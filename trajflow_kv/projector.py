from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


class LowRankResidual(nn.Module):
    """x -> x + alpha * A(B(x)); initialized as an identity modification."""

    def __init__(self, width: int, rank: int, alpha: float):
        super().__init__()
        if not 0 < rank <= width:
            raise ValueError(f"rank must be in [1, {width}], got {rank}")
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.orthogonal_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank
        self.last_delta: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.up(self.down(x)) * self.scale
        self.last_delta = delta
        return x + delta

    def energy(self) -> torch.Tensor:
        if self.last_delta is None:
            return self.up.weight.square().mean()
        return self.last_delta.float().square().mean()

    def orthogonality_loss(self) -> torch.Tensor:
        rows = self.down.weight.float()
        gram = rows @ rows.T
        eye = torch.eye(gram.shape[0], device=gram.device)
        return (gram - eye).square().mean()


@dataclass
class HookedProjectors:
    modules: nn.ModuleList
    handles: list
    names: list[str]

    def energy(self) -> torch.Tensor:
        return torch.stack([m.energy() for m in self.modules]).mean()

    def orthogonality_loss(self) -> torch.Tensor:
        return torch.stack([m.orthogonality_loss() for m in self.modules]).mean()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def attach_kv_projectors(
    model: nn.Module,
    rank: int,
    alpha: float,
    target: str = "both",
    last_n_layers: int | None = None,
) -> HookedProjectors:
    """Attach projectors to real decoder K/V projection outputs.

    Hooks preserve autograd. Projectors are registered on the root model so
    optimizers and checkpoints see them, while the frozen base stays fixed.
    """
    if target not in {"k", "v", "both"}:
        raise ValueError("target must be k, v, or both")
    suffixes = {"k": ("k_proj",), "v": ("v_proj",), "both": ("k_proj", "v_proj")}[target]
    selected = [(name, module) for name, module in model.named_modules()
                if name.rsplit(".", 1)[-1] in suffixes and isinstance(module, nn.Linear)]
    if not selected:
        raise RuntimeError("No decoder k_proj/v_proj Linear modules found")
    if last_n_layers is not None:
        if last_n_layers <= 0:
            raise ValueError("last_n_layers must be positive")
        # Keep K and V from the requested number of final transformer layers.
        keep = last_n_layers * len(suffixes)
        selected = selected[-keep:]

    projectors = nn.ModuleList()
    handles, names = [], []
    for name, linear in selected:
        projector = LowRankResidual(linear.out_features, rank, alpha).to(
            device=linear.weight.device, dtype=linear.weight.dtype
        )
        projectors.append(projector)
        handles.append(linear.register_forward_hook(
            lambda _module, _inputs, output, p=projector: p(output)
        ))
        names.append(name)
    model.add_module("trajflow_kv_projectors", projectors)
    return HookedProjectors(projectors, handles, names)


def trainable_parameters(bundle: HookedProjectors) -> Iterable[nn.Parameter]:
    return bundle.modules.parameters()


def merge_projectors_into_model(model: nn.Module, bundle: HookedProjectors) -> dict[str, torch.Tensor]:
    """Fold linear residual projectors into their K/V projection weights."""
    modules = dict(model.named_modules())
    merged = {}
    bundle.close()
    with torch.no_grad():
        for name, projector in zip(bundle.names, bundle.modules, strict=True):
            linear = modules[name]
            delta = (
                projector.up.weight.float()
                @ projector.down.weight.float()
                @ linear.weight.float()
            ) * projector.scale
            linear.weight.add_(delta.to(linear.weight.dtype))
            merged[f"{name}.weight"] = linear.weight.detach().cpu().clone()
            if linear.bias is not None:
                bias_delta = (
                    projector.up.weight.float()
                    @ projector.down.weight.float()
                    @ linear.bias.float()
                ) * projector.scale
                linear.bias.add_(bias_delta.to(linear.bias.dtype))
                merged[f"{name}.bias"] = linear.bias.detach().cpu().clone()
    return merged


def load_merged_weights(model: nn.Module, checkpoint: str) -> list[str]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weights = payload.get("weights", payload)
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in weights.items():
            parameters[name].copy_(value.to(parameters[name].device, parameters[name].dtype))
    return sorted(weights)


def resize_low_rank_factors(
    down: torch.Tensor,
    up: torch.Tensor,
    source_alpha: float,
    target_rank: int,
    target_alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SVD-reparameterize a residual map at another rank and scale."""
    source_scale = source_alpha / down.shape[0]
    target_scale = target_alpha / target_rank
    effective = source_scale * up.float() @ down.float()
    left, singular, right = torch.linalg.svd(effective, full_matrices=False)
    take = min(target_rank, len(singular))
    new_up = torch.zeros(up.shape[0], target_rank)
    new_down = torch.zeros(target_rank, down.shape[1])
    for index in range(take):
        if singular[index] > 1e-10:
            root = torch.sqrt(singular[index] / target_scale)
            new_up[:, index] = left[:, index] * root
            new_down[index] = right[index] * root
        else:
            # A zero-up/nonzero-down slot preserves the map but can learn.
            new_down[index] = right[index]
    return new_down.to(down.dtype), new_up.to(up.dtype)
