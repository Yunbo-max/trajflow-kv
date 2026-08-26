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
    model: nn.Module, rank: int, alpha: float, target: str = "both"
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

