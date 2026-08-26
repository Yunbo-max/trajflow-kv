from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import yaml
from torch import nn

from .data import load_jsonl
from .objective import normalized_advantages, shuffle_within_tasks, trajectory_policy_loss
from .projector import attach_kv_projectors


class ToyKVPolicy(nn.Module):
    def __init__(self, width: int = 16, actions: int = 5):
        super().__init__()
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.head = nn.Linear(width, actions, bias=False)

    def forward(self, x):
        return self.head(torch.tanh(self.k_proj(x) + self.v_proj(x)))


def toy_batch(device: str):
    generator = torch.Generator(device=device).manual_seed(19)
    x = torch.randn(8, 16, generator=generator, device=device)
    actions = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2], device=device)
    returns = torch.tensor([1., 0., 1., 0., 1., 0., 0., 1.], device=device)
    task_ids = ["a", "a", "b", "b", "c", "c", "d", "d"]
    return x, actions, returns, task_ids


def train_toy(cfg):
    device = cfg["device"]
    model = ToyKVPolicy().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bundle = attach_kv_projectors(
        model,
        cfg["rank"],
        cfg["alpha"],
        cfg["target"],
        cfg.get("last_n_layers"),
    )
    optimizer = torch.optim.AdamW(bundle.modules.parameters(), lr=cfg["lr"])
    history = []
    for epoch in range(cfg["epochs"]):
        x, actions, returns, task_ids = toy_batch(device)
        logits = model(x)
        logprobs = logits.log_softmax(-1).gather(1, actions[:, None]).squeeze(1)
        advantages = normalized_advantages(returns, task_ids)
        pg = trajectory_policy_loss(logprobs, advantages)
        loss = pg + cfg["lambda_energy"] * bundle.energy() + cfg["lambda_orth"] * bundle.orthogonality_loss()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch, "loss": float(loss.detach()), "pg": float(pg.detach())})
    return model, bundle, history


def train_qwen(cfg):
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from .qwen_policy import build_action_prompt

    device = cfg["device"]
    processor = AutoProcessor.from_pretrained(cfg["model_path"], max_pixels=cfg["max_pixels"])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg["model_path"], dtype=torch.bfloat16, device_map=device
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # Training never reuses generation caches. Keeping them would retain a
    # full extra KV copy and can exhaust a 16 GiB card before backward.
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    bundle = attach_kv_projectors(
        model,
        cfg["rank"],
        cfg["alpha"],
        cfg["target"],
        cfg.get("last_n_layers"),
    )
    projector_checkpoint = cfg.get("projector_checkpoint")
    if projector_checkpoint:
        bundle.modules.load_state_dict(
            torch.load(projector_checkpoint, map_location=device, weights_only=True)
        )
    optimizer = torch.optim.AdamW(bundle.modules.parameters(), lr=cfg["lr"])
    trajectories = load_jsonl(cfg["data_path"])
    trajectories = trajectories[: cfg.get("max_trajectories", len(trajectories))]

    # Teacher-forced action log probabilities retain gradient through hooked K/V.
    history = [{
        "event": "setup",
        "hooked_modules": len(bundle.names),
        "trainable_parameters": sum(p.numel() for p in bundle.modules.parameters()),
        "first_hook": bundle.names[0],
        "last_hook": bundle.names[-1],
        "projector_checkpoint": projector_checkpoint,
    }]
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(cfg["epochs"]):
        observed_returns = torch.tensor(
            [float(t["return"]) for t in trajectories], device=device
        )
        returns = observed_returns.clone()
        task_ids = [t["task_id"] for t in trajectories]
        return_mode = cfg.get("return_mode", "observed")
        if return_mode == "shuffle":
            returns = shuffle_within_tasks(returns, task_ids)
        elif return_mode == "zero":
            returns = torch.zeros_like(returns)
        elif return_mode != "observed":
            raise ValueError(f"unsupported return_mode: {return_mode}")
        advantages = normalized_advantages(returns, task_ids)
        if cfg.get("lambda_action", 0.0) == 0 and torch.count_nonzero(advantages) == 0:
            raise ValueError(
                "pure return training has zero advantages; collect mixed returns "
                "for at least one repeated task or set lambda_action > 0 for imitation"
            )
        optimizer.zero_grad(set_to_none=True)
        for index, trajectory in enumerate(trajectories):
            trajectory_logprob = torch.zeros((), device=device)
            for step in trajectory["steps"]:
                content = []
                loaded_image = None
                if step.get("image"):
                    loaded_image = Image.open(step["image"]).convert("RGB")
                    content.append({"type": "image", "image": loaded_image})
                inferred_size = loaded_image.size if loaded_image else tuple(step.get("screen_size", (1000, 1000)))
                prompt = build_action_prompt(
                    trajectory["instruction"], step.get("history", []), inferred_size
                )
                content.append({"type": "text", "text": prompt})
                messages = [{"role": "user", "content": content},
                            {"role": "assistant", "content": [
                                {"type": "text", "text": step["action"]}
                            ]}]
                prompt_messages = messages[:1]
                prompt_batch = processor.apply_chat_template(
                    prompt_messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt"
                )
                batch = processor.apply_chat_template(messages, tokenize=True, return_dict=True,
                                                       return_tensors="pt").to(device)
                labels = batch["input_ids"].clone()
                # Supervise the complete assistant span, including the action;
                # final-N masking is wrong because chat templates add EOS tokens.
                prompt_tokens = prompt_batch["input_ids"].shape[1]
                labels[:, :prompt_tokens] = -100
                output = model(**batch, labels=labels, use_cache=False)
                valid = (labels != -100).sum().clamp_min(1)
                trajectory_logprob = trajectory_logprob - output.loss * valid
            # AITW demos have no failure return; action NLL keeps that offline
            # warm-start useful. Set lambda_action=0 for pure online return RL.
            loss = -advantages[index].detach() * trajectory_logprob
            action_weight = cfg.get("lambda_action", 0.0)
            if cfg.get("positive_action_only", False) and observed_returns[index] <= 0:
                action_weight = 0.0
            loss = loss - action_weight * trajectory_logprob
            energy = bundle.energy()
            orthogonality = bundle.orthogonality_loss()
            loss = loss + cfg["lambda_energy"] * energy + cfg["lambda_orth"] * orthogonality
            (loss / cfg["gradient_accumulation_steps"]).backward()
            if (index + 1) % cfg["gradient_accumulation_steps"] == 0 or index + 1 == len(trajectories):
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            metric = {"epoch": epoch, "trajectory": index, "loss": float(loss.detach()),
                      "return": float(returns[index]), "advantage": float(advantages[index]),
                      "energy": float(energy.detach()),
                      "orthogonality": float(orthogonality.detach())}
            if device.startswith("cuda"):
                metric["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2**30
            history.append(metric)
    return model, bundle, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-trajectories", type=int)
    parser.add_argument("--data-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--projector-checkpoint")
    parser.add_argument("--no-projector-checkpoint", action="store_true")
    parser.add_argument("--target", choices=("k", "v", "both"))
    parser.add_argument("--rank", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--last-n-layers", type=int)
    parser.add_argument("--return-mode", choices=("observed", "shuffle", "zero"))
    parser.add_argument("--lambda-action", type=float)
    parser.add_argument("--positive-action-only", action="store_true")
    parser.add_argument("--lambda-energy", type=float)
    parser.add_argument("--lambda-orth", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-pixels", type=int)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.max_trajectories is not None:
        cfg["max_trajectories"] = args.max_trajectories
    if args.data_path is not None:
        cfg["data_path"] = args.data_path
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.projector_checkpoint is not None:
        cfg["projector_checkpoint"] = args.projector_checkpoint
    if args.no_projector_checkpoint:
        cfg["projector_checkpoint"] = None
    if args.positive_action_only:
        cfg["positive_action_only"] = True
    for key in (
        "return_mode", "lambda_action", "lambda_energy", "lambda_orth",
        "epochs", "max_pixels", "target", "rank", "alpha", "last_n_layers",
    ):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    model, bundle, history = train_toy(cfg) if cfg["toy"] else train_qwen(cfg)
    output = Path(cfg["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    torch.save(bundle.modules.state_dict(), output / "kv_projectors.pt")
    (output / "metrics.json").write_text(json.dumps(history, indent=2))
    print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    main()
