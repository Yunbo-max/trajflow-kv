from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import yaml
from torch import nn

from .data import load_jsonl
from .objective import normalized_advantages, trajectory_policy_loss
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
    bundle = attach_kv_projectors(model, cfg["rank"], cfg["alpha"], cfg["target"])
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

    device = cfg["device"]
    processor = AutoProcessor.from_pretrained(cfg["model_path"], max_pixels=cfg["max_pixels"])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg["model_path"], torch_dtype=torch.bfloat16, device_map=device
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.gradient_checkpointing_enable()
    bundle = attach_kv_projectors(model, cfg["rank"], cfg["alpha"], cfg["target"])
    optimizer = torch.optim.AdamW(bundle.modules.parameters(), lr=cfg["lr"])
    trajectories = load_jsonl(cfg["data_path"])

    # Teacher-forced action log probabilities retain gradient through hooked K/V.
    history = []
    for epoch in range(cfg["epochs"]):
        returns = torch.tensor([float(t["return"]) for t in trajectories], device=device)
        advantages = normalized_advantages(returns, [t["task_id"] for t in trajectories])
        optimizer.zero_grad(set_to_none=True)
        for index, trajectory in enumerate(trajectories):
            trajectory_logprob = torch.zeros((), device=device)
            for step in trajectory["steps"]:
                content = []
                if step.get("image"):
                    content.append({"type": "image", "image": Image.open(step["image"]).convert("RGB")})
                prompt = f"Task: {trajectory['instruction']}\nHistory: {json.dumps(step.get('history', []))}\nEmit one JSON action."
                content.append({"type": "text", "text": prompt})
                messages = [{"role": "user", "content": content},
                            {"role": "assistant", "content": step["action"]}]
                batch = processor.apply_chat_template(messages, tokenize=True, return_dict=True,
                                                       return_tensors="pt").to(device)
                labels = batch["input_ids"].clone()
                # Minimal robust masking: supervise only final action token span.
                action_ids = processor.tokenizer(step["action"], add_special_tokens=False)["input_ids"]
                labels[:, :-len(action_ids)] = -100
                output = model(**batch, labels=labels)
                valid = (labels != -100).sum().clamp_min(1)
                trajectory_logprob = trajectory_logprob - output.loss * valid
            # AITW demos have no failure return; action NLL keeps that offline
            # warm-start useful. Set lambda_action=0 for pure online return RL.
            loss = -advantages[index].detach() * trajectory_logprob
            loss = loss - cfg.get("lambda_action", 0.0) * trajectory_logprob
            loss = loss + cfg["lambda_energy"] * bundle.energy() + cfg["lambda_orth"] * bundle.orthogonality_loss()
            (loss / cfg["gradient_accumulation_steps"]).backward()
            if (index + 1) % cfg["gradient_accumulation_steps"] == 0 or index + 1 == len(trajectories):
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            history.append({"epoch": epoch, "trajectory": index, "loss": float(loss.detach())})
    return model, bundle, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    model, bundle, history = train_toy(cfg) if cfg["toy"] else train_qwen(cfg)
    output = Path(cfg["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    torch.save(bundle.modules.state_dict(), output / "kv_projectors.pt")
    (output / "metrics.json").write_text(json.dumps(history, indent=2))
    print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    main()
