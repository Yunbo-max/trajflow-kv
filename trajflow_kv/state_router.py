from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn


def routed_action_type(action: str) -> str | None:
    action_type = json.loads(action)["action_type"]
    if action_type in {"swipe", "scroll"}:
        return "pan"
    if action_type == "click":
        return "click"
    return None


def router_features(image: Image.Image, history_length: int, size: int = 24) -> torch.Tensor:
    pixels = np.asarray(image.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
    flat = torch.from_numpy(pixels).flatten()
    return torch.cat([flat, torch.tensor([min(history_length, 20) / 20.0])])


class StateRouter(nn.Module):
    def __init__(self, input_width: int, classes: list[str], hidden: int = 64):
        super().__init__()
        self.classes = classes
        self.network = nn.Sequential(
            nn.Linear(input_width, hidden), nn.ReLU(), nn.Linear(hidden, len(classes))
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    def predict(self, image: Image.Image, history_length: int) -> str:
        features = router_features(image, history_length).to(next(self.parameters()).device)
        with torch.inference_mode():
            index = int(self(features.unsqueeze(0)).argmax(-1))
        return self.classes[index]

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "StateRouter":
        payload = torch.load(path, map_location=device, weights_only=True)
        model = cls(payload["input_width"], payload["classes"], payload["hidden"])
        model.load_state_dict(payload["state_dict"])
        return model.to(device).eval()

    def save(self, path: str | Path, hidden: int = 64) -> None:
        torch.save({
            "input_width": self.network[0].in_features,
            "classes": self.classes,
            "hidden": hidden,
            "state_dict": self.state_dict(),
        }, path)
