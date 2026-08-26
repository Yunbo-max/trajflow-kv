from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from .projector import attach_kv_projectors


SYSTEM = """You control Android. Return exactly one JSON action and no prose.
Valid examples:
{"action_type":"click","x":120,"y":300}
{"action_type":"scroll","direction":"down"}
{"action_type":"input_text","text":"hello"}
{"action_type":"navigate_back"}
{"action_type":"status","goal_status":"complete"}
Coordinates may be absolute pixels or normalized floats in [0,1]."""


def build_action_prompt(instruction: str, history: list[str], screen_size: tuple[int, int]) -> str:
    return (f"{SYSTEM}\nScreen size: {screen_size[0]}x{screen_size[1]}\n"
            f"Task: {instruction}\nHistory: {json.dumps(history)}")


class QwenKVPolicy:
    def __init__(self, model_path: str, checkpoint: str | None = None, *, rank: int = 8,
                 alpha: float = 0.1, target: str = "both", device: str = "cuda",
                 max_pixels: int = 401408, temperature: float = 0.7,
                 max_new_tokens: int = 128):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device
        ).eval()
        self.bundle = attach_kv_projectors(self.model, rank, alpha, target)
        if checkpoint:
            state = torch.load(Path(checkpoint), map_location=device, weights_only=True)
            self.bundle.modules.load_state_dict(state)
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def act(self, instruction: str, image: Image.Image, history: list[str],
            screen_size: tuple[int, int]) -> str:
        prompt = build_action_prompt(instruction, history, screen_size)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt}
        ]}]
        batch = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        ).to(self.device)
        sampling = ({"do_sample": True, "temperature": self.temperature, "top_p": 0.9}
                    if self.temperature > 0 else {"do_sample": False, "temperature": None, "top_p": None})
        generated = self.model.generate(**batch, max_new_tokens=self.max_new_tokens, **sampling)
        suffix = generated[0, batch["input_ids"].shape[1]:]
        return self.processor.decode(suffix, skip_special_tokens=True).strip()
