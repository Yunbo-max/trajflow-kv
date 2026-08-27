#!/usr/bin/env python3
"""Scan which decoder K/V layers causally carry a historical visual block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluate_counterfactual_qwen import _groups, _score_groups, summarize_policy_scores
from trajflow_kv.causal_ablation import attach_kv_block_ablator, decoder_layer_index
from trajflow_kv.counterfactual import load_counterfactual_jsonl


def _layer_groups(layer_count: int) -> dict[str, list[int]]:
    if layer_count < 3:
        return {"all": list(range(layer_count))}
    first = layer_count // 3
    second = 2 * layer_count // 3
    return {
        "early": list(range(0, first)),
        "middle": list(range(first, second)),
        "late": list(range(second, layer_count)),
        "all": list(range(layer_count)),
    }


def _compact(summary: dict) -> dict:
    return {
        "prefixes": summary["prefixes"],
        "candidate_top1_accuracy": summary["candidate_top1_accuracy"],
        "critical_prefixes": summary["critical_prefixes"],
        "critical_fork_accuracy": summary["critical_fork_accuracy"],
        "families": summary["families"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--family", default="hidden_memory")
    parser.add_argument("--targets", nargs="+", choices=("k", "v", "both"), default=("k", "v", "both"))
    parser.add_argument("--image-indices", nargs="+", type=int, default=(0, 1))
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device
    ).eval()
    rows = load_counterfactual_jsonl(args.data)
    rows = [row for row in rows if not args.family or row.get("task_family") == args.family]
    groups = _groups(rows)
    if not groups:
        raise ValueError("no matching counterfactual prefix groups")

    layer_indices = sorted({
        layer for name, _ in model.named_modules()
        if (layer := decoder_layer_index(name)) is not None
    })
    if not layer_indices:
        raise RuntimeError("could not discover decoder layers")
    layer_groups = _layer_groups(max(layer_indices) + 1)

    baseline = _compact(summarize_policy_scores(_score_groups(model, processor, groups, args.device)))
    conditions = []
    for image_index in args.image_indices:
        for target in args.targets:
            for group_name, layers in layer_groups.items():
                ablator = attach_kv_block_ablator(model, target=target, layers=layers)
                try:
                    scored = _score_groups(
                        model, processor, groups, args.device,
                        kv_ablator=ablator, ablate_image_index=image_index,
                    )
                finally:
                    ablator.close()
                summary = _compact(summarize_policy_scores(scored))
                conditions.append({
                    "image_index": image_index,
                    "target": target,
                    "layer_group": group_name,
                    "layers": layers,
                    "evaluation": summary,
                    "critical_accuracy_delta": (
                        summary["critical_fork_accuracy"] - baseline["critical_fork_accuracy"]
                    ),
                })

    result = {
        "model": args.model,
        "data": args.data,
        "family": args.family,
        "layer_count": max(layer_indices) + 1,
        "baseline": baseline,
        "conditions": conditions,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
