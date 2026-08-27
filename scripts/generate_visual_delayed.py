#!/usr/bin/env python3
"""Generate the screenshot-backed TANGO delayed-consequence pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajflow_kv.counterfactual import write_counterfactual_jsonl
from trajflow_kv.visual_delayed import build_visual_counterfactual_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/visual_delayed/pilot.jsonl"))
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds (0..N-1).")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--aggregation", choices=("mean", "max"), default="mean")
    args = parser.parse_args()
    image_dir = args.image_dir or args.output.parent / "images"
    rows = build_visual_counterfactual_dataset(
        range(args.seed_start, args.seed_start + args.seeds),
        output_dir=image_dir,
        horizon=args.horizon,
        aggregation=args.aggregation,
    )
    count = write_counterfactual_jsonl(rows, args.output)
    prefixes = {row["prefix_id"] for row in rows}
    result = {
        "output": str(args.output),
        "image_dir": str(image_dir),
        "rows": count,
        "prefixes": len(prefixes),
        "families": sorted({row["task_family"] for row in rows}),
        "critical_rows": sum(bool(row["is_critical_action"]) for row in rows),
        "critical_prefixes": sum(bool(row["critical_step"]) for row in rows if row["action_index"] == 0),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

