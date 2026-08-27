#!/usr/bin/env python3
"""Paired online statistics with task-clustered bootstrap and exact McNemar."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajflow_kv.statistics import summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL with task_id, baseline, candidate")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    result = summarize(rows, args.samples, args.seed)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
