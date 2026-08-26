#!/usr/bin/env bash
set -euo pipefail
sample_count="${1:-16}"
mkdir -p data/aitw
huggingface-cli download KMK040412/aitw-processed-labeled-full \
  viewer_examples/gmail_mobile_use_examples.parquet \
  official_splits/standard.json \
  --repo-type dataset --local-dir data/aitw-source
python3 scripts/prepare_aitw.py \
  --local-parquet data/aitw-source/viewer_examples/gmail_mobile_use_examples.parquet \
  --limit "$sample_count" --output data/aitw/train.jsonl --image-dir data/aitw/images
