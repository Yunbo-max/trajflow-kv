#!/usr/bin/env bash
set -euo pipefail
mkdir -p models
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir models/Qwen2.5-VL-3B-Instruct

