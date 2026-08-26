# TrajFlow-KV

Minimal, reproducible Phase-1 implementation of return-driven low-rank KV
memory transport for VLM GUI agents.

The code freezes a Qwen2.5-VL backbone, injects trainable residual low-rank
projectors into decoder `k_proj` and `v_proj` activations, and optimizes them
with a trajectory-level return-weighted objective plus transport-energy and
orthogonality penalties. It includes a deterministic toy policy for CI and a
Qwen/AITW path for real experiments.

## What is and is not implemented

- Implemented: online KV hooks (no cached activation dataset), K/V/both
  injection, REINFORCE-style trajectory loss, per-task baseline, normalized
  advantages, energy and orthogonality regularization, JSONL trajectory
  format, checkpoints, and smoke tests.
- Adapter-ready: AndroidWorld rollouts can be exported into the same JSONL
  schema using `scripts/androidworld_to_jsonl.py`.
- Deferred deliberately: flow action sampler, learned critic, and student
  distillation. These are Phase 2-4 and should be evaluated only after the
  return-driven projector baseline works.

## Quick start (no model download)

```bash
cd /root/trajflow-kv
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest -q
.venv/bin/python -m trajflow_kv.train --config configs/toy.yaml
```

## Real model and data

```bash
./scripts/download_model.sh
./scripts/download_aitw_sample.sh 1000
.venv/bin/pip install -e '.[train]'
.venv/bin/python -m trajflow_kv.train --config configs/qwen_aitw.yaml
```

The Qwen run requires substantially more memory than the toy test. On a 16GB
GPU use batch size 1, gradient accumulation, gradient checkpointing and a
small image pixel budget. The backbone is frozen; only projectors train.

## Trajectory JSONL schema

Each line represents one trajectory:

```json
{"task_id":"clock-1","instruction":"Set an alarm","return":1.0,"steps":[{"image":"/abs/s0.png","history":[],"action":"{\"action_type\":\"click\",\"x\":0.5,\"y\":0.8}"}]}
```

`return` is the environment return. Every action log-probability in a
trajectory receives the same normalized trajectory advantage in this minimal
version. Later work can replace this with reward-to-go or a critic.

The bundled AITW viewer sample contains successful demonstrations only, so
`lambda_action` supplies an imitation warm-start. Pure return optimization
requires mixed-return rollouts from AndroidWorld; set `lambda_action: 0` for
that experiment. Treating all-success demonstrations as policy-gradient data
would yield zero centered advantage and is intentionally avoided.

## Reproducibility notes

- Model: `Qwen/Qwen2.5-VL-3B-Instruct`, pinned by the snapshot downloader.
- Online benchmark: AndroidWorld; its emulator is intentionally not bundled.
- Offline source: AITW. The helper downloads a bounded sample/index rather
  than the multi-million-step corpus.
- Never commit Hugging Face or GitHub tokens. `.env`, model files, datasets,
  checkpoints and logs are ignored.
