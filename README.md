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
.venv/bin/pip install -e '.[train]'
./scripts/setup_cuda126.sh
./scripts/download_model.sh
./scripts/download_aitw_sample.sh 16
.venv/bin/python -m trajflow_kv.train --config configs/qwen_aitw.yaml
```

The Qwen run requires substantially more memory than the toy test. On a 16GB
GPU use batch size 1, gradient accumulation, gradient checkpointing and a
small image pixel budget. The backbone is frozen; only projectors train.
`max_trajectories: 1` is the committed real-model smoke-test default; increase
it only after observing peak GPU memory on your machine.

Verified on an RTX A4000 (16GB): 16 viewer examples use 9.28 GiB peak allocated
GPU memory, hook all 72 K/V projections across 36 language layers, and train
294,912 projector parameters. Run the verified batch with
`--max-trajectories 16`.

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

## AndroidWorld mixed-return loop

AndroidWorld's official Docker image exposes a FastAPI server on port 5000.
It can run on the same host or a separate emulator host; this repository only
needs HTTP access to it. Collect repeated stochastic rollouts for one fixed
task instance so the task-group baseline is meaningful:

```bash
.venv/bin/python scripts/collect_androidworld.py \
  --server-url http://ANDROID_HOST:5000 \
  --task ContactsAddContact --task-index 0 \
  --rollouts 8 --temperature 0.7

.venv/bin/python scripts/validate_rollouts.py data/androidworld/rollouts.jsonl
.venv/bin/python -m trajflow_kv.train --config configs/qwen_androidworld.yaml
```

The collector uses the official `/screenshot`, `/execute_action`, task
initialize/tear-down, goal, and score endpoints. Screenshots and canonical
actions are saved per step. Invalid model JSON becomes a recorded `wait`
action rather than crashing or silently dropping the trajectory. AITW-style
point coordinates and two-point swipes are converted to AndroidWorld's action
schema at this boundary.

Pure return training (`lambda_action: 0`) refuses to run if every task group
has constant return. This prevents an all-success or all-failure batch from
silently producing zero centered policy-gradient signal. The AndroidWorld
config loads the AITW projector checkpoint before applying return updates.

To exercise the exact return path without an emulator, create a small mixed
fixture from the downloaded sample:

```bash
.venv/bin/python scripts/make_mixed_return_smoke.py \
  data/aitw/train.jsonl /tmp/trajflow-mixed.jsonl
.venv/bin/python scripts/validate_rollouts.py /tmp/trajflow-mixed.jsonl
.venv/bin/python -m trajflow_kv.train \
  --config configs/qwen_androidworld.yaml \
  --data-path /tmp/trajflow-mixed.jsonl \
  --output-dir /tmp/trajflow-return-smoke
```
