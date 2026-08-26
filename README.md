# TrajFlow-KV

Minimal, reproducible Phase-1 implementation of return-driven low-rank KV
memory transport for VLM GUI agents.

The code freezes a Qwen2.5-VL backbone, injects trainable residual low-rank
projectors into decoder `k_proj` and `v_proj` activations, and optimizes them
with a trajectory-level return-weighted objective plus transport-energy and
orthogonality penalties. It includes a deterministic toy policy for CI and a
Qwen/AITW path for real experiments.

## Current go/no-go result

The latest controlled run is a **single-task online go / cross-task pending**,
not a broad end-to-end success claim. With Qwen2.5-VL-3B, V-only projectors on the last
eight layers (rank 8, alpha 8), 41 training trajectories and 20 epochs, the
held-out success-minus-failure log-probability margin improved from `0.1965`
to `0.2598`. Shuffled-return training reached `0.1955`, and successful-action
CE reached `0.1874`. However, learned free-form rollouts still achieved no
success on the tested Wi-Fi/Bluetooth system tasks. The exact snapshot is in
[`results/gonogo_current.json`](results/gonogo_current.json).

A follow-up with positive-trajectory-only action CE (`lambda=0.01`) retained
the return margin (`0.2602`) and improved MRR from `0.6250` to `0.6349`, but
Top-1 stayed at `0.5`; its online Wi-Fi rollout emitted legal actions yet
repeated the same swipe for all eight steps and failed. This narrows the next
problem to state-conditioned/fork-point action selection rather than syntax.

State-conditioned fork preferences, click-coordinate hard negatives, and
hierarchical type/parameter ranking subsequently raise held-out return margin
to `0.3685`, action MRR to `0.6722`, and type-conditioned Top-1 to `0.8333`.
On real AndroidWorld Bluetooth, the identical structured policy yields
baseline `0/4` versus Return/Fork/Coordinate-KV `4/4` across one deterministic
and three temperature-0.2 seeds. Wi-Fi still fails because the policy acts
before the quick-settings shade is fully expanded. The next gate is therefore
learned state routing and cross-task online replication.

A lightweight screenshot-conditioned state router is also included. On the
current independent system-task split it reaches `5/6` action-type accuracy
(`3/3` on click states). Its cross-task online result is still pending because
the software-only emulator developed a black-render/system-server failure;
those corrupted probes are explicitly excluded. The committed AndroidWorld
patch now permits a physical-size fallback so action execution does not depend
on a fragile `dumpsys input` call under TCG.

The energy term was also calibrated rather than left at its numerically
negligible initial coefficient. In a contemporaneous 10-epoch comparison,
unregularized Return-KV has held-out margin `0.2209` at mean activation energy
`3.00e-4`; `lambda_energy=3000` retains margin `0.2160` while reducing energy
to `6.72e-5` (4.47x lower). `lambda_energy=100000` reaches `8.19e-7` but drops
margin below the initial checkpoint, demonstrating the expected Pareto
trade-off and providing a practical low-energy setting.

The learned-critic gate is currently negative. An MLP critic trained on 29
system trajectories obtains train AUC `1.0` but only held-out AUC
`0.75–0.8125` with saturated probabilities; a regularized linear critic is
unstable across seeds (`0.375/0.625/0.625`) and underperforms raw trajectory
log-probability ranking (`0.75`). The critic code is retained for
reproduction, but critic-guided KV backprop is deliberately not reported as
an improvement until substantially more independent trajectories exist.

A task/screenshot-conditioned rectified-flow sampler over four-step action
chunks is implemented as a separate exploration module. On two independent
held-out successful trajectories, its mean endpoint MSE is `0.1580` versus
Gaussian noise `1.1474`. After projection to the structured action manifold,
best-of-128 MSE is `0.00247` for Wi-Fi and `0.0924` for Bluetooth. However,
mean behavior-cloning baselines are better than average flow samples, and the
critic gate above cannot yet select the best endpoints reliably. The honest
status is therefore candidate-coverage go, average-policy/critic-chain no-go.

For the current linear KV residual, internalization can be exact rather than
approximate: `y'=(I+AB)y` and `y=Wx` imply a merged projection
`W'=(I+AB)W` (and the same transform for bias). Folding the best checkpoint
into eight V projections removes all inference hooks. The hooked teacher and
merged student have identical held-out Top-1 (`0.5`), MRR (`0.6722`), and
type-conditioned Top-1/MRR (`0.8333/0.8889`); return margin is `0.3767` versus
`0.3836`, with the small difference attributable to bf16 merge rounding.

The K/V ablation is not one-sided. At 10 return-training epochs, K-only moves
held-out margin from `0.1818` to `0.2369` (`+0.0550`), while V-only moves from
`0.1835` to `0.2209` (`+0.0373`). After fork/coordinate shaping, however,
K reaches margin/MRR `0.2818/0.6389` whereas V reaches
`0.3767/0.6722`. This suggests K is stronger for coarse trajectory
separation and V for concrete action/coordinate shaping.

Using SVD to reparameterize the same AITW warm-start makes the rank ablation
comparable. At 10 epochs, ranks `4/8/16` obtain return margins
`0.2094/0.2209/0.2129`; all have Top-1/MRR `0.5/0.6210`. Rank 8 is best for
return separation, rank 16 is already saturated on 41 trajectories, and rank
4 is a reasonable lower-energy budget point.

As a cross-dataset retention check, 16 AITW trajectories have mean action
log-prob `-5.9998` at warm-start, `-6.0107` after AndroidWorld return training,
`-6.0238` with the `lambda_energy=3000` checkpoint, and `-6.1120` after the
strongest fork/coordinate shaping. Pure return training therefore changes the
AITW metric by only about `0.18%`; fork shaping costs about `1.87%`. The
low-energy checkpoint lowers activation energy but does not beat unregularized
return training on this retention metric.

The principal 20-epoch run is reproducible with:

```bash
.venv/bin/python -m trajflow_kv.train \
  --config configs/qwen_androidworld.yaml \
  --data-path data/androidworld/multitask_return_train_v2.jsonl \
  --output-dir outputs/gonogo/multitask_v2_warm_return_v_r8_l8_a8_e20 \
  --projector-checkpoint outputs/gonogo/initial_v_l8/kv_projectors.pt \
  --target v --last-n-layers 8 --rank 8 --alpha 8 --epochs 20
```

## What is and is not implemented

- Implemented: online KV hooks (no cached activation dataset), K/V/both
  injection, REINFORCE-style trajectory loss, per-task baseline, normalized
  advantages, energy and orthogonality regularization, JSONL trajectory
  format, checkpoints, fork/coordinate preference training, hierarchical
  candidate ranking, loop guards, and smoke tests.
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
  --task OpenAppTaskEval --task-index 0 \
  --rollouts 8 --max-steps 1 --temperature 1.5 \
  --output data/androidworld/open_app_rollouts.jsonl

.venv/bin/python scripts/validate_rollouts.py data/androidworld/open_app_rollouts.jsonl
.venv/bin/python -m trajflow_kv.train --config configs/qwen_androidworld.yaml
```

For a paired check, collect baseline and candidate rollouts with the same
`--seed`, task, horizon, and temperature, then run:

```bash
.venv/bin/python scripts/compare_rollouts.py \
  --baseline data/androidworld/eval_pre.jsonl \
  --candidate data/androidworld/eval_post.jsonl \
  --output outputs/qwen-androidworld-return/controlled_eval.json
```

The summary reports paired improvements/regressions, success-rate delta, and
invalid-action counts. Small smoke samples are a pipeline check, not evidence
of a statistically significant policy improvement.

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

### Running without `/dev/kvm`

The verified software-emulation setup is reproducible with:

```bash
./scripts/setup_androidworld_software.sh
./scripts/run_androidworld_software.sh
```

The setup pins AndroidWorld commit `3e508885`, installs API 33/Pixel 6 and a
Python 3.11 environment, and applies `patches/androidworld-tcg.patch`. The patch
makes ADB timeouts configurable, allows screenshot-only operation without the
accessibility forwarder, and makes first-time app setup optional. For smoke
tasks it also enables lightweight task initialization, skipping date and app
snapshot resets that are not required by `OpenAppTaskEval`. Do not use that
mode for tasks whose evaluator depends on restored app state. The runner uses
`-accel off`, increases Android's watchdog timeout multiplier, retries an
Android ActivityController during boot, skips Setup Wizard, and serves the
official API on `127.0.0.1:5000`.

On the reference RTX A4000 container, first boot took roughly 25 minutes.
Native AndroidWorld initialization then returned a 1080x2400 screenshot in
15.4 seconds. A real `ContactsAddContact` initialize/score/teardown cycle and
a Qwen + KV checkpoint rollout were both verified.

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
