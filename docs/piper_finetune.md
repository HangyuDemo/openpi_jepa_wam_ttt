# Piper pick-can-place-plate fine-tuning

The downloaded dataset is LeRobot v3. This repository's pinned LeRobot reader
uses v2.1, so train from the converted copy at
`data/pick_can_place_plate_150_trimmed_v21`. Regenerate it with:

```bash
.venv/bin/python scripts/convert_piper_v3_to_v21.py
```

The three Piper configs in `src/openpi/training/config.py` are:

| Config | Purpose | Initial weights |
| --- | --- | --- |
| `pi05_piper` | Compute Piper normalization statistics | π0.5 base, if trained |
| `pi05_piper_vjepa_aux` | Fine-tune π0.5 with the JEPA-WAM future objective | π0.5 base |
| `pi05_piper_ttt_wrapper_explicit_jepa_ttt_only` | Train four wrapper TTT blocks with frozen JEPA-WAM policy | Trained `pi05_piper_vjepa_aux` checkpoint |

The dataset has a 14D state and 7D joint/gripper action. The first six action
dimensions are trained as deltas from the current joint state; the gripper is
absolute. The front image maps to `base_0_rgb`, the wrist image to
`left_wrist_0_rgb`, and the absent right wrist image is masked. The task text
comes from LeRobot metadata. Model actions retain π0.5's 32D padded shape.
Robot rollout must supply the same state ordering and action units as the
dataset and execute eight actions between TTT observations.

## π0.5 base checkpoint

The official initialization is `gs://openpi-assets/checkpoints/pi05_base/params`.
OpenPI downloads it automatically when training starts and caches it under
`$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params`, defaulting to
`~/.cache/openpi/openpi-assets/checkpoints/pi05_base/params`. To fetch it
explicitly on a machine where it is not already cached:

```bash
.venv/bin/python -c 'from openpi.shared.download import maybe_download; print(maybe_download("gs://openpi-assets/checkpoints/pi05_base/params"))'
```

This workspace already has a completed 12 GiB cache entry, so no duplicate
download is needed here. The V-JEPA teacher checkpoint is separate and already
exists at `data/weights/vjepa2/vjepa2_1_vitg_384.pt`.

The `CokeAnd1ce/JEPA_WAM` repository has task checkpoints for LIBERO and
RoboTwin. The two `*_vjepa_aux` checkpoints include a trained JEPA branch;
their corresponding non-auxiliary checkpoints do not. The main Piper run
starts from the official `pi05_base` to avoid carrying over a simulated task's
policy. If testing a JEPA-initialized transfer run, use
`pi05_libero_vjepa_aux/.../29999/params` as the first comparison: its 7D
action layout and 31-frame teacher offset match the Piper setup more closely
than the RoboTwin clean-20 checkpoint. Keep the Piper normalization statistics
and target files; do not load the LIBERO checkpoint's assets. This transfer
route has not yet been validated on a GPU. To select it for the JEPA Slurm job,
set `BASE_PARAMS` to that local `params` directory and use a new `EXP_NAME`.

## 1. Compute normalization statistics

```bash
.venv/bin/python scripts/compute_piper_norm_stats.py
```

This writes to `assets/pi05_piper/piper_pick_can_place_plate_150_v21`. Both
later configs read that same asset. The script uses the same 16-step clamped
action windows and joint-delta transform as the training loader. It excludes
the eight episode-level validation episodes selected with seed 2026 and records
their IDs in `norm_stats_provenance.json`. It reads only Parquet columns rather
than decoding 38,492 camera frames.

## 2. Precompute frozen JEPA targets

```bash
export DATASET_ROOT="$PWD/data/pick_can_place_plate_150_trimmed_v21"
export VJEPA_CHECKPOINT="$PWD/data/weights/vjepa2/vjepa2_1_vitg_384.pt"
export VJEPA_SOURCE_ROOT="$PWD/third_party/vjepa2"
export OUTPUT_ROOT="$PWD/data/vjepa_targets/piper_pick_can_place_plate_offset31"
export IMAGE_KEY=observation.images.front
export FUTURE_OFFSET=31

bash scripts/run_precompute_vjepa_targets.sh plan
GPU_IDS=0 bash scripts/run_precompute_vjepa_targets.sh smoke
GPU_IDS=0 bash scripts/run_precompute_vjepa_targets.sh run
```

The planning command reports about 58 GiB for future targets plus current
features. `smoke` writes one episode; `run` resumes and completes the rest.
Choose GPU IDs assigned to your job.

## 3. Train JEPA-WAM from π0.5 base

Run from the repository root after statistics and targets are complete:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
HF_DATASETS_CACHE="$PWD/data/.hf_datasets_cache" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  .venv/bin/python scripts/train.py pi05_piper_vjepa_aux \
  --exp-name piper_jepa --fsdp-devices 2 --batch-size 16 \
  --no-wandb-enabled
```

The starting schedule is 3,000 steps, 5e-5 learning rate, 500-step JEPA loss
warmup, and a 5% episode-level validation split. Adjust batch size and FSDP
to the actual GPU allocation. Resume with the same command plus `--resume`.

## 4. Train TTT from the completed JEPA checkpoint

Replace `<step>` with the saved JEPA checkpoint step:

```bash
JEPA_PARAMS="$PWD/checkpoints/pi05_piper_vjepa_aux/piper_jepa/best/<step>/params"
CUDA_VISIBLE_DEVICES=0,1 \
HF_DATASETS_CACHE="$PWD/data/.hf_datasets_cache" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  .venv/bin/python scripts/train.py pi05_piper_ttt_wrapper_explicit_jepa_ttt_only \
  --exp-name piper_ttt --weight-loader.params-path "$JEPA_PARAMS" \
  --fsdp-devices 2 --batch-size 2 --no-wandb-enabled
```

The TTT stage uses the precomputed current-frame V-JEPA features, a 16
observation window, stride 8, and TBPTT every 8 observations. It freezes the
JEPA-WAM policy and initializes only TTT parameters from scratch. Do not point
this stage directly at `pi05_base`: its JEPA predictor must first be trained.

## Slurm submission

Run `sbatch` from the repository root so relative log paths resolve under
`slurm_logs`. The Piper job files target the local `normal` partition: one H100
for target generation and one node with two H100s for each training stage.
They use the installed conda Python and local checkpoint cache. Confirm the
partition and account allocation on the cluster before submitting. The jobs
check GPU visibility at startup, and training runs resume their latest saved
step when submitted again with the same experiment name.

```bash
cd /lustre/fs1/home/ha865618/project/openpi_jepawam
mkdir -p slurm_logs
PRE=$(sbatch --parsable scripts/run_piper_vjepa_precompute.sbatch)
JEPA=$(sbatch --parsable --dependency=afterok:"$PRE" scripts/run_piper_jepa_train.sbatch)
sbatch --dependency=afterok:"$JEPA" scripts/run_piper_ttt_train.sbatch
```

The three files are `scripts/run_piper_vjepa_precompute.sbatch`,
`scripts/run_piper_jepa_train.sbatch`, and `scripts/run_piper_ttt_train.sbatch`.
If an account is required, add `-A <allocation>` to each `sbatch` command.
The TTT file reads the best validated JEPA checkpoint from
`checkpoints/pi05_piper_vjepa_aux/piper_jepa/best` after JEPA training
completes. To inspect a single episode before the full target run, submit the
precompute file once with `--export=ALL,MAX_EPISODES=1`; it writes reusable
episode output, but it does not authorize the dependent training job until all
150 episodes are complete. The full target tree needs about 58 GiB, plus room
for model checkpoints and temporary files.

## Approved joint real-robot run (2026-10-25)

Use `scripts/run_piper_joint_train.sbatch` for the joint experiment. The earlier
separate JEPA and TTT scripts remain alternative experiments; this run initializes
directly from official `pi05_base` and does not depend on a prior Piper checkpoint.

| Setting | Joint run |
| --- | --- |
| Config | `pi05_piper_jepa_ttt_joint` |
| Slurm | `normal`, two nodes, two H100 GPUs and four CPUs per node, 128 GiB RAM per node, 72 hours |
| Global batch | 64 temporal windows per optimizer update |
| Accumulation | Global microbatch 4 (one window per GPU), 16 microbatches |
| Temporal context | 16 observations, stride 8 frames, TBPTT segment 8 |
| Training | Policy backbone, action expert, JEPA predictor, four TTT wrappers jointly train |
| Loss | Action flow loss + warmed-up JEPA future-feature loss |
| JEPA weight | Linear warmup from 0 to 0.1 over 500 optimizer updates |
| Learning rate | Warmup 200 updates to 5e-5, then cosine decay to 1e-6 by update 10,000 |
| Optimizer | AdamW, global gradient clipping at 1.0 after accumulation |
| Max optimizer updates | 10,000; 72 hours is the job time limit, not a guaranteed completion time |
| EMA | Disabled; validation and exported checkpoint use the same weights |
| W&B project | `hang-yu-team/openpi-piper-real-robot-jepa-ttt` |
| Run name | `piper_real_joint_gbs64_20261025` |

Accumulation weights gradients by the number of valid observations. Left-padded
observations do not change the loss denominator. Learning-rate/JEPA warmup,
checkpoint cadence, and W&B steps all count **optimizer updates**, not microbatches.
The normal partition supplies only two H100s per node, so four-way FSDP spans
both nodes. At GBS 64, 3,000 updates would sample 192,000 temporal windows,
about 10.5 times the 18,264 available training windows. The 10,000-update cap
corresponds to 640,000 samples, about 35 passes; the best checkpoint still
comes from held-out validation, and the job may reach fewer updates in 72 hours.

Training uses 142 episodes; validation holds out episodes
`55, 76, 108, 109, 124, 127, 137, 148`, also excluded from normalization statistics.
Every 25 optimizer updates, evaluate the same 64 temporal windows spanning all
eight held-out episodes, with fixed noise/time RNG seeds and no augmentation.
The split uses seed 2026 at the episode level; seed 2027 selects a fixed set of
64 endpoints from those episodes (10, 7, 7, 4, 7, 11, 11, and 7 windows by
episode, respectively). This is a fixed subset of the held-out episodes, not
an evaluation of every one of their 982 frames at every checkpoint.
The first update and final update are also saved and validated. Validation uses
TTT memory updates within each window; memory starts empty for each window.

Choose the best checkpoint using the lowest `validation/flow_loss`. Also log
`validation/vjepa_loss` and training `flow_loss`, `vjepa_loss`, `vjepa_cosine`,
`vjepa_weight`, `weighted_vjepa_loss`, and `grad_norm`. Do not select on the total
weighted training loss, since its JEPA weight changes during warmup.

The root directory is
`checkpoints/pi05_piper_jepa_ttt_joint/piper_real_joint_gbs64_20261025/`:

- `latest/<step>/`: latest complete training state, refreshed every 25 updates.
- `best/<step>/`: checkpoint with lowest validation flow loss.
- `best_validation.json`: selected step and its validation flow loss.
- `wandb_id.txt`: run identity used for resuming.

Only the latest and best checkpoints are retained, with temporary extra disk
usage while a new checkpoint is finalized. A finished offline validation score
is a proxy; the final selection should be checked with repeated robot rollouts.
For a controlled adaptation comparison, load the same checkpoint twice and pass
`sample_kwargs={"ttt_update": True}` or `sample_kwargs={"ttt_update": False}` to
`create_trained_policy`. Reset the policy at each episode boundary. Disabling
updates retains the learned static TTT path; it does not remove the TTT modules.
Both inference modes still require current V-JEPA features. The current TTT path
reads current frozen V-JEPA tokens, not predicted future WAM tokens.

Submit from the repository root:

```bash
sbatch scripts/run_piper_joint_train.sbatch
```

The job validates cached features and W&B authentication before distributed
startup. The first real GPU microbatch checks the joint gradient path, and the
first full optimizer update checks save/validation; CPU preflight does not
establish full-model GPU memory fit. Preflight results are recorded under
`docs/training_checks/`.
