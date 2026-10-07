# RoboTwin 2.0 LeRobot v2.1 training

This repository has single-task pilot configurations: `pi05_robotwin2_beat_block_hammer`
(Pi0.5) and `pi05_robotwin2_beat_block_hammer_vjepa_aux` (Pi0.5 plus the V-JEPA
future-representation auxiliary loss). Both read the local `beat_block_hammer`
dataset at `/lustre/fs1/home/ha865618/project/robotwin_lerobot/robotwin2_beat_block_hammer_v21_full`.
This dataset has 50 episodes, 5,682 frames at 15 FPS, three RGB cameras, and
14D state/action vectors. The configs are a small 3,000-step pilot, not a
multi-task RoboTwin model. Separate `pi05_robotwin2_clean20` and
`pi05_robotwin2_clean20_vjepa_aux` configurations select the paper's 20 Clean
tasks. Their optimizer, batch-size, and training-step defaults remain pilot
settings, not a claimed reproduction of the released checkpoint.

The local conversion produces next-frame absolute joint targets. The current
training transform applies Aloha/pi joint-sign and gripper conversions, then
converts the first six joint values of each arm to state-relative deltas; the
gripper values (indices 6 and 13) stay absolute in pi space. Inference reverses
these conversions. **The paper does not specify whether the released RoboTwin
Pi0.5 + JEPA checkpoint uses these transforms. Do not assume this local choice
matches that checkpoint.** Once the authors confirm the action convention, use
`--data.no-adapt-to-pi` and/or `--data.no-use-delta-joint-actions` if needed,
then recompute normalization statistics with the final settings.
Pi0.5 still uses its pretrained 32D action head: the 14D data are padded by
the model transform, and inference returns only the first 14 dimensions.
The paper describes x-prediction for its RoboTwin JEPA-WAM action expert, but
its Pi0.5 transfer section says the original Pi0.5 action pathway is retained.
This pilot keeps OpenPI's original flow-matching objective; whether the
released RoboTwin Pi0.5 checkpoint used x-prediction remains unconfirmed.

Use the repository's Python 3.11 environment, created with `uv sync`. The
existing `jepa_wam` Python 3.10 environment has LeRobot 0.3.3 but lacks the
older `lerobot.common.datasets` API that this OpenPI checkout imports. The
OpenPI lockfile pins a LeRobot revision that reads v2.1 data. Do not replace
that pinned dependency with the `jepa_wam` installation.

From the `openpi_jepawam` root, calculate normalization stats once using the
plain Pi0.5 config:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_robotwin2_beat_block_hammer
```

This writes stats under
`assets/pi05_robotwin2_beat_block_hammer/robotwin2_beat_block_hammer_v21_full`.
Both configurations use these same stats because their state/action transforms
are identical. Check that `norm_stats.json` exists before training.

For a plain Pi0.5 pilot, from a GPU allocation with enough memory:

```bash
uv run scripts/train.py pi05_robotwin2_beat_block_hammer \
  --exp-name beat_block_hammer_pi05 \
  --batch-size 32 --num-train-steps 3000 --no-wandb-enabled
```

The JEPA auxiliary configuration additionally needs a V-JEPA 2.1 ViT-g/384
pretraining checkpoint, the `facebookresearch/vjepa2` source tree, and
precomputed targets for **this exact dataset**. The current implementation
selects the top camera feature `observation.images.cam_high` for the JEPA target.
The paper establishes three RoboTwin policy camera views but does not identify
the teacher target view(s) for the released Pi0.5 checkpoint. Confirm this
selection before reproducing that checkpoint. The paper specifies a future
offset of 50 frames:

```bash
export DATASET_ROOT=/lustre/fs1/home/ha865618/project/robotwin_lerobot/robotwin2_beat_block_hammer_v21_full
export VJEPA_SOURCE_ROOT=/path/to/vjepa2
export VJEPA_CHECKPOINT=/path/to/vjepa2_vitg_384.pt
export IMAGE_KEY=observation.images.cam_high
export FUTURE_OFFSET=50
export OUTPUT_ROOT=./data/vjepa_targets/robotwin2_beat_block_hammer_v21_full_vjepa2_1_vitg_384_offset50
bash scripts/run_precompute_vjepa_targets.sh plan
bash scripts/run_precompute_vjepa_targets.sh smoke
bash scripts/run_precompute_vjepa_targets.sh run
bash scripts/run_precompute_vjepa_targets.sh status
```

The wrapper uses `.venv/bin/python` unless `PYTHON_BIN` is set. Run it in a
GPU allocation for `smoke` and `run`. The future-pair target and the matching
single-view current representation each use roughly 9 GiB for 5,682 frames,
before temporary files. The preprocessor writes future supervision under
`targets/` and current explicit TTT K/V under `current/`. It can upgrade a
format-1 manifest and retain valid existing targets, so rerunning resumes by
computing only missing current arrays. Once `manifest.json` and every episode
array are present, run:

```bash
uv run scripts/train.py pi05_robotwin2_beat_block_hammer_vjepa_aux \
  --exp-name beat_block_hammer_pi05_vjepa \
  --batch-size 32 --num-train-steps 3000 --no-wandb-enabled
```

Both RoboTwin configs now use the paper's 50-step action horizon. Recompute
normalization statistics after changing the horizon, since the statistics
script samples action chunks using the configured horizon. Existing offset-31
JEPA targets cannot be reused: regenerate them with `FUTURE_OFFSET=50`.

The JEPA loader checks target shape, dtype, frame count, image key, and future
offset before training. The target directory can be overridden with
`--data.vjepa-target-root`. The local LeRobot dataset directory can be changed
with `--data.dataset-root` for either configuration. The Pi0.5 base parameters
are loaded from the official `gs://openpi-assets/checkpoints/pi05_base/params`
path; the new JEPA layers are initialized separately. These commands configure
training only; they do not provide a RoboTwin simulator evaluation adapter.

## Paper Clean-20 data selection

Appendix D/Table 14 identifies exactly 20 RoboTwin tasks. The existing
`robotwin2_demo_clean_all_v21` dataset contains more tasks and must not be
used as a substitute. This wrapper validates all 20 raw HDF5 task directories
and creates a separate LeRobot v2.1 dataset without overwriting an existing
target:

```bash
bash scripts/prepare_robotwin_clean20.sh plan
PYTHON_BIN=/absolute/path/to/python-with-lerobot-0.3.3 \
  bash scripts/prepare_robotwin_clean20.sh run
```

It writes `robotwin2_clean20_v21` under `HF_LEROBOT_HOME` (by default
`/lustre/fs1/home/ha865618/project/robotwin_lerobot`). It uses all available
Clean demonstrations for each listed task; the paper does not disclose the
exact number used for the released Pi0.5 checkpoint. Conversion is time- and
storage-intensive, so `plan` does not start it.

After conversion, compute statistics for the Clean-20 dataset separately:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_robotwin2_clean20
```

For JEPA training, precompute targets from this exact dataset with
`IMAGE_KEY=observation.images.cam_high`, `FUTURE_OFFSET=50`, and
`OUTPUT_ROOT=./data/vjepa_targets/robotwin2_clean20_v21_vjepa2_1_vitg_384_offset50`.
The current-observation JEPA representation is always generated from the same
`IMAGE_KEY` view as the future-pair target.
The image key remains provisional until the checkpoint's teacher views are
confirmed. The eight `pi05_robotwin2_clean20_ttt_*` configurations inherit
the 50-step horizon and 50-frame offset, but still require a trained and
compatible Pi0.5 + JEPA checkpoint and confirmed action transforms.
