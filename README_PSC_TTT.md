# PSC Bridges-2: LIBERO TTT-only with online V-JEPA current encoding

This route keeps the Pi0.5 JEPA query architecture, freezes every parameter outside TTT, and trains on action flow loss only. The explicit-JEPA TTT memory uses current-frame tokens encoded by a frozen V-JEPA teacher during training and inference. No precomputed `data/vjepa_targets/.../current` or `targets` directory is needed.

## Files to stage on PSC

Set `ROOT` to the PSC checkout. Git carries the code, but these inputs must be copied separately:

- `$ROOT/data/libero_lerobot` (LeRobot dataset, including `meta/info.json` and parquet data).
- `$ROOT/pretrained/CokeAnd1ce_JEPA_WAM/checkpoints/openpi/pi05_libero_vjepa_aux/pi05_vjepa_pair32_q64_w01_seed42_fsdp2_b128/29999` (at least `params` and `assets`).
- `$ROOT/data/weights/vjepa2/vjepa2_1_vitg_384.pt` (frozen teacher, about 16 GiB).
- `$ROOT/third_party/vjepa2` (teacher source; this directory is excluded from Git).

The training and evaluation Python environments need the repository dependencies. Set `PYTHON_BIN` to the training environment's Python 3.11 executable. The 4-GPU script checks the key paths and GPUs before starting JAX.

## Submit and resume

```bash
export ROOT=/path/to/psc/openpi_jepa_wam_ttt
export PYTHON_BIN=/path/to/psc/python
export CONFIG_NAME=pi05_libero_ttt_wrapper_explicit_jepa_ttt_only_online
mkdir -p "$ROOT/slurm_logs"
cd "$ROOT"
sbatch -A YOUR_PSC_ALLOCATION --export=ALL scripts/run_libero_online_h100.sbatch
```

The job requests one Bridges-2 `GPU-shared` node with four H100 80 GiB GPUs. Its walltime is 48 hours. Submit the same command again after a walltime stop; `--resume` loads the latest complete checkpoint. Keep the same `EXP_NAME`, `CONFIG_NAME`, and checkpoint directory across submissions. Set `WANDB_ENABLED=1` if the environment has a working W&B login.

## Checkpoint and validation rules

- At every **200 completed global steps**, save a full resume checkpoint under `latest/<step>`. Orbax keeps only the newest numbered step in `latest`.
- At every **400 completed global steps**, and at the final step, measure action `flow_loss` on a fixed validation set. A seeded 5% of LIBERO episodes is held out from training; the same 8 global batches (32 temporal windows at global batch size 4) are used at every validation. The online teacher object is shared between the training and validation loaders on each GPU.
- Only a **strictly lower** validation flow loss saves a new full checkpoint under `best/<step>`. Orbax keeps only the newest best checkpoint. `best_validation.json` records its step and score. Equal or worse results leave `best` unchanged.
- `latest/<step>` is for resuming training. `best/<step>` is for selecting a checkpoint by held-out action loss. This validation does not measure LIBERO simulation success rate.

For inference, use the matching `_online` config and a `params`/`assets` checkpoint step. The policy server online-encodes the current camera frame before calling the TTT model; no stored current-token array is needed.

The first PSC run still needs a short smoke check on actual H100 GPUs. The local CPU tests cover configuration, fixed episode splitting, checkpoint aliases, and a TTT forward/backward pass without future targets.
