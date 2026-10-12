#!/usr/bin/env bash
# Resume RoboTwin Clean-20 TTT training on one 8x H100 highgpu node.
# Keep the experiment directory and global batch size from the 4-GPU run.
#SBATCH -J openpi-rw-8-resume
#SBATCH -p highgpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=h100
#SBATCH --gres=gpu:8
#SBATCH -c 32
#SBATCH --mem=400G
#SBATCH -t 2-00:00:00
#SBATCH --requeue
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH -e slurm_logs/%x-%j.err

set -euo pipefail

ROOT="/lustre/fs1/home/ha865618/project/openpi_jepawam"
PYTHON_BIN="/lustre/fs1/home/ha865618/.conda/envs/openpi_jepawam/bin/python"
CONFIG_NAME="pi05_robotwin2_clean20_ttt_wrapper_explicit_jepa_ttt_only"
EXP_NAME="robotwin_clean20_ttt_wrapper_explicit_gbs16"
DATASET_ROOT="/lustre/fs1/home/ha865618/project/robotwin_lerobot/robotwin2_clean20_v21"
BASE_CKPT="${ROOT}/pretrained/CokeAnd1ce_JEPA_WAM/checkpoints/openpi/pi05_robotwin_clean_20_vjepa_aux/pi05_robotwin_vjepa_delta50_no_geomaug_b256_fsdp4_gpu0123_seed42/14999"
ASSETS_ROOT="${BASE_CKPT}/assets"
TARGET_ROOT="${ROOT}/data/vjepa_targets/robotwin2_clean20_v21_vjepa2_1_vitg_384_offset50"
CHECKPOINT_BASE_DIR="${ROOT}/checkpoints"
CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}"

export PYTHONPATH="${ROOT}/src:${ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
unset JAX_LOCAL_DEVICE_IDS

cd "${ROOT}"
mkdir -p slurm_logs

for path in \
    "${DATASET_ROOT}/meta/info.json" \
    "${TARGET_ROOT}/manifest.json" \
    "${BASE_CKPT}/params" \
    "${ASSETS_ROOT}/local/robotwin_clean_20/norm_stats.json"; do
    if [[ ! -f "${path}" && ! -d "${path}" ]]; then
        echo "Missing required path: ${path}" >&2
        exit 2
    fi
done

"${PYTHON_BIN}" - "${CHECKPOINT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

from orbax.checkpoint import utils

root = Path(sys.argv[1])
complete = []
for checkpoint in root.iterdir():
    if not checkpoint.name.isdigit() or int(checkpoint.name) < 500:
        continue
    required = (
        checkpoint / "_CHECKPOINT_METADATA",
        checkpoint / "train_state/manifest.ocdbt",
        checkpoint / "params/manifest.ocdbt",
        checkpoint / "assets/local/robotwin_clean_20/norm_stats.json",
    )
    if not all(path.is_file() for path in required) or not utils.is_checkpoint_finalized(checkpoint):
        continue
    metadata = json.loads(required[0].read_text())
    if metadata.get("commit_timestamp_nsecs"):
        complete.append(checkpoint)
if not complete:
    raise SystemExit(f"No finalized checkpoint at or after step 500 in {root}")
print(f"Resuming from finalized checkpoint: {max(complete, key=lambda p: int(p.name))}", flush=True)
PY

echo "job=${SLURM_JOB_ID} node=${SLURM_JOB_NODELIST} config=${CONFIG_NAME} exp=${EXP_NAME}"
echo "gpus=8 global_batch=16 fsdp=8 steps=30000 time_limit=2d"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

srun --kill-on-bad-exit=1 --ntasks=1 --gres=gpu:8 \
    "${PYTHON_BIN}" scripts/train.py "${CONFIG_NAME}" \
    --exp-name "${EXP_NAME}" \
    --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}" \
    --weight-loader.params-path "${BASE_CKPT}/params" \
    --data.assets.assets-dir "${ASSETS_ROOT}" \
    --data.dataset-root "${DATASET_ROOT}" \
    --data.vjepa-target-root "${TARGET_ROOT}" \
    --model.ttt-context-length 128 \
    --model.ttt-observation-stride 8 \
    --model.ttt-tbptt-step-size 8 \
    --batch-size 16 \
    --fsdp-devices 8 \
    --num-workers 2 \
    --num-train-steps 30000 \
    --save-interval 500 \
    --keep-period 5000 \
    --resume
