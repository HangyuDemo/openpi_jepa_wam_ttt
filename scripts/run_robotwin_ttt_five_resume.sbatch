#!/usr/bin/env bash
# Resume the finalized Clean-20 step-500 TTT checkpoint on the five paper tasks.
# Submit with --partition, --nodes, --ntasks, --gres and --time overrides.
#SBATCH -J rw-clean5-resume
#SBATCH -p normal
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=h100
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=400G
#SBATCH -t 3-00:00:00
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH -e slurm_logs/%x-%j.err

set -euo pipefail

ROOT="/lustre/fs1/home/ha865618/project/openpi_jepawam"
PYTHON_BIN="/lustre/fs1/home/ha865618/.conda/envs/openpi_jepawam/bin/python"
CONFIG_NAME="pi05_robotwin2_clean20_ttt_wrapper_explicit_jepa_ttt_only"
EXP_NAME="robotwin_clean20_ttt_wrapper_explicit_gbs16"
DATASET_ROOT="/lustre/fs1/home/ha865618/project/robotwin_lerobot/robotwin2_clean20_v21"
TARGET_ROOT="${ROOT}/data/vjepa_targets/robotwin2_clean20_v21_vjepa2_1_vitg_384_offset50"
BASE_CKPT="${ROOT}/pretrained/CokeAnd1ce_JEPA_WAM/checkpoints/openpi/pi05_robotwin_clean_20_vjepa_aux/pi05_robotwin_vjepa_delta50_no_geomaug_b256_fsdp4_gpu0123_seed42/14999"
ASSETS_ROOT="${BASE_CKPT}/assets"
CHECKPOINT_BASE_DIR="${ROOT}/checkpoints"
CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}"
EPISODE_INDICES_PATH="${ROOT}/scripts/robotwin_clean5_episode_indices.json"

export PYTHONPATH="${ROOT}/src:${ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TF_FORCE_GPU_ALLOW_GROWTH=true
unset JAX_LOCAL_DEVICE_IDS

cd "${ROOT}"
mkdir -p "${CHECKPOINT_DIR}"
# Both jobs enter scheduling independently. The first to run makes the other
# pending job depend on its completion, so waiting does not consume GPU time.
# The filesystem lock also covers the rare case where both start at once.
PAIR_FILE="${CHECKPOINT_DIR}/clean5_job_pair.txt"
[[ -f "${PAIR_FILE}" ]] || { echo "Missing job pair: ${PAIR_FILE}" >&2; exit 2; }
read -r NORMAL_JOB_ID HIGHGPU_JOB_ID < "${PAIR_FILE}"
case "${SLURM_JOB_ID}" in
    "${NORMAL_JOB_ID}") OTHER_JOB_ID="${HIGHGPU_JOB_ID}" ;;
    "${HIGHGPU_JOB_ID}") OTHER_JOB_ID="${NORMAL_JOB_ID}" ;;
    *) echo "Job ${SLURM_JOB_ID} is not in ${PAIR_FILE}" >&2; exit 2 ;;
esac
exec 9>"${CHECKPOINT_DIR}/.clean5_train.lock"
flock 9
OTHER_STATE="$(squeue --noheader --jobs "${OTHER_JOB_ID}" --format=%T)"
if [[ "${OTHER_STATE}" == PENDING ]]; then
    if scontrol update "JobId=${OTHER_JOB_ID}" "Dependency=afterany:${SLURM_JOB_ID}"; then
        echo "Job ${OTHER_JOB_ID} will wait in queue until job ${SLURM_JOB_ID} finishes"
    else
        echo "Could not add a dependency to job ${OTHER_JOB_ID}; filesystem lock still prevents concurrent training" >&2
    fi
fi
for path in "${PYTHON_BIN}" "${DATASET_ROOT}/meta/info.json" "${TARGET_ROOT}/manifest.json" \
    "${BASE_CKPT}/params" "${ASSETS_ROOT}/local/robotwin_clean_20/norm_stats.json" \
    "${EPISODE_INDICES_PATH}"; do
    [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 2; }
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
    )
    if not all(path.is_file() for path in required) or not utils.is_checkpoint_finalized(checkpoint):
        continue
    if json.loads(required[0].read_text()).get("commit_timestamp_nsecs"):
        complete.append(checkpoint)
if not complete:
    raise SystemExit(f"No finalized checkpoint at or after step 500 in {root}")
print(f"Resuming from {max(complete, key=lambda p: int(p.name))}", flush=True)
PY

case "${SLURM_JOB_PARTITION}" in
    normal) GPUS_PER_NODE=2; FSDP_DEVICES=4 ;;
    highgpu) GPUS_PER_NODE=4; FSDP_DEVICES=4 ;;
    *) echo "Unsupported partition: ${SLURM_JOB_PARTITION}" >&2; exit 2 ;;
esac
[[ "${SLURM_NTASKS}" -eq "${SLURM_JOB_NUM_NODES}" ]] || {
    echo "Expected one JAX process per node" >&2; exit 2;
}
export JAX_COORDINATOR_ADDRESS="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1):$((12000 + SLURM_JOB_ID % 1000))"
export ROOT PYTHON_BIN CONFIG_NAME EXP_NAME DATASET_ROOT TARGET_ROOT BASE_CKPT ASSETS_ROOT
export CHECKPOINT_BASE_DIR CHECKPOINT_DIR EPISODE_INDICES_PATH FSDP_DEVICES GPUS_PER_NODE

echo "job=${SLURM_JOB_ID} partition=${SLURM_JOB_PARTITION} nodes=${SLURM_JOB_NODELIST} checkpoint=${CHECKPOINT_DIR}/500"
echo "episodes=250 global_batch=16 fsdp=${FSDP_DEVICES} time_limit=3d"
srun --kill-on-bad-exit=1 bash -lc '
    set -euo pipefail
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    "${PYTHON_BIN}" -c "import torch; assert torch.cuda.device_count() == ${GPUS_PER_NODE}, torch.cuda.device_count()"
'
srun --kill-on-bad-exit=1 bash -lc '
    set -euo pipefail
    exec "${PYTHON_BIN}" scripts/train.py "${CONFIG_NAME}" \
        --exp-name "${EXP_NAME}" \
        --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}" \
        --weight-loader.params-path "${BASE_CKPT}/params" \
        --data.assets.assets-dir "${ASSETS_ROOT}" \
        --data.dataset-root "${DATASET_ROOT}" \
        --data.episode-indices-path "${EPISODE_INDICES_PATH}" \
        --data.vjepa-target-root "${TARGET_ROOT}" \
        --model.ttt-context-length 128 \
        --model.ttt-observation-stride 8 \
        --model.ttt-tbptt-step-size 8 \
        --batch-size 16 \
        --fsdp-devices "${FSDP_DEVICES}" \
        --num-workers 2 \
        --num-train-steps 30000 \
        --save-interval 500 \
        --keep-period 5000 \
        --resume
'
