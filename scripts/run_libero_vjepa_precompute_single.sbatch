#!/usr/bin/env bash
# Single-H100 LIBERO V-JEPA precompute.
# The output is episode-granular and resumable, so this can continue a
# multi-GPU or interrupted preprocessing run in the same output directory.
#SBATCH -J openpi-libero-vjepa-1gpu
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=h100
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH -c 16
#SBATCH --mem=400G
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH --exclude=evc22,evc24,evc25,evc26,evc28,evc34,evc42,evc45,evc48,evc50
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH -e slurm_logs/%x-%j.err

set -euo pipefail

ROOT="/lustre/fs1/home/ha865618/project/openpi_jepawam"
PYTHON_BIN="/lustre/fs1/home/ha865618/.conda/envs/openpi_jepawam/bin/python"
DATASET_ROOT="${DATASET_ROOT:-/home/ha865618/data/libero_lerobot}"
VJEPA_CHECKPOINT="${VJEPA_CHECKPOINT:-${ROOT}/data/weights/vjepa2/vjepa2_1_vitg_384.pt}"
VJEPA_SOURCE_ROOT="${VJEPA_SOURCE_ROOT:-${ROOT}/third_party/vjepa2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/data/vjepa_targets/libero_vjepa2_1_vitg_384_offset31}"
VJEPA_BATCH_SIZE="${VJEPA_BATCH_SIZE:-8}"

export PYTHONPATH="${ROOT}/src:${ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROOT PYTHON_BIN DATASET_ROOT VJEPA_CHECKPOINT VJEPA_SOURCE_ROOT OUTPUT_ROOT VJEPA_BATCH_SIZE

cd "${ROOT}"
mkdir -p "${OUTPUT_ROOT}/logs"

for path in "${DATASET_ROOT}/meta/info.json" "${VJEPA_CHECKPOINT}" "${VJEPA_SOURCE_ROOT}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required path: ${path}" >&2
        exit 2
    fi
done

echo "job=${SLURM_JOB_ID} host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"
echo "dataset=${DATASET_ROOT} output=${OUTPUT_ROOT} batch=${VJEPA_BATCH_SIZE}"

exec srun --ntasks=1 --gpu-bind=single:1 bash -lc '
    set -euo pipefail
    exec "${PYTHON_BIN}" scripts/precompute_vjepa_pair_targets.py \
        --dataset-root "${DATASET_ROOT}" \
        --checkpoint "${VJEPA_CHECKPOINT}" \
        --vjepa-source-root "${VJEPA_SOURCE_ROOT}" \
        --output-root "${OUTPUT_ROOT}" \
        --image-key image \
        --future-offset 31 \
        --batch-size "${VJEPA_BATCH_SIZE}" \
        --device cuda:0 \
        --worker-rank 0 \
        --world-size 1
'
