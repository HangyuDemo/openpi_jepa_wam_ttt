#!/usr/bin/env bash
# Precompute LIBERO future-pair targets and current-observation tokens.
# normal H100 PCIe nodes expose two GPUs each, so four workers use two nodes.
# The output is episode-granular and resumable: completed episodes are skipped.
#SBATCH -J openpi-libero-vjepa
#SBATCH -p normal
#SBATCH -N 2
#SBATCH -n 4
#SBATCH --ntasks-per-node=2
#SBATCH --constraint=h100
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH -c 16
#SBATCH --mem=400G
#SBATCH -t 1-00:00:00
#SBATCH --requeue
# evc22/evc48 also failed direct CUDA/nvidia-smi checks in prior launches.
# Keep the known drain/maintenance nodes out as well.
#SBATCH --exclude=evc22,evc24,evc25,evc26,evc28,evc34,evc42,evc45,evc48,evc50
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH -e slurm_logs/%x-%j.err

set -euo pipefail

ROOT="/lustre/fs1/home/ha865618/project/openpi_jepawam"
PYTHON_BIN="/lustre/fs1/home/ha865618/.conda/envs/openpi_jepawam/bin/python"
DATASET_ROOT="${DATASET_ROOT:-/home/ha865618/data/libero_lerobot}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/pretrained/CokeAnd1ce_JEPA_WAM/checkpoints/openpi/pi05_libero_vjepa_aux/pi05_vjepa_pair32_q64_w01_seed42_fsdp2_b128/29999/params}"
VJEPA_CHECKPOINT="${VJEPA_CHECKPOINT:-${ROOT}/data/weights/vjepa2/vjepa2_1_vitg_384.pt}"
VJEPA_SOURCE_ROOT="${VJEPA_SOURCE_ROOT:-${ROOT}/third_party/vjepa2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/data/vjepa_targets/libero_vjepa2_1_vitg_384_offset31}"
VJEPA_BATCH_SIZE="${VJEPA_BATCH_SIZE:-4}"

export PYTHONPATH="${ROOT}/src:${ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROOT PYTHON_BIN DATASET_ROOT CHECKPOINT VJEPA_CHECKPOINT VJEPA_SOURCE_ROOT OUTPUT_ROOT VJEPA_BATCH_SIZE

cd "${ROOT}"
mkdir -p "${OUTPUT_ROOT}/logs"

for path in "${DATASET_ROOT}/meta/info.json" "${VJEPA_CHECKPOINT}" "${VJEPA_SOURCE_ROOT}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required path: ${path}" >&2
        exit 2
    fi
done

COORDINATOR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
export JAX_COORDINATOR_ADDRESS="${COORDINATOR}:$((12000 + SLURM_JOB_ID % 1000))"
echo "job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} coordinator=${JAX_COORDINATOR_ADDRESS}"
echo "dataset=${DATASET_ROOT} output=${OUTPUT_ROOT} workers=${SLURM_NTASKS} batch=${VJEPA_BATCH_SIZE}"

srun --ntasks="${SLURM_NTASKS}" --ntasks-per-node=2 --gpu-bind=map_gpu:0,1 bash -lc '
    set -euo pipefail
    echo "worker rank=${SLURM_PROCID} host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"
    exec "${PYTHON_BIN}" scripts/precompute_vjepa_pair_targets.py \
        --dataset-root "${DATASET_ROOT}" \
        --checkpoint "${VJEPA_CHECKPOINT}" \
        --vjepa-source-root "${VJEPA_SOURCE_ROOT}" \
        --output-root "${OUTPUT_ROOT}" \
        --image-key image \
        --future-offset 31 \
        --batch-size "${VJEPA_BATCH_SIZE}" \
        --device cuda:0 \
        --worker-rank "${SLURM_PROCID}" \
        --world-size "${SLURM_NTASKS}"
'

"${PYTHON_BIN}" scripts/precompute_vjepa_pair_targets.py \
    --dataset-root "${DATASET_ROOT}" \
    --checkpoint "${VJEPA_CHECKPOINT}" \
    --vjepa-source-root "${VJEPA_SOURCE_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --image-key image \
    --future-offset 31 \
    --batch-size "${VJEPA_BATCH_SIZE}" \
    --status-only
