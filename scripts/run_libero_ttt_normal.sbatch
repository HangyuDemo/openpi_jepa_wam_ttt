#!/usr/bin/env bash
# Four-card LIBERO TTT training wrapper for normal H100 PCIe nodes.
# A normal node has two H100 PCIe cards, therefore this launches one JAX
# process per card over two nodes. Re-running the same experiment resumes the
# latest completed checkpoint automatically.
#SBATCH -J openpi-libero-ttt
#SBATCH -p normal
#SBATCH -N 2
#SBATCH -n 4
#SBATCH --ntasks-per-node=2
#SBATCH --constraint=h100
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=400G
#SBATCH -t 2-00:00:00
#SBATCH --requeue
# evc22/evc48 also failed direct CUDA/nvidia-smi checks in prior launches.
# Keep the known drain/maintenance nodes out as well.
#SBATCH --exclude=evc22,evc24,evc25,evc26,evc28,evc34,evc38,evc40,evc42,evc44,evc45,evc46,evc48,evc50
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH -e slurm_logs/%x-%j.err

set -euo pipefail

ROOT="/lustre/fs1/home/ha865618/project/openpi_jepawam"
PYTHON_BIN="/lustre/fs1/home/ha865618/.conda/envs/openpi_jepawam/bin/python"
CONFIG_NAME="${CONFIG_NAME:?Set CONFIG_NAME to one LIBERO TTT config}"
EXP_NAME="${EXP_NAME:?Set a unique EXP_NAME for this route}"
DATASET_ROOT="${DATASET_ROOT:-/home/ha865618/data/libero_lerobot}"
BASE_CKPT="${BASE_CKPT:-${ROOT}/pretrained/CokeAnd1ce_JEPA_WAM/checkpoints/openpi/pi05_libero_vjepa_aux/pi05_vjepa_pair32_q64_w01_seed42_fsdp2_b128/29999}"
TARGET_ROOT="${TARGET_ROOT:-${ROOT}/data/vjepa_targets/libero_vjepa2_1_vitg_384_offset31}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${ROOT}/checkpoints}"
BATCH_SIZE="${BATCH_SIZE:-16}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"
NUM_STEPS="${NUM_STEPS:-40000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"

export PYTHONPATH="${ROOT}/src:${ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# The LIBERO parquet files are staged locally. Never let concurrent ranks/jobs
# try to repair this shared dataset through Hugging Face downloads.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# Each local rank must see both GPUs so NCCL can establish peer connections.
unset JAX_LOCAL_DEVICE_IDS

cd "${ROOT}"
mkdir -p "${CHECKPOINT_BASE_DIR}" slurm_logs

for path in "${DATASET_ROOT}/meta/info.json" "${TARGET_ROOT}/manifest.json" "${BASE_CKPT}/params" "${BASE_CKPT}/assets/physical-intelligence/libero/norm_stats.json"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required path: ${path}" >&2
        exit 2
    fi
done

"${PYTHON_BIN}" - "${DATASET_ROOT}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
info = json.loads((root / "meta/info.json").read_text())
missing = []
for episode_index in range(info["total_episodes"]):
    relative_path = info["data_path"].format(
        episode_chunk=episode_index // info["chunks_size"],
        episode_index=episode_index,
    )
    path = root / relative_path
    if not path.is_file() or path.stat().st_size < 8:
        missing.append(str(path))
if missing:
    sys.exit(f"LIBERO dataset incomplete: {len(missing)} missing/empty parquet files; first: {missing[:5]}")
print(f"LIBERO dataset preflight OK: {info['total_episodes']} local parquet files", flush=True)
PY

CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}"
# Create the shared directory before launching ranks to avoid a startup race.
mkdir -p "${CHECKPOINT_DIR}"

if [[ "${WANDB_ENABLED}" == "0" ]]; then
    WANDB_DISABLED=1
else
    WANDB_DISABLED=0
fi

export ROOT PYTHON_BIN CONFIG_NAME EXP_NAME DATASET_ROOT BASE_CKPT TARGET_ROOT
export CHECKPOINT_BASE_DIR BATCH_SIZE FSDP_DEVICES NUM_STEPS NUM_WORKERS WANDB_ENABLED WANDB_DISABLED
export RESUME_ENABLED=1

COORDINATOR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
export JAX_COORDINATOR_ADDRESS="${COORDINATOR}:$((12000 + SLURM_JOB_ID % 1000))"
echo "job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} coordinator=${JAX_COORDINATOR_ADDRESS}"
echo "config=${CONFIG_NAME} exp=${EXP_NAME} gbs=${BATCH_SIZE} fsdp=${FSDP_DEVICES} steps=${NUM_STEPS}"
echo "checkpoint=${CHECKPOINT_DIR} resume_requested=1"

# Fail before JAX initialization if any assigned task cannot see a healthy GPU.
srun --kill-on-bad-exit=1 --ntasks="${SLURM_NTASKS}" --ntasks-per-node=2 --gres=gpu:2 bash -lc '
    set -euo pipefail
    echo "GPU preflight rank=${SLURM_PROCID} host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    "${PYTHON_BIN}" -c "import torch; assert torch.cuda.device_count() == 2, torch.cuda.device_count(); torch.empty(1, device=\"cuda:${SLURM_LOCALID}\"); print(torch.cuda.get_device_name(${SLURM_LOCALID}), flush=True)"
'

srun --kill-on-bad-exit=1 --ntasks="${SLURM_NTASKS}" --ntasks-per-node=2 --gres=gpu:2 bash -lc '
    set -euo pipefail
    echo "rank=${SLURM_PROCID} host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"
    train_args=(
        "${CONFIG_NAME}"
        --exp-name "${EXP_NAME}" \
        --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}" \
        --weight-loader.params-path "${BASE_CKPT}/params" \
        --data.dataset-root "${DATASET_ROOT}" \
        --data.assets.assets-dir "${BASE_CKPT}/assets" \
        --data.vjepa-target-root "${TARGET_ROOT}" \
        --batch-size "${BATCH_SIZE}" \
        --fsdp-devices "${FSDP_DEVICES}" \
        --num-workers "${NUM_WORKERS}" \
        --num-train-steps "${NUM_STEPS}" \
        --save-interval 500 \
        --keep-period 5000
    )
    if [[ "${RESUME_ENABLED}" == "1" ]]; then
        train_args+=(--resume)
    fi
    if [[ "${WANDB_DISABLED}" == "1" ]]; then
        train_args+=(--no-wandb-enabled)
    fi
    exec "${PYTHON_BIN}" scripts/train.py "${train_args[@]}"
'
