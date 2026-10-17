#!/usr/bin/env bash
# Four-card RoboTwin Clean-20 TTT-only training on normal H100 PCIe nodes.
# This route uses wrapper TTT with explicit current-frame V-JEPA memory and
# keeps every non-TTT parameter frozen through the *_ttt_only config.
# A normal node has two H100 PCIe cards, so this launches one JAX process per
# card over two nodes. Re-running the same experiment resumes automatically.
#SBATCH -J openpi-rw-w-exp
#SBATCH -p normal
#SBATCH -N 2
#SBATCH -n 4
#SBATCH --ntasks-per-node=2
#SBATCH --constraint=h100
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=400G
#SBATCH -t 5-00:00:00
#SBATCH --requeue
# evc22/evc48 failed direct CUDA/nvidia-smi checks in prior launches. Keep
# currently drained/down/maintenance nodes and other known bad nodes out too.
#SBATCH --exclude=evc22,evc24,evc25,evc26,evc28,evc34,evc38,evc40,evc42,evc44,evc45,evc46,evc48,evc50
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH -e slurm_logs/%x-%j.err

set -euo pipefail

ROOT="/lustre/fs1/home/ha865618/project/openpi_jepawam"
PYTHON_BIN="/lustre/fs1/home/ha865618/.conda/envs/openpi_jepawam/bin/python"
CONFIG_NAME="${CONFIG_NAME:-pi05_robotwin2_clean20_ttt_wrapper_explicit_jepa_ttt_only}"
EXP_NAME="${EXP_NAME:-robotwin_clean20_ttt_wrapper_explicit_gbs16}"
DATASET_ROOT="${DATASET_ROOT:-/lustre/fs1/home/ha865618/project/robotwin_lerobot/robotwin2_clean20_v21}"
BASE_CKPT="${BASE_CKPT:-${ROOT}/pretrained/CokeAnd1ce_JEPA_WAM/checkpoints/openpi/pi05_robotwin_clean_20_vjepa_aux/pi05_robotwin_vjepa_delta50_no_geomaug_b256_fsdp4_gpu0123_seed42/14999}"
ASSETS_ROOT="${ASSETS_ROOT:-${BASE_CKPT}/assets}"
TARGET_ROOT="${TARGET_ROOT:-${ROOT}/data/vjepa_targets/robotwin2_clean20_v21_vjepa2_1_vitg_384_offset50}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${ROOT}/checkpoints}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-128}"
OBSERVATION_STRIDE="${OBSERVATION_STRIDE:-8}"
TBPTT_STEP_SIZE="${TBPTT_STEP_SIZE:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"
NUM_STEPS="${NUM_STEPS:-30000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"

export PYTHONPATH="${ROOT}/src:${ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Each local rank must see both GPUs so NCCL can establish peer connections.
unset JAX_LOCAL_DEVICE_IDS

cd "${ROOT}"
mkdir -p "${CHECKPOINT_BASE_DIR}" slurm_logs

required_paths=(
    "${PYTHON_BIN}"
    "${DATASET_ROOT}/meta/info.json"
    "${TARGET_ROOT}/manifest.json"
    "${TARGET_ROOT}/targets/chunk-000/episode_000000.npy"
    "${TARGET_ROOT}/current/chunk-000/episode_000000.npy"
    "${BASE_CKPT}/params"
    "${ASSETS_ROOT}/local/robotwin_clean_20/norm_stats.json"
)
for path in "${required_paths[@]}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required path: ${path}" >&2
        exit 2
    fi
done

CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}"
# Create the shared directory before launching ranks to avoid a startup race.
mkdir -p "${CHECKPOINT_DIR}"

if [[ "${WANDB_ENABLED}" == "0" ]]; then
    WANDB_DISABLED=1
else
    WANDB_DISABLED=0
fi

export ROOT PYTHON_BIN CONFIG_NAME EXP_NAME DATASET_ROOT BASE_CKPT ASSETS_ROOT TARGET_ROOT
export CHECKPOINT_BASE_DIR CONTEXT_LENGTH OBSERVATION_STRIDE TBPTT_STEP_SIZE
export BATCH_SIZE FSDP_DEVICES NUM_STEPS NUM_WORKERS WANDB_ENABLED WANDB_DISABLED
export RESUME_ENABLED=1

COORDINATOR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
export JAX_COORDINATOR_ADDRESS="${COORDINATOR}:$((12000 + SLURM_JOB_ID % 1000))"
echo "job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} coordinator=${JAX_COORDINATOR_ADDRESS}"
echo "config=${CONFIG_NAME} exp=${EXP_NAME} gbs=${BATCH_SIZE} fsdp=${FSDP_DEVICES} steps=${NUM_STEPS}"
echo "context=${CONTEXT_LENGTH} stride=${OBSERVATION_STRIDE} tbptt=${TBPTT_STEP_SIZE}"
echo "data=${DATASET_ROOT} targets=${TARGET_ROOT}"
echo "checkpoint=${CHECKPOINT_DIR} resume_requested=1"

# Fail before model initialization if any assigned task cannot see a healthy GPU.
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
        --exp-name "${EXP_NAME}"
        --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}"
        --weight-loader.params-path "${BASE_CKPT}/params"
        --data.assets.assets-dir "${ASSETS_ROOT}"
        --data.dataset-root "${DATASET_ROOT}"
        --data.vjepa-target-root "${TARGET_ROOT}"
        --model.ttt-context-length "${CONTEXT_LENGTH}"
        --model.ttt-observation-stride "${OBSERVATION_STRIDE}"
        --model.ttt-tbptt-step-size "${TBPTT_STEP_SIZE}"
        --batch-size "${BATCH_SIZE}"
        --fsdp-devices "${FSDP_DEVICES}"
        --num-workers "${NUM_WORKERS}"
        --num-train-steps "${NUM_STEPS}"
        --save-interval 100
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
