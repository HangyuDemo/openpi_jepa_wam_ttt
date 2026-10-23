#!/usr/bin/env bash
# PSC Bridges-2: wrapper + explicit-JEPA (WE), with current-frame V-JEPA
# encoded online. DATASET=libero or DATASET=robotwin (Clean-20).
# Submit scripts/run_libero_we_online_psc.sbatch or
# scripts/run_robotwin_we_online_psc.sbatch from ROOT after `mkdir -p slurm_logs`.
# Re-submit with the same DATASET/EXP_NAME to resume the existing WE checkpoint.
# This TTT-only route optimizes action flow loss; it does not compute the
# auxiliary future-pair JEPA loss or read precomputed V-JEPA target files.
#SBATCH -J we-online
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=h100-80:4
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH -t 48:00:00
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH -e slurm_logs/%x-%j.err

set -euo pipefail

ROOT="${ROOT:-/ocean/projects/cis260071p/ha865618/project/openpi_jepa_wam_ttt}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
DATASET="${DATASET:-libero}"
VJEPA_CHECKPOINT="${VJEPA_CHECKPOINT:-${ROOT}/data/weights/vjepa2/vjepa2_1_vitg_384.pt}"
VJEPA_SOURCE_ROOT="${VJEPA_SOURCE_ROOT:-${ROOT}/third_party/vjepa2}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${ROOT}/checkpoints}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-128}"
OBSERVATION_STRIDE="${OBSERVATION_STRIDE:-8}"
TBPTT_STEP_SIZE="${TBPTT_STEP_SIZE:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_STEPS="${NUM_STEPS:-40000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
VJEPA_BATCH_SIZE="${VJEPA_BATCH_SIZE:-4}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
WANDB_ENABLED=1
export WANDB_API_KEY='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'
export WANDB_MODE=online
unset WANDB_DISABLED

case "${DATASET}" in
    libero)
        CONFIG_NAME=pi05_libero_ttt_wrapper_explicit_jepa_ttt_only
        EXP_NAME="${EXP_NAME:-libero_ttt_wrapper_explicit_gbs16}"
        DATASET_ROOT="${DATASET_ROOT:-${ROOT}/data/libero_lerobot}"
        ASSET_ID=physical-intelligence/libero
        ;;
    robotwin)
        CONFIG_NAME=pi05_robotwin2_clean20_ttt_wrapper_explicit_jepa_ttt_only
        EXP_NAME="${EXP_NAME:-robotwin_clean20_ttt_wrapper_explicit_gbs16}"
        DATASET_ROOT="${DATASET_ROOT:-/ocean/projects/cis260071p/ha865618/project/robotwin_lerobot/robotwin2_clean20_v21}"
        ASSET_ID=local/robotwin_clean_20
        ;;
    *) echo "DATASET must be libero or robotwin; got ${DATASET}" >&2; exit 2 ;;
esac
CACHE_ROOT="${CACHE_ROOT:-${ROOT}/cache/${DATASET}}"
CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}"

if (( BATCH_SIZE < 4 || BATCH_SIZE % 4 != 0 )); then
    echo "BATCH_SIZE must be positive and divisible by four" >&2
    exit 2
fi
# `--resume` silently starts from base weights when the directory is empty.
# Require the transferred full train state and original W&B run ID instead.
if [[ ! -s "${CHECKPOINT_DIR}/wandb_id.txt" ]]; then
    echo "Missing W&B run ID: ${CHECKPOINT_DIR}/wandb_id.txt" >&2
    exit 2
fi
latest_step_dir=""
for candidate in "${CHECKPOINT_DIR}"/[0-9]*; do
    [[ -d "${candidate}" && "${candidate##*/}" =~ ^[0-9]+$ ]] || continue
    if [[ -z "${latest_step_dir}" ]] || (( ${candidate##*/} > ${latest_step_dir##*/} )); then
        latest_step_dir="${candidate}"
    fi
done
if [[ -z "${latest_step_dir}" ]]; then
    echo "No numbered resume checkpoint found under ${CHECKPOINT_DIR}" >&2
    exit 2
fi
for path in "${latest_step_dir}/params/_METADATA" "${latest_step_dir}/train_state/_METADATA"; do
    [[ -s "${path}" ]] || { echo "Incomplete resume checkpoint: ${path}" >&2; exit 2; }
done
WEIGHT_PARAMS="${latest_step_dir}/params"
ASSETS_ROOT="${ASSETS_ROOT:-${latest_step_dir}/assets}"
for path in \
    "${PYTHON_BIN}" \
    "${ROOT}/scripts/train.py" \
    "${DATASET_ROOT}/meta/info.json" \
    "${ASSETS_ROOT}/${ASSET_ID}/norm_stats.json" \
    "${VJEPA_CHECKPOINT}" \
    "${VJEPA_SOURCE_ROOT}/app/vjepa_2_1/models/vision_transformer.py"; do
    [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 2; }
done

cd "${ROOT}"
mkdir -p "${CHECKPOINT_BASE_DIR}" "${CACHE_ROOT}"/datasets "${CACHE_ROOT}"/huggingface/hub
export PYTHONPATH="${ROOT}/src:${ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export TF_FORCE_GPU_ALLOW_GROWTH=true HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export ROOT PYTHON_BIN CONFIG_NAME EXP_NAME DATASET_ROOT WEIGHT_PARAMS ASSETS_ROOT
export VJEPA_CHECKPOINT VJEPA_SOURCE_ROOT CHECKPOINT_BASE_DIR CACHE_ROOT
export CONTEXT_LENGTH OBSERVATION_STRIDE TBPTT_STEP_SIZE BATCH_SIZE NUM_STEPS
export NUM_WORKERS VJEPA_BATCH_SIZE SAVE_INTERVAL WANDB_ENABLED

COORDINATOR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export JAX_COORDINATOR_ADDRESS="${COORDINATOR}:$((12000 + SLURM_JOB_ID % 1000))"
echo "job=${SLURM_JOB_ID} dataset=${DATASET} config=${CONFIG_NAME} exp=${EXP_NAME}"
echo "checkpoint=${CHECKPOINT_DIR} steps=${NUM_STEPS} batch=${BATCH_SIZE}"
echo "resuming_from=${latest_step_dir} wandb=online"
echo "online V-JEPA current: ${VJEPA_CHECKPOINT}"

# Slurm may expose one bound GPU or all four allocated GPUs to each task.
# In either case JAX and the PyTorch teacher must select the same single GPU.
srun --kill-on-bad-exit=1 --ntasks=4 --ntasks-per-node=4 bash -lc '
    set -euo pipefail
    gpu_count="$("${PYTHON_BIN}" -c "import torch; print(torch.cuda.device_count())")"
    case "${gpu_count}" in
        1) export JAX_LOCAL_DEVICE_IDS=0 ;;
        4) export JAX_LOCAL_DEVICE_IDS="${SLURM_LOCALID}" ;;
        *) echo "Rank ${SLURM_PROCID} sees ${gpu_count} GPUs; expected 1 or 4" >&2; exit 2 ;;
    esac
    "${PYTHON_BIN}" -c "import os, torch; d=int(os.environ[\"JAX_LOCAL_DEVICE_IDS\"]); torch.empty(1, device=f\"cuda:{d}\"); print(\"rank=%s gpu=%d %s\" % (os.environ.get(\"SLURM_PROCID\"), d, torch.cuda.get_device_name(d)), flush=True)"

    NODE_CACHE="${LOCAL:-${CACHE_ROOT}/tmp/${SLURM_JOB_ID}}"
    mkdir -p "${NODE_CACHE}/jax/${SLURM_PROCID}" "${NODE_CACHE}/tmp/${SLURM_PROCID}" \
        "${CACHE_ROOT}/openpi" "${CACHE_ROOT}/wandb/${SLURM_JOB_ID}/${SLURM_PROCID}"
    export HF_HOME="${CACHE_ROOT}/huggingface"
    export HF_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
    export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
    export OPENPI_DATA_HOME="${CACHE_ROOT}/openpi"
    export JAX_COMPILATION_CACHE_DIR="${NODE_CACHE}/jax/${SLURM_PROCID}"
    export TMPDIR="${NODE_CACHE}/tmp/${SLURM_PROCID}"
    export WANDB_DIR="${CACHE_ROOT}/wandb/${SLURM_JOB_ID}/${SLURM_PROCID}"

    args=(
        "${CONFIG_NAME}"
        --exp-name "${EXP_NAME}"
        --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}"
        --weight-loader.params-path "${WEIGHT_PARAMS}"
        --data.assets.assets-dir "${ASSETS_ROOT}"
        --data.dataset-root "${DATASET_ROOT}"
        --data.vjepa-target-root None
        --data.vjepa-online-current
        --data.vjepa-online-checkpoint "${VJEPA_CHECKPOINT}"
        --data.vjepa-online-source-root "${VJEPA_SOURCE_ROOT}"
        --data.vjepa-online-batch-size "${VJEPA_BATCH_SIZE}"
        --model.ttt-context-length "${CONTEXT_LENGTH}"
        --model.ttt-observation-stride "${OBSERVATION_STRIDE}"
        --model.ttt-tbptt-step-size "${TBPTT_STEP_SIZE}"
        --batch-size "${BATCH_SIZE}"
        --fsdp-devices 4
        --num-workers "${NUM_WORKERS}"
        --num-train-steps "${NUM_STEPS}"
        --save-interval "${SAVE_INTERVAL}"
        --resume
    )
    exec "${PYTHON_BIN}" scripts/train.py "${args[@]}"
'
