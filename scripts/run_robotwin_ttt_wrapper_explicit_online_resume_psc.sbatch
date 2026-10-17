#!/usr/bin/env bash
# PSC Bridges-2: inline implicit-full TTT, action flow loss only.
# Submit from ROOT after creating slurm_logs; WANDB_API_KEY is required.
#SBATCH -J lib-i-implicit
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
export WANDB_API_KEY='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'
ROOT="${ROOT:-/ocean/projects/cis260071p/ha865618/project/openpi_jepa_wam_ttt}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
CONFIG_NAME="${CONFIG_NAME:-pi05_robotwin2_clean20_ttt_wrapper_explicit_jepa_ttt_only}"
EXP_NAME="${EXP_NAME:-robotwin_clean20_ttt_wrapper_explicit_gbs16}"
DATASET_ROOT="${DATASET_ROOT:-/ocean/projects/cis260071p/ha865618/project/robotwin_lerobot/robotwin2_clean20_v21}"
VJEPA_CHECKPOINT="${VJEPA_CHECKPOINT:-${ROOT}/data/weights/vjepa2/vjepa2_1_vitg_384.pt}"
VJEPA_SOURCE_ROOT="${VJEPA_SOURCE_ROOT:-${ROOT}/third_party/vjepa2}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${ROOT}/checkpoints}"
CACHE_ROOT="${CACHE_ROOT:-/ocean/projects/cis260071p/ha865618/cache/robotwin}"
CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}"
RESUME_CHECKPOINT="${CHECKPOINT_DIR}/500"
BASE_CKPT="${BASE_CKPT:-${ROOT}/pretrained/CokeAnd1ce_JEPA_WAM/checkpoints/openpi/pi05_robotwin_clean_20_vjepa_aux/pi05_robotwin_vjepa_delta50_no_geomaug_b256_fsdp4_gpu0123_seed42/14999}"
# The resumed checkpoint already contains the normalization statistics.  This
# default lets a run proceed even when only the 500 checkpoint was transferred;
# set ASSETS_ROOT explicitly if you also keep the base checkpoint on PSC.
ASSETS_ROOT="${ASSETS_ROOT:-${RESUME_CHECKPOINT}/assets}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-128}"
OBSERVATION_STRIDE="${OBSERVATION_STRIDE:-8}"
TBPTT_STEP_SIZE="${TBPTT_STEP_SIZE:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"
NUM_STEPS="${NUM_STEPS:-30000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
VJEPA_BATCH_SIZE="${VJEPA_BATCH_SIZE:-4}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"

if [[ "${WANDB_ENABLED}" != "1" ]]; then
    echo "WANDB_ENABLED must remain 1 for this script; export a valid WANDB_API_KEY." >&2
    exit 2
fi
: "${WANDB_API_KEY:?Export WANDB_API_KEY before sbatch; the key is intentionally not stored in this file}"
export WANDB_API_KEY WANDB_MODE=online

export PYTHONPATH="${ROOT}/src:${ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export TF_FORCE_GPU_ALLOW_GROWTH=true
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "${ROOT}"
mkdir -p "${CHECKPOINT_BASE_DIR}" slurm_logs "${CACHE_ROOT}"/datasets "${CACHE_ROOT}"/huggingface "${CACHE_ROOT}"/openpi

required_paths=(
    "${PYTHON_BIN}"
    "${ROOT}/scripts/train.py"
    "${DATASET_ROOT}/meta/info.json"
    "${ASSETS_ROOT}/local/robotwin_clean_20/norm_stats.json"
    "${VJEPA_CHECKPOINT}"
    "${VJEPA_SOURCE_ROOT}/app/vjepa_2_1/models/vision_transformer.py"
    "${RESUME_CHECKPOINT}/params/_METADATA"
    "${RESUME_CHECKPOINT}/train_state/_METADATA"
)
for path in "${required_paths[@]}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required path: ${path}" >&2
        exit 2
    fi
done
if [[ -e "${BASE_CKPT}/params" ]]; then
    WEIGHT_PARAMS="${BASE_CKPT}/params"
elif [[ -e "${RESUME_CHECKPOINT}/params" ]]; then
    # Not used while --resume is active, but keeps the CLI value valid when
    # only the transferred resume checkpoint is present.
    WEIGHT_PARAMS="${RESUME_CHECKPOINT}/params"
else
    echo "Missing both base and resume parameter trees" >&2
    exit 2
fi

"${PYTHON_BIN}" - "${DATASET_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
info = json.loads((root / "meta/info.json").read_text())
expected = int(info["total_episodes"])
episodes = list((root / "data").glob("chunk-*/episode_*.parquet"))
if len(episodes) != expected:
    raise SystemExit(f"RoboTwin parquet count mismatch: {len(episodes)}/{expected}")
print(f"RoboTwin dataset OK: {len(episodes)} episodes")
PY

export ROOT PYTHON_BIN CONFIG_NAME EXP_NAME DATASET_ROOT BASE_CKPT ASSETS_ROOT WEIGHT_PARAMS
export VJEPA_CHECKPOINT VJEPA_SOURCE_ROOT CHECKPOINT_BASE_DIR CHECKPOINT_DIR CACHE_ROOT
export CONTEXT_LENGTH OBSERVATION_STRIDE TBPTT_STEP_SIZE BATCH_SIZE FSDP_DEVICES
export NUM_STEPS NUM_WORKERS VJEPA_BATCH_SIZE WANDB_ENABLED
export RESUME_ENABLED=1

COORDINATOR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
export JAX_COORDINATOR_ADDRESS="${COORDINATOR}:$((12000 + SLURM_JOB_ID % 1000))"
echo "job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} coordinator=${JAX_COORDINATOR_ADDRESS}"
echo "config=${CONFIG_NAME} exp=${EXP_NAME} resume=${RESUME_CHECKPOINT}"
echo "data=${DATASET_ROOT} vjepa=${VJEPA_CHECKPOINT} cache=${CACHE_ROOT}"

# Depending on the Bridges-2 GRES binding, a task can see either its one bound
# GPU or both GPUs allocated to its node.  In both cases JAX and the online
# V-JEPA encoder must use exactly one GPU per process.
srun --kill-on-bad-exit=1 --ntasks=4 --ntasks-per-node=2 --gres=gpu:2 bash -lc '
    set -euo pipefail
    gpu_count="$(${PYTHON_BIN} -c "import torch; print(torch.cuda.device_count())")"
    case "${gpu_count}" in
        1) export JAX_LOCAL_DEVICE_IDS=0 ;;
        2) export JAX_LOCAL_DEVICE_IDS="${SLURM_LOCALID}" ;;
        *) echo "Rank ${SLURM_PROCID} sees ${gpu_count} GPUs; expected 1 or 2" >&2; exit 2 ;;
    esac
    "${PYTHON_BIN}" -c "import os, torch; d=int(os.environ['JAX_LOCAL_DEVICE_IDS']); torch.empty(1, device=f'cuda:{d}'); print('rank=%s gpu=%d name=%s' % (os.environ['SLURM_PROCID'], d, torch.cuda.get_device_name(d)), flush=True)"
'

srun --kill-on-bad-exit=1 --ntasks=4 --ntasks-per-node=2 --gres=gpu:2 bash -lc '
    set -euo pipefail
    gpu_count="$(${PYTHON_BIN} -c "import torch; print(torch.cuda.device_count())")"
    case "${gpu_count}" in
        1) export JAX_LOCAL_DEVICE_IDS=0 ;;
        2) export JAX_LOCAL_DEVICE_IDS="${SLURM_LOCALID}" ;;
        *) echo "Rank ${SLURM_PROCID} sees ${gpu_count} GPUs; expected 1 or 2" >&2; exit 2 ;;
    esac

    # Keep Hugging Face Arrow materialization out of the 25 GB HOME quota.
    # This project cache is shared and reused by later jobs; JAX/TMP remain
    # node-local when PSC provides $LOCAL.
    NODE_CACHE="${LOCAL:-${CACHE_ROOT}/tmp/${SLURM_JOB_ID}/${SLURM_NODEID}}"
    mkdir -p "${CACHE_ROOT}/datasets" "${CACHE_ROOT}/huggingface/hub" "${CACHE_ROOT}/openpi" \
        "${CACHE_ROOT}/wandb/${SLURM_JOB_ID}/${SLURM_PROCID}" "${NODE_CACHE}/jax" "${NODE_CACHE}/tmp"
    export HF_HOME="${CACHE_ROOT}/huggingface"
    export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
    export HF_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
    export OPENPI_DATA_HOME="${CACHE_ROOT}/openpi"
    export TMPDIR="${NODE_CACHE}/tmp"
    export JAX_COMPILATION_CACHE_DIR="${NODE_CACHE}/jax"
    export WANDB_DIR="${CACHE_ROOT}/wandb/${SLURM_JOB_ID}/${SLURM_PROCID}"

    echo "rank=${SLURM_PROCID} host=$(hostname) local_gpu=${JAX_LOCAL_DEVICE_IDS}"
    train_args=(
        "${CONFIG_NAME}"
        --exp-name "${EXP_NAME}"
        --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}"
        --weight-loader.params-path "${WEIGHT_PARAMS}"
        --data.assets.assets-dir "${ASSETS_ROOT}"
        --data.dataset-root "${DATASET_ROOT}"
        # The old checkpoint used precomputed targets.  Null disables that
        # tree; online V-JEPA supplies only the current explicit K/V tokens.
        --data.vjepa-target-root None
        --data.vjepa-online-current
        --data.vjepa-online-checkpoint "${VJEPA_CHECKPOINT}"
        --data.vjepa-online-source-root "${VJEPA_SOURCE_ROOT}"
        --data.vjepa-online-batch-size "${VJEPA_BATCH_SIZE}"
        --model.ttt-context-length "${CONTEXT_LENGTH}"
        --model.ttt-observation-stride "${OBSERVATION_STRIDE}"
        --model.ttt-tbptt-step-size "${TBPTT_STEP_SIZE}"
        --batch-size "${BATCH_SIZE}"
        --fsdp-devices "${FSDP_DEVICES}"
        --num-workers "${NUM_WORKERS}"
        --num-train-steps "${NUM_STEPS}"
        --save-interval 500
        --keep-period 5000
        --resume
    )
    exec "${PYTHON_BIN}" scripts/train.py "${train_args[@]}"
'
