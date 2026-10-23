#!/bin/bash
# TACC Stampede3 h100: 1 node x 4 GPUs, one JAX process per GPU.
# Copy this file to /scratch/11441/demo23331/project/openpi_jepa_wam_ttt/.
# From that directory: ROUTE=II sbatch -A <allocation> run_libero_ii_ie_tacc_stampede3.slurm
# Use ROUTE=IE for inline explicit JEPA. The two routes have separate checkpoints.
# Stampede3 does not accept --gres, --gpus-per-task, or --mem.
#SBATCH -J libero-ii-ie
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH -c 12
#SBATCH -t 48:00:00
#SBATCH --chdir=/scratch/11441/demo23331/project/openpi_jepa_wam_ttt
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

ROOT="${ROOT:-/scratch/11441/demo23331/project/openpi_jepa_wam_ttt}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/scratch/11441/demo23331/project/openpi_jepa_wam_ttt/data/libero_lerobot}"
BASE_CKPT="${BASE_CKPT:-/scratch/11441/demo23331/project/openpi_jepa_wam_ttt/pretrained/CokeAnd1ce_JEPA_WAM/checkpoints/openpi/pi05_libero_vjepa_aux/pi05_vjepa_pair32_q64_w01_seed42_fsdp2_b128/29999}"
VJEPA_CHECKPOINT="${VJEPA_CHECKPOINT:-${ROOT}/data/weights/vjepa2/vjepa2_1_vitg_384.pt}"
VJEPA_SOURCE_ROOT="${VJEPA_SOURCE_ROOT:-${ROOT}/third_party/vjepa2}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${ROOT}/checkpoints}"
ROUTE="${ROUTE:?Set ROUTE=II or ROUTE=IE before sbatch}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-64}"
TBPTT_STEP_SIZE="${TBPTT_STEP_SIZE:-4}"
NUM_STEPS="${NUM_STEPS:-40000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
WANDB_ENABLED="${WANDB_ENABLED:-0}"

case "${ROUTE^^}" in
    II)
        CONFIG_NAME=pi05_libero_ttt_inline_implicit_full_ttt_only_online
        EXP_NAME="${EXP_NAME:-inline_implicit_ctx${CONTEXT_LENGTH}_gbs16}"
        ;;
    IE)
        CONFIG_NAME=pi05_libero_ttt_inline_explicit_jepa_ttt_only_online
        EXP_NAME="${EXP_NAME:-inline_explicit_ctx${CONTEXT_LENGTH}_gbs16}"
        ;;
    *) echo "ROUTE must be II or IE, got: ${ROUTE}" >&2; exit 2 ;;
esac

if [[ "${SLURM_NTASKS:-}" != 4 || "${SLURM_JOB_NUM_NODES:-}" != 1 ]]; then
    echo "Expected 4 Slurm tasks on 1 Stampede3 H100 node" >&2
    exit 2
fi
if (( CONTEXT_LENGTH < 1 || TBPTT_STEP_SIZE < 1 || TBPTT_STEP_SIZE > CONTEXT_LENGTH )); then
    echo "Invalid CONTEXT_LENGTH or TBPTT_STEP_SIZE" >&2
    exit 2
fi

for path in \
    "${PYTHON_BIN}" \
    "${ROOT}/scripts/train.py" \
    "${DATASET_ROOT}/meta/info.json" \
    "${BASE_CKPT}/params" \
    "${BASE_CKPT}/assets/physical-intelligence/libero/norm_stats.json"; do
    [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 2; }
done
"${PYTHON_BIN}" - "${DATASET_ROOT}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
info = json.loads((root / "meta/info.json").read_text())
for episode_index in range(int(info["total_episodes"])):
    relative = info["data_path"].format(
        episode_chunk=episode_index // int(info["chunks_size"]),
        episode_index=episode_index,
    )
    path = root / relative
    if not path.is_file() or path.stat().st_size < 8:
        raise SystemExit(f"LIBERO dataset incomplete: {path}")
print(f"LIBERO dataset OK: {root} ({info['total_episodes']} episodes)", flush=True)
PY
if [[ "${ROUTE^^}" == IE ]]; then
    for path in \
        "${VJEPA_CHECKPOINT}" \
        "${VJEPA_SOURCE_ROOT}/app/vjepa_2_1/models/vision_transformer.py"; do
        [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 2; }
    done
fi

cd "${ROOT}"
mkdir -p "${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}"
export PYTHONPATH="${ROOT}/src:${ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true
# Do not set a fixed JAX memory fraction: IE also runs PyTorch V-JEPA on the same GPU.
unset JAX_LOCAL_DEVICE_IDS
COORDINATOR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export JAX_COORDINATOR_ADDRESS="${COORDINATOR}:$((12000 + SLURM_JOB_ID % 1000))"
export ROOT PYTHON_BIN DATASET_ROOT BASE_CKPT VJEPA_CHECKPOINT VJEPA_SOURCE_ROOT
export CHECKPOINT_BASE_DIR CONFIG_NAME EXP_NAME ROUTE CONTEXT_LENGTH TBPTT_STEP_SIZE
export NUM_STEPS NUM_WORKERS WANDB_ENABLED

echo "route=${ROUTE^^} config=${CONFIG_NAME} experiment=${EXP_NAME}"
echo "nodes=1 GPUs/node=4 JAX_processes=4 batch/GPU=4 global_batch=16 fsdp_devices=4"
echo "root=${ROOT} dataset=${DATASET_ROOT} base_checkpoint=${BASE_CKPT}"
echo "context=${CONTEXT_LENGTH} tbptt=${TBPTT_STEP_SIZE} coordinator=${JAX_COORDINATOR_ADDRESS}"

srun --kill-on-bad-exit=1 -n 4 --ntasks-per-node=4 bash -lc '
    set -euo pipefail
    gpu_count="$("${PYTHON_BIN}" -c "import torch; print(torch.cuda.device_count())")"
    case "${gpu_count}" in
        1) export JAX_LOCAL_DEVICE_IDS=0 ;;
        4) export JAX_LOCAL_DEVICE_IDS="${SLURM_LOCALID}" ;;
        *) echo "Rank ${SLURM_PROCID}: expected 1 or 4 visible GPUs, got ${gpu_count}" >&2; exit 2 ;;
    esac
    export JAX_COMPILATION_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/openpi-jax-${SLURM_JOB_ID}-${SLURM_PROCID}"
    mkdir -p "${JAX_COMPILATION_CACHE_DIR}"
    "${PYTHON_BIN}" -c "import os, torch; i=int(os.environ[\"JAX_LOCAL_DEVICE_IDS\"]); torch.empty(1, device=f\"cuda:{i}\"); print(\"rank=%s gpu=%d name=%s\" % (os.environ[\"SLURM_PROCID\"], i, torch.cuda.get_device_name(i)), flush=True)"

    args=(
        "${CONFIG_NAME}"
        --exp-name "${EXP_NAME}"
        --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}"
        --weight-loader.params-path "${BASE_CKPT}/params"
        --data.dataset-root "${DATASET_ROOT}"
        --data.assets.assets-dir "${BASE_CKPT}/assets"
        --model.ttt-context-length "${CONTEXT_LENGTH}"
        --model.ttt-tbptt-step-size "${TBPTT_STEP_SIZE}"
        --batch-size 16
        --fsdp-devices 4
        --num-workers "${NUM_WORKERS}"
        --num-train-steps "${NUM_STEPS}"
        --save-interval 200
        --resume
    )
    if [[ "${ROUTE^^}" == IE ]]; then
        args+=(--data.vjepa-online-checkpoint "${VJEPA_CHECKPOINT}")
        args+=(--data.vjepa-online-source-root "${VJEPA_SOURCE_ROOT}")
        args+=(--data.vjepa-online-batch-size 1)
    fi
    if [[ "${WANDB_ENABLED}" == 0 ]]; then
        args+=(--no-wandb-enabled)
    fi
    exec "${PYTHON_BIN}" scripts/train.py "${args[@]}"
'
