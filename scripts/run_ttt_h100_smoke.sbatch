#!/usr/bin/env bash
# One-H100 smoke/probe for Pi0.5 inline + explicit-JEPA TTT.
# The input data is reduced to a few real LIBERO episodes at job start; the
# V-JEPA pair/current features are precomputed for that bounded smoke view.
# evc45 previously reported CUDA initialization failures and is excluded.
#SBATCH -J openpi-ttt-i-exp-smoke
#SBATCH -p highgpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 00:30:00
# Keep this probe on the separate H100-80GB pool; the normal-partition PCIe
# nodes currently fail CUDA initialization in this environment.
#SBATCH --nodelist=evc101,evc102,evc103,evc104
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH -e slurm_logs/%x-%j.err

set -euo pipefail

ROOT="/lustre/fs1/home/ha865618/project/openpi_jepawam"
PYTHON_BIN="/lustre/fs1/home/ha865618/.conda/envs/openpi_jepawam/bin/python"
# The user-provided path is the data/ subdirectory; LeRobotDataset needs the
# parent root because metadata lives in ../meta.
DATA_ROOT="/home/ha865618/data/libero_lerobot"
BASE_CKPT="${ROOT}/pretrained/CokeAnd1ce_JEPA_WAM/checkpoints/openpi/pi05_libero_vjepa_aux/pi05_vjepa_pair32_q64_w01_seed42_fsdp2_b128/29999"
VJEPA_CKPT="${ROOT}/data/weights/vjepa2/vjepa2_1_vitg_384.pt"
VJEPA_SOURCE="${ROOT}/third_party/vjepa2"
ASSETS_ROOT="${BASE_CKPT}/assets"
WORK_ROOT="${SLURM_TMPDIR:-/tmp}/${USER}/openpi-ttt-smoke-${SLURM_JOB_ID:-local}"
MINI_ROOT="${WORK_ROOT}/libero-mini"
TARGET_ROOT="${WORK_ROOT}/vjepa-targets"
RUN_ROOT="${WORK_ROOT}/runs"

# Match the requested experiment route and temporal settings.
CONFIG_NAME="${CONFIG_NAME:-pi05_libero_ttt_inline_explicit_jepa_ttt_only}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-128}"
OBSERVATION_STRIDE="${OBSERVATION_STRIDE:-8}"
TBPTT_STEP_SIZE="${TBPTT_STEP_SIZE:-8}"
SMOKE_EPISODES="${SMOKE_EPISODES:-2}"
STEPS_PER_BATCH="${STEPS_PER_BATCH:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"

export PYTHONPATH="${ROOT}/src:${ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "${WORK_ROOT}" "${RUN_ROOT}"
cd "${ROOT}"

echo "host=$(hostname)"
echo "job=${SLURM_JOB_ID:-local} node=${SLURM_NODELIST:-unknown}"
echo "config=${CONFIG_NAME} context=${CONTEXT_LENGTH} stride=${OBSERVATION_STRIDE} tbptt=${TBPTT_STEP_SIZE}"
echo "data=${DATA_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "wandb_enabled=${WANDB_ENABLED}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "missing python: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -d "${DATA_ROOT}" || ! -f "${DATA_ROOT}/meta/info.json" ]]; then
    echo "missing LeRobot data root: ${DATA_ROOT}" >&2
    exit 2
fi
if [[ ! -d "${BASE_CKPT}/params" || ! -f "${ASSETS_ROOT}/physical-intelligence/libero/norm_stats.json" ]]; then
    echo "missing base checkpoint or norm stats under ${BASE_CKPT}" >&2
    exit 2
fi
if [[ ! -f "${VJEPA_CKPT}" || ! -d "${VJEPA_SOURCE}" ]]; then
    echo "missing V-JEPA encoder assets" >&2
    exit 2
fi

"${PYTHON_BIN}" scripts/prepare_libero_ttt_smoke.py \
    --source-root "${DATA_ROOT}" \
    --output-root "${MINI_ROOT}" \
    --episodes "${SMOKE_EPISODES}"

"${PYTHON_BIN}" scripts/precompute_vjepa_pair_targets.py \
    --dataset-root "${MINI_ROOT}" \
    --checkpoint "${VJEPA_CKPT}" \
    --vjepa-source-root "${VJEPA_SOURCE}" \
    --output-root "${TARGET_ROOT}" \
    --image-key image \
    --future-offset 31 \
    --batch-size 4 \
    --device cuda:0

echo "precompute complete"

run_one() {
    local batch="$1"
    local exp="batch_${batch}"
    local log="${RUN_ROOT}/${exp}.log"
    local wandb_args=()
    if [[ "${WANDB_ENABLED}" == "0" ]]; then
        wandb_args+=(--no-wandb-enabled)
    fi
    echo "===== BEGIN batch=${batch} ====="
    set +e
    timeout 09m "${PYTHON_BIN}" scripts/train.py "${CONFIG_NAME}" \
        --exp-name "${exp}" \
        --checkpoint-base-dir "${RUN_ROOT}/checkpoints" \
        --weight-loader.params-path "${BASE_CKPT}/params" \
        --data.assets.assets-dir "${ASSETS_ROOT}" \
        --data.dataset-root "${MINI_ROOT}" \
        --data.vjepa-target-root "${TARGET_ROOT}" \
        --model.ttt-context-length "${CONTEXT_LENGTH}" \
        --model.ttt-observation-stride "${OBSERVATION_STRIDE}" \
        --model.ttt-tbptt-step-size "${TBPTT_STEP_SIZE}" \
        --batch-size "${batch}" \
        --num-workers "${NUM_WORKERS}" \
        --num-train-steps "${STEPS_PER_BATCH}" \
        --log-interval 1 \
        --save-interval 1000 \
        --keep-period 1000 \
        --overwrite \
        "${wandb_args[@]}" >"${log}" 2>&1
    local rc=$?
    set -e
    tail -40 "${log}"
    echo "batch=${batch} rc=${rc}" | tee -a "${RUN_ROOT}/summary.txt"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    return 0
}

# Probe conservatively upward in strict TTT-only mode; adjust BATCHES if needed.
BATCHES="${BATCHES:-1 2 4 8}"
for batch in ${BATCHES}; do
    run_one "${batch}"
done

echo "===== SUMMARY ====="
cat "${RUN_ROOT}/summary.txt"
