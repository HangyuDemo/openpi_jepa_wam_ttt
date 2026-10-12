#!/usr/bin/env bash
# One-H100 smoke/probe for RobotWin Clean-20 Pi0.5 inline + explicit-JEPA TTT.
# Use the separate highgpu H100-80GB pool; the normal-partition H100 PCIe
# nodes previously failed CUDA initialization in this environment.
#SBATCH -J openpi-rw-ttt-i-exp
#SBATCH -p highgpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 00:30:00
#SBATCH --nodelist=evc101,evc102,evc103,evc104
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH -e slurm_logs/%x-%j.err

set -euo pipefail

ROOT="/lustre/fs1/home/ha865618/project/openpi_jepawam"
PYTHON_BIN="/lustre/fs1/home/ha865618/.conda/envs/openpi_jepawam/bin/python"
DATA_ROOT="/home/ha865618/project/robotwin_lerobot/robotwin2_clean20_v21"
BASE_CKPT="${ROOT}/pretrained/CokeAnd1ce_JEPA_WAM/checkpoints/openpi/pi05_robotwin_clean_20_vjepa_aux/pi05_robotwin_vjepa_delta50_no_geomaug_b256_fsdp4_gpu0123_seed42/14999"
TARGET_ROOT="${ROOT}/data/vjepa_targets/robotwin2_clean20_v21_vjepa2_1_vitg_384_offset50"
ASSETS_ROOT="${BASE_CKPT}/assets"
WORK_ROOT="${SLURM_TMPDIR:-/tmp}/${USER}/openpi-robotwin-ttt-smoke-${SLURM_JOB_ID:-local}"
RUN_ROOT="${WORK_ROOT}/runs"

CONFIG_NAME="${CONFIG_NAME:-pi05_robotwin2_clean20_ttt_inline_explicit_jepa_ttt_only}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-128}"
OBSERVATION_STRIDE="${OBSERVATION_STRIDE:-8}"
TBPTT_STEP_SIZE="${TBPTT_STEP_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
STEPS_PER_BATCH="${STEPS_PER_BATCH:-1}"
PER_RUN_TIMEOUT="${PER_RUN_TIMEOUT:-04m}"
# Increase one power of two at a time. Override BATCHES to probe a different
# range without changing the Slurm submission.
BATCHES="${BATCHES:-1 2 4 8 16 32}"

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
echo "targets=${TARGET_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "wandb_enabled=${WANDB_ENABLED}"
echo "batches=${BATCHES} steps_per_batch=${STEPS_PER_BATCH} timeout_per_batch=${PER_RUN_TIMEOUT}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "missing python: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -f "${DATA_ROOT}/meta/info.json" ]]; then
    echo "missing LeRobot data root: ${DATA_ROOT}" >&2
    exit 2
fi
if [[ ! -d "${BASE_CKPT}/params" || ! -f "${ASSETS_ROOT}/local/robotwin_clean_20/norm_stats.json" ]]; then
    echo "missing base checkpoint or RobotWin norm stats under ${BASE_CKPT}" >&2
    exit 2
fi
if [[ ! -f "${TARGET_ROOT}/manifest.json" || ! -f "${TARGET_ROOT}/targets/chunk-000/episode_000000.npy" || ! -f "${TARGET_ROOT}/current/chunk-000/episode_000000.npy" ]]; then
    echo "missing precomputed RobotWin V-JEPA target/current arrays under ${TARGET_ROOT}" >&2
    exit 2
fi

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
    timeout "${PER_RUN_TIMEOUT}" "${PYTHON_BIN}" scripts/train.py "${CONFIG_NAME}" \
        --exp-name "${exp}" \
        --checkpoint-base-dir "${RUN_ROOT}/checkpoints" \
        --weight-loader.params-path "${BASE_CKPT}/params" \
        --data.assets.assets-dir "${ASSETS_ROOT}" \
        --data.dataset-root "${DATA_ROOT}" \
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
    tail -50 "${log}"
    echo "batch=${batch} rc=${rc}" | tee -a "${RUN_ROOT}/summary.txt"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    return 0
}

: > "${RUN_ROOT}/summary.txt"
for batch in ${BATCHES}; do
    run_one "${batch}"
done

echo "===== SUMMARY ====="
cat "${RUN_ROOT}/summary.txt"
