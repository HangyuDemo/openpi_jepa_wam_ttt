#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-check}"
if [[ "$MODE" != "check" && "$MODE" != "start" && "$MODE" != "resume" ]]; then
    echo "Usage: $0 [check|start|resume]" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
CONFIG_NAME="${CONFIG_NAME:-pi05_libero_vjepa_aux}"
EXP_NAME="${EXP_NAME:-pi05_libero_vjepa_aux}"
TARGET_ROOT="${TARGET_ROOT:-$ROOT/data/vjepa_targets/libero_vjepa2_1_vitg_384_offset31}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$ROOT/checkpoints}"
BASE_PARAMS="${BASE_PARAMS:-gs://openpi-assets/checkpoints/pi05_base/params}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-128}"
FSDP_DEVICES="${FSDP_DEVICES:-2}"
NUM_STEPS="${NUM_STEPS:-30000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
XLA_MEMORY_FRACTION="${XLA_MEMORY_FRACTION:-0.90}"
WANDB_ENABLED="${WANDB_ENABLED:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found at $PYTHON_BIN. Run 'GIT_LFS_SKIP_SMUDGE=1 uv sync' first." >&2
    exit 2
fi
if [[ ! -f "$TARGET_ROOT/manifest.json" ]]; then
    echo "Missing precomputed target manifest: $TARGET_ROOT/manifest.json" >&2
    exit 2
fi

IFS=',' read -r -a gpu_array <<<"$GPU_IDS"
gpu_count="${#gpu_array[@]}"
if (( gpu_count < 1 || BATCH_SIZE % gpu_count != 0 || gpu_count % FSDP_DEVICES != 0 )); then
    echo "BATCH_SIZE must be divisible by GPU count, and GPU count by FSDP_DEVICES." >&2
    exit 2
fi

checkpoint_dir="$CHECKPOINT_BASE_DIR/$CONFIG_NAME/$EXP_NAME"
if [[ "$MODE" == "start" && -e "$checkpoint_dir" ]]; then
    echo "Refusing to overwrite existing checkpoint directory: $checkpoint_dir" >&2
    exit 2
fi
if [[ "$MODE" == "resume" && ! -d "$checkpoint_dir" ]]; then
    echo "Cannot resume missing checkpoint directory: $checkpoint_dir" >&2
    exit 2
fi

train_args=(
    "$CONFIG_NAME"
    --exp-name "$EXP_NAME"
    --checkpoint-base-dir "$CHECKPOINT_BASE_DIR"
    --weight-loader.params-path "$BASE_PARAMS"
    --data.vjepa-target-root "$TARGET_ROOT"
    --batch-size "$BATCH_SIZE"
    --fsdp-devices "$FSDP_DEVICES"
    --num-workers "$NUM_WORKERS"
    --num-train-steps "$NUM_STEPS"
)
if [[ "$MODE" == "resume" ]]; then
    train_args+=(--resume)
fi
if [[ "$WANDB_ENABLED" == "0" ]]; then
    train_args+=(--no-wandb-enabled)
fi

printf 'TRAIN_COMMAND='
printf ' %q' "$PYTHON_BIN" "$ROOT/scripts/train.py" "${train_args[@]}"
printf '\n'
echo "CHECKPOINT_DIR=$checkpoint_dir"
echo "TARGET_ROOT=$TARGET_ROOT"
if [[ "$MODE" == "check" ]]; then
    exit 0
fi

CUDA_VISIBLE_DEVICES="$GPU_IDS" \
XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_MEMORY_FRACTION" \
TF_FORCE_GPU_ALLOW_GROWTH=true \
PYTHONUNBUFFERED=1 \
TOKENIZERS_PARALLELISM=false \
    "$PYTHON_BIN" "$ROOT/scripts/train.py" "${train_args[@]}"
