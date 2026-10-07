#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-check}"
if [[ "$MODE" != "check" && "$MODE" != "start" ]]; then
    echo "Usage: $0 [check|start]" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
CONFIG_NAME="${CONFIG_NAME:-pi05_libero_vjepa_aux}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a trained checkpoint step directory}"
PORT="${PORT:-8000}"
GPU_IDS="${GPU_IDS:-0}"
XLA_MEMORY_FRACTION="${XLA_MEMORY_FRACTION:-0.90}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "OpenPI environment not found at $PYTHON_BIN" >&2
    exit 2
fi
if [[ "$CHECKPOINT_DIR" != gs://* && ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Checkpoint directory not found: $CHECKPOINT_DIR" >&2
    exit 2
fi

server_args=(
    "$ROOT/scripts/serve_policy.py"
    --env LIBERO
    --port "$PORT"
    policy:checkpoint
    --policy.config "$CONFIG_NAME"
    --policy.dir "$CHECKPOINT_DIR"
)

printf 'SERVER_COMMAND='
printf ' %q' "$PYTHON_BIN" "${server_args[@]}"
printf '\n'
if [[ "$MODE" == "check" ]]; then
    exit 0
fi

CUDA_VISIBLE_DEVICES="$GPU_IDS" \
XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_MEMORY_FRACTION" \
TF_FORCE_GPU_ALLOW_GROWTH=true \
PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" "${server_args[@]}"
