#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-status}"
if [[ "$MODE" != "plan" && "$MODE" != "status" && "$MODE" != "smoke" && "$MODE" != "run" ]]; then
    echo "Usage: $0 [plan|status|smoke|run]" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
: "${DATASET_ROOT:?Set DATASET_ROOT to the local LeRobot dataset directory}"
: "${VJEPA_CHECKPOINT:?Set VJEPA_CHECKPOINT to the V-JEPA 2.1 ViT-g/384 pretraining checkpoint}"
: "${VJEPA_SOURCE_ROOT:?Set VJEPA_SOURCE_ROOT to a clone of facebookresearch/vjepa2}"

IMAGE_KEY="${IMAGE_KEY:-image}"
FUTURE_OFFSET="${FUTURE_OFFSET:-31}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/data/vjepa_targets/vjepa2_1_vitg_384_offset${FUTURE_OFFSET}}"
GPU_IDS="${GPU_IDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MIN_FREE_GIB="${MIN_FREE_GIB:-16}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found at $PYTHON_BIN. Run 'GIT_LFS_SKIP_SMUDGE=1 uv sync' first." >&2
    exit 2
fi

common_args=(
    --dataset-root "$DATASET_ROOT"
    --checkpoint "$VJEPA_CHECKPOINT"
    --vjepa-source-root "$VJEPA_SOURCE_ROOT"
    --output-root "$OUTPUT_ROOT"
    --image-key "$IMAGE_KEY"
    --future-offset "$FUTURE_OFFSET"
    --batch-size "$BATCH_SIZE"
    --min-free-gib "$MIN_FREE_GIB"
)
case "$MODE" in
    plan)
        "$PYTHON_BIN" "$ROOT/scripts/precompute_vjepa_pair_targets.py" "${common_args[@]}" --plan-only
        ;;
    status)
        "$PYTHON_BIN" "$ROOT/scripts/precompute_vjepa_pair_targets.py" "${common_args[@]}" --status-only
        ;;
    smoke)
        first_gpu="${GPU_IDS%%,*}"
        CUDA_VISIBLE_DEVICES="$first_gpu" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            "$PYTHON_BIN" "$ROOT/scripts/precompute_vjepa_pair_targets.py" "${common_args[@]}" \
            --device cuda:0 --worker-rank 0 --world-size 1 --max-episodes 1
        ;;
    run)
        IFS=',' read -r -a gpu_array <<<"$GPU_IDS"
        world_size="${#gpu_array[@]}"
        if (( world_size < 1 )); then
            echo "GPU_IDS must select at least one GPU" >&2
            exit 2
        fi
        mkdir -p "$OUTPUT_ROOT/logs"
        pids=()
        cleanup() {
            for pid in "${pids[@]:-}"; do
                kill "$pid" 2>/dev/null || true
            done
        }
        trap cleanup INT TERM
        for rank in "${!gpu_array[@]}"; do
            gpu="${gpu_array[$rank]}"
            log="$OUTPUT_ROOT/logs/worker-${rank}.gpu${gpu}.log"
            CUDA_VISIBLE_DEVICES="$gpu" \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                "$PYTHON_BIN" "$ROOT/scripts/precompute_vjepa_pair_targets.py" "${common_args[@]}" \
                --device cuda:0 --worker-rank "$rank" --world-size "$world_size" >"$log" 2>&1 &
            pids+=("$!")
            echo "started rank=$rank gpu=$gpu pid=${pids[-1]} log=$log"
        done

        failed=0
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                failed=1
            fi
        done
        trap - INT TERM
        if (( failed )); then
            echo "At least one V-JEPA preprocessing worker failed; inspect $OUTPUT_ROOT/logs." >&2
            exit 1
        fi
        "$PYTHON_BIN" "$ROOT/scripts/precompute_vjepa_pair_targets.py" "${common_args[@]}" --status-only
        ;;
esac
