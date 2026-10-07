#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-plus}"
if [[ "$MODE" != "plus" && "$MODE" != "standard" && "$MODE" != "summary" && "$MODE" != "merge" ]]; then
    echo "Usage: $0 [plus|standard|summary|merge] [journal paths...]" >&2
    exit 2
fi
shift || true

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLUS_REVISION="4976dc30028e805ff8094b55501d532c48fec182"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-$ROOT/data/libero-plus/LIBERO-plus}"
STANDARD_LIBERO_ROOT="${STANDARD_LIBERO_ROOT:-$ROOT/third_party/libero}"

configure_runtime() {
    local source_root="$1"
    local config_root="$2"
    local benchmark_root="$source_root/libero/libero"
    mkdir -p "$config_root"
    {
        printf 'benchmark_root: %s\n' "$benchmark_root"
        printf 'bddl_files: %s\n' "$benchmark_root/bddl_files"
        printf 'init_states: %s\n' "$benchmark_root/init_files"
        printf 'datasets: %s\n' "$source_root/libero/datasets"
        printf 'assets: %s\n' "$benchmark_root/assets"
    } >"$config_root/config.yaml"
}

if [[ "$MODE" == "standard" ]]; then
    SOURCE_ROOT="$STANDARD_LIBERO_ROOT"
    EVAL_PYTHON="${EVAL_PYTHON:-$ROOT/examples/libero/.venv/bin/python}"
    CONFIG_ROOT="${LIBERO_CONFIG_PATH:-$ROOT/data/libero-standard/config}"
    BENCHMARK_REVISION="${BENCHMARK_REVISION:-$(git -C "$SOURCE_ROOT" rev-parse HEAD 2>/dev/null || true)}"
    NUM_TRIALS=50
else
    SOURCE_ROOT="$LIBERO_PLUS_ROOT"
    EVAL_PYTHON="${EVAL_PYTHON:-$ROOT/examples/libero/.venv-plus/bin/python}"
    CONFIG_ROOT="${LIBERO_CONFIG_PATH:-$ROOT/data/libero-plus/config}"
    BENCHMARK_REVISION="${BENCHMARK_REVISION:-$PLUS_REVISION}"
    NUM_TRIALS=1
fi

if [[ ! -x "$EVAL_PYTHON" ]]; then
    echo "Evaluation Python not found: $EVAL_PYTHON" >&2
    exit 2
fi
if [[ ! -d "$SOURCE_ROOT/libero/libero" ]]; then
    echo "LIBERO source/runtime tree not found: $SOURCE_ROOT" >&2
    exit 2
fi
if [[ -z "$BENCHMARK_REVISION" ]]; then
    echo "Could not determine benchmark revision; set BENCHMARK_REVISION explicitly" >&2
    exit 2
fi
configure_runtime "$SOURCE_ROOT" "$CONFIG_ROOT"

export LIBERO_CONFIG_PATH="$CONFIG_ROOT"
export PYTHONPATH="$SOURCE_ROOT:$ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"

if [[ "$MODE" == "summary" ]]; then
    if (( $# < 1 )); then
        echo "Usage: $0 summary JOURNAL [JOURNAL ...]" >&2
        exit 2
    fi
    exec "$EVAL_PYTHON" "$ROOT/examples/libero/main.py" --args.summarize-results-paths "$@"
fi

if [[ "$MODE" == "merge" ]]; then
    if (( $# < 2 )); then
        echo "Usage: $0 merge OUTPUT_JOURNAL SHARD_JOURNAL [SHARD_JOURNAL ...]" >&2
        exit 2
    fi
    output_journal="$1"
    shift
    merge_args=(
        "$ROOT/examples/libero/main.py"
        --args.results-path "$output_journal"
        --args.merge-results-paths "$@"
    )
    if [[ "${OVERWRITE_RESULTS:-0}" == "1" ]]; then
        merge_args+=(--args.overwrite-results)
    fi
    exec "$EVAL_PYTHON" "${merge_args[@]}"
fi

RUN_ID="${RUN_ID:?Set RUN_ID to a stable identifier for the served checkpoint}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/data/libero-eval/$RUN_ID}"
RESULTS_PATH="${RESULTS_PATH:-$RESULTS_ROOT/$MODE-$TASK_SUITE.jsonl}"
VIDEO_ROOT="${VIDEO_ROOT:-$RESULTS_ROOT/videos}"
TASK_START="${TASK_START:-0}"
TASK_END="${TASK_END:-}"
NUM_TASK_SHARDS="${NUM_TASK_SHARDS:-1}"
TASK_SHARD_ID="${TASK_SHARD_ID:-0}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"
EVAL_GPU="${EVAL_GPU:-0}"

eval_args=(
    "$ROOT/examples/libero/main.py"
    --args.host "$HOST"
    --args.port "$PORT"
    --args.run-id "$RUN_ID"
    --args.task-suite-name "$TASK_SUITE"
    --args.benchmark-mode "$MODE"
    --args.benchmark-revision "$BENCHMARK_REVISION"
    --args.num-trials-per-task "$NUM_TRIALS"
    --args.task-start "$TASK_START"
    --args.num-task-shards "$NUM_TASK_SHARDS"
    --args.task-shard-id "$TASK_SHARD_ID"
    --args.replan-steps "$REPLAN_STEPS"
    --args.results-path "$RESULTS_PATH"
    --args.video-out-path "$VIDEO_ROOT"
)
if [[ -n "$TASK_END" ]]; then
    eval_args+=(--args.task-end "$TASK_END")
fi
if [[ "$SAVE_VIDEO" == "0" ]]; then
    eval_args+=(--args.no-save-video)
fi
if [[ "${RESUME:-1}" == "0" ]]; then
    eval_args+=(--args.no-resume)
fi
if [[ "${RETRY_ERRORS:-0}" == "1" ]]; then
    eval_args+=(--args.retry-errors)
fi
if [[ "$MODE" == "plus" ]]; then
    eval_args+=(
        --args.classification-path
        "$SOURCE_ROOT/libero/libero/benchmark/task_classification.json"
    )
fi

printf 'EVAL_COMMAND='
printf ' %q' "$EVAL_PYTHON" "${eval_args[@]}"
printf '\n'
printf 'RESULTS_PATH=%s\n' "$RESULTS_PATH"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi

CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}" \
PYTHONNOUSERSITE=1 \
    "$EVAL_PYTHON" "${eval_args[@]}"
