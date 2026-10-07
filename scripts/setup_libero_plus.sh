#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-check}"
if [[ "$MODE" != "install" && "$MODE" != "check" ]]; then
    echo "Usage: $0 [install|check]" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV_BIN:-uv}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-$ROOT/data/libero-plus/LIBERO-plus}"
EVAL_ENV="${EVAL_ENV:-$ROOT/examples/libero/.venv-plus}"
CONFIG_ROOT="${LIBERO_CONFIG_PATH:-$ROOT/data/libero-plus/config}"
DOWNLOAD_ROOT="${LIBERO_PLUS_DOWNLOAD_ROOT:-$ROOT/data/libero-plus/downloads}"

LIBERO_PLUS_REVISION="4976dc30028e805ff8094b55501d532c48fec182"
ASSET_REVISION="dd2bd61b7d9a6fef1abc52d606e983b41886a149"
ASSET_SHA256="96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf"
ASSET_URL="https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/${ASSET_REVISION}/assets.zip?download=true"
ASSET_ARCHIVE="$DOWNLOAD_ROOT/assets-${ASSET_REVISION}.zip"
ASSET_MARKER="$LIBERO_PLUS_ROOT/libero/libero/assets/.jepawam-assets-sha256"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Required command not found: $1" >&2
        exit 2
    fi
}

write_config() {
    local benchmark_root="$LIBERO_PLUS_ROOT/libero/libero"
    mkdir -p "$CONFIG_ROOT"
    {
        printf 'benchmark_root: %s\n' "$benchmark_root"
        printf 'bddl_files: %s\n' "$benchmark_root/bddl_files"
        printf 'init_states: %s\n' "$benchmark_root/init_files"
        printf 'datasets: %s\n' "$LIBERO_PLUS_ROOT/libero/datasets"
        printf 'assets: %s\n' "$benchmark_root/assets"
    } >"$CONFIG_ROOT/config.yaml"
}

check_install() {
    if [[ ! -d "$LIBERO_PLUS_ROOT/.git" ]]; then
        echo "LIBERO-Plus source is missing: $LIBERO_PLUS_ROOT" >&2
        return 1
    fi
    local source_revision
    source_revision="$(git -C "$LIBERO_PLUS_ROOT" rev-parse HEAD)"
    if [[ "$source_revision" != "$LIBERO_PLUS_REVISION" ]]; then
        echo "LIBERO-Plus revision mismatch: expected $LIBERO_PLUS_REVISION, got $source_revision" >&2
        return 1
    fi
    if [[ ! -f "$LIBERO_PLUS_ROOT/libero/libero/benchmark/task_classification.json" ]]; then
        echo "Missing LIBERO-Plus task classification metadata" >&2
        return 1
    fi
    if [[ ! -f "$ASSET_MARKER" ]] || [[ "$(<"$ASSET_MARKER")" != "$ASSET_SHA256" ]]; then
        echo "LIBERO-Plus assets are missing or were not installed from the pinned archive" >&2
        return 1
    fi
    if [[ ! -x "$EVAL_ENV/bin/python" ]]; then
        echo "LIBERO-Plus evaluation environment is missing: $EVAL_ENV" >&2
        return 1
    fi
    if [[ ! -f "$CONFIG_ROOT/config.yaml" ]]; then
        echo "LIBERO config is missing: $CONFIG_ROOT/config.yaml" >&2
        return 1
    fi

    LIBERO_CONFIG_PATH="$CONFIG_ROOT" \
    PYTHONPATH="$LIBERO_PLUS_ROOT:$ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$EVAL_ENV/bin/python" -c \
        'from libero.libero import benchmark; expected={"libero_spatial":2402,"libero_object":2518,"libero_goal":2591,"libero_10":2519}; actual={name: benchmark.get_benchmark_dict()[name]().n_tasks for name in expected}; assert actual == expected, (actual, expected); import openpi_client; print(actual)'
    echo "LIBERO-Plus installation is ready"
}

if [[ "$MODE" == "check" ]]; then
    check_install
    exit 0
fi

for command_name in git curl sha256sum bsdtar; do
    require_command "$command_name"
done
if ! command -v "$UV_BIN" >/dev/null 2>&1; then
    echo "uv was not found. Install uv or set UV_BIN to its executable." >&2
    exit 2
fi
if command -v pkg-config >/dev/null 2>&1 && ! pkg-config --exists MagickWand; then
    echo "MagickWand development files are required for LIBERO-Plus Sensor Noise tasks." >&2
    echo "On Ubuntu, install: libmagickwand-dev libexpat1 libfontconfig1-dev libpython3-dev" >&2
    exit 2
fi

mkdir -p "$(dirname "$LIBERO_PLUS_ROOT")" "$DOWNLOAD_ROOT"
if [[ ! -d "$LIBERO_PLUS_ROOT/.git" ]]; then
    git clone https://github.com/sylvestf/LIBERO-plus.git "$LIBERO_PLUS_ROOT"
fi
git -C "$LIBERO_PLUS_ROOT" fetch --depth 1 origin "$LIBERO_PLUS_REVISION"
git -C "$LIBERO_PLUS_ROOT" checkout --detach FETCH_HEAD

if [[ ! -f "$ASSET_ARCHIVE" ]] || ! printf '%s  %s\n' "$ASSET_SHA256" "$ASSET_ARCHIVE" | sha256sum --check --status; then
    curl --location --fail --retry 8 --retry-all-errors --continue-at - \
        --output "$ASSET_ARCHIVE" "$ASSET_URL"
fi
printf '%s  %s\n' "$ASSET_SHA256" "$ASSET_ARCHIVE" | sha256sum --check

if [[ ! -f "$ASSET_MARKER" ]] || [[ "$(<"$ASSET_MARKER")" != "$ASSET_SHA256" ]]; then
    if [[ -d "$LIBERO_PLUS_ROOT/libero/libero/assets" ]] && \
        [[ -n "$(find "$LIBERO_PLUS_ROOT/libero/libero/assets" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "Refusing to mix pinned assets with a non-empty untracked assets directory:" >&2
        echo "$LIBERO_PLUS_ROOT/libero/libero/assets" >&2
        echo "Use a fresh LIBERO_PLUS_ROOT or move the existing assets aside." >&2
        exit 2
    fi
    mkdir -p "$LIBERO_PLUS_ROOT/libero/libero"
    bsdtar -xf "$ASSET_ARCHIVE" --strip-components 10 -C "$LIBERO_PLUS_ROOT/libero/libero"
    printf '%s\n' "$ASSET_SHA256" >"$ASSET_MARKER"
fi

if [[ ! -x "$EVAL_ENV/bin/python" ]]; then
    "$UV_BIN" venv --python 3.8 "$EVAL_ENV"
fi
"$UV_BIN" pip sync --python "$EVAL_ENV/bin/python" \
    "$ROOT/examples/libero/requirements.txt" \
    "$ROOT/third_party/libero/requirements.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --index-strategy unsafe-best-match
"$UV_BIN" pip install --python "$EVAL_ENV/bin/python" --index-strategy unsafe-best-match \
    -r "$LIBERO_PLUS_ROOT/requirements.txt" \
    -r "$LIBERO_PLUS_ROOT/extra_requirements.txt"
"$UV_BIN" pip install --python "$EVAL_ENV/bin/python" hatchling editables \
    'dm-tree>=0.1.8' 'msgpack>=1.0.5' 'websockets>=11.0'
"$UV_BIN" pip install --python "$EVAL_ENV/bin/python" --no-deps --no-build-isolation \
    -e "$LIBERO_PLUS_ROOT" -e "$ROOT/packages/openpi-client"
"$UV_BIN" pip check --python "$EVAL_ENV/bin/python"

write_config
check_install
