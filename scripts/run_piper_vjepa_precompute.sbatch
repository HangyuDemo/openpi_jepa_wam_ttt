#!/usr/bin/env bash
# Resumable, episode-level V-JEPA target generation for the Piper dataset.
#SBATCH -J piper-vjepa-targets
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --constraint=h100
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH --requeue
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH -e slurm_logs/%x-%j.err

set -euo pipefail

ROOT=/lustre/fs1/home/ha865618/project/openpi_jepawam
PYTHON_BIN=/lustre/fs1/home/ha865618/.conda/envs/openpi_jepawam/bin/python
DATASET_ROOT="${DATASET_ROOT:-${ROOT}/data/pick_can_place_plate_150_trimmed_v21}"
VJEPA_CHECKPOINT="${VJEPA_CHECKPOINT:-${ROOT}/data/weights/vjepa2/vjepa2_1_vitg_384.pt}"
VJEPA_SOURCE_ROOT="${VJEPA_SOURCE_ROOT:-${ROOT}/third_party/vjepa2}"
TARGET_ROOT="${TARGET_ROOT:-${ROOT}/data/vjepa_targets/piper_pick_can_place_plate_offset31}"
VJEPA_BATCH_SIZE="${VJEPA_BATCH_SIZE:-4}"

cd "$ROOT"
for path in "$PYTHON_BIN" "$DATASET_ROOT/meta/info.json" "$VJEPA_CHECKPOINT" "$VJEPA_SOURCE_ROOT/app/vjepa_2_1/models/vision_transformer.py"; do
    test -e "$path" || { echo "Missing: $path" >&2; exit 2; }
done

export PYTHONPATH="$ROOT/src:$ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROOT PYTHON_BIN DATASET_ROOT VJEPA_CHECKPOINT VJEPA_SOURCE_ROOT TARGET_ROOT VJEPA_BATCH_SIZE MAX_EPISODES

echo "job=${SLURM_JOB_ID} host=$(hostname) target_root=${TARGET_ROOT}"
srun --kill-on-bad-exit=1 --ntasks=1 --gpu-bind=single:1 bash -lc '
    set -euo pipefail
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    "$PYTHON_BIN" -c "import torch; assert torch.cuda.device_count() >= 1; torch.empty(1, device=\"cuda:0\")"
    args=(
        --dataset-root "$DATASET_ROOT"
        --checkpoint "$VJEPA_CHECKPOINT"
        --vjepa-source-root "$VJEPA_SOURCE_ROOT"
        --output-root "$TARGET_ROOT"
        --image-key observation.images.front
        --future-offset 31
        --batch-size "$VJEPA_BATCH_SIZE"
        --device cuda:0
    )
    if [[ -n "${MAX_EPISODES:-}" ]]; then args+=(--max-episodes "$MAX_EPISODES"); fi
    exec "$PYTHON_BIN" scripts/precompute_vjepa_pair_targets.py "${args[@]}"
'

# A successful job must have every episode, including after a resumed run.
if [[ -z "${MAX_EPISODES:-}" ]]; then
    "$PYTHON_BIN" scripts/precompute_vjepa_pair_targets.py \
        --dataset-root "$DATASET_ROOT" --checkpoint "$VJEPA_CHECKPOINT" \
        --vjepa-source-root "$VJEPA_SOURCE_ROOT" --output-root "$TARGET_ROOT" \
        --image-key observation.images.front --future-offset 31 --status-only |
        "$PYTHON_BIN" -c 'import json,sys; s=json.load(sys.stdin); print(s,flush=True); assert s["complete_episodes"] == s["total_episodes"] == 150'
fi
