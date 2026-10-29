#!/usr/bin/env bash
# Train the Piper TTT wrapper from the best validated JEPA-WAM checkpoint.
#SBATCH -J piper-ttt-train
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --ntasks-per-node=2
#SBATCH --constraint=h100
#SBATCH --gres=gpu:2
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 2-00:00:00
#SBATCH --requeue
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH -e slurm_logs/%x-%j.err

set -euo pipefail

ROOT=/lustre/fs1/home/ha865618/project/openpi_jepawam
PYTHON_BIN=/lustre/fs1/home/ha865618/.conda/envs/openpi_jepawam/bin/python
DATASET_ROOT="${DATASET_ROOT:-${ROOT}/data/pick_can_place_plate_150_trimmed_v21}"
TARGET_ROOT="${TARGET_ROOT:-${ROOT}/data/vjepa_targets/piper_pick_can_place_plate_offset31}"
JEPA_EXP_NAME="${JEPA_EXP_NAME:-piper_jepa}"
EXP_NAME="${EXP_NAME:-piper_ttt}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_STEPS="${NUM_STEPS:-3000}"
NUM_WORKERS="${NUM_WORKERS:-2}"

cd "$ROOT"
for path in "$PYTHON_BIN" "$DATASET_ROOT/meta/info.json" "$TARGET_ROOT/manifest.json" "$ROOT/assets/pi05_piper/piper_pick_can_place_plate_150_v21/norm_stats.json"; do
    test -e "$path" || { echo "Missing: $path" >&2; exit 2; }
done
if (( SLURM_NTASKS != 2 || BATCH_SIZE % SLURM_NTASKS != 0 )); then
    echo "Expected two GPU ranks and a global batch divisible by two" >&2
    exit 2
fi

JEPA_DIR="$ROOT/checkpoints/pi05_piper_vjepa_aux/$JEPA_EXP_NAME"
JEPA_PARAMS="${JEPA_PARAMS:-$("$PYTHON_BIN" - "$JEPA_DIR" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
record = root / "best_validation.json"
if not record.is_file():
    sys.exit(f"Missing JEPA validation record: {record}")
step = int(json.loads(record.read_text())["step"])
params = root / "best" / str(step) / "params"
if not (params / "commit_success.txt").is_file():
    sys.exit(f"Best JEPA checkpoint is incomplete: {params}")
print(params)
PY
)}"
test -e "$JEPA_PARAMS/commit_success.txt" || { echo "Incomplete JEPA params: $JEPA_PARAMS" >&2; exit 2; }

"$PYTHON_BIN" scripts/precompute_vjepa_pair_targets.py \
    --dataset-root "$DATASET_ROOT" \
    --checkpoint "$ROOT/data/weights/vjepa2/vjepa2_1_vitg_384.pt" \
    --vjepa-source-root "$ROOT/third_party/vjepa2" \
    --output-root "$TARGET_ROOT" --image-key observation.images.front \
    --future-offset 31 --status-only |
    "$PYTHON_BIN" -c 'import json,sys; s=json.load(sys.stdin); print(s,flush=True); assert s["complete_episodes"] == s["total_episodes"] == 150'

export PYTHONPATH="$ROOT/src:$ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE="$ROOT/data/.hf_datasets_cache"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TF_FORCE_GPU_ALLOW_GROWTH=true
unset JAX_LOCAL_DEVICE_IDS
COORDINATOR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export JAX_COORDINATOR_ADDRESS="${COORDINATOR}:$((12000 + SLURM_JOB_ID % 1000))"
export ROOT PYTHON_BIN DATASET_ROOT TARGET_ROOT JEPA_PARAMS EXP_NAME BATCH_SIZE NUM_STEPS NUM_WORKERS

echo "job=${SLURM_JOB_ID} coordinator=${JAX_COORDINATOR_ADDRESS} jepa_params=${JEPA_PARAMS} batch=${BATCH_SIZE} steps=${NUM_STEPS}"
srun --kill-on-bad-exit=1 --ntasks=2 --ntasks-per-node=2 --gres=gpu:2 bash -lc '
    set -euo pipefail
    export JAX_LOCAL_DEVICE_IDS="$SLURM_LOCALID"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    "$PYTHON_BIN" -c "import torch; assert torch.cuda.device_count() == 2; torch.empty(1, device=\"cuda:${SLURM_LOCALID}\")"
    exec "$PYTHON_BIN" scripts/train.py pi05_piper_ttt_wrapper_explicit_jepa_ttt_only \
        --exp-name "$EXP_NAME" \
        --data.dataset-root "$DATASET_ROOT" \
        --data.vjepa-target-root "$TARGET_ROOT" \
        --weight-loader.params-path "$JEPA_PARAMS" \
        --batch-size "$BATCH_SIZE" --fsdp-devices 2 \
        --num-workers "$NUM_WORKERS" --num-train-steps "$NUM_STEPS" \
        --no-wandb-enabled --resume
'
