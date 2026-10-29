#!/usr/bin/env bash
# Real Piper robot: pi05_base + jointly trained action, JEPA, and TTT branches.
#SBATCH -J piper-real-jepa-ttt
#SBATCH -p normal
#SBATCH -N 2
#SBATCH -n 4
#SBATCH --ntasks-per-node=2
#SBATCH --constraint=h100
#SBATCH --gres=gpu:2
#SBATCH -c 2
#SBATCH --mem=128G
#SBATCH -t 3-00:00:00
#SBATCH --requeue
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH -e slurm_logs/%x-%j.err

set -euo pipefail
ROOT=/lustre/fs1/home/ha865618/project/openpi_jepawam
PYTHON_BIN=/lustre/fs1/home/ha865618/.conda/envs/openpi_jepawam/bin/python
DATASET_ROOT="$ROOT/data/pick_can_place_plate_150_trimmed_v21"
TARGET_ROOT="$ROOT/data/vjepa_targets/piper_pick_can_place_plate_offset31"
BASE_PARAMS=/lustre/fs1/home/ha865618/.cache/openpi/openpi-assets/checkpoints/pi05_base/params
CONFIG_NAME=pi05_piper_jepa_ttt_joint
PROJECT_NAME=openpi-piper-real-robot-jepa-ttt
EXP_NAME="${EXP_NAME:-piper_real_joint_gbs64_20261025}"
BATCH_SIZE=64
ACCUMULATION_STEPS=4
NUM_STEPS=10000
NUM_WORKERS=1

cd "$ROOT"
for path in "$PYTHON_BIN" "$DATASET_ROOT/meta/info.json" "$TARGET_ROOT/manifest.json" "$BASE_PARAMS/_METADATA" "$BASE_PARAMS/manifest.ocdbt" "$ROOT/assets/pi05_piper/piper_pick_can_place_plate_150_v21/norm_stats.json"; do
    test -e "$path" || { echo "Missing: $path" >&2; exit 2; }
done
if (( SLURM_NTASKS != 4 || BATCH_SIZE != 64 || ACCUMULATION_STEPS != 4 )); then
    echo "Requires four GPU ranks, per-GPU microbatch 4, and accumulation 4 for GBS 64" >&2
    exit 2
fi

export PYTHONPATH="$ROOT/src:$ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WANDB_MODE=online WANDB_PROJECT="$PROJECT_NAME"
export WANDB_ENTITY=hang-yu-team
export WANDB_DIR="$ROOT/wandb"
unset WANDB_DISABLED JAX_LOCAL_DEVICE_IDS WANDB_RUN_ID WANDB_RESUME
mkdir -p "$WANDB_DIR"
COORDINATOR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export JAX_COORDINATOR_ADDRESS="${COORDINATOR}:$((12000 + SLURM_JOB_ID % 1000))"
export ROOT PYTHON_BIN DATASET_ROOT TARGET_ROOT BASE_PARAMS CONFIG_NAME PROJECT_NAME EXP_NAME
export BATCH_SIZE ACCUMULATION_STEPS NUM_STEPS NUM_WORKERS

"$PYTHON_BIN" scripts/precompute_vjepa_pair_targets.py \
    --dataset-root "$DATASET_ROOT" \
    --checkpoint "$ROOT/data/weights/vjepa2/vjepa2_1_vitg_384.pt" \
    --vjepa-source-root "$ROOT/third_party/vjepa2" \
    --output-root "$TARGET_ROOT" --image-key observation.images.front \
    --future-offset 31 --status-only |
    "$PYTHON_BIN" -c 'import json,sys; s=json.load(sys.stdin); print(s,flush=True); assert s["complete_episodes"] == s["total_episodes"] == 150'

# Authentication is read from the existing environment/netrc; never copy keys into scripts.
"$PYTHON_BIN" -c 'import wandb; api=wandb.Api(timeout=30); assert api.default_entity; print("W&B authentication verified", flush=True)'

echo "job=${SLURM_JOB_ID} coordinator=${JAX_COORDINATOR_ADDRESS} GBS=${BATCH_SIZE} global_microbatch=$((BATCH_SIZE / ACCUMULATION_STEPS)) per_gpu_microbatch=$((BATCH_SIZE / ACCUMULATION_STEPS / SLURM_NTASKS)) accumulation=${ACCUMULATION_STEPS} max_optimizer_steps=${NUM_STEPS}"
srun --kill-on-bad-exit=1 --ntasks=4 --ntasks-per-node=2 --gres=gpu:2 bash -lc '
    set -euo pipefail
    export JAX_LOCAL_DEVICE_IDS="$SLURM_LOCALID"
    RANK_CACHE_ROOT="${TMPDIR:-/tmp}/piper-joint-${SLURM_JOB_ID}-${SLURM_PROCID}"
    export HF_DATASETS_CACHE="$RANK_CACHE_ROOT/hf-datasets"
    export JAX_COMPILATION_CACHE_DIR="$RANK_CACHE_ROOT/jax"
    mkdir -p "$HF_DATASETS_CACHE" "$JAX_COMPILATION_CACHE_DIR"
    echo "rank=${SLURM_PROCID} hf_datasets_cache=${HF_DATASETS_CACHE}"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    "$PYTHON_BIN" -c "import torch; assert torch.cuda.device_count() == 2; torch.empty(1, device=\"cuda:${SLURM_LOCALID}\")"
    exec "$PYTHON_BIN" scripts/train.py "$CONFIG_NAME" \
        --exp-name "$EXP_NAME" --project-name "$PROJECT_NAME" \
        --data.dataset-root "$DATASET_ROOT" \
        --data.vjepa-target-root "$TARGET_ROOT" \
        --weight-loader.params-path "$BASE_PARAMS" \
        --batch-size "$BATCH_SIZE" --gradient-accumulation-steps "$ACCUMULATION_STEPS" \
        --fsdp-devices 4 --num-workers "$NUM_WORKERS" --num-train-steps "$NUM_STEPS" \
        --save-interval 100 --validation-interval 100 --validation-num-batches 4 \
        --log-interval 1 --wandb-enabled --resume
'
