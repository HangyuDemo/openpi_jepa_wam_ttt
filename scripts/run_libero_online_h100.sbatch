#!/usr/bin/env bash
# PSC Bridges-2: submit with sbatch -A <allocation-id> --export=ALL run_libero_online_h100.sbatch
#SBATCH -J libero-ttt-online
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=h100-80:4
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH -t 48:00:00
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH -e slurm_logs/%x-%j.err

set -euo pipefail
: "${ROOT:?Set ROOT to the PSC OpenPI project directory}"
: "${PYTHON_BIN:?Set PYTHON_BIN to the PSC Python 3.11 executable}"
: "${CONFIG_NAME:?Set CONFIG_NAME to a pi05_libero_ttt_*_ttt_only_online config}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT/data/libero_lerobot}"
BASE_CKPT="${BASE_CKPT:-$ROOT/pretrained/CokeAnd1ce_JEPA_WAM/checkpoints/openpi/pi05_libero_vjepa_aux/pi05_vjepa_pair32_q64_w01_seed42_fsdp2_b128/29999}"
VJEPA_CHECKPOINT="${VJEPA_CHECKPOINT:-$ROOT/data/weights/vjepa2/vjepa2_1_vitg_384.pt}"
VJEPA_SOURCE_ROOT="${VJEPA_SOURCE_ROOT:-$ROOT/third_party/vjepa2}"
EXP_NAME="${EXP_NAME:-${CONFIG_NAME#pi05_libero_ttt_}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_STEPS="${NUM_STEPS:-40000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
WANDB_ENABLED="${WANDB_ENABLED:-0}"

case "$CONFIG_NAME" in
  pi05_libero_ttt_*_explicit_jepa_ttt_only_online) EXPLICIT=1 ;;
  pi05_libero_ttt_*_implicit_full_ttt_only_online) EXPLICIT=0 ;;
  *) echo "Unsupported storage-light TTT config: $CONFIG_NAME" >&2; exit 2 ;;
esac
for path in "$PYTHON_BIN" "$ROOT/scripts/train.py" "$DATASET_ROOT/meta/info.json" "$BASE_CKPT/params" "$BASE_CKPT/assets/physical-intelligence/libero/norm_stats.json"; do
  test -e "$path" || { echo "Missing: $path" >&2; exit 2; }
done
if (( EXPLICIT )); then
  for path in "$VJEPA_CHECKPOINT" "$VJEPA_SOURCE_ROOT/app/vjepa_2_1/models/vision_transformer.py"; do
    test -e "$path" || { echo "Missing: $path" >&2; exit 2; }
  done
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset JAX_LOCAL_DEVICE_IDS
COORDINATOR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export JAX_COORDINATOR_ADDRESS="${COORDINATOR}:$((12000 + SLURM_JOB_ID % 1000))"
export ROOT PYTHON_BIN CONFIG_NAME DATASET_ROOT BASE_CKPT VJEPA_CHECKPOINT VJEPA_SOURCE_ROOT EXP_NAME BATCH_SIZE NUM_STEPS NUM_WORKERS WANDB_ENABLED EXPLICIT
mkdir -p checkpoints

srun --kill-on-bad-exit=1 --ntasks=4 --ntasks-per-node=4 bash -lc '
  set -euo pipefail
  gpu_count="$("$PYTHON_BIN" -c "import torch; print(torch.cuda.device_count())")"
  case "$gpu_count" in
    1) export JAX_LOCAL_DEVICE_IDS=0 ;;
    4) export JAX_LOCAL_DEVICE_IDS="$SLURM_LOCALID" ;;
    *) echo "Rank $SLURM_PROCID sees $gpu_count GPUs; expected 1 or 4" >&2; exit 2 ;;
  esac
  echo "rank=$SLURM_PROCID localid=$SLURM_LOCALID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} JAX_LOCAL_DEVICE_IDS=$JAX_LOCAL_DEVICE_IDS"
  "$PYTHON_BIN" -c "import os, torch; device = \"cuda:\" + os.environ[\"JAX_LOCAL_DEVICE_IDS\"]; torch.empty(1, device=device); print(\"rank\", os.environ[\"SLURM_PROCID\"], \"GPU\", device, torch.cuda.get_device_name(device), flush=True)"
  args=(
    "$CONFIG_NAME" --exp-name "$EXP_NAME"
    --weight-loader.params-path "$BASE_CKPT/params"
    --data.dataset-root "$DATASET_ROOT"
    --data.assets.assets-dir "$BASE_CKPT/assets"
    --batch-size "$BATCH_SIZE" --fsdp-devices 4
    --num-workers "$NUM_WORKERS" --num-train-steps "$NUM_STEPS"
    --checkpoint-base-dir "$ROOT/checkpoints" --save-interval 200 --validation-interval 400 --resume
  )
  if (( EXPLICIT )); then
    args+=(--data.vjepa-online-checkpoint "$VJEPA_CHECKPOINT")
    args+=(--data.vjepa-online-source-root "$VJEPA_SOURCE_ROOT")
  fi
  if [[ "$WANDB_ENABLED" == 0 ]]; then args+=(--no-wandb-enabled); fi
  exec "$PYTHON_BIN" scripts/train.py "${args[@]}"
'
