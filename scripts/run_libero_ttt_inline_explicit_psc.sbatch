#!/usr/bin/env bash
# PSC Bridges-2: inline explicit-JEPA TTT, action flow loss only.
# Submit from ROOT after creating slurm_logs; WANDB_API_KEY is required.
#SBATCH -J lib-i-explicit
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=h100-80:4
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH -t 30:00
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH -e slurm_logs/%x-%j.err

set -euo pipefail
ROOT="${ROOT:-/ocean/projects/cis260071p/ha865618/project/openpi_jepa_wam_ttt}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
CONFIG_NAME=pi05_libero_ttt_inline_explicit_jepa_ttt_only_online
EXP_NAME="${EXP_NAME:-inline_explicit_action_only}"
WANDB_ENABLED=1
: "${WANDB_API_KEY:?Export WANDB_API_KEY before sbatch}"
export ROOT PYTHON_BIN CONFIG_NAME EXP_NAME WANDB_ENABLED WANDB_API_KEY
exec bash "$ROOT/scripts/run_libero_online_h100.sbatch"
