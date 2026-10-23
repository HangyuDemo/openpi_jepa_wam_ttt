#!/usr/bin/env bash
# PSC Bridges-2 LIBERO wrapper explicit-JEPA TTT resume (online current encoding).
# Run from the PSC project root after creating slurm_logs.
#SBATCH -J lib-we-online
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
export DATASET=libero
ROOT="${ROOT:-/ocean/projects/cis260071p/ha865618/project/openpi_jepa_wam_ttt}"
export ROOT
exec bash "${ROOT}/scripts/run_we_online_psc.sbatch"
