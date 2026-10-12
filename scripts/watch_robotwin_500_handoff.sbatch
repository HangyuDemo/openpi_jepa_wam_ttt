#!/usr/bin/env bash
# Wait for job 898547's finalized step-500 checkpoint, then hand off to 8 GPUs.
#SBATCH -J watch-rw-500
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 2-00:00:00
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH -e slurm_logs/%x-%j.err

set -euo pipefail
ROOT="/lustre/fs1/home/ha865618/project/openpi_jepawam"
cd "${ROOT}"
exec /lustre/fs1/home/ha865618/.conda/envs/openpi_jepawam/bin/python \
    scripts/watch_robotwin_500_handoff.py
