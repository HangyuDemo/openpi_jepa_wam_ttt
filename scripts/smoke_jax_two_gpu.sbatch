#!/usr/bin/env bash
#SBATCH -J openpi-jax-2gpu-smoke
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --ntasks-per-node=2
#SBATCH --constraint=h100
#SBATCH --gres=gpu:2
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 00:10:00
#SBATCH --exclude=evc22,evc24,evc25,evc26,evc28,evc34,evc38,evc40,evc42,evc44,evc45,evc46,evc48,evc50
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH -e slurm_logs/%x-%j.err

set -euo pipefail

ROOT="/lustre/fs1/home/ha865618/project/openpi_jepawam"
PYTHON_BIN="/lustre/fs1/home/ha865618/.conda/envs/openpi_jepawam/bin/python"
export PYTHONPATH="${ROOT}/src:${ROOT}/packages/openpi-client/src:${ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset JAX_LOCAL_DEVICE_IDS
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT

cd "${ROOT}"
COORDINATOR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
export JAX_COORDINATOR_ADDRESS="${COORDINATOR}:$((12000 + SLURM_JOB_ID % 1000))"
echo "job=${SLURM_JOB_ID} node=${SLURM_JOB_NODELIST} coordinator=${JAX_COORDINATOR_ADDRESS}"

srun --kill-on-bad-exit=1 --ntasks=2 --ntasks-per-node=2 --gres=gpu:2 \
    "${PYTHON_BIN}" scripts/smoke_jax_two_gpu.py
