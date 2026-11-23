#!/bin/bash
#SBATCH --job-name=kmeans_sweep_sil
#SBATCH --partition=cpu_short,cpu_medium
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/data/tsirigoslab/home/jz5490/LUMC_breast_cancer/SSL/logs/kmeans_sweep_sil_%j.out
#SBATCH --error=/gpfs/data/tsirigoslab/home/jz5490/LUMC_breast_cancer/SSL/logs/kmeans_sweep_sil_%j.err

# Silhouette (5 x 10,000 tiles) for all six models at k = 20-100; run after the fits.
set -euo pipefail
umask 007
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
cd /gpfs/data/tsirigoslab/home/jz5490/LUMC_breast_cancer/SSL
/gpfs/scratch/jz5490/myhome/.conda/envs/dpath_benchmark/bin/python kmeans_sweep_silhouette.py
