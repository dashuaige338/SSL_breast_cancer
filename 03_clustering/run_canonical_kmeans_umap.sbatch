#!/bin/bash
#SBATCH --job-name=k20_umap
#SBATCH --partition=cpu_long
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-2
#SBATCH --output=/gpfs/data/tsirigoslab/home/jz5490/LUMC_breast_cancer/SSL/logs/k20_umap_%A_%a.out
#SBATCH --error=/gpfs/data/tsirigoslab/home/jz5490/LUMC_breast_cancer/SSL/logs/k20_umap_%A_%a.err

set -euo pipefail
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

models=(CONCH PROVGIGAPATH KEEP)
model=${models[$SLURM_ARRAY_TASK_ID]}

/gpfs/scratch/jz5490/myhome/.conda/envs/dpath_benchmark/bin/python \
  /gpfs/data/tsirigoslab/home/jz5490/LUMC_breast_cancer/SSL/run_canonical_kmeans_umap.py \
  --model "$model"
