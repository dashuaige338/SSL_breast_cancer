#!/bin/bash
#SBATCH --job-name=kmeans_k5_k10
#SBATCH --partition=cpu_medium,cpu_long
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-11
#SBATCH --output=/gpfs/data/tsirigoslab/home/jz5490/LUMC_breast_cancer/SSL/logs/kmeans_k5_k10_%A_%a.out
#SBATCH --error=/gpfs/data/tsirigoslab/home/jz5490/LUMC_breast_cancer/SSL/logs/kmeans_k5_k10_%A_%a.err

# K-means k = 5 and 10 for all six models (extends the k = 20-100 sweep below k = 20 for
# Supplementary Fig. 2). Same settings as kmeans_sweep_fit.py; never overwrites a model.
set -euo pipefail
umask 007
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
models=(Virchow UNI Hoptimusv0 CONCH PROVGIGAPATH KEEP)
ks=(5 10)
model=${models[$((SLURM_ARRAY_TASK_ID / 2))]}
k=${ks[$((SLURM_ARRAY_TASK_ID % 2))]}
cd /gpfs/data/tsirigoslab/home/jz5490/LUMC_breast_cancer/SSL
/gpfs/scratch/jz5490/myhome/.conda/envs/dpath_benchmark/bin/python kmeans_sweep_fit.py --model "$model" --k "$k"
