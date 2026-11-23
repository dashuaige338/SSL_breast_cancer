#!/bin/bash
#SBATCH --job-name=kmeans_sweep_fit
#SBATCH --partition=cpu_medium,cpu_long
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-23
#SBATCH --output=/gpfs/data/tsirigoslab/home/jz5490/LUMC_breast_cancer/SSL/logs/kmeans_sweep_fit_%A_%a.out
#SBATCH --error=/gpfs/data/tsirigoslab/home/jz5490/LUMC_breast_cancer/SSL/logs/kmeans_sweep_fit_%A_%a.err

# K-means k = 30-100 (step 10) for the three models that only have k = 20, completing the
# k = 20-100 sweep for Supplementary Fig. 2. One (model, k) per task: 3 models x 8 k.
set -euo pipefail
umask 007
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
models=(CONCH PROVGIGAPATH KEEP)
ks=(30 40 50 60 70 80 90 100)
model=${models[$((SLURM_ARRAY_TASK_ID / 8))]}
k=${ks[$((SLURM_ARRAY_TASK_ID % 8))]}
cd /gpfs/data/tsirigoslab/home/jz5490/LUMC_breast_cancer/SSL
/gpfs/scratch/jz5490/myhome/.conda/envs/dpath_benchmark/bin/python kmeans_sweep_fit.py --model "$model" --k "$k"
