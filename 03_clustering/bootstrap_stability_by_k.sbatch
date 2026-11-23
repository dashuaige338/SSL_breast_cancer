#!/bin/bash
#SBATCH --job-name=boot_stab_k
#SBATCH --partition=cpu_short,cpu_medium
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/gpfs/data/tsirigoslab/home/jz5490/LUMC_breast_cancer/SSL/logs/boot_stab_k_%A_%a.out
#SBATCH --error=/gpfs/data/tsirigoslab/home/jz5490/LUMC_breast_cancer/SSL/logs/boot_stab_k_%A_%a.err
# Bootstrap stability at k = 5 and 10 (+ k = 20 rerun as a reproduction check).
set -euo pipefail
umask 007
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
ks=(5 10 20)
cd /gpfs/data/tsirigoslab/home/jz5490/LUMC_breast_cancer/SSL
/gpfs/scratch/jz5490/myhome/.conda/envs/dpath_benchmark/bin/python calculate_bootstrap_stability_by_k.py --k "${ks[$SLURM_ARRAY_TASK_ID]}"
