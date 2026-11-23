#!/bin/bash
#SBATCH --job-name=attn_ablation_dr
#SBATCH --partition=cpu_short,cpu_medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=/gpfs/data/tsirigoslab/home/jz5490/dPath_benchmark/run/CLAM_Surv/logs/attn_ablation_dr_%A_%a.out
#SBATCH --error=/gpfs/data/tsirigoslab/home/jz5490/dPath_benchmark/run/CLAM_Surv/logs/attn_ablation_dr_%A_%a.err
# Uniform-attention ablation of the DR SurvCLAM models (writes eval_results/attention_ablation_dr/).
set -euo pipefail
umask 007
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
models=(hoptimus virchow)
cd /gpfs/data/tsirigoslab/home/jz5490/dPath_benchmark/run/CLAM_Surv
/gpfs/scratch/jz5490/myhome/.conda/envs/dpath_benchmark/bin/python attention_ablation_dr.py --model "${models[$SLURM_ARRAY_TASK_ID]}"
