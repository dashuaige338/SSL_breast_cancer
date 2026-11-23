#!/bin/bash
#SBATCH --job-name=k20_ch_db
#SBATCH --partition=cpu_short,cpu_medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/gpfs/data/tsirigoslab/home/jz5490/LUMC_breast_cancer/SSL/logs/k20_ch_db_%j.out
#SBATCH --error=/gpfs/data/tsirigoslab/home/jz5490/LUMC_breast_cancer/SSL/logs/k20_ch_db_%j.err
set -euo pipefail
umask 007
cd /gpfs/data/tsirigoslab/home/jz5490/LUMC_breast_cancer/SSL
/gpfs/scratch/jz5490/myhome/.conda/envs/dpath_benchmark/bin/python kmeans_k20_ch_db.py
