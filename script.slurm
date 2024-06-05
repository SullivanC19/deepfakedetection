#!/bin/bash
#SBATCH --job-name=sullivan
#SBATCH --time=1-0
#SBATCH --mem-per-cpu=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=colins26@stanford.edu
#SBATCH --array=0-7

python -u main.py -i ${SLURM_ARRAY_TASK_ID}