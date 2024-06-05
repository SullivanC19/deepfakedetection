#!/bin/bash
#SBATCH --job-name=sullivan
#SBATCH --time=1-0
#SBATCH --mem-per-cpu=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:tesla:8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=colins26@stanford.edu

python -u main.py