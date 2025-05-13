#!/bin/bash

#SBATCH --job-name="¯\\_(ツ)_/¯"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=150000
#SBATCH --time=1-00:00:00

srun python scripts/train_transit.py
