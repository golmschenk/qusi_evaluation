#!/bin/bash

#SBATCH --job-name="¯\\_(ツ)_/¯"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=600000
#SBATCH --time=1-00:00:00

srun python scripts/train_variable_team_simulations_dataset.py
