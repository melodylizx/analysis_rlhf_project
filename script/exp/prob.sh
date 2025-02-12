#!/bin/bash

#SBATCH --partition=short-unkillable                    # Ask for unkillable job
#SBATCH --gres=gpu:a100l:4
#SBATCH --mem=128G                                        # Ask for 10 GB of RAM
#SBATCH --time=03:00:00                                   # The job will run for 3 hours
#SBATCH --output=./logs/prob.txt
#SBATCH --error=./logs/prob.txt
#SBATCH -c 4


# 1. Load the required modules
module load gcc/9.3.0
module load  cudatoolkit/11.7
source vrlhf/bin/activate

nvidia-smi

cd ./src/training/

deepspeed ./reward_model/compute_uncertain.py