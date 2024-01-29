#!/bin/bash

#SBATCH --partition=long                     # Ask for unkillable job
#SBATCH --gres=gpu:a100l:2 --constraint="dgx&ampere"
#SBATCH --mem=96G                                        # Ask for 10 GB of RAM
#SBATCH --time=4:00:00                                   # The job will run for 3 hours
#SBATCH --output=./logs/uncertain_out.txt
#SBATCH --error=./logs/uncertain_error.txt
#SBATCH --constraint=80gb
#SBATCH -c 2

module --quiet load anaconda/3
conda activate "rlhf"

#load the dataset this will create pkl file in the data folder
cd ./src/training/reward_model
python uncertain_model.py