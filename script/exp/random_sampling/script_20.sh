#!/bin/bash

#SBATCH --partition=long                     # Ask for unkillable job
#SBATCH --gres=gpu:a100l:2 --constraint="dgx&ampere"                                     
#SBATCH --mem=96G                                        # Ask for 10 GB of RAM
#SBATCH --time=6:00:00                                   # The job will run for 3 hours
#SBATCH --output=./logs/20_out.txt
#SBATCH --error=./logs/20_error.txt
#SBATCH --constraint=80gb
#SBATCH -c 2


# 1. Load the required modules
module --quiet load anaconda/3
conda activate "rlhf"

cd ./src/training/
foldername=$(date +%Y_%m_%d_%H_%M)
CHPTPATH=/network/scratch/i/ines.arous/experiment_reward_model/20/2024_01_22_09_41

deepspeed ./reward_model/train_reward_model_gptj.py --local_rank=0 --seed=3 --data_path="/network/scratch/i/ines.arous/data_rlhf/percent/0.2/"  --chpt_path="${CHPTPATH}"
python ./reward_model/gptj_reward_test.py --ckpt_path="${CHPTPATH}"
