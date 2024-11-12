#!/bin/bash

#SBATCH --partition=long                     # Ask for unkillable job
#SBATCH --gres=gpu:a100l:2 --constraint="dgx&ampere"                                     
#SBATCH --mem=96G                                        # Ask for 10 GB of RAM
#SBATCH --time=6:00:00                                   # The job will run for 3 hours
#SBATCH --output=./logs/low_confidence_out.txt
#SBATCH --error=./logs/low_confidence_error.txt
#SBATCH --constraint=80gb
#SBATCH -c 2


# 1. Load the required modules
module --quiet load anaconda/3
conda activate "rlhf"

cd ./src/training/
foldername=$(date +%Y_%m_%d_%H_%M)
CHPTPATH=/network/scratch/i/ines.arous/experiment_reward_model/low_conf/2024_01_23_10_38

deepspeed ./reward_model/train_reward_model_gptj.py --local_rank=0 --seed=3 --data_path="/network/scratch/i/ines.arous/data_rlhf/low_conf/"  --chpt_path="${CHPTPATH}"
python ./reward_model/gptj_reward_test.py --ckpt_path="${CHPTPATH}"
