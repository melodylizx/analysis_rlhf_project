#!/bin/bash

#SBATCH --partition=long
#SBATCH --no-requeue 
#SBATCH --gres=gpu:a100l:2 --constraint="dgx&ampere"  
#SBATCH --mem=96G                                        # Ask for 10 GB of RAM
#SBATCH --time=3-12:00:00
#SBATCH --output=./logs/medium_out.txt
#SBATCH --error=./logs/medium_error.txt
#SBATCH --constraint=80gb
#SBATCH -c 2


# 1. Load the required modules
module --quiet load anaconda/3
conda activate "rlhf"

cd ./src/training/
foldername=$(date +%Y_%m_%d_%H_%M)
CHPTPATH=/network/scratch/i/ines.arous/experiment_reward_model/medium/"$foldername"

mkdir -p ${CHPTPATH}

deepspeed ./reward_model/train_reward_model_gptj.py --data_path="/network/scratch/i/ines.arous/data_rlhf/reliability/medium" --chpt_path="${CHPTPATH}"

python ./reward_model/gptj_reward_test.py --ckpt_path="${CHPTPATH}"
