#!/bin/bash

#SBATCH --partition=main                     # Ask for unkillable job
#SBATCH --gres=gpu:1 --constraint="dgx&ampere"
#SBATCH --mem=47G                                        # Ask for 10 GB of RAM
#SBATCH --time=48:00:00                                   # The job will run for 3 hours
#SBATCH --output=./logs/random/01_out.txt
#SBATCH --error=./logs/random/01_error.txt
#SBATCH -c 2


# 1. Load the required modules
module --quiet load anaconda/3
conda init
conda activate "vrlhf"

cd ./src/training/
foldername=$(date +%Y_%m_%d_%H_%M)
CHPTPATH=/network/scratch/i/ines.arous/experiment_reward_model/random_sampling/01/"$foldername"
SAVEPATH=/network/scratch/i/ines.arous/ppo/random_sampling/01/"$foldername"
mkdir -p ${CHPTPATH}
mkdir -p ${SAVEPATH}


deepspeed ./reward_model/train_reward_model_gptj.py --local_rank=0 --seed=3  --data_path="../../data/percent/0.01"  --chpt_path="${CHPTPATH}"
python ./reward_model/gptj_reward_test.py --ckpt_path="${CHPTPATH}"
accelerate launch --config_file configs/default_accelerate_config.yaml new_ppo.py --ckpt_path="${CHPTPATH}" --save_path="${SAVEPATH}"