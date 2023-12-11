#!/bin/bash

#SBATCH --partition=long                     # Ask for unkillable job
#SBATCH --gres=gpu:a100l:2                                        # Ask for 1 GPU
#SBATCH --mem=96G                                        # Ask for 10 GB of RAM
#SBATCH --time=48:00:00                                   # The job will run for 2 days
#SBATCH --output=./logs/extreme_out.txt
#SBATCH --error=./logs/extreme_error.txt
#SBATCH --constraint=80gb
#SBATCH -c 2


# 1. Load the required modules
module --quiet load anaconda/3
conda activate "rlhf"


cd ./src/
deepspeed examples/summarize_rlhf/reward_model/train_reward_model_gptj.py --data_path="/network/scratch/z/zixuan.li/generated_dataset/extreme" --chpt_path="/network/scratch/z/zixuan.li/experiment_reward_model/extreme"

BEST_CHECKPOINT_PATH=$(jq -r '.best_model_checkpoint' /network/scratch/z/zixuan.li/experiment_reward_model/extreme/checkpoint-5000/trainer_state.json)

python examples/summarize_rlhf/reward_model/gptj_reward_test.py --ckpt_path="${BEST_CHECKPOINT_PATH}/pytorch_model.bin"
