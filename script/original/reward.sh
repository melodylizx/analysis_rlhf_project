#!/bin/bash
#SBATCH --job-name=reward_original
#SBATCH --ntasks=1
#SBATCH --partition=long                          # Ask for unkillable job                         
#SBATCH --gres=gpu:a100l:2  --constraint="dgx&ampere"                                      # Ask for 1 GPU
#SBATCH --mem=96G                                        # Ask for 10 GB of RAM
#SBATCH --time=120:00:00                                   # The job will run for 3 hours
#SBATCH --output=reward_out.txt
#SBATCH --error=reward_error.txt# Write the log on scratch
#SBATCH --constraint=80gb
#SBATCH -c 3



# 1. Load the required modules
module --quiet load anaconda/3
conda init

# 2. Load your environment
conda activate "rlhf"

cd ../../src/training/
deepspeed ./reward_model/train_reward_model_gptj.py --chpt_path /network/scratch/i/ines.arous/reward_model --deepspeed --deepspeed_config ./reward_model/ds_config_gpt_j.json



#conda deactivate
