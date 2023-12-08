#!/bin/bash

#SBATCH --partition=main                          # Ask for unkillable job                         
#SBATCH --gres=gpu:a100l:2                                        # Ask for 1 GPU
#SBATCH --mem=48G                                        # Ask for 10 GB of RAM
#SBATCH --time=24:00:00                                   # The job will run for 3 hours
#SBATCH -o /home/mila/z/zixuan.li/output/unbiased/try-%j.out  # Write the log on scratch
#SBATCH --constraint=80gb
#SBATCH -c 3



# 1. Load the required modules
module --quiet load anaconda/3
conda init

# 2. Load your environment
conda activate "sum"

BEST_CHECKPOINT_PATH=$(jq -r '.best_model_checkpoint' /network/scratch/z/zixuan.li/experiment_reward_model/perfect0/checkpoint-3000/trainer_state.json)

cd /home/mila/z/zixuan.li/trlx
python examples/summarize_rlhf/reward_model/gptj_reward_test.py --ckpt_path="${BEST_CHECKPOINT_PATH}/pytorch_model.bin"
