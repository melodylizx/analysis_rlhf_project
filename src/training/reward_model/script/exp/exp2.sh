#!/bin/bash

#SBATCH --partition=long                          # Ask for unkillable job                         
#SBATCH --gres=gpu:a100l:2                                        # Ask for 1 GPU
#SBATCH --mem=96G                                        # Ask for 10 GB of RAM
#SBATCH --time=48:00:00                                   # The job will run for 3 hours
#SBATCH -o /home/mila/z/zixuan.li/output/original_ppo-%j.out  # Write the log on scratch
#SBATCH --constraint=80gb
#SBATCH -c 3


# 1. Load the required modules
module --quiet load anaconda/3
conda init
# 2. Load your environment
conda activate "sum"

cd /home/mila/z/zixuan.li/trlx/examples/summarize_rlhf
accelerate launch --config_file configs/default_accelerate_config.yaml new_ppo.py -- --ckpt_path="/network/scratch/z/zixuan.li/experiment_reward_model/coverage/checkpoint-100/pytorch_model.bin" --save_path="/network/scratch/z/zixuan.li/biased_coverage"

#cd /home/mila/z/zixuan.li/trlx
#deepspeed examples/summarize_rlhf/trlx_gptj_text_summarization.py
#--checkpoint_dir /network/scratch/z/zixuan.li/

