#!/bin/bash

#SBATCH --partition=long                          # Ask for unkillable job                         
#SBATCH --gres=gpu:a100l:2                                        # Ask for 1 GPU
#SBATCH --mem=96G                                        # Ask for 10 GB of RAM
#SBATCH --time=96:00:00                                   # The job will run for 3 hours
#SBATCH -o /home/mila/z/zixuan.li/output/unbiased/hlhh-%j.out  # Write the log on scratch
#SBATCH --constraint=80gb
#SBATCH -c 3


# 1. Load the required modules
module --quiet load anaconda/3
conda init

# 2. Load your environment
conda activate "sum"


cd /home/mila/z/zixuan.li/trlx
deepspeed examples/summarize_rlhf/reward_model/train_reward_model_new.py --path="/network/scratch/z/zixuan.li/generated_dataset/hlhh" --output="/network/scratch/z/zixuan.li/experiment_reward_model/hlhh"

#cd /home/mila/z/zixuan.li/trlx/examples/summarize_rlhf
#accelerate launch --config_file configs/default_accelerate_config.yaml new_ppo.py --ckpt_path="/network/scratch/z/zixuan.li/experiment_reward_model/hlhh/checkpoint-300/pytorch_model.bin" --save_path="/network/scratch/z/zixuan.li/experiment_ppo_model/low"

#conda deactive

#python trlx_inference_gptj.py --ckpt_path="/network/scratch/z/zixuan.li/experiment_reward_model/low/checkpoint-300/pytorch_model.bin" --save_path="/network/scratch/z/zixuan.li/experiment_ppo_model/low" --csv_path="/network/scratch/z/zixuan.li/result_of_experiments/low/ppo_with_reward_scores.csv"