#!/bin/bash

#SBATCH --partition=long           # Ask for unkillable job                         
#SBATCH --gres=gpu:a100l:2                                        # Ask for 1 GPU
#SBATCH --mem=96G                                        # Ask for 10 GB of RAM
#SBATCH --time=48:00:00                                   # The job will run for 3 hours
#SBATCH -o /home/mila/z/zixuan.li/output/unbiased/medium-%j.out  # Write the log on scratch
#SBATCH --constraint=80gb
#SBATCH -c 3


# 1. Load the required modules
module --quiet load anaconda/3
conda init

# 2. Load your environment
conda activate "sum"



cd /home/mila/z/zixuan.li/trlx
deepspeed examples/summarize_rlhf/reward_model/train_reward_model_gptj.py  --data_path="/network/scratch/z/zixuan.li/generated_dataset/medium"  --chpt_path="/network/scratch/z/zixuan.li/experiment_reward_model/medium"

BEST_CHECKPOINT_PATH=$(jq -r '.best_model_checkpoint' /network/scratch/z/zixuan.li/experiment_reward_model/medium/checkpoint-5000/trainer_state.json)

python examples/summarize_rlhf/reward_model/gptj_reward_test.py --ckpt_path="${BEST_CHECKPOINT_PATH}/pytorch_model.bin"

#cd /home/mila/z/zixuan.li/trlx/examples/summarize_rlhf
#accelerate launch --config_file configs/default_accelerate_config.yaml new_ppo.py --ckpt_path="${BEST_CHECKPOINT_PATH}/pytorch_model.bin" --save_path="/network/scratch/z/zixuan.li/experiment_ppo_model/medium"

#python rouge.py --ckpt_path="${BEST_CHECKPOINT_PATH}/pytorch_model.bin" --save_path="/network/scratch/z/zixuan.li/experiment_ppo_model/medium" --csv_path="/network/scratch/z/zixuan.li/result_of_experiments/medium/ppo_with_reward_scores.csv"