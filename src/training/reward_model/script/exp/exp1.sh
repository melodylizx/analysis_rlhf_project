#!/bin/bash

#SBATCH --partition=long                          # Ask for unkillable job                         
#SBATCH --gres=gpu:a100l:2                                        # Ask for 1 GPU
#SBATCH --mem=96G                                        # Ask for 10 GB of RAM
#SBATCH --time=48:00:00                                   # The job will run for 3 hours
#SBATCH -o /home/mila/z/zixuan.li/output/coverage——reward-%j.out  # Write the log on scratch
#SBATCH --constraint=80gb
#SBATCH -c 3



# 1. Load the required modules
module --quiet load anaconda/3
conda init

# 2. Load your environment
conda activate "sum"

cd /home/mila/z/zixuan.li/trlx
deepspeed examples/summarize_rlhf/reward_model/train_reward_model_new.py --path="/network/scratch/z/zixuan.li/generated_dataset" --output="/network/scratch/z/zixuan.li/experiment_reward_model/coverage"


#conda deactive
