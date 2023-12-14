#!/bin/bash

#SBATCH --partition=long                     # Ask for unkillable job
#SBATCH --gres=gpu:a100l:2   --constraint="dgx&ampere"                                      # Ask for 1 GPU
#SBATCH --mem=96G                                        # Ask for 10 GB of RAM
#SBATCH --time=3-6:00:00                                   # The job will run for 3 hours
#SBATCH --output=./logs/perfect_testasval_out.txt
#SBATCH --error=./logs/perfect_testasval_error.txt
#SBATCH --constraint=80gb
#SBATCH -c 2


# 1. Load the required modules
module --quiet load anaconda/3
conda activate "rlhf"

cd ./src/training/
foldername=$(date +%Y_%m_%d_%H_%M)
CHPTPATH=/network/scratch/i/ines.arous/experiment_reward_model/perfect/test_as_val_"$foldername"

mkdir -p ${CHPTPATH}
deepspeed ./reward_model/train_reward_model_gptj.py --local_rank=0  --data_path="/network/scratch/i/ines.arous/data_rlhf/reliability/perfect"  --chpt_path="${CHPTPATH}"


BEST_CHECKPOINT_PATH=$(jq -r '.best_model_checkpoint' ${CHPTPATH}/checkpoint-5000/trainer_state.json)

python ./reward_model/gptj_reward_test.py --ckpt_path="${BEST_CHECKPOINT_PATH}/pytorch_model.bin"