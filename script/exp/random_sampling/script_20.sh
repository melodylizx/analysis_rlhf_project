#!/bin/bash

#SBATCH --partition=short-unkillable                    # Ask for unkillable job
#SBATCH --gres=gpu:a100l:4
#SBATCH --mem=128G                                        # Ask for 10 GB of RAM
#SBATCH --time=03:00:00                                   # The job will run for 3 hours
#SBATCH --output=./logs/random/20_out.txt
#SBATCH --error=./logs/random/20_error.txt
#SBATCH -c 4


# 1. Load the required modules
module load gcc/9.3.0
module load  cudatoolkit/11.7
source vrlhf/bin/activate

nvidia-smi

cd ./src/training/
#foldername=$(date +%Y_%m_%d_%H_%M)
#CHPTPATH=/network/scratch/i/ines.arous/experiment_reward_model/random_sampling/20/"$foldername"
#mkdir -p ${CHPTPATH}


#echo CHPTPATH
#echo $CHPTPATH
#deepspeed ./reward_model/train_reward_model_gptj.py --local_rank=0 --seed=3  --data_path="../../data/percent/0.2"  --chpt_path="${CHPTPATH}"
#BESTCHPT=$(deepspeed ./reward_model/gptj_reward_test.py --ckpt_path="${CHPTPATH}"| grep "${CHPTPATH}" | tail -n 1)
#echo BESTCHPT
#echo $BESTCHPT
BESTCHPT=/network/scratch/i/ines.arous/experiment_reward_model/random_sampling/20/2024_12_24_21_03/checkpoint-1300/pytorch_model.bin
foldername=$(date +%Y_%m_%d_%H_%M)
SAVEPATH=/network/scratch/i/ines.arous/ppo/random_sampling/20/2025_01_08_13_25
mkdir -p ${SAVEPATH}
accelerate launch --config_file configs/default_accelerate_config.yaml new_ppo.py --ckpt_path="${BESTCHPT}" --run_id='eqvy0vto' --save_path="${SAVEPATH}" --perc=20