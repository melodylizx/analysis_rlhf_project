#!/bin/bash

#SBATCH --partition=short-unkillable                  # Ask for unkillable job
#SBATCH --gres=gpu:a100l:4
#SBATCH --mem=0
#SBATCH --time=03:00:00                                   # The job will run for 3 hours
#SBATCH --output=./logs/margin_uncertainty_20_out.txt
#SBATCH --error=./logs/margin_uncertainty_20_error.txt
#SBATCH -c 4


# 1. Load the required modules
module load gcc/9.3.0
module load cudatoolkit/11.7
module load python/3.10
source vrlhf/bin/activate

nvidia-smi

#cd ./src/data_generation/
#deepspeed ./generate_uncertainty_sampling_data_old.py
##cd ../../


cd ./src/training/
foldername=$(date +%Y_%m_%d_%H_%M)


CHPTPATH=/network/scratch/i/ines.arous/experiment_reward_model/margin_confidence/"$foldername"
mkdir -p ${CHPTPATH}
echo ">>>>>>>> CHPTPATH: $CHPTPATH"
deepspeed ./reward_model/train_reward_model_gptj.py --local_rank=0 --seed=3  --data_path="../../data/margin_confidence"  --chpt_path="${CHPTPATH}"
reward_test_out=$(deepspeed ./reward_model/gptj_reward_test.py --ckpt_path="${CHPTPATH}")
echo $reward_test_out
BESTCHPT=$(echo "${reward_test_out}" | grep "${CHPTPATH}" | tail -n 1)
echo ">>>>>>>> BESTCHPT: $BESTCHPT"