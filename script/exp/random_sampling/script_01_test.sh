#!/bin/bash

#SBATCH --partition=long                     # Ask for unkillable job
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=0                                        # Ask for 10 GB of RAM
#SBATCH --time=1:00:00                                   # The job will run for 3 hours
#SBATCH --output=./logs/random/01_out_res.txt
#SBATCH --error=./logs/random/01_error_res.txt
#SBATCH -c 2


# 1. Load the required modules
module load gcc/9.3.0
module load  cudatoolkit/11.7
source vrlhf/bin/activate

nvidia-smi

cd ./src/training/
foldername=$(date +%Y_%m_%d_%H_%M)
CHPTPATH=/network/scratch/i/ines.arous/experiment_reward_model/random_sampling/01/"$foldername"

mkdir -p ${CHPTPATH}



deepspeed ./reward_model/train_reward_model_gptj.py --local_rank=0 --seed=3  --data_path="../../data/percent/0.01"  --chpt_path="${CHPTPATH}"
#BESTCHPT=$(python ./reward_model/gptj_reward_test.py --ckpt_path="${CHPTPATH}"| tail -n 1)
#echo BESTCHPT
#echo $BESTCHPT
#SAVEPATH=/network/scratch/i/ines.arous/ppo/random_sampling/01/"$foldername"
#mkdir -p ${SAVEPATH}
#accelerate launch --config_file configs/default_accelerate_config.yaml new_ppo.py --ckpt_path="${BESTCHPT}" --save_path="${SAVEPATH}"
#python trlx_inference_gptj.py --ckpt_path="${BESTCHPT}" --save_path="${SAVEPATH}" --csv_path="${SAVEPATH}"