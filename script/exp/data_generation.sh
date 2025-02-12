#!/bin/bash
#SBATCH --partition=main                   # Ask for unkillable job
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=128G                                        # Ask for 10 GB of RAM
#SBATCH --time=03:00:00                                  # The job will run for 3 hours
#SBATCH --output=./logs/data.txt
#SBATCH --error=./logs/data.txt
#SBATCH -c 2

module load gcc/9.3.0
module load  cudatoolkit/11.7
source vrlhf/bin/activate

rm -rf ./data/*
#load the dataset this will create pkl file in the data folder
cd ./src/data_generation
python data_loading.py

# generate dataset with different levels of reliability
python reliability.py

# generate dataset with different types of bias
python generate_biased_data.py

# random sample from the data
python generate_percent_data.py


# cluster data
#python cluster_data.py