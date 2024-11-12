#!/bin/bash
#SBATCH --job-name=data_genertation
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --partition=main
#SBATCH --time=2:00:00                                   # The job will run for 3 hours
#SBATCH --output=./logs/data_gen_out.txt
#SBATCH --error=./logs/data_gen_error.txt# Write the log on scratch
#SBATCH -c 3

module load miniconda/3
module load  cuda/11.7
conda activate "vrlhf"

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
python cluster_data.py