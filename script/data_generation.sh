#!/bin/bash
#SBATCH --job-name=data_genertation
#SBATCH --ntasks=1
#SBATCH --partition=main
#SBATCH --time=2:00:00                                   # The job will run for 3 hours
#SBATCH --output=data_gen_out.txt
#SBATCH --error=data_gen_error.txt# Write the log on scratch
#SBATCH -c 3

module --quiet load anaconda/3
conda activate "rlhf"

#load the dataset this will create pkl file in the data folder
cd ./src/data_generation
python data_loading.py

# generate dataset with different levels of reliability
python reliability.py

# generate dataset with different levels of bias



