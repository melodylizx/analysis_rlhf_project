#!/bin/bash

#SBATCH --partition=long                          # Ask for unkillable job                         
#SBATCH --gres=gpu:a100l:2                                        # Ask for 1 GPU
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH --time=10:00:00                                   # The job will run for 3 hours
#SBATCH -o /home/mila/z/zixuan.li/try-%j.out  # Write the log on scratch
#SBATCH --constraint=80gb
#SBATCH -c 2


# 1. Load the required modules
module --quiet load anaconda/3
conda init
# 2. Load your environment
conda activate "sum"

python alpha.py

