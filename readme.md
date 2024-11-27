# Analysis of RLHF
This is the repo for the project the H in RLHF.
First, you need to preapre your environment using the following steps.

## Installation

You need cuda 11.7 and Python 3.10.4.

``module load miniconda/3``
``module load  cuda/11.7``

First, create a virtual environment using:

``conda create -n vrlhf python==3.10.4``
``conda activate vrlhf``

Second install pytorch 2.0.0
``conda install pytorch==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia``

Then, install the rest of the packages using the provided requirements file.
``pip install -r requirements.txt``

## Data Generation
- Generate the data needed for the experiments:
``sbatch script/exp/data_generation.sh``

Successfully installed huggingface-hub-0.26.2 safetensors-0.4.5 tokenizers-0.20.3 transformers-4.46.2

pip install bert-score
