import pandas as pd
import ast
import random
import numpy as np
import worker_modeling
from utils import create_directory

# Read data from CSV files
comp_validation_df = pd.read_pickle('../../data/comp_validation.pkl')
comp_train_df = pd.read_pickle('../../data/comp_train.pkl')

directory_path = '../../data/reliability'
create_directory(directory_path)
for rel_perc in np.arange(0.0, 1.1, 0.1):
    training_reliability = worker_modeling.fixed_reliability(round(rel_perc,1), comp_train_df)
    training_file = worker_modeling.to_parquet(training_reliability, directory_path, "train", str(int(round(rel_perc,1)*100)))
    validation_reliability = worker_modeling.fixed_reliability(round(rel_perc,1), comp_validation_df)
    extreme_validation = worker_modeling.to_parquet(validation_reliability, directory_path, "validation",
                                                    str(int(round(rel_perc,1)*100)))

