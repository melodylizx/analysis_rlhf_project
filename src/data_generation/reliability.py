import pandas as pd
import ast
import random
import worker_modeling
from utils import create_directory
# Read data from CSV files
comp_val_df = pd.read_pickle('../../data/comp_val.pkl')
comp_train_df = pd.read_pickle('../../data/comp_train.pkl')

directory_path = '../../data/reliability'
create_directory(directory_path)
#scenario with one worker with fixed reliability

extreme_training_reliability = worker_modeling.fixed_reliability(0, comp_train_df)
low_training_reliability = worker_modeling.fixed_reliability(0.2, comp_train_df)
medium_training_reliability = worker_modeling.fixed_reliability(0.5, comp_train_df)
high_training_reliability = worker_modeling.fixed_reliability(0.8, comp_train_df)


# Save files to a parquet file
extreme_file = worker_modeling.to_parquet(extreme_training_reliability, directory_path+"/train", "extreme")
low_file = worker_modeling.to_parquet(low_training_reliability, directory_path+"/train", "low")
medium_file = worker_modeling.to_parquet(medium_training_reliability, directory_path+"/train", "medium")
high_file = worker_modeling.to_parquet(high_training_reliability, directory_path+"/train", "high")



extreme_val_reliability = worker_modeling.fixed_reliability(0, comp_val_df)
low_val_reliability = worker_modeling.fixed_reliability(0.2, comp_val_df)
medium_val_reliability = worker_modeling.fixed_reliability(0.5, comp_val_df)
high_val_reliability = worker_modeling.fixed_reliability(0.8, comp_val_df)


# Save files to a parquet file
extreme_val = worker_modeling.to_parquet(extreme_val_reliability, directory_path+"/val", "extreme")
low_val = worker_modeling.to_parquet(low_val_reliability, directory_path+"/val", "low")
medium_val = worker_modeling.to_parquet(medium_val_reliability, directory_path+"/val", "medium")
high_val = worker_modeling.to_parquet(high_val_reliability, directory_path+"/val", "high")
