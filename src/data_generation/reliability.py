import pandas as pd
import ast
import random
import worker_modeling
from utils import create_directory
# Read data from CSV files
comp_validation_df = pd.read_pickle('../../data/comp_validation.pkl')
comp_train_df = pd.read_pickle('../../data/comp_train.pkl')

directory_path = '../../data/reliability'
create_directory(directory_path)
#scenario with one worker with fixed reliability

extreme_training_reliability = worker_modeling.fixed_reliability(0, comp_train_df)
low_training_reliability = worker_modeling.fixed_reliability(0.2, comp_train_df)
medium_training_reliability = worker_modeling.fixed_reliability(0.5, comp_train_df)
high_training_reliability = worker_modeling.fixed_reliability(0.8, comp_train_df)
perfect_training_reliability = worker_modeling.fixed_reliability(1, comp_train_df)



# Save files to a parquet file
extreme_file = worker_modeling.to_parquet(extreme_training_reliability, directory_path, "train", "extreme")
low_file = worker_modeling.to_parquet(low_training_reliability, directory_path,"train", "low")
medium_file = worker_modeling.to_parquet(medium_training_reliability, directory_path,"train", "medium")
high_file = worker_modeling.to_parquet(high_training_reliability, directory_path, "train", "high")
perfect_file = worker_modeling.to_parquet(perfect_training_reliability, directory_path, "train", "perfect")



extreme_validation_reliability = worker_modeling.fixed_reliability(0, comp_validation_df)
low_validation_reliability = worker_modeling.fixed_reliability(0.2, comp_validation_df)
medium_validation_reliability = worker_modeling.fixed_reliability(0.5, comp_validation_df)
high_validation_reliability = worker_modeling.fixed_reliability(0.8, comp_validation_df)
perfect_validation_reliability = worker_modeling.fixed_reliability(1, comp_validation_df)


# Save files to a parquet file
extreme_validation = worker_modeling.to_parquet(extreme_validation_reliability, directory_path, "validation", "extreme")
low_validation = worker_modeling.to_parquet(low_validation_reliability, directory_path, "validation", "low")
medium_validation = worker_modeling.to_parquet(medium_validation_reliability, directory_path, "validation", "medium")
high_validation = worker_modeling.to_parquet(high_validation_reliability, directory_path, "validation", "high")
perfect_validation = worker_modeling.to_parquet(perfect_validation_reliability, directory_path, "validation", "perfect")

