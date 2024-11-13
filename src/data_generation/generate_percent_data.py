import pandas as pd
import worker_modeling
from utils import create_directory
# Load the dataset
train_comparisons = pd.read_pickle('../../data/comp_train.pkl')
valid_comparisons = pd.read_pickle('../../data/comp_validation.pkl')

train_comparisons = train_comparisons.rename(columns={"choice": "worker_label"})
valid_comparisons = valid_comparisons.rename(columns={"choice": "worker_label"})
# Define the percentages to be used for training and validation
percentages = [0.01, 0.05, 0.10, 0.20, 0.50, 0.80]
directory_path = '../../data/percent'
create_directory(directory_path)
# Generate datasets for each percentage case
def get_percent_dataset(pct):
    # Sample the dataset
    return_train = train_comparisons.sample(frac=pct,random_state=42)
    return_valid = valid_comparisons.sample(frac=pct,random_state=42)
    
    return return_train,return_valid

for pct in percentages:
    return_train,return_valid = get_percent_dataset(pct)
    create_directory(directory_path+"/" + str(pct))
    worker_modeling.to_parquet(return_train, directory_path+"/", "train", str(pct))
    worker_modeling.to_parquet(return_valid, directory_path+"/", "validation", str(pct))


    