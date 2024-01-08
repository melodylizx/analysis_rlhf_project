import pandas as pd
from utils import create_directory
# Load the dataset
train_comparisons = pd.read_parquet('../../data/reliability/perfect/train_perfect.parquet')
valid_comparisons = pd.read_parquet('../../data/reliability/perfect/validation_perfect.parquet')

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
    return_train.to_parquet(directory_path +"/" + str(pct) +'/train_'+ str(pct) +'.parquet')
    return_valid.to_parquet(directory_path + "/" + str(pct) +'/validation_'+str(pct) +'.parquet')


    