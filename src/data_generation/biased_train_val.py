import pandas as pd
import worker_modeling
from utils import create_directory

bias_name = ["accuracy", "coherence", "coverage"]
splits = ["train","validation"]

#axis validationidation
overlap_axis_validation = pd.read_pickle('../../data/overlap_axis_validation.pkl')
comparisons_validation_df = pd.read_pickle('../../data/overlap_comp_validation.pkl')
# # Group and average the evaluationuation scores
aggregated_evaluation = overlap_axis_validation[['summary_id', 'overall', 'accuracy', 'coverage', 'coherence']].groupby('summary_id').mean().reset_index()
midpoint = 1+(len(comparisons_validation_df) // 2)
train_set = comparisons_validation_df[:midpoint]
val_set = comparisons_validation_df[midpoint:]

comparisons_train = pd.read_pickle('../../data/comp_train.pkl')
comparisons_val = pd.read_pickle('../../data/comp_validation.pkl')

# Extract the intersection of 'id' values as a list
id_inter_train = set(comparisons_validation_df['id']).intersection(set(comparisons_train['id']))
filtered_train = comparisons_train[~comparisons_train['id'].isin(id_inter_train)]

id_inter_val = set(comparisons_validation_df['id']).intersection(set(comparisons_val['id']))
filtered_val = comparisons_val[~comparisons_val['id'].isin(id_inter_val)]

# create 90% of the dataset  non-biased
compl_size_train = len(train_set) *95//5
compl_size_val = len(val_set) *95//5

compl_train = filtered_train[:compl_size_train]
compl_val = filtered_val[:compl_size_val]

appended_train = pd.concat([train_set, compl_train], ignore_index=True)
appended_val = pd.concat([val_set, compl_val], ignore_index=True)


# Randomly shuffle the rows
shuffled_train = appended_train.sample(frac=1, random_state=42)  # frac=1 means shuffle all rows
shuffled_val = appended_val.sample(frac=1, random_state=42)  # frac=1 means shuffle all rows

directory_path = '../../data/witness'
create_directory(directory_path)

shuffled_train = shuffled_train.rename(columns={'choice':'worker_label'})
shuffled_val = shuffled_val.rename(columns={'choice':'worker_label'})
worker_modeling.to_parquet(shuffled_train,directory_path, 'train','sample')
worker_modeling.to_parquet(shuffled_val,directory_path, 'validation','sample')