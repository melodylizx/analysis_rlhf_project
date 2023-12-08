import pandas as pd
import worker_modeling
from utils import create_directory
# Read data from CSV files
comp_val_df = pd.read_pickle('../../data/comp_val.pkl')
comp_train_df = pd.read_pickle('../../data/comp_train.pkl')
directory_path = '../../data/multiple'
create_directory(directory_path)

# Generate data for unbiased comparisons with multiple workers
num_comparisons = len(comp_train_df)   #727,92862
total_num_workers = len(comp_train_df.worker.unique()) * 3
scenario_reliability = ['majority_low','majority_high','quarter_reliabilities' ,'random']

assignments = {}
assignments['min'] = comp_train_df.worker.value_counts().min()
assignments['max'] = comp_train_df.worker.value_counts().max()
assignments['mean'] = comp_train_df.worker.value_counts().mean()
assignments['std'] = comp_train_df.worker.value_counts().std()
worker_assignments = worker_modeling.n_assignments(num_comparisons,total_num_workers,assignments)
dict_gt = dict(zip(comp_train_df.index, comp_train_df['choice']))
dict_prompt = dict(zip(comp_train_df.index, comp_train_df['post']))
dict_sum_0 = dict(zip(comp_train_df.index, comp_train_df['summary_text_0']))
dict_sum_1 = dict(zip(comp_train_df.index, comp_train_df['summary_text_1']))

#scenario with multiple workers
for scenario in scenario_reliability:
    worker_assignments = worker_modeling.assign_reliability(worker_assignments,scenario)
    pair_worker, assignments, pairs_w = worker_modeling.assign_instances(num_comparisons, worker_assignments)
    pair_worker_assignment = pd.DataFrame(pair_worker.transpose(),columns=['pairs','worker'],dtype=int)
    pair_worker_assignment['choice'] = pair_worker_assignment['pairs'].map(dict_gt)
    dict_rel = dict(zip(worker_assignments['worker_id'], worker_assignments['reliability']))
    pair_worker_assignment_rel = worker_modeling.assign_rel_workers(pair_worker_assignment,dict_rel)
    mv_pairs = pair_worker_assignment_rel.groupby('pairs').mean().reset_index()
    mv_pairs['worker_label'] = mv_pairs.round({'worker_label': 0}).worker_label.astype(int)
    mv_pairs['post'] = mv_pairs['pairs'].map(dict_prompt)
    mv_pairs['summary_text_0'] = mv_pairs['pairs'].map(dict_sum_0)
    mv_pairs['summary_text_1'] = mv_pairs['pairs'].map(dict_sum_1)
    worker_modeling.to_parquet(mv_pairs, directory_path+"/train", scenario)