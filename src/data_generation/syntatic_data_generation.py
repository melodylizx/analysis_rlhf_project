import pandas as pd
import ast
import random
import worker_modeling

# Read data from CSV files
comp_val_df = pd.read_pickle('../../data/comp_val.pkl')
comp_train_df = pd.read_pickle('../../data/comp_train.pkl')

#axis validation
overlap_axis_val = pd.read_pickle('../../data/overlap_axis_val.pkl')

extreme_training_reliability = worker_modeling.fixed_reliability(0, comp_train_df)
low_training_reliability = worker_modeling.fixed_reliability(0.2, comp_train_df)
medium_training_reliability = worker_modeling.fixed_reliability(0.5, comp_train_df)
high_training_reliability = worker_modeling.fixed_reliability(0.8, comp_train_df)


# Group and average the evaluation scores
unique_eval = overlap_axis_val[['summary_id', 'overall', 'accuracy', 'coverage', 'coherence']].groupby('summary_id').mean().reset_index()

# Generate data for unbiased comparisons
num_comparisons = 92862  #727,92862
total_num_workers = 53
num_generated = 92862
comparisons_per_worker = random.randint(100, int(5000))
reliability = "high"

generated_df = worker_modeling.get_the_generated_df_for_comp(
    total_num_workers, num_generated, reliability, num_comparisons, comparisons_per_worker, comp_train_df
).reset_index() #comparisons_val_df_overlap
generated_df = generated_df.drop(['assigned', 'index'], axis=1)
generated_df = generated_df.dropna()
generated_df = generated_df.astype({'worker_label': 'int'})
generated_df = generated_df.reset_index(drop=True)

# Save the final DataFrame to a parquet file
final_file = worker_modeling.to_parquet(generated_df, "train", reliability)

#assign bias
bias_towards_0 = False    #Bias towards class 0 over class 1
bias_towards_1 = False  #Bias towards class 1 over class 0
#has to use comparisons_val_df_overlap before generating data
bias_accuracy = True # Bias towards highly accurate summaries. 
bias_coverage = False  # Bias towards high coverage summaries
bias_coherence = False  # Bias towards highly coherent summaries
generated_df = introduce_bias(generated_df, bias_accuracy, bias_coverage, bias_coherence, bias_towards_0, bias_towards_1,unique_eval )
generated_df