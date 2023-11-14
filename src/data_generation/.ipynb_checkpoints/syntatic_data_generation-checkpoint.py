import pandas as pd
import ast
import random
import working_modeling

# Read data from CSV files
comp_df = pd.read_csv('comp.csv')
eval_df = pd.read_csv('eval.csv')
comp_unbiased_df = pd.read_csv('comp_biased.csv')

# Rename columns for clarity
rename_columns = {'Unnamed: 0': 'post_id', 'Unnamed: 1': 'comp_id_of_post'}
comp_df.rename(columns=rename_columns, inplace=True)
eval_df.rename(columns=rename_columns, inplace=True)
comp_unbiased_df.rename(columns=rename_columns, inplace=True)

# Process 'summary_eval' column
eval_df['summary_eval'] = eval_df['summary_eval'].apply(ast.literal_eval)
summary_eval_df = pd.DataFrame(eval_df['summary_eval'].tolist())
summary_eval_df = summary_eval_df.drop('compatible', axis=1)  # Drop 'compatible' column if present
eval_table = pd.concat([eval_df.drop('summary_eval', axis=1), summary_eval_df], axis=1)

# Group and average the evaluation scores
unique_eval = eval_table.groupby('summary_id').mean().drop("eval_id_of_post", axis=1).reset_index()

# Generate data for unbiased comparisons
num_comparisons = 92862
total_num_workers = 53
num_generated = 92862
comparisons_per_worker = random.randint(1, int(5000))
reliability = "high"

generated_df = working_modeling.get_the_generated_df_for_comp(
    total_num_workers, num_generated, reliability, num_comparisons, comparisons_per_worker, comp_unbiased_df
).reset_index()
generated_df = generated_df.drop(['assigned', 'index'], axis=1)
generated_df.rename(columns={'generated_label': 'worker_label', 'worker_choice': 'expert_label'}, inplace=True)
generated_df = generated_df.dropna()
generated_df = generated_df.astype({'expert_label': 'int', 'worker_label': 'int', 'id_1': 'int'})
generated_df = generated_df.reset_index(drop=True)

# Save the final DataFrame to a parquet file
final_file = working_modeling.to_parquet(generated_df, "train", reliability)


#assign bias
bias_accuracy = False # Bias towards highly accurate summaries
bias_coverage = False  # Bias towards high coverage summaries
bias_coherence = False  # Bias towards highly coherent summaries
bias_towards_0 = False  #Bias towards class 0 over class 1
bias_towards_1 = False  #Bias towards class 1 over class 0
generated_df = working_modeling.introduce_bias(generated_df, bias_accuracy, bias_coverage, bias_coherence, bias_towards_0, bias_towards_1)
generated_df = generated_df.drop_duplicates()
generated_df = generated_df.dropna()
generated_df
