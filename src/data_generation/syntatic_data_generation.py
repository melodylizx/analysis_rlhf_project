import pandas as pd
import ast

comp_df = pd.read_csv('comp.csv')
eval_df = pd.read_csv('eval.csv')
comp_df.rename(columns={'Unnamed: 0': 'post_id'}, inplace=True)
comp_df.rename(columns={'Unnamed: 1': 'comp_id_of_post'}, inplace=True)

eval_df.rename(columns={'Unnamed: 0': 'post_id'}, inplace=True)
eval_df.rename(columns={'Unnamed: 1': 'eval_id_of_post'}, inplace=True)
eval_df['summary_eval'] = eval_df['summary_eval'].apply(ast.literal_eval)

comp_unbiased_df = pd.read_csv('comp_biased.csv')
comp_unbiased_df.rename(columns={'Unnamed: 0': 'post_id'}, inplace=True)
comp_unbiased_df.rename(columns={'Unnamed: 1': 'comp_id_of_post'}, inplace=True)

# Extract the 'summary_eval' column as a DataFrame
summary_eval_df = pd.DataFrame(eval_df['summary_eval'].tolist())

# Drop the 'compatible' column if present
summary_eval_df = summary_eval_df.drop('compatible', axis=1)

# Concatenate the original DataFrame without 'summary_eval' column and the extracted columns
eval_table = pd.concat([eval_df.drop('summary_eval', axis=1), summary_eval_df], axis=1)

# Calculate the average and standard deviation for multiple evaluations
overall_stats = eval_table.groupby('summary_id')['overall'].agg(['mean', 'std'])
overall_stats = overall_stats[overall_stats['std'].notna()]  # Exclude summaries with single evaluations
accuracy_stats = eval_table.groupby('summary_id')['accuracy'].agg(['mean', 'std'])
accuracy_stats = accuracy_stats[accuracy_stats['std'].notna()]  # Exclude summaries with single evaluations
coverage_stats = eval_table.groupby('summary_id')['coverage'].agg(['mean', 'std'])
coverage_stats = coverage_stats[coverage_stats['std'].notna()]  # Exclude summaries with single evaluations
coherence_stats = eval_table.groupby('summary_id')['coherence'].agg(['mean', 'std'])
coherence_stats = coherence_stats[coherence_stats['std'].notna()]  # Exclude summaries with single evaluations

#collect all the evaluation scores for the summaries
unique_eval = eval_table.groupby('summary_id').mean().drop("eval_id_of_post", axis=1).reset_index()

import random
import numpy as np



"""# some scratch and analysis stuff"""

# Extract the relevant columns
cols = ['overall', 'accuracy', 'coverage', 'coherence']

# Compare each column against the others
comparison = {}
for col1 in cols:
    comparison[col1] = {}
    for col2 in cols:
        if col1 != col2:
            comparison[col1][col2] = (eval_table[col1] > eval_table[col2]).sum()

comparison_df = pd.DataFrame(comparison)
print(comparison_df)



correlation_with_coherence = eval_table.corrwith(eval_table['coverage'])

print(correlation_with_coherence)

# Merge comp_df with eval_unique for id_0
merged_df = comp_df.merge(unique_eval[['summary_id', 'coverage', 'overall']],
                          left_on='id_0', right_on='summary_id', suffixes=('', '_0'))

# Merge comp_df with eval_unique for id_1
merged_df = merged_df.merge(unique_eval[['summary_id', 'coverage', 'overall']],
                            left_on='id_1', right_on='summary_id', suffixes=('', '_1'))

# Filter based on your conditions
filtered_df = merged_df[
    (merged_df['worker_choice'] == 0) &
     (merged_df['coverage'] < merged_df['coverage_1'])
     #(merged_df['overall'] > merged_df['overall_1']))
    |
    (merged_df['worker_choice'] == 1) &
     (merged_df['coverage'] > merged_df['coverage_1'])
     #(merged_df['overall'] < merged_df['overall_1']))
]

count = len(filtered_df)

print(count)
#filtered_df

"""# unbiased"""

#unbiased
num_comaprisons= 92862#92862
total_num_experts=90  #comp_biased_df[:6543]['worker_id'].nunique()
total_num_workers=90
num_generated=92862  #total number of comparions we want to generate
comparisons_per_worker = random.randint(2, int(5000))
reliability="high"
generated_df = get_the_generated_df_for_comp(total_num_workers,num_generated, reliability,num_comaprisons, comparisons_per_worker,comp_unbiased_df).reset_index()
comp_unbiased_df= comp_unbiased_df.drop('assigned', axis=1)
generated_df= generated_df.drop('assigned', axis=1)
generated_df.rename(columns={'generated_label': 'worker_label','worker_choice': 'expert_label'}, inplace=True)
generated_df = generated_df.dropna()
generated_df['expert_label'] = generated_df['expert_label'].astype(int)
generated_df['worker_label'] = generated_df['worker_label'].astype(int)
generated_df['id_1'] = generated_df['id_1'].astype(int)
#generated_df = generated_df.drop_duplicates()
generated_df = generated_df.reset_index(drop=True)
generated_df

dd=to_parquet(generated_df, "train", reliability)

duplicate_rows = comp_unbiased.duplicated(subset=['id_0', 'id_1'], keep=False)

# Count the number of duplicate rows
count_duplicates = duplicate_rows.sum()

print(count_duplicates)

#unbiased
num_comaprisons= 727#92862
total_num_experts=53  #comp_biased_df[:6543]['worker_id'].nunique()
total_num_workers=53
num_generated=727  #total number of comparions we want to generate
reliability="perfect"
generated_df = get_the_generated_df_for_comp(total_num_workers,num_generated, reliability,num_comaprisons, comp_df).reset_index()
generated_df.rename(columns={'generated_label': 'worker_label','worker_choice': 'expert_label'}, inplace=True)

generated_df = generated_df.dropna()
generated_df['expert_label'] = generated_df['expert_label'].astype(int)
generated_df['worker_label'] = generated_df['worker_label'].astype(int)
generated_df['id_1'] = generated_df['id_1'].astype(int)
#generated_df = generated_df.drop_duplicates()
generated_df = generated_df.reset_index(drop=True)
generated_df

#medium
same_values = generated_df['expert_label'] == generated_df['worker_label']


# Count the number of rows with the same elements
count_same = same_values.sum()

print(count_same)

#unbiased= to_json(generated_df, reliability)

# !zip -r /content/my_files.zip /content/generated_unbiased/  # replace 'my_folder' with your directory's path
#
# from google.colab import files
# files.download('/content/my_files.zip')
#
# """# biased"""
#
# #biased
# num_comaprisons=727
# total_num_experts=59
# total_num_workers=59
# num_generated=727 #total number of comparions we want to generate
# reliability="low"
# generated_df = get_the_generated_df_for_comp(total_num_workers,num_generated, reliability,num_comaprisons, comp_df).reset_index()
# #comp_df.rename(columns={'generated_label': 'worker_label','worker_choice': 'expert_label'}, inplace=True)
#
#
# generated_df
#
# #assign bias
# bias_accuracy = False # Bias towards highly accurate summaries
# bias_coverage = False  # Bias towards high coverage summaries
# bias_coherence = False  # Bias towards highly coherent summaries
# bias_towards_0 = False  #Bias towards class 0 over class 1
# bias_towards_1 = False  #Bias towards class 1 over class 0
# generated_df = introduce_bias(generated_df, bias_accuracy, bias_coverage, bias_coherence, bias_towards_0, bias_towards_1)
# generated_df.rename(columns={'generated_label': 'worker_label','worker_choice': 'expert_label'}, inplace=True)
# #generated_df = generated_df.iloc[:, :-1]
# #generated_df  = generated_df .drop(generated_df .columns[1], axis=1)
# generated_df = generated_df.drop_duplicates()
# generated_df = generated_df.dropna()
# generated_df
#
# to_parquet(generated_df, "train", "no_bias")
#
# #bia= to_json_biased(generated_df, reliability, "bias_accuracy")
#
# pip install datasets
#
# from datasets import load_dataset
#
# dataset = load_dataset("CarperAI/openai_summarize_tldr", split="valid")
#
# dataset
#
# example = dataset[504]  # Dataset indexing starts at 0, so row number 505 corresponds to index 504.
# example
#
# pip install matplotlib
#
# import matplotlib.pyplot as plt
#
# # Data
# metrics = ['Rouge1', 'Rouge2', 'Rouge L']
# original_dataset = [0.302, 0.101, 0.233]
# bias_accurate = [0.284, 0.091, 0.222]
# bias_coverage = [0.283, 0.091, 0.222]
# bias_coherent = ["Still waiting", "for result", ""]
# bias_0 = [0.292, 0.100, 0.228]
# bias_1 = [0.291, 0.102, 0.228]
#
# # Setting the width of bars and position of bars on x-axis
# barWidth = 0.15
# r1 = list(range(len(original_dataset)))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]
# r4 = [x + barWidth for x in r3]
# r5 = [x + barWidth for x in r4]
#
# # Create bars
# plt.bar(r1, original_dataset, width=barWidth, color='b', edgecolor='grey', label='Original Dataset')
# plt.bar(r2, bias_accurate, width=barWidth, color='c', edgecolor='grey', label='Bias: Accurate Summaries')
# plt.bar(r3, bias_coverage, width=barWidth, color='m', edgecolor='grey', label='Bias: High Coverage')
# plt.bar(r4, bias_0, width=barWidth, color='y', edgecolor='grey', label='Bias: Towards 0')
# plt.bar(r5, bias_1, width=barWidth, color='r', edgecolor='grey', label='Bias: Towards 1')
#
# # Title & Subtitle
# plt.title('Comparison of Rouge Metrics across Datasets')
# plt.xlabel('Metrics', fontweight='bold')
# plt.ylabel('Score', fontweight='bold')
#
# # X axis
# plt.xticks([r + barWidth for r in range(len(original_dataset))], metrics)
#
# # Create legend & Show graphic
# plt.legend()
# plt.show()