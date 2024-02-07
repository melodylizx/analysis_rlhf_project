from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import numpy as np
import pandas as pd
import worker_modeling
from utils import create_directory

directory_path = '../../data/worker_answer'
create_directory(directory_path)

comp_train_df = pd.read_pickle('../../data/comp_validation.pkl')
values = comp_train_df['worker'].value_counts()
comp_train_df['worker_val'] = comp_train_df['worker'].map(values)
comp_train_df.loc[:, 'worker_label'] = comp_train_df['choice']
values_df = pd.DataFrame(values)
values_df['perc'] = values_df['count']*100/comp_train_df.shape[0]

bins = np.outer(10.0 ** np.arange(-1, 7), [1, 2, 5]).ravel()[:-2]
# Define the percentage intervals for the bins
bin_edges = np.array([0,1, 2, 4, 6, 8, 100])  # The last interval goes up to 100%


plt.figure(figsize=(16, 5))
heights, _ = np.histogram(values_df['perc'] , bins=bin_edges)
labels = ['0to1','1to2','2to4','4to6','6to8','above8']

ax1 = sns.barplot(x=[lbl for lbl, h in zip(labels, heights) if h > 0], y=heights[heights > 0])
ax1.margins(x=0.01)
sns.despine()
ax1.set(xlabel="# Annotated Pairs", ylabel="# Workers")
sns.despine()
plt.title("Workers Answers Distribution")

directory_path
comp_train_df['worker_count_bin'] = pd.cut(comp_train_df['worker'].map(values_df['perc']), bins=bin_edges, labels=labels)
for lbl in labels:
    w_comp = comp_train_df[comp_train_df['worker_count_bin']==lbl]
    w_comp = worker_modeling.to_parquet(w_comp, directory_path, "validation", lbl)

# # Compute average worker confidence for each bin
# comp_train_df['worker_count_bin'] = pd.cut(comp_train_df['worker'].map(values), bins=bins, labels=labels)
# options = [lbl for lbl, h in zip(labels, heights) if h > 0]
# options[0]='0.5-1.0'
# list_opt=[]
# for opt in options:
#     comp_opt = comp_train_df[comp_train_df['worker_count_bin']==opt]
#     list_opt.append(comp_opt['conf'].mean())
# # Plot average worker confidence per bin
# # Plot average worker confidence per bin using a line plot
# ax2 = ax1.twinx()
# ax2 = sns.lineplot(x=[lbl for lbl, h in zip(labels, heights) if h > 0], y=list_opt, marker='o',color='red')
# ax2.set( ylabel="Average Worker Confidence")
#
# plt.tight_layout()
# # plt.show()
#
# # plt.savefig('../../data/workers_answers_dist.png')
# plt.show()
#
# values_df[['min_conf','max_conf','mean_conf']] = 0
# values_df = values_df.reset_index()
# for idx, worker in values_df['worker'].items():
#     comp_train_worker = comp_train_df[comp_train_df['worker'] == worker]
#     values_df.loc[idx,'min_conf'] = comp_train_worker[comp_train_worker['conf']!=0].conf.min()
#     values_df.loc[idx,'max_conf'] = comp_train_worker[comp_train_worker['conf']!=0].conf.max()
#     values_df.loc[idx,'mean_conf'] = comp_train_worker[comp_train_worker['conf']!=0].conf.mean()
#
# least_confident = comp_train_df[comp_train_df['worker_count_bin']=='10-20K']
# average_confident = comp_train_df[comp_train_df['worker_count_bin'].isin(['500-1000','1000-2000'])]
# low_confident = comp_train_df[comp_train_df['worker_count_bin'].isin(['2000-5000'])]