from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import numpy as np
import pandas as pd

comp_train_df = pd.read_pickle('../../data/comp_train.pkl')
values = comp_train_df['worker'].value_counts()
bins = np.outer(10.0 ** np.arange(-1, 7), [1, 2, 5]).ravel()[:-2]
plt.figure(figsize=(16, 5))
heights, _ = np.histogram(values, bins=bins)
labels = [
    f'{x0:.1f}-{x1:.1f}' if x0 < 1 else f'{x0:.0f}-{x1:.0f}' if x0 < 10000 else f'{x0 / 1000:.0f}-{x1 / 1000:.0f}K'
    for x0, x1 in zip(bins[:-1], bins[1:])]

ax1 = sns.barplot(x=[lbl for lbl, h in zip(labels, heights) if h > 0], y=heights[heights > 0])
ax1.margins(x=0.01)
sns.despine()
ax1.set(xlabel="# Annotated Pairs", ylabel="# Workers")
sns.despine()
plt.title("Workers Answers Distribution")


# Compute average worker confidence for each bin
comp_train_df['worker_count_bin'] = pd.cut(comp_train_df['worker'].map(values), bins=bins, labels=labels)
options = [lbl for lbl, h in zip(labels, heights) if h > 0]
options[0]='0.5-1.0'
list_opt=[]
for opt in options:
    comp_opt = comp_train_df[comp_train_df['worker_count_bin']==opt]
    list_opt.append(comp_opt['conf'].mean())
# Plot average worker confidence per bin
# Plot average worker confidence per bin using a line plot
ax2 = ax1.twinx()
ax2 = sns.lineplot(x=[lbl for lbl, h in zip(labels, heights) if h > 0], y=list_opt, marker='o',color='red')
ax2.set( ylabel="Average Worker Confidence")

plt.tight_layout()
# plt.show()

plt.savefig('../../data/workers_answers_dist.png')
# plt.show()