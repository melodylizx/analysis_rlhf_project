from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import numpy as np
import pandas as pd

comp_train_df = pd.read_csv('../../data/comp_train.csv')
values = comp_train_df['worker'].value_counts()
bins = np.outer(10.0 ** np.arange(-1, 7), [1, 2, 5]).ravel()[:-2]
plt.figure(figsize=(16, 5))
heights, _ = np.histogram(values, bins=bins)
labels = [
    f'{x0:.1f}-{x1:.1f}' if x0 < 1 else f'{x0:.0f}-{x1:.0f}' if x0 < 10000 else f'{x0 / 1000:.0f}-{x1 / 1000:.0f}K'
    for x0, x1 in zip(bins[:-1], bins[1:])]

ax = sns.barplot(x=[lbl for lbl, h in zip(labels, heights) if h > 0], y=heights[heights > 0])
ax.margins(x=0.01)
sns.despine()
plt.tight_layout()
plt.show()