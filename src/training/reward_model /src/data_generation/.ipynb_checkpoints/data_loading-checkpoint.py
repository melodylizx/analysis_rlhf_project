# -*- coding: utf-8 -*-
# load datasets
from datasets import load_dataset
import pandas as pd
import warnings
from utils import create_post2summaries_ratings, create_dict_comparision, filter_comparisons, filter_axis, update_summary_t2id, filter_biased_comp

warnings.filterwarnings('ignore')

comparisons = load_dataset("openai/summarize_from_feedback", name="comparisons")
axis = load_dataset("openai/summarize_from_feedback", name="axis")

comparisons_train = comparisons["train"]
comparisons_val = comparisons["validation"]
axis_val = axis["validation"]
axis_test = axis["test"]

"""# filtering datasets"""
# create a dictionary small_post_map {post_id: post_content}
# used only on the validation dataset because it contains evaluation on properties
# created for the validation set

##### Ines refactor #######
# create a dictionary comparisons_val_dict {post_id: post_content}
# used only on the validation dataset because it contains evaluation on properties
# created for the validation set
# create a dictionary comparisons_train_val_dict {post_id: post_content}
# used for the training and validation datasets
comparisons_val_dict = create_dict_comparision(comparisons_val)
comparisons_train_val_dict = create_dict_comparision(comparisons_train)
comparisons_train_val_dict.update(comparisons_val_dict)

#create_post2summaries_ratings
post2summaries, summary_text2id, repeated_summaries_n = create_post2summaries_ratings(axis_val)

# filtering axis:
axis_filtered = {}
summary_id, repeated_summaries = 0, 0
summary_t2id = {}
axis_filtered, summary_id, summary_t2id, repeated_summaries = filter_axis(axis_val, axis_filtered, summary_id, summary_t2id,repeated_summaries)
#axis_filtered, summary_id, summary_t2id, repeated_summaries = filter_axis(axis_test, axis_filtered, summary_id, summary_t2id,repeated_summaries)

# filtering comparisons: post_id, summary_text, choice_worker, worker_id, confidence
comparisons_filtered = {}
filtered_ids = []
#comparisons_filtered, filtered_ids = filter_comparisons(comparisons_train, comparisons_filtered, summary_t2id,filtered_ids)
comparisons_filtered, filtered_ids = filter_comparisons(comparisons_val, comparisons_filtered, summary_t2id,filtered_ids)
# 739 unique summaries

#full unbiased dataset
filtered_ids_unbiased = []
comparisons_filtered_unbiased = {}
complete_summary_t2id = update_summary_t2id(comparisons_train, summary_t2id)
comp_unbiased_train= filter_biased_comp(comparisons_train, comparisons_filtered_unbiased, summary_t2id,complete_summary_t2id,filtered_ids)
comp_unbiased_train = pd.concat({k: pd.DataFrame(v) for k, v in comparisons_filtered_unbiased.items()})

#save results
comp = pd.concat({k: pd.DataFrame(v) for k, v in comparisons_filtered.items()})
axis_ = pd.concat({k: pd.DataFrame(v) for k, v in axis_filtered.items()})
comp_unbiased = pd.concat({k: pd.DataFrame(v) for k, v in comparisons_filtered_unbiased.items()})
axis_f = axis_[axis_['summary_id'].isin(filtered_ids)]

# comp is the comparison from validation
comp.to_csv('../../data/comp.csv')
axis_f.to_csv('../../data/eval.csv')
# comp_unbiased is the comparison from training
comp_unbiased.to_csv('../../data/comp_unbiased.csv')