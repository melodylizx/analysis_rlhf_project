# -*- coding: utf-8 -*-
# load datasets
from datasets import load_dataset
import pandas as pd
import warnings
from utils import create_post2summaries_ratings, create_dict_comparision, process_comparisons_df, update_summary_t2id

warnings.filterwarnings('ignore')

comparisons = load_dataset("openai/summarize_from_feedback", name="comparisons")
axis = load_dataset("openai/summarize_from_feedback", name="axis")

comparisons_train = comparisons["train"]
comparisons_val = comparisons["validation"]
axis_val = axis["validation"]


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
rating_val_overlap, repeated_summaries, summary_text2id = create_post2summaries_ratings(axis_val)

# processing comparisons to keep only those from validation set: post_id, summary_text, choice_worker, worker_id, confidence
# 739 unique summaries
comparisons_val_df_overlap = process_comparisons_df(comparisons_val,summary_text2id)

#full training dataset
complete_summary_t2id = update_summary_t2id(comparisons_train, summary_text2id)
comparisons_train_df = process_comparisons_df(comparisons_train,complete_summary_t2id)

#full validation set
complete_summary_t2id = update_summary_t2id(comparisons_val, summary_text2id)
comparisons_val_df = process_comparisons_df(comparisons_val,complete_summary_t2id)


#save results
#dataset preparation for biased experiment with overlap between evaluation criteria and comparisons
comparisons_val_df_overlap.to_pickle('../../data/overlap_comp_val.pkl')
rating_val_overlap.to_pickle('../../data/overlap_axis_val.pkl')

#dataset preparation for the comparison sets
comparisons_train_df.to_pickle('../../data/comp_train.pkl')
comparisons_val_df.to_pickle('../../data/comp_val.pkl')