# -*- coding: utf-8 -*-
# load datasets
from datasets import load_dataset
import pandas as pd
import warnings
from utils import filter_duplicates_disagreement, create_post2summaries_ratings, create_dict_comparision, process_comparisons_df, update_summary_t2id
import worker_modeling

warnings.filterwarnings('ignore')

comparisons = load_dataset("openai/summarize_from_feedback", name="comparisons")
axis = load_dataset("openai/summarize_from_feedback", name="axis")

comparisons_train = comparisons["train"]
comparisons_val = comparisons["validation"]
axis_val = axis["validation"]


##### Ines refactor #######
# create a dictionary comparisons_validation_dict {post_id: post_content}
# used only on the validation dataset because it contains evaluation on properties
# created for the validation set
# create a dictionary comparisons_train_validation_dict {post_id: post_content}
# used for the training and validation datasets
comparisons_validation_dict = create_dict_comparision(comparisons_val)
comparisons_train_validation_dict = create_dict_comparision(comparisons_train)
comparisons_train_validation_dict.update(comparisons_validation_dict)

#create_post2summaries_ratings
rating_validation_overlap, repeated_summaries, summary_text2id = create_post2summaries_ratings(axis_val)

# processing comparisons to keep only those from validation set: post_id, summary_text, choice_worker, worker_id, confidence
# 739 unique summaries
comparisons_validation_df_overlap = process_comparisons_df(comparisons_val,summary_text2id)
comp_val_overlap_df_dup = filter_duplicates_disagreement(comparisons_validation_df_overlap)

#full training dataset
complete_summary_t2id = update_summary_t2id(comparisons_train, summary_text2id)
comparisons_train_df = process_comparisons_df(comparisons_train,complete_summary_t2id)
comp_train_df_dup = filter_duplicates_disagreement(comparisons_train_df)

#full validation set
complete_summary_t2id = update_summary_t2id(comparisons_val, summary_text2id)
comparisons_validation_df = process_comparisons_df(comparisons_val,complete_summary_t2id)
comp_val_df_dup = filter_duplicates_disagreement(comparisons_validation_df)


#save results
#dataset preparation for biased experiment with overlap between evaluation criteria and comparisons

comp_val_overlap_df_dup.to_pickle('../../data/overlap_comp_validation.pkl')
rating_validation_overlap.to_pickle('../../data/overlap_axis_validation.pkl')

#dataset preparation for the comparison sets
comp_train_df_dup.to_pickle('../../data/comp_train.pkl')

#keep 10k of pairs for testing
validation_set = comp_val_df_dup[:-10000]
test_set = comp_val_df_dup[-10000:]

validation_set.to_pickle('../../data/comp_validation.pkl')
test_set.to_pickle('../../data/comp_test.pkl')

#save the test set in the perfect setup
directory_path = '../../data/reliability'
test_set.loc[:,'worker_label'] = test_set['choice']
test_set.reset_index(drop=True, inplace=True)
worker_modeling.to_parquet(test_set, directory_path, "test", "perfect")