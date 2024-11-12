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
confusing_train = comp_train_df_dup[(comp_train_df_dup['conf'] > 0) & (comp_train_df_dup['conf'] < 5)]
confident_train =  comp_train_df_dup[(comp_train_df_dup['conf'] > 0) & (comp_train_df_dup['conf'] > 5)]

#keep 10k of pairs for testing
validation_set = comp_val_df_dup[:-10000]
test_set = comp_val_df_dup[-10000:]

confusing_val = validation_set[(validation_set['conf'] > 0) & (validation_set['conf'] < 5)]
confident_val =  validation_set[(validation_set['conf'] > 0) & (validation_set['conf'] > 5)]


validation_set.to_pickle('../../data/comp_validation.pkl')
test_set.to_pickle('../../data/comp_test.pkl')

#save the test set in the perfect setup
directory_path = '../../data/reliability'
test_set.loc[:,'worker_label'] = test_set['choice']
test_set.reset_index(drop=True, inplace=True)
worker_modeling.to_parquet(test_set, directory_path, "test", "100")

#save the confusing cases
#save the training set
def save_conf_parquet(df,original_df,split,exp, path):
    df.loc[:, 'worker_label'] = df['choice']
    df.reset_index(drop=True, inplace=True)
    sample_size = int(len(original_df) * 0.2)
    df_sample = df.sample(n=sample_size, random_state=42)
    worker_modeling.to_parquet(df_sample, path, split, exp)
    return df_sample


low_conf_path = '../../data/'
confusing_train_sample = save_conf_parquet(confusing_train,comp_train_df_dup,"train","low_conf", low_conf_path)
confusing_val_sample = save_conf_parquet(confusing_val,validation_set,"validation","low_conf", low_conf_path)

#high confidence
high_conf_path = '../../data/'
confident_train_sample = save_conf_parquet(confident_train,comp_train_df_dup,"train","high_conf", high_conf_path)
confident_val_sample = save_conf_parquet(confident_val,validation_set,"validation","high_conf", high_conf_path)

