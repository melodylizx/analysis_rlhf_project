import pdb

import pandas as pd
from datasets import load_dataset

def chosen_colum(row):
    if row['choice']==1:
        return row['summary_text_1']
    else:
        return row['summary_text_0']


#check generated reliability data
directory_path = '../../data/reliability/'
reliability_name = {"extreme":0,"low":0.2,"medium":0.5,"high":0.8, "perfect" : 1}
splits = ["train","validation"]
for split in splits:
    # load original sets
    comparisons = pd.read_pickle('../../data/comp_'+split+'.pkl')
    comparisons = comparisons.reset_index()
    comparisons['post'] = "POST: " + comparisons['post']
    comparisons['chosen'] = "TL;DR: " + comparisons.apply(chosen_colum, axis=1)
    comp_len = comparisons.shape[0]
    for title in reliability_name:
        generated_file = pd.read_parquet(directory_path+'/'+title+'/'+split + "_" + title + '.parquet')
        reliability = reliability_name[title]
        matches_nbr = generated_file[generated_file['chosen']==comparisons['chosen']].shape[0]
        perc_matching = round(matches_nbr/comp_len,1)
        assert perc_matching==reliability_name[title]




