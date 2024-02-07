import pandas as pd
import numpy as np
from evaluate import load
from sentence_transformers import SentenceTransformer, models,util
import worker_modeling
from utils import create_directory
directory_path = '../../data/cluster'
create_directory(directory_path)
hub_path = '/network/scratch/i/ines.arous/models-hub/'
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2', cache_folder=hub_path)

bertscore = load("bertscore")

comp_train_df = pd.read_pickle('../../data/comp_train.pkl')
comp_val_df = pd.read_pickle('../../data/comp_val.pkl')

def df2sim(comp_train_df):
    columns_text = ['prompt', 'summary_text_0', 'summary_text_1']

    embeddings_prompt = model.encode(comp_train_df['prompt'].to_list())
    embeddings_sum_text_0 = model.encode(comp_train_df['summary_text_0'].to_list())
    embeddings_sum_text_1 = model.encode(comp_train_df['summary_text_1'].to_list())

    sim_prompt_text0,sim_prompt_text1 = [], []

    for idx in range(len(embeddings_prompt)):
        sim_prompt_text0.append(util.pytorch_cos_sim(embeddings_prompt[idx], embeddings_sum_text_0[idx]).item())
        sim_prompt_text1.append(util.pytorch_cos_sim(embeddings_prompt[idx], embeddings_sum_text_1[idx]).item())

    comp_train_df['sim_text0'] = sim_prompt_text0
    comp_train_df['sim_text1'] = sim_prompt_text1

    comp_train_df['sim_diff'] = np.abs(comp_train_df['sim_text0']-comp_train_df['sim_text1'])


    comp_train_df_sim = comp_train_df.sort_values(by='sim_diff')
    size_20 = int(comp_train_df.shape[0]*0.2)
    sim_sample = comp_train_df_sim.iloc[:size_20]
    return sim_sample

sim_sample = df2sim(comp_train_df)
sim_sample.loc[:, 'worker_label'] = sim_sample['choice']
sim_sample_pqt = worker_modeling.to_parquet(sim_sample, directory_path, "train", "cos_sim")
comp_val_df.loc[:,'prompt'] = comp_val_df['post']
sim_sample_val = df2sim(comp_val_df)
sim_sample_val.loc[:, 'worker_label'] = sim_sample_val['choice']
sim_val_pqt = worker_modeling.to_parquet(sim_sample, directory_path, "validation", "cos_sim")
# sum_0 = bertscore.compute(predictions=comp_train_df['summary_text_0'], references=comp_train_df['prompt'], lang="en")
# sum_1 = bertscore.compute(predictions=comp_train_df['summary_text_1'], references=comp_train_df['prompt'], lang="en")