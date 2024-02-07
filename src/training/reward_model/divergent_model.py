# -*- coding: utf-8 -*-
# load datasets
from datasets import load_dataset
import pandas as pd
import deepspeed
from reward_model import GPTRewardModel
from gptj_reward_test import create_comparison_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import GPT2TokenizerFast
import torch
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import pdb
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Analysis")
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank passed from distributed launcher')
    parser.add_argument("--hub_path",
                        type=str,
                        default='/network/scratch/i/ines.arous/models-hub/',
                        help="path of the hub")
    # DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    return args

def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        for pair in pairs:
            chosen, rejected = pair["chosen"], pair["rejected"]
            chosen_encodings_dict = tokenizer(
                "<|startoftext|>" + chosen + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_encodings_dict = tokenizer(
                "<|startoftext|>" + rejected + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            if not torch.all(torch.eq(chosen_encodings_dict["input_ids"], rejected_encodings_dict["input_ids"])).item():
                self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
                self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
                self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
                self.rejected_attn_masks.append(rejected_encodings_dict["attention_mask"])

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return (
            self.chosen_input_ids[idx],
            self.chosen_attn_masks[idx],
            self.rejected_input_ids[idx],
            self.rejected_attn_masks[idx],
            idx,
        )


class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data])
        batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data])
        batch["labels"] = torch.tensor([0] * len(data) + [1] * len(data))
        batch["idx"] = torch.tensor([f[4] for f in data])
        return batch

set_seed(3)
args = parse_args()
# model = GPTRewardModel("CarperAI/openai_summarize_tldr_sft",args.hub_path)
model = GPTRewardModel("gpt2",args.hub_path)
data_path = '../../../data/uncertain_spl/'

max_length = 550
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
PAD_ID = tokenizer(tokenizer.pad_token)["input_ids"][0]

data_train_pqt = pd.read_parquet(data_path+'train'+'_uncertain_spl.parquet')
train_pairs = create_comparison_dataset(data_path, "train")
train_dataset = PairwiseDataset(train_pairs, tokenizer, max_length=max_length)
train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=6, collate_fn=DataCollatorReward())
device = torch.device("cpu")
model.to(device)
model.eval()
# model.half()
# Evaluate accuracy for the current checkpoint
correct = 0
chosen_list = []
reject_list = []
uncertain_instances = []
disagreement_instances = []
uncertain_pqt = pd.DataFrame(columns=['prompt','chosen','rejected'])
disagreement_pqt = pd.DataFrame(columns=['prompt','chosen','rejected'])
with torch.no_grad():
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

        for x in batch:
            batch[x] = batch[x].to(device)
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        correct += sum(outputs["chosen_end_scores"] > outputs["rejected_end_scores"])
        mask = torch.abs(outputs["chosen_end_scores"] - outputs["rejected_end_scores"]) < 2
        for idx in range(len(mask)):
            if mask[idx]:
                uncertain_pqt = pd.concat([uncertain_pqt,data_train_pqt.iloc[batch['idx'][idx].item(),:].to_frame().transpose()], ignore_index=True)
                uncertain_instances.extend([
                    {
                        'chosen': train_pairs[batch['idx'][idx].item()]['chosen'],  # Replace 'text' with the actual attribute in your dataset
                        'rejected': train_pairs[batch['idx'][idx].item()]['rejected'],
                        'chosen_score': outputs["chosen_end_scores"][idx].item(),
                        'rejected_score': outputs["rejected_end_scores"][idx].item()
                    }
                ])
        mask_disagreement = outputs["chosen_end_scores"] < outputs["rejected_end_scores"]
        for idx_dis in range(len(mask_disagreement)):
            if mask_disagreement[idx_dis]:
                disagreement_pqt = pd.concat([disagreement_pqt,data_train_pqt.iloc[batch['idx'][idx_dis].item(),:].to_frame().transpose()], ignore_index=True)
                disagreement_instances.extend([
                    {
                        'chosen': train_pairs[batch['idx'][idx_dis].item()]['chosen'],  # Replace 'text' with the actual attribute in your dataset
                        'rejected': train_pairs[batch['idx'][idx_dis].item()]['rejected'],
                        'chosen_score': outputs["chosen_end_scores"][idx_dis].item(),
                        'rejected_score': outputs["rejected_end_scores"][idx_dis].item()
                    }
                ])
        pdb.set_trace()
        chosen_list.append(outputs["chosen_end_scores"].cpu())
        reject_list.append(outputs["rejected_end_scores"].cpu())
    # uncertain_pqt.to_parquet('../../../data/reliability/uncertain_di_val.parquet',index=False)
    # pd.DataFrame(uncertain_instances).to_parquet('../../../data/reliability/uncertain_scores_di_val.parquet',index=False)
    # disagreement_pqt.to_parquet('../../../data/reliability/disagreement_val.parquet',index=False)
    # pd.DataFrame(disagreement_instances).to_parquet('../../../data/reliability/disagreement_pqt_scores_val.parquet',index=False)


train_disagreement = pd.read_parquet('../../data/disagreement_arxiv/train_disagreement.parquet')
valid_disagreement = pd.read_parquet('../../data/disagreement_arxiv/validation_disagreement.parquet')


train_uncertain = pd.read_parquet('../../data/uncertain/train_uncertain.parquet')
valid_uncertain = pd.read_parquet('../../data/uncertain/validation_uncertain.parquet')

train_disagreement_sc = pd.read_parquet('../../data/out_scores/disagreement_pqt_scores.parquet')
valid_disagreement_sc = pd.read_parquet('../../data/out_scores/disagreement_pqt_scores_val.parquet')

train_disagreement_pqt = map_pqtsc_pqt(train_disagreement_sc)
valid_disagreement_pqt = map_pqtsc_pqt(valid_disagreement_sc)
train_disagreement_pqt.to_parquet('../../data/disagreement/train_disagreement.parquet',index=False)
valid_disagreement_pqt.to_parquet('../../data/disagreement/validation_disagreement.parquet',index=False)

train_uncertain_sc = pd.read_parquet('../../data/out_scores/uncertain_scores_di.parquet')
valid_uncertain_sc = pd.read_parquet('../../data/out_scores/uncertain_scores_di_val.parquet')
train_uncertain_sc['diff'] = np.abs(train_uncertain_sc['chosen_score']-train_uncertain_sc['rejected_score'])
valid_uncertain_sc['diff'] = np.abs(valid_uncertain_sc['chosen_score']-valid_uncertain_sc['rejected_score'])
size_sample = int(train_comparisons.shape[0]*20/100)
size_val = int(valid_comparisons.shape[0]*20/100)
train_uncertain_sc_sorted = train_uncertain_sc.sort_values(by='diff').iloc[: size_sample]
valid_uncertain_sc_sorted = valid_uncertain_sc.sort_values(by='diff').iloc[: size_val]


uncertain_train = map_pqtsc_pqt(train_uncertain_sc_sorted)
uncertain_valid = map_pqtsc_pqt(valid_uncertain_sc_sorted)
uncertain_train.to_parquet('../../data/uncertain_spl/train_uncertain_spl.parquet',index=False)
uncertain_valid.to_parquet('../../data/uncertain_spl/validation_uncertain_spl.parquet',index=False)