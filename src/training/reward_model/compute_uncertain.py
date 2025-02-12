import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse
import deepspeed
import sys
from src.data_generation.uncertainty import UncertaintySampling

import logging

logger = logging.getLogger(__name__)


sys.path.insert(0, '../')
from trlx.utils import set_seed
from data_generation.utils import create_directory
from reward_model import GPTRewardModel

# Configuration for DeepSpeed
ds_config = {
    "fp16": {
        "enabled": True,
        "min_loss_scale": 1,
        "opt_level": "O2"
    },

    "zero_optimization": {
        "stage": 2,
        "offload_param": {
            "device": "cpu",
        },
        "offload_optimizer": {
            "device": "cpu"
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "contiguous_gradients": True
    },
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False
}


def create_comparison_dataset(path="CarperAI/openai_summarize_comparisons", split="train"):
    dataset = load_dataset(path, split=split)
    # if split == "test":
    #     dataset = dataset.select(range(5000))

    pairs = []
    for sample in tqdm(dataset):
        pair = {}
        prompt = sample["prompt"]
        chosen_summary = sample["chosen"]
        rejected_summary = sample["rejected"]
        if chosen_summary == rejected_summary:
            continue
        if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
            continue
        pair["chosen"] = prompt + "\n" + chosen_summary
        pair["rejected"] = prompt + "\n" + rejected_summary
        pairs.append(pair)
    return pairs


class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        self.chosen = []
        self.rejected = []
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
                self.chosen.append(chosen)
                self.rejected.append(rejected)

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return (
            self.chosen_input_ids[idx],
            self.chosen_attn_masks[idx],
            self.rejected_input_ids[idx],
            self.rejected_attn_masks[idx],
            self.chosen[idx],
            self.rejected[idx],
            idx
        )


class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data])
        batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data])
        batch["labels"] = torch.tensor([0] * len(data) + [1] * len(data))
        batch["chosen"] = [f[4] for f in data]
        batch["rejected"] = [f[5] for f in data]
        batch["idx"] = [f[6] for f in data]
        return batch


def parse_args():
    parser = argparse.ArgumentParser(description="Analysis")

    # Existing arguments
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank passed from distributed launcher')
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed")
    parser.add_argument("--hub_path",
                        type=str,
                        default='/network/scratch/i/ines.arous/models-hub/',
                        help="path of the checkpoint")

    # DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    uncertainty_sampler = UncertaintySampling()
    measure_uncertainty_method = lambda x: uncertainty_sampler.margin_confidence(uncertainty_sampler.softmax(x))

    logger.info("########### loading the tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=args.hub_path)
    tokenizer.pad_token = tokenizer.eos_token
    PAD_ID = tokenizer(tokenizer.pad_token)["input_ids"][0]

    logger.info("########### loading the model")
    model = GPTRewardModel("CarperAI/openai_summarize_tldr_sft", args.hub_path)

    logger.info("########### initialize deepspeed")
    deepspeed.init_distributed()
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        config_params='../training/reward_model/ds_config_gpt_j.json'
    )

    logger.info("########### loading the dataset")
    max_length = 550
    all_data_path ='/home/mila/i/ines.arous/rlhf_reproduce/data'
    data_path = all_data_path+'/reliability/100/'
    train_pairs = create_comparison_dataset(data_path, "train")
    train_dataset = PairwiseDataset(train_pairs, tokenizer, max_length=max_length)
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=32, collate_fn=DataCollatorReward())

    val_pairs = create_comparison_dataset(data_path, "validation")
    val_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=max_length)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=32, collate_fn=DataCollatorReward())

    logger.info("done loading the dataset")
    # Divide checkpoints among GPUs
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    logger.info("doing eval")
    model_engine.module.eval()
    # Evaluate accuracy for the current checkpoint
    correct = 0
    logger.info("computing accuracy")
    train_ids_with_uncertainty_score = []
    val_ids_with_uncertainty_score = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            ids = batch.pop('idx')
            chosen = batch.pop('chosen')
            rejected = batch.pop('rejected')
            batch = {key: value.cuda() for key, value in batch.items()}
            outputs = model(**batch)
            for idx, chosen_score, rejected_score in zip(ids, outputs["chosen_end_scores"],
                                                         outputs["rejected_end_scores"]):
                uncertainty_score = measure_uncertainty_method(torch.tensor([chosen_score, rejected_score]))
                train_ids_with_uncertainty_score.append([idx, uncertainty_score])
    train_ids_with_uncertainty_score.sort(key=lambda x: x[1], reverse=True)
    top_20_percent_train = int(len(train_ids_with_uncertainty_score) * 0.2)
    top_uncertain_train_ids = [idx for idx, _ in train_ids_with_uncertainty_score[:top_20_percent_train]]
    train_output_df = load_dataset(data_path, split="train")
    train_output_df = train_output_df.select(top_uncertain_train_ids)
    directory_path = all_data_path+"/uncertainty/margin_confidence/"
    create_directory(directory_path)
    # worker_modeling.to_parquet(train_output_df.to_pandas(), directory_path + "/", "train","margin_confidence")
    train_output_df.to_pandas().to_parquet(directory_path + "/uncertainty/margin_confidence/" + '/' + "train" + "_" + "margin_confidence" + '.parquet', index=False)

    with torch.no_grad():
        for step, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            ids = batch.pop('idx')
            chosen = batch.pop('chosen')
            rejected = batch.pop('rejected')
            batch = {key: value.cuda() for key, value in batch.items()}
            outputs = model(**batch)
            for idx, chosen_score, rejected_score in zip(ids, outputs["chosen_end_scores"],
                                                         outputs["rejected_end_scores"]):
                uncertainty_score = measure_uncertainty_method(torch.tensor([chosen_score, rejected_score]))
                val_ids_with_uncertainty_score.append([idx, uncertainty_score])

    # sort ids by decreasing uncertainty
    val_ids_with_uncertainty_score.sort(key=lambda x: x[1], reverse=True)

    # select top 20%
    top_20_percent_val = int(len(val_ids_with_uncertainty_score) * 0.2)
    top_uncertain_val_ids = [idx for idx, _ in val_ids_with_uncertainty_score[:top_20_percent_val]]

    val_output_df = load_dataset(data_path, split="validation")

    # filter out ids which are not in the top uncertain ids
    val_output_df = val_output_df.select(top_uncertain_val_ids)

    # save modified datasets to parquets
    val_output_df.to_pandas().to_parquet(
        directory_path + "/uncertainty/margin_confidence/" + "validation" + "_" + "margin_confidence" + '.parquet',
        index=False)
    # worker_modeling.to_parquet(val_output_df, directory_path + "/", "validation","margin_confidence")