import os
import deepspeed
import torch
from datasets import load_dataset
from reward_model import GPTRewardModel
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments
import argparse
import pdb
import sys
sys.path.insert(0, '..')

from trlx.utils import set_seed
def parse_args():
    parser = argparse.ArgumentParser(description="Analysis")

    # Existing arguments
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank passed from distributed launcher')
    parser.add_argument("--chpt_path", type=str, required=True,
                        help="path of the checkpoint")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed")
    parser.add_argument("--hub_path",
                        type=str,
                        default='/network/scratch/i/ines.arous/models-hub/',
                        help="path of the checkpoint")

    # New argument for dataset path
    parser.add_argument("--data_path", type=str, required=True,
                        help="path to load the dataset")

    # DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    return args


def create_comparison_dataset(path="CarperAI/openai_summarize_comparisons", split="train"):
    dataset = load_dataset(path, split=split)
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
        for pair in tqdm(pairs):
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
        )


class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data])
        batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data])
        batch["labels"] = torch.tensor([0] * len(data) + [1] * len(data))
        return batch


def compute_metrics(eval_preds):
    chosen_end_scores = eval_preds.predictions[0]  # chosen scores
    rejected_end_scores = eval_preds.predictions[1]  # rejected scores

    result = {}
    acc = sum(chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
    result["accuracy"] = acc
    print("accuracy",acc)
    return result

def find_largest_checkpoint(ckpt_path):
    checkpoints = [entry  for entry in os.listdir(ckpt_path) if entry.startswith('checkpoint')]
    if len(checkpoints) == 0:
        return False
    else:
        largest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
        return os.path.join(ckpt_path, largest_checkpoint)

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    last_checkpoint= find_largest_checkpoint(args.chpt_path)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=args.hub_path)
    tokenizer.pad_token = tokenizer.eos_token
    if not os.path.exists(args.chpt_path):
        os.mkdir(args.chpt_path)
    training_args = TrainingArguments(
        output_dir=args.chpt_path,
        num_train_epochs=1,
        logging_steps=10,
        gradient_accumulation_steps=4,
        save_strategy="steps",
        evaluation_strategy="steps",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_accumulation_steps=1,
        eval_steps=100,
        save_steps=100,
        warmup_steps=100,
        logging_dir="./logs",
        run_name='_'.join(args.chpt_path.rsplit('/', 2)[1:]),
        fp16=True,
        bf16=False,
        learning_rate=1e-5,
        deepspeed='./reward_model/ds_config_gpt_j.json',
        save_total_limit=5,
        load_best_model_at_end=True
    )
    # Initialize the reward model from the (supervised) fine-tuned GPT-J
    # double check the model to use

    model = GPTRewardModel("CarperAI/openai_summarize_tldr_sft",args.hub_path)
    # Freeze the first 70% of the hidden layers of the reward model backbone
    layers = model.transformer.h
    num_layers = len(layers)
    num_unfrozen = int(0.3 * num_layers)
    for layer in layers[:-num_unfrozen]:
        layer.requires_grad_(False)

    # Create the comparisons datasets
    data_path = args.data_path
    train_pairs = create_comparison_dataset(data_path, "train")
    val_pairs = create_comparison_dataset(data_path, "validation")

    # Make pairwise datasets for training
    max_length = 550
    train_dataset = PairwiseDataset(train_pairs, tokenizer, max_length=max_length)
    val_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=max_length)

    # Create the collator to gather batches of pairwise comparisons
    data_collator = DataCollatorReward()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    if last_checkpoint==False:
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=last_checkpoint)