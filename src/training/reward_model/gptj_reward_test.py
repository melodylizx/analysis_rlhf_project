import random
import os
import json
import csv
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from reward_model import GPTRewardModel
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse
import pdb

def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


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
        )


class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data])
        batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data])
        batch["labels"] = torch.tensor([0] * len(data) + [1] * len(data))
        return batch

def find_largest_checkpoint(ckpt_path):
    checkpoints = [entry for entry in os.listdir(ckpt_path) if entry.startswith('checkpoint')]
    largest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
    return os.path.join(ckpt_path, largest_checkpoint, 'trainer_state.json')

def get_best_checkpoint(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    return data.get('best_model_checkpoint', None)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='reward model checkpoint')
    parser.add_argument('--ckpt_path', type=str, help='Path to the reward model.')
    parser.add_argument("--hub_path",
                        type=str,
                        default='/network/scratch/i/ines.arous/models-hub/',
                        help="path of the checkpoint")
    args = parser.parse_args()
    largest_checkpoint_path = find_largest_checkpoint(args.ckpt_path)
    if largest_checkpoint_path:
        best_checkpoint = get_best_checkpoint(largest_checkpoint_path)
        print("Best Checkpoint:", best_checkpoint)
    else:
        print("No valid checkpoints found.")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.pad_token = tokenizer.eos_token
    PAD_ID = tokenizer(tokenizer.pad_token)["input_ids"][0]
    print("loading the model")
    model = GPTRewardModel("CarperAI/openai_summarize_tldr_sft",args.hub_path)
    max_length = 550
    data_path = '/home/mila/i/ines.arous/rlhf/data/reliability/perfect'
    test_pairs = create_comparison_dataset(data_path, "test")
    test_dataset = PairwiseDataset(test_pairs, tokenizer, max_length=max_length)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=6, collate_fn=DataCollatorReward())
    # Initialize a list to store accuracy for each checkpoint
    accuracy_records = []
    best_path = ""
    best_accuracy = 0

    # Iterate through all checkpoints
    for checkpoint_name in os.listdir(args.ckpt_path):
        print(checkpoint_name)
        checkpoint_path = os.path.join(args.ckpt_path, checkpoint_name)
        if os.path.isdir(checkpoint_path):
            model_state_path = os.path.join(checkpoint_path, 'pytorch_model.bin')

            # Load the model state dictionary from pytorch_model.bin
            model.load_state_dict(torch.load(model_state_path))
            model = model.cuda().half().eval()

            # Evaluate accuracy for the current checkpoint
            correct = 0
            with torch.no_grad():
                for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                    for x in batch:
                        batch[x] = batch[x].cuda()
                    outputs = model(**batch)
                    correct += sum(outputs["chosen_end_scores"] > outputs["rejected_end_scores"])
            # Calculate and store accuracy for the current checkpoint
            accuracy = correct / len(test_dataset)
            if accuracy> best_accuracy:
                best_accuracy = accuracy
                best_path = checkpoint_path+'/pytorch_model.bin'
            accuracy_records.append({"Checkpoint": checkpoint_name, "Accuracy": accuracy, "correct": correct, "wrong": len(test_dataset)-correct})

    # Save accuracy records to a CSV file
    csv_file_path = os.path.join(args.ckpt_path, "accuracy_records.csv")
    fields = ["Checkpoint", "Accuracy", "correct", "wrong"]
    print("writing in csv...")
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)

        # Write the header
        writer.writeheader()

        # Write accuracy records
        writer.writerows(accuracy_records)

    print(best_path)








