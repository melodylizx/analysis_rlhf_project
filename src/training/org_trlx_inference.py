import os

import evaluate
import pandas as pd
import torch
from datasets import load_dataset
from reward_model.reward_model import GPTRewardModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trlx.models.modeling_ppo import AutoModelForCausalLMWithHydraValueHead

import argparse
import numpy as np

def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    #tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(path)
    model.config.pad_token_id = tokenizer.bos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.bos_token_id
    tokenizer.padding_side = "left"
    return model, tokenizer


#REWARD_CHECKPOINT_PATH = "/network/scratch/z/zixuan.li/experiment_reward_model/coverage/checkpoint-120/pytorch_model.bin"

#if not os.path.exists(REWARD_CHECKPOINT_PATH):
    #os.makedirs("/network/scratch/z/zixuan.li/reward_model/rm_checkpoint", exist_ok=True)
    #os.system(
        #f"wget -O {REWARD_CHECKPOINT_PATH} \
        #https://huggingface.co/CarperAI/openai_summarize_tldr_rm_checkpoint/resolve/main/pytorch_model.bin"
    #)
parser = argparse.ArgumentParser(description='reward model checkpoint')
parser.add_argument('--ckpt_path', type=str, help='Path to the reward model.')
parser.add_argument('--save_path', type=str, help='Path to the save ppo model.')
parser.add_argument('--csv_path', type=str, help='Path to the csv eval.')
args = parser.parse_args()
    
random_seed = 3 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

rw_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
#rw_tokenizer = AutoTokenizer.from_pretrained("gpt2")
rw_tokenizer.pad_token = rw_tokenizer.eos_token
#rw_model =GPTRewardModel("/home/mila/z/zixuan.li/trlx/examples/summarize_rlhf/ckpts/checkpoint_320")
rw_model = GPTRewardModel(args.save_path)
rw_model.load_state_dict(torch.load(args.ckpt_path))
rw_model.half()
rw_model.eval()
rw_device = torch.device("cuda:{}".format(1))
rw_model.to(rw_device)


def reward_fn(samples):
    scores_list = []
    batch_size = 2
    for i in range(0, len(samples), batch_size):
        sub_samples = samples[i : i + batch_size]
        sub_samples = ["<|startoftext|>" + chosen + "<|endoftext|>" for chosen in sub_samples]
        encodings_dict = rw_tokenizer(
            sub_samples,
            truncation=True,
            max_length=550,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encodings_dict["input_ids"].to(rw_device)
        attn_masks = encodings_dict["attention_mask"].to(rw_device)
        input_ids = input_ids.repeat(2, 1)
        attn_masks = attn_masks.repeat(2, 1)
        with torch.no_grad():
            sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
        scores_list.append(sub_scores["chosen_end_scores"])
    scores = torch.cat(scores_list, dim=0)
    return scores


def inference(model, tokenizer):
    model.to("cuda")
    model.eval()

    pred_list = []
    summarize_list = []
    post_list = []
    rouge = evaluate.load("rouge")
    count = 0
    for post, summarize in tqdm(zip(test_post_list, test_summ_list), total=len(test_post_list)):
        encode_dict = tokenizer(post, return_tensors="pt", padding=False, truncation=True)
        txt_tokens = encode_dict["input_ids"].cuda()
        attention_mask = encode_dict["attention_mask"].cuda()
        kwargs = {"max_new_tokens": 50, "eos_token_id": 50256, "pad_token_id": 50256}
        summ_tokens = model.generate(txt_tokens, attention_mask=attention_mask, **kwargs)
        pred = tokenizer.batch_decode(summ_tokens)[0]
        pred = pred.split("TL;DR:")[1].replace("<|endoftext|>", "")
        pred_list.append(pred)
        summarize_list.append(summarize)
        post_list.append(post)
        if count % 10 == 0:
            result = rouge.compute(predictions=pred_list, references=summarize_list)
            print(result)
        count += 1
    df = pd.DataFrame.from_dict({"pred": pred_list, "truth": summarize_list, "post": post_list})
    result = rouge.compute(predictions=pred_list, references=summarize_list)
    print(result)
    return df


def inference_batches(model, tokenizer, test_post_list, test_summ_list, batch_size=32):
    model.to("cuda")
    model.eval()

    pred_list = []
    summarize_list = []
    post_list = []
    rouge = evaluate.load("rouge")

    # Iterate over the input data in mini-batches
    for i in tqdm(range(0, len(test_post_list), batch_size)):
        batch_post_list = test_post_list[i : i + batch_size]
        batch_summ_list = test_summ_list[i : i + batch_size]

        # Convert input data to tensors
        encode_dict = tokenizer(
            batch_post_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        txt_tokens = encode_dict["input_ids"].cuda()
        attention_mask = encode_dict["attention_mask"].cuda()

        # Perform inference on the batch
        kwargs = {"max_new_tokens": 50, "eos_token_id": 50256, "pad_token_id": 50256}
        summ_tokens = model.generate(txt_tokens, attention_mask=attention_mask, **kwargs)

        # Decode output tokens
        preds = tokenizer.batch_decode(summ_tokens)

        # Add predictions, truths, and input posts to lists
        pred_list += preds
        summarize_list += batch_summ_list
        post_list += batch_post_list

        # Compute rouge scores every 10 mini-batches
        result = rouge.compute(predictions=pred_list, references=summarize_list)
        print(result)

    # Compute final rouge scores and create a dataframe
    result = rouge.compute(predictions=pred_list, references=summarize_list)
    print(result)
    df = pd.DataFrame.from_dict({"pred": pred_list, "truth": summarize_list, "post": post_list, "rouge": result})
    return df


if __name__ == "__main__":
    
    
    #model, tokenizer = load_model("/home/mila/z/zixuan.li/trlx/examples/summarize_rlhf/ckpts/checkpoint_320/pytorch_model/mp_rank_00_model_states.pt")
    model, tokenizer = load_model(args.save_path)

    test_post_list = [sample["prompt"] for sample in load_dataset("CarperAI/openai_summarize_tldr", split="test")]
    test_summ_list = [sample["label"] for sample in load_dataset("CarperAI/openai_summarize_tldr", split="test")]

    df_result = inference(model, tokenizer)
    sup_pred = df_result["pred"].values
    truth = df_result["truth"].values

    scores_pred = []
    scores_truth = []
    preds_list = []
    truth_list = []
    post_list = []
    batch_size = 32
    for i in range(0, len(df_result), batch_size):
        predicts = df_result["pred"].values[i : i + batch_size]
        labels = df_result["truth"].values[i : i + batch_size]
        posts = df_result["post"].values[i : i + batch_size]
        data_pred = [posts[i] + predicts[i] for i in range(len(predicts))]
        data_truth = [posts[i] + labels[i] for i in range(len(labels))]
        preds_list.extend(list(predicts))
        truth_list.extend(list(labels))
        post_list.extend(list(posts))
        scores_pred.extend(list(reward_fn(data_pred).cpu().numpy()))
        scores_truth.extend(list(reward_fn(data_truth).cpu().numpy()))

    df = pd.DataFrame.from_dict(
        {
            "pred": preds_list,
            "truth": truth_list,
            "post": post_list,
            "score_pred": scores_pred,
            "score_truth": scores_truth,
        }
    )
    #df.to_csv("/network/scratch/z/zixuan.li/result_of_experiments/coverage/ppo_with_reward_scores.csv", index=False)
    df.to_csv(args.csv_path, index=False)
    print("Reward score pred: ", df.score_pred.values.mean())
    print("Reward score truth: ", df.score_truth.values.mean())
