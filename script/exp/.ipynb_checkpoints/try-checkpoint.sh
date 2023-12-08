#!/bin/bash

# Extract the best model checkpoint path from the JSON file
BEST_MODEL_CHECKPOINT=$(jq -r '.best_model_checkpoint' /network/scratch/z/zixuan.li/experiment_reward_model/medium/checkpoint-2000/training_state.json)

# Append the filename to get the full path to the checkpoint
CKPT_PATH="$BEST_MODEL_CHECKPOINT/pytorch_model.bin"

# Print or use the variable
echo $CKPT_PATH
