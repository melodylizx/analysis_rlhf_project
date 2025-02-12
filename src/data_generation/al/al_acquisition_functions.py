# import torch
# from torch.utils.data import DataLoader
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
# from datasets import load_dataset
# import numpy as np
# from sklearn.metrics import f1_score
# import random
# import json
# from scipy.stats import entropy
# import numpy as np
# from batchbald_redux import batchbald
# import torch.nn.functional as F
#
#
# def mc_dropout_uncertainty(model, dataloader,DEVICE,N_PASSES):
#     model.train()  # Enable dropout for MC sampling
#     uncertainties = []
#
#     for batch in dataloader:
#         inputs = {key: val.to(DEVICE) for key, val in batch.items() if key in ['input_ids', 'attention_mask']}
#         predictions = []
#
#         for _ in range(N_PASSES):
#             with torch.no_grad():
#                 outputs = model(**inputs)
#                 probs = torch.softmax(outputs.logits, dim=-1)
#                 predictions.append(probs.cpu().numpy())
#
#         predictions = np.stack(predictions)  # Shape: (N_PASSES, batch_size, num_classes)
#         stds = np.std(predictions, axis=0)  # Std across passes
#         sample_uncertainties = stds.mean(axis=1)  # Mean over classes
#         uncertainties.extend(sample_uncertainties)
#
#     return np.array(uncertainties)
#
# def compute_bald_scores(model, dataloader, n_passes=10,DEVICE):
#     model.train()  # Enable dropout for MC sampling
#     bald_scores = []
#
#     for batch in dataloader:
#         inputs = {key: val.to(DEVICE) for key, val in batch.items() if key in ['input_ids', 'attention_mask']}
#         mc_probs = []
#
#         for _ in range(n_passes):
#             with torch.no_grad():
#                 logits = model(**inputs).logits
#                 probs = F.softmax(logits, dim=-1)
#                 mc_probs.append(probs.cpu().numpy())
#
#         mc_probs = np.stack(mc_probs)  # Shape: (n_passes, batch_size, num_classes)
#         mean_probs = mc_probs.mean(axis=0)  # Average over MC passes
#         entropy_mean = entropy(mean_probs.T)  # Entropy of mean prediction
#         mean_entropy = mc_probs.mean(axis=0).mean(axis=-1)  # Mean entropy across classes
#
#         bald_score = entropy_mean - mean_entropy  # BALD score
#         bald_scores.extend(bald_score)
#
#     return np.array(bald_scores)
#
#
# def compute_batchbald_scores(model, dataloader, n_passes=10,DEVICE):
#     model.train()  # Enable dropout for MC sampling
#     batchbald_scores = []
#
#     for batch in dataloader:
#         inputs = {key: val.to(DEVICE) for key, val in batch.items() if key in ['input_ids', 'attention_mask']}
#         mc_logits = []
#
#         for _ in range(n_passes):
#             with torch.no_grad():
#                 logits = model(**inputs).logits
#                 mc_logits.append(logits.cpu())
#
#         mc_logits = torch.stack(mc_logits)  # Shape: (n_passes, batch_size, num_classes)
#
#         # Compute BatchBALD scores
#         batch_scores = batchbald.get_batchbald_batch(mc_logits, num_samples=n_passes)
#         batchbald_scores.extend(batch_scores)
#
#     return np.array(batchbald_scores)