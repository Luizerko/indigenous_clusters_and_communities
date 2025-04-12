# Centralizing main imports so we can run the models separately
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchmetrics.classification import Accuracy, Precision, Recall

import matplotlib.pyplot as plt

from transformers import AutoModel, AutoTokenizer
# from transformers import AutoModelForPreTraining
from captum.attr import LayerIntegratedGradients

# Cleaning up memory function
def clean_mem(tensors):
    for tensor in tensors:
        del tensor
    torch.cuda.empty_cache()

# Class for the TextDataset and to avoid loading everything simultaneously and to better interact with toch data pipelines
class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256, clf_col=None):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.clf_col = clf_col

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.df.iloc[idx]['descricao']
        
        # Getting input_ids from the text with the tokenizer
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = inputs['input_ids'].squeeze(0)
        
        # Creating labels if dataset is used for fine-tuning
        label = []
        if self.clf_col is not None:
            label = self.df.iloc[idx][self.clf_col]

        return idx, input_ids, label

# Function to get the dataloaders for a dataset
def get_dataloaders(dataset, batch_size=8, splits=None):
    # Splitting the data into train, validation and test datasets and then initializing their respective dataloader
    if splits is not None:
        train_dataset, val_dataset, test_dataset = random_split(dataset, splits)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return [train_dataset, val_dataset, test_dataset], [train_dataloader, val_dataloader, test_dataloader]

    # Returning the dataloader for the entire dataset
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Captum compatible wrapper model for the vanilla BERTimbau. Notice that we need to use a scalar representation for the [CLS] token so we can evaluate how ir varies with the other tokens (integrated gradients as the name suggests). In this case, we decided to go for the norm of the embedding vector
class VanillaWrappedModel(nn.Module):
    def __init__(self, model, pad_token_id, baseline_embedding, device, target_type='l2-norm'):
        super(VanillaWrappedModel, self).__init__()
        self.model = model.to(device)
        self.pad_token_id = pad_token_id
        self.baseline_embedding = baseline_embedding.detach().to(device)
        self.device = device
        self.target_type = target_type
        
    def forward(self, input_ids):
        # Moving (or guaranteeing) input_ids to proper device
        input_ids = input_ids.to(self.device)

        # Computing attention mask and moving it to device as well
        attention_mask = (input_ids != self.pad_token_id).to(self.device).long()

        # Extracting [CLS] token embedding and computing its L2 norm (for attributions target)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]
        
        if self.target_type == 'l2-norm':
            cls_scalar = cls_embedding.norm(p=2, dim=1)
        elif self.target_type == 'cos-sim':
            cls_scalar = F.cosine_similarity(cls_embedding, self.baseline_embedding, dim=1)
        
        return cls_scalar
    
# Function to compute [CLS] token embeddings
def get_embeddings(model, tokenizer, input_ids, device):
    # Computing attention mask
    attention_mask = (input_ids != tokenizer.pad_token_id).to(device).long()
    
    # And now the [CLS] token embeddings
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0]

    return cls_embeddings

# Function to compute attributions via integrated gradients torwards the [CLS] token
def get_attributions(lig, tokenizer, input_ids, baseline_input_ids, attrib_aggreg_type='sum', verbose=False, sample_num=0):
    # Computing attributions and then summing over the embedding dimensions (because we compute attributions for every dimension of the tokens' embeddings, so we need some kind of aggregation to idedntify tokens individually)
    attributions, delta = lig.attribute(inputs=input_ids, baselines=baseline_input_ids, return_convergence_delta=True, n_steps=50)
    if attrib_aggreg_type == 'sum':
        attributions = attributions.sum(dim=-1)
    elif attrib_aggreg_type == 'l2-norm':
        attributions = attributions.norm(p=2, dim=-1)
    attributions = attributions.cpu().detach()

    # Normalizing to output easier to interpret and because we are more interested in the absolute importance of tokens rather then their signal
    attributions = torch.abs(attributions)
    try:
        attributions = attributions/attributions.sum(axis=1).unsqueeze(1)
    except:
        print('Check for a zero division on one of the samples! Maybe an empty string?')

    # If verbose, decoding tokens and getting attributions for the [CLS] token for the first sample
    if verbose:
        sample_input_ids = input_ids[min(sample_num, len(input_ids)-1)]
        tokens = tokenizer.convert_ids_to_tokens(sample_input_ids.cpu().tolist())
        sample_attribution = attributions[min(sample_num, len(input_ids)-1)]

        new_tokens, new_scores = [], []
        counter = -1
        for i, (token, score) in enumerate(zip(tokens[1:-1], sample_attribution[1:-1])):
            if token == '[PAD]':
                new_tokens = new_tokens[:-1]
                break

            if token[0] == '#':
                new_tokens[counter] += token[2:]
                new_scores[counter] += score
            else:
                new_tokens.append(token)
                new_scores.append(score)
                counter += 1

        print("Token importances:")
        for token, score in zip(new_tokens, new_scores):
            print(f"{token:20} -> {score:.4f}")

    return attributions, delta