# Centralizing main imports so we can run the models separately
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
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
    def __init__(self, model, pad_token_id, device):
        super(VanillaWrappedModel, self).__init__()
        self.model = model.to(device)
        self.pad_token_id = pad_token_id
        self.device = device
        
    def forward(self, input_ids):
        # Moving (or guaranteeing) input_ids to proper device
        input_ids = input_ids.to(self.device)

        # Computing attention mask and moving it to device as well
        attention_mask = (input_ids != self.pad_token_id).to(self.device).long()

        # Extracting [CLS] token embedding and computing its L2 norm (for attributions target)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]
        cls_scalar = cls_embedding.norm(p=2, dim=1)
        
        return cls_scalar