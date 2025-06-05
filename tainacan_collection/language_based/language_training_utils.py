# Centralizing main imports so we can run the models separately
import os
import random
from tqdm import tqdm

import numpy as np
from scipy.stats import pearsonr
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset

import trimap
from sklearn.manifold import TSNE
import umap

import matplotlib.pyplot as plt

# Cleaning up memory function
def clean_mem(tensors):
    for tensor in tensors:
        del tensor
    torch.cuda.empty_cache()

# Function for normalizing projections to [-norm_factor, norm_factor] while maintaining relative distances
def normalize(data, norm_factor=2):
    mean = np.mean(data)
    max_dev = np.max(np.abs(data-mean))
    
    return norm_factor*(data-mean)/max_dev

# Class for the text dataset for the unsupervised fine-tuning, to avoid loading everything simultaneously and to better interact with torch data pipelines
class UnsupervisedTextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.df.iloc[idx]['descricao_resumida']
        
        # Getting input_ids from the text with the tokenizer
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = inputs['input_ids'].squeeze(0)
        
        return self.df.iloc[idx].name, input_ids
    
# Class for the text dataset for the supervised fine-tuning, to avoid loading everything simultaneously and to better interact with torch data pipelines
class SupervisedTextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Getting pieces of text needed for contrastive training
        anchor_text = row['descricao_resumida']
        positive_text = row['positive_contrastive']
        negatives_list = row['multi_negative_contrastive']
        
        # Getting input_ids from all the pieces of text with the tokenizer
        anchor_inputs = self.tokenizer(anchor_text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        anchor_input_ids = anchor_inputs['input_ids'].squeeze(0)
        
        pos_inputs = self.tokenizer(positive_text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        pos_input_ids = pos_inputs['input_ids'].squeeze(0)
        
        neg_input_ids_list = []
        for neg_text in negatives_list:
            neg_inputs = self.tokenizer(neg_text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
            neg_input_ids_list.append(neg_inputs['input_ids'].squeeze(0))
            
        # Stacking negatives to get a tensor of size (num_negatives, max_length)
        neg_input_ids = torch.stack(neg_input_ids_list, dim=0)
        
        return self.df.iloc[idx].name, anchor_input_ids, pos_input_ids, neg_input_ids

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

# Captum compatible wrapper model for the vanilla models. Notice that we need to use a scalar representation for the [CLS] token so we can evaluate how ir varies with the other tokens (integrated gradients as the name suggests). In this case, we decided to go for the norm of the embedding vector
class CaptumWrappedModel(nn.Module):
    def __init__(self, model, pad_token_id, baseline_embedding, device, target_type='l2-norm'):
        super(CaptumWrappedModel, self).__init__()
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
        # cls_embedding = outputs.last_hidden_state[:, 0]
        last_embeddings = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embedding = (last_embeddings*attention_mask_expanded).sum(dim=1)
        length = attention_mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_embedding = sum_embedding/length
        
        if self.target_type == 'l2-norm':
            cls_scalar = mean_embedding.norm(p=2, dim=1)
        elif self.target_type == 'cos-sim':
            cls_scalar = F.cosine_similarity(mean_embedding, self.baseline_embedding, dim=1)
        
        return cls_scalar
    
# Function to compute [CLS] token embeddings
def get_embeddings(model, tokenizer, input_ids, device, fine_tuned=False):
    # Computing attention mask
    attention_mask = (input_ids != tokenizer.pad_token_id).to(device).long()
    
    # And now the mean pooling embeddings
    with torch.no_grad():
        model.eval()
        if not fine_tuned:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
        # cls_embeddings = outputs.last_hidden_state[:, 0]
        last_embeddings = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embedding = (last_embeddings*attention_mask_expanded).sum(dim=1)
        length = attention_mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_embedding = sum_embedding/length

    return mean_embedding

# Function to compute attributions via integrated gradients torwards the [CLS] token
def get_attributions(lig, tokenizer, input_ids, baseline_input_ids, attrib_aggreg_type='sum', return_tokens=True, verbose=False, sample_num=0):
    # Computing attributions and then summing over the embedding dimensions (because we compute attributions for every dimension of the tokens' embeddings, so we need some kind of aggregation to identify tokens individually)
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
    if return_tokens:
        sample_input_ids = input_ids[min(sample_num, len(input_ids)-1)]
        tokens = tokenizer.convert_ids_to_tokens(sample_input_ids.cpu().tolist())
        sample_attribution = attributions[min(sample_num, len(input_ids)-1)]

        new_tokens, new_attributions = [], []
        counter = -1
        for token, attribution in zip(tokens[1:-1], sample_attribution[1:-1]):
            if token == '[PAD]':
                new_tokens = new_tokens[:-1]
                break

            if token[0] == '#':
                new_tokens[counter] += token[2:]
                new_attributions[counter] += attribution
            else:
                new_tokens.append(token)
                new_attributions.append(attribution)
                counter += 1

        if verbose:
            print("Token importances:")
            for token, attribution in zip(new_tokens, new_attributions):
                print(f"{token:20} -> {attribution:.4f}")

        return tokens, attributions, delta

    return attributions, delta

# Function for computing data projection
def data_projections(image_embeddings, **kwargs):
    proj_trimap = trimap.TRIMAP(n_dims=2, n_inliers=12, n_outliers=6, n_random=3,\
                                weight_temp=0.5, lr=0.1, apply_pca=True)
    vanilla_vit_trimap = proj_trimap.fit_transform(image_embeddings)
    
    proj_tsne = TSNE(n_components=2, perplexity=5, learning_rate='auto', init='random')
    vanilla_vit_tsne = proj_tsne.fit_transform(image_embeddings)
    
    proj_umap = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation')
    vanilla_vit_umap = proj_umap.fit_transform(image_embeddings)

    return vanilla_vit_trimap, vanilla_vit_tsne, vanilla_vit_umap

# Class for the unsupervised  SimCSE models. The idea is to fine-tune different models using the SimCSE approach with NT-Xent loss, which is an unsupervised contrastive loss, good for specializing the embedding model to our data without having to group it by categories
class USimCSEModel(nn.Module):
    def __init__(self, model, pad_token_id, device, dropout_prob=0.1):
        super(USimCSEModel, self).__init__()
        self.model = model.to(device)
        self.dropout = nn.Dropout(dropout_prob)
        
        self.device = device
        self.pad_token_id = pad_token_id
        
    def forward(self, input_ids):
        # Moving (or guaranteeing) input_ids to proper device
        input_ids = input_ids.to(self.device)

        # Computing attention mask and moving it to device as well
        attention_mask = (input_ids != self.pad_token_id).to(self.device).long()

        # Extracting mean pooling embedding and passing it through dropout to late compute NT-Xent loss
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # cls_embedding = outputs.last_hidden_state[:, 0]
        last_embeddings = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embedding = (last_embeddings*attention_mask_expanded).sum(dim=1)
        length = attention_mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_embedding = sum_embedding/length
        mean_embedding = self.dropout(mean_embedding)
        
        return mean_embedding
    
# Class for the (supervised)  InfoNCE models. The idea is to fine-tune different models using the contrastive InfoNCE approach with InfoNCE loss, which is a supervised contrastive loss based on our distribution of positive against all negatives, good for specializing the embedding model to our data without having to group it by categories
class InfoNCEModel(nn.Module):
    def __init__(self, model, pad_token_id, device, hidden_dim=768, proj_dim=512):
        super(InfoNCEModel, self).__init__()
        self.model = model.to(device)
        self.pad_token_id = pad_token_id
        self.device = device

        # Getting encoder hiddem_dim
        encoder_dim = model.config['hidden_size']

        # Simple MLP for fine-tuning
        self.mlp_head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.Relu(),
            nn.Linear(hidden_dim, proj_dim)
        )
        
    def forward(self, input_ids):
        # Moving (or guaranteeing) input_ids to proper device
        input_ids = input_ids.to(self.device)

        # Computing attention mask and moving it to device as well
        attention_mask = (input_ids != self.pad_token_id).to(self.device).long()

        # Mean-pooling embedding of tokens (weighted by attention mask) and passing it through the MLP head
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        last_embeddings = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embedding = (last_embeddings*attention_mask_expanded).sum(dim=1)
        length = attention_mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_embedding = sum_embedding/length
        
        projected = self.mlp_head(mean_embedding)
        
        return projected

# Computing the NT-Xent loss. The embeddings in this case should have shape [2*batch_size, dim], because we want two different views (different masked dimensions) of each sample to use as positive pairs
def nt_xent_loss(embeddings, device, temperature=0.05):
    # Computing cosine similarity in a more optimized way
    normalized_embeddings = F.normalize(embeddings, dim=1)
    sim_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.T).to(device)
    
    # Masking self-similarity
    sim_matrix.fill_diagonal_(-1e9)

    # Computing "logits" for cross-entropy
    logits = sim_matrix/temperature

    # Creating labels (positive pair mapping). Each embedding i matches with i+N if i<N, or i-N if i>=N
    batch_size = embeddings.shape[0]//2
    labels = torch.arange(2*batch_size).to(device)
    positive_indices = (labels+batch_size)%(2*batch_size)
    positive_indices = positive_indices.long()

    # Finally computing loss
    loss = F.cross_entropy(logits, positive_indices)
    return loss

# Training loop for the unsupervised SimCSE
def usimcse_training_loop(model, tokenizer, optimizer, train_dataloader, val_dataloader, test_dataset, df, device, epochs=10, temperature=0.05, patience=3, model_name='usimcse_bertimbau'):
    # Getting test indices to run in-context STS-B later
    test_indices = test_dataset.df.index.tolist()
    test_df = df.loc[test_indices, ['descricao_resumida', 'positive_contrastive', 'single_negative_contrastive']]
    
    # Varibales for saving the best model and early-stopping
    best_val_loss = float('inf')
    patience_counter = 0

    # Saving indicies and embeddings for later usage
    all_indices = []
    all_embeddings = []

    # Tracking loss and STS-Bs
    train_losses = []
    val_losses = []
    stsb_track = []
    in_context_stsb_track = []

    # Effective training loop
    for epoch in tqdm(range(epochs), desc='Epoch'):
        model.train()
        epoch_train_loss = 0

        for indices, input_ids in train_dataloader:
            # Saving indices
            all_indices.append(indices)

            # Moving appropriate tensors to device
            input_ids = input_ids.to(device)
            
            # Duplicating batch for SimCSE
            input_ids = torch.cat([input_ids, input_ids], dim=0)
            
            # Computing embeddings, saving them and computing loss
            optimizer.zero_grad()
            embeddings = model(input_ids)
            all_embeddings.append(embeddings.cpu().detach())

            loss = nt_xent_loss(embeddings, device, temperature)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss/len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation loss computation for early stopping and hyperparameter tunning. Also we need to turn on the lat dropout or else our validation loss becomes very small on our contrastive context since it depends on the dropout noise to create different embeddings from the same input
        model.eval()
        model.dropout.train()
        model.zero_grad()
        epoch_val_loss = 0
        with torch.no_grad():
            for indices, input_ids in val_dataloader:
                # Repeating process for validation dataset
                input_ids = input_ids.to(device)
                input_ids = torch.cat([input_ids, input_ids], dim=0)
                embeddings = model(input_ids)
                epoch_val_loss += nt_xent_loss(embeddings, device, temperature).item()
        
        avg_val_loss = epoch_val_loss/len(val_dataloader)
        val_losses.append(avg_val_loss)

        # Tracking STS-B score as a validation method as well
        stsb_track.append(stsb_test(model, device, tokenizer, max_length=input_ids.size(1), model_loss='usimcse', verbose=False))

        # Tracking in-context STS-B score as the validation method 
        in_context_stsb_track.append(in_context_stsb_test(model, device, tokenizer, test_df, max_length=input_ids.size(1), model_loss='usimcse', verbose=False))

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | STS-B: {stsb_track[-1]:.4f} | in-context STS-B: {in_context_stsb_track[-1]:.4f}")

        # Implementing early-stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'../data/models_weights/{model_name}.pth')
        elif avg_val_loss < best_val_loss*1.25:
            continue        
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping!")
                break

    return all_indices, all_embeddings, train_losses, val_losses, stsb_track, in_context_stsb_track

# Computing the InfoNCE loss. The anchor embeddings in this case should have shape [batch_size, dim], as well as the positive embeddings. The negative embeddings should have shape [batch_size, number_of_negatives, dim]. We then compute cosine similarity between the examples and the anchor and use them all in a smart way as logits for our cross-entropy loss (like the positive sample is our correct label and the negative samples are wrong labels)
def infonce_loss(anchor_embeddings, pos_embeddings, neg_embeddings, device, temperature=0.07):
    # Normalizing embeddings for InfoNCE
    anchor_norm = F.normalize(anchor_embeddings, dim=1)
    pos_norm = F.normalize(pos_embeddings, dim=1)
    neg_norm = F.normalize(neg_embeddings, dim=2)

    # Computing positive similarity to acnhor (batch_size)
    pos_sim = torch.sum(anchor_norm*pos_norm, dim=1)

    # Computing negative similarities to anchor (batch_size, N)
    neg_sim = torch.bmm(anchor_norm.unsqueeze(1), neg_norm.transpose(1, 2)).squeeze(1)  

    # Computing logits (normalized by the temperature)
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    logits = logits/temperature
    
    # Computing InfoNCE loss
    labels = torch.zeros(anchor_norm.size(0), dtype=torch.long, device=device)
    loss = F.cross_entropy(logits, labels)

    return loss

# Training loop for the supervised contrastive learning (InfoNCE) 
def infonce_training_loop(model, tokenizer, optimizer, train_dataloader, val_dataloader, test_dataset, df, device, epochs=10, temperature=0.07, patience=3, model_name='infonce_bertimbau'):
    # Getting test indices to run in-context STS-B later
    test_indices = test_dataset.df.index.tolist()
    test_df = df.loc[test_indices, ['descricao_resumida', 'positive_contrastive', 'single_negative_contrastive']]
    
    # Varibales for saving the best model and early-stopping
    best_val_loss = float('inf')
    patience_counter = 0

    # Saving indicies and embeddings for later usage
    all_indices = []
    all_embeddings = []

    # Tracking loss and STS-B
    train_losses = []
    val_losses = []
    stsb_track = []
    in_context_stsb_track = []

    # Effective training loop
    for epoch in tqdm(range(epochs), desc="Epoch"):
        model.train()
        epoch_train_loss = 0.0

        for indices, anchor_ids, pos_ids, neg_ids in train_dataloader:
            # Saving indices
            all_indices.append(indices)

            # Moving tensors to device
            anchor_ids = anchor_ids.to(device)
            pos_ids = pos_ids.to(device)
            neg_ids = neg_ids.to(device)

            # Computing embeddings, saving them and computing loss
            optimizer.zero_grad()
            anchor_emb = model(anchor_ids)
            pos_emb    = model(pos_ids)
            all_embeddings.append(anchor_emb.cpu().detach())

            B, N, L = neg_ids.size()
            neg_flat = neg_ids.view(B*N, L)
            neg_emb_flat = model(neg_flat)
            neg_emb = neg_emb_flat.view(B, N, -1)

            loss = infonce_loss(anchor_emb, pos_emb, neg_emb, device, temperature=0.07)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss/len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation part of training loop
        model.eval()
        model.zero_grad()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for indices, anchor_ids, pos_ids, neg_ids in val_dataloader:
                # Moving tensors to dedvice
                anchor_ids = anchor_ids.to(device)
                pos_ids = pos_ids.to(device)
                neg_ids = neg_ids.to(device)

                # Computing embeddings and computing loss
                anchor_emb = model(anchor_ids)
                pos_emb = model(pos_ids)

                B, N, L = neg_ids.size()
                neg_flat = neg_ids.view(B*N, L)
                neg_emb_flat = model(neg_flat)
                neg_emb = neg_emb_flat.view(B, N, -1)

                loss = infonce_loss(anchor_emb, pos_emb, neg_emb, device, temperature=0.07)

                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss/len(val_dataloader)
        val_losses.append(avg_val_loss)

        # Tracking STS-B score as a validation method as well
        stsb_track.append(stsb_test(model, device, tokenizer, max_length=anchor_ids.size(1), model_loss='infonce', verbose=False))

        # Tracking in-context STS-B score as the validation method 
        in_context_stsb_track.append(in_context_stsb_test(model, device, tokenizer, test_df, max_length=anchor_ids.size(1), model_loss='infonce', verbose=False))

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | STS-B: {stsb_track[-1]:.4f} | in-context STS-B: {in_context_stsb_track[-1]:.4f}")

        # Implementing early-stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'../data/models_weights/{model_name}.pth')
        elif avg_val_loss < best_val_loss*1.25:
            continue
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping!")
                break

    return all_indices, all_embeddings, train_losses, val_losses, stsb_track

# Function to evaluate model on STS-B PTBR
def stsb_test(model, device, tokenizer, max_length=64, model_loss='vanilla', verbose=False):
    # Loading STS-B PTBR version from Hugging Face. Validation split is used because it's more concise and it has labels (the test doesn't)
    sts = load_dataset("PORTULAN/extraglue", "stsb_pt-BR", split="validation")

    model.eval()
    with torch.no_grad():
        if model_loss == 'vanilla':
            # Computing encodings for sentences, their similarity and comparing to the given label
            sims, labels = [], []
            for example in sts:
                sent1, sent2 = example["sentence1"], example["sentence2"]
                
                # Generating input_ids and attention_masks to foward pass sentences
                input_ids1 = tokenizer(sent1, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)['input_ids'].to(device)
                attention_mask1 = (input_ids1 != tokenizer.pad_token_id).to(device).long()

                input_ids2 = tokenizer(sent2, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)['input_ids'].to(device)
                attention_mask2 = (input_ids2 != tokenizer.pad_token_id).to(device).long()

                # Effectively passing sentences through model
                z1 = model(input_ids1, attention_mask=attention_mask1).last_hidden_state
                sum_emb1 = (z1*attention_mask1.unsqueeze(-1).float()).sum(dim=1)
                lengths1 = attention_mask1.unsqueeze(-1).float().sum(dim=1).clamp(min=1e-9)
                z1 = sum_emb1/lengths1
                
                z2 = model(input_ids2, attention_mask=attention_mask2).last_hidden_state
                sum_emb2 = (z2*attention_mask2.unsqueeze(-1).float()).sum(dim=1)
                lengths2 = attention_mask2.unsqueeze(-1).float().sum(dim=1).clamp(min=1e-9)
                z2 = sum_emb2/lengths2

                sims.append(F.cosine_similarity(z1, z2).item())
                labels.append(example["label"])

        # Turning on dropout because of the unsupervised SimCSE method
        elif model_loss == 'usimcse':
            model.dropout.train()

            # Computing encodings for sentences, their similarity and comparing to the given label
            sims, labels = [], []
            for example in sts:
                sent1, sent2 = example["sentence1"], example["sentence2"]

                # Generating input_ids and attention_masks to foward pass sentences
                input_ids1 = tokenizer(sent1, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)['input_ids'].to(device)
                input_ids2 = tokenizer(sent2, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)['input_ids'].to(device)

                # Effectively passing sentences through model
                z1 = model(input_ids1)
                z2 = model(input_ids2)
                sims.append(F.cosine_similarity(z1, z2).item())
                labels.append(example["label"])

        # Evaluating InfoNCE models by dropping out MLP head and getting CLS token
        elif model_loss == 'infonce':
            model = model.model

            # Computing encodings for sentences, their similarity and comparing to the given label
            sims, labels = [], []
            for example in sts:
                sent1, sent2 = example["sentence1"], example["sentence2"]
                
                # Generating input_ids and attention_masks to foward pass sentences
                input_ids1 = tokenizer(sent1, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)['input_ids'].to(device)
                attention_mask1 = (input_ids1 != tokenizer.pad_token_id).to(device).long()

                input_ids2 = tokenizer(sent2, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)['input_ids'].to(device)
                attention_mask2 = (input_ids2 != tokenizer.pad_token_id).to(device).long()

                # Effectively passing sentences through model
                z1 = model(input_ids1, attention_mask=attention_mask1).last_hidden_state
                sum_emb1 = (z1*attention_mask1.unsqueeze(-1).float()).sum(dim=1)
                lengths1 = attention_mask1.unsqueeze(-1).float().sum(dim=1).clamp(min=1e-9)
                z1 = sum_emb1/lengths1
                
                z2 = model(input_ids2, attention_mask=attention_mask2).last_hidden_state
                sum_emb2 = (z2*attention_mask2.unsqueeze(-1).float()).sum(dim=1)
                lengths2 = attention_mask2.unsqueeze(-1).float().sum(dim=1).clamp(min=1e-9)
                z2 = sum_emb2/lengths2

                sims.append(F.cosine_similarity(z1, z2).item())
                labels.append(example["label"])

    pearson, _ = pearsonr(sims, labels)
    if verbose:
        print(f"STS-B (Pearson): {pearson:.4f}")
    
    return pearson

# Function to evaluate model on our (in-context) generated dataset in STS-B fashion
def in_context_stsb_test(model, device, tokenizer, test_df, max_length=64, model_loss='vanilla', verbose=False):
    model.eval()
    with torch.no_grad():
        if model_loss == 'vanilla':
            # Computing encodings for sentences, their similarity and comparing to the given label
            sims, labels = [], []
            for index, row in test_df.iterrows():
                anchor, pos, neg = row['descricao_resumida'], row['positive_contrastive'], row['single_negative_contrastive']
                
                # Generating input_ids and attention_masks to foward pass sentences
                anchor_ids = tokenizer(anchor, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)['input_ids'].to(device)
                anchor_attention_mask = (anchor_ids != tokenizer.pad_token_id).to(device).long()

                pos_ids = tokenizer(pos, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)['input_ids'].to(device)
                pos_attention_mask = (pos_ids != tokenizer.pad_token_id).to(device).long()

                neg_ids = tokenizer(neg, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)['input_ids'].to(device)
                neg_attention_mask = (neg_ids != tokenizer.pad_token_id).to(device).long()

                # Effectively passing sentences through model
                anchor_z = model(anchor_ids, attention_mask=anchor_attention_mask).last_hidden_state
                anchor_sum_emb = (anchor_z*anchor_attention_mask.unsqueeze(-1).float()).sum(dim=1)
                anchor_lengths = anchor_attention_mask.unsqueeze(-1).float().sum(dim=1).clamp(min=1e-9)
                anchor_z = anchor_sum_emb/anchor_lengths
                
                pos_z = model(pos_ids, attention_mask=pos_attention_mask).last_hidden_state
                pos_sum_emb = (pos_z*pos_attention_mask.unsqueeze(-1).float()).sum(dim=1)
                pos_lengths = pos_attention_mask.unsqueeze(-1).float().sum(dim=1).clamp(min=1e-9)
                pos_z = pos_sum_emb/pos_lengths

                neg_z = model(neg_ids, attention_mask=neg_attention_mask).last_hidden_state
                neg_sum_emb = (neg_z*neg_attention_mask.unsqueeze(-1).float()).sum(dim=1)
                neg_lengths = neg_attention_mask.unsqueeze(-1).float().sum(dim=1).clamp(min=1e-9)
                neg_z = neg_sum_emb/neg_lengths

                # Appending cosine similarities and pre-set (artificial) labels
                sims.append(F.cosine_similarity(anchor_z, pos_z).item())
                sims.append(F.cosine_similarity(anchor_z, neg_z).item())
                labels.append(4.0)
                labels.append(1.0)

        # Turning on dropout because of the unsupervised SimCSE method
        elif model_loss == 'usimcse':
            model.dropout.train()

            # Computing encodings for sentences, their similarity and comparing to the given label
            sims, labels = [], []
            for index, row in test_df.iterrows():
                anchor, pos, neg = row['descricao_resumida'], row['positive_contrastive'], row['single_negative_contrastive']
                
                # Generating input_ids and attention_masks to foward pass sentences
                anchor_ids = tokenizer(anchor, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)['input_ids'].to(device)
                pos_ids = tokenizer(pos, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)['input_ids'].to(device)
                neg_ids = tokenizer(neg, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)['input_ids'].to(device)
                
                # Effectively passing sentences through model
                anchor_z = model(anchor_ids, attention_mask=anchor_attention_mask)
                pos_z = model(pos_ids, attention_mask=pos_attention_mask)
                neg_z = model(neg_ids, attention_mask=neg_attention_mask)

                # Appending cosine similarities and pre-set (artificial) labels
                sims.append(F.cosine_similarity(anchor_z, pos_z).item())
                sims.append(F.cosine_similarity(anchor_z, neg_z).item())
                labels.append(4.0)
                labels.append(1.0)

        # Evaluating InfoNCE models by dropping out MLP head and getting CLS token
        elif model_loss == 'infonce':
            model = model.model

            # Computing encodings for sentences, their similarity and comparing to the given label
            sims, labels = [], []
            for index, row in test_df.iterrows():
                anchor, pos, neg = row['descricao_resumida'], row['positive_contrastive'], row['single_negative_contrastive']
                
                # Generating input_ids and attention_masks to foward pass sentences
                anchor_ids = tokenizer(anchor, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)['input_ids'].to(device)
                anchor_attention_mask = (anchor_ids != tokenizer.pad_token_id).to(device).long()

                pos_ids = tokenizer(pos, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)['input_ids'].to(device)
                pos_attention_mask = (pos_ids != tokenizer.pad_token_id).to(device).long()

                neg_ids = tokenizer(neg, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)['input_ids'].to(device)
                neg_attention_mask = (neg_ids != tokenizer.pad_token_id).to(device).long()

                # Effectively passing sentences through model
                anchor_z = model(anchor_ids, attention_mask=anchor_attention_mask).last_hidden_state
                anchor_sum_emb = (anchor_z*anchor_attention_mask.unsqueeze(-1).float()).sum(dim=1)
                anchor_lengths = anchor_attention_mask.unsqueeze(-1).float().sum(dim=1).clamp(min=1e-9)
                anchor_z = anchor_sum_emb/anchor_lengths
                
                pos_z = model(pos_ids, attention_mask=pos_attention_mask).last_hidden_state
                pos_sum_emb = (pos_z*pos_attention_mask.unsqueeze(-1).float()).sum(dim=1)
                pos_lengths = pos_attention_mask.unsqueeze(-1).float().sum(dim=1).clamp(min=1e-9)
                pos_z = pos_sum_emb/pos_lengths

                neg_z = model(neg_ids, attention_mask=neg_attention_mask).last_hidden_state
                neg_sum_emb = (neg_z*neg_attention_mask.unsqueeze(-1).float()).sum(dim=1)
                neg_lengths = neg_attention_mask.unsqueeze(-1).float().sum(dim=1).clamp(min=1e-9)
                neg_z = neg_sum_emb/neg_lengths

                # Appending cosine similarities and pre-set (artificial) labels
                sims.append(F.cosine_similarity(anchor_z, pos_z).item())
                sims.append(F.cosine_similarity(anchor_z, neg_z).item())
                labels.append(4.0)
                labels.append(1.0)

    pearson, _ = pearsonr(sims, labels)
    if verbose:
        print(f"STS-B (Pearson): {pearson:.4f}")
    
    return pearson

# Function to plot losses obtained during training
def plot_training_curves(train_losses, val_losses, stsb_track, in_context_stsb_track, model_name):
    plt.figure(figsize=(8,4))
    plt.suptitle(f'Training Loss and Validation Loss for {model_name}')
    
    # Plotting training loss curve
    plt.subplot(2, 2, 1)
    plt.plot([i+1 for i in range(len(train_losses))], train_losses, 'x-', c='b')
    plt.title("Training Loss x Epoch")
    plt.xlabel("")
    plt.ylabel("")
    
    # Plotting validation loss curve
    plt.subplot(2, 2, 2)
    plt.plot([i+1 for i in range(len(val_losses))], val_losses, 'x-', c='r')
    plt.title("Validation Loss x Epoch")
    plt.xlabel("")
    plt.ylabel("")

    # Plotting STS-B curve
    plt.subplot(2, 2, 3)
    plt.plot([i+1 for i in range(len(stsb_track))], stsb_track, 'x-', c='g')
    plt.title("STS-B Pearson Correlation x Epoch")
    plt.xlabel("")
    plt.ylabel("")

    # Plotting in-context STS-B curve
    plt.subplot(2, 2, 4)
    plt.plot([i+1 for i in range(len(in_context_stsb_track))], in_context_stsb_track, 'x-', c='y')
    plt.title("In-Context STS-B Pearson Correlation x Epoch")
    plt.xlabel("")
    plt.ylabel("")
    
    plt.tight_layout()
    plt.show()

# Function to safe outputs for visualization tool
def saving_outputs(projections, tokens, attributions, text_indices, save_file='vanilla_bertimbau_umap.csv'):
    # Processing attributions to go back to more understandable tokens for visual tool
    new_tokens, new_attributions = [[] for i in tokens], [[] for i in attributions]
    for i, (sentence_tokens, sentence_attributions) in enumerate(zip(tokens, attributions)):
        counter = -1
        for token, attribution in zip(sentence_tokens[1:-1], sentence_attributions[1:-1]):
            if token == '[PAD]':
                new_tokens[i] = new_tokens[i][:-1]
                break

            if token[0] == '#':
                new_tokens[i][counter] += token[2:]
                new_attributions[i][counter] += attribution
            else:
                new_tokens[i].append(token)
                new_attributions[i].append(attribution)
                counter += 1

    # Getting dictionary of tokens and attributions
    token_attribution_map = [[] for i in range(len(new_tokens))]
    for i in range(len(new_tokens)):
        token_attribution_map[i] = {t: a for t, a in zip(new_tokens[i], new_attributions[i])}
    
    # Computing clusters and cluster_names columns (because we need to use them for the visual tool)
    clusters = np.full(len(projections), -1)
    cluster_names = np.full(len(projections), '', dtype=object)

    # Creating dataframe and saving projections
    visualization_df = pd.DataFrame(index=text_indices, data={'x': projections[:, 0], 'y': projections[:, 1], 'cluster': clusters, 'cluster_names': cluster_names, 'token_attribution_map': token_attribution_map})
    visualization_df.index.name='id'
    visualization_df.to_csv('../data/projections/' + save_file)

    return visualization_df