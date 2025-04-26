# Centralizing main imports so we can run the models separately
import random
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset

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
        
        # Creating labels if dataset is used for supervised fine-tuning
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
        cls_embedding = outputs.last_hidden_state[:, 0]
        
        if self.target_type == 'l2-norm':
            cls_scalar = cls_embedding.norm(p=2, dim=1)
        elif self.target_type == 'cos-sim':
            cls_scalar = F.cosine_similarity(cls_embedding, self.baseline_embedding, dim=1)
        
        return cls_scalar
    
# Function to compute [CLS] token embeddings
def get_embeddings(model, tokenizer, input_ids, device, fine_tuned=False):
    # Computing attention mask
    attention_mask = (input_ids != tokenizer.pad_token_id).to(device).long()
    
    # And now the [CLS] token embeddings
    with torch.no_grad():
        if not fine_tuned:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        if fine_tuned:
            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0]

    return cls_embeddings

# Function to compute attributions via integrated gradients torwards the [CLS] token
def get_attributions(lig, tokenizer, input_ids, baseline_input_ids, attrib_aggreg_type='sum', return_tokens=True, verbose=False, sample_num=0):
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

# Class for the SimCSE models. The idea is to use fine-tune different models using the SimCSE approach with NT-Xent loss, which is an unsupervised contrastive loss, ideal for specializing the embedding model to our data without having to group it by categories
class SimCSEModel(nn.Module):
    def __init__(self, model, pad_token_id, device, dropout_prob=0.1):
        super(SimCSEModel, self).__init__()
        self.model = model.to(device)
        self.dropout = nn.Dropout(dropout_prob)
        
        self.device = device
        self.pad_token_id = pad_token_id
        
    def forward(self, input_ids, train=True):
        # Moving (or guaranteeing) input_ids to proper device
        input_ids = input_ids.to(self.device)

        # Computing attention mask and moving it to device as well
        attention_mask = (input_ids != self.pad_token_id).to(self.device).long()

        # Extracting [CLS] token embedding and passing it through dropout to late compute NT-Xent loss
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]
        if train:
            cls_embedding = self.dropout(cls_embedding)
        
        return cls_embedding
    
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

# Training loop for the SimCSE
def contrastive_training_loop(model, optimizer, train_dataloader, val_dataloader, device, epochs=10, temperature=0.05, patience=3, model_name='simcse_bertimbau'):
    # Varibales for saving the best model and early-stopping
    best_val_loss = float('inf')
    patience_counter = 0

    # Saving indicies and embeddings for later usage
    all_indices = []
    all_embeddings = []

    # Tracking loss
    train_losses = []
    val_losses = []

    # Effective training loop
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0

        for indices, input_ids, _ in train_dataloader:
            # Saving indices
            all_indices.append(indices)

            # Moving appropriate tensors to device
            input_ids = input_ids.to(device)
            
            # Duplicating batch for SimCSE
            input_ids = torch.cat([input_ids, input_ids], dim=0)
            
            # Computing embeddings, saving them, and computing loss
            embeddings = model(input_ids)
            all_embeddings.append(embeddings.cpu().detach())

            loss = nt_xent_loss(embeddings, temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss/len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation loss computation for early stopping and hyperparameter tunning
        model.eval()
        model.zero_grad()
        val_loss = 0
        with torch.no_grad():
            for indices, input_ids, _ in val_dataloader:
                # Repeating process for validation dataset
                input_ids = input_ids.to(device)
                input_ids = torch.cat([input_ids, input_ids], dim=0)
                embeddings = model(input_ids)
                val_loss += nt_xent_loss(embeddings, temperature).item()
        
        avg_val_loss = val_loss/len(val_dataloader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Implementing early-stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, '../data/models_weights/'+model_name+'.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping!")
                break

    return all_indices, all_embeddings, train_losses, val_losses

# Function to plot losses obtained during training
def plot_training_curves(train_losses, val_losses, model_name):
    plt.figure(figsize=(8,4))
    plt.suptitle(f'Training Loss and Validation Loss for {model_name}')
    
    # Plotting loss curve
    plt.subplot(1, 2, 1)
    plt.plot([i+1 for i in range(len(train_losses))], train_losses, 'x-', c='b')
    plt.title("Training Loss x Epoch")
    plt.xlabel("")
    plt.ylabel("")
    
    # Plotting accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot([i+1 for i in range(len(val_losses))], val_losses, 'x-', c='r')
    plt.title("Validation Loss x Epoch")
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