import os
from tqdm import tqdm
from PIL import Image

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics.classification import Accuracy, Precision, Recall

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

# Function for preparing labels for dataset training
def preparing_image_labels(df, label_column='povo'):
    # label_column = 'povo', 'categoria', 'ano_de_aquisicao'
    name_to_num = {c: i for i, c in enumerate(df[label_column].unique())}
    labels = {row['image_path_br']: name_to_num[row[label_column]] \
              for index, row in df.loc[df['image_path_br'].notna()].iterrows()}
    return labels, name_to_num

# Class for the ImageDataset and to avoid loading all the images simultaneously and run out of GPU memory
class ImageDataset(Dataset):
    def __init__(self, image_dir, labels, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) \
                            if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels.get(image_path, -1)
        return image, label, idx

# Handling class imbalance


# Function for getting traininig/validation split
def get_train_val_split(dataset, train_size, batch_size=32):
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], \
                                          generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, \
                                  num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, \
                                num_workers=0, pin_memory=True)
    
    return train_dataloader, val_dataloader

# Function to iterating over data to get projections
def get_vit_embeddings(model, dataloader, device, fine_tuned=False):
    image_embeddings = []
    model.eval()
    for batch_images, _, _ in tqdm(dataloader, desc="Computing embeddings"):
        batch_images = batch_images.to(device)
        with torch.no_grad():
            # Soft "removing" classifier head, if fine-tuned model
            if fine_tuned:
                outputs = model.vit(batch_images)
            else:
                outputs = model(batch_images)
        
        # Do I get the last_hidden_state of CLS token or the pooler_output?
        embeddings = outputs['last_hidden_state'][:, 0, :]
        # embeddings = outputs['pooler_output']
        image_embeddings.append(embeddings.cpu())
    return image_embeddings

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

# Training function
def train_loop(model, num_classes, train_dataloader, val_dataloader, device, criterion, opt, model_name, epochs=20):
    losses = []
    accuracies = []
    class_precisions = [[] for i in range(num_classes)]
    class_recalls = [[] for i in range(num_classes)]

    # Early-stopping set up
    best_val_acc = 0
    patience = max(3, int(0.1*epochs))
    patience_counter = 0
    tolerance = 0.02

    acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    prec_metric = Precision(task="multiclass", num_classes=num_classes, \
                            average=None).to(device)
    rec_metric = Recall(task="multiclass", num_classes=num_classes, \
                        average=None).to(device)
    
    for epoch in tqdm(range(epochs), desc=f"Training model", leave=True):
        model.train()
        epoch_loss = .0
        for batch_images, batch_labels, _ in train_dataloader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            
            opt.zero_grad()
            logits = model(batch_images)
            loss = criterion(logits, batch_labels)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()

        losses.append(torch.tensor(epoch_loss, dtype=torch.float16).item())

        # Validation set for early-stopping on metrics that are not directly optimized
        model.eval()
        with torch.no_grad():
            all_preds = []
            all_labels = []

            for batch_images, batch_labels, _ in val_dataloader:
                batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
                logits = model(batch_images)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.append(preds)
                all_labels.append(batch_labels)
            
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            
            val_acc = acc_metric(all_preds, all_labels).item()
            val_prec = prec_metric(all_preds, all_labels).tolist()
            val_rec = rec_metric(all_preds, all_labels).tolist()

        accuracies.append(torch.tensor(val_acc, dtype=torch.float16).item())
        for i, prec, rec in enumerate(zip(val_prec, val_rec)):
            class_precisions[i].append(torch.tensor(prec, dtype=torch.float16).item())
            class_recalls[i].append(torch.tensor(rec, dtype=torch.float16).item())

         # Early-stopping check and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'best_val_acc': best_val_acc
            }, 'data/models_weights/' + model_name + '.pth')
        
            tqdm.write(f'Best model saved at epoch {epoch+1}')

        elif best_val_acc-val_acc > tolerance:
            patience_counter += 1
            if patience_counter >= patience:
                tqdm.write("Early-stopping training!")
                break
        
        tqdm.write((f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, '
                    f'Validation Accuracy: {val_acc:.4f}'))

    return losses, accuracies, class_precisions, class_recalls

# Function for plotting loss and accuracy curves
def plot_train_curves(losses, accuracies, model_name):
    plt.figure(figsize=(8,4))
    plt.suptitle(f'Loss and Accuracy Curves for {model_name}')
    
    # Plotting loss curve
    plt.subplot(1, 2, 1)
    plt.plot([i+1 for i in range(len(losses))], losses, 'x-', c='b')
    plt.title("Loss x Epoch")
    plt.xlabel("")
    plt.ylabel("")
    
    # Plotting accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot([i+1 for i in range(len(accuracies))], accuracies, 'x-', c='r')
    plt.title("Accuracy x Epoch")
    plt.xlabel("")
    plt.ylabel("")
    
    plt.tight_layout()
    plt.show()