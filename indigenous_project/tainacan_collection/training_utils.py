import os
from tqdm import tqdm
from PIL import Image
from math import ceil

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import transforms
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
def preparing_image_labels(df, label_column='povo', offset=0):
    # label_column = 'povo', 'categoria', 'ano_de_aquisicao'
    name_to_num = {c: i+offset for i, c in enumerate(df[label_column].unique())}
    num_to_name = {c: i for i, c in name_to_num.items()}
    labels = {row['image_path_br']: name_to_num[row[label_column]] \
              for index, row in df.loc[df['image_path_br'].notna()].iterrows()}
    return labels, name_to_num, num_to_name

# Class for the ImageDataset and to avoid loading all the images simultaneously and run out of GPU memory
class ImageDataset(Dataset):
    def __init__(self, labels, transform=None, augment=False):
        self.image_files = list(labels.keys())
        self.labels = labels
        self.augment = augment
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = self.rgb_with_br(Image.open(image_path))
        
        if self.augment:
            augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.6),
                transforms.RandomVerticalFlip(p=0.6),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))], p=0.6)
            ])
            image = augment_transform(image)

        if self.transform:
            image = self.transform(image)
        label = self.labels.get(image_path, -1)
        return image, label, idx

    def rgb_with_br(self, image):
        # Creating white background to convert original image to RGB without reconstructing background
        background = Image.new("RGB", image.size, (255, 255, 255))
        image = Image.alpha_composite(background.convert("RGBA"), image).convert("RGB")
        return image

# Studying class distribution
def study_class_distribution(labels):
    # Counting categories
    categories = {}
    for l in labels.values():
        try:
            categories[l] += 1
        except:
            categories[l] = 1
    categories = dict(sorted(categories.items()))
    categories_keys = list(categories.keys())
    categories_freq = np.array(list(categories.values()))

    # Studying data distribution to filter out rare classes
    total_data = categories_freq.sum()
    q_10, q_25, q_50, q_75, q_90 = np.quantile(categories_freq, 0.10), np.quantile(categories_freq, 0.25), np.quantile(categories_freq, 0.50), np.quantile(categories_freq, 0.75), np.quantile(categories_freq, 0.90)
    mask_10, mask_25, mask_50, mask_75, mask_90 = np.where(categories_freq > q_10), np.where(categories_freq > q_25), np.where(categories_freq > q_50), np.where(categories_freq > q_75), np.where(categories_freq > q_90)

    print('Quantile X Data Percentage:')
    print(f'''Q-10: {q_10:.2f}, {categories_freq[mask_10].sum()/total_data*100:.2f}% of data''')
    print(f'''Q-25: {q_25:.2f}, {categories_freq[mask_25].sum()/total_data*100:.2f}% of data''')
    print(f'''Q-50: {q_50:.2f}, {categories_freq[mask_50].sum()/total_data*100:.2f}% of data''')
    print(f'''Q-75: {q_75:.2f}, {categories_freq[mask_75].sum()/total_data*100:.2f}% of data''')
    print(f'''Q-90: {q_90:.2f}, {categories_freq[mask_90].sum()/total_data*100:.2f}% of data\n''')

    qs = [q_10, q_25, q_50, q_75, q_90]
    masks = [mask_10, mask_25, mask_50, mask_75, mask_90]

    return categories, categories_keys, categories_freq, qs, masks

# Filtering dataset based on class distribution for image content
def filter_image_data_distribution(df, filtered_categories_names, transform, threshold_multiplier=2, column_name='povo'):
    # Filtering dataframe for selected categories
    filtered_ind_df = df[df[column_name].isin(list(filtered_categories_names.keys())) & df['image_path'].notna()]

    # Selecting minority and majority classes
    filtered_categories_freq = np.array(list(filtered_categories_names.values()))
    threshold = threshold_multiplier*np.median(filtered_categories_freq)

    minority_classes = []
    majority_classes = []
    for k, v in filtered_categories_names.items():
        if v <= threshold:
            minority_classes.append(k)
        else:
            majority_classes.append(k)

    minority_ind_df=filtered_ind_df[filtered_ind_df[column_name].isin(minority_classes)]
    majority_ind_df=filtered_ind_df[filtered_ind_df[column_name].isin(majority_classes)]

    # Undersampling majority classes
    undersampled_majority_ind_df = (
        majority_ind_df
        .groupby(column_name, group_keys=False)
        .apply(lambda x: x.sample(n=min(int(1.5*threshold), len(x)), replace=False))
    )

    # Creating augmented dataset for training
    labels_minority, minority_name_to_num, _ = preparing_image_labels(minority_ind_df, column_name)
    labels_majority, majority_name_to_num, _ = preparing_image_labels(undersampled_majority_ind_df, column_name, len(minority_classes))

    # print("New 'name_to_num':\n")
    # print(f'Minority classes: {minority_name_to_num}')
    # print()
    # print(f'Majority classes: {majority_name_to_num}')

    minority_datasets = [ImageDataset(labels_minority, transform=transform, augment=True) for i in range(ceil(threshold_multiplier))]
    minority_datasets.append(ImageDataset(labels_minority, transform=transform, augment=False))

    majority_datasets = [ImageDataset(labels_majority, transform=transform, augment=True)]
    majority_datasets.append(ImageDataset(labels_majority, transform=transform, augment=False))

    augmented_dataset = ConcatDataset(minority_datasets + majority_datasets)

    return minority_classes, majority_classes, labels_minority, labels_majority, augmented_dataset

# Plotting old and new class distributions
def plot_class_distributions(categories, filtered_categories, labels_minority, labels_majority, threshold_multiplier=2, column_name='povo'):
    new_categories = {i: 0 for i in range(len(filtered_categories))}
    for i in range(ceil(threshold_multiplier)+1):
        indices, counts = np.unique(list(labels_minority.values()), return_counts=True)
        for idx, count in zip(indices, counts):
            new_categories[idx] += count

    for i in range(2):
        indices, counts = np.unique(list(labels_majority.values()), return_counts=True)
        for idx, count in zip(indices, counts):
            new_categories[idx] += count

    plt.figure(figsize=(10,4))
    plt.suptitle(f"Class Distribution Before and After Rebalancing for '{column_name}' Column")

    plt.subplot(2, 1, 1)
    plt.bar(list(categories.keys()), list(categories.values()))
    plt.title("Before Rebalancing")

    plt.subplot(2, 1, 2)
    plt.bar(list(new_categories.keys()), list(new_categories.values()))
    plt.title("After Rebalancing")

    plt.tight_layout()
    plt.show()
    
# Computing class weights for unbalanced dataset
def compute_class_weights(filtered_categories, labels_minority, labels_majority, device, threshold_multiplier=2):
    # Because the dataset can still be unbalanced, we also create class weights for the loss function
    num_classes = len(filtered_categories)
    y = []
    for i in range(ceil(threshold_multiplier)+1):
        y = y + list(labels_minority.values())
    for i in range(2):
        y = y + list(labels_majority.values())
    y = np.array(y)

    class_weights = compute_class_weight(class_weight='balanced', \
                                        classes=np.arange(num_classes), y=y)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    return class_weights

# Function for getting traininig/validation split
def get_train_val_test_split(dataset, train_size, val_size, batch_size=32):
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    return train_dataloader, val_dataloader, test_dataloader

# Function to iterating over data to get projections
def get_vit_embeddings(model, dataloader, device, fine_tuned=False):
    image_embeddings = []
    image_indices = []
    model.eval()
    for batch_images, _, batch_indices in tqdm(dataloader, desc="Computing embeddings"):
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

        image_indices.append(batch_indices)

    return image_embeddings, image_indices

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
        for i, (prec, rec) in enumerate(zip(val_prec, val_rec)):
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

# Function for setting-up training, executing training and then running tests
def execute_train_test(dataset, dataloader, device, batch_size, epochs, num_classes, model, criterion, opt, model_name, column_name='povo'):
    # Creating training, validation and test datasets
    train_size = int(0.75*len(dataset))
    val_size = int(0.15*len(dataset))
    train_dataloader, val_dataloader, \
    test_dataloader = get_train_val_test_split(dataset, train_size, val_size, batch_size)

    # Training set-up and execution
    losses, accuracies, class_precisions, class_recalls = train_loop(model, num_classes, train_dataloader, val_dataloader, device, criterion, opt, model_name, epochs)
    plot_train_curves(losses, accuracies, f"ViT Fine-Tuned on '{column_name}'")
    print(f'Per class precision: {class_precisions[-1]}')
    print(f'Per class recall: {class_recalls[-1]}')

    # Evaluating model on test dataset
    test_acc, test_prec, test_rec = evaluate_model(model, model_name, num_classes, test_dataloader, device)
    print(f'Test accuracy: {test_acc}')
    print(f'Test per class precisions: {test_prec}')
    print(f'Test per class recalls: {test_rec}')

    # Computing image embeddings
    model.classifier = nn.Identity()
    image_embeddings, image_indices = get_vit_embeddings(model, dataloader, device, True)
    image_indices = np.concatenate(image_indices, axis=0)
    image_embeddings = np.concatenate(image_embeddings, axis=0)

    # Computing data projections
    trimap_proj, tsne_proj, umap_proj = data_projections(image_embeddings)

    return trimap_proj, tsne_proj, umap_proj, image_indices

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

# Function for evaluating the model on the test dataset
def evaluate_model(model, model_name, num_classes, test_dataloader, device):
    # Initializing evaluation metrics
    acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    prec_metric = Precision(task="multiclass", num_classes=num_classes, average=None).to(device)
    rec_metric = Recall(task="multiclass", num_classes=num_classes, average=None).to(device)

    # Loading best model, setting it to eval and forward passing
    model.load_state_dict(torch.load('data/models_weights/' + model_name + '.pth', weights_only=True))
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []

        for batch_images, batch_labels, _ in test_dataloader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            logits = model(batch_images)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.append(preds)
            all_labels.append(batch_labels)
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        test_acc = acc_metric(all_preds, all_labels).item()
        test_prec = prec_metric(all_preds, all_labels).tolist()
        test_rec = rec_metric(all_preds, all_labels).tolist()

    return test_acc, test_prec, test_rec