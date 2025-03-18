# Clustering Experiments

This page outlines the clustering experiments we conducted, ranging from simple baseline methods to more sophisticated grouping techniques. Here, you will find details on the approaches we implemented, insights into what worked well, reflections on the challenges we encountered, and conclusions we can draw from all that.

## The Baselines

Before applying advanced machine learning techniques, we began by exploring fundamental aspects of our data. Instead of immediately relying on algorithms, we first sought to understand what insights could be directly extracted from the dataset and how domain knowledge from museum specialists could be integrated. Establishing these baselines was crucial for two reasons:

1. **Comparing our clusters to existing knowledge:** This helped evaluate how well our models captured essential patterns in the collection.

2. **Identifying knowledge gaps:** By analyzing our clustering results, we could pinpoint connections between items and communities that were previously not identified.

This analysis extends beyond traditional quantitative measures, such as embedding space distribution and clustering algorithm performance. Instead, it also focus on the **qualitative** aspects - evaluating whether the clusters align with meaningful patterns within the collection and how effectively they reveal new insights.

### Random Orthonormal Projections

To establish these baselines, we first explored clustering categorical features by mapping them into a higher-dimensional space. We couldn't, however, assume any inherent relationships between categories, making it crucial to treat clusters as equidistant. To enforce that, we applied **random orthonormal projections**. For each feature (in our expetiment `categoria` with 10 classes and `tipo_de_materia_prima` with 4 classes), we generated a random orthonormal matrix, ensuring every category had a unit vector, all of them orthogonal to one another. For multi-category features, we summed categories' vectors when a datapoint belonged to more than 1 category. This produced "high-dimensional" representations of data that we could use for clustering.

While theoretically interesting, this approach naturally failed. The resulting space was highly sparse, leading to poor clustering and later visualization. Even before clustering, lower-dimensional projections showed no clear groupings or symmetry.

To better preserve distances, we tested two projection techniques:

- **MDS:** Effective in low dimensions (so 14 shouldn't be a problem) and optimizes for distance preservation.

- **TriMap:** Suitable for higher-dimensional spaces while maintaining global structure.

Despite slight improvements with TriMap, neither method yielded meaningful clusters.

<div align="center">
    <br>
    <img src="assets/projections_orthonormal.png" alt="Projections using MDS and TriMap with Random Orthonormal Projections." width="500">
</div>
<div align='center'>
    <span>Plot showing MDS and TriMap 2D projections from random orthonormal vectors for 2 (concatenated) features (14 dimensions).</span>
    <br>
</div>

### Categorical Clustering

After the failure of random orthonormal projections, we returned directly to **categorical clustering**. We used **multi-hot encoding** to transform features with multiple category assignments into binary vectors. Then, we applied **K-Modes**, a clustering algorithm designed for categorical data that measures dissimilarity based on category mismatches instead of Euclidean distance.

Using `categoria` and `tipo_de_materia_prima` again, we found that **16 clusters** provided the best balance, slightly exceeding the sum of individual category counts (14) - not suggesting an overestimation but still accounting for possible correlations between categories across the features. While this approach prevented cluster assignment issues through the direct use of the categories, we still had a very sparse feature space and visualization remained challenging.

- **t-SNE** failed, producing unnatural circular patterns due to its KL-Divergence minimization on (sparse) categorical data.

- **UMAP** struggled, as the sparse feature space violated its assumption of an underlying manifold, leading to a chaotic point cloud.

- **TriMap** performed best, forming a few identifiable clusters. However, some clusters split across multiple areas - necessary for preserving some kind of equidistance between all clusters in 2D. Despite this improvement, visualization remained unclear.

<div align="center">
    <br>
    <img src="assets/projections_categorical.png" alt="Projections using t-SNE, UMAP and TriMap with Categorical Clustering." width="550">
</div>
<div align='center'>
    <span>Plot showing t-SNE, UMAP, and TriMap 2D projections from categorical vectors for 2 features (14 dimensions).</span>
    <br>
</div>

### Basic Feature-Based Clustering  

Due to the failures of the previous methods, we opted for a simpler approach, **directly using only the most well-defined and easily visualized feature** as baseline: `tipo_de_materia_prima`. This feature has three meaningful categories - *animal*, *vegetal*, and *mineral*. A fourth category (*sintetico*) exists, but no data points fall into this group. Items can also belong to multiple categories.

To represent clusters in 2D, we used a triangle representation:

- Each vertex represents one category (*animal*, *vegetal*, or *mineral*).

- Midpoints between vertices represent items that belong to two categories (each one of the closest vertices).

- Items that belong to all three categories are placed in the center of the triangle.

Since each category would otherwise collapse into a single point, making visualization difficult, we added 2D Gaussian noise to create a point-cloud effect.

<div align="center">
    <br>
    <img src="assets/tipo_de_materia_prima_baseline.png", alt="Plot showing tipo_de_materia_prima clusters." width="500">
</div>
<div align='center'>
    <span>Plot showing clusters of <i>tipo_de_materia_prima</i>.</span>
    <br>
</div>

### Specialist Taxonomy

We consulted museum specialists in indigenous cultures, but each expert typically focuses on a specific group or small communities. While specialists exist for every community, there is no centralized structure connecting or organizing relationships between them.

Given this, we identified **language** as the best proxy for mapping community relationships, as we have a well-defined hierarchical structure for most indigenous languages in Brazil. Using this framework, we built a **hierarchical graph** for communities, connecting them to one another through their languages.

## Clustering Through Machine Learning

### Image-Based Clustering

This section delves into the technical aspects of one of our machine learning approaches: using image-based clustering to group objects, aiming to understand how they connect through visual similarities. The results of this process serve two main purposes:

1. **Enhancing collection navigation:** By clustering visually similar objects together, we create an interactive and intuitive way for users to explore the collection. Similar objects will be positioned in close proximity within our final projection, allowing users to navigate different “micro-universes” of the collection, observe category transitions, and explore relationships between items.  

2. **Uncovering latent relationships:** Our models help to reveal previously undocumented connections between groups of objects or cultural communities. This is particularly valuable for researchers studying indigenous peoples, as it provides insights into shared artistic or manufacturing traditions. Given the lack of centralized taxonomies for indigenous groups in Brazil, our tool could serve as a pivotal resource for broader ethnographic studies in the country.  

To implement this, we use **image feature extractors** to project background-removed images (see [dataset documentation](https://github.com/Luizerko/indigenous_clusters_and_communities/tree/main/DATASET.md) for details) into high-dimensional space. We then **apply dimensionality reduction** techniques to visualize the clusters.

Beyond simple projections, we experimented with fine-tuning models to improve item dispersion and enable subdivision by specific attributes (e.g., `povo` or `categoria`). This allows users to explore both individual item neighborhoods and broader categorical relationships within the dataset. 

We now proceed to describe the technical pipelines implemented, report the obtained results and show some of the generated images for clarity. For this stage, we used two main feature extraction models, both based on transformers. Transformer-based architectures are currently state-of-the-art for feature extraction, as they leverage pretrained backbones with the best results when optimized on large-scale classification tasks (such as ImageNet21K in our case).

#### ViT Base (Patch 16x16)  

We started with the **ViT Base model with 16x16 patches**, trained on ImageNet21K, available on [Hugging Face](https://huggingface.co/google/vit-base-patch16-224-in21k). Although no longer cutting-edge, ViT remains a foundational model in the field and serves as a solid reference point for transformer-based architectures. Many state-of-the-art models, including the next one we discuss (DINOv2), build upon it.

For preprocessing, we resized images to 224x224 (cropping if larger in any dimension and bilinear interpolation if smaller), then normalized them using a mean of 0.5 and a standard deviation of 0.5 for all channels, following the model’s preprocessing pipeline.

Using only the **pretrained backbone**, we projected the images into a high-dimensional space and applied **dimensionality reduction techniques** to generate 2D visualizations for the interactive tool. We tested three different techniques:

- **TriMap:** Poor results, with minimal data dispersion and poor visual separation.

- **t-SNE:** Produced an entangled, chaotic cloud with no clear clusters.

- **UMAP:** Successfully created a meaningful manifold, capturing structure with the vanilla pretrained model and groupings when fine-tuned (as discussed later).  

<!-- Image showing vanilla projections for all methods (TriMap, t-SNE, and UMAP) -->

The resulting projection reveals a dense point cloud due to the lack of a specific training category. However, a **continuous manifold emerges**, where visually similar objects are positioned close together. This reflects the model’s ability to capture diverse visual similarities, including shape, colors, texture and details. This manifold alone offers a unique and interactive way to navigate the collection. But what happens when we introduce more structured knowledge into the data?

To refine clustering, we performed **fine-tuning** using the `povo` and `categoria` features, aiming for semantically distinct object groupings. This allows for categorical exploration and a more nuanced understanding of relationships between indigenous communities and their artistic traditions.

For that, we added a **classification head** to the network’s backbone - a **single linear layer** with 768-dimensional output from the backbone as the input size and the number of classes in the chosen feature as the output size. While common fine-tuning methods involve adding a small fully connected network, ViT fine-tuning is typically performed by adding a single linear layer at the top of the network. This approach is supported by section **3.1** of the [ViT original paper](https://ar5iv.labs.arxiv.org/html/2010.11929) and section **3.2** of [this paper](https://openreview.net/pdf?id=4nPswr1KcP), which explores ViT training strategies.

##### Training Models and Results

We trained several models to achieve the best possible results and assess the effectiveness of each adjustment we were implementing. Going into the implementation details, the dataset was split into 80% training, 10% validation, 10% test, and the original collection contained approximately **11,000 images**.

For each model, we tracked:  
  - **Loss**  
  - **Validation accuracy**  
  - **Average class precision**  
  - **Average class recall**  

All models were trained on a 8GB RTX 4070 until convergence (typically between 20 and 30 epochs) using:  
- **Adam optimizer** (with weight decay)  
- **Cross-entropy loss** (either with or without weights)  
- **Early stopping** (1% accuracy tolerance, 3-iteration patience)  
- **Five runs per model** (for mean and standard deviation analysis)  

Most models, however, were not trained directly on the original dataset due to **severe class imbalance** for `povo` and, to a lesser extent, `categoria`. To address this, we developed a **rebalancing pipeline**, which significantly improved model performance.  

<!-- Images showing class imbalance for 'povo' and 'categoria' -->

For `povo`, we started by understanding the distribution. We analyzed the quantiles of class sizes. `povo` contains 187 classes, but 25% of these (~47 classes) have only 4 images - insufficient for training. Even after removing these 25% least populated classes, around 99% of the dataset remains intact. We ultimately removed 75% of the least populated classes (~138 classes), keeping only classes with more than 65 images, preserving around 85% of the original data.

Despite filtering, class sizes still varied significantly. To address this we performed a **class median analysis**: classes with more than 2 times the median image count were labeled as *majority* classes, and others were labeled as *minority* classes. After that we started **data augmentation for minority classes** through random horizontal flips, random vertical flips and random Gaussian blur. For the majority classes, in turn, we **randomly (under)sampled images** to match minority class sizes. Notice, however, that only augmenting minority classes could introduce a bias where the model differentiates minority/majority classes based on artificially added noise. Thus, we applied stronger undersampling to majority classes and then also augmented them.  

Even after balancing, class disparities remained though. Because of that, we **assigned weights** inversely proportional to the amount of data the class had, ensuring equal contribution during training.  

For `categoria`, the procedure was nearly identical to `povo`, with one key difference: only one class (*"etnobotânica"*) was significantly underrepresented. Hence, instead of a full quantile study, we filtered this single class. The remaining balancing steps followed the same augmentation, undersampling, and weight adjustment process.

The tables below summarizes the parameters for different models and the corresponding quantitative results.

| Dataset | Learning Rate | Weight Decay | Frozen Layers (%) | Weighted Loss | Test Accuracy (%) | Avg. Precision | Avg. Recall | Avg. Precision on Selected Classes | Avg. Recall on Selected Classes | 
|-|-|-|-|-|-|-|-|-|-|
| Original | 5e-5 | 2e-6 | 0 | False | **67.72 ± 3.46** | 0.27 ± 0.02 | 0.25 ± 0.02 | 0.59 ± 0.03 | <ins>0.63 ± 0.05</ins> |
| Balanced | 2e-5 | 2e-6 | 0 | True | <ins>69.47 ± 1.58</ins> | - | - | **0.70 ± 0.02** | **0.70 ± 0.04** |
| Balanced | 2e-5 | 2e-6 | 50 | True | 68.87 ± 3.95 | - | - | <ins>0.68 ± 0.05</ins> | <ins>0.63 ± 0.08</ins> |
| Balanced | 2e-5 | 2e-6 | 80 | True | 66.56 ± 2.37 | - | - | 0.63 ± 0.04 | 0.62 ± 0.06 |
<p align="center" style="margin-bottom: 25px;">
  Parameters and results for ViT models fine-tuned on `povo`.
</p>

| Dataset | Learning Rate | Weight Decay | Frozen Layers (%) | Weighted Loss | Test Accuracy (%) | Avg. Precision | Avg. Recall | Avg. Precision on Selected Classes | Avg. Recall on Selected Classes | 
|-|-|-|-|-|-|-|-|-|-|
| Original | 1e-5 | 2e-6 | 0 | False | 88.11 ± 1.61 | 0.78 ± 0.04 | 0.75 ± 0.03 | <ins>0.87 ± 0.02</ins> | 0.84 ± 0.02 |
| Balanced | 3e-6 | 1e-6 | 0 | True | **88.65 ± 1.27** | - | - | **0.88 ± 0.03** | **0.85 ± 0.02** |
| Balanced | 3e-6 | 1e-6 | 50 | True | <ins>88.63 ± 1.75</ins> | - | - | <ins>0.87 ± 0.02</ins> | **0.85 ± 0.02** |
| Balanced | 3e-6 | 1e-6 | 80 | True | 87.12 ± 2.01 | - | - | 0.86 ± 0.04 | 0.84 ± 0.05 |
<p align="center" style="margin-bottom: 25px;">
  Parameters and results for ViT models fine-tuned on `categoria`.
</p>

As tabelas acima mostram que o modelo 

In addition to the previously mentioned models, we developed a multi-head model to explore the semantics of the network’s image projections when optimizing both features simultaneously. We implemented two classification heads - one for `povo` and another for `categoria` - with the loss being the weighted average of both losses.

Balancing the dataset was even more challenging in this case, as optimizing for one feature could disrupt the other. Joint distribution balancing was impractical due to the vast number of classes and because of the even worse joint-class imbalance, with little correlation between `povo` and `categoria`. Ultimately, we used the previously filtered classes for both features and initially balanced only for `povo`. This approach had minimal impact on `categoria`, which was already less imbalanced and not a major issue.

The table below summarizes the parameters for different head weights and the corresponding quantitative results.

| Learning Rate | Weight Decay | Head Weights (`povo`/`categoria`) | `povo` Head Test Accuracy (%) | `povo` Head Avg. Precision on Selected Classes | `povo` Head Avg. Recall on Selected Classes | `categoria` Head Test Accuracy (%) | `categoria` Head Avg. Precision on Selected Classes | `categoria` Head Avg. recall on Selected Classes |
|-|-|-|-|-|-|-|-|-|
| 1e-5 | 3e-6 | 50/50 | 68.82 ± 3.46 | 0.68 ± 0.02 | 0.67 ± 0.02 | 86.87 ± 1.95 | 0.85 ± 0.03 | 0.83 ± 0.05 |
| 1e-5 | 3e-6 | 70/30 | 71.11 ± 2.06 | 0.72 ± 0.03 | 0.70 ± 0.02 | 87.74 ± 2.34 | 0.88 ± 0.02 | 0.87 ± 0.03 |
| 1e-5 | 3e-6 | 30/70 | 67.91 ± 3.46 | 0.25 ± 0.02 | 0.27 ± 0.02 | 68.73 ± 1.95 | 0.60 ± 0.03 | 0.67 ± 0.05 |
<p align="center" style="margin-bottom: 25px;">
  Parameters and results for multi-head ViT models fine-tuned on both `povo` and `categoria`. The columns <i>Dataset</i>, <i>Frozen Layers (%)</i>, <i>Weighted Loss</i>, <i>Avg. Precision</i> and <i>Avg. Recall</i> are not found in this table because we trained all models with the same (balanced) dataset, no frozen layers, always with weighted loss for both heads and only on the selected categories.
</p>

#### DINOv2

| Dataset | Learning Rate | Weight Decay | Frozen Layers (%) | Weighted Loss | Test Accuracy (%) | Avg. Precision | Avg. Recall | Avg. Precision on Selected Classes | Avg. Recall on Selected Classes | 
|-|-|-|-|-|-|-|-|-|-|
| Original | 1e-6 | 3e-7 | 0 | False | 69.12 ± 4.27 | 0.29 ± 0.07 | 0.28 ± 0.09 | 0.61 ± 0.07 | 0.60 ± 0.08 |
| Balanced | 1e-6 | 3e-7 | 0 | True | 69.67 ± 1.88 | - | - | 0.69 ± 0.02 | 0.68 ± 0.03 |
| Balanced | 1e-6 | 3e-7 | 50 | True | **72.23 ± 3.32** | - | - | **0.72 ± 0.05** | **0.72 ± 0.03** |
| Balanced | 1e-6 | 3e-7 | 80 | True | 68.76 ± 2.94 | - | - | 0.69 ± 0.03 | 0.67 ± 0.04 |
<p align="center" style="margin-bottom: 25px;">
  Parameters and results for DINOv2 models fine-tuned on `povo`.
</p>

| Dataset | Learning Rate | Weight Decay | Frozen Layers (%) | Weighted Loss | Test Accuracy (%) | Avg. Precision | Avg. Recall | Avg. Precision on Selected Classes | Avg. Recall on Selected Classes | 
|-|-|-|-|-|-|-|-|-|-|
| Original | 1e-5 | 2e-6 | 0 | False | 88.11 ± 1.61 | 0.78 ± 0.01 | 0.75 ± 0.02 | **0.87 ± 0.02** | 0.84 ± 0.02 |
| Balanced | 3e-6 | 1e-6 | 0 | True | 88.04 ± 1.27 | - | - | 0.86 ± 0.01 | **0.85 ± 0.02** |
| Balanced | 3e-6 | 1e-6 | 50 | True | **88.23 ± 1.75** | - | - | 0.86 ± 0.02 | **0.85 ± 0.03** |
| Balanced | 3e-6 | 1e-6 | 80 | True | 87.63 ± 0.89 | - | - | 0.85 ± 0.01 | 0.84 ± 0.01 |
<p align="center" style="margin-bottom: 25px;">
  Parameters and results for DINOv2 models fine-tuned on `categoria`.
</p>

### Text-Based Clustering



### Multimodal clustering


