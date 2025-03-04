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

We now proceed to describe the technical details, report the obtained results and show some of the generated images for clarity:

#### Technical Pipeline

Utilizamos dois principais modelos de extração de feautures para essa etapa, ambos baseados em transformers. Isso porque, para extração de features, normalmente fazemos uso do backbone de um modelo treinado em alguma tarefa de classificação geral relacionada a imagens (classificação do imagenet21K nos casos dos modelos que utilizamos) e as arquiteturas que usam de transformers atualmente são as state-of-the-art nessas tarefas.

Começamos com o modelo ViT Base com patches de tamanho 16x16 já que se trata de uma rede de extrema importância na área e que, apesar de não ser mais cutting-edge, ainda é uma boa referência inicial para modelos transformers (base de vários modelos state-of-the-art atuais, inclusive o outro que treinamos e o próximo que discutiremos DINOv2).

#### Results



### Text-Based Clustering



### Multimodal clustering


