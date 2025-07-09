# Initiative for Indigenous Cultural Preservation in Brazil: a Study of Clusters and Communities

The modern world presents overwhelming challenges for most indigenous peoples, and Brazil is no exception. Urban development, often lacking in social responsibility, has led to the physical displacement of many indigenous people and a cultural erasure from the nation’s collective memory. Preserving their heritage has become a crucial task today. The [Brazilian Indigenous Museum](https://tainacan.museudoindio.gov.br/), through the [Tainacan platform](https://tainacan.org/), makes an effort in that direction by offering an extensive collection of indigenous artifacts, yet the accessibility and usability of this data remain limited due to the lack of proper structuring and interaction. Using data science and machine learning, we can explore new ways of organizing, visualizing, and analyzing this rich cultural heritage. This master thesis project thus aims to create an end-to-end pipeline that bridges indigenous knowledge with modern computational tools, being a cultural preservation initiative that contributes both to academic research and public engagement.

## Project Goals

The main challenges lie in extracting, organizing, and analyzing the data from the [Tainacan platform](https://tainacan.org/), which is currently not readily usable for large-scale computational tasks. The metadata structuring is inconsistent, the images require preprocessing, there are almost no possibilities of exploring relationships between artifacts based on dimensions such as visual or textual similarity and the interactivity with the collection is very limited. This project seeks to address these issues by building an end-to-end pipeline that starts with data extraction and normalization, followed by various applications of feature extraction, clustering and projection techniques to explore connections within the dataset, and finally the development of a tool for globally visualizing the collection. The aim is not only to understand these clusters but to compare them against one another, established indigenous literature and geographic data, creating an exploratory application that can inform and engage the general public through an interactive platform, as well as become a useful research instrument for future cataloging.

<p align="center">
  <img src="assets/visual_tool_summ.gif" alt="Visual tool overview." width="70%" style="margin-top: 20px;" />
</p>
<p align="center" style="margin-bottom: 15px;">
  Animated overview of visual tool interactions.
</p>

## Repository Hierarchy

    .
    ├── docs                  # Project's documentation folder
    ├── tainacan_collection   # Code collection for Tainacan data exploration (core code folder)
    |   ├── assets            # Stylesheets for visualization tool
    |   ├── image_based       # Code focused on image based feature extraction and clustering
    |   ├── language_based    # Code focused on language based feature extraction and clustering
    |   └── ...
    └── ...

This is the base folder for the project. In here, you'll find:
 
 - The [install documentation](https://github.com/Luizerko/indigenous_clusters_and_communities/tree/main/docs/INSTALL.md) to help set up your own environment.

 - The [dataset documentation](https://github.com/Luizerko/indigenous_clusters_and_communities/tree/main/docs/DATASET.md) describing how the data was extracted and processed, as well as explaining its attributes. 
 
 - The [visualization tool documentation](https://github.com/Luizerko/indigenous_clusters_and_communities/tree/main/docs/VISUALIZATION.md) containing details regarding the tool egineering and visual examples of its parts.
 
 - The [clustering documentation](https://github.com/Luizerko/master_thesis/tree/main/docs/CLUSTERING.md) describing all the clustering technical implementations and obtained results.
 
If you want to take a look at the code itself, go into the [`tainacan_collection`](https://github.com/Luizerko/indigenous_clusters_and_communities/tree/main/tainacan_collection) folder and explore the individual files inside the desired pipeline (image-based or textual-based).

I hope you have as much fun going through the repo as I had developing it all :grin: