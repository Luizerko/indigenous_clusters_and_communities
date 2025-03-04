# Tainacan-Based Dataset

This page provides an overview of how we utilized the [Tainacan collection](https://tainacan.museudoindio.gov.br/) to build our dataset, which was later used for clustering and analysis.

## Dataset Creation Process

The first step in our workflow is data collection. We accomplish this through a two-stage process:

1. **Scraping the data:** A [Bash script](https://github.com/Luizerko/indigenous_clusters_and_communities/tree/main/tainacan_collection/scrapping_data.sh) is used to extract data from the Tainacan platform.

2. **Processing and structuring:** A [Python script](https://github.com/Luizerko/indigenous_clusters_and_communities/tree/main/tainacan_collection/creating_dataset.py) then processes the raw data and organizes it into a structured CSV format.

Once the dataset is prepared, we proceed with **data cleaning and exploratory analysis** to assess the volume and distribution of the data, identify key categories and their relationships and explore potential clustering opportunities. For a more in-depth analysis of the dataset, refer to the [Jupyter notebook](https://github.com/Luizerko/indigenous_clusters_and_communities/tree/main/tainacan_collection/dataset_exploration.ipynb).

Below, we provide a high-level summary of the dataset attributes, based on our initial inspection and documentation efforts.

| Column Name           | Description                                                                                                                                          |
|-----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `url`                | Link to the object in Tainacan's archive.                                                                                                           |
| `thumbnail`          | Link to the thumbnail image of the element. `NaN` if there's no image associated with the object.                                                   |
| `creation_date`      | Creation date of the object in Tainacan's archive. This is an internal variable from the platform.                                                   |
| `modification_date`  | Date of the last modification of the object in Tainacan's archive. This is an internal variable from the platform.                                   |
| `numero_do_item`     | String identifying the item. It has multiple formats (e.g., `DD.D.DD`, `DD.D.DDC`, `D`, `DD`, `DDD`, etc.).                                          |
| `tripticos`          | No relevant explanation (`D.DD` or `DD.DD`).                                                                                                        |
| `categoria`          | Category of the item. There are ten different and well-defined categories.                                                                          |
| `nome_do_item`       | Name of the object. Sometimes followed by an observation in parentheses.                                                                            |
| `nome_do_item_dic`   | Name of the item according to a dictionary. A second, more generic name for the object.                                                             |
| `colecao`            | Name of the collection the item belongs to.                                                                                                        |
| `coletor`            | Person or institution responsible for collecting the item.                                                                                         |
| `doador`             | Person or institution responsible for donating the item to the museum.                                                                              |
| `modo_de_aquisicao`  | How the item was obtained: bought, donated, exchanged, other, or unknown.                                                                           |
| `data_de_aquisicao`  | Date when the item was acquired by the museum.                                                                                                      |
| `ano_de_aquisicao`   | Year in which the item was acquired by the museum.                                                                                                 |
| `data_de_confeccao`  | Date when the item was made.                                                                                                                       |
| `autoria`            | Person or institution that made the item. Sometimes includes observations in parentheses.                                                          |
| `nome_etnico`        | Indigenous name of the item. Often includes the noun in quotes and additional information.                                                         |
| `descricao`          | Description of the object, including material, components, and functionality.                                                                      |
| `dimensoes`          | Dimensions of the object.                                                                                                                          |
| `funcao`             | Function of the object.                                                                                                                            |
| `materia_prima`      | Material the object is made of, categorized as *animal*, *vegetal*, *mineral*, or *sintetico*.                                                     |
| `tecnica_confeccao`  | Techniques used to make the item.                                                                                                                  |
| `descritor_tematico` | Keywords describing themes related to the item.                                                                                                    |
| `descritor_comum`    | Keywords describing generic categories related to the item.                                                                                         |
| `numero_de_pecas`    | Number of pieces for the item, often with a short description of the pieces.                                                                       |
| `itens_relacionados` | List of related items in `numero_do_item` format.                                                                                                  |
| `responsavel_guarda` | Museum responsible for the item.                                                                                                                   |
| `inst_detentora`     | Museum that owns the item (always "Museu do Índio").                                                                                               |
| `povo`               | Community associated with the item.                                                                                                               |
| `autoidentificacao`  | List of communities identified by the original owner as related to the item.                                                                       |
| `lingua`             | Language of the community associated with the item.                                                                                               |
| `estado_de_origem`   | List of Brazilian states associated with the item.                                                                                                |
| `geolocalizacao`     | Specific location where the item originated (e.g., city, community, or other description).                                                        |
| `pais_de_origem`     | Country where the item is from.                                                                                                                    |
| `exposicao`          | Exhibitions where the item was displayed, possibly including the date it was returned to the museum.                                               |
| `referencias`        | Bibliographic references related to the item.                                                                                                      |
| `disponibilidade`    | Accessibility status of the item: inaccessible, locally accessible, or fully accessible.                                                           |
| `qualificacao`       | Additional descriptive information.                                                                                                                |
| `historia_adm`       | Administrative history of the item, including how it was acquired by the museum. May include random related or unrelated information.              |
| `notas_gerais`       | General notes with various information, related or unrelated to the item.                                                                          |
| `observacao`         | Additional observations about the item, often unrelated information.                                                                               |
| `conservacao`        | Conservation state of the item: good, regular, or bad.                                                                                            |
| `image_path`         | Local path to the associated image.                                                                                                               |

## Data Processing

While many analyses were conducted during dataset assembly, additional preprocessing is required to prepare the data for clustering. Beyond handling inconsistencies such as formatting issues, missing data, and incorrect data structures, we focus on two key preprocessing steps crucial for clustering both images and text: **background removal for images** and **text normalization**.

### Image Background Removal

Although the collection's metadata contains some formatting inconsistencies, the images themselves are generally high-quality and well-processed. The problem, however, lies mostly on the consistency of the backgrounds. If they are not removed, the fine-tuned image extractors might group images based on background characteristics rather than object features, leading to misleading results.

To mitigate this issue, we made use of a state-of-the-art [open-source background removal pipeline](https://huggingface.co/briaai/RMBG-2.0). This solution includes built-in preprocessing, ensuring images are numerically normalized to have consistent size before background removal. Below are examples of the pipeline applied to our collection - images are mapped back to their original sizes before plotting:

<p align="center">
  <img src="assets/vase_br.jpg" alt="Original vase image." width="30%" style="margin: 5px;" />
  <img src="assets/vase_br_r.png" alt="Background removed vase image." width="30%" style="margin: 5px;" />
</p>
<p align="center">
  <img src="assets/bracelet_br.jpg" alt="Original bracelet image." width="30%" style="margin: 5px;" />
  <img src="assets/bracelet_br_r.png" alt="Background removed bracelet image." width="30%" style="margin: 5px;" />
</p>
<p align="center">
  <img src="assets/fiber_br.jpg" alt="Original fiber image." width="30%" style="margin: 5px;" />
  <img src="assets/fiber_br_r.png" alt="Background removed fiber image." width="30%" style="margin: 5px;" />
</p>

<p align="center">
  The top row presents a simple example featuring a vase - a single, well-defined object. The middle row, in turn, demonstrates background removal for multiple objects, while the bottom row showcases an object with a very complex (not well-defined) form. These examples illustrate both the variety of objects in the collection and the effectiveness of the background removal pipeline.
</p>

### Text Normalization