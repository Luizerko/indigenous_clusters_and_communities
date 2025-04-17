# Visualization Tool  

This page explains how we built our visualization tool. We go over the design choices and implementation, sharing the reasoning behind each step and highlighting the tool’s features. We also discuss its limitations and future improvements.

## The Foundation  

Our tool was built using `Python` and two powerful interactive libraries: `Plotly` and `Dash`. We chose these because our dataset creation, processing, and clustering experiments were all done in `Python`, making integration straight forward.

The system is organized into three main sections, each offering a distinct way to explore the dataset. The first tab presents an interactive view of the data as point clouds, allowing users to explore visual and semantic relationships between items through various projection and filtering tools. The second tab focuses on the temporal dimension of the collection, enabling users to browse the museum's acquisition timeline and analyze how items are distributed across specific years. The third and final tab offers a geographic perspective, displaying the full dataset on a map of Brazil and linking each object and community to its associated location.

<!-- Image of tabs -->

Below, we describe each of these components in more detail.

## Collection Semantic Space and Groupings

This tab presents several point-cloud based visualizations of the dataset. Each point represents an object, and users can choose between two display modes: marker-based - where colors can correspond to different clusters if applicable -  or image-based, where each point is replaced by the object’s actual image.

<!-- Image of markers and images -->

To ensure smooth performance - especially given the high number of points - we implement lazy loading throughout this tab. Rather than processing and rendering all points at once, the system first determines whether each point is currently visible within the user’s viewport. Only visible points are processed for actions like color assignment, plotting, and interactions. This significantly improves responsiveness and prevents unnecessary computations.

This approach, however, does come with a few trade-offs. For instance, when panning across the canvas, users may briefly see an empty area outside their initial view. The new region's points will load as soon as the mouse button is released, appearing instantly.

In image mode, we went a step further: images are fetched in real time rather than preloaded. Preloading hundreds of images upfront severely impacted interactivity - screens could take up to 25 or 30 seconds to render. Now, as you navigate, images dynamically reload, even for regions you've visited before. While this can result in a brief flicker when dragging around the view, we believe this is a worthwhile trade-off, as it keeps the interface responsive and fluid.

We mention these limitations to be transparent, but in practice, they have minimal impact on the user experience. The overall performance gains and interactivity more than make up for the occasional moment of minimally delayed rendering.

### Collapse Markers

Again due to the large volume of items in the collection, both visualization modes (markers and images) support collapse markers - translucent circles that represent multiple nearby points that can't be individually shown at the current zoom level. When points are too densely packed, plotting them all individually would lead to visual clutter and reduced performance. Instead, we display a larger, semi-transparent circle with a number inside, indicating how many objects are aggregated in that region. The more points within a collapsed group, the larger the circle. This feature not only improves readability, but also enhances performance by avoiding the need to render thousands of overlapping points simultaneously.

We initially implemented collapsing using *DBSCAN*, a density-based clustering algorithm. While it worked reasonably well, *DBSCAN*'s propagation mechanism sometimes grouped together points that were visually far apart, especially in denser regions. This made the collapsed markers feel imprecise or unintuitive in certain areas.

To address this, we transitioned to a more direct and spatially aware method using a *KD-Tree*. We build a *KD-Tree* over all visible points and perform efficient neighborhood searches to determine which points are close enough (based on zoom-normalized Euclidean distance) to be grouped. A distance threshold controls the grouping sensitivity, and small clusters - those with fewer than 5 points - are ignored, as their visual overlap is minimal and doesn't hinder exploration.

This approach produces more natural and intuitive groupings, better aligned with the user’s visual expectations, and while it’s slightly slower than the *DBSCAN* version, it still runs in real time.

### Hovering and Clicking

The marker-based point cloud is fully interactive and designed to support exploration. Depending on whether the user is browsing visual similarities or textual similarities, hovering over a point reveals a tooltip card:

- For visual similarity, the card displays the item’s image.

- For textual similarity, it shows the item's description with highlighted keywords - the terms considered the most relevant for the embedding computation of the item.

In both cases, the card also includes key metadata: the item’s name, the community it comes from, and the year it was acquired by the museum.

<!-- Image of the hovering boxes both for the image case and for the textual case -->

Clicking on a point opens the corresponding item’s page on the [Tainacan collection](https://tainacan.museudoindio.gov.br/), allowing for deeper inspection of that object.

The image-based point cloud is more straightforward and intentionally less interactive. Its goal is not detailed exploration but rather to provide a visually engaging overview of projected content - allowing users to immediately grasp grouping patterns through the visual density and arrangement of images. This version does not support hovering or clicking. It is intended as a passive, yet impactful, visualization.

### Filtering

## Collection on Brazil

Not much for now..... Let's work on this one a bit more before updating the documentation.....