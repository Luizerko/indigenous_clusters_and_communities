# Visualization Tool  

This page explains how we built our visualization tool. We go over the design choices and implementation, sharing the reasoning behind each step and highlighting the tool’s features. We also discuss its limitations and future improvements.  

## The Foundation  

Our tool was built using `Python` and two powerful interactive libraries: `Plotly` and `Dash`. We chose these because our dataset creation, processing, and clustering experiments were all done in `Python`, making integration straight forward.

The system has two main parts. The first tab allows users to explore clusters in different ways, with interactive features that enhance the experience. The second tab visualizes the full dataset across Brazil, linking objects and communities to their locations. Below, we describe each part in more detail.  

## Cluster Graphs

This tab shows different cluster visualizations of the points in a point-cloud format. The points are displayed either with markers - where different colors represent different clusters (illustrated by a legend), and each point corresponds to an object in our dataset - or through direct images (with removed backgrounds).

![IMAGE OF BOTH]

Both formats include transparent circles with numbers inside to indicate collapsed points. When points are too close together, instead of plotting them all in a crowded space, we display a larger circle showing how many points are in that small neighborhood. The more points in a collapse, the bigger the circle. Not only the collapsing helps the visualization, it also makes the tool much faster by avoiding plotting thousands of points at the same time.

![IMAGE OF COLLAPSING POINTS]

For this feature, we initially implemented collapsing using DBSCAN clustering, which assigns clusters based on density. While it worked reasonably well, DBSCAN’s method of propagating core points sometimes ended up grouping points that were far apart into the same cluster.

To improve this, we switched to a more direct approach using a KD-Tree. We built a KD-Tree with all points and then checked each point’s spatial neighborhood using the data-structure. By directly measuring Euclidean distances (normalized by the graph’s zoom level), we determined which points were close enough to be in the same cluster. We then applied a threshold: if points were within a certain distance, they were grouped together. Small clusters (fewer than 10 points) were filtered out since minor overlaps don’t clutter the image much. This approach provided a more natural user experience while keeping computation fast - only slightly slower but still real-time.

The point-cloud with markers is also interactive. Hovering over a point reveals a card showing an image of the item, its name, the community it comes from, and the year it was acquired by the museum. Clicking on a marker redirects the user to the item's page on the [Tainacan collection](https://tainacan.museudoindio.gov.br/).

![IMAGE OF THE HOVER BOX]

The point-cloud with direct images is simpler and much less interactive. Its main purpose is to provide a visually interesting way to see clusters, showing a cloud of images grouped together. Unlike the marker-based plot, this visualization offers minimal interactivity and does not update in real-time - it takes around 2 to 3 seconds to refresh when zooming or panning, as plotting all the images is computationally heavy.  

To speed up both visualizations (though the direct image plot is still slower than ideal), we use a lazy-loading approach that only plots the visible points. This significantly improves performance and scalability but introduces a minor drawback: when panning the graph, the new points don’t immediately appear, creating a slightly odd effect where points only load after releasing the mouse button.

## Collection on Brazil

Not much for now..... Let's work on this one a bit more before updating the documentation.....