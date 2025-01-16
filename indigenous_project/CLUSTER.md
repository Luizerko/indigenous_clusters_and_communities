# Clustering Experiments

This page explains all the cluster experiments we made, from the most basic baselines to the most complex grouping computations. In here, one can find information about all the implemented methods, everything that went well and all that has gone wrong too.

## The Baselines

Before getting deep into heavy and fancy machine learning systems, we need to look into the basics of our data: what information can we extract directly from the dataset or from knowledge coming from our Museum specialists, and what does it tell us about the items? This is very important because, at some point, we are going to need to compare our sophisticated clusters to something, to understand their usuflness, more than just studying the success of an embedding space in spreading the data or the success of a certain clustering method.

For that, we tried multiple approaches. First, we began by selecting a few categorical features that were very easy to cluster by and tried to combine them in a higher dimensional space for clustering. The problem is, even thought they are easily clusterable since they are categorical, they 