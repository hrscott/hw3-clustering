"""
A Python module for kmeans clustering
"""
from .kmeans import Kmeans
from .silhouette import Silhouette
from .utils import (
        make_clusters, 
        plot_clusters,
        plot_multipanel)

__version__ = "0.1.1"