# Write your k-means unit tests here
import pytest
import cluster
import numpy as np
from sklearn.cluster import KMeans


#testing to see if my function provides output that's "reasonably" similar to that provided by sklearn 
# note that this is an anticipated failure on account of int vs float implementation)
def test_kmeans_vs_sklearn():
    np.random.seed(123)
    X = np.random.rand(100, 2)
    
    kmeans = cluster.Kmeans(K=3)
    kmeans.fit(X)
    custom_labels = kmeans.labels
    custom_centroids = kmeans.centroids
    
    sklearn_kmeans = KMeans(n_clusters=3)
    sklearn_kmeans.fit(X)
    sklearn_labels = sklearn_kmeans.labels_
    sklearn_centroids = sklearn_kmeans.cluster_centers_
    
    assert np.array_equal(custom_labels, sklearn_labels), "Labels do not match"
    assert np.allclose(custom_centroids, sklearn_centroids), "Centroids do not match"

# testing that the number of clusters is accurate
def test_number_of_clusters():
    kmeans = cluster.Kmeans(K=3)
    X = np.random.rand(100, 2)
    kmeans.fit(X)
    assert kmeans.K == 3

#test asserts that the error attribute after fitting the model is of type float, indicating that the error has been calculated.
def test_error_after_fit():
    kmeans = cluster.Kmeans(K=3)
    X = np.random.rand(100, 2)
    kmeans.fit(X)
    assert isinstance(kmeans.error, float)
