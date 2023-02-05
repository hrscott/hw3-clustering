import pytest
import numpy as np
from scipy.spatial.distance import cdist
import cluster


# write your silhouette score unit tests here

#testing to see if a score of 1 is correctly assigned to data with only 1 cluster
def test_single_cluster():
    X = np.array([[1,2],[2,3],[3,4],[4,5]])
    labels = np.zeros(X.shape[0])
    sil = cluster.Silhouette()
    score = sil.avg_score(X, labels)
    assert score == 1, f"Expected score of 1, but got {score}"

# checking if function correctly handles different densities of data points within each cluster. 
# reminder: The expected output is a score close to 1 for dense clusters and a score close to -1 for sparse clusters.
def test_varying_densities():
    X = np.vstack([np.random.normal(0, 1, (100, 2)), np.random.normal(5, 1, (20, 2))])
    labels = np.concatenate([np.zeros(100), np.ones(20)])
    score = cluster.Silhouette.avg_score(X, labels)
    assert score > 0.5, f"Expected score > 0.5, but got {score}"

#testing to see if the scoring function can correctly handle clusters with different shapes
def test_varying_shapes():
    X = np.vstack([np.random.normal(0, 1, (100, 2)), np.random.normal(5, 3, (100, 2))])
    labels = np.concatenate([np.zeros(100), np.ones(100)])
    score = cluster.Silhouette.avg_score(X, labels)
    assert abs(score) < 0.5, f"Expected score close to 0, but got {score}"
