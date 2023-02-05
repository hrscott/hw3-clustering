import pytest
import numpy as np
from scipy.spatial.distance import cdist
import cluster


# Check if silhouette scores are correctly calculated
def test_Silhouette_score():
    silhouette = cluster.Silhouette()

    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 1])
    expected = np.array([0.854, 0.276, 0.276])
    result = silhouette.score(X, y)
    assert np.allclose(result, expected, atol=1e-3)

    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([0, 1])
    expected = np.array([0.000, 0.000])
    result = silhouette.score(X, y)
    assert np.allclose(result, expected, atol=1e-3)

# Check if average silhouette score is correctly calculated
def test_Silhouette_avg_score():
    silhouette = cluster.Silhouette()

    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 1])
    expected = "Average silhouette score: 0.449"
    result = silhouette.avg_score(X, y)
    assert result == expected

    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([0, 1])
    expected = "Average silhouette score: 0.000"
    result = silhouette.avg_score(X, y)
    assert result == expected
