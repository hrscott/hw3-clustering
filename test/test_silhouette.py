import pytest
import numpy as np
from scipy.spatial.distance import cdist
import cluster

#testing to see whether function returns the correct object shape
def test_silhouette_score_returns_correct_shape():
    silhou = cluster.Silhouette()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 0, 1])
    scores = silhou.score(X, y)
    assert scores.shape == (3,)
    
#testing to see whether avg_score function returns intended format 
def test_avg_score_result_format():
    silhouette = cluster.Silhouette()
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([0, 1, 0])
    result = silhouette.avg_score(X, y)
    assert isinstance(result, str)
    assert "Average silhouette score" in result

