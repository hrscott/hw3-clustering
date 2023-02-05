import pytest
import numpy as np
from scipy.spatial.distance import cdist
import cluster


def test_silhouette_score_returns_correct_shape():
    silhou = cluster.Silhouette()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 0, 1])
    scores = silhou.score(X, y)
    assert scores.shape == (3,)

def test_silhouette_score_returns_expected_values():
    silhou = cluster.Silhouette()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 0, 1])
    scores = silhou.score(X, y)
    expected_scores = np.array([-0.5, -0.5, 0.0])
    np.testing.assert_array_almost_equal(scores, expected_scores)

def test_silhouette_avg_score_returns_expected_value():
    silhou = cluster.Silhouette()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 0, 1])
    avg_score = silhou.avg_score(X, y)
    expected_avg_score = "-0.167"
    assert avg_score == f"Average silhouette score: {expected_avg_score}"
    y = np.array([0, 1])
    expected = "Average silhouette score: 0.000"
    result = silhouette.avg_score(X, y)
    assert result == expected
