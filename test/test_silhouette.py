import pytest
import numpy as np
from scipy.spatial.distance import cdist
import cluster


# Test 1: Check if silhouette scores are correctly calculated
X = np.array([[1,2],[2,3],[3,3],[3,2],[1,1]])
y = np.array([0,0,1,1,2])
sil = cluster.Silhouette()
scores = sil.score(X, y)
expected_scores = np.array([0.63, 0.63, -0.22, -0.22, 1.00])
assert np.allclose(scores, expected_scores, atol=1e-2), f"Expected {expected_scores}, but got {scores}"

# Test 2: Check if average silhouette score is correctly calculated
X = np.array([[1,2],[2,3],[3,3],[3,2],[1,1]])
y = np.array([0,0,1,1,2])
sil = cluster.Silhouette()
avg_score = sil.avg_score(X, y)
expected_avg_score = "Average silhouette score: 0.395"
assert avg_score == expected_avg_score, f"Expected {expected_avg_score}, but got {avg_score}"
