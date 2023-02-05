import pytest
import numpy as np
from scipy.spatial.distance import cdist
import cluster
import unittest

class TestSilhouette(unittest.TestCase):
    def test_score(self):
        silhouette = cluster.Silhouette()
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])
        scores = silhouette.score(X, y)
        np.testing.assert_allclose(scores, np.array([0.63, 0.63, 0.37, 0.37]))
        
    def test_avg_score(self):
        silhouette = cluster.Silhouette()
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])
        avg_score = silhouette.avg_score(X, y)
        self.assertEqual(avg_score, "Average silhouette score: 0.500")
        
if __name__ == '__main__':
    unittest.main()
