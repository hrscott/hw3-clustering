# Write your k-means unit tests here
import pytest
from cluster import Kmeans
import numpy as np


#testing to see if my function provides output that's reasonably similar to that provided by sklearn

# Test that the number of centroids generated is equal to the number of clusters specified in the K parameter
def test_number_of_centroids(self):
    kmeans = KMeans(K=5)
    X = np.random.randn(100, 2)
    kmeans.fit(X)
    self.assertEqual(kmeans.centroids.shape[0], 5) 