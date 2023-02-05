
import numpy as np
from scipy.spatial.distance import cdist

class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        n_samples = X.shape[0]
        distances = cdist(X, X)
        cluster_sizes = np.bincount(y)
        intra_cluster_distance = np.zeros(n_samples)
        for cluster in range(cluster_sizes.shape[0]):
            mask = y == cluster
            intra_cluster_distance[mask] = np.mean(distances[mask][:, mask], axis=1)
        centroid_distance = np.zeros(n_samples)
        for cluster in range(cluster_sizes.shape[0]):
            mask = y != cluster
            centroid_distance[mask] = np.mean(np.min(distances[mask][:, y == cluster], axis=1))
        scores = (centroid_distance - intra_cluster_distance) / np.maximum(intra_cluster_distance, centroid_distance)
        return scores
    
    def avg_score(self, X, y):
        """
        calculates the average silhouette score for all observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            float
                the average silhouette score for all observations
        """
        scores = self.score(X, y)
        avg_score = np.mean(scores)
        return (f"Average silhouette score: {avg_score:.3f}")
       