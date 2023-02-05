import numpy as np
from numpy.linalg import norm

class Kmeans:

    # initializing the number of clusters, maximum number of iterations, and the random seed to be used for random first placement 
    # of the centroids.
    def __init__(self, K: int, max_iter=300, random_seed=123,):
        self.K = K
        self.max_iter = max_iter
        self.random_seed = random_seed

    # defining  my helper functions as private methods (i.e. my API)        
    # selects random initial centroids from the input data.
    def _initialize_centroids(self, X):
        np.random.RandomState(self.random_seed)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.K]]
        return centroids
    
    # computing the squared Euclidean distance between each data point and each centroid.
    def _compute_euc_dist(self, X, centroids):
        distance = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            row_norm = norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance
    
    # assigning each data point to the closest centroid based on the computed distances.
    def _closest_cluster(self, distance):
        return np.argmin(distance, axis=1)
    
    ## Main implmentation of K-Means clustering
    # initializes the centroids, then repeatedly updates the centroids and assigns data points to the closest centroids 
    # until either the maximum number of iterations is reached or the centroids no longer change.
    def fit(self, X):
        self.centroids = self._initialize_centroids(X)
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distance = self._compute_euc_dist(X, old_centroids)
            self.labels = self._closest_cluster(distance)
            self.centroids = self.get_centroids(X, self.labels)
            if np.all(old_centroids == self.centroids):
                break
        self.error = self.get_error(X, self.labels, self.centroids)
    

    # assigning new data points to the closest cluster based on the trained centroids.
    def predict(self, X):
        distance = self._compute_euc_dist(X, self.centroids)
        return self._closest_cluster(distance)
    

    # calculating the sum of the squared distances between each data point and its assigned centroid. 
    def get_error(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.K):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))
    
    # calculating the mean of all data points assigned to each centroid and updating the  centroids accoridngly.
    def get_centroids(self, X, labels):
        centroids = np.zeros((self.K, X.shape[1]))
        for k in range(self.K):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return centroids
    