import numpy as np

class KMeans:
    def __init__(self, n_clusters: int, max_iter: int = 100, random_state: int = None):
        """
        Initialize K-means clustering algorithm.

        Args:
            n_clusters (int): Number of clusters
            max_iter (int): Maximum number of iterations
            random_state (int): Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using k-means++ initialization.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)

        Returns:
            np.ndarray: Initial centroids of shape (n_clusters, n_features)
        """

        """
        I Used kmeans++ to select initial points which chooses first point randomly and chooses other points in probability proportional to their squared distance from the nearest existing centroid.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))

        centroids[0] = X[np.random.randint(n_samples)]

        for c_id in range(1, self.n_clusters):
            distances = np.zeros(n_samples)
            for i, point in enumerate(X):
                min_dist = float('inf')
                for j in range(c_id):
                    dist = np.sum((point - centroids[j]) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                distances[i] = min_dist

            probabilities = distances / np.sum(distances)
            cumulative_probs = np.cumsum(probabilities)
            r = np.random.rand()

            for i, cum_prob in enumerate(cumulative_probs):
                if r <= cum_prob:
                    centroids[c_id] = X[i]
                    break

        return centroids

    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each data point to the nearest centroid.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)

        Returns:
            np.ndarray: Cluster assignments for each data point
        """
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)

        for i, point in enumerate(X):
            distances = np.zeros(self.n_clusters)
            for j, centroid in enumerate(self.centroids):
                distances[j] = np.sum((point - centroid) ** 2)

            labels[i] = np.argmin(distances)

        return labels

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Update centroids based on current cluster assignments.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
            labels (np.ndarray): Current cluster assignments

        Returns:
            np.ndarray: Updated centroids
        """
        n_features = X.shape[1]
        new_centroids = np.zeros((self.n_clusters, n_features))

        for k in range(self.n_clusters):
            cluster_points = X[labels == k]

            if len(cluster_points) > 0:
                new_centroids[k] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[k] = self.centroids[k]

        return new_centroids

    def fit(self, X: np.ndarray) -> 'KMeans':
        """
        Fit the K-means clustering algorithm to the data.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)

        Returns:
            self: The fitted KMeans instance
        """
        self.centroids = self._initialize_centroids(X)

        for iteration in range(self.max_iter):
            self.labels = self._assign_clusters(X)

            new_centroids = self._update_centroids(X, self.labels)

            if np.allclose(self.centroids, new_centroids, rtol=1e-8):
                break

            self.centroids = new_centroids

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignments for new data.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)

        Returns:
            np.ndarray: Cluster assignments for each data point
        """

        return self._assign_clusters(X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the K-means clustering algorithm to the data and return cluster assignments.
        """
        self.fit(X)
        return self.predict(X)

    def inertia_(self, X: np.ndarray) -> float:
        """
        Calculate the inertia (within-cluster sum of squares).

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)

        Returns:
            float: The inertia value
        """

        inertia = 0.0
        labels = self._assign_clusters(X)

        for i, point in enumerate(X):
            cluster_id = labels[i]
            centroid = self.centroids[cluster_id]
            inertia += np.sum((point - centroid) ** 2)

        return inertia