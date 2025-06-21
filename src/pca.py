import numpy as np
from sklearn.preprocessing import StandardScaler


class PCA:
    def __init__(self, n_components=None):
        """
        Initialize PCA with specified number of components.

        Args:
            n_components (int, optional): Number of components to keep. If None, all components are kept.
        """
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.scaler = StandardScaler()

    def fit(self, X):
        """
        Fit the PCA model to the data.(Don't use np.cov, instead implement it from scratch. standardize the data first)

        Args:
            X (numpy.ndarray): Training data of shape (n_samples, n_features)

        Returns:
            self: The fitted PCA instance
        """
        X = np.array(X)
        n_samples, n_features = X.shape

        X_scaled = self.scaler.fit_transform(X)

        self.mean_ = np.mean(X_scaled, axis=0)

        X_centered = X_scaled - self.mean_

        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        if self.n_components is None:
            self.n_components = n_features

        self.n_components = min(self.n_components, n_features)

        self.components_ = eigenvectors[:, :self.n_components].T

        self.explained_variance_ = eigenvalues[:self.n_components]

        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance

        return self

    def transform(self, X):
        """
        Transform the data using the fitted PCA model. (Don't forget to standardize the data before transforming)

        Args:
            X (numpy.ndarray): Data to transform of shape (n_samples, n_features)

        Returns:
            numpy.ndarray: Transformed data of shape (n_samples, n_components)
        """

        X = np.array(X)

        X_scaled = self.scaler.transform(X)

        X_centered = X_scaled - self.mean_

        X_transformed = np.dot(X_centered, self.components_.T)

        return X_transformed

    def inverse_transform(self, X_transformed):
        """
        Transform data back to its original space.

        Args:
            X_transformed (numpy.ndarray): Transformed data of shape (n_samples, n_components)

        Returns:
            numpy.ndarray: Data in original space of shape (n_samples, n_features)
        """

        X_transformed = np.array(X_transformed)

        X_reconstructed = np.dot(X_transformed, self.components_)

        X_reconstructed += self.mean_

        X_original = self.scaler.inverse_transform(X_reconstructed)

        return X_original

    def get_explained_variance_ratio(self):
        """
        Get the explained variance ratio for each component.

        Returns:
            numpy.ndarray: Explained variance ratio for each component
        """
        return self.explained_variance_ratio_