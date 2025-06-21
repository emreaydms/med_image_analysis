import numpy as np
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment


def calculate_psnr(pred: np.ndarray, true: np.ndarray, eps: float = 1e-4) -> float:
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.

    PSNR is a metric used to measure the quality of reconstruction of an image.
    It is calculated as the ratio between the maximum possible power of a signal 
    and the power of corrupting noise that affects the fidelity of its representation.

    Args:
        pred (np.ndarray): The predicted image or reconstructed image as a NumPy array.
        true (np.ndarray): The ground truth image as a NumPy array.
        eps (float, optional): A small constant added to avoid division by zero. Default is 1e-4.

    Returns:
        float: The PSNR value in decibels (dB).
    """

    mse = np.mean((pred - true) ** 2)

    mse = mse + eps

    max_val = np.max(true)
    if max_val <= 1.0:
        max_val = 1.0
    else:
        max_val = 255.0

    psnr = 10 * np.log10((max_val ** 2) / mse)

    return float(psnr)


def map_clusters_to_labels_2d(pred_2d: np.ndarray, true_2d: np.ndarray, k: int) -> np.ndarray:
    """
        This function takes a 2D array of predicted cluster labels and a corresponding 2D array of ground truth labels.
        It computes a mapping from each cluster label to the most frequent true label within that cluster, and applies
        this mapping to the predicted labels. This algorithm uses np.vectorise to efficiently map the predicted labels to the true labels.

        Example:
            >>> import numpy as np
            >>> from scipy.stats import mode
            >>> pred_2d = np.array([[0, 0, 1], [1, 2, 2]])
            >>> true_2d = np.array([[1, 1, 2], [2, 3, 3]])
            >>> k = 3
            >>> map_clusters_to_labels_2d(pred_2d, true_2d, k)
            array([[1, 1, 2],
                   [2, 3, 3]])

    Maps predicted cluster labels in a 2D array to true labels based on the mode of true labels within each cluster.

    Args:
        pred_2d (np.ndarray): 2D array of predicted cluster labels.
        true_2d (np.ndarray): 2D array of ground truth labels.
        k (int): Number of clusters.

    Returns:
        np.ndarray: 2D array where each predicted cluster label is replaced by the most frequent true label in that cluster.
    """
    pred_flat = pred_2d.flatten()
    true_flat = true_2d.flatten().astype(int)
    mapping = {}

    for cluster_id in range(k):

        cluster_mask = (pred_flat == cluster_id)

        if np.any(cluster_mask):

            true_labels_in_cluster = true_flat[cluster_mask]


            if len(true_labels_in_cluster) > 0:

                mode_result = mode(true_labels_in_cluster, keepdims=False)
                most_frequent_label = mode_result.mode
                mapping[cluster_id] = most_frequent_label
            else:

                mapping[cluster_id] = cluster_id
        else:

            mapping[cluster_id] = cluster_id

    def map_func(cluster_label):
        return mapping.get(cluster_label, cluster_label)

    vectorized_map = np.vectorize(map_func)
    mapped_pred_2d = vectorized_map(pred_2d)

    return mapped_pred_2d


def clustering_accuracy(true_labels, pred_labels):
    """
    Calculate the clustering accuracy by finding the optimal assignment
    between true labels and predicted labels using the Hungarian algorithm.

    Parameters:
    -----------
    true_labels : numpy.ndarray
        Array of true labels for the data points. Must be a 1D array or
        flattenable to 1D.
    pred_labels : numpy.ndarray
        Array of predicted labels for the data points. Must be a 1D array or
        flattenable to 1D.

    Returns:
    --------
    float
        The clustering accuracy, computed as the sum of the optimal assignment
        values divided by the total sum of the confusion matrix.

    Notes:
    ------
    - This function uses the confusion matrix to compute the accuracy and
      applies the Hungarian algorithm (linear sum assignment) to find the
      optimal mapping between clusters and true labels.
    - The confusion matrix is adjusted to exclude the first row and column
      before performing the assignment.

    Example:
    --------
    >>> from sklearn.metrics import confusion_matrix
    >>> true_labels = np.array([0, 1, 1, 0])
    >>> pred_labels = np.array([1, 0, 0, 1])
    >>> clustering_accuracy(true_labels, pred_labels)
    0.5
    """
    true_flat = np.array(true_labels).flatten()
    pred_flat = np.array(pred_labels).flatten()

    cm = confusion_matrix(true_flat, pred_flat)

    n_true_classes, n_pred_classes = cm.shape

    max_classes = max(n_true_classes, n_pred_classes)
    if n_true_classes != n_pred_classes:

        square_cm = np.zeros((max_classes, max_classes), dtype=cm.dtype)
        square_cm[:n_true_classes, :n_pred_classes] = cm
        cm = square_cm

    row_indices, col_indices = linear_sum_assignment(-cm)

    optimal_sum = cm[row_indices, col_indices].sum()

    total_samples = cm.sum()

    accuracy = optimal_sum / total_samples

    return float(accuracy)