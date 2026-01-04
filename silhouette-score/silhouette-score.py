import numpy as np

def silhouette_score(X, labels):
    
    dist = np.linalg.norm(X[:, None, :] - X, axis=2)
    unique_labels = np.unique(labels)
    onehot = (labels[:, None] == unique_labels).astype(float)
    sizes = onehot.sum(axis=0)

    sum_to_clusters = dist @ onehot 
    cluster_idx = onehot.argmax(axis=1)

    a = sum_to_clusters[np.arange(len(X)), cluster_idx] / np.maximum(sizes[cluster_idx] - 1, 1)
    mean_to_clusters = sum_to_clusters / sizes
    mean_to_clusters[np.arange(len(X)), cluster_idx] = np.inf
    b = mean_to_clusters.min(axis=1)

    return np.where(np.maximum(a, b) == 0, 0, 
    (b - a) / np.maximum(a, b)).mean()
    