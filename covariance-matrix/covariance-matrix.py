import numpy as np

def covariance_matrix(X):
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 2 or X.ndim != 2:
        return None
    X -= np.mean(X, axis=0)
    return np.asarray((X.T @ X) / (X.shape[0] - 1))
