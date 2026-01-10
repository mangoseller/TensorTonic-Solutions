import numpy as np

def covariance_matrix(X):
    return None if ((X_a := np.asarray(X, dtype=np.float64)).shape[0] < 2 or X_a.ndim != 2) else \
    np.atleast_1d(((X_cent := X_a - np.mean(X_a, axis=0)).T @ X_cent) / (X_a.shape[0] - 1))


