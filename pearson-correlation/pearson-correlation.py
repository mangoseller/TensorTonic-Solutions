import numpy as np

def pearson_correlation(X):
    if (X := np.asarray(X, dtype=np.float64)).ndim != 2 or X.shape[0] < 2:
        return None
    X -= X.mean(axis=0)
    dev = np.std(X, axis=0)
    denom = np.outer(dev, dev)
    denom = np.where(denom == 0, np.nan, denom)
    num = X.T @ X / (len(X))
    res = num / denom
    np.fill_diagonal(res, 1)
    return res
    
