import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    return ((x := np.asarray(X, dtype=float)) - np.mean(x, axis=axis, keepdims=True)) / (np.std(x, keepdims=True, axis=axis) + eps)
