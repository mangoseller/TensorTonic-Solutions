import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    return ((x := np.asarray(X, dtype=float)) - np.min(x, axis=axis, keepdims=True)) / (np.max(x, axis=axis, keepdims=True) - np.min(x, axis=axis, keepdims=True) + eps)
