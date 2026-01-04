import numpy as np

def normalize_3d(v):
    return np.where(
        (norm := np.linalg.norm(v, axis=-1, keepdims=True)) < 1e-10,
         0,
         v / np.maximum(norm, 1e-10)
    ) # np.where is not lazily evaluated
