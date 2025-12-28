import numpy as np

def dot_product(x, y):
    def _raise(e): raise e
    return _raise(ValueError("Mismatched lengths!")) if np.asarray(x).shape != np.asarray(y).shape \
    else np.sum(np.asarray(x)*np.asarray(y), dtype=float)