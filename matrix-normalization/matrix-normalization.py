import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    return None if (matrix := np.asarray(matrix)).ndim != 2 else \
    {
        "max": lambda: matrix / np.where((norms := np.linalg.norm(matrix, ord=np.inf, axis=axis, keepdims = True)) == 0, 1, norms),
        "l1": lambda: matrix / np.where((norms := np.linalg.norm(matrix, ord=1, axis=axis, keepdims = True)) == 0, 1, norms),
        "l2": lambda: matrix / np.where((norms := np.sqrt(np.sum(matrix**2, axis=axis, keepdims=True))) == 0, 1, norms)
    }[norm_type]()

