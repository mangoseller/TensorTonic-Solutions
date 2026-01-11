import numpy as np

def calculate_eigenvalues(matrix):
    return None if (not matrix or (matrix := np.asarray(matrix)).ndim != 2 or matrix.shape[0] != matrix.shape[1]) else (eigvals := np.linalg.eigvals(matrix))[np.lexsort((-eigvals.imag, -eigvals.real))]

