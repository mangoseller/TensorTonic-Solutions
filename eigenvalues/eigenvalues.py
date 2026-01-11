import numpy as np

def calculate_eigenvalues(matrix):
    if not matrix or (matrix := np.asarray(matrix)).ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    real = -(eigenvalues := np.linalg.eigvals(matrix)).real
    im = -eigenvalues.imag
    idx = np.lexsort((im, real))
    return eigenvalues[idx]
