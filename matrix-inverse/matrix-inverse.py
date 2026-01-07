import numpy as np

def matrix_inverse(A):
    return None if any((abs(np.linalg.det(A)) < 1e-10, A.shape[0] != A.shape[1])) \
    else np.linalg.inv(A)
