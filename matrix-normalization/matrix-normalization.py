import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        return None
        
    match norm_type:
        case 'max':
            return matrix / np.where((
                norms := np.linalg.norm(matrix, ord=np.inf, axis=axis, keepdims=True)) == 0, 1, norms)
        case 'l1':
            return matrix / np.where((
                norms := np.linalg.norm(matrix, ord=1, axis=axis, keepdims=True)) == 0, 1, norms)
        case 'l2':
            return matrix / np.where(
                (norms := np.sqrt(np.sum(matrix ** 2, axis=axis, keepdims=True))) == 0, 1, norms)

