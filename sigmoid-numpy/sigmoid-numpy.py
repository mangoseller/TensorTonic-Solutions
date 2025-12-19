import numpy as np

def sigmoid(x):
    return np.apply_along_axis(
            lambda y: 1/(np.exp(-y, dtype=np.float32) + 1 + 1e-8), 0, np.array(x)) \
            if not isinstance(x, int) \
            else 1/(np.exp(-x, dtype=np.float32) + 1 + 1e-8)
