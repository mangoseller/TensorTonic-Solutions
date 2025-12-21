import numpy as np

def relu(x):
    return np.maximum(0.0, np.atleast_1d(x).astype(np.float64))
