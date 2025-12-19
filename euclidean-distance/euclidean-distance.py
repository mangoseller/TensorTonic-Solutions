import numpy as np

def euclidean_distance(x, y):
    return np.sqrt(np.sum((np.array(x) - np.array(y)).astype(np.float64)**2))
    