import numpy as np

def manhattan_distance(x, y):
    return np.sum(np.abs(np.asarray(x, dtype=float) - np.asarray(y, dtype=float)))
