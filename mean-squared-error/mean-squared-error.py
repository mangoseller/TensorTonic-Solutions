import numpy as np

def mean_squared_error(y_pred, y_true):
    return None if (pred := np.asarray(y_pred, dtype=float)).shape != (true := (np.asarray(y_true, dtype=float))).shape else np.mean(((pred - true)**2))
