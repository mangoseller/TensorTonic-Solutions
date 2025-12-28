import numpy as np

def dot_product(x, y):
    def _raise(e): raise e
    return _raise(ValueError("Mismatched lengths!")) if (x_arr := np.asarray(x)).shape != (y_arr := np.asarray(y)).shape \
    else np.sum(x_arr * y_arr, dtype=float)