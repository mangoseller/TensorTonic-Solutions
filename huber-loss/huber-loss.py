import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    return np.where(np.abs((e := np.asarray(y_true) - np.asarray(y_pred))) <= delta, 0.5 * (e ** 2), delta * (np.abs(e) - (0.5 * delta))).mean()