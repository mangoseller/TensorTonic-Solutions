import numpy as np

def cross_entropy_loss(y_true, y_pred):
    return (lambda y_p, y_t: -np.mean(np.log(y_p[np.arange(y_p.shape[0]), y_t])))(
        np.asarray(y_pred),
        np.asarray(y_true),
    )
