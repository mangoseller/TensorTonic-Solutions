import numpy as np

def dice_loss(p, y, eps=1e-8):
    return (lambda p_a, y_a:(
        1 - (2 * np.sum((p_a * y_a)) + eps) / (np.sum(p_a) + np.sum(y_a) + eps)
    ))(
        np.ravel(np.asarray(p, dtype=np.float64)),
        np.ravel(np.asarray(y, dtype=np.float64)),
    )

