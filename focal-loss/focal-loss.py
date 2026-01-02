import numpy as np

def focal_loss(p, y, gamma=2.0):
    return np.vectorize(
        lambda p_l, y_l:(-(1-p_l)**gamma * (y_l * np.log(np.clip(p_l, 1e-15, (1-1e-15)))) - (p_l**gamma * (1-y_l) * np.log(np.clip((1-p_l), 1e-15, (1-1e-15)))))
        )(p, y).mean()


