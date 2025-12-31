import numpy as np

def angle_between_3d(v, w):
    return np.nan if ((v_norm := np.linalg.norm((v_arr := np.asarray(v)))) == 0 or \
    (w_norm := np.linalg.norm((w_arr := np.asarray(w)))) == 0) else \
    np.arccos((np.dot(v_arr, w_arr)) / (w_norm * v_norm))

