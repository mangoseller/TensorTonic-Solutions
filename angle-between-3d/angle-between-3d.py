import numpy as np

def angle_between_3d(v, w):
    return np.nan if (np.linalg.norm((v_arr := np.asarray(v))) == 0 or \
    np.linalg.norm((w_arr := np.asarray(w))) == 0) else \
    np.arccos(np.clip((np.dot(v_arr, w_arr)) / (np.linalg.norm(v_arr) * np.linalg.norm(w_arr)))) 
