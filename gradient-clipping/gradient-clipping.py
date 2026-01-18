import numpy as np

def clip_gradients(g, max_norm):
    return (g := np.asarray(g, dtype=np.float64)) \
    if ((g_norm := np.linalg.norm(g, keepdims=True)) <= max_norm or max_norm <= 0) \
    else g * (max_norm / g_norm)
