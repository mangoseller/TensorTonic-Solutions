import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    w = np.asarray(w)
    g = np.asarray(g)
    s = np.asarray(s)
    s_t = (beta * s) + ((1-beta) * g**2)
    w_t = w - (lr / (np.sqrt(s_t) + eps)) * g
    return w_t, s_t
