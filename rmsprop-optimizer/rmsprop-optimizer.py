import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    return (lambda w_arr, g_arr, s_t: (
        w - (lr / (np.sqrt(s_t) + eps)) * g,
        s_t
    ))(
     np.asarray(w),
     np.asarray(g),
     (beta * np.asarray(s)) + ((1-beta) * np.asarray(g)**2)
     )

