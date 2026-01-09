import numpy as np

def kl_divergence(p, q, eps=1e-12):
    return np.vectorize((lambda p_a, q_a: (
        np.where(p_a > 0, p_a * np.log(p_a / q_a), 0))))(
        np.asarray(p) + eps,
        np.asarray(q) + eps
    ).sum()
