import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    return (lambda x, gam, beta, mu, var: (gam * (x - mu) / np.sqrt(var + eps)) + beta)(
        x,
        gamma[None, :, None, None] if (fourD := len(x.shape) == 4) else gamma,
        beta[None, :, None, None] if fourD else beta,
        np.mean(x.copy(), axis=(0, 2, 3) if fourD else 0, keepdims=True),
        np.var(x.copy(), axis=(0, 2, 3) if fourD else 0, keepdims=True)
    )

    