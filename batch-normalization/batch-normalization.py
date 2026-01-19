import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):

    fourD = len(x.shape) == 4
    sample = x.copy()
    if fourD:
        gamma = gamma[None, :, None, None]
        beta = beta[None, :, None, None]
        mu = np.mean(sample, axis=(0, 2, 3), keepdims=True)
        var = np.var(sample, axis=(0, 2, 3), keepdims=True)
    else:
        mu = np.mean(sample, axis=0, keepdims=True)
        var = np.var(sample, axis=0, keepdims=True) 

    x_hat = (x - mu) / np.sqrt(var + eps)
    y_hat = (gamma * x_hat) + beta

    return y_hat    


    