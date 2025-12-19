import numpy as np

def leaky_relu(x, alpha=0.01):
    if isinstance(x, (int, float)):
        return np.array(alpha*x) if x < 0 else np.array(x)
    elif isinstance(x, list):
        x = np.array(x)
    return np.where(x < 0, alpha*x, x)
    """
    Vectorized Leaky ReLU implementation.
    """
