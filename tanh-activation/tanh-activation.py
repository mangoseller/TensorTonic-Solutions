import numpy as np

def tanh(x):
    return np.atleast_1d(
        (lambda n: (np.exp(n) - np.exp(-n)) / (np.exp(n) + np.exp(-n)))
        (np.clip(np.asarray(x, dtype=np.float64), -600, 600)) # Handle numerical overflow/underflow
        )
        
