import numpy as np

def swish(x):
    return (lambda y: y / (1 + np.exp(-y))
    )(np.clip(np.atleast_1d(x), -500, 500))