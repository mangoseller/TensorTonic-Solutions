import numpy as np

def t_test_one_sample(x, mu0):
    return (lambda avg, std: (avg - mu0) / (std / np.sqrt(len(x))))(
        (mean := np.asarray(x).mean()),
        np.sqrt((1 / (len(x) - 1)) * np.linalg.norm(x - mean)**2)
    )
