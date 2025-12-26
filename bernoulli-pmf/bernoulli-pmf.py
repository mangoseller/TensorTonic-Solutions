import numpy as np

def bernoulli_pmf_and_moments(x, p):
    return (np.where(np.array(x) == 0, 1-p, p), float(p), float(p*(1-p)))
