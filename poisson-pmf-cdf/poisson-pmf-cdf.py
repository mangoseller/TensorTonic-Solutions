import numpy as np

def poisson_pmf_cdf(lam, k):
    pm = 0
    for i in range(k+1):
        pm += (np.exp(-lam) * lam ** i) / np.prod(np.arange(1, i+1))

    return((np.exp(-lam) * lam ** k) / np.prod(np.arange(1, k+1)), pm)
