import numpy as np

def poisson_pmf_cdf(lam, k):
    return (lambda pmf: ((pmf(k), sum(pmf(i) for i in range(k+1)))))(
        lambda i: (np.exp(-lam) * lam ** i) / np.prod(np.arange(1, i+1))
        )

