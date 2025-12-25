import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    return((comb(n, k) * p**k * (1-p)**(n-k)), 
    sum(comb(n, x) * p**x * (1-p)**(n-x) for x in range(0, k+1)))
