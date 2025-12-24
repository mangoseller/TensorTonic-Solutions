import numpy as np
import math

def gelu(x):
    return np.vectorize((lambda y: (0.5 * y) * (1 + math.erf(y/math.sqrt(2)))))(np.array(x))
