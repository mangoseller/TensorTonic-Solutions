import numpy as np

def cosine_similarity(a, b):
    return 0.0 if (denom := np.linalg.norm(a) * np.linalg.norm(b)) == 0 else np.dot(a, b) / denom
