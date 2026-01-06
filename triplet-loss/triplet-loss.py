import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    return (lambda d, an, p, n: (max(0, (d(an, p) - d(an, n) + margin))))(
        lambda x, y: np.linalg.norm(x - y)**2,
        np.asarray(anchor),
        np.asarray(positive),
        np.asarray(negative),
    ) 
