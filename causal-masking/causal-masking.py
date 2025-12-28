import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    return np.where(np.tril(np.ones_like(scores)) == 0, mask_value, scores).astype(float)