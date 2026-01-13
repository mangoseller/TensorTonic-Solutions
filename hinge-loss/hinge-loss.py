import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    return getattr(np.maximum(margin - (np.asarray(y_true, dtype=float) * np.asarray(y_score, dtype=float)), 0), reduction)()
