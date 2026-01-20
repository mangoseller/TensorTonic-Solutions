import numpy as np

def vector_norm_3d(v):
    return (lambda k, y: np.linalg.norm(k, axis=y))(
        (V := np.asarray(v)),
        0 if V.ndim == 1 else -1,
    )
