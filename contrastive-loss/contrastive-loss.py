import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    return (lambda ar, br, yr: (lambda loss: {"mean": loss.mean, "sum": loss.sum}[reduction]())(
        (yr*(d := np.linalg.norm(ar - br, axis=-1))**2) + ((1-yr) * (np.maximum(0, margin - d)**2))))(
        np.asarray(a),
        np.asarray(b),
        np.asarray(y),
    )
