import numpy as np

def silhouette_score(X, labels):
    return (lambda dist, onehot: (lambda sizes, idx, sum_to:
    np.where((denom := np.maximum(
        (a := sum_to[np.arange(len(X)), idx] / np.maximum(sizes[idx] - 1, 1)),
        (b := np.where(np.arange(len(sizes)) == idx[:, None], np.inf, sum_to / sizes).min(axis=1))
    )) == 0, 0, (b - a) / denom).mean()
    )(
        onehot.sum(axis=0),
        onehot.argmax(axis=1),
        dist @ onehot
    ))(
        np.linalg.norm(X[:, None] - X, axis=2),
        (labels[:, None] == np.unique(labels)).astype(float)
    )

    