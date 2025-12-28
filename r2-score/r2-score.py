import numpy as np

def r2_score(y_true, y_pred) -> float:
    return (lambda y_t, y_p: (
      1.0 if np.array_equal(y_true, y_pred)
      else 0.0 if np.var(y_t) == 0
      else 1.0 - np.sum((y_t - y_p)**2) / np.sum((y_t - np.mean(y_t))**2)
    ))(np.asarray(y_true), np.asarray(y_pred))