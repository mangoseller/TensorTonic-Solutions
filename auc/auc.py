import numpy as np

def auc(fpr, tpr):
    return np.trapezoid(tpr, fpr)
