import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    w, v, grad = np.array(w), np.array(v), np.array(grad)
    return (
        w - (momentum * v + lr*grad),
        momentum*v + lr*grad
    )
