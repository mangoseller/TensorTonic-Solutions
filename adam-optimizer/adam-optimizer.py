import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    return (lambda m_t, v_t: (
        param - lr * (m_t / (1-beta1**t)) / (np.sqrt(v_t / (1-beta2**t)) + eps),
        m_t,
        v_t
    ))(
        beta1 * m + (1-beta1) * grad,
        beta2 * v + (1-beta2) * grad ** 2
    )
    