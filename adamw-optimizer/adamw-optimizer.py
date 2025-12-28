import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    return (lambda m_t, v_t, w_g: (
        w_g - (lr * (weight_decay * w_g)) - ((lr * m_t) / (np.sqrt(v_t) + eps)),
        m_t,
        v_t
    ))(
        beta1 * np.asarray(m) + (1-beta1) * np.asarray(grad),
        beta2 * np.asarray(v) + (1-beta2) * np.asarray(grad) ** 2,
        np.asarray(w)
    )