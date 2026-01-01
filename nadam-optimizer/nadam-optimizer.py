import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    return (lambda w_t, m_t, v_t, grad_:(
        w - (lr * ((beta1*m_t) + ((1-beta1) * grad_)) / (np.sqrt(v_t) + eps)),
        m_t,
        v_t
    ))(
        np.asarray(w),
        (beta1*np.asarray(m)) + (1-beta1) * (grad_t := np.asarray(grad)),
        (beta2*np.asarray(v)) + (1-beta2) * grad_t**2,
        grad_t      
    )
