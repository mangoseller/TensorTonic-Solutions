import numpy as np

def adadelta_step(w, grad, E_grad_sq, E_update_sq, rho=0.9, eps=1e-6):
  w, grad, E_grad_sq, E_update_sq = (np.asarray(x) for x in [w, grad, E_grad_sq, E_update_sq])
  e_g = (rho * E_grad_sq) + (1-rho) * (grad ** 2)
  d_t = (-np.sqrt((E_update_sq + eps) / (e_g + eps))) * grad
  e_dt = (rho * E_update_sq) + (1 - rho) * (d_t **2)
  w_t = w + d_t
  return (w_t, e_g, e_dt)
    