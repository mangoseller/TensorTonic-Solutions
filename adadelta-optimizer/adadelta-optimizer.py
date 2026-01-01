import numpy as np

def adadelta_step(w, grad, E_grad_sq, E_update_sq, rho=0.9, eps=1e-6):
  return (lambda w_t, grad_, E_grad_sq_a, E_update_sq_a: (
    w_t + (d_t := (-np.sqrt(
      (E_update_sq_a + eps) / ((e_t := (rho * E_grad_sq_a) + (1-rho) * (grad_ ** 2)) + eps))) * grad_),
      e_t,
      (rho * E_update_sq_a) + (1 - rho) * (d_t **2)
  ))(
    *(map(np.asarray, [w, grad, E_grad_sq, E_update_sq]))
  )

    