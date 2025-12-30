import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    return (lambda new_g, old_w: (
       old_w - (lr * np.asarray(g)) / (np.sqrt(new_g) + eps), 
       new_g  
    ))(
        np.asarray(G) + np.asarray(g)**2,
        np.asarray(w)
    )
