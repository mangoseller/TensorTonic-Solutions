import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    return np.tanh(
        np.array(x_t) @ np.array(Wx) + \
        np.array(h_prev) @ np.array(Wh) + b
    )

