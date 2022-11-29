import numpy as np
from gradient import gradient

def sto_grad(y, tx, w, i_t):
    y_sgd = y[i_t]
    x_sgd = np.array(tx[i_t,:])
    return gradient(y_sgd, x_sgd, w)