from src.logistic_regression.log_reg_gradient import log_reg_gradient
from src.logistic_regression.stochastic_gradient import stochastic_gradient

from src.utils.method_utils import *


def SVRG(w_0, tx, y, max_iter, q):
    w = [w_0]
    grads = []
    gamma = 1 / np.max([np.linalg.norm(tx[i])**2 for i in range(len(tx))])

    for k in range(max_iter):
        i_k = np.random.choice(np.arange(len(y)))
        
        if k % q == 0:
            z = w[k]
            v = log_reg_gradient(y, tx, w[k])

        grad = stochastic_gradient(y, tx, w[k], i_k) - stochastic_gradient(y, tx, z, i_k) + v

        next_w = w[k] - gamma * grad

        w.append(next_w)
        grads.append(grad)
    
    return grads