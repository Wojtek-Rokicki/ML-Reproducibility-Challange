from src.logistic_regression.log_reg_gradient import log_reg_gradient
from src.logistic_regression.stochastic_gradient import stochastic_gradient

from src.utils.method_utils import *


def AdaSVRG(w_0, tx, y, max_iter, q, lamb):
    w = [w_0]
    grads = []
    G = [0]
    # L = np.linalg.norm(tx, 'fro')**2 + lamb <- regularizer param
    # l_max (below) is max l for each row
    step_size = 1 / (np.max([np.linalg.norm(tx[i])**2 for i in range(len(tx))]) + lamb)

    for k in range(max_iter):
        i_k = np.random.choice(np.arange(len(y)))
        
        if k % q == 0:
            z = w[k]
            v = log_reg_gradient(y, tx, w[k])

        g_t = stochastic_gradient(y, tx, w[k], i_k) - stochastic_gradient(y, tx, z, i_k) + v
        G_t = G[k] + np.linalg.norm(g_t)**2
        A_t = np.sqrt(G_t)

        next_w = w[k] - step_size * g_t / A_t

        w.append(next_w)
        G.append(G_t)
        grads.append(g_t)

    return grads
