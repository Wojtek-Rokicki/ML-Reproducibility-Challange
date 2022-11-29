import numpy as np

from src.logistic_regression.log_reg_gradient import log_reg_gradient
from src.logistic_regression.stochastic_gradient import stochastic_gradient
"""
inputs:
- w_0: initial weights
- tx: dataset
- y: labels
- max_iter: number of iterations
"""


def Spider(w_0, max_iter, tx, y, parameters):
    """
    Compute spider
    :param w_0: initial weights vector.
    :param max_iter:
    :param tx:
    :param y:
    :param S2: Number of samples to compute gradient from.
    :return:
    """
    S2 = parameters.S2
    n_0 = parameters.n_0
    epsilon = parameters.epsilon


    grads = []
    w = [w_0]
    n = len(y)

    for t in range(max_iter):
        if t % n == 0:
            v_k = log_reg_gradient(y, tx, w[t])
        else:
            sample_indices = np.random.choice(range(0, len(y)), size=S2, replace=False)
            # i_t = np.random.choice(np.arange(1, n))
            # v_k = sgd(y, tx, w[t], [i_t]) - sgd(y, tx, w[t-1], [i_t]) - grads[t-1]
            v_k = stochastic_gradient(y, tx, w[t], sample_indices) - stochastic_gradient(y, tx, w[t - 1], sample_indices) - grads[t - 1]

        term1 = epsilon/(n_0*np.linalg.norm(v_k, 2))
        term2 = 1/(2*n_0)
        eta_k = np.min([term1, term2])  # choose minimum between two terms

        w_next = w[t] - eta_k*v_k

        w.append(w_next)
        grads.append(v_k)

    return grads
