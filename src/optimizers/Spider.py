import numpy as np
from typing import List

from src.logistic_regression.log_reg_gradient import log_reg_gradient
from src.logistic_regression.stochastic_gradient import stochastic_gradient


def Spider(w_0, tx, y, max_iter, parameters) -> List:
    """
    Compute Spider
    :param w_0: initial weights vector.
    :param max_iter:
    :param tx: Built model
    :param y: Target data
    :param S2: Number of samples to compute gradient from.
    :return: List of Gradients
    """
    S2 = parameters.S2
    n_0 = parameters.n_0
    epsilon = parameters.epsilon


    grads = []
    oracle_grads = []
    w = [w_0]
    n = len(y)

    is_oracle_grad = False
    for t in range(max_iter):
        if t % n == 0:
            v_k = log_reg_gradient(y, tx, w[t])
            is_oracle_grad = False
        else:
            sample_indices = np.random.choice(range(0, len(y)), size=S2, replace=False)
            # i_t = np.random.choice(np.arange(1, n))
            # v_k = sgd(y, tx, w[t], [i_t]) - sgd(y, tx, w[t-1], [i_t]) - grads[t-1]
            v_k = stochastic_gradient(y, tx, w[t], sample_indices) - stochastic_gradient(y, tx, w[t - 1], sample_indices) - grads[t - 1]
            is_oracle_grad = True
        term1 = epsilon/(n_0*np.linalg.norm(v_k, 2))
        term2 = 1/(2*n_0)
        eta_k = np.min([term1, term2])  # choose minimum between two terms

        w_next = w[t] - eta_k*v_k

        w.append(w_next)
        grads.append(v_k)
        if is_oracle_grad is False:
            oracle_grads.append(v_k)

    return oracle_grads