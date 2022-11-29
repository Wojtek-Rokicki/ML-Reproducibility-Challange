import numpy as np
from typing import List

from src.logistic_regression.log_reg_gradient import log_reg_gradient
from src.logistic_regression.stochastic_gradient import stochastic_gradient


def sgd(w_0, max_iter, tx, y, params):
    """
    Compute ADASpider
    :param w_0: Initial weights vector.
    :param max_iter: Maximum number of iterations
    :param tx: Built model
    :param y: Target data
    :param params: Additional parameters for SGD. 
    :return: List of Gradients
    """
    gamma = params.gamma
    grads = []
    w = [w_0]
    n = len(y)

    for t in range(max_iter):
        i_t = np.random.choice(np.arange(1, n))  # get index of sample for which to compute gradient
        gradient = stochastic_gradient(y, tx, w[t], [i_t])
        w_next = w[t] - gamma*gradient

        w.append(w_next)
        grads.append(gradient)
    return grads
