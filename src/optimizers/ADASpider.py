from src.logistic_regression.log_reg_gradient import log_reg_gradient
from src.logistic_regression.stochastic_gradient import stochastic_gradient

from src.utils.method_utils import *


def ADASpider(w_0, tx, y, max_iter):
    """
    Compute ADASpider
    :param w_0: Initial weights vector.
    :param max_iter: Maximum number of iterations
    :param tx: Built model
    :param y: Target data
    :return: List of Gradients
    """
    grads = []
    oracle_grads = []
    w = [w_0]
    n = len(y)
    is_oracle_grad = False

    for t in range(max_iter):
        if t % len(y) == 0:
            t_grad = log_reg_gradient(y, tx, w[t])
            is_oracle_grad = False
        else:
            i_t = np.random.choice(np.arange(len(y)))
            t_grad = stochastic_gradient(y, tx, w[t], i_t) - stochastic_gradient(y, tx, w[t - 1], i_t) - grads[t - 1]
            is_oracle_grad = True

        grads.append(t_grad)
        gamma = 1 / (n ** (1 / 4) * np.sqrt(np.sqrt(n) + grad_sum(grads)))

        w_next = w[t] - gamma * t_grad
        w.append(w_next)
        if is_oracle_grad is False:
            oracle_grads.append(t_grad)

    return oracle_grads
