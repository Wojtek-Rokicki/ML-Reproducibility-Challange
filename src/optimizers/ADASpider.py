import numpy as np

from src.logistic_regression.log_reg_gradient import log_reg_gradient
from src.logistic_regression.stochastic_gradient import stochastic_gradient


def ADASpider(w_0, max_iter, tx, y, **kwargs):
    grads = []
    w = [w_0]
    n = len(y)

    for t in range(max_iter):
        if t % n == 0:
            t_grad = log_reg_gradient(y, tx, w[t])
        else:
            i_t = np.random.choice(np.arange(1, n))
            t_grad = stochastic_gradient(y, tx, w[t], [i_t]) - stochastic_gradient(y, tx, w[t - 1], [i_t]) - grads[t - 1]

        grad_sum = np.sum(np.linalg.norm(grads)**2)
        gamma = 1 / (n**(1/4) * np.sqrt(np.sqrt(n) + grad_sum))
        w_next = w[t] - gamma * t_grad

        w.append(w_next)
        grads.append(t_grad)

    return grads
