import numpy as np
from gradient import gradient
from sto_grad import sto_grad

"""
inputs:
- w_0: initial weights
- tx: dataset
- y: labels
- max_iter: number of iterations
"""

def ADASpider(w_0, max_iter, tx, y):
    grads = []
    w = [w_0]
    n = len(y)

    for t in range(max_iter):
        if t % n == 0:
            t_grad = gradient(tx, y, w[t])
        else:
            i_t = np.random.choice(np.arange(1,n))
            t_grad = sto_grad(y, tx, w[t], i_t) - sto_grad(y, tx, w[t-1], i_t) - grads[t-1]

        grad_sum = np.sum(np.linalg.norm(grads)**2)
        gamma = 1 / (n**(1/4) * np.sqrt(np.sqrt(n) + grad_sum))
        w_next = w[t] - gamma * t_grad

        w.append(w_next)
        grads.append(t_grad)

    return w