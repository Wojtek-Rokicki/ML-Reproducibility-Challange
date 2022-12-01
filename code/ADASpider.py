import numpy as np
from MethodUtils import *

"""
inputs:
- w_0: initial weights
- tx: dataset
- y: labels
- max_iter: number of iterations
"""

def ADASpider(w_0, tx, y, max_iter):
    grads = []
    w = [w_0]

    for t in range(max_iter):
        if t % len(y) == 0:
            t_grad = gradient(tx, y, w[t])
        else:
            i_t = np.random.choice(np.arange(len(y)))
            t_grad = sto_grad(y, tx, w[t], i_t) - sto_grad(y, tx, w[t-1], i_t) - grads[t-1]
        
        grads.append(t_grad)
        gamma = 1 / (n**(1/4) * np.sqrt(np.sqrt(n) + grad_sum(grads)))

        w_next = w[t] - gamma * t_grad
        w.append(w_next)

    return grads