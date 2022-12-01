import numpy as np
from AdaSpiderUtil import *

"""
inputs:
- w_0: initial weights
- tx: dataset
- y: labels
- max_iter: number of iterations
"""

def ADASpider(w_0, tx, y, max_iter):
    grads = []#np.empty((max_iter,len(w_0)))
    w = [w_0]
    n = len(y)

    for t in range(max_iter):
        if t % n == 0:
            t_grad = gradient(tx, y, w[t])
        else:
            i_t = np.random.choice(np.arange(1,n))
            t_grad = sto_grad(y, tx, w[t], i_t) - sto_grad(y, tx, w[t-1], i_t) - grads[t-1]
        
        grads.append(t_grad)
        gamma = 1 / (n**(1/4) * np.sqrt(np.sqrt(n) + grad_sum(grads)))

        w_next = w[t] - gamma * t_grad
        w.append(w_next)

    return w