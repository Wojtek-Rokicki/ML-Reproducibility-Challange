import numpy as np
from MethodUtils import *

def SVRG(y, tx, w_0, max_iter, q):
    w = [w_0]
    grads = []
    gamma = 1 / np.max([np.linalg.norm(tx[i])**2 for i in range(len(tx))])

    for k in range(max_iter):
        i_k = np.random.choice(np.arange(len(y)))
        
        if k % q == 0:
            z = w[k]
            v = gradient(y, tx, w[k])

        grad = sto_grad(y, tx, w[k], i_k) - sto_grad(y, tx, z, i_k) + v

        next_w = w[k] - gamma * grad

        w.append(next_w)
        grads.append(grad)
    
    return grads