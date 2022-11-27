import numpy as np
from src.logistic_regression.gradient import gradient
from sgd import sto_grad

def AdaSVRG(w_0, tx, y, K, m):
    w = [w_0]
    n = len(y)

    for k in range(K - 1):
        k_grad = gradient(y, tx, w)
        step_size =  np.max([np.linalg.norm(tx[i])**2 for i in range(len(tx))])

        x = [w[k]]
        for t in range(1,m):
            i_t = np.random.choice(np.arange(1,n))
            g_t = sto_grad(y, tx, x[t - 1], i_t) - sto_grad(y, tx, w[k], i_t) + gradient(y, tx, w[k])

            next_x = eucl_proj(x[t - 1] - step_size * g_t) ## need to code this function

            x.append(next_x)
        
        next_w = np.mean(x)
        w.append(next_w)

    return np.mean(w)
