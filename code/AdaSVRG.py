import numpy as np
from MethodUtils import *

def AdaSVRG(w_0, tx, y, K, m):
    w = [w_0]
    step_size = 1 / np.max([np.linalg.norm(tx[i])**2 for i in range(len(tx))])

    for k in range(K - 1):
        k_grad = gradient(y, tx, w[k])

        x = [w[k]]
        G = [0]
        for t in range(1, m + 1):
            i_t = np.random.choice(np.arange(len(y)))
            g_t = sto_grad(y, tx, x[t - 1], i_t) - sto_grad(y, tx, w[k], i_t) + k_grad
            G_t = G[t - 1] + np.linalg.norm(g_t)**2
            A_t = np.sqrt(G_t)

            next_x = x[t - 1] - step_size * g_t / A_t
            x.append(next_x)
            G.append(G_t)
        
        next_w = np.mean(x)
        w.append(next_w)

    return np.mean(w)
