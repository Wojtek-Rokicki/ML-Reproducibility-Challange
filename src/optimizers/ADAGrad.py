import numpy as np
from src.logistic_regression.log_reg_gradient import log_reg_gradient

### AdaGrad

def AdaGrad(w_0, tx, y, max_iter, lambda_, epsilon = 1e-8):
    '''Algoritm for adaptive gradient optimization.

    Adapts learing parameter - smaller rate for frequent features (well-suited for sparse data). 

    Parameters
    ----------
    w_0 : ndarray of shape (D, 1)
        Initial weights of the model
    tx : ndarray of shape (N, D)
        Array of input features
    y : ndarray of shape (N, 1)
        Array of output
    max_iter : int
        Maximum number of iterations
    lambda_ : float
        Initial learning rate.
    epsilon : float
        Smoothing term, which avoids dividing by zero.

    Returns
    -------
    grads : ndarray of shape (max_iter, D)
        Array of gradient estimators in each step of the algorithm.
    '''
    K = max_iter
    lambda_ = lambda_
    epsilon = epsilon
    D = len(w_0)
    G_t = np.zeros((D,D))
    
    # Outputs
    grads = []
    w = [w_0]

    for k in range(K):
        g_t = log_reg_gradient(tx, y, w[k])
        G_t += np.diag([g_t_i ** 2 for g_t_i in g_t])
        v_k = lambda_ / np.sqrt(G_t + epsilon) * g_t
        w_next = w[k] - v_k
        w.append(w_next)
        grads.append(v_k)

    return grads