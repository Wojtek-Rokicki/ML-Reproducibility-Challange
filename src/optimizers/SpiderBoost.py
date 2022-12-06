import numpy as np
from src.logistic_regression.log_reg_gradient import log_reg_gradient
from src.logistic_regression.stochastic_gradient import stochastic_gradient

### SpiderBoost

def compute_estimator_vk(tx, y, w, w_prev, prev_vk, S):
    '''Computes gradient estimator for SpiderBoost algorithm
    
    Parameters
    ----------
    tx : ndarray of shape (N, D)
        Array of input features
    y : ndarray of shape (N, 1)
        Array of output 
    w : ndarray of shape (D, 1)
        Current weights of the model
    w_prev : ndarray of shape (D, 1)
        Weights of the model from previous iteration
    prev_v_k : ndarray of shape (D, 1)
        Estimation of the gradient from previous iteration
    S : ndarray of shape (D, 1)
        Set of samples' indexes randomly chosen with replacement

    Returns
    -------
    v_k : ndarray of shape (D, 1)
        Estimation of the gradient
    '''
    res = 0
    for i in S:
        partial_sum = stochastic_gradient(y, tx, w, i) - stochastic_gradient(y, tx, w_prev, i) + prev_vk
        res = res + partial_sum
    v_k = res/len(S)

    return v_k

def spider_boost(w_0, tx, y, max_iter, L, q, S_size):
    """Algorithm for gradient optimization, which estimates gradients and reduces their iterative variance.

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
    L : float
        Lipshitz constant.
    q : int
        Parameter, for which each q-th iteration standard gradient is computed.

    Returns
    -------
    grads : ndarray of shape (max_iter, D)
        Array of gradient estimators in each step of the algorithm.
    """
    # Intrinsic parameters initialization
    eta = 1/(2*L)
    q = q
    K = max_iter
    N = len(tx)

    # Outputs
    grads = []
    w = [w_0]

    # Algorithm
    for k in range(K):
        if k%q == 0:
            v_k = log_reg_gradient(tx, y, w[k]) # logistic cost function full gradient
        else:
            S = np.random.choice(N, size = S_size)
            v_k = compute_estimator_vk(tx, y, w[k], w[k-1], v_k, S)
        w_next = w[k] - eta*v_k
        w.append[w_next]
        grads.append(v_k)
    
    return grads