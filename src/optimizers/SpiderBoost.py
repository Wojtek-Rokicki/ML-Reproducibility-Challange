import numpy as np

from src.logistic_regression.log_reg_gradient import log_reg_gradient
from src.logistic_regression.stochastic_gradient import stochastic_gradient


class SpiderBoost:
    """
    Implementation of SPIDER boost method.
    """
    name = "SpiderBoost"

    def optimize(self, w_0, tx, y, max_iter):
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
        Returns
        -------
        grads : ndarray of shape (max_iter, D)
            Array of gradient estimators in each step of the algorithm.
        """
        # Intrinsic parameters initialization
        N = len(tx)
        n = len(y)

        # Outputs
        grads = []
        w = [w_0]

        # Algorithm
        for t in max_iter:
            lipshitz_const = np.linalg.norm(tx[t], 'fro')
            if t % n == 0:
                v_k = log_reg_gradient(tx, y, w[t]) # logistic cost function full gradient
            else:
                i = np.random.choice(np.arange(1, n))
                v_k = partial_sum = stochastic_gradient(y, tx, w, i) - stochastic_gradient(y, tx, w[t-1], i) + v_k
            w_next = w[t] - 1/(2*lipshitz_const)*v_k
            w.append[w_next]
            grads.append(v_k)

        return grads
