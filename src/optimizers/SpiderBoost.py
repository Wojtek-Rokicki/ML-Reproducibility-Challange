import numpy as np

from src.optimizers.Optimizer import Optimizer

from src.logistic_regression.log_reg_gradient import log_reg_gradient
from src.logistic_regression.stochastic_gradient import stochastic_gradient


class SpiderBoost(Optimizer):
    name = "SpiderBoost"

    def __init__(self,
                 q: int):
        """
        Implementation of SPIDER boost method.
        Args:
            q: Number of iterations for each the variance reduction gradient should be saved
        """
        self.q = q

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
        v_k = 0

        lipshitz_const = 200  # np.linalg.norm(tx, 'fro') ** 2

        # Algorithm
        for t in range(max_iter):
            if t % self.q == 0:
                v_k = log_reg_gradient(y, tx, w[t])
                is_oracle_grad = False
            else:
                i_t = np.random.choice(np.arange(n))
                v_k = stochastic_gradient(y, tx, w[t], [i_t]) - stochastic_gradient(y, tx, w[t-1], [i_t]) + v_k
                is_oracle_grad = True

            w_next = w[t] - 1/(2*lipshitz_const)*v_k
            w.append(w_next)

            if is_oracle_grad is False:
                grads.append(v_k)

        return grads
