import numpy as np

from src.optimizers.Optimizer import Optimizer

from src.logistic_regression.log_reg import log_reg_gradient, calculate_loss
from src.logistic_regression.stochastic_gradient import stochastic_gradient


class SpiderBoost(Optimizer):
    name = "SpiderBoost"
    n_params_to_tune = 1

    def __init__(self,
                 q: int,
                 lambda_):
        """
        Implementation of SPIDER boost method.
        Args:
            q: Number of iterations for each the variance reduction gradient should be saved
        """
        self.q = q
        self.lambda_ = lambda_
        
    def set_params(self, new_lambda_):
        self.lambda_ = new_lambda_

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
        losses = []
        full_grads = []
        w = [w_0]
        v_k = 0

        lipshitz_const = np.linalg.norm(tx, 'fro') ** 2 + self.lambda_
        print(1/(2*lipshitz_const), lipshitz_const)
        
        # Algorithm
        for t in range(max_iter):
            if t % self.q == 0:
                v_k = log_reg_gradient(y, tx, w[t])
                full_grads.append(v_k)
            else:
                i_t = np.random.choice(range(0, len(y)), size=500, replace=False)
                # i_t = np.random.choice(np.arange(n))
                v_k = stochastic_gradient(y, tx, w[t], i_t) - stochastic_gradient(y, tx, w[t-1], i_t) + grads[t-1]

            w_next = w[t] - 1/(2*lipshitz_const)*v_k
            w.append(w_next)
            grads.append(v_k)
            losses.append(calculate_loss(y, tx, w_next))

        return full_grads, losses
