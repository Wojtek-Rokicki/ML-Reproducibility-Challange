import numpy as np
from typing import List

from src.optimizers.Optimizer import Optimizer

from src.logistic_regression.log_reg_gradient import log_reg_gradient
from src.logistic_regression.stochastic_gradient import stochastic_gradient


class Spider(Optimizer):
    name = "Spider"

    def __init__(self,
                 n_0: float,
                 epsilon: float,
                 q: int):
        """
        Implementation of SPIDER method.
        Args:
            n_0:
            epsilon:
            q: Number of iterations for each the variance reduction gradient should be saved
        """
        self.n_0 = n_0
        self.epsilon = epsilon
        self.q = q

    def optimize(self, w_0, tx, y, max_iter) -> List:
        """
        Compute Spider
        :param w_0: initial weights vector
        :param tx: Built model
        :param y: Target data
        :param max_iter:
        :return: List of oracle gradients
        """
        grads = []
        oracle_grads = []
        w = [w_0]
        n = len(y)
        lipshitz_const = 100  # np.linalg.norm(tx, 'fro') ** 2

        is_oracle_grad = False
        for t in range(max_iter):
            if t % self.q == 0:
                v_k = log_reg_gradient(y, tx, w[t])
                is_oracle_grad = True
            else:
                # sample_indices = np.random.choice(range(0, len(y)), size=S2, replace=False)
                i_t = np.random.choice(np.arange(n))
                # v_k = sgd(y, tx, w[t], [i_t]) - sgd(y, tx, w[t-1], [i_t]) - grads[t-1]
                v_k = stochastic_gradient(y, tx, w[t], [i_t]) - stochastic_gradient(y, tx, w[t-1], [i_t]) + grads[t - 1]
                is_oracle_grad = False

            # term1 = self.epsilon/(lipshitz_const*self.n_0*np.linalg.norm(v_k, 2))
            term2 = 1/(2*lipshitz_const*self.n_0)
            eta_k = term2  # np.min([term1, term2])

            w_next = w[t] - eta_k*v_k

            w.append(w_next)
            grads.append(v_k)
            if is_oracle_grad is True:
                oracle_grads.append(v_k)

        return oracle_grads
