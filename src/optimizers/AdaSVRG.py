from src.optimizers.Optimizer import Optimizer

from src.logistic_regression.log_reg import log_reg_gradient, calculate_loss
from src.logistic_regression.stochastic_gradient import stochastic_gradient

from src.utils.method_utils import *


class AdaSVRG(Optimizer):
    name = "AdaSVRG"
    n_params_to_tune = 1

    def __init__(self,
                 q: int,
                 lambda_: float):
        """
        Implementation of SPIDER method.
        Args:
            q: Number of iterations for each the variance reduction gradient should be saved
            lambda_:
        """
        self.q = q
        self.lambda_ = lambda_

    def set_params(self, new_lambda):
        self.lambda_ = new_lambda

    def optimize(self, w_0, tx, y, max_iter):
        w = [w_0]
        grads = []
        losses = []
        G = [0]
        n = len(y)

        # L = np.linalg.norm(tx, 'fro')**2 + lamb <- regularizer param
        # l_max (below) is max l for each row
        step_size = 1 / (np.max([np.linalg.norm(tx[i]) ** 2 for i in range(len(tx))]) + self.lambda_)

        for k in range(max_iter):
            i_k = np.random.choice(np.arange(len(y)))

            if k % self.q == 0:
                z = w[k]
                v = log_reg_gradient(y, tx, w[k])
                grads.append(v)

            g_t = stochastic_gradient(y, tx, w[k], [i_k]) - stochastic_gradient(y, tx, z, [i_k]) + v
            G_t = G[k] + np.linalg.norm(g_t) ** 2
            A_t = np.sqrt(G_t)

            next_w = w[k] - step_size * g_t / A_t

            w.append(next_w)
            G.append(g_t)
            losses.append(calculate_loss(y, tx, next_w))

        return grads, losses
