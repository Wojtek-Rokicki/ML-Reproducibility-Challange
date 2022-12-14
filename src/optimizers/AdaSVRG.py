from src.optimizers.Optimizer import Optimizer

from src.logistic_regression.log_reg import log_reg_gradient, calculate_loss
from src.logistic_regression.stochastic_gradient import stochastic_gradient

from src.utils.method_utils import *


class AdaSVRG(Optimizer):
    name = "AdaSVRG"
    n_params_to_tune = 2

    def __init__(self,
                 q: int,
                 lambda_: float,
                 epsilon: float = 1e-8):
        """
        Implementation of AdaSVRG method.
        Args:
            q: Number of iterations for each the variance reduction gradient should be saved
            lambda_:
        """
        self.q = q
        self.lambda_ = lambda_

    def set_params(self, new_lambda):
        self.lambda_ = new_lambda

    def optimize(self, w_0, tx, y, max_iter):
        D = len(w_0)
        G_k = np.zeros((D, D))
        
        # Outputs
        w = [w_0]
        grads = []
        losses = []

        step_size = lip_const(tx, self.lambda_)

        for k in range(max_iter):
            i_k = np.random.choice(np.arange(len(y)))

            if k % self.q == 0:
                z = w[k]
                v = log_reg_gradient(y, tx, w[k])
                grads.append(v)

            g_k = stochastic_gradient(y, tx, w[k], [i_k]) - stochastic_gradient(y, tx, z, [i_k]) + v
            G_k += np.linalg.norm(g_k) ** 2
            A_k = np.diag(step_size / np.sqrt(G_k))

            next_w = w[k] - A_k @ g_k

            w.append(next_w)
            losses.append(calculate_loss(y, tx, next_w))

        return grads, losses
