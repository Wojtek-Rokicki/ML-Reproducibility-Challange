from src.optimizers.Optimizer import Optimizer

from src.logistic_regression.log_reg import log_reg_gradient, calculate_loss
from src.logistic_regression.stochastic_gradient import stochastic_gradient

from src.utils.method_utils import *


class SVRG(Optimizer):
    name = "SVRG"
    n_params_to_tune = 1

    def __init__(self,
                 lambda_: float,
                 q: float):
        """
        Implementation of SVRG method.
        Args:
            q:
        """
        self.q = q
        self.lambda_ = lambda_

    def set_params(self, new_lambda):
        self.lambda_ = new_lambda

    def optimize(self, w_0, tx, y, max_iter):
        w = [w_0]
        grads = []
        full_grads = [log_reg_gradient(y, tx, w_0)]
        losses = []
        n = len(y)

        L_max = np.linalg.norm(tx, 'fro') ** 2 + 5  # 100  #
        step_size = 1 / L_max  #(np.max([np.linalg.norm(tx[i]) ** 2 for i in range(len(tx))]) + self.lambda_)

        for k in range(max_iter):
            if k % self.q == 0:
                z = w[k]
                v = log_reg_gradient(y, tx, w[k])
                grads.append(v)

            i_k = np.random.choice(np.arange(len(y)))
            grad = stochastic_gradient(y, tx, w[k], [i_k]) - stochastic_gradient(y, tx, z, [i_k]) + v

            next_w = w[k] - step_size * grad

            w.append(next_w)
            full_grads = [log_reg_gradient(y, tx, w[k+1])]
            losses.append(calculate_loss(y, tx, next_w))

        return full_grads, losses
