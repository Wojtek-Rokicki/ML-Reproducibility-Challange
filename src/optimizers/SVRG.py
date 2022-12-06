from src.optimizers.Optimizer import Optimizer

from src.logistic_regression.log_reg_gradient import log_reg_gradient
from src.logistic_regression.stochastic_gradient import stochastic_gradient

from src.utils.method_utils import *


class SVRG(Optimizer):
    name = "SVRG"

    def __init__(self,
                 q: float,
                 lambda_: float):
        """
        Implementation of SVRG method.
        Args:
            q:
        """
        self.q = q
        self.lambda_ = lambda_

    def optimize(self, w_0, tx, y, max_iter):
        w = [w_0]
        grads = []
        step_size = 1 / (np.max([np.linalg.norm(tx[i]) ** 2 for i in range(len(tx))]) + self.lambda_)

        for k in range(max_iter):
            if k % self.q == 0:
                z = w[k]
                v = log_reg_gradient(y, tx, w[k])

            i_k = np.random.choice(np.arange(len(y)))
            grad = stochastic_gradient(y, tx, w[k], i_k) - stochastic_gradient(y, tx, z, i_k) + v

            next_w = w[k] - step_size * grad

            w.append(next_w)
            grads.append(grad)

        return grads
