from src.optimizers.Optimizer import Optimizer

from src.logistic_regression.log_reg import log_reg_gradient, calculate_loss
from src.logistic_regression.stochastic_gradient import stochastic_gradient

from src.utils.method_utils import *


class AdaSpider(Optimizer):
    name = "AdaSpider"
    n_params_to_tune = 0

    def __init__(self,
                 q: int):
        """
        Implementation of AdaSpider method.
        Args:
            q: Number of iterations for each the variance reduction gradient should be saved
        """
        self.q = q

    def optimize(self, w_0, tx, y, max_iter):
        """
        Compute ADASpider
        :param w_0: Initial weights vector
        :param max_iter: Maximum number of iterations
        :param tx: Built model
        :param y: Target data
        :return: List of Gradients
        """
        grads = []
        oracle_grads = []
        losses = []
        w = [w_0]
        n = len(y)
        is_oracle_grad = False

        for t in range(max_iter):
            if t % self.q == 0:
                t_grad = log_reg_gradient(y, tx, w[t])
                oracle_grads.append(t_grad)
            else:
                i_t = np.random.choice(np.arange(len(y)))
                t_grad = stochastic_gradient(y, tx, w[t], [i_t]) - stochastic_gradient(y, tx, w[t - 1], [i_t]) + grads[t - 1]

            gamma = 1 / (n ** (1 / 4) * np.sqrt(np.sqrt(n) + grad_sum(grads)))

            w_next = w[t] - gamma * t_grad
            
            w.append(w_next)
            grads.append(t_grad)
        
            losses.append(calculate_loss(y, tx, w_next))

        return oracle_grads, losses
