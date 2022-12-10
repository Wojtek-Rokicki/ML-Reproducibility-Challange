import numpy as np

from src.optimizers.Optimizer import Optimizer

from src.logistic_regression.stochastic_gradient import stochastic_gradient
from src.logistic_regression.log_reg import calculate_loss


class SGD(Optimizer):
    name = "SGD"
    n_params_to_tune = 1

    def __init__(self,
                 q: int,
                 lambda_: float):
        """
        Implementation of SGD method.
        Args:
            q: Number of iterations for each the variance reduction gradient should be saved
            lambda_: Step size
        """
        self.q = q
        self.lambda_ = lambda_

    def set_params(self, new_lambda):
        self.lambda_ = new_lambda

    def optimize(self, w_0, tx, y, max_iter):
        """
        Compute Stochastic gradient Descent
        :param w_0: Initial weights vector
        :param max_iter: Maximum number of iterations
        :param tx: Built model
        :param y: Target data
        :return: List of Gradients
        """
        grads = []
        losses = []
        w = [w_0]
        n = len(y)

        for t in range(max_iter):
            i_t = np.random.choice(np.arange(n))  # get index of sample for which to compute gradient
            gradient = stochastic_gradient(y, tx, w[t], [i_t])
            w_next = w[t] - self.lambda_ * gradient

            w.append(w_next)
            grads.append(gradient)
            losses.append(calculate_loss(y, tx, w_next))

        return grads, losses
