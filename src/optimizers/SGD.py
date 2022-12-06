import numpy as np

from src.optimizers.Optimizer import Optimizer

from src.logistic_regression.stochastic_gradient import stochastic_gradient


class SGD(Optimizer):
    name = "SGD"

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
        self.gamma = lambda_

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
        w = [w_0]
        n = len(y)

        for t in range(max_iter):
            i_t = np.random.choice(np.arange(n))  # get index of sample for which to compute gradient
            gradient = stochastic_gradient(y, tx, w[t], [i_t])
            w_next = w[t] - self.gamma*gradient

            w.append(w_next)

            # save only oracle calls
            if t % self.q == 0:
                grads.append(gradient)

        return grads
