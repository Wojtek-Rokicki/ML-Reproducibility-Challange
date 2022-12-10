import numpy as np

from src.optimizers.Optimizer import Optimizer

from src.logistic_regression.log_reg import log_reg_gradient, calculate_loss


class AdaGrad(Optimizer):
    name = "AdaGrad"
    n_params_to_tune = 2

    def __init__(self,
                 lambda_: float,
                 q: int,
                 epsilon: float = 1e-8):
        """
        Implementation of AdaGrad method.
        Args:
            lambda_:
            q: Number of iterations for each the variance reduction gradient should be saved
            epsilon:
        """
        self.lambda_ = lambda_
        self.q = q
        self.epsilon = epsilon

    def set_params(self, new_lambda, new_epsilon):
        self.lambda_ = new_lambda
        self.epsilon = new_epsilon

    def optimize(self, w_0, tx, y, max_iter):
        '''Algoritm for adaptive gradient optimization.

        Adapts learing parameter - smaller rate for frequent features (well-suited for sparse data).

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
        '''
        D = len(w_0)
        G_t = np.zeros((D, D))

        # Outputs
        grads = []
        losses = []
        w = [w_0]

        for t in range(max_iter):
            g_t = log_reg_gradient(y, tx, w[t])
            G_t += np.linalg.norm(g_t)**2
            v_k = np.diag(self.lambda_ / np.sqrt(G_t) + self.epsilon) @ g_t
            w_next = w[t] - v_k
            w.append(w_next)

            grads.append(g_t)
            losses.append(calculate_loss(y, tx, w_next))

        return grads, losses
