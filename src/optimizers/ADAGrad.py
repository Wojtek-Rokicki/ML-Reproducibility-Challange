import numpy as np
from src.logistic_regression.log_reg_gradient import log_reg_gradient


class AdaGrad:
    name = "AdaGrad"

    def __init__(self,
                 lambda_: float,
                 epsilon: float = 1e-8):
        """
        Implementation of AdaGrad method.
        Args:
            lambda_:
            epsilon:
        """
        self.lambda_ = lambda_
        self.epsilon = epsilon

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
        w = [w_0]

        for t in range(max_iter):
            g_t = log_reg_gradient(tx, y, w[t])
            G_t += np.diag([g_t_i ** 2 for g_t_i in g_t])
            v_k = self.lambda_ / np.sqrt(G_t + self.epsilon) * g_t
            w_next = w[t] - v_k
            w.append(w_next)
            grads.append(v_k)

        return grads
