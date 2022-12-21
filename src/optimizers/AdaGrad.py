import numpy as np

from src.optimizers.Optimizer import Optimizer

from src.logistic_regression.log_reg import log_reg_gradient, calculate_loss
from src.logistic_regression.stochastic_gradient import stochastic_gradient


class AdaGrad(Optimizer):
    name = "AdaGrad"
    n_params_to_tune = 2

    def __init__(self,
                 lambda_: float,
                 q: int,
                 epsilon: float = 1e-8,
                 conv_rate: float = 1e-6):
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
        self.conv_rate = conv_rate

    def set_params(self, new_lambda, new_epsilon):
        self.lambda_ = new_lambda
        self.epsilon = new_epsilon
        
    def grad(self, X, y):
        """Returns the gradient vector"""
        y_pred = self.predict(X)
        d_intercept = -2*sum(y - y_pred)                    # dJ/d w_0.
        d_x = -2*sum(X[:,1:] * (y - y_pred).reshape(-1,1))  # dJ/d w_i.
        g = np.append(np.array(d_intercept), d_x)           # Gradient.
        return g / X.shape[0]  

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
        n = len(y)
        # Outputs
        grads = []
        losses = []
        w = [w_0]

        for t in range(max_iter):
            i_t = np.random.choice(np.arange(n))  # get index of sample for which to compute gradient
            g_t = log_reg_gradient(y, tx, w[t])
            grad = stochastic_gradient(y, tx, w[t], [i_t])
            G_t += np.linalg.norm(g_t)**2
            # v_k = self.lambda_/(np.sqrt(G_t) + self.epsilon) * g_t
            # G_t += np.linalg.norm(g_t)**2
            v_k = np.diag(self.lambda_ / np.sqrt(G_t) + self.epsilon) @ g_t
            w_next = w[t] - v_k
            w.append(w_next)

            if t % 10000 == 0:
                print(t)
            grads.append(grad)
            losses.append(calculate_loss(y, tx, w_next))
        
        return grads, losses
