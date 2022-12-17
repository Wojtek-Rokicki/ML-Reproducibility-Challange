
import numpy as np

from src.logistic_regression.log_reg import log_reg_gradient, calculate_loss
from src.logistic_regression.stochastic_gradient import stochastic_gradient

class SAG:
  name = "SAG"
  n_params_to_tune = 1
  def __init__(self, lambda_=0.001):
    self.learning_rate = lambda_
  
  def optimize(self, w_0, tx, y, max_iter):
    w = [w_0]
    grads = []
    losses = []
    past_grads = []
    full_grads = [log_reg_gradient(y, tx, w_0)]

    for t in range(max_iter):
        i_t = np.random.choice(np.arange(len(y)))
        t_grad = stochastic_gradient(y, tx, w[t], [i_t])
      
        past_grads.append(t_grad)
        avg_grad = np.mean(past_grads, axis=0)

        w_next = w[t] - self.learning_rate * (avg_grad + t_grad)
        w.append(w_next)
        full_grads.append(log_reg_gradient(y, tx, w[t+1]))
        grads.append(avg_grad + t_grad)
        losses.append(calculate_loss(y, tx, w_next))

    return full_grads, losses
