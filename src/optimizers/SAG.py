
import numpy as np

from src.logistic_regression.log_reg import log_reg_gradient, calculate_loss
from src.logistic_regression.stochastic_gradient import stochastic_gradient

class SAG:
  def __init__(self, learning_rate=0.001):
    self.learning_rate = learning_rate
  
def optimize(self, w_0, tx, y, max_iter):
    w = [w_0]
    grads = []
    losses = []
    past_grads = []
    avg_grad 

    for t in range(max_iter):
        i_t = np.random.choice(np.arange(len(y)))
        t_grad = stochastic_gradient(y, tx, w[t], [i_t])
      
        past_grads.append(t_grad)
        avg_grad = np.mean(past_grads, axis=0)

        w_next = w[t] - self.learning_rate * (avg_grad + t_grad)
        w.append(w_next)
        grads.append(avg_grad + t_grad)
        losses.append(calculate_loss(y, tx, w_next))
