import numpy as np

class Adam:
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0
        self.m = 0
        self.v = 0
        
    def update(self, params, grads):
        self.t += 1
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grads
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (grads ** 2)
        m_hat = self.m / (1 - self.beta_1 ** self.t)
        v_hat = self.v / (1 - self.beta_2 ** self.t)
        params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
