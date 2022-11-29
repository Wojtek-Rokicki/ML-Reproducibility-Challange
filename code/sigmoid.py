import numpy as np

# compute sigmoid function
def sigmoid(t):
    return 1.0 / (1 + np.exp(-t))