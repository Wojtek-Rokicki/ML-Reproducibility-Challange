import numpy as np


def sigmoid(t):
    # compute sigmoid function
    return 1.0 / (1 + np.exp(-t))
