from src.logistic_regression.gradient import gradient
import numpy as np


def sgd(y, tx, w):
    sample_index = np.random.randint(0, len(y))
    y_sgd = y[sample_index]
    x_sgd = np.array(tx[sample_index, :])
    return gradient(y_sgd, x_sgd, w)
