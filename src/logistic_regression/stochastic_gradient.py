from src.logistic_regression.log_reg_gradient import log_reg_gradient
import numpy as np


def stochastic_gradient(y, tx, w, sample_indices: np.array):
    if sample_indices is None:
        sample_indices = np.random.choice(range(0, len(y)), size=1, replace=False)
    y_sgd = y[sample_indices]
    x_sgd = np.array(tx[sample_indices, :])
    return log_reg_gradient(y_sgd, x_sgd, w)
