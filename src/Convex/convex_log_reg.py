import numpy as np

from src.logistic_regression.sigmoid import sigmoid

def log_reg_gradient(y, tx, w):
    """
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1)
    """
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y) * (1 / y.size)
    return grad


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss
    """

    N = len(y)
    first_component = np.log(1 + np.exp(tx.dot(w)))
    second_component = y * (tx.dot(w))
    cost = sum(first_component - second_component) / N 
    return cost