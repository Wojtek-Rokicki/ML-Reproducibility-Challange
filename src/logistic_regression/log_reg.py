import numpy as np

from src.logistic_regression.sigmoid import sigmoid

from src.utils.method_utils import non_convex

LAMBDA = 0.5


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
    lamb = 0.5
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y) * (1 / y.size) + 2 * LAMBDA * non_convex(w)
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
    cost = sum(first_component - second_component) / N + (LAMBDA * non_convex(w)**2)
    return cost
    # pred = sigmoid(tx.dot(w))
    # loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    # return np.squeeze(-loss).item() * (1 / y.shape[0])
