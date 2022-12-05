from src.logistic_regression.sigmoid import sigmoid

from src.utils.method_utils import non_convex


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
    grad = tx.T.dot(pred - y) * (1 / y.size) + 2 * lamb * non_convex(w)
    return grad
