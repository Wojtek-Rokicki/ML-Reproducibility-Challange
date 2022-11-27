from src.logistic_regression.sigmoid import sigmoid


def gradient(y, tx, w):
    # gradient of the logistic loss function
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y) * (1 / tx.shape[0])
    return grad
