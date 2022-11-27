import numpy

# gradient of the logistic loss function
def gradient(y, tx, w):
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y) * (1 / y.shape[0])
    return grad