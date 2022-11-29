import numpy as np
from sigmoid import sigmoid

#def nonconvex(w):
#    temp = np.zeros(len(w))
#   for i in len(w):
#       temp[i] = w[i] / (1 + w[i]**2)**2
#
#   return np.sum(temp)

# gradient of the logistic loss function
def gradient(y, tx, w):
    pred = sigmoid(tx.T.dot(w))
    grad = tx.T.dot(pred - y) * (1 / y.shape[0]) #+ 2 * lamb * nonconvex(w)
    return grad