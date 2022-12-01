import numpy as np

# compute sigmoid function
def sigmoid(t):
    return 1.0 / (1 + np.exp(-t))

# compute the derivative of the nonconvex regularizer
def nonconvex(w):
    temp = np.zeros(len(w))
    for i in range(len(w)):
        temp[i] = w[i] / (1 + w[i]**2)**2
    return np.sum(temp)

# gradient of the logistic loss function
def gradient(y, tx, w):
    lamb = 0.5
    pred = sigmoid(np.dot(tx,w))
    grad = tx.T.dot(pred - y) * (1 / y.size) + 2 * lamb * nonconvex(w)
    return grad

# compute stochastic gradient
def sto_grad(y, tx, w, i_t):
    y_sgd = y[i_t]
    x_sgd = np.array(tx[i_t,:])
    return gradient(y_sgd, x_sgd, w)

# compute the sum of gradients for t th stepsize
def grad_sum(grads):
    temp = np.empty(len(grads))
    for s in range(len(grads)):
        temp.append(np.linalg.norm(grads[s])**2)
    return np.sum(temp)
