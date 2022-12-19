import numpy as np


def non_convex(w):
    # compute the derivative of the nonconvex regularizer
    temp = np.zeros(len(w))
    for i in range(len(w)):
        temp[i] = w[i] / (1 + w[i]**2)**2
    return np.sum(temp)


def grad_sum(grads):
    # compute the sum of gradients for t'th stepsize
    temp = list()
    for s in range(len(grads)):
        temp.append(np.linalg.norm(grads[s])**2)
    return np.sum(temp)

def lip_const(tx, lambda_=None):
    if lambda_ != None:
        L_max = np.linalg.norm(tx, 'fro') ** 2 + lambda_
    else:
        L_max = np.linalg.norm(tx, 'fro')

    return 1 / L_max
