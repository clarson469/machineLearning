import sys
import numpy as np
from scipy.special import expit
import settings

def init_params(sizes):
    thetas = [np.random.randn(*size) for size in sizes]
    return np.hstack(([t.flatten() for t in thetas]))

def unwrap(params, sizes):
    thetas = []
    start, end = 0, 0
    for size in sizes:
        end += np.prod(size)
        thetas.append(params[start : end].reshape(*size))
        start += np.prod(size)

    return thetas

def sigmoid(z):
    return expit(z)

def sig_prime(z):
    return expit(z) * (1 - expit(z))

def add_intercept(A):
    intercept = np.ones((A.shape[0], 1))
    return np.hstack((intercept, A))

class IterCount(object):
    def __init__(self, msg, auto=False):
        self.msg = msg + ' {}'
        if auto:
            self.auto = auto
            self.count = 0

    def update(self, i):
        sys.stdout.write(self.msg.format(i))
        sys.stdout.write('\b' * (len(self.msg) + 4))
        sys.stdout.flush()

    def auto_update(self):
        self.count += 1
        self.update(self.count)
