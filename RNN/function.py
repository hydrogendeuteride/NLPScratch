import numpy as np


def softmax(x):
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=1, keepdims=True)
        return x_exp / x_sum
    else:
        x = x - np.max(x)
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp)
        return x_exp / x_sum
