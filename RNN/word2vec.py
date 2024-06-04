from function import *
import numpy as np


class SkipGram:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.W1 = np.random.randn(embedding_dim, vocab_size) * 0.01
        self.W2 = np.random.randn(vocab_size, embedding_dim) * 0.01

    def forward(self, x):
        h = np.dot(self.W1, x)
        u = np.dot(self.W2, h)
        y = softmax(u)
        return y, h, u

    def backward(self, x, y):
        o, h, _ = self.forward(x)

        dLdu = o - y
        dLdW1 = np.outer(dLdu, h)
        dLdh = np.dot(self.W2.T, dLdu)
        dLdW2 = np.zeros(dLdh, x)

        return dLdW1, dLdW2
