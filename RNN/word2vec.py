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

    def predict(self, x):
        o, _, _ = self.forward(x)
        return np.argmax(o)

    def calculate_total_loss(self, indexed_corpus, vocab_size, window_size):
        total_loss = 0
        for sentence in indexed_corpus:
            for i, target in enumerate(sentence):
                x = np.zeros(self.vocab_size)
                x[target] = 1
                o, _, _ = self.forward(x)

                targets = np.zeros(vocab_size)
                start = max(0, i - window_size)
                end = min(len(sentence), i + window_size + 1)
                for j in range(start, end):
                    if i != j:
                        targets[sentence[j]] = 1

                total_loss += -np.sum(targets * np.log(o + 1e-10))

            return total_loss

    def calculate_loss(self, indexed_corpus, vocab_size, window_size):
        total_loss = self.calculate_total_loss(indexed_corpus, vocab_size, window_size)
        num_targets = sum(len(sentence) for sentence in indexed_corpus)
        average_loss = total_loss / num_targets if num_targets > 0 else 0
        return average_loss

    def backward(self, x, context_words):
        o, h, _ = self.forward(x)

        dLdW1 = np.zeros_like(self.W1)
        dLdW2 = np.zeros_like(self.W2)

        for y in context_words:
            dLdu = o - y
            dLdW2 = np.outer(dLdu, h)
            dLdh = np.dot(self.W2.T, dLdu)
            dLdW1 = np.outer(dLdh, x)

        dLdW1 /= len(context_words)
        dLdW2 /= len(context_words)

        return dLdW1, dLdW2

    def sgd_step(self, x, context_words, learning_rate=0.01):
        dLdW1, dLdW2 = self.backward(x, context_words)
        self.W1 -= learning_rate * dLdW1
        self.W2 -= learning_rate * dLdW2
