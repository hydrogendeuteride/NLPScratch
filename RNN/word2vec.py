from function import *
import numpy as np
import pickle


class SkipGram:
    def __init__(self, vocab_size, embedding_dim, params_path=None, use_gpu=False):
        self.use_gpu = use_gpu and (default_library == 'cupy')
        self.np = cupy if self.use_gpu else numpy

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        if params_path:
            self.load(params_path)
        else:
            self.W1 = self.np.random.uniform(-np.sqrt(1.0 / vocab_size), np.sqrt(1.0 / vocab_size),
                                             (embedding_dim, vocab_size)).astype(np.float32)
            self.W2 = self.np.random.uniform(-np.sqrt(1.0 / embedding_dim), np.sqrt(1.0 / embedding_dim),
                                             (vocab_size, embedding_dim)).astype(np.float32)

    def forward(self, x):
        h = self.np.dot(self.W1, x)
        u = self.np.dot(self.W2, h)
        y = softmax(u)
        return y, h, u

    def predict(self, x):
        o, _, _ = self.forward(x)
        return np.argmax(o)

    def calculate_total_loss(self, indexed_corpus, vocab_size, window_size):
        total_loss = 0
        for sentence in indexed_corpus:
            for i, target in enumerate(sentence):
                x = self.np.zeros(self.vocab_size)
                x[target] = 1
                o, _, _ = self.forward(x)

                targets = self.np.zeros(vocab_size)
                start = max(0, i - window_size)
                end = min(len(sentence), i + window_size + 1)
                for j in range(start, end):
                    if i != j:
                        targets[sentence[j]] = 1

                total_loss += -self.np.sum(targets * np.log(o + 1e-10))

            return total_loss

    def calculate_loss(self, indexed_corpus, vocab_size, window_size):
        total_loss = self.calculate_total_loss(indexed_corpus, vocab_size, window_size)
        num_targets = sum(len(sentence) for sentence in indexed_corpus)
        average_loss = total_loss / num_targets if num_targets > 0 else 0
        return average_loss

    def backward(self, x, context_words):
        o, h, _ = self.forward(x)

        dLdW1 = self.np.zeros_like(self.W1)
        dLdW2 = self.np.zeros_like(self.W2)

        EI = self.np.sum(self.np.array([o - word for word in context_words]), axis=0)

        dLdW2 = self.np.outer(EI, h)
        dLdh = self.np.dot(self.W2.T, EI)
        dLdW1 = self.np.outer(dLdh, x)

        dLdW1 /= len(context_words)
        dLdW2 /= len(context_words)

        return dLdW1, dLdW2

    def sgd_step(self, x, context_words, learning_rate=0.01):
        dLdW1, dLdW2 = self.backward(x, context_words)
        self.W1 -= learning_rate * dLdW1
        self.W2 -= learning_rate * dLdW2

    def save(self, file_path):
        parameters = {
            'W1': self.W1.get() if self.use_gpu else self.W1,
            'W2': self.W2.get() if self.use_gpu else self.W2,
        }
        with open(file_path, 'wb') as f:
            pickle.dump(parameters, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            parameters = pickle.load(f)
        self.W1 = self.np.array(parameters['W1']).astype(self.np.float32)
        self.W2 = self.np.array(parameters['W2']).astype(self.np.float32)
