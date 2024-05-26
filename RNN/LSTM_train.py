import numpy as np

from function import *
import pathlib
import pickle

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


class LSTM:
    def __init__(self, word_dim, tag_dim, hidden_dim=100, bptt_truncate=4, params_path=None, use_gpu=False):
        self.use_gpu = use_gpu and (default_library == 'cupy')
        self.np = cupy if self.use_gpu else numpy

        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.tag_dim = tag_dim
        self.bptt_truncate = bptt_truncate

        if params_path:
            self.load(params_path)
        else:
            self.E = self.np.random.uniform(-self.np.sqrt(1. / word_dim), self.np.sqrt(1. / word_dim),
                                            (word_dim, word_dim))
            self.Wf = self.np.random.uniform(-self.np.sqrt(1. / hidden_dim), self.np.sqrt(1. / hidden_dim),
                                             (hidden_dim, hidden_dim + word_dim))
            self.Wi = self.np.random.uniform(-self.np.sqrt(1. / hidden_dim), self.np.sqrt(1. / hidden_dim),
                                             (hidden_dim, hidden_dim + word_dim))
            self.Wo = self.np.random.uniform(-self.np.sqrt(1. / hidden_dim), self.np.sqrt(1. / hidden_dim),
                                            (hidden_dim, hidden_dim + word_dim))
            self.Wc = self.np.random.uniform(-self.np.sqrt(1. / hidden_dim), self.np.sqrt(1. / hidden_dim),
                                             (hidden_dim, hidden_dim + word_dim))

            self.V = self.np.random.uniform(-self.np.sqrt(1. / hidden_dim), self.np.sqrt(1. / hidden_dim), (tag_dim, hidden_dim))

    def forward(self, x):
        T = len(x)
        x = self.np.array(x)
        s = self.np.zeros((T + 1, self.hidden_dim))
        s[-1] = self.np.zeros(self.hidden_dim)

        c = self.np.zeros((T + 1, self.hidden_dim))
        c[-1] = self.np.zeros(self.hidden_dim)

        o = self.np.zeros((T, self.tag_dim))

        for t in range(T):
            x_t = self.E[:, x[t]]
            z = self.np.concatenate([s[t-1], x_t])
            f = sigmoid(self.np.dot(self.Wf, z))
            i = sigmoid(self.np.dot(self.Wi, z))
            o_ = sigmoid(self.np.dot(self.Wo, z))
            c_tilde = self.np.tanh(self.np.dot(self.Wc, z))

            c[t] = f * c[t - 1] + i * c_tilde
            s[t] = o_ * np.tanh(c[t])
            o[t] = softmax(np.dot(self.V, s[t]))

        return [o, s, c]

    def predict(self, x):
        o, _, _ = self.forward(x)
        return self.np.argmax(o, axis=1)

    def calculate_total_loss(self, x, y):
        L = 0

        for i in range(len(y)):
            o, _, _ = self.forward(x[i])
            correct_word_predictions = o[self.np.arange(len(y[i])), y[i]]
            L += -1 * self.np.sum(self.np.log(correct_word_predictions))
        return L

    def calculate_loss(self, x, y):
        N = sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y) / N

    def backward(self, x, y):
        T = len(x)
        o, s, c= self.forward(x)

        dLdWf = self.np.zeros(self.Wf.shape)
        dLdWi = self.np.zeros(self.Wi.shape)
        dLdWo = self.np.zeros(self.Wo.shape)
        dLdWc = self.np.zeros(self.Wc.shape)
        dLdV = self.np.zeros(self.V.shape)
        dLdE = self.np.zeros(self.E.shape)

        delta_o = o
        delta_o[range(len(y)), y] -= 1

        delta_c = self.np.zeros(c[0].shape)

        for t in range(T)[::-1]:
            dLdV += self.np.outer(delta_o[t], s[t].T)

            z = self.np.concatenate([s[t - 1], self.E[:, x[t]]])

            delta_s = (self.np.dot(self.V.T, delta_o[t]) * (1 - self.np.tanh(c[t]) ** 2) *
                       sigmoid(self.np.dot(self.Wo, delta_o[t]) * z))
            delta_o_ = delta_s * self.np.tanh(c[t])

            delta_c += (delta_s * sigmoid(self.np.dot(self.Wo, z)) * (1 - self.np.tanh(c[t]) ** 2))
            delta_c_tilde = delta_c * sigmoid(self.np.dot(self.Wi, z))

            delta_f = delta_c * c[t - 1]
            delta_i = delta_c * self.np.tanh(self.np.dot(self.Wc, z))

            dLdWf += self.np.outer(delta_f, z)
            dLdWi += self.np.outer(delta_i, z)
            dLdWo += self.np.outer(delta_c, z)
            dLdWc += self.np.outer(delta_c_tilde, z)

            dLdE[:, x[t]] += self.np.dot(self.Wf[:, self.hidden_dim:].T, delta_f)
            dLdE[:, x[t]] += self.np.dot(self.Wi[:, self.hidden_dim:].T, delta_i)
            dLdE[:, x[t]] += self.np.dot(self.Wo[:, self.hidden_dim:].T, delta_o_)
            dLdE[:, x[t]] += self.np.dot(self.Wc[:, self.hidden_dim:].T, delta_c_tilde)

            delta_c = self.np.dot(self.Wf.T, delta_f)[:, :self.hidden_dim] * sigmoid(self.np.dot(self.Wf, z)) * (
                        1 - sigmoid(self.np.dot(self.Wf, z)))
            delta_c += self.np.dot(self.Wi.T, delta_i)[:, :self.hidden_dim] * sigmoid(self.np.dot(self.Wi, z)) * (
                        1 - sigmoid(self.np.dot(self.Wi, z)))
            delta_c += self.np.dot(self.Wo.T, delta_o_)[:, :self.hidden_dim] * sigmoid(self.np.dot(self.Wo, z)) * (
                        1 - sigmoid(self.np.dot(self.Wo, z)))
            delta_c += self.np.dot(self.Wc.T, delta_c_tilde)[:, :self.hidden_dim] * (
                        1 - self.np.tanh(self.np.dot(self.Wc, z)) ** 2)

        return [dLdWf, dLdWi, dLdWo, dLdWc, dLdV, dLdE]

    def sgd_step(self, x, y, learning_rate=0.01):
        dLdWf, dLdWi, dLdWo, dLdWc, dLdV, dLdE = self.backward(x, y)

        self.Wf -= learning_rate * dLdWf
        self.Wi -= learning_rate * dLdWi
        self.Wo -= learning_rate * dLdWo
        self.Wc -= learning_rate * dLdWc
        self.V -= learning_rate * dLdV
        self.E -= learning_rate * dLdE

    def save(self, file_path):
        parameters = {
            'E': self.E.get() if self.use_gpu else self.E,
            'U': self.U.get() if self.use_gpu else self.U,
            'V': self.V.get() if self.use_gpu else self.V,
            'W': self.W.get() if self.use_gpu else self.W
        }
        with open(file_path, 'wb') as f:
            pickle.dump(parameters, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            parameters = pickle.load(f)
        self.E = self.np.array(parameters['E'])
        self.U = self.np.array(parameters['U'])
        self.V = self.np.array(parameters['V'])
        self.W = self.np.array(parameters['W'])