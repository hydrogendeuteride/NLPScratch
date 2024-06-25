import numpy as np

from utils.function import *
import pathlib
import pickle

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


class LSTM:
    def __init__(self, word_dim, word_embed_dim, tag_dim, hidden_dim=100, params_path=None, bptt_truncate=4,
                 max_norm=5, embedding_weights=None, use_gpu=False):
        self.use_gpu = use_gpu and (default_library == 'cupy')
        self.np = cupy if self.use_gpu else numpy

        self.word_dim = word_dim
        self.word_embed_dim = word_embed_dim
        self.hidden_dim = hidden_dim
        self.tag_dim = tag_dim
        self.bptt_truncate = bptt_truncate
        self.max_norm = max_norm

        if params_path:
            self.load(params_path)
        else:
            if embedding_weights is not None:
                self.E = embedding_weights
            else:
                self.E = self.np.random.uniform(-self.np.sqrt(1. / word_embed_dim), self.np.sqrt(1. / word_embed_dim),
                                            (word_dim, word_embed_dim)).astype(self.np.float32)
            self.Wf = self.np.random.uniform(-self.np.sqrt(1. / hidden_dim), self.np.sqrt(1. / hidden_dim),
                                             (hidden_dim, hidden_dim + word_embed_dim)).astype(self.np.float32)
            self.Wi = self.np.random.uniform(-self.np.sqrt(1. / hidden_dim), self.np.sqrt(1. / hidden_dim),
                                             (hidden_dim, hidden_dim + word_embed_dim)).astype(self.np.float32)
            self.Wo = self.np.random.uniform(-self.np.sqrt(1. / hidden_dim), self.np.sqrt(1. / hidden_dim),
                                             (hidden_dim, hidden_dim + word_embed_dim)).astype(self.np.float32)
            self.Wc = self.np.random.uniform(-self.np.sqrt(1. / hidden_dim), self.np.sqrt(1. / hidden_dim),
                                             (hidden_dim, hidden_dim + word_embed_dim)).astype(self.np.float32)

            self.V = self.np.random.uniform(-self.np.sqrt(1. / hidden_dim), self.np.sqrt(1. / hidden_dim),
                                            (tag_dim, hidden_dim)).astype(self.np.float32)

    def forward(self, x):
        T = len(x)
        x = self.np.array(x)
        s = self.np.zeros((T + 1, self.hidden_dim)).astype(self.np.float32)
        s[-1] = self.np.zeros(self.hidden_dim)

        c = self.np.zeros((T + 1, self.hidden_dim)).astype(self.np.float32)
        c[-1] = self.np.zeros(self.hidden_dim)

        o = self.np.zeros((T, self.tag_dim)).astype(self.np.float32)

        for t in range(T):
            x_t = self.E[x[t]]
            z = self.np.concatenate([s[t - 1], x_t]).astype(self.np.float32)
            f = sigmoid(self.np.dot(self.Wf, z))
            i = sigmoid(self.np.dot(self.Wi, z))
            o_ = sigmoid(self.np.dot(self.Wo, z))
            c_tilde = self.np.tanh(self.np.dot(self.Wc, z))

            c[t] = f * c[t - 1] + i * c_tilde
            s[t] = o_ * self.np.tanh(c[t])
            o[t] = softmax(self.np.dot(self.V, s[t]))

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
        o, s, c = self.forward(x)

        dLdWf = self.np.zeros(self.Wf.shape).astype(self.np.float32)
        dLdWi = self.np.zeros(self.Wi.shape).astype(self.np.float32)
        dLdWo = self.np.zeros(self.Wo.shape).astype(self.np.float32)
        dLdWc = self.np.zeros(self.Wc.shape).astype(self.np.float32)
        dLdV = self.np.zeros(self.V.shape).astype(self.np.float32)
        dLdE = self.np.zeros(self.E.shape).astype(self.np.float32)

        delta_o = o  # output vector grad
        delta_o[range(len(y)), y] -= 1

        delta_c = self.np.zeros(c[0].shape).astype(self.np.float32)  # memory cell grad

        for t in range(T)[::-1]:
            dLdV += self.np.outer(delta_o[t], s[t].T)

            if t > 0:
                z = self.np.concatenate([s[t - 1], self.E[x[t]]]).astype(self.np.float32)
            else:
                z = self.np.concatenate([self.np.zeros(self.hidden_dim), self.E[x[t]]]).astype(self.np.float32)

            delta_o_ = self.np.dot(self.V.T, delta_o[t]) * sigmoid(self.np.dot(self.Wo, z)) * \
                       (1 - sigmoid(self.np.dot(self.Wo, z)))  # output gate grad

            delta_s = (self.np.dot(self.V.T, delta_o[t]) +
                       self.np.dot(self.Wo.T[:self.hidden_dim, :], delta_o_[:self.hidden_dim]) *
                       (1 - self.np.tanh(c[t]) ** 2))  # hidden state grad

            delta_c += (delta_s * sigmoid(self.np.dot(self.Wo, z)) * (1 - self.np.tanh(c[t]) ** 2))
            delta_c_tilde = delta_c * sigmoid(self.np.dot(self.Wi, z))

            delta_f = delta_c * c[t - 1] if t > 0 else delta_c * self.np.zeros(c[t].shape)
            delta_i = delta_c * self.np.tanh(self.np.dot(self.Wc, z))
            dLdWf += self.np.outer(delta_f, z)
            dLdWi += self.np.outer(delta_i, z)
            dLdWo += self.np.outer(delta_o_, z)
            dLdWc += self.np.outer(delta_c_tilde, z)

            dLdE[x[t]] += self.np.dot(self.Wf[:, self.hidden_dim:].T, delta_f)
            dLdE[x[t]] += self.np.dot(self.Wi[:, self.hidden_dim:].T, delta_i)
            dLdE[x[t]] += self.np.dot(self.Wo[:, self.hidden_dim:].T, delta_o_)
            dLdE[x[t]] += self.np.dot(self.Wc[:, self.hidden_dim:].T, delta_c_tilde)

            if t > max(0, t - self.bptt_truncate):
                delta_c = self.np.dot(self.Wf.T, delta_f)[:self.hidden_dim] * sigmoid(self.np.dot(self.Wf, z)) * (
                        1 - sigmoid(self.np.dot(self.Wf, z)))
                delta_c += self.np.dot(self.Wi.T, delta_i)[:self.hidden_dim] * sigmoid(self.np.dot(self.Wi, z)) * (
                        1 - sigmoid(self.np.dot(self.Wi, z)))
                delta_c += self.np.dot(self.Wo.T, delta_o_)[:self.hidden_dim] * sigmoid(self.np.dot(self.Wo, z)) * (
                        1 - sigmoid(self.np.dot(self.Wo, z)))
                delta_c += self.np.dot(self.Wc.T, delta_c_tilde)[:self.hidden_dim] * (
                        1 - self.np.tanh(self.np.dot(self.Wc, z)) ** 2)
            else:
                delta_c = self.np.zeros(c[0].shape)

        return [dLdWf, dLdWi, dLdWo, dLdWc, dLdV, dLdE]

    def sgd_step(self, x, y, learning_rate=0.01):
        gradients = self.backward(x, y)

        clip_grads(gradients, self.max_norm)

        self.Wf -= learning_rate * gradients[0]
        self.Wi -= learning_rate * gradients[1]
        self.Wo -= learning_rate * gradients[2]
        self.Wc -= learning_rate * gradients[3]
        self.V -= learning_rate * gradients[4]
        self.E -= learning_rate * gradients[5]

    def save(self, file_path):
        parameters = {
            'Wf': self.Wf.get() if self.use_gpu else self.Wf,
            'Wi': self.Wi.get() if self.use_gpu else self.Wi,
            'Wo': self.Wo.get() if self.use_gpu else self.Wo,
            'Wc': self.Wc.get() if self.use_gpu else self.Wc,
            'E': self.E.get() if self.use_gpu else self.E,
            'V': self.V.get() if self.use_gpu else self.V
        }
        with open(file_path, 'wb') as f:
            pickle.dump(parameters, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            parameters = pickle.load(f)
        self.Wf = self.np.array(parameters['Wf']).astype(self.np.float32)
        self.Wi = self.np.array(parameters['Wi']).astype(self.np.float32)
        self.Wo = self.np.array(parameters['Wo']).astype(self.np.float32)
        self.Wc = self.np.array(parameters['Wc']).astype(self.np.float32)
        self.E = self.np.array(parameters['E']).astype(self.np.float32)
        self.V = self.np.array(parameters['V']).astype(self.np.float32)
