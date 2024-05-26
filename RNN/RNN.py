from function import *
import pathlib
import pickle

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


class RNN:
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
            self.E = self.np.random.uniform(-self.np.sqrt(1. / word_dim), self.np.sqrt(1. / word_dim), (word_dim, word_dim))
            self.U = self.np.random.uniform(-self.np.sqrt(1. / word_dim), self.np.sqrt(1. / word_dim), (hidden_dim, word_dim))
            self.V = self.np.random.uniform(-self.np.sqrt(1. / hidden_dim), self.np.sqrt(1. / hidden_dim), (tag_dim, hidden_dim))
            self.W = self.np.random.uniform(-self.np.sqrt(1. / hidden_dim), self.np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))

    def forward(self, x):
        T = len(x)
        x = self.np.array(x)
        s = self.np.zeros((T + 1, self.hidden_dim))
        s[-1] = self.np.zeros(self.hidden_dim)

        o = self.np.zeros((T, self.tag_dim))

        for t in self.np.arange(T):
            x_t = self.E[:, x[t]]
            s[t] = self.np.tanh(self.U.dot(x_t) + self.W.dot(s[t - 1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]

    def predict(self, x):
        o, s = self.forward(x)
        return self.np.argmax(o, axis=1)

    def calculate_total_loss(self, x, y):
        L = 0

        for i in range(len(y)):
            o, s = self.forward(x[i])
            correct_word_predictions = o[self.np.arange(len(y[i])), y[i]]
            L += -1 * self.np.sum(self.np.log(correct_word_predictions))
        return L

    def calculate_loss(self, x, y):
        N = sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y) / N

    def backward(self, x, y):
        T = len(x)
        o, s = self.forward(x)

        dLdU = self.np.zeros(self.U.shape)
        dLdV = self.np.zeros(self.V.shape)
        dLdW = self.np.zeros(self.W.shape)
        dLdE = self.np.zeros(self.E.shape)

        delta_o = o
        delta_o[range(len(y)), y] -= 1

        for t in range(T)[::-1]:
            dLdV += self.np.outer(delta_o[t], s[t].T)
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            for bptt_step in range(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                dLdW += self.np.outer(delta_t, s[bptt_step - 1])
                dLdU[:, x[bptt_step]] += delta_t
                dLdE[:, x[bptt_step]] += self.U.T.dot(delta_t)
                delta_t = self.W.T.dot(delta_t) * (1 - (s[bptt_step] ** 2))

        return [dLdU, dLdV, dLdW, dLdE]

    def sgd_step(self, x, y, learning_rate=0.01):
        dLdU, dLdV, dLdW, dLdE = self.backward(x, y)

        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW
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
