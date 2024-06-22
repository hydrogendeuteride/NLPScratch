import numpy as np

from function import *
import pathlib
import pickle


class Transformer:
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len, use_gpu=False):
        self.use_gpu = use_gpu and (default_library == 'cupy')
        self.np = cupy if self.use_gpu else numpy

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.max_len = max_len

        self.We = lecun_init((self.vocab_size, self.embed_dim), self.vocab_size, self.np)

        self.Wq = lecun_init((self.num_layers, self.num_heads, self.embed_dim, self.embed_dim // self.num_heads),
                             self.embed_dim, self.np)
        self.Wk = lecun_init((self.num_layers, self.num_heads, self.embed_dim, self.embed_dim // self.num_heads),
                             self.embed_dim, self.np)
        self.Wv = lecun_init((self.num_layers, self.num_heads, self.embed_dim, self.embed_dim // self.num_heads),
                             self.embed_dim, self.np)

        self.Wo = lecun_init((self.num_layers, self.embed_dim, self.embed_dim), self.embed_dim, self.np)

        self.W1 = lecun_init((self.num_layers, self.embed_dim, self.ff_dim), self.embed_dim, self.np)
        self.W2 = lecun_init((self.num_layers, self.ff_dim, self.embed_dim), self.ff_dim, self.np)

        self.b1 = self.np.zeros((self.num_layers, self.ff_dim)).astype(self.np.float32)
        self.b2 = self.np.zeros((self.num_layers, self.embed_dim)).astype(self.np.float32)

        self.pe = positional_encoding(self.max_len, self.embed_dim, self.np)
        self.look_ahead_mask = create_look_ahead_mask(max_len)

    def forward(self, x):
        x = pad_sequence(x, self.max_len)
        padding_mask = create_padding_mask(x)

        x = self.np.array(x)
        H = self.We[x] + self.pe

        cache = {'H': [], 'Q': [], 'K': [], 'V': [], 'attention_weights': [], 'relu_input':[]}

        for l in range(self.num_layers):
            Q = H.dot(self.Wq[l].reshape(self.embed_dim, self.embed_dim))
            K = H.dot(self.Wk[l].reshape(self.embed_dim, self.embed_dim))
            V = H.dot(self.Wv[l].reshape(self.embed_dim, self.embed_dim))

            attention_scores = Q.dot(K.T) / self.np.sqrt(self.embed_dim)
            attention_scores += self.look_ahead_mask
            attention_scores += padding_mask[:, np.newaxis] * -1e9
            attention_weights = softmax(attention_scores)
            attention_output = attention_weights.dot(V)

            multi_head_output = attention_output.dot(self.Wo[l])
            H = layer_norm(H + multi_head_output)

            relu_input = H.dot(self.W1[l]) + self.b1[l]
            ffn_output = relu(relu_input)
            H = layer_norm(H + ffn_output.dot(self.W2[l]) + self.b2[l])

            cache['H'].append(H)
            cache['Q'].append(Q)
            cache['K'].append(K)
            cache['V'].append(V)
            cache['attention_weights'].append(attention_weights)
            cache['relu_input'].append(relu_input)

        O = softmax(H.dot(self.We.T))
        return O, H, cache

    def backward(self, x, y):
        O, H, cache = self.forward(x)

        gradients = {
            'We': self.np.zeros(self.We.shape).astype(self.np.float32),
            'Wq': self.np.zeros(self.Wq.shape).astype(self.np.float32),
            'Wk': self.np.zeros(self.Wk.shape).astype(self.np.float32),
            'Wv': self.np.zeros(self.Wv.shape).astype(self.np.float32),
            'Wo': self.np.zeros(self.Wo.shape).astype(self.np.float32),
            'W1': self.np.zeros(self.W1.shape).astype(self.np.float32),
            'W2': self.np.zeros(self.W2.shape).astype(self.np.float32),
            'b1': self.np.zeros(self.b1.shape).astype(self.np.float32),
            'b2': self.np.zeros(self.b2.shape).astype(self.np.float32),
        }

        dO = O - y
        gradients['We'] = dO.T.dot(H)

        dH = dO.dot(self.We)

        for l in reversed(range(self.num_layers)):
            Q = cache['Q'][l]
            K = cache['K'][l]
            V = cache['V'][l]
            attention_weights = cache['attention_weights'][l]
            H_prev = cache['H'][l - 1] if l > 0 else self.We[x]

            dH_norm_ffn = layer_norm_backward(dH, cache['H'][l])
            dFFN = relu_backward(dH_norm_ffn.dot(self.W2[l].T), cache['relu_input'][l])
            gradients['W2'][l] = cache['relu_input'][l].T.dot(dH_norm_ffn)
            gradients['b2'][l] = dH_norm_ffn.sum(axis=0)

            dFFN_input = dFFN.dot(self.W1[l].T)
            gradients['W1'][l] = H_prev.T.dot(dFFN)
            gradients['b1'][l] = dFFN.sum(axis=0)

            dH = dFFN_input

            dh_norm_mha = layer_norm_backward(dH, cache['H'][l])
            gradients['Wo'][l] = cache['H'][l].T.dot(dh_norm_mha)

            dAttention = dh_norm_mha.dot(self.Wo[l].T)

            dV = attention_weights.T.dot(dAttention)
            gradients['Wv'][l] = cache['H'][l].T.dot(dV).reshape(8, 512, 64)

            dAttention_weights = dAttention.dot(V.T)
            dK = dAttention_weights.T.dot(Q)
            gradients['Wk'][l] = H_prev.T.dot(dK).reshape(8, 512, 64)

            dQ = dAttention_weights.T.dot(K)
            gradients['Wq'][l] = H_prev.T.dot(dQ).reshape(8, 512, 64)

            # dH = dAttention.dot(attention_weights.T)

        return gradients

    def sgd_step(self, x, y, learning_rate=0.01):
        gradients = self.backward(x, y)

        self.We -= learning_rate * gradients['We']
        self.Wq -= learning_rate * gradients['Wq']
        self.Wk -= learning_rate * gradients['Wk']
        self.Wv -= learning_rate * gradients['Wv']
        self.Wo -= learning_rate * gradients['Wo']
        self.W1 -= learning_rate * gradients['W1']
        self.W2 -= learning_rate * gradients['W2']
        self.b1 -= learning_rate * gradients['b1']
        self.b2 -= learning_rate * gradients['b2']

    def calculate_total_loss(self, x, y):
        y_pred, _, _ = self.forward(x)
        y_true_one_hot = self.np.eye(self.vocab_size)[y]

        epsilon = 1e-9
        y_pred = self.np.clip(y_pred, epsilon, 1.0 - epsilon)

        loss = -self.np.sum(y_true_one_hot * self.np.log(y_pred)) / y_pred.shape[0]
        return loss

    def calculate_loss(self, xs, ys):
        total_loss = 0.0
        for x, y in zip(xs, ys):
            total_loss += self.calculate_total_loss(x, y)
        return total_loss / len(xs)

    def save(self, filename):
        weights = {
            'We': self.We,
            'Wq': self.Wq,
            'Wk': self.Wk,
            'Wv': self.Wv,
            'Wo': self.Wo,
            'W1': self.W1,
            'W2': self.W2,
            'b1': self.b1,
            'b2': self.b2,
        }
        if self.use_gpu:
            weights = {k: v.get() for k, v in weights.items()}
        with open(filename, 'wb') as f:
            pickle.dump(weights, f)


def pad_sequence(sequence, max_len, pad_token=0):
    padded_sequence = sequence + [pad_token] * (max_len - len(sequence))
    return padded_sequence


def create_padding_mask(sequence, pad_token=0):
    mask = np.array([1 if token == pad_token else 0 for token in sequence])
    return np.array(mask)


def lecun_init(shape, fan_in, lib=np):
    scale = lib.sqrt(1 / fan_in)
    return lib.random.uniform(-scale, scale, shape).astype(lib.float32)


def create_look_ahead_mask(size):
    mask = np.triu(np.ones((size, size)), k=1).astype('float32')
    return mask * -1e9


def positional_encoding(max_len, embed_dim, np_module):
    pe = np_module.zeros((max_len, embed_dim))
    position = np_module.arange(0, max_len)[:, np_module.newaxis]
    div_term = np_module.exp(np_module.arange(0, embed_dim, 2) * -(np_module.log(10000.0) / embed_dim))
    pe[:, 0::2] = np_module.sin(position * div_term)
    pe[:, 1::2] = np_module.cos(position * div_term)
    return pe.astype(np_module.float32)

