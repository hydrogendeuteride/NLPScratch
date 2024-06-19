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

        self.We = self.np.random.uniform(-0.1, 0.1, (self.vocab_size, self.embed_dim)).astype(self.np.float32)

        self.Wq = self.np.random.uniform(-0.1, 0.1, (
            self.num_layers, self.num_heads, self.embed_dim, self.embed_dim // self.num_heads)).astype(self.np.float32)
        self.Wk = self.np.random.uniform(-0.1, 0.1, (
            self.num_layers, self.num_heads, self.embed_dim, self.embed_dim // self.num_heads)).astype(self.np.float32)
        self.Wv = self.np.random.uniform(-0.1, 0.1, (
            self.num_layers, self.num_heads, self.embed_dim, self.embed_dim // self.num_heads)).astype(self.np.float32)

        self.Wo = self.np.random.uniform(-0.1, 0.1, (self.num_layers, self.embed_dim, self.embed_dim)).astype(
            self.np.float32)

        self.W1 = self.np.random.uniform(-0.1, 0.1, (self.num_layers, self.embed_dim, self.ff_dim)).astype(
            self.np.float32)
        self.W2 = self.np.random.uniform(-0.1, 0.1, (self.num_layers, self.ff_dim, self.embed_dim)).astype(
            self.np.float32)
        self.b1 = self.np.zeros((self.num_layers, self.ff_dim)).astype(self.np.float32)
        self.b2 = self.np.zeros((self.num_layers, self.embed_dim)).astype(self.np.float32)

    def forward(self, x):
        x = pad_sequence(x, self.max_len)
        padding_mask = create_padding_mask(x)

        x = self.np.array(x)
        H = self.We[x]

        cache = {'H': [], 'Q': [], 'K': [], 'V': [], 'attention_weights': []}

        for l in range(self.num_layers):
            Q = H.dot(self.Wq[l].reshape(self.embed_dim, self.embed_dim))
            K = H.dot(self.Wk[l].reshape(self.embed_dim, self.embed_dim))
            V = H.dot(self.Wv[l].reshape(self.embed_dim, self.embed_dim))

            attention_scores = Q.dot(K.T) / np.sqrt(self.embed_dim)
            attention_scores = attention_scores + (padding_mask * -1e9)
            attention_weights = softmax(attention_scores)
            attention_output = attention_weights.dot(V)

            multi_head_output = attention_output.dot(self.Wo[l])
            H = layer_norm(H + multi_head_output)

            ffn_output = relu(H.dot(self.W1[l]) + self.b1[l])
            H = layer_norm(H + ffn_output.dot(self.W2[l]) + self.b2[l])

            cache['H'].append(H)
            cache['Q'].append(Q)
            cache['K'].append(K)
            cache['V'].append(V)
            cache['attention_weights'].append(attention_weights)

        O = softmax(H.dot(self.We.T))
        return O, H, cache

    def backward(self, x, y):
        T = len(x)
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
        gradients['We'] = dO.T.dot(H).T

        dH = dO.dot(self.We)

        for l in reversed(range(self.num_layers)):
            Q = cache['Q'][l]
            K = cache['K'][l]
            V = cache['V'][l]
            attention_weights = cache['attention_weights'][l]
            H_prev = cache['H'][l - 1] if l > 0 else self.We[x]

            dH_norm_ffn = layer_norm_backward(dH, cache['H'][l])
            dFFN = relu_backward(dH_norm_ffn.dot(self.W2[l].T), cache['H'][l])
            gradients['W2'][l] = cache['H'][l].T.dot(dH_norm_ffn)
            gradients['b2'][l] = dH_norm_ffn.sum(axis=0)

            dFFN_input = dFFN.dot(self.W1[l].T)
            gradients['W1'][l] = H_prev.T.dot(dFFN)
            gradients['b1'][l] = dFFN.sum(axis=0)

            dH = dFFN_input

            dh_norm_mha = layer_norm_backward(dH, cache['H'][l])
            gradients['Wo'][l] = dh_norm_mha.T.dot(cache['H'][l])

            dAttention = dh_norm_mha.dot(self.Wo[l].T)

            dV = dAttention.dot(attention_weights.T)
            gradients['Wv'][l] = cache['H'][l].T.dot(dV)

            dK = dAttention.T.dot(Q)
            gradients['Wk'][l] = H_prev.T.dot(dK)

            dQ = dAttention.T.dot(K)
            gradients['Wq'][l] = H_prev.T.dot(dQ)

            dH = dAttention.dot(attention_weights.T)

        return gradients


def pad_sequence(sequence, max_len, pad_token=0):
    padded_sequence = sequence + [pad_token] * (max_len - len(sequence))
    return padded_sequence


def create_padding_mask(sequence, pad_token=0):
    mask = [1 if token == pad_token else 0 for token in sequence]
    return np.array(mask)
