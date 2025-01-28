import numpy as np

from utils.function import *
import pathlib
import pickle


class Transformer:
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len,
                 embedding_weight=None, use_gpu=False):
        self.use_gpu = use_gpu and (default_library == 'cupy')
        self.np = cupy if self.use_gpu else numpy

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.max_len = max_len
        self.head_dim = embed_dim // num_heads

        if embedding_weight is None:
            self.We = lecun_init((self.vocab_size, self.embed_dim), self.embed_dim, self.np)
        else:
            self.We = self.np.array(embedding_weight) if self.use_gpu else embedding_weight

        self.Wq = lecun_init((self.num_layers, self.embed_dim, self.embed_dim),
                             self.embed_dim, self.np)
        self.Wk = lecun_init((self.num_layers, self.embed_dim, self.embed_dim),
                             self.embed_dim, self.np)
        self.Wv = lecun_init((self.num_layers, self.embed_dim, self.embed_dim),
                             self.embed_dim, self.np)

        self.Wo = lecun_init((self.num_layers, self.embed_dim, self.embed_dim), self.embed_dim, self.np)

        self.W1 = lecun_init((self.num_layers, self.embed_dim, self.ff_dim), self.embed_dim, self.np)
        self.W2 = lecun_init((self.num_layers, self.ff_dim, self.embed_dim), self.ff_dim, self.np)

        self.b1 = self.np.zeros((self.num_layers, self.ff_dim)).astype(self.np.float32)
        self.b2 = self.np.zeros((self.num_layers, self.embed_dim)).astype(self.np.float32)

        self.W_vocab = lecun_init((embed_dim, vocab_size), fan_in=embed_dim)
        self.b_vocab = self.np.zeros((vocab_size,), dtype=self.np.float32)

        self.pe = positional_encoding(self.max_len, self.embed_dim, self.np)

    def forward(self, x):
        # x : (B, S)
        B, S = x.shape

        H = self.We[x] + self.pe[:S]  # (B, S, E)

        padding_mask = create_padding_mask(x, lib=self.np)  # (B, 1, 1, S)
        look_ahead = create_look_ahead_mask(S)  # (1, 1, S, S)
        combined_mask = look_ahead + padding_mask  # (B, 1, S, S)

        layers_cache = []

        for l in range(self.num_layers):
            layer_c = {}

            attn_input = H

            Q = attn_input @ self.Wq[l]  # (B, S, E)
            K = attn_input @ self.Wk[l]  # (B, S, E)
            V = attn_input @ self.Wv[l]  # (B, S, E)

            Q_4d = Q.reshape(B, S, self.num_heads, -1).transpose(0, 2, 1, 3)  # (B,h,S,d)
            K_4d = K.reshape(B, S, self.num_heads, -1).transpose(0, 2, 1, 3)
            V_4d = V.reshape(B, S, self.num_heads, -1).transpose(0, 2, 1, 3)

            attn_scores = Q_4d @ K_4d.transpose(0, 1, 3, 2) / self.np.sqrt(self.head_dim)  # (B, h, S, S)
            attn_scores += combined_mask

            attn_weights = softmax(attn_scores, axis=-1)  # (B, h, S, S)

            attn_out_4d = attn_weights @ V_4d
            attn_out = attn_out_4d.transpose(0, 2, 1, 3).reshape(B, S, self.embed_dim)
            attn_out = attn_out @ self.Wo[l]

            attn_output = layer_norm(attn_input + attn_out)

            ffn_input = attn_output
            ffn_hidden = relu(ffn_input @ self.W1[l] + self.b1[l])
            ffn_out = ffn_hidden @ self.W2[l] + self.b2[l]

            H_new = layer_norm(ffn_input + ffn_out)

            layer_c['attn_input'] = attn_input
            layer_c['Q'] = Q_4d
            layer_c['K'] = K_4d
            layer_c['V'] = V_4d
            layer_c['scores'] = attn_scores
            layer_c['attn_weights'] = attn_weights
            layer_c['attn_out_beforeWo'] = attn_out_4d
            layer_c['attn_out'] = attn_out
            layer_c['attn_output'] = attn_output

            layer_c['ffn_input'] = ffn_input
            layer_c['ffn_hidden'] = ffn_hidden
            layer_c['ffn_out'] = ffn_out

            layers_cache.append(layer_c)
            H = H_new

        logits = H @ self.W_vocab + self.b_vocab  # (B, S, vocab_size)

        cache = {
            'x': x,
            'H_final': H,
            'layers': layers_cache
        }

        return logits, cache

    def backward(self, cache, dlogits):
        """
        dlogits: (B, S, vocab_size) = ∂Loss/∂logits
        """
        B, S, V = dlogits.shape

        gradients = {
            'We': self.np.zeros_like(self.We),
            'Wq': self.np.zeros_like(self.Wq),
            'Wk': self.np.zeros_like(self.Wk),
            'Wv': self.np.zeros_like(self.Wv),
            'Wo': self.np.zeros_like(self.Wo),
            'W1': self.np.zeros_like(self.W1),
            'W2': self.np.zeros_like(self.W2),
            'b1': self.np.zeros_like(self.b1),
            'b2': self.np.zeros_like(self.b2),
            'W_vocab': self.np.zeros_like(self.W_vocab),
            'b_vocab': self.np.zeros_like(self.b_vocab)
        }

        H_final = cache['H_final']  # (B, S, E)
        E = self.embed_dim
        H_2d = H_final.reshape(B * S, E)  # (N, E)
        dlogits_2d = dlogits.reshape(B * S, V)  # (N, V)

        gradients['W_vocab'] += H_2d.T @ dlogits_2d  # (E,V)
        gradients['b_vocab'] += self.np.sum(dlogits_2d, axis=0)  # (V)

        dH_final_2d = dlogits_2d @ self.W_vocab.T  # (N, E)
        dH = dH_final_2d.reshape(B, S, E)

        layers = cache['layers']

        for l in reversed(range(self.num_layers)):
            layer_c = layers[l]

            ffn_input = layer_c['ffn_input']  # (B, S, E)
            ffn_out = layer_c['ffn_out']  # (B, S ,E)
            ffn_hidden = layer_c['ffn_hidden']  # (B, S, ff_dim)

            residual_in = ffn_input + ffn_out  # (B ,S ,E)
            d_residual_in = layer_norm_backward(dH, residual_in)

            d_ffn_out = d_residual_in  # (B, S, E)

            gradients['W2'][l] += self.np.einsum('bse,bsf->ef', ffn_hidden, d_ffn_out)
            gradients['b2'][l] += self.np.sum(d_ffn_out, axis=(0, 1))

            d_ffn_hidden = d_ffn_out @ self.W2[l].T  # (B, S, ff_dim)

            z = ffn_input @ self.W1[l] + self.b1[l]  # (B, S, ff_dim)
            dz = relu_backward(d_ffn_hidden, z)

            gradients['W1'][l] += self.np.einsum('bse,bsf->ef', ffn_input, dz)
            gradients['b1'][l] += self.np.sum(dz, axis=(0, 1))

            d_ffn_input = dz @ self.W1[l].T  # (B, S ,E)

            dH_attn_out = d_residual_in
            dH = d_ffn_input + dH_attn_out

            attn_input = layer_c['attn_input']  # (B,S,E)
            attn_out = layer_c['attn_out']  # (B,S,E)
            attn_out_4d = layer_c['attn_out_beforeWo']  # (B,h,S,d)
            attn_weights = layer_c['attn_weights']  # (B,h,S,S)
            scores = layer_c['scores']  # (B,h,S,S)
            Q_4d = layer_c['Q']  # (B,h,S,d)
            K_4d = layer_c['K']  # (B,h,S,d)
            V_4d = layer_c['V']  # (B,h,S,d)

            attn_residual_in = attn_input + attn_out
            d_attn_res_in = layer_norm_backward(dH, attn_residual_in)

            d_attn_out = d_attn_res_in

            dWo_in = d_attn_out  # (B,S,E)
            attn_out_noWo_2d = attn_out_4d.transpose(0, 2, 1, 3).reshape(B * S, self.embed_dim)
            dWo_in_2d = dWo_in.reshape(B * S, self.embed_dim)

            gradients['Wo'][l] += attn_out_noWo_2d.T @ dWo_in_2d  # (E, E)

            d_attn_out_noWo_2d = dWo_in_2d @ self.Wo[l].T  # (B*S,E)
            d_attn_out_noWo_4d = d_attn_out_noWo_2d.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

            d_attn_weight = d_attn_out_noWo_4d @ V_4d.transpose(0, 1, 3, 2)
            dV_4d = attn_weights.transpose(0, 1, 3, 2) @ d_attn_out_noWo_4d

            d_attn_score = softmax_backward(d_attn_weight, attn_weights)  # (B, h, S, S)

            scale = 1.0 / self.np.sqrt(self.head_dim)
            dQ_4d = scale * (d_attn_score @ K_4d)
            dK_4d = scale * (d_attn_score.transpose(0, 1, 3, 2) @ Q_4d)

            dQ = dQ_4d.transpose(0, 2, 1, 3).reshape(B, S, self.embed_dim)
            dK = dK_4d.transpose(0, 2, 1, 3).reshape(B, S, self.embed_dim)
            dV = dV_4d.transpose(0, 2, 1, 3).reshape(B, S, self.embed_dim)

            d_attn_input = (dQ @ self.Wq[l].T) + (dK @ self.Wk[l].T) + (dV @ self.Wv[l].T)

            dH = d_attn_input + d_attn_res_in

        x = cache['x']  # (B, S)

        flat_x = x.flatten()
        dH_flat = dH.reshape(-1, self.embed_dim)
        self.np.add.at(gradients['We'], flat_x, dH_flat)

        return gradients

    def sgd_step(self, gradients, learning_rate=0.01):
        self.We -= learning_rate * gradients['We']
        self.Wq -= learning_rate * gradients['Wq']
        self.Wk -= learning_rate * gradients['Wk']
        self.Wv -= learning_rate * gradients['Wv']
        self.Wo -= learning_rate * gradients['Wo']
        self.W1 -= learning_rate * gradients['W1']
        self.b1 -= learning_rate * gradients['b1']
        self.W2 -= learning_rate * gradients['W2']
        self.b2 -= learning_rate * gradients['b2']
        self.W_vocab -= learning_rate * gradients['W_vocab']
        self.b_vocab -= learning_rate * gradients['b_vocab']

    def calculate_loss_dlogits(self, logits, y):
        """
        logits : (B, S, vocab_size)
        y : (B, S)
        return:
            loss : scalar
            dlogits : (B, S, vocab_size)
        """

        B, S, V = logits.shape

        max_log = self.np.max(logits, axis=1, keepdims=True)
        x_shift = logits - max_log
        exp_x = self.np.exp(x_shift)
        sum_exp = self.np.sum(exp_x, axis=-1, keepdims=True)
        log_probs = x_shift - self.np.log(sum_exp)  # (B, S, V)

        N = B * S
        y_flat = y.reshape(N)
        log_probs_2d = log_probs.reshape(N, V)
        correct_lp = log_probs_2d[np.arange(N), y_flat]
        loss = -self.np.mean(correct_lp)

        softmax_out = exp_x / sum_exp
        dlogits = softmax_out

        for i in range(N):
            dlogits.reshape(N, V)[i, y_flat[i]] -= 1.0
        dlogits /= N

        return loss, dlogits

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


def pad_sequence(sequence, max_len, pad_token=0, lib=np):
    padded_sequence = sequence + [pad_token] * (max_len - len(sequence))
    return padded_sequence


def create_padding_mask(x, pad_token=0, lib=np):
    """
    x: (B, S)
    return: shape (B, 1, 1, S)
    """
    return (x == pad_token)[:, lib.newaxis, lib.newaxis, :].astype('float32')


def lecun_init(shape, fan_in, lib=np):
    scale = lib.sqrt(1 / fan_in)
    return lib.random.uniform(-scale, scale, shape).astype(lib.float32)


def create_look_ahead_mask(size, lib=np):
    """
       return: shape (1, 1, s, s)
    """
    return lib.triu(lib.ones((1, 1, size, size)), k=1).astype('float32') * -1e9


def positional_encoding(max_len, embed_dim, np_module):
    pe = np_module.zeros((max_len, embed_dim))
    position = np_module.arange(0, max_len)[:, np_module.newaxis]
    div_term = np_module.exp(np_module.arange(0, embed_dim, 2) * -(np_module.log(10000.0) / embed_dim))
    pe[:, 0::2] = np_module.sin(position * div_term)
    pe[:, 1::2] = np_module.cos(position * div_term)
    return pe.astype(np_module.float32)


if __name__ == "__main__":
    np.random.seed(42)

    vocab_size = 1000
    embed_dim = 64
    num_heads = 8
    ff_dim = 256
    num_layers = 2
    max_len = 50

    model = Transformer(vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len)

    x = np.array([[1, 2, 3, 4, 5],
                  [5, 4, 3, 2, 1]], dtype=np.int32)
    y = np.array([[2, 3, 4, 5, 6],
                  [1, 1, 1, 1, 1]], dtype=np.int32)

    logits, cache = model.forward(x)

    loss, dlogits = model.calculate_loss_dlogits(logits, y)
    print("loss:", loss)

    grads = model.backward(cache, dlogits)
    model.sgd_step(grads, learning_rate=0.001)
