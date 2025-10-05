import numpy as np

from utils.function import *
import pathlib
import pickle


class Transformer:
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len,
                 embedding_weight=None, use_gpu=False, dropout_p=0.1, use_adam=True, enable_tf32=True):
        self.use_gpu = use_gpu and (default_library == 'cupy')
        self.np = cupy if self.use_gpu else numpy

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.max_len = max_len
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads

        # training/eval mode
        self.training = True
        self.dropout_p = float(dropout_p)

        # Optimizer (Adam)
        self.use_adam = use_adam
        self.adam_b1 = 0.9
        self.adam_b2 = 0.999
        self.adam_eps = 1e-8
        self.adam_t = 0

        if embedding_weight is None:
            self.We = lecun_init((self.vocab_size, self.embed_dim), self.embed_dim, self.np)
        else:
            self.We = self.np.array(embedding_weight)

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

        self.W_vocab = lecun_init((embed_dim, vocab_size), fan_in=embed_dim, lib=self.np)
        self.b_vocab = self.np.zeros((vocab_size,), dtype=self.np.float32)

        self.pe = positional_encoding(self.max_len, self.embed_dim, self.np)

        # LayerNorm parameters (learnable)
        self.ln1_gamma = self.np.ones((self.num_layers, self.embed_dim), dtype=self.np.float32)
        self.ln1_beta = self.np.zeros((self.num_layers, self.embed_dim), dtype=self.np.float32)
        self.ln2_gamma = self.np.ones((self.num_layers, self.embed_dim), dtype=self.np.float32)
        self.ln2_beta = self.np.zeros((self.num_layers, self.embed_dim), dtype=self.np.float32)

        # Adam state containers
        if self.use_adam:
            self._init_adam_states()

        # Try to enable TF32 on CuPy/cuBLAS for faster matmuls
        if self.use_gpu and enable_tf32:
            try:
                # Set cuBLAS math mode to TF32 tensor ops if available
                h = cupy.cuda.device.get_cublas_handle()
                cupy.cuda.cublas.setMathMode(h, cupy.cuda.cublas.CUBLAS_TF32_TENSOR_OP_MATH)  # CUDA 11+
            except Exception:
                try:
                    # Some versions expose allow_tf32 flag
                    cupy.cuda.matmul.allow_tf32 = True
                except Exception:
                    pass

    def _init_adam_states(self):
        npb = self.np
        def zlike(x):
            return npb.zeros_like(x)
        self.m = {
            'We': zlike(self.We), 'Wq': zlike(self.Wq), 'Wk': zlike(self.Wk), 'Wv': zlike(self.Wv),
            'Wo': zlike(self.Wo), 'W1': zlike(self.W1), 'W2': zlike(self.W2),
            'b1': zlike(self.b1), 'b2': zlike(self.b2),
            'W_vocab': zlike(self.W_vocab), 'b_vocab': zlike(self.b_vocab),
            'ln1_gamma': zlike(self.ln1_gamma), 'ln1_beta': zlike(self.ln1_beta),
            'ln2_gamma': zlike(self.ln2_gamma), 'ln2_beta': zlike(self.ln2_beta),
        }
        self.v = {k: self.np.zeros_like(v) for k, v in self.m.items()}

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def _dropout(self, x):
        p = self.dropout_p if self.training else 0.0
        if p <= 0.0:
            return x, None
        keep = 1.0 - p
        mask = (self.np.random.random(x.shape) < keep).astype(self.np.float32)
        return x * mask / keep, mask

    def forward(self, x, return_hidden_only=False):
        # x : (B, S)
        B, S = x.shape

        H = self.We[x] + self.pe[:S]  # (B, S, E)

        padding_mask = create_padding_mask(x, lib=self.np)  # (B, 1, 1, S)
        look_ahead = create_look_ahead_mask(S, lib=self.np)  # (1, 1, S, S)
        combined_mask = self.np.maximum(look_ahead, padding_mask)  # (B, 1, S, S)

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

            # dropout on attention output
            attn_out, drop_attn_mask = self._dropout(attn_out)

            # LayerNorm with affine on residual
            attn_residual = attn_input + attn_out
            attn_output, ln1_cache = layer_norm_affine_forward(
                attn_residual, self.ln1_gamma[l], self.ln1_beta[l], eps=1e-6)

            ffn_input = attn_output
            ffn_hidden = relu(ffn_input @ self.W1[l] + self.b1[l])
            ffn_out = ffn_hidden @ self.W2[l] + self.b2[l]

            # dropout on FFN output
            ffn_out, drop_ffn_mask = self._dropout(ffn_out)

            ffn_residual = ffn_input + ffn_out
            H_new, ln2_cache = layer_norm_affine_forward(
                ffn_residual, self.ln2_gamma[l], self.ln2_beta[l], eps=1e-6)

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

            # caches for norm and dropout
            layer_c['ln1'] = ln1_cache
            layer_c['ln2'] = ln2_cache
            layer_c['drop_attn'] = drop_attn_mask
            layer_c['drop_ffn'] = drop_ffn_mask

            layers_cache.append(layer_c)
            H = H_new

        if return_hidden_only:
            cache = {
                'x': x,
                'H_final': H,
                'layers': layers_cache
            }
            return H, cache

        logits = H @ self.W_vocab + self.b_vocab  # (B, S, vocab_size)

        cache = {
            'x': x,
            'H_final': H,
            'layers': layers_cache
        }

        return logits, cache

    def backward_last_token(self, cache, t_index, dlogits_last):
        """
        Memory-efficient backward pass for last-token-only loss.
        t_index: (B,) indices of last real token per sequence
        dlogits_last: (B, V)
        """
        B = t_index.shape[0]
        V = dlogits_last.shape[1]
        E = self.embed_dim

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
            'b_vocab': self.np.zeros_like(self.b_vocab),
            'ln1_gamma': self.np.zeros_like(self.ln1_gamma),
            'ln1_beta': self.np.zeros_like(self.ln1_beta),
            'ln2_gamma': self.np.zeros_like(self.ln2_gamma),
            'ln2_beta': self.np.zeros_like(self.ln2_beta),
        }

        H_final = cache['H_final']  # (B, S, E)
        S = H_final.shape[1]
        idx = self.np.arange(B)
        H_last = H_final[idx, t_index, :]  # (B, E)

        # Vocab head grads
        gradients['W_vocab'] += H_last.T @ dlogits_last  # (E,V)
        gradients['b_vocab'] += self.np.sum(dlogits_last, axis=0)

        # Gradient to H, only last positions
        dH = self.np.zeros_like(H_final)
        dH[idx, t_index, :] = dlogits_last @ self.W_vocab.T  # (B,E)

        # Backward through encoder stack (same as backward, starting from dH)
        layers = cache['layers']
        for l in reversed(range(self.num_layers)):
            layer_c = layers[l]

            ffn_input = layer_c['ffn_input']  # (B, S, E)
            ffn_out = layer_c['ffn_out']  # (B, S ,E)
            ffn_hidden = layer_c['ffn_hidden']  # (B, S, ff_dim)

            # LN2 backward
            d_residual_in, d_ln2_gamma, d_ln2_beta = layer_norm_affine_backward(dH, layer_c['ln2'])
            gradients['ln2_gamma'][l] += d_ln2_gamma
            gradients['ln2_beta'][l] += d_ln2_beta

            d_ffn_out = d_residual_in
            if layer_c['drop_ffn'] is not None:
                keep = 1.0 - self.dropout_p
                d_ffn_out = d_ffn_out * layer_c['drop_ffn'] / keep

            # FFN grads
            gradients['W2'][l] += (ffn_hidden.reshape(-1, self.ff_dim).T @
                                   d_ffn_out.reshape(-1, self.embed_dim))
            gradients['b2'][l] += self.np.sum(d_ffn_out, axis=(0, 1))

            d_ffn_hidden = d_ffn_out @ self.W2[l].T
            z = ffn_input @ self.W1[l] + self.b1[l]
            dz = relu_backward(d_ffn_hidden, z)
            gradients['W1'][l] += (ffn_input.reshape(-1, self.embed_dim).T @
                                   dz.reshape(-1, self.ff_dim))
            gradients['b1'][l] += self.np.sum(dz, axis=(0, 1))

            d_ffn_input = dz @ self.W1[l].T
            dH_attn_out = d_residual_in
            dH = d_ffn_input + dH_attn_out

            # Attention block
            attn_input = layer_c['attn_input']
            attn_out = layer_c['attn_out']
            attn_out_4d = layer_c['attn_out_beforeWo']
            attn_weights = layer_c['attn_weights']
            scores = layer_c['scores']
            Q_4d = layer_c['Q']
            K_4d = layer_c['K']
            V_4d = layer_c['V']

            d_attn_res_in, d_ln1_gamma, d_ln1_beta = layer_norm_affine_backward(dH, layer_c['ln1'])
            gradients['ln1_gamma'][l] += d_ln1_gamma
            gradients['ln1_beta'][l] += d_ln1_beta
            d_attn_out = d_attn_res_in
            if layer_c['drop_attn'] is not None:
                keep = 1.0 - self.dropout_p
                d_attn_out = d_attn_out * layer_c['drop_attn'] / keep

            dWo_in = d_attn_out
            attn_out_noWo_2d = attn_out_4d.transpose(0, 2, 1, 3).reshape(B * S, self.embed_dim)
            dWo_in_2d = dWo_in.reshape(B * S, self.embed_dim)
            gradients['Wo'][l] += attn_out_noWo_2d.T @ dWo_in_2d

            d_attn_out_noWo_2d = dWo_in_2d @ self.Wo[l].T
            d_attn_out_noWo_4d = d_attn_out_noWo_2d.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            d_attn_weight = d_attn_out_noWo_4d @ V_4d.transpose(0, 1, 3, 2)
            dV_4d = attn_weights.transpose(0, 1, 3, 2) @ d_attn_out_noWo_4d
            d_attn_score = softmax_backward(d_attn_weight, attn_weights)

            scale = 1.0 / self.np.sqrt(self.head_dim)
            dQ_4d = scale * (d_attn_score @ K_4d)
            dK_4d = scale * (d_attn_score.transpose(0, 1, 3, 2) @ Q_4d)

            dQ = dQ_4d.transpose(0, 2, 1, 3).reshape(B, S, self.embed_dim)
            dK = dK_4d.transpose(0, 2, 1, 3).reshape(B, S, self.embed_dim)
            dV = dV_4d.transpose(0, 2, 1, 3).reshape(B, S, self.embed_dim)

            X2d = attn_input.reshape(-1, self.embed_dim)
            gradients['Wq'][l] += X2d.T @ dQ.reshape(-1, self.embed_dim)
            gradients['Wk'][l] += X2d.T @ dK.reshape(-1, self.embed_dim)
            gradients['Wv'][l] += X2d.T @ dV.reshape(-1, self.embed_dim)

            d_attn_input = (dQ @ self.Wq[l].T) + (dK @ self.Wk[l].T) + (dV @ self.Wv[l].T)
            dH = d_attn_input + d_attn_res_in

        # Embedding gradient (scatter add)
        x = cache['x']  # (B, S)
        flat_x = x.flatten()
        dH_flat = dH.reshape(-1, self.embed_dim)
        self.np.add.at(gradients['We'], flat_x, dH_flat)

        return gradients

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

            # LN2 backward
            d_residual_in, d_ln2_gamma, d_ln2_beta = layer_norm_affine_backward(dH, layer_c['ln2'])
            # grads for ln2 params
            gradients.setdefault('ln2_gamma', self.np.zeros_like(self.ln2_gamma))
            gradients.setdefault('ln2_beta', self.np.zeros_like(self.ln2_beta))
            gradients['ln2_gamma'][l] += d_ln2_gamma
            gradients['ln2_beta'][l] += d_ln2_beta

            d_ffn_out = d_residual_in  # (B, S, E) path through residual to FFN output

            # dropout backward on FFN output
            if layer_c['drop_ffn'] is not None:
                keep = 1.0 - self.dropout_p
                d_ffn_out = d_ffn_out * layer_c['drop_ffn'] / keep

            # dW2: (ff_dim, E)
            gradients['W2'][l] += (ffn_hidden.reshape(-1, self.ff_dim).T @
                                   d_ffn_out.reshape(-1, self.embed_dim))
            gradients['b2'][l] += self.np.sum(d_ffn_out, axis=(0, 1))

            d_ffn_hidden = d_ffn_out @ self.W2[l].T  # (B, S, ff_dim)

            z = ffn_input @ self.W1[l] + self.b1[l]  # (B, S, ff_dim)
            dz = relu_backward(d_ffn_hidden, z)

            # dW1: (E, ff_dim)
            gradients['W1'][l] += (ffn_input.reshape(-1, self.embed_dim).T @
                                   dz.reshape(-1, self.ff_dim))
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
            d_attn_res_in, d_ln1_gamma, d_ln1_beta = layer_norm_affine_backward(dH, layer_c['ln1'])
            gradients.setdefault('ln1_gamma', self.np.zeros_like(self.ln1_gamma))
            gradients.setdefault('ln1_beta', self.np.zeros_like(self.ln1_beta))
            gradients['ln1_gamma'][l] += d_ln1_gamma
            gradients['ln1_beta'][l] += d_ln1_beta

            d_attn_out = d_attn_res_in

            # dropout backward on attention output
            if layer_c['drop_attn'] is not None:
                keep = 1.0 - self.dropout_p
                d_attn_out = d_attn_out * layer_c['drop_attn'] / keep

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

            # dWq/dWk/dWv: (E, E)
            X2d = attn_input.reshape(-1, self.embed_dim)
            gradients['Wq'][l] += X2d.T @ dQ.reshape(-1, self.embed_dim)
            gradients['Wk'][l] += X2d.T @ dK.reshape(-1, self.embed_dim)
            gradients['Wv'][l] += X2d.T @ dV.reshape(-1, self.embed_dim)

            d_attn_input = (dQ @ self.Wq[l].T) + (dK @ self.Wk[l].T) + (dV @ self.Wv[l].T)

            dH = d_attn_input + d_attn_res_in

        x = cache['x']  # (B, S)

        flat_x = x.flatten()
        dH_flat = dH.reshape(-1, self.embed_dim)
        self.np.add.at(gradients['We'], flat_x, dH_flat)

        return gradients

    def _adam_update(self, param_name, param, grad, lr):
        m = self.m[param_name]
        v = self.v[param_name]
        b1, b2, eps = self.adam_b1, self.adam_b2, self.adam_eps
        m[:] = b1 * m + (1 - b1) * grad
        v[:] = b2 * v + (1 - b2) * (grad * grad)
        m_hat = m / (1 - b1 ** self.adam_t)
        v_hat = v / (1 - b2 ** self.adam_t)
        param -= lr * m_hat / (self.np.sqrt(v_hat) + eps)

    def step(self, gradients, learning_rate=0.001):
        if self.use_adam and (not hasattr(self, 'm') or not hasattr(self, 'v')):
            self._init_adam_states()

        if self.use_adam:
            self.adam_t += 1
            self._adam_update('We', self.We, gradients['We'], learning_rate)
            self._adam_update('Wq', self.Wq, gradients['Wq'], learning_rate)
            self._adam_update('Wk', self.Wk, gradients['Wk'], learning_rate)
            self._adam_update('Wv', self.Wv, gradients['Wv'], learning_rate)
            self._adam_update('Wo', self.Wo, gradients['Wo'], learning_rate)
            self._adam_update('W1', self.W1, gradients['W1'], learning_rate)
            self._adam_update('b1', self.b1, gradients['b1'], learning_rate)
            self._adam_update('W2', self.W2, gradients['W2'], learning_rate)
            self._adam_update('b2', self.b2, gradients['b2'], learning_rate)
            self._adam_update('W_vocab', self.W_vocab, gradients['W_vocab'], learning_rate)
            self._adam_update('b_vocab', self.b_vocab, gradients['b_vocab'], learning_rate)
            self._adam_update('ln1_gamma', self.ln1_gamma, gradients['ln1_gamma'], learning_rate)
            self._adam_update('ln1_beta', self.ln1_beta, gradients['ln1_beta'], learning_rate)
            self._adam_update('ln2_gamma', self.ln2_gamma, gradients['ln2_gamma'], learning_rate)
            self._adam_update('ln2_beta', self.ln2_beta, gradients['ln2_beta'], learning_rate)
        else:
            # SGD fallback
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
            # LN params
            self.ln1_gamma -= learning_rate * gradients['ln1_gamma']
            self.ln1_beta -= learning_rate * gradients['ln1_beta']
            self.ln2_gamma -= learning_rate * gradients['ln2_gamma']
            self.ln2_beta -= learning_rate * gradients['ln2_beta']

    # Backward-compat name
    def sgd_step(self, gradients, learning_rate=0.01):
        self.step(gradients, learning_rate)

    def calculate_loss_dlogits(self, logits, y):
        """
        logits : (B, S, vocab_size) or (B, V) if S==1 and squeezed
        y : (B, S) or (B,) aligned with logits
        return:
            loss : scalar
            dlogits : (B, S, vocab_size)
        """

        # Normalize shapes to (B, S, V)
        if logits.ndim == 2:
            logits = logits[:, None, :]
        if y.ndim == 1:
            y = y[:, None]

        B, S, V = logits.shape

        max_log = logits.max(axis=-1, keepdims=True)
        x_shift = logits - max_log
        exp_x = self.np.exp(x_shift)
        sum_exp = self.np.sum(exp_x, axis=-1, keepdims=True)
        log_probs = x_shift - self.np.log(sum_exp)  # (B, S, V)

        N = B * S
        y_flat = y.reshape(N)
        log_probs_2d = log_probs.reshape(N, V)
        correct_lp = log_probs_2d[self.np.arange(N), y_flat]
        loss = -self.np.mean(correct_lp)

        softmax_out = exp_x / sum_exp
        dlogits = softmax_out.reshape(N, V)
        dlogits[self.np.arange(N), y_flat] -= 1.0
        dlogits = (dlogits / N).reshape(B, S, V)

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
            'W_vocab': self.W_vocab,
            'b_vocab': self.b_vocab,
            'ln1_gamma': self.ln1_gamma,
            'ln1_beta': self.ln1_beta,
            'ln2_gamma': self.ln2_gamma,
            'ln2_beta': self.ln2_beta,
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
    mask = (x == pad_token).astype(lib.float32)
    mask = mask[:, lib.newaxis, lib.newaxis, :]  # (B,1,1,S)
    mask *= -1e9
    return mask


def lecun_init(shape, fan_in, lib=np):
    scale = lib.sqrt(1 / fan_in)
    return lib.random.uniform(-scale, scale, shape).astype(lib.float32)


def create_look_ahead_mask(size, lib=np):
    """
       return: shape (1, 1, s, s)
    """
    mask = lib.triu(lib.ones((1, 1, size, size), dtype='float32'), k=1) * -1e9
    return mask


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
