import numpy

try:
    import cupy

    default_library = 'cupy'
except ImportError:
    default_library = 'numpy'


def softmax(x, axis=-1):
    np = cupy.get_array_module(x) if 'cupy' in str(type(x)) else numpy

    max_val = np.max(x, axis=axis, keepdims=True)
    x_exp = np.exp(x - max_val)
    sum_exp = np.sum(x_exp, axis=axis, keepdims=True)
    return x_exp / sum_exp


def sigmoid(x):
    np = cupy.get_array_module(x) if 'cupy' in str(type(x)) else numpy

    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def clip_grads(gradients, max_norm):
    if not gradients:
        return

    np = cupy.get_array_module(gradients[0]) if 'cupy' in str(type(gradients[0])) else numpy

    total_norm = np.sqrt(sum(np.sum(np.square(g)) for g in gradients))

    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for g in gradients:
            g *= scale


def relu(x):
    np = cupy.get_array_module(x) if 'cupy' in str(type(x)) else numpy
    return np.maximum(0, x)


def layer_norm(x, eps=1e-6):
    np = cupy.get_array_module(x) if 'cupy' in str(type(x)) else numpy
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def relu_backward(dout, x):
    dx = dout.copy()
    dx[x <= 0] = 0
    return dx


def softmax_backward(dout, softmax_output, axis=-1):
    np = cupy.get_array_module(dout) if 'cupy' in str(type(dout)) else numpy

    sum_dout = np.sum(dout * softmax_output, axis=axis, keepdims=True)

    dx = softmax_output * (dout - sum_dout)
    return dx


def layer_norm_backward(dout, x, eps=1e-5):
    # x, dout shape = (..., D)
    # LayerNorm은 마지막 차원 D에 대해 mean/var 계산
    np = cupy.get_array_module(x) if 'cupy' in str(type(x)) else numpy

    mean = np.mean(x, axis=-1, keepdims=True)  # shape (..., 1)
    var = np.var(x, axis=-1, keepdims=True)  # shape (..., 1)
    std = np.sqrt(var + eps)  # shape (..., 1)
    D = x.shape[-1]

    # dout_sum = \sum_j dout_ij
    dout_sum = np.sum(dout, axis=-1, keepdims=True)  # shape (..., 1)
    # dxmu_sum = \sum_j dout_ij*(x_ij - mu_i)
    dxmu_sum = np.sum(dout * (x - mean), axis=-1, keepdims=True)

    # 최종 식
    dx = (1. / std) * (
            dout
            - dout_sum / D
            - (x - mean) * dxmu_sum / ((var + eps) * D)
    )
    return dx

