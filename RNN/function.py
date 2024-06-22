import numpy

try:
    import cupy

    default_library = 'cupy'
except ImportError:
    default_library = 'numpy'


def softmax(x):
    np = cupy.get_array_module(x) if 'cupy' in str(type(x)) else numpy

    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=1, keepdims=True)
        return x_exp / x_sum
    else:
        x = x - np.max(x)
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp)
        return x_exp / x_sum


def sigmoid(x):
    np = cupy.get_array_module(x) if 'cupy' in str(type(x)) else numpy

    return 1.0 / (1.0 + np.exp(-x))


def clip_grads(gradients, max_norm):
    np = cupy.get_array_module(gradients) if 'cupy' in str(type(gradients)) else numpy

    total_norm = 0
    for grad in gradients:
        total_norm += np.sum(np.square(grad))
    total_norm = np.sqrt(total_norm)

    if total_norm + 1e-6 > max_norm:
        for i in range(len(gradients)):
            gradients[i] *= max_norm / total_norm


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


def layer_norm_backward(dout, x, eps=1e-6):
    np = cupy.get_array_module(x) if 'cupy' in str(type(x)) else numpy
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    N, D = x.shape
    dx_normalized = dout / (std + eps)
    dmean = np.sum(dout * -1 / (std + eps), axis=-1, keepdims=True)
    dstd = np.sum(dout * (x - mean) * -1 / (std + eps)**2, axis=-1, keepdims=True)
    dx = dx_normalized + dmean / N + dstd * 2 * (x - mean) / N
    return dx
