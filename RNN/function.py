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
