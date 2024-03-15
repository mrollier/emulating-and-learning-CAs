import numpy as np

# define base representation function
def _base_repr(number, base=2, length=8):
    base_repr = np.base_repr(number, base=base)
    if len(base_repr) < length:
        extra_zeros = (length - len(base_repr)) * '0'
        base_repr = extra_zeros + base_repr
    return base_repr