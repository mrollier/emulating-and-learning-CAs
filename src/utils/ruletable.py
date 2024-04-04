# %% import packages
import numpy as np

# %% define base representation function
def base_repr(number, base=2, length=8):
    br = np.base_repr(number, base=base)
    if len(br) < length:
        extra_zeros = (length - len(br)) * '0'
        br = extra_zeros + br
    return br