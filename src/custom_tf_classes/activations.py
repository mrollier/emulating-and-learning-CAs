# %% import packages

import tensorflow as tf
import numpy as np

# %% incomplete

def general_sigmoid(x, hstretch=1):
    vstretch = (1+np.exp(hstretch/2))/(np.exp(hstretch/2)-1)
    vtrans = 1/(1-np.exp(hstretch/2))
    htrans = -hstretch/2
    gs = vstretch / (1 + tf.exp(-(x*hstretch + htrans))) + vtrans
    return gs
