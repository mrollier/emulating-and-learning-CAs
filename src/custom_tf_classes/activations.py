# %% import packages

import tensorflow as tf

# %% incomplete

def general_sigmoid(x, hstretch=1):
    vstretch = (1+np.exp(hstretch/2))/(np.exp(hstretch/2)-1)
    vtrans = 1/(1-np.exp(hstretch/2))
    htrans = -hstretch/2
    gs = vstretch / (1 + tf.exp(-(x*hstretch + htrans))) + vtrans
    return gs

# %% test and show
import numpy as np
import matplotlib.pyplot as plt

x = tf.constant(np.linspace(-1, 2, 10000))


hstretches = np.logspace(0, 2, 10)


for hstretch in hstretches:
    hstretch=10
    gs = general_sigmoid(x, hstretch=hstretch).numpy()
    plt.plot(x, gs)

plt.plot([-1,2], [0,0], 'k--', lw=1)
plt.plot([-1,2], [1,1], 'k--', lw=1)
plt.plot([0,0],  [-1,2], 'k--', lw=1)
plt.plot([1,1],  [-1,2], 'k--', lw=1)

plt.xlim(-.5,1.5)
plt.ylim(-.5, 1.5)

plt.title("Activation function: modified sigmoid")
# %%
