# %% load packages
from src.custom_tf_classes.activations import general_sigmoid
import tensorflow as tf

# %% test and show
import numpy as np
import matplotlib.pyplot as plt

x = tf.constant(np.linspace(-1, 2, 10000))


hstretches = np.logspace(0, 2, 10)


for hstretch in hstretches:
    gs = general_sigmoid(x, hstretch=hstretch).numpy()
    plt.plot(x, gs, label=str(round(hstretch)))

plt.plot([-1,2], [0,0], 'k--', lw=1)
plt.plot([-1,2], [1,1], 'k--', lw=1)
plt.plot([0,0],  [-1,2], 'k--', lw=1)
plt.plot([1,1],  [-1,2], 'k--', lw=1)

plt.xlim(-.5,1.5)
plt.ylim(-.5, 1.5)

plt.legend()
plt.title("Activation function: modified sigmoid for various hstretch factors")
# %%
