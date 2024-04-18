# %% import packages

# helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cellpylib as cpl

plt.rcParams['text.usetex'] = True

# TensorFlow and tf.keras
import tensorflow as tf

print(tf.__version__)

# fix random seed for reproducibility
seed = 2023
np.random.seed(seed)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation
from tensorflow.keras.layers import Conv1D, Layer
from tensorflow.keras.models import load_model
from tensorflow.keras import Input

images_dir = "./figures"

# %% Import custom CNN layer

class CylindricalPadding1D(Layer):
    # copied from ChatGPT
    def __init__(self, padding=1, **kwargs):
        super(CylindricalPadding1D, self).__init__(**kwargs)
        self.padding = padding

    def call(self, inputs):
        padded_inputs = inputs
        # pad in height
        if self.padding:
            padded_inputs = tf.concat([inputs[:, -self.padding:, :],
                                    inputs,
                                    inputs[:, :self.padding, :]], axis=1)
        return padded_inputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1] + 2*self.padding, input_shape[2]

    def get_config(self):
        config = super(CylindricalPadding1D, self).get_config()
        config.update({'padding': self.padding})
        return config

# %% Load data

# x_train_all = np.load("../files/ECAs_1024sims-per-rule_64x64_train_diagrams.npy")
# r_train_all = np.load("../files/ECAs_1024sims-per-rule_64x64_train_labels.npy")

# ECA_rule = 110

# x_train_sel = x_train_all[1024*ECA_rule:1024*(ECA_rule+1)]
# x_train_input  = x_train_sel[:,0]
# x_train_output = x_train_sel[:,1]

x_train_input = 

# %% visualise a single input row

fig, ax = plt.subplots(1,1,figsize=(10,1))
ax.imshow(x_train_input[0].T, cmap='Greys')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Random first row")

# %% Define model described in Appendix of Gilspin (2019)

def gilspin_model(diagram_dims=(64,1), sum_channels=True):
    # input
    inputs = Input(diagram_dims, dtype=tf.float32)

    # add cylindrical padding (periodic boundary conditions)
    x = CylindricalPadding1D(padding=1)(inputs)

    # Convolution
    x = Conv1D(8,3, activation='relu', padding='valid')(x) # filters, kernel size

    # 1x1 convolution to add up the channels
    if sum_channels:
        x = Conv1D(1,1, padding='valid')(x)

    # sequence model
    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model

# define base representation function
def _base_repr(number, base=2, length=8):
    base_repr = np.base_repr(number, base=base)
    if len(base_repr) < length:
        extra_zeros = (length - len(base_repr)) * '0'
        base_repr = extra_zeros + base_repr
    return base_repr

# %% Initiate CNN

diagram_dims = (64,1)
cnn = gilspin_model(diagram_dims=diagram_dims,
                    sum_channels=False)
cnn.summary()
cnn.compile()

# %% Define and assign weights and biases

ECA_rule_bin = _base_repr(ECA_rule)

# weights
weights = np.array([np.array(list(_base_repr(triplet, length=3))).astype(int) for triplet in range(8)])

# biases for locating triplets
biases_triplets = 1 - np.sum(weights, axis=1)

# large negative weight for non-ocurring triplets
weights[weights==0] = -100
weights = np.expand_dims(weights.T, axis=1)

cnn.set_weights([weights,
                 biases_triplets])

# %% Show decomposition
SAVEFIG=False

fig, axs = plt.subplots(3,2,height_ratios=[1,11,1], width_ratios=[1,64], figsize=(8,2))

axs[0,0].remove()
axs[2,0].remove()

axs[0,1].imshow(x_train_input[0].T, cmap='Greys')
decomposition = cnn.predict(x_train_input[0:1])
axs[1,1].imshow(decomposition.T, cmap='Greys')
axs[2,1].imshow(x_train_output[0].T, cmap='Greys')

ECA_rule_image = np.expand_dims(np.array(list(_base_repr(ECA_rule)[::-1])).astype(int),axis=1)
axs[1,0].imshow(ECA_rule_image, cmap='Greys')

for i in range(3):
    for j in range(2):
        axs[i,j].set_xticks([])
        axs[i,j].set_yticks([])

fontsize=16
axs[0,1].set_title("Input array", size=fontsize)
axs[1,0].set_ylabel(f'Rule {ECA_rule}', size=fontsize)
axs[1,1].set_title("Triplet decomposition", size=fontsize)
axs[2,1].set_title("Output array", size=fontsize)

plt.subplots_adjust(wspace=0.05)

if SAVEFIG:
    savename = f"triplet-decomposition_rule{ECA_rule}.pdf"
    plt.savefig(f"{images_dir}/{savename}", bbox_inches='tight')
# %%
