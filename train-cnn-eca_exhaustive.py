# %% import packages

# helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cellpylib as cpl
import os
import logging

import pickle

plt.rcParams['text.usetex'] = True

# TensorFlow and tf.keras
import tensorflow as tf
print(tf.__version__)

# fix random seed for reproducibility
seed = 2024
np.random.seed(seed)

import tensorflow as tf

from tensorflow.keras.layers import Conv1D, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import Input

from keras.callbacks import Callback
from keras import regularizers

# Set TensorFlow logging to only display ERROR messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Suppress INFO messages
logging.getLogger('tensorflow').setLevel(logging.ERROR) 

import tensorflow as tf

# Your TensorFlow code here


images_dir = "./figures"
models_dir = "./models"


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
    
# %% Define model described in Appendix of Gilspin (2019)

def gilspin_model(diagram_dims=(62,1), activation=None):
    # input
    inputs = Input(diagram_dims, dtype=tf.float32)

    # add cylindrical padding (periodic boundary conditions)
    x = CylindricalPadding1D(padding=1)(inputs)

    # Convolution
    x = Conv1D(8,3, activation='relu', padding='valid')(x) # filters, kernel size

    # 1x1 convolution to add up the channels. This layer has fixed weights and biases of zero.
    outputs = Conv1D(1,1, padding='valid',trainable=False, activation=activation)(x)

    # sequence model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

# define base representation function
def _base_repr(number, base=2, length=8):
    base_repr = np.base_repr(number, base=base)
    if len(base_repr) < length:
        extra_zeros = (length - len(base_repr)) * '0'
        base_repr = extra_zeros + base_repr
    return base_repr

# %% Training loop

RUN_AGAIN = False

N_train = 2**14
N_val = 2**14
N = 62
diagram_dims = (N,1)

N_epochs = 50
batch_size = 64
batches_per_epoch = int(N_train / batch_size)

loss = 'mse'
LR = 0.001
activation='tanh'

histories = []
verbose = 0
SAVE_MODEL = False
SAVE_NAME_PREFIX = f"cnn-eca-{N_epochs}-epochs_"

if RUN_AGAIN:
    for ECA_rule in range(256):
        print(64*"=")
        if ECA_rule<10:
            print(f"Training ECA rule: 00{ECA_rule}")
            savename = SAVE_NAME_PREFIX + "00" + str(ECA_rule)
        elif ECA_rule<100:
            print(f"Training ECA rule: 0{ECA_rule}")
            savename = SAVE_NAME_PREFIX + "0" + str(ECA_rule)
        else:
            print(f"Training ECA rule: {ECA_rule}")
            savename = SAVE_NAME_PREFIX + str(ECA_rule)
        print()

        ECA_rule_bin = _base_repr(ECA_rule)

        # Initiate CNN

        cnn = gilspin_model(diagram_dims=diagram_dims) # no activation function
        cnn.compile()

        # Calculate desired labels by using perfect CNN.

        # weights
        weights = np.array([np.array(list(_base_repr(triplet, length=3))).astype(int) for triplet in range(8)])

        # biases for locating triplets
        biases = np.array(list(ECA_rule_bin[::-1])).astype(int)-np.sum(weights, axis=1)

        # large negative weight for non-ocurring triplets
        weights[weights==0] = -100
        weights = np.expand_dims(weights.T, axis=1)

        cnn.set_weights([weights,
                        biases,
                        np.ones((1,8,1)),
                        np.zeros(1)])

        # create new random data (using 'perfect' cnn)
        x_train = np.random.randint(2,size=(N_train, N, 1),
                                    dtype=np.int8)
        x_val = np.random.randint(2, size=(N_val, N, 1),
                                dtype=np.int8)

        r_train = cnn.predict(x_train,
                            batch_size=N,
                            verbose=verbose).astype(np.int8)
        r_val = cnn.predict(x_val,
                            batch_size=N,
                            verbose=verbose).astype(np.int8)

        # Now re-initialise the model with random trainable weights and biases, and try to train it to perfection.
        cnn = gilspin_model(diagram_dims=diagram_dims, activation=activation)

        cnn.compile(loss=loss,
                    optimizer=Adam(learning_rate=LR),
                    metrics=['mse'])
        
        # fix weights of final conv1D (this just sums all channels)
        cnn.layers[-1].set_weights([np.ones((1,8,1)), np.zeros(1)])
        
        history = cnn.fit(x_train,
                        r_train,
                        batch_size=batch_size,
                        epochs=N_epochs,
                        verbose=verbose,
                        validation_data=(x_val, r_val),
                        shuffle=True)

        histories += [history]

        final_val_mse = history.history['val_mse'][-1]
        print(f"Final validation MSE: {final_val_mse}")

        if SAVE_MODEL:
            cnn.save(f"{models_dir}/{savename}")
            print(f"Saved model {savename}.")

        print()
# %% Save histories

SAVE_HISTORIES=False

if SAVE_HISTORIES:
    for ECA_rule in range(256):
        if ECA_rule<10:
            savename = SAVE_NAME_PREFIX + "00" + str(ECA_rule)
        elif ECA_rule<100:
            savename = SAVE_NAME_PREFIX + "0" + str(ECA_rule)
        else:
            savename = SAVE_NAME_PREFIX + str(ECA_rule)

        history = histories[ECA_rule]
        with open(f'{models_dir}/{savename}/history.pkl', 'wb') as file:
            pickle.dump(history.history, file)
# %% Load histories

histories = []
for ECA_rule in range(256):
    if ECA_rule<10:
        savename = SAVE_NAME_PREFIX + "00" + str(ECA_rule)
    elif ECA_rule<100:
        savename = SAVE_NAME_PREFIX + "0" + str(ECA_rule)
    else:
        savename = SAVE_NAME_PREFIX + str(ECA_rule)

    with open(f'{models_dir}/{savename}/history.pkl', 'rb') as file:
        history = pickle.load(file)
        histories += [history]

# %% Plot histories
        
def reduced_langton_param(rule):
    # only works for ECA
    number_of_1s = np.base_repr(rule).count('1')
    lp = number_of_1s/8
    rlp = -abs(lp-1/2)+1/2
    return rlp

cmap = plt.get_cmap('viridis')
colours = [cmap(i) for i in np.linspace(0,1,5)]

fig, ax = plt.subplots(1,1,figsize=(10,3))
linthresh=1e-6
ax.set_yscale('symlog', linthresh=linthresh)
ax.axhline(linthresh, color='k', lw=1, ls='--', alpha=.5)

labelsize=16

ax.set_xlim(1, 50)
ax.set_ylim(0, 1)
ax.set_xlabel('Epochs', fontsize=labelsize)
ax.set_ylabel('Validation MSE', fontsize=labelsize)

ax.set_yticks([0, 1e-6, 1e-4, 1e-2, 1])

for ECA_rule in range(256):
    mses = histories[ECA_rule]['val_mse']
    colour = colours[int(reduced_langton_param(ECA_rule)*8)]
    ax.plot(np.arange(1, 51), mses, color=colour, lw=1, alpha=.3)

# Creating custom legend from the color array
from matplotlib.patches import Patch

labels = ['0', '1/8', '2/8', '3/8', '4/8']
legend_elements = [Patch(facecolor=colour,
                         edgecolor='black',
                         label=labels[i]) for i, colour in enumerate(colours)]
ax.legend(handles=legend_elements, ncols=5, loc='upper center', title='Reduced Langton parameter')

# plt.savefig("./figures/val_mse_vs_epochs_rlp_colour.pdf", bbox_inches='tight')

# %% List the ECA rules for which no convergence occurs

nonconvergent_rules = []
for ECA_rule in range(256):
    final_mse = histories[ECA_rule]['val_mse'][-1]
    if final_mse > 0.1:
        nonconvergent_rules += [ECA_rule]
# %%
