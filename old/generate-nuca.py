# %% import packages

# helper libraries
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

# TensorFlow and tf.keras
import tensorflow as tf

print(tf.__version__)

# fix random seed for reproducibility
seed = 2023
np.random.seed(seed)

from tensorflow.keras.layers import Conv1D, Layer, MaxPooling1D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import Input

import sys
sys.path.append('C:\\Users\\mrollier\\OneDrive - UGent\\Research\\Cellular Automata\\CA Programming\\learning_automata')
from cnn_models import nuca_emulator_1D
from custom_tf_classes import WeightsBiasesHistory

from keras.callbacks import EarlyStopping, Callback
# from keras import regularizers

import time

images_dir = "./figures"


# %%

vert=3
horz=3
fig, axs = plt.subplots(vert,horz,figsize=(9,9))

N = 32
T = N # // 2

for i in range(vert):
    for j in range(horz):

        rule1 = np.random.randint(256)
        rule2 = np.random.randint(256)
        title = f"{rule1} and {rule2}"

        print(f"Working on {title}.     ", end='\r')

        init_config = np.random.randint(2, size=(1,N))
        rule_alloc = np.random.randint(2, size=N)

        rules = [rule1, rule2]

        timesteps=1
        output_hidden=False

        nuca_cnn = nuca_emulator_1D(N, rules, timesteps=timesteps, rule_alloc=rule_alloc, train_triplet_id=False, output_hidden=output_hidden)
        nuca_cnn.compile()

        diagram = np.empty((N,T))
        diagram[:,0] = init_config[0]
        for t in range(1,T):
            next_config = nuca_cnn.predict(diagram[:,t-1:t].T, verbose=0)[0]
            diagram[:,t] = next_config[:,0]


        axs[i,j].imshow(diagram.T, cmap='Greys')
        # axs[i,j].set_title(title)
        axs[i,j].set_title(None)
        axs[i,j].set_xticks([])
        axs[i,j].set_yticks([])

plt.savefig(f'{images_dir}/examples-of-nucas_{vert}x{horz}.pdf', bbox_inches='tight')
# %%