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
sys.path.append('C:\\Users\\mrollier\\OneDrive - UGent\\Research\\Cellular Automata\\CA Programming\\learning_automata\\src')
# from nn import nuca_emulator_1D
from nn.eca import EcaEmulator
from custom_tf_classes.callbacks import WeightsBiasesHistory

from keras.callbacks import EarlyStopping, Callback
# from keras import regularizers

import time

images_dir = "../figures/eca"


# %%

SAVEFIG = True

vert=6
horz=6
fig, axs = plt.subplots(vert,horz,figsize=(9,6))

N = 32
T = N // 2

for i in range(vert):
    for j in range(horz):

        rule = np.random.randint(256)
        title = f"rule {rule}"

        print(f"Working on {title}.     ", end='\r')

        init_config = np.random.randint(2, size=(1,N))

        timesteps=1
        output_hidden=False

        eca_cnn = EcaEmulator(N, rule, timesteps=timesteps,
                                train_triplet_id=False, output_hidden=output_hidden).model()
        eca_cnn.compile()

        diagram = np.empty((N,T))
        diagram[:,0] = init_config[0]
        for t in range(1,T):
            next_config = eca_cnn.predict(diagram[:,t-1:t].T, verbose=0)[0]
            diagram[:,t] = next_config[:,0]


        axs[i,j].imshow(diagram.T, cmap='Greys')
        axs[i,j].set_title(title)
        # axs[i,j].set_title(None)
        axs[i,j].set_xticks([])
        axs[i,j].set_yticks([])

if SAVEFIG:
    plt.savefig(f'{images_dir}/examples-of-ecas_{vert}x{horz}.pdf', bbox_inches='tight')
# %%