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

from cnn_models import eca_emulator
from custom_tf_classes import WeightsBiasesHistory

from keras.callbacks import EarlyStopping, Callback
# from keras import regularizers

import time

images_dir = "./figures"


# %% Synthesise initial configurations

N_train = 2**12
N_val = 2**12

N = 256

x_train = np.random.randint(2,size=(N_train, N, 1),
                            dtype=np.int8)
x_val = np.random.randint(2, size=(N_val, N, 1),
                          dtype=np.int8)

# problems with: [1, 9, 33, 41, 73, 97, 124, 129, 137, 151, 161, 169, 193, 209, 225, 229, 233]

ECA_rule = 110 # right-moving # np.random.randint(256)
print(f"ECA rule: {ECA_rule}")



# %% Initiate CNN

diagram_dims = (N,1)
timesteps = 1 # N//2-1
output_hidden = True

cnn = eca_emulator(N, ECA_rule, timesteps=timesteps, output_hidden=output_hidden)
cnn.summary()
cnn.compile()

# %% make perfect predictions (labels). Save economically.
# NB this goes SUPER fast (relatively)

verbose=0

t0 = time.time()
if output_hidden:
    all_configs_train, r_train = cnn.predict(x_train, batch_size=len(x_train), verbose=verbose)
    all_configs_val, r_val = cnn.predict(x_val, batch_size=len(x_val), verbose=verbose)
    all_configs_train = np.array(all_configs_train, dtype=np.int8)
    r_train = np.array(r_train, dtype=np.int8)
    all_configs_val = np.array(all_configs_val, dtype=np.int8)
    r_val = np.array(r_val, dtype=np.int8)
else:
    r_train = cnn.predict(x_train, batch_size=len(x_train), verbose=verbose).astype(np.int8)
    r_val = cnn.predict(x_val, batch_size=len(x_train), verbose=verbose).astype(np.int8)
tf = time.time()

print(f"N={N} cells, N_train={N_train} diagrams, T={timesteps+1} timesteps.\nTime: {round(tf-t0,2)} seconds.")

# %% Make a spacetime diagram as a sanity check
    
verbose=True
diagram_train = x_train.copy()
t0 = time.time()

next_configs = np.array(cnn.predict(x_train, batch_size=len(x_train), verbose=verbose)).astype(np.int8)

# for next_config in next_configs:
#     diagram_train = np.append(diagram_train, next_config, axis=2)

# show_idx=420
# plt.imshow(diagram_train[show_idx].T, cmap='Greys')

# %% Now randomly re-initialise the model with random weights and biases, and try to train it to perfection.

# Note that this worked better without the intermediate step of first identifying the triplets, and then applying the rule. This needs some fine-tuning, indeed.

RUN_AGAIN = True

if RUN_AGAIN:
    batch_size = 16 # Gilpin chooses 10, that's not that much!
    batches_per_epoch = int(N_train / batch_size)
    epochs=100
    LR = 0.001
    stopping_patience = 5
    stopping_delta = 0.0001
    activation= 'tanh' # relu_ceiling does not work
    loss = 'mse' # tf.keras.losses.BinaryCrossentropy(from_logits=False) # bce (no logits) or mse

    cnn = eca_emulator(N, timesteps=timesteps, train_triplet_id=True, activation=activation)
    cnn.summary()

    cnn.compile(loss=loss,
                optimizer=Adam(learning_rate=LR),
                metrics=['mse'])
    
    stopping_callback = EarlyStopping(monitor='val_mse',
                                      patience=stopping_patience,
                                      min_delta=stopping_delta,
                                      verbose = 1,
                                      restore_best_weights=True)
    
    weights_biases_callback = WeightsBiasesHistory()

    history = cnn.fit(x_train,
                      r_train.astype(np.float32),
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      validation_data=(x_val, r_val.astype(np.float32)),
                      shuffle=True,
                      callbacks = [weights_biases_callback, stopping_callback])
# %% Retrieve histories of weights and biases
    
# dimensions: (epochs, layers)
wab_history = weights_biases_callback.wab_history

example_idx = 5
x_train_example = x_train[example_idx:example_idx+1]
r_train_example = r_train[example_idx:example_idx+1]

r_pred_history = []
mses = []

for wab_epoch in wab_history:
    # add historical weights and biases to CNN
    cnn.set_weights(wab_epoch)

    r_pred = cnn.predict(x_train_example, verbose=0)[0]
    r_pred_history += [r_pred]

    mse = np.square(r_pred - r_train_example).mean()
    mses += [mse]

r_pred_history = np.array(r_pred_history)
mses = np.array(mses)

# %% Plot everything nicely together

from matplotlib.colors import SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, axs = plt.subplots(4,2, figsize=(5, 3),
                        width_ratios=[5,1], height_ratios=[1, 7, 1, 1])
axs[0,1].remove()
axs[2,1].remove()
axs[3,1].remove()

# input array
axs[0,0].imshow(x_train_example, cmap='Greys')
axs[0,0].set_title("Input configuration")

# training
try:
    square_error_history = (r_pred_history - r_train_example)**2
except:
    square_error_history = (np.expand_dims(r_pred_history,axis=2) - r_train_example)**2

# errors
im = axs[1,0].imshow(square_error_history, cmap='RdYlGn_r',
              norm=SymLogNorm(1e-6, vmin=0, vmax=1), interpolation='none')
# prediction history
# im = axs[1,0].imshow(r_pred_history, cmap='Greys',
#               norm=SymLogNorm(1e-4, vmin=0, vmax=1), interpolation='none')

cbar_ticks = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
cbar_tick_labels = [r'$0$', r'$10^{-6}$', r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$1$']
cbar = fig.colorbar(im, ax=axs[1,0], orientation='horizontal', aspect=75)
cbar.ax.set_xticks(cbar_ticks,
                   minor=False)
# cbar.ax.tick_params(length=0, which='major', color='white')
cbar.ax.set_xticklabels(cbar_tick_labels,
                        fontdict={'fontsize':8})
axs[1,0].set_ylabel(r"$\leftarrow$ Epochs")
axs[1,0].set_title(f"CNN model convergence over {len(mses)} epochs")
axs[1,0].set_aspect('auto')

# CNN output
try:
    axs[2,0].imshow(r_pred_history[-1].T, cmap='Greys')
except:
    axs[2,0].imshow(np.expand_dims(r_pred_history[-1],axis=1).T, cmap='Greys')
axs[2,0].set_title("CNN model final output")

# MSE history
axs[1,1].plot(mses, np.arange(len(mses),0,-1), color='k', lw=1)
axs[1,1].set_xscale('log')
axs[1,1].set_title(r"MSE (log)")

vert_shift = .4*len(mses)
axs[1,1].set_ylim([-vert_shift, len(mses)+1])

axs[1,1].set_axis_off()

# output array
axs[3,0].imshow(r_train_example, cmap='Greys')
axs[3,0].set_title(f"Desired output, rule {ECA_rule}, {timesteps} timestep(s)")

for i in range(4):
    for j in range(2):
        axs[i,j].set_yticks([])
        axs[i,j].set_xticks([])

plt.tight_layout()

savename = f"cnn-eca-convergence_rule-{ECA_rule}_timesteps-{timesteps}.png"
# plt.savefig(images_dir + '/' + savename, bbox_inches='tight')
