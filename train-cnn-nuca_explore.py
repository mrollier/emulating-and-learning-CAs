# %% import packages

# helper libraries
import numpy as np
import time
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

# import existing packages

import tensorflow as tf
print(f'TensorFlow version {tf.__version__}')

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from custom_tf_classes import WeightsBiasesHistory

# import custom-made models
from cnn_models import nuca_emulator_1D
from cnn_training import predict_with_hidden

# magic to automatically reload packages (handy in development phase)
%load_ext autoreload
%autoreload 2

# %% choose directories and random seed
images_dir = "./figures"
models_dir = "./models"

seed = 2023
np.random.seed(seed)

# %% Synthesise initial configurations

# choose size and quantity. There is a virtually limitless supply.
N_train = 2**14
N_val = 2**14
N = 64

x_train = np.random.randint(2,size=(N_train, N, 1),
                            dtype=np.int8)
x_val = np.random.randint(2, size=(N_val, N, 1),
                          dtype=np.int8)

# %% Define models with 'perfect' weights and biases and calculate output of the model

# pick rules
rules = [15, 240]

# pick rule allocation
# rule_alloc = np.zeros(N)
# rule_alloc[N//2] = 1
rule_alloc = np.random.randint(0,2,size=N)
# np.ones(N)
# rule_alloc[0] = 1 # a single defect in first cell

# initiate model and make all 'labels'
timesteps=1
nuca_cnn = nuca_emulator_1D(N, rules, timesteps=timesteps, rule_alloc=rule_alloc, train_triplet_id=False)
nuca_cnn.summary()

nuca_cnn.compile()
r_train = nuca_cnn.predict(x_train, batch_size=N_train)
r_val = nuca_cnn.predict(x_val, batch_size=N_val)

# %% Output content of hidden layers

# Does not work yet properly yet. Check out TensorBoard!

# r_train, hidden_outputs = predict_with_hidden(nuca_cnn, x_train, output_hidden=True)

# submodel = tf.keras.Model(inputs=nuca_cnn.input,
#                           outputs=nuca_cnn.layers[1].output)

# r_triplet_id = submodel.predict(x_train, batch_size=N_train)

# %% Make and show a couple of spacetime diagrams

# safety switch (because this can take a while)
MAKE_AGAIN = True

if MAKE_AGAIN:

    T = N//2 # divide by two to avoid interference
    previous_configs = x_train.copy()
    diagrams = x_train.copy()

    start_time = time.time()
    time_array = [0]
    for t in range(1,T):
        print(f'Working on t={t}', end='\r')
        new_configs = nuca_cnn.predict(previous_configs, batch_size=N_train, verbose=0)
        diagrams = np.append(diagrams, new_configs, axis=2)
        previous_configs = new_configs
        time_since = time.time() - start_time
        time_array.append(time_since)

# %%  this process is clearly NOT linear in number of time steps! but ...
# four times more samples ~ 4 times more time
# four times more cells ~ 4 times more time
# that's pretty good news: we can simulate CAs with more cells without being bothered by interference problems.

plt.plot(time_array)
plt.plot([0, N//2-1], [time_array[0], time_array[-1]], 'k--')
plt.xlabel("Timestep")
plt.ylabel("Cumulative required time (seconds)")
plt.title(f"{len(rules)} rules, {N} cells, {N_train} samples")

# %% Show some diagrams

N_diagrams = 4
fig, axs = plt.subplots(N_diagrams, 1, figsize=(5, 12))
random_idxs = np.random.choice(N_train, size=N_diagrams)
axs[0].set_title(f"Rules: {rules}")
for idx, ax in zip(random_idxs, axs):
    ax.imshow(diagrams[idx].T, cmap='Greys', interpolation=None)
    ax.set_xticks([])
    ax.set_yticks([])


# %% Now randomly re-initialise the model with random weights and biases, and try to train it to perfection.

RUN_AGAIN = True
verbose=1

if RUN_AGAIN:
    # we don't know the rules, but we can indicate the number of rules
    rules = [None, None]
    # note that a batch size of 4096 is ridiculous, but we have a very particular data set.
    batch_size = 1024 # Gilpin chooses 10, that's not that much!
    batches_per_epoch = int(N_train / batch_size)
    epochs=500
    LR = 0.002
    stopping_patience = 20
    stopping_delta = 0.0001
    loss = 'mse' # tf.keras.losses.BinaryCrossentropy(from_logits=False) # bce (no logits) or mse

    # initialise model again
    nuca_cnn = nuca_emulator_1D(N, rules, train_triplet_id=False)
    nuca_cnn.summary()
    nuca_cnn.compile(loss=loss,
                optimizer=Adam(learning_rate=LR),
                metrics=['mse'])

    # callback for early termination of the training (plateau)
    stopping_callback = EarlyStopping(monitor='val_mse',
                                      patience=stopping_patience,
                                      min_delta=stopping_delta,
                                      verbose = 1,
                                      restore_best_weights=True)
    
    # keep history of the output for nice plotting
    weights_biases_callback = WeightsBiasesHistory()

    # fit the model. This is the big one.
    history = nuca_cnn.fit(x_train,
                      r_train.astype(np.float32),
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=verbose,
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
    nuca_cnn.set_weights(wab_epoch)

    r_pred = nuca_cnn.predict(x_train_example, verbose=0)[0]
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
axs[3,0].set_title(f"Desired output")

for i in range(4):
    for j in range(2):
        axs[i,j].set_yticks([])
        axs[i,j].set_xticks([])

plt.tight_layout()

# savename = f"cnn-eca-convergence_rule-{ECA_rule}_timesteps-{timesteps}.png"
# plt.savefig(images_dir + '/' + savename, bbox_inches='tight')

# %% Check out TensorBoard for inspecting node values of hidden layers!