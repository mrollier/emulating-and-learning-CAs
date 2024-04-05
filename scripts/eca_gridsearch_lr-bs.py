# %% import packages

# classic
import numpy as np
import matplotlib.pyplot as plt

# custom
import sys
sys.path.insert(0, '..') # TODO: this is probably not the right way to do this
from src.nn.eca import EcaEmulator
from src.train.train import Train1D
from src.visual.histories import History1D

%load_ext autoreload
%autoreload 2

# choose appropriate directory
dir_figs = '../figures/eca/grid_search/'
dir_files = '../files/models/eca/'

# %% set params

N = 32
N_train = 2**12
N_val = 2**12
timesteps = 1
rule = 110 # preferably a rule that is generally difficult

# skip pretraining?
TRAIN = True

SAVE_FIG = True

# training params
N_batch = 12
batch_sizes = np.logspace(2, 7, N_batch, base=2, dtype=int)
N_lr = 14
learning_rates = np.logspace(-5, -2, N_lr, base=10)

epochs = 40
loss = 'mse'
stopping_patience = None # 20
stopping_delta = None # 0.0001

# model init

activation = None
# model with perfect weights and biases
train_triplet_id = False
ECA = EcaEmulator(N, rule=rule, timesteps=timesteps,
                activation=activation, train_triplet_id=train_triplet_id)
model_perfect = ECA.model()

# model with random weights and biases, and activation function (for training)
ECA.rule = None
ECA.activation = 'tanh' # a modified sigmoid may be better
ECA.train_triplet_id = True

# inputs
x_train = np.random.randint(2,size=(N_train, N, 1),
                            dtype=np.int8)
x_val = np.random.randint(2, size=(N_val, N, 1),
                        dtype=np.int8)

# desired outputs
verbose = False
r_train = model_perfect.predict(x_train, batch_size=len(x_train), verbose=verbose)
r_val = model_perfect.predict(x_val, batch_size=len(x_val), verbose=verbose)

# %% grid search

verbose=False
losses = np.zeros((N_batch, N_lr))
for i, batch_size in enumerate(batch_sizes):
    for j, learning_rate in enumerate(learning_rates):
        print(f"Working on batch_size {batch_size}, learning_rate {learning_rate}.")
        # re-initialise model
        model = ECA.model()
        # train
        tr = Train1D(model, x_train, r_train, x_val, r_val,
                    batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, loss=loss,stopping_patience=stopping_patience, stopping_delta=stopping_delta)
        history = tr.train(verbose=verbose)
        current_loss = history.history['val_loss'][-1]
        losses[i,j] = current_loss
        print(f"Final val_loss: {current_loss}.\n")
# %% plot losses matrix

L = np.array(losses)

# Plotting the heatmap
plt.figure(figsize=(10, 6))
plt.imshow(L, cmap='RdYlGn_r', interpolation='nearest')

# Adding labels to axes
plt.xticks(np.arange(N_lr), ['{:.0e}'.format(lr) for lr in learning_rates], rotation=45)
plt.yticks(np.arange(N_batch), batch_sizes)
plt.xlabel('Learning Rates')
plt.ylabel('Batch Sizes')
plt.colorbar(label='Losses')

plt.title(f'Losses for rule {rule}, {epochs} epochs')

savename = f"grid_search_losses-bs-lr_rule{rule}_{epochs}epochs.pdf"
plt.savefig(dir_figs+savename, bbox_inches='tight')