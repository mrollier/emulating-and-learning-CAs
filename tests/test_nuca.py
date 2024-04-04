# %% import packages

# classic
import numpy as np

# custom
import sys
sys.path.insert(0, '..') # TODO: this is probably not the right way to do this
from src.nn.nuca import NucaEmulator
from src.train.train import Train1D
from src.visual.histories import History1D

%load_ext autoreload
%autoreload 2

# choose appropriate directory
dir_figs = '../figures/nuca/'

# %% test nuca emulator

N = 64
rules = [110, 54]
rule_alloc = np.random.randint(0,2,size=N)
timesteps = 1
activation = None

# model with perfect weights and biases
train_triplet_id = False
nuCA = NucaEmulator(N, rules=rules, timesteps=timesteps,
                  activation=activation, train_triplet_id=train_triplet_id,
                  rule_alloc=rule_alloc)
model_perfect = nuCA.model()

# model with random weights and biases, and activation function (for training)
nuCA.rules = None
nuCA.activation = 'tanh'
nuCA.train_triplet_id = True
nuCA.rule_alloc = None
model = nuCA.model()
model.summary()

# %% define inputs and outputs

N_train = 2**12
N_val = 2**12

# inputs
x_train = np.random.randint(2,size=(N_train, N, 1),
                            dtype=np.int8)
x_val = np.random.randint(2, size=(N_val, N, 1),
                          dtype=np.int8)

# desired outputs
verbose = False
r_train = model_perfect.predict(x_train, batch_size=len(x_train), verbose=verbose)
r_val = model_perfect.predict(x_val, batch_size=len(x_val), verbose=verbose)

# %% training

TRAIN_AGAIN = True

# training params
batch_size = 32
epochs = 100
learning_rate = 0.0005
loss = 'mse'
stopping_patience = 20
stopping_delta = 0.0001
wab_callback = True

if TRAIN_AGAIN:
    tr = Train1D(model, x_train, r_train, x_val, r_val,
                 batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, loss=loss,stopping_patience=stopping_patience, stopping_delta=stopping_delta, wab_callback=wab_callback)
    if wab_callback:
        history, wab_history = tr.train()
    else:
        history = tr.train()

# %% visualisation

SAVE_FIG = True

idx_ex = np.random.randint(N_train)
input_example = x_train[idx_ex]
output_example = r_train[idx_ex]

hist = History1D(history, model=model, wab_history=wab_history)
rules_str = f'{rules[0]}-{rules[1]}'
fig, ax = hist.plot_configs(input_example, output_example, rules_str, timesteps)

if SAVE_FIG:
    import matplotlib.pyplot as plt
    lr_str = str(learning_rate).replace('.','p')
    savename = f"plot_configs_nuCA_rules{rules_str}_{epochs}epochs_bs{batch_size}_lr{lr_str}.pdf"
    plt.savefig(dir_figs+savename, bbox_inches='tight')