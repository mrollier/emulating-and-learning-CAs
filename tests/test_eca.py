# %% import packages

# classic
import numpy as np

# custom
from src.nn.eca import EcaEmulator
from src.train.train import Train1D
from src.visual.histories import History1D

%load_ext autoreload
%autoreload 2

# choose appropriate directory
dir_figs = './figures/eca/'

# %% test eca emulator

N = 64
rule = 42
timesteps = 1
activation = None

# model with perfect weights and biases
train_triplet_id = False
ECA = EcaEmulator(N, rule=rule, timesteps=timesteps,
                  activation=activation, train_triplet_id=train_triplet_id)
model_perfect = ECA.model()

# model with random weights and biases, and activation function (for training)
ECA.rule = None
ECA.activation = 'tanh'
ECA.train_triplet_id = True
model = ECA.model()
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
learning_rate = 0.001
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
fig, ax = hist.plot_configs(input_example, output_example, rule, timesteps)

if SAVE_FIG:
    import matplotlib.pyplot as plt
    savename = f"plot_configs_ECA_rule{rule}_{epochs}epochs_bs{batch_size}_lr{learning_rate}.pdf"
    plt.savefig(dir_figs+savename, bbox_inches='tight')