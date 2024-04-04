# %% import packages

# classic
import numpy as np

# custom
import sys
sys.path.insert(0, '..') # TODO: this is probably not the right way to do this
from src.nn.eca import EcaEmulator
from src.train.train import Train1D
from src.visual.histories import History1D

%load_ext autoreload
%autoreload 2

# choose appropriate directory
dir_figs = '../figures/eca/'

# %% test eca emulator

N = 64
rule = 110
timesteps = 1
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

# %% pretraining and training
# TODO: can pretraining be parallellised?
# TODO: can pretraining go in a class as well?

PRETRAIN = True
TRAIN = True

# pretraining params
N_pt = 50
batch_size_pt = 128

# training params
batch_size = 8
epochs = 40
learning_rate = 0.0005
loss = 'mse'
stopping_patience = None # 20
stopping_delta = None # 0.0001
wab_callback = True

if PRETRAIN:
    best_loss = np.infty
    models = [ECA.model() for _ in range(N_pt)]
    for i in range(N_pt):
        print(f'Working on pretraining {i+1}/{N_pt}. Best loss: {best_loss}.', end='\r')
        current_model = models[i]
        tr = Train1D(current_model, x_train, r_train, x_val, r_val,
                     batch_size=batch_size_pt, epochs=1,
                     learning_rate=learning_rate, loss=loss)
        history = tr.train(verbose=False)
        current_loss = history.history['loss'][0]
        if current_loss < best_loss:
            best_loss = current_loss
            best_model = current_model
    print(f'Best pretraining loss: {best_loss}.')
    model = best_model

if TRAIN:
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
    savename = f"plot_configs_ECA_rule{rule}_{epochs}epochs_bs{batch_size}_lr{str(learning_rate).replace('.','p')}.pdf"
    plt.savefig(dir_figs+savename, bbox_inches='tight')