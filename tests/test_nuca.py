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
rules = [255, 0, 0]
Nrules = len(rules)
# rule_alloc = np.random.randint(0,Nrules,size=N)
rule_alloc = [2]*32 + [0]*32
timesteps = 1
activation = None

# model with perfect weights and biases
train_triplet_id = False
nuCA = NucaEmulator(N, rules=rules, timesteps=timesteps,
                  activation=activation, train_triplet_id=train_triplet_id,
                  rule_alloc=rule_alloc)
model_perfect = nuCA.model_dense()

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

# %% pretraining and training
# TODO: can pretraining be parallellised?
# TODO: can pretraining go in a class as well?

TRAIN = True
PRETRAIN = True

# pretraining params
N_pt = 50
batch_size_pt = 128

# training params
batch_size = 32
epochs = 40
learning_rate = 0.0005
loss = 'mse'
stopping_patience = None # 20
stopping_delta = None # 0.0001
wab_callback = True

if PRETRAIN:
    best_loss = np.infty
    models = [nuCA.model() for _ in range(N_pt)]
    for i in range(N_pt):
        print(f'Working on pretraining {i+1}/{N_pt}. Best loss: {best_loss}.         ', end='\r')
        current_model = models[i]
        tr = Train1D(current_model, x_train, r_train, x_val, r_val,
                     batch_size=batch_size_pt, epochs=1,
                     learning_rate=learning_rate, loss=loss)
        history = tr.train(verbose=False)
        current_loss = history.history['loss'][0]
        if current_loss < best_loss:
            best_loss = current_loss
            best_model = current_model
    print('')
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
rules_str = f'{rules[0]}-{rules[1]}'
fig, ax = hist.plot_configs(input_example, output_example, rules_str, timesteps)

if SAVE_FIG:
    import matplotlib.pyplot as plt
    lr_str = str(learning_rate).replace('.','p')
    savename = f"plot_configs_nuCA_{N}cells_rules{rules_str}_{epochs}epochs_bs{batch_size}_lr{lr_str}.pdf"
    plt.savefig(dir_figs+savename, bbox_inches='tight')