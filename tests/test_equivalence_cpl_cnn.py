# %% import packages

# change system path
import sys
sys.path.insert(0, '..') # TODO: this is probably not the right way to do this

# classic
import numpy as np

# particular
import cellpylib as cpl

# custom
from src.nn.nuca import NucaEmulator

%load_ext autoreload
%autoreload 2

# choose appropriate directory
dir_figs = '../figures/nuca/'

# %% test nuca emulator

# dimensions
N = 32
T = 32

# states and rules
k=2
Nrules = 3
rules = np.sort(np.random.randint(0,256,Nrules))
print(f"The rules are {rules}.")

# initial configuration
init_config = np.random.randint(k,size=(1,N))

# rule allocation (index of rules list)
rule_alloc = np.random.randint(Nrules,size=N)

# does not change over time
# rule_alloc_2D = np.tile(init_rule_alloc, (N,1))

# %% initialise and run cellpylib

diagram_cpl = cpl.evolve(
    init_config, timesteps=T,
    apply_rule=lambda n, c, t: cpl.nks_rule(
        n,
        rules[rule_alloc[c]]),
        memoize=False)

# plt.imshow(diagram_cpl, cmap='Greys')

# %% initialise and run CNNs

# model with perfect weights and biases
train_triplet_id = False
nuCA = NucaEmulator(
    N, rules=rules, timesteps=T,
    activation=None,
    train_triplet_id=train_triplet_id,
    rule_alloc=init_rule_alloc)

cnn_loc_connected = nuCA.model()
# cnn_sparse_dense = nuCA.model_dense()
# %%
