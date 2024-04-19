# %% import packages

# change system path
import sys
sys.path.insert(0, '..') # TODO: this is probably not the right way to do this

# classic
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['text.usetex'] = True

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
Nrules = 8
rules = np.sort(np.random.randint(0,256,Nrules))
print(f"The rules are {rules}.")

# initial configuration
init_config = np.random.randint(k,size=(1,N))

# rule allocation (index of rules list)
rule_alloc = np.random.randint(Nrules,size=N)

# does not change over time
# rule_alloc_2D = np.tile(init_rule_alloc, (
# N,1))

# %% initialise and run cellpylib

diagram_cpl = cpl.evolve(
    init_config, timesteps=T,
    apply_rule=lambda n, c, t: cpl.nks_rule(
        n,
        rules[rule_alloc[c]]),
        memoize=False)

# plt.imshow(diagram_cpl, cmap='Greys')

# %% initialise CNNs

# model with perfect weights and biases
train_triplet_id = False
nuCA = NucaEmulator(
    N, rules=rules, timesteps=1,
    activation=None,
    train_triplet_id=train_triplet_id,
    rule_alloc=rule_alloc)

cnn_loc_connected = nuCA.model()
cnn_sparse_dense = nuCA.model_dense()

# %% run CNNs

input = init_config.T[np.newaxis]
diagram_cnn_loc = input
diagram_cnn_dense = input
for t in range(T-1):
    output = cnn_loc_connected.predict(
        diagram_cnn_loc[:,:,-1],
        verbose=False)
    diagram_cnn_loc = np.append(
        diagram_cnn_loc,
        output, axis=2)
    output = cnn_sparse_dense.predict(
        diagram_cnn_dense[:,:,-1],
        verbose=False)
    diagram_cnn_dense = np.append(
        diagram_cnn_dense,
        output, axis=2)

# %% plot results

# prepare figure
width=15; height=5
fig, axs = plt.subplots(1,3,figsize=(width,height))

# plot diagrams
axs[0].imshow(diagram_cpl, cmap='Greys')
axs[1].imshow(diagram_cnn_loc[0].T, cmap='Greys')
axs[2].imshow(diagram_cnn_dense[0].T, cmap='Greys')

# aesthetics and information
labelsize=18
axs[0].set_title(f"Evolved with CellPyLib", size=labelsize+4)
axs[1].set_title(f"CNN with locally connected layer", size=labelsize+4)
axs[2].set_title(f"CNN with dense layer", size=labelsize+4)
for ax in axs:
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel(f"$\\leftarrow$ Time", size=labelsize)

fig.suptitle(f"Three different techniques for simulating a $\\nu$CA with rules {rules}", size=labelsize+8)