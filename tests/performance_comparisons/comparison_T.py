### comparing the performance of the various models

# %% import packages

# change system path
import sys
sys.path.insert(0, '../..') # TODO: this is probably not the right way to do this

# classic
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['text.usetex'] = True

# particular
import cellpylib as cpl
import time

# custom
from src.nn.nuca import NucaEmulator

%load_ext autoreload
%autoreload 2

# choose appropriate directory
dir_figs = '../../figures/nuca/'
dir_data = '../../data/nuca/'

# %% static values

# test size
test_per_setting = 10

# dimensions (static)
k=2
N = 64
Nrules = 4
rules = np.sort(np.random.choice(
    range(256),
    size=Nrules,
    replace=False))
rule_alloc = np.random.randint(
    Nrules,size=N)
S = 32

# %% dynamic values

T_list = np.linspace(10, 100, 10, dtype=int)

# initial configuration (sample-dependent)
init_configs = np.random.randint(k,size=(S,1,N))

# %% initialise and run cellpylib

time_deltas_cpl_list = []
for test_idx in range(test_per_setting):
    time_deltas_cpl = []
    for T in T_list:
        print(f"Working on test number {test_idx+1}/{test_per_setting}", end='\r')
        time_start = time.time()
        for init_config in init_configs:
            diagram_cpl = cpl.evolve(
                init_config, timesteps=T,
                apply_rule=lambda n, c, t: cpl.nks_rule(
                    n,
                    rules[rule_alloc[c]]),
                    memoize=False)
        time_end = time.time()
        time_delta = time_end - time_start
        time_deltas_cpl.append(time_delta)
    time_deltas_cpl_list.append(time_deltas_cpl)
time_deltas_cpl_array = np.array(time_deltas_cpl_list)

# %% plot results

cpl_means = np.mean(time_deltas_cpl_array, axis=0)
cpl_errors = np.std(time_deltas_cpl_array, axis=0)
plt.errorbar(T_list, cpl_means, yerr=cpl_errors, fmt='o', capsize=5)
# plt.xscale('log', base=2)

# %% initialise CNN class

# model with perfect weights and biases
train_triplet_id = False
nuCA = NucaEmulator(
    N, rules=rules, timesteps=1,
    activation=None,
    train_triplet_id=train_triplet_id,
    rule_alloc=rule_alloc)

# %% run locally connected CNNs

time_deltas_cnn_lc_list = []
for test_idx in range(test_per_setting):
    time_deltas_cnn_lc = []
    for T in T_list:
        cnn_lc = nuCA.model()
        input = np.transpose(init_configs, (0, 2, 1))
        diagram_cnn_lc = input
        print(f"Working on test number {test_idx+1}/{test_per_setting}", end='\r')
        # start the clock
        time_start = time.time()
        for t in range(T-1):
            output = cnn_lc.predict(
                diagram_cnn_lc[:,:,-1],
                verbose=False)
            diagram_cnn_lc = np.append(
                diagram_cnn_lc,
                output, axis=2)
        time_end = time.time()
        time_delta = time_end - time_start
        time_deltas_cnn_lc.append(time_delta)
    time_deltas_cnn_lc_list.append(time_deltas_cnn_lc)
time_deltas_cnn_lc_array = np.array(time_deltas_cnn_lc_list)

# %% plot results

cnn_lc_means = np.mean(time_deltas_cnn_lc_array, axis=0)
cnn_lc_errors = np.std(time_deltas_cnn_lc_array, axis=0)
plt.errorbar(T_list, cnn_lc_means, yerr=cnn_lc_errors, fmt='o', capsize=5)
# plt.xscale('log', base=2)


# %% run locally connected CNNs

time_deltas_cnn_dense_list = []
for test_idx in range(test_per_setting):
    time_deltas_cnn_dense = []
    for T in T_list:
        cnn_dense = nuCA.model_dense()
        input = np.transpose(init_configs, (0, 2, 1))
        diagram_cnn_dense = input
        print(f"Working on test number {test_idx+1}/{test_per_setting}", end='\r')
        # start the clock
        time_start = time.time()
        for t in range(T-1):
            output = cnn_dense.predict(
                diagram_cnn_dense[:,:,-1],
                verbose=False)
            diagram_cnn_dense = np.append(
                diagram_cnn_dense,
                output, axis=2)
        time_end = time.time()
        time_delta = time_end - time_start
        time_deltas_cnn_dense.append(time_delta)
    time_deltas_cnn_dense_list.append(time_deltas_cnn_dense)
time_deltas_cnn_dense_array = np.array(time_deltas_cnn_dense_list)

# %% plot results

cnn_dense_means = np.mean(time_deltas_cnn_dense_array, axis=0)
cnn_dense_errors = np.std(time_deltas_cnn_dense_array, axis=0)
plt.errorbar(T_list, cnn_dense_means, yerr=cnn_dense_errors, fmt='o', capsize=5)
# plt.xscale('log', base=2)


# %% master plot (full comparison)

SAVEFIG=True
savename=f"nuca-comparison-T-N{N}_Nrules{Nrules}_S{S}_avg-from-{test_per_setting}.pdf"

# setup
width=7; height=3
labelsize = 14
fig, ax = plt.subplots(1,1,figsize=(width,height))
jitter_factor = 1

# plot
ax.errorbar(T_list - jitter_factor, cpl_means, yerr=cpl_errors, fmt='+', capsize=5, label='CellPyLib')
ax.errorbar(T_list, cnn_lc_means, yerr=cnn_lc_errors, fmt='o', capsize=5, label='CNN (locally connected)')
ax.errorbar(T_list + jitter_factor, cnn_dense_means, yerr=cnn_dense_errors, fmt='x', capsize=5, label='CNN (densely connected)')

# aesthetics and labels
ax.set_title(f"Computation time for {S} $\\nu$CAs of {N} cells and {Nrules} rules", size=labelsize+2)
# ax.set_xscale('log', base=2)
ax.legend(ncols=3)
ax.set_yticks([0, 2, 4, 6, 8])
ax.set_xlim([0, 110])
ax.set_xlabel(f'Number of timesteps $T$', size=labelsize)
ax.set_ylabel(f'Time to compute (s)', size=labelsize)
ax.tick_params(axis='both', which='major', labelsize=labelsize-2)

if SAVEFIG:
    plt.savefig(dir_figs+savename, bbox_inches='tight')

# %% save data for future reference

dataname = f"nuca-comparison-T-N{N}_Nrules{Nrules}_S{S}_avg-from-{test_per_setting}.npy"

# with open(dir_data+dataname, 'wb') as f:
#     # note the order!
#     np.save(f, T_list)
#     np.save(f, time_deltas_cpl_array)
#     np.save(f, time_deltas_cnn_lc_array)
#     np.save(f, time_deltas_cnn_dense_array)

with open(dir_data+dataname, 'rb') as f:
    T_list = np.load(f)
    time_deltas_cpl_array = np.load(f)
    time_deltas_cnn_lc_array = np.load(f)
    time_deltas_cnn_dense_array = np.load(f)

# calculate relevant quantities
cpl_means = np.mean(time_deltas_cpl_array, axis=0)
cpl_errors = np.std(time_deltas_cpl_array, axis=0)

cnn_lc_means = np.mean(time_deltas_cnn_lc_array, axis=0)
cnn_lc_errors = np.std(time_deltas_cnn_lc_array, axis=0)

cnn_dense_means = np.mean(time_deltas_cnn_dense_array, axis=0)
cnn_dense_errors = np.std(time_deltas_cnn_dense_array, axis=0)
