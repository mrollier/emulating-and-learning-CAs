# %% import packages

# classic
import numpy as np
# from functools import partial

# custom
import sys
sys.path.insert(0, '..') # TODO: this is probably not the right way to do this
from src.visual.decomposition import DecompositionECA

%load_ext autoreload
%autoreload 2

# choose appropriate directory
dir_figs = '../figures/eca/'

# %% plot

input = np.random.randint(0,2,64)
rule = 110

decomposition = DecompositionECA(input, rule)
fig, ax = decomposition.plot()
# %%
