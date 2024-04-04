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
dir_files = '../files/eca/'

# %% optimise eca emulator for all 256 rules

