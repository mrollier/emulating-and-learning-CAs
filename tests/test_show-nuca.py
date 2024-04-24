# %% import packages

# change system path
import sys
sys.path.insert(0, '..') # TODO: this is probably not the right way to do this

# classic
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
plt.rcParams['text.usetex'] = True

# particular
import cellpylib as cpl

%load_ext autoreload
%autoreload 2

# choose appropriate directory
dir_figs = '../../figures/nuca/'

# %%

# dimensions
N = 32
T = 32

# states and rules
k=2
Nrules = 2
# rules = np.random.randint(0,256,Nrules)
rules = [30,90]
rules = np.sort(rules)
print(f"The rules are {rules}.")

# initial configuration
init_config = np.random.randint(k,size=(1,N))

# rule alloc alternates over time
rule_alloc_2D = np.zeros((N,T))
rule_alloc_2D[::2] = 1
rule_alloc_2D = np.array(rule_alloc_2D, dtype=int)

# %% initialise and run cellpylib

diagram_cpl = cpl.evolve(
    init_config, timesteps=T,
    apply_rule=lambda n, c, t: cpl.nks_rule(
        n,
        rules[rule_alloc_2D[t-1,c]]),
        memoize=False)

# %% plot

SAVEFIG = False
savename = f"acri-example-rules{rules[0]}_{rules[1]}.pdf"

labelsize=14
fig, axs = plt.subplots(1,2,figsize=(5,3))
dividers = [make_axes_locatable(ax) for ax in axs]
caxs = [divider.append_axes('right', size='5%', pad=0.1) for divider in dividers]

cmap_colours = [(1,1,1),(30/255,100/255,200/255)]
cmap_ra = ListedColormap(cmap_colours)
cmap_se = plt.get_cmap('Greys', 2)

ra = axs[0].imshow(rule_alloc_2D, cmap=cmap_ra)
se = axs[1].imshow(diagram_cpl, cmap=cmap_se)

ticks_ra = [1/4, 3/4]
ticks_se = [1/4, 3/4]
cbar_ra = plt.colorbar(ra, ax=axs[0], cax=caxs[0], ticks=ticks_ra)
cbar_ra.set_ticklabels(rules, size=labelsize)
cbar_se = plt.colorbar(se, ax=axs[1], cax=caxs[1], ticks=ticks_se)
cbar_se.set_ticklabels([0,1], size=labelsize)

axs[0].set_title('Rule allocation', size=labelsize)
axs[1].set_title('State evolution', size=labelsize)

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(f"$\\leftarrow$ Time", size=labelsize)

# set the spacing between subplots
plt.subplots_adjust(wspace=0.45)

if SAVEFIG:
    plt.savefig(dir_figs+savename, bbox_inches='tight')
# %%
