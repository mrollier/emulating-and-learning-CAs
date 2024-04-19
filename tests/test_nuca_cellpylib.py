# %% import packages

# classics
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['text.usetex'] = True

# specifics
import cellpylib as cpl

# choose appropriate directory
dir_figs = '../figures/nuca/'

# %% nuCA definition

# dimensions
N = 32
T = 32

# states and rules
k=2
Nrules = 8
rules = np.sort(np.random.randint(0,256,Nrules))

# initial configuration
init_config = np.random.randint(k,size=(1,N))

# rule allocation (index of rules list)
init_rule_alloc = np.random.randint(Nrules,size=(1,N))

# does not change over time
rule_alloc = np.tile(init_rule_alloc, (N,1))

# shifted to the right over time
rule_alloc = np.array([np.roll(rule_alloc[row_idx], -row_idx) for row_idx in range(T)])
# %% nuCA simulation

spacetime_diagram = cpl.evolve(
    init_config, timesteps=T,
    apply_rule=lambda n, c, t: cpl.nks_rule(
        n,
        rules[rule_alloc[t-1,c]]),
        memoize=True)

# %% plot nuCA spacetime diagram
# TODO: currently only renders nicely with 8 rules

SAVEFIG=False
savename=f"cellpylib-spacetime_diagram-Nrules{Nrules}.pdf"

fig, axs = plt.subplots(1,2,figsize=(11,5))
dividers = [make_axes_locatable(ax) for ax in axs]
caxs = [divider.append_axes('right', size='5%', pad=0.1) for divider in dividers]

alloc_map = axs[0].imshow(rule_alloc, cmap='Set2')
cb_alloc = plt.colorbar(alloc_map, ax=axs[0], cax=caxs[0])
state_map = axs[1].imshow(spacetime_diagram, cmap='Greys')
cb_state = plt.colorbar(state_map, cax=caxs[1])

# aesthetics
labelsize=18
cbar_ticks = np.linspace(0,Nrules-1,2*Nrules+1)[1::2]
cb_alloc.set_ticks(cbar_ticks)
cb_alloc.set_ticklabels(rules, size=labelsize-4)
cb_state.remove()
axs[0].set_title(f'Rule allocation ({Nrules} rules)', size=labelsize+4)
axs[1].set_title(f'Spacetime diagram of the $\\nu$CA', size=labelsize+4)
for ax in axs:
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel(f"$\\leftarrow$ Time", size=labelsize)

if SAVEFIG:
    plt.savefig(dir_figs+savename, bbox_inches='tight')

# %%
