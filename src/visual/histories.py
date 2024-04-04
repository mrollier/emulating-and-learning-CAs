# %% import packages

# existing
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from matplotlib.colors import SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# custom

# %% history visualisation classes

class History1D:
    def __init__(self, history, model=None, wab_history=None):
        self.history = history
        self.model = model
        self.wab_history = wab_history
        
        # exceptions
        if (model is not None and wab_history is None) or (model is None and wab_history is not None):
            raise Exception("When visualising the configurations over the epochs, both model and wab_history are required.")
        
    def plot_loss(self):
        # TODO: add function that simply plots loss vs epochs of training and val set
        pass

    def plot_configs(self, input_example, output_example, rule, timesteps):
        # TODO: increase flexibility of plot layout
        if self.wab_history is None:
            raise Exception("The plots_configs function requires the History1D initialisation with wab_history.")
        # add batch dimension in case this has been forgotten
        if len(input_example.shape)==2:
            input_example = input_example[np.newaxis]
        if len(output_example.shape)==2:
            output_example = output_example[np.newaxis]

        # make plot layout
        fig, axs = plt.subplots(4,2, figsize=(5, 3),
                                width_ratios=[5,1], height_ratios=[1, 7, 1, 1])
        axs[0,1].remove()
        axs[2,1].remove()
        axs[3,1].remove()

        # input array
        axs[0,0].imshow(input_example, cmap='Greys')
        axs[0,0].set_title("Input configuration")

        # predictions over epochs
        output_pred_history = []
        mses = []
        for wab_epoch in self.wab_history:
        # add historical weights and biases to CNN
            self.model.set_weights(wab_epoch)

            output_pred = self.model.predict(input_example, verbose=0)[0]
            output_pred_history += [output_pred]

            mse = np.square(output_pred - output_example).mean()
            mses += [mse]
        output_pred_history = np.array(output_pred_history)
        mses = np.array(mses)

        # training
        # try:
        error_history = np.abs(output_pred_history - output_example)
        # except:
        #     square_error_history = (np.expand_dims(r_pred_history,axis=2) - r_train_example)**2

        # errors
        im = axs[1,0].imshow(error_history, cmap='RdYlGn_r',
                    norm=SymLogNorm(1e-3, vmin=0, vmax=1), interpolation='none')
        # prediction history
        # im = axs[1,0].imshow(r_pred_history, cmap='Greys',
        #               norm=SymLogNorm(1e-4, vmin=0, vmax=1), interpolation='none')

        cbar_ticks = [0, 1e-3, 1e-2, 1e-1, 1]
        cbar_tick_labels = [r'$0$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$1$']
        cbar = fig.colorbar(im, ax=axs[1,0], orientation='horizontal', aspect=75)
        cbar.ax.set_xticks(cbar_ticks,minor=False)
        # cbar.ax.tick_params(length=0, which='major', color='white')
        cbar.ax.set_xticklabels(cbar_tick_labels,
                                fontdict={'fontsize':8})
        axs[1,0].set_ylabel(r"$\leftarrow$ Epochs")
        axs[1,0].set_title(f"CNN model convergence over {len(mses)} epochs")
        axs[1,0].set_aspect('auto')

        # CNN output
        try:
            axs[2,0].imshow(output_pred_history[-1].T, cmap='Greys')
        except:
            axs[2,0].imshow(np.expand_dims(output_pred_history[-1],axis=1).T, cmap='Greys')
        axs[2,0].set_title("CNN model final output")

        # MSE history
        axs[1,1].plot(mses, np.arange(len(mses),0,-1), color='k', lw=1)
        axs[1,1].set_xscale('log')
        axs[1,1].set_title(r"MSE (log)")

        vert_shift = .4*len(mses)
        axs[1,1].set_ylim([-vert_shift, len(mses)+1])

        axs[1,1].set_axis_off()

        # output array
        axs[3,0].imshow(output_example, cmap='Greys')
        axs[3,0].set_title(f"Desired output, rule(s) {rule}, {timesteps} timestep(s)")

        for i in range(4):
            for j in range(2):
                axs[i,j].set_yticks([])
                axs[i,j].set_xticks([])

        plt.tight_layout()
        return fig, axs