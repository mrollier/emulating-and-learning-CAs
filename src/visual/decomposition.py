# %% import packages

# classics
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

# custom
from src.nn.eca import EcaEmulator
from src.utils.ruletable import base_repr

# %% define class
class DecompositionECA:
    """
    Class used to show the decomposition and re-composition performed by the simple ECA emulator
    """
    def __init__(self, input, rule):
        self.input = input[np.newaxis,:,np.newaxis]
        self.N = len(input)
        self.rule = rule
        # calculate output
        self.output = self.input_to_output(
            self.input,
            self.N,
            self.rule)
        # calculate nbhs as grey value
        self.nbh_int = self.input_to_nbh_int(
            input)
        # calculate one-hot matrix
        self.one_hot = self.nbh_int_to_one_hot(
            self.nbh_int,
            self.N)
        # calculate binary repr of local update rule
        self.rule_table = self.rule_to_table(rule)
        # calculate output matrix (unsummed)
        self.update = self.one_hot_to_update(
            self.one_hot,
            self.rule_table)
        
    def plot(self,saveas=None):
        title_size = 14
        width=5; width_ratios=[1,self.N]
        height=5; height_ratios=[1, 1, 3, 3, 1]
        cmap='Greys'
        fig, axs = plt.subplots(
            len(height_ratios),len(width_ratios),
            height_ratios=height_ratios,
            width_ratios=width_ratios,
            figsize=(width,height))
        axs[0,1].set_title("Input configuration", size=title_size)
        axs[0,1].imshow(self.input[0].T, cmap=cmap)
        axs[1,1].set_title("Integer neighbourhood encoding", size=title_size)
        axs[1,1].imshow(self.nbh_int[0].T, cmap=cmap)
        axs[2,0].set_ylabel(f"Rule {self.rule}", size=title_size+2)
        axs[2,0].imshow(self.rule_table.T, cmap=cmap)
        axs[2,1].set_title("One-hot neighbourhood encoding", size=title_size)
        axs[2,1].imshow(self.one_hot[0].T, cmap=cmap)
        axs[3,1].set_title("Output of neighbourhood after local update", size=title_size)
        axs[3,1].imshow(self.update[0].T, cmap=cmap)
        axs[4,1].set_title("Output configuration", size=title_size)
        axs[4,1].imshow(self.output[0].T, cmap=cmap)
        # cleanup
        for ax_col in axs:
            for ax in ax_col:
                ax.set_xticks([])
                ax.set_yticks([])
        for ax in [axs[0,0], axs[1,0], axs[3,0], axs[4,0]]:
            ax.remove()
        # save
        if saveas:
            plt.savefig(saveas, bbox_inches='tight')
        return fig, axs
    
    def input_to_nbh_int(self, input):
        input_periodic = np.concatenate(
            (input[-1:], input, input[:1]))
        kernel = [1, 2, 4]
        nbh_int = np.convolve(
            input_periodic,
            kernel,
            mode='valid')
        nbh_int = nbh_int[np.newaxis,:,np.newaxis]
        return nbh_int
    
    def nbh_int_to_one_hot(self, nbh_int, N):
        nbh_int = nbh_int[0,:,0]
        one_hot = np.zeros((N, 8))
        one_hot[np.arange(N), nbh_int] = 1
        one_hot = one_hot[np.newaxis,:,:]
        return one_hot
    
    def one_hot_to_update(self, one_hot, rule_table):
        one_hot = one_hot[0]
        update = rule_table[:,np.newaxis] * one_hot
        update = update[np.newaxis,:,:]
        return update

    def input_to_output(self, input, N, rule):
        ECA = EcaEmulator(
            N,
            rule=rule,
            train_triplet_id=False)
        model_perfect = ECA.model()
        output = model_perfect.predict(
            input,
            verbose=False
            )
        return output
    
    def rule_to_table(self, rule):
        rule_as_str = base_repr(rule)
        rule_as_list = list(rule_as_str)
        rule_table = np.array([eval(char) for char in rule_as_list])
        rule_table = rule_table[np.newaxis,::-1]
        return rule_table