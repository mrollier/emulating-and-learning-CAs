# %% import packages

from src.nn.eca import EcaEmulator
import numpy as np
import matplotlib.pyplot as plt

# %% define class
class DecompositionECA:
    def __init__(self, input, rule):
        self.input = input
        self.N = len(self.input)
        self.rule = rule
        # calculate output
        ECA = EcaEmulator(self.N, rule=self.rule)
        model_perfect = ECA.model()
        self.output = model_perfect(
            self.input[np.newaxis,:,np.newaxis],
            verbose=False
            )[0,:,0]
        
    def plot(self):
        width=5
        height=5
        fig, axs = plt.subplots(2,1,size=(width,height))
        axs[0].imshow(self.input)
        axs[1].imshow(self.output)
        return fig, axs