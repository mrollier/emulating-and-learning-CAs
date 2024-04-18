# %% import packages

from src.nn.eca import EcaEmulator
import numpy as np
import matplotlib.pyplot as plt

# %% define class
class DecompositionECA:
    def __init__(self, input, rule):
        self.input = input[np.newaxis,:,np.newaxis]
        self.N = len(input)
        self.rule = rule
        # calculate output
        ECA = EcaEmulator(self.N, rule=self.rule)
        model_perfect = ECA.model()
        self.output = model_perfect.predict(
            self.input,
            verbose=False
            )
        
    def plot(self):
        width=5
        height=5
        fig, axs = plt.subplots(2,1,figsize=(width,height))
        axs[0].imshow(self.input[0].T)
        axs[1].imshow(self.output[0].T)
        return fig, axs