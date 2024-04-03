# %% import packages

import numpy as np
from keras.callbacks import Callback

# %% custom callbacks
    
class WeightsBiasesHistory(Callback):
    def __init__(self):
        super(WeightsBiasesHistory, self).__init__()
        self.wab_history = []
        self.weights_history = []
        self.biases_history = []

    def on_epoch_end(self, epoch, logs=None):
        # I could also save at the end of every batch instead!
        model_weights = self.model.get_weights()
        wab = [np.array(w) for w in model_weights]
        self.wab_history.append(wab)
        self.weights_history.append(wab[0::2])
        self.biases_history.append(wab[1::2])
# %%
