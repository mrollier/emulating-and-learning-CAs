# %% import packages

# existing
import numpy as np
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# custom
from src.custom_tf_classes.callbacks import WeightsBiasesHistory

# %% train class

class Train1D:
    def __init__(self, model, input, output, input_val, output_val,
                 batch_size=16, epochs=1, learning_rate= 0.001, loss='mse',stopping_patience=None, stopping_delta=None, wab_callback=False):
        self.model = model
        self.input = input
        self.output = output
        self.input_val = input_val
        self.output_val = output_val
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss = loss
        self.stopping_patience = stopping_patience
        self.stopping_delta = stopping_delta
        self.wab_callback = wab_callback

        # exceptions
        if (stopping_patience is not None and stopping_delta is None) or (stopping_patience is None and stopping_delta is not None):
            raise Exception("When using EarlyStopping, specify both stopping_patience and stopping_delta. When neither is selected, EarlyStopping is not activated.")

    def train(self, verbose=True):
        self.model.compile(loss=self.loss,
                           optimizer=Adam(learning_rate=self.learning_rate),
                           metrics=['mse'])
    
        callbacks=[]
        if self.stopping_delta:
            stopping_callback = EarlyStopping(monitor='val_mse',
                                            patience=self.stopping_patience,
                                            min_delta=self.stopping_delta,
                                            verbose = verbose,
                                            restore_best_weights=True)
            callbacks.append(stopping_callback)
        if self.wab_callback:
            weights_biases_callback = WeightsBiasesHistory()
            callbacks.append(weights_biases_callback)

        history = self.model.fit(self.input,
                        self.output.astype(np.float32), # TODO: probably better as tf.float32?
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        verbose=verbose,
                        validation_data=(self.input_val, self.output_val.astype(np.float32)),
                        shuffle=True,
                        callbacks=callbacks)
        
        if self.wab_callback:
            return history, weights_biases_callback.wab_history
        return history