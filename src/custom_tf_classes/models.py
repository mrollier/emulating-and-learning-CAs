# %% import packages
import tensorflow as tf

# %% custom Models

class CustomModel(tf.keras.Model):
    def __init__(self, base_model):
        super(CustomModel, self).__init__()
        self.base_model = base_model

    def call(self, inputs):
        return self.base_model(inputs)

    def predict_with_hidden(self, x):
        outputs = [x]  # Add input as the first output
        for layer in self.base_model.layers:
            x = layer(x)
            outputs.append(x)
        return outputs