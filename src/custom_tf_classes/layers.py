# %% import packages
import tensorflow as tf
from tensorflow.keras.layers import Conv1D

# %% custom Layers

class PeriodicConv1D(Conv1D):
    # WARNING: copied from ChatGPT (contains bugs)
    def __init__(self, filters, kernel_size, **kwargs):
        super(PeriodicConv1D, self).__init__(filters, kernel_size, **kwargs)

    def call(self, inputs):
        # Compute the padding required for periodic padding
        pad_size = self.kernel_size[0] // 2

        # Apply periodic padding
        inputs = self.periodic_padding(inputs, pad_size)

        # Perform convolution operation
        return super(PeriodicConv1D, self).call(inputs)

    def periodic_padding(self, input_tensor, pad_size):
        # Pad the tensor with zeros
        # padded_tensor = tf.pad(input_tensor, [[0, 0], [pad_size, pad_size], [0, 0]], mode='constant')

        # Perform periodic padding
        # padded_tensor = tf.roll(padded_tensor, shift=pad_size, axis=1)
        left_values = input_tensor[:,0:pad_size,:]
        right_values = input_tensor[:,-pad_size:,:]
        padded_tensor = tf.concat([right_values, input_tensor, left_values], axis=1)

        return padded_tensor