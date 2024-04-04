# %% import packages

import tensorflow as tf
from tensorflow.keras.constraints import Constraint

# %% custom constraints. May be handy in training phase (but may also be too restrictive and/or obstructive to back-propagation)

class FixedSumConstraint(Constraint):
    # it feels like this is going to cause issues
    def __init__(self, target_sum, axis=0):
        self.target_sum = target_sum
        self.axis = axis

    def __call__(self, w):
        return w / tf.reduce_sum(w, axis=self.axis, keepdims=True) * self.target_sum

    def get_config(self):
        return {'target_sum': self.target_sum, 'axis': self.axis}
    
class WeakSigmoid(Constraint):
    # I can also just leave this out, and let the system figure things out by itself
    def __init__(self, flattening=1):
        self.flattening = flattening
    # increase flattening to avoid vanishing gradients
    # initialise weights at 0 (outputs 1/2)
    def __call__(self, w):
        return tf.nn.sigmoid(w/self.flattening)