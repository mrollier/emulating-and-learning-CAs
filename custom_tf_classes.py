# %% import packages and classes
import tensorflow as tf
import keras
import numpy as np

from tensorflow.keras.layers import Conv1D
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import Initializer
from keras.callbacks import Callback

from utils import _base_repr


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
    
# %% custom initializers

class WeightsTripletFinder(Initializer):
    def __init__(self, omega=5):
        triplet_finder = tf.constant([[-omega, -omega, -omega],
                                      [-omega, -omega, 1     ],
                                      [-omega, 1     , -omega],
                                      [-omega, 1     , 1     ],
                                      [1     , -omega, -omega],
                                      [1     , -omega, 1     ],
                                      [1     , 1     , -omega],
                                      [1     , 1     , 1     ]], dtype=tf.float32)
        triplet_finder = tf.expand_dims(tf.transpose(triplet_finder),axis=1)
        self.triplet_finder = triplet_finder

    def __call__(self, shape, dtype=None):
        return self.triplet_finder
    
class BiasesTripletFinder(Initializer):
    def __init__(self):
        triplet_finder = tf.constant([1, 0, 0, -1, 0, -1, -1, -2], dtype=tf.float32)
        self.triplet_finder = triplet_finder

    def __call__(self, shape, dtype=None):
        return self.triplet_finder
    
class WeightsLocalUpdate(Initializer):
    """
    TODO: this class uses numpy. It is probably best to keep everything in tf terminology.
    """
    def __init__(self, rules):
        local_update = np.zeros((8, len(rules)))
        for i, rule in enumerate(rules):
            rule_array = list(_base_repr(rule)[::-1])
            rule_array_tmp = [eval(i) for i in rule_array]
            rule_array = np.array(rule_array_tmp, dtype=np.float32)
            local_update[:,i] = rule_array
        local_update = tf.constant(local_update, dtype=tf.float32)
        local_update = tf.expand_dims(local_update,axis=0)
        self.local_update = local_update

    def __call__(self, shape, dtype=None):
        return self.local_update

class WeightsRuleAllocation(Initializer):
    """
    Used to initialise the weights of the LocallyConnected1D layer,
    which picks out the required cells in a non-uniform situation.
    """
    def __init__(self, Nrules, rule_alloc):
        N = len(rule_alloc)
        alloc_weights = tf.one_hot(rule_alloc, Nrules)[:,:,tf.newaxis]
        self.alloc_weights = alloc_weights

    def __call__(self, shape, dtype=None):
        return self.alloc_weights

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
