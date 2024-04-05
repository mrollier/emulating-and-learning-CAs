# %% import packages

# existing
import tensorflow as tf

from tensorflow.keras.layers import Conv1D, Activation
from tensorflow.keras import Input

# custom
from src.custom_tf_classes.layers import PeriodicConv1D
from src.custom_tf_classes.initializers import WeightsTripletFinder, BiasesTripletFinder, WeightsLocalUpdate, WeightsHalfway

# %% ECA emulator class

class EcaEmulator:
    """
    TODO: change the class description to the conventional form (below is the description copied from the original function)
    This model first finds the triplets, then performs a 1x1 convolution that is rule-specific.
    
    Parameters
    ----------
    N : int
        Width of the ECA configuration. The input must be of shape (N_samples, N, 1).
    rule : int
        Rule that determines the ECA evolution. Default is None.
    timesteps : int
        Number of timesteps after which the configuration is calculated. Default is 1.
    output_hidden : bool
        Choose to output the values of the hidden layers as well. This is useful when generating entire spacetime diagrams. Default is True.
    train_triplet_id : bool
        If True, the network is also capable of changing the weights and biases that lead to the triplet identification. Enabling this option gives more freedom to the network in the training phase. Disabling this generally makes more sense, as it is always the required first step in a 'human' approach to finding the next configuration. Default is False.
    activation : str or None or tf.keras.layers.Activation
        The activation function for the final layer (useful for training). Default is None.

    Returns
    -------
    model : keras.src.engine.functional.Functional
        Keras model object that can be used for inference, training ... The output of this model is either the content of all intermediate CA configurations (if output_hidden is True), or only the final configuration (if output_hidden is False).
    """
    def __init__(self, N:int, rule=None, timesteps=1, output_hidden=False, train_triplet_id=True, activation=None, kernel_initializer='he_normal'):
        self.N = N
        self.rule = rule
        self.timesteps = timesteps
        self.output_hidden = output_hidden
        self.train_triplet_id = train_triplet_id
        self.activation = activation
        self.kernel_initializer = kernel_initializer

    def model(self):
        # model input
        inputs = Input((self.N,1), dtype=tf.float32)

        # 1x3 convolution to identify each of the eight triplets
        if not self.train_triplet_id:
            kernel_initializer = WeightsTripletFinder()
            bias_initializer = BiasesTripletFinder()
        else:
            kernel_initializer = 'he_normal'
            bias_initializer = 'zeros'
        triplet_id = PeriodicConv1D(8,3, activation='relu',
                            kernel_initializer=kernel_initializer,
                            bias_initializer=bias_initializer,
                            trainable=self.train_triplet_id)
        
        # 1x1 convolution, summing according to each of the rules
        if self.rule is not None: # all rules provided
            kernel_initializer = WeightsLocalUpdate([self.rule])
            use_bias=False
            train_local_update=False
        else:
            if self.kernel_initializer=='halfway':
                self.kernel_initializer = WeightsHalfway([self.rule])
            kernel_initializer = self.kernel_initializer
            use_bias=True
            train_local_update=True
        global_update = Conv1D(1, 1, activation='relu',
                                kernel_initializer=kernel_initializer,
                                use_bias=use_bias,
                                trainable=train_local_update)

        # rinse and repeat over several timesteps
        x = inputs
        if self.output_hidden:
            all_configs = tf.identity(inputs)
            for _ in range(self.timesteps-1):
                x = triplet_id(x)
                x = global_update(x)
                all_configs = tf.concat([all_configs, x], axis=2)
            x = triplet_id(x)
            x = global_update(x)
            outputs = Activation(self.activation)(x)
            all_configs = tf.concat([all_configs, outputs], axis=2)
        else:
            for _ in range(self.timesteps):
                x = triplet_id(x)
                x = global_update(x)
            outputs = Activation(self.activation)(x)

        # sequence and return model
        if self.output_hidden:
            model = tf.keras.Model(inputs=inputs, outputs=[all_configs,outputs])
        else:
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model
