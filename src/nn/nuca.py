# %% import packages

# existing packages
import tensorflow as tf
import numpy as np

# note: LocallyConnected1D is removed from newer versions of TensorFlow!
from tensorflow.keras.layers import Conv1D, Activation, LocallyConnected1D
from tensorflow.keras import Input

# custom packages and classes
from src.custom_tf_classes.layers import PeriodicConv1D
from src.custom_tf_classes.initializers import WeightsTripletFinder, BiasesTripletFinder, WeightsLocalUpdate, WeightsRuleAllocation

# %% nuCA emulator class

class NucaEmulator:
    """
    This model first finds the triplets, then evolves them according to a number of convolutions (one set of weights per rule). Then a locally connected layer picks out cell-by-cell which of the rules should apply.

    TODO: change the class description to the conventional form (below is the description copied from the original function)
    This model first finds the triplets, then performs a 1x1 convolution that is rule-specific.
    TODO: check whether biases are required. Perhaps limiting the degrees of freedom is beneficial for training.
    TODO: allow rule_alloc to change over subsequent time steps (no priority)
    TODO: allow a framework where the rules are known, but the rule allocation is not.
    TODO: find a better initial weight distribution (cf. Leonardo Scabini?)
    TODO: find a nice way to export values of hidden layers, probably by means of a custom Model class.
    
    Parameters
    ----------
    N : int
        Width of the 1D nuCA configuration. The input must be of shape (N_samples, N, 1)
    rules : list of ints
        Rules that can determine the nuCA evolution. May be None (if rule is unknown)
    timesteps : int
        Number of timesteps after which the configuration is calculated. Default is 1.
    output_hidden : bool
        Choose to output the values of the hidden layers as well. This is useful when generating entire spacetime diagrams. Default is True.
    train_triplet_id : bool
        If True, the network is also capable of changing the weights and biases that lead to the triplet identification. Enabling this option gives more freedom to the network in the training phase. Disabling this generally makes more sense, as it is always the required first step in a 'human' approach to finding the next configuration. Default is False.
    rule_alloc : (nested) list of ints or None
        List of integers corresponding to the index of the rule in param rules. If None, no perfect weights are initialised. Default is None.
    activation : str or None or tf.keras.layers.Activation
        The activation function for the final layer (useful for training). Default is None.

    Returns
    -------
    model : keras.src.engine.functional.Functional
        Keras model object that can be used for inference, training ... The output of this model is either the content of all intermediate CA configurations (if output_hidden is True), or only the final configuration (if output_hidden is False).
    """
    def __init__(self, N:int, rules=None, timesteps=1, output_hidden=False, train_triplet_id=True, rule_alloc=None, activation=None):
        self.N = N
        self.rules = rules
        self.timesteps = timesteps
        self.output_hidden = output_hidden
        self.train_triplet_id = train_triplet_id
        self.rule_alloc = rule_alloc
        self.activation = activation

    def model(self):
    # TODO: add an option for a more powerful model (e.g. with a fully-connected layer instead of the LocallyConnected1D)
    # exceptions and preprocessing
        rules = np.atleast_1d(self.rules)
        Nrules = len(rules)
        if self.rule_alloc is not None:
            if len(self.rule_alloc) != self.N:
                raise Exception("The parameter rule_dist should have size N.")
            if np.max(self.rule_alloc) > Nrules - 1:
                raise Exception("The rule allocation does not correspond to the number of rules.")

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
        Nrules = len(rules)
        if rules.all() is not None: # all rules provided
            kernel_initializer = WeightsLocalUpdate(rules)
            use_bias=False
        else:
            kernel_initializer = 'he_normal'
            use_bias=True
        global_updates = Conv1D(Nrules, 1, activation='relu',
                                kernel_initializer=kernel_initializer,
                                use_bias=use_bias,
                                trainable=self.train_triplet_id)

        # locally connected network to select the desired cell state
        if self.rule_alloc is not None:
            kernel_initializer = WeightsRuleAllocation(Nrules, self.rule_alloc)
            use_bias=False
            train_rule_alloc=False
        else:
            kernel_initializer = 'he_normal'
            use_bias=True
            train_rule_alloc=True
        cell_selector = LocallyConnected1D(1, 1, activation='relu',
                                        kernel_initializer=kernel_initializer,
                                        use_bias=use_bias,
                                        trainable=train_rule_alloc)

        # rinse and repeat over several timesteps
        x = inputs
        if self.output_hidden:
            all_configs = tf.identity(inputs)
            for _ in range(self.timesteps-1):
                x = triplet_id(x)
                x = global_updates(x)
                x = cell_selector(x)
                all_configs = tf.concat([all_configs, x], axis=2)
            x = triplet_id(x)
            x = global_updates(x)
            x = cell_selector(x)
            outputs = Activation(self.activation)(x)
            all_configs = tf.concat([all_configs, outputs], axis=2)
        else:
            for _ in range(self.timesteps):
                x = triplet_id(x)
                x = global_updates(x)
                x = cell_selector(x)
            outputs = Activation(self.activation)(x)

        # sequence and return model
        if self.output_hidden:
            model = tf.keras.Model(inputs=inputs, outputs=[all_configs,outputs])
        else:
            model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model