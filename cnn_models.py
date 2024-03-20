# %% import packages

# helper libraries
import numpy as np

# import existing packages
import tensorflow as tf

from tensorflow.keras.layers import Conv1D, LocallyConnected1D, Activation
from tensorflow.keras import Input

# custom packages and classes
from custom_tf_classes import PeriodicConv1D
from custom_tf_classes import WeightsTripletFinder, BiasesTripletFinder, WeightsLocalUpdate, WeightsRuleAllocation

# %% Model for ECA

def eca_emulator(N:int, rule=None, timesteps=1, output_hidden=True, train_triplet_id=False, activation=None):
    """
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

    Returns
    -------
    model : keras.src.engine.functional.Functional
        Keras model object that can be used for inference, training ... The output of this model is either the content of all intermediate CA configurations (if output_hidden is True), or only the final configuration (if output_hidden is False).
    """
    # model input
    inputs = Input((N,1), dtype=tf.float32)

    # 1x3 convolution to identify each of the eight triplets
    if not train_triplet_id:
        kernel_initializer = WeightsTripletFinder()
        bias_initializer = BiasesTripletFinder()
    else:
        kernel_initializer = 'he_normal'
        bias_initializer = 'zeros'
    triplet_id = PeriodicConv1D(8,3, activation='relu',
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        trainable=train_triplet_id)
    
    # 1x1 convolution, summing according to each of the rules
    if rule is not None: # all rules provided
        kernel_initializer = WeightsLocalUpdate([rule])
        use_bias=False
    else:
        kernel_initializer = 'he_normal'
        use_bias=True
    global_update = Conv1D(1, 1, activation='relu',
                            kernel_initializer=kernel_initializer,
                            use_bias=use_bias,
                            trainable=train_triplet_id)

    # rinse and repeat over several timesteps
    x = inputs
    if output_hidden:
        all_configs = tf.identity(inputs)
        for _ in range(timesteps-1):
            x = triplet_id(x)
            x = global_update(x)
            all_configs = tf.concat([all_configs, x], axis=2)
        x = triplet_id(x)
        x = global_update(x)
        outputs = Activation(activation)(x)
        all_configs = tf.concat([all_configs, outputs], axis=2)
    else:
        for _ in range(timesteps):
            x = triplet_id(x)
            x = global_update(x)
        outputs = Activation(activation)(x)

    # sequence and return model
    if output_hidden:
        model = tf.keras.Model(inputs=inputs, outputs=[all_configs,outputs])
    else:
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# %% Model for nuCA

def nuca_emulator_1D(N:int, rules, timesteps=1, train_triplet_id=False, rule_alloc=None,):
    """
    This model first finds the triplets, then evolves them according to a number of convolutions (one set of weights per rule). Then a locally connected layer picks out cell-by-cell which of the rules should apply.

    TODO: check whether biases are required. Perhaps limiting the degrees of freedom is beneficial for training.
    TODO: allow rule_alloc to change over subsequent time steps (no priority)
    TODO: find a better initial weight distribution (cf. Leonardo Scabini)
    TODO: find a nice way to export values of hidden layers, probably by means of a custom Model class.
    
    Parameters
    ----------
    N : int
        Width of the 1D nuCA configuration. The input must be of shape (N_samples, N, 1)
    rules : list of ints
        Rules that can determine the nuCA evolution. May be None (if rule is unknown)
    timesteps : int
        Number of timesteps after which the configuration is calculated. Default is 1.
    train_triplet_id : bool
        If True, the network is also capable of changing the weights and biases that lead to the triplet identification. Enabling this option gives more freedom to the network in the training phase. Disabling this generally makes more sense, as it is always the required first step in a 'human' approach to finding the next configuration. Default is False.
    rule_alloc : (nested) list of ints or None
        List of integers corresponding to the index of the rule in param rules. If None, no perfect weights are initialised. Default is None.

    Returns
    -------
    model : keras.src.engine.functional.Functional
        Keras model object that can be used for inference, training ...

    """
    # exceptions and preprocessing
    if rule_alloc is not None:
        if len(rule_alloc) != N:
            raise Exception("The parameter rule_dist should have size N.")
    rules = np.atleast_1d(rules)
    Nrules = len(rules)

    # model input
    inputs = Input((N,1), dtype=tf.float32)

    # 1x3 convolution to identify each of the eight triplets
    if not train_triplet_id:
        kernel_initializer = WeightsTripletFinder()
        bias_initializer = BiasesTripletFinder()
    else:
        kernel_initializer = 'he_normal'
        bias_initializer = 'zeros'
    triplet_id = PeriodicConv1D(8,3, activation='relu',
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        trainable=train_triplet_id)

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
                            trainable=train_triplet_id)

    # locally connected network to select the desired cell state
    if rule_alloc is not None:
        kernel_initializer = WeightsRuleAllocation(Nrules, rule_alloc)
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
    for _ in range(timesteps):
        x = triplet_id(x)
        x = global_updates(x)
        x = cell_selector(x)
    outputs = x

    # sequence and return model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# %% other models, e.g. for larger neighbourhoods and for multiple states (which is equivalent)