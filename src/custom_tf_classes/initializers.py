# %% import packages

import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import Initializer

from src.utils.ruletable import base_repr

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
            rule_array = list(base_repr(rule)[::-1])
            rule_array_tmp = [eval(i) for i in rule_array]
            rule_array = np.array(rule_array_tmp, dtype=np.float32)
            local_update[:,i] = rule_array
        local_update = tf.constant(local_update, dtype=tf.float32)
        local_update = tf.expand_dims(local_update,axis=0)
        self.local_update = local_update

    # TODO: why are shape and dtype required?
    def __call__(self, shape, dtype=None):
        return self.local_update
    
class WeightsHalfway(Initializer):
    def __init__(self, rules):
        local_update = .5*np.ones((8, len(rules)))
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
    def __init__(self, Nrules, rule_alloc, dense=False):
        N = len(rule_alloc)
        alloc_one_hot = tf.one_hot(rule_alloc, Nrules)
        if not dense:
            alloc_weights = alloc_one_hot[:,:,tf.newaxis]
            self.alloc_weights = alloc_weights
        else:
            alloc_weights = tf.zeros([0,N])
            for column in tf.transpose(alloc_one_hot):
                column_diag = tf.linalg.diag(column)
                alloc_weights = tf.concat([alloc_weights, column_diag], axis=0)
            self.alloc_weights = alloc_weights

    def __call__(self, shape, dtype=None):
        return self.alloc_weights