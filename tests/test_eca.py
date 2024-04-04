# %% import packages

# classic
import numpy as np

# custom
from src.nn.eca import EcaEmulator


%load_ext autoreload
%autoreload 2

# %% test eca emulator

N = 16
rule=110
timesteps=1
activation=None

# model with perfect weights and biases
train_triplet_id = False
ECA = EcaEmulator(N, rule=rule, timesteps=timesteps,
                  activation=activation, train_triplet_id=train_triplet_id)
model_perfect = ECA.model()

# model with random weights and biases, and activation function (for training)
ECA.rule = None
ECA.activation = 'tanh'
ECA.train_triplet_id = True
model = ECA.model()
model.summary()

# %% define inputs and outputs

N_train = 2**12
N_val = 2**12

# inputs
x_train = np.random.randint(2,size=(N_train, N, 1),
                            dtype=np.int8)
x_val = np.random.randint(2, size=(N_val, N, 1),
                          dtype=np.int8)

# desired outputs
verbose=False
r_train = model_perfect.predict(x_train, batch_size=len(x_train), verbose=verbose)
r_val = model_perfect.predict(x_val, batch_size=len(x_val), verbose=verbose)

r_train = np.array(r_train, dtype=np.int8)
r_val = np.array(r_val, dtype=np.int8)

# %% training

