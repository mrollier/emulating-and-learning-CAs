# %% import packages

# helper libraries
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

# TensorFlow and tf.keras
import tensorflow as tf

print(tf.__version__)

# fix random seed for reproducibility
seed = 2023
np.random.seed(seed)

from cnn_models import eca_emulator
from custom_tf_classes import CustomModel

images_dir = "./figures"


# %% Synthesise initial configurations

N_diagrams = 2**16
N = 64

init_configs = np.random.randint(2,size=(N_diagrams, N, 1),
                            dtype=np.int8)

# %% Synthesise next timestep

ECA_rule = 110
timesteps = 8
model = eca_emulator(N, ECA_rule,timesteps=timesteps)

final_configs = model.predict(init_configs, batch_size=N_diagrams//4)

# %%

custom_model = CustomModel(model)

