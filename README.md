# 1D Learning Automata

## Project description
This project contains classes of simple neural networks that are capable of perfectly emulating a number of 1D cellular automata (CAs), including elementary CAs, non-uniform CAs, and network automata.

## Overview

**tests/**: scripts used for testing the available classes. This is a good start for figuring out how this package works.
- `test_eca.py`: scripts for testing the ECA emulator
- `test_nuca.py`: scripts for testing the nuCA emulator
- `test_na.py`: scripts for testing the NA emulator (under development)

 **scripts/**: all scripts used for actual research (under development).

**src/**: all source files.
- **custom_tf_classes/**: Custom TensorFlow classed
 - `callbacks.py`: custom callbacks (e.g. for saving weights and biases during training)
 - `constraints.py`: custom constraints imposed on the parameters during training (currently not used)
 - `initializers.py`: custom initializers, used to establish the perfect weights and biases for CA emulation
 - `layers.py`: custom layers, including a Conv layer with periodic padding
 - `models.py`: custom models (currently not used)
- **nn/**: neural networks (NNs) that emulate the CAs
 - `eca.py`: NN that emulates an elementary CA
 - `nuca.py`: NN that emulates a non-uniform CA
 - `na.py`: NN that emulates a network automaton (under development)
- **train/**: classes used for training the NN
 - `train.py`: contains wrapper for training procedure in TensorFlow
- **visual/**: classes used for visualising the training procedure and outcome
 - `histories.py`: visualise history of configurations and of loss values (under development)
- **utils/**: generic classes and definitions that are used throughout
 - `ruletable.py`: definitions related to the CA ruletable, such as `base_repr`

 **old/**: scripts that will be removed, but contain valuable information for now.

 ## Important results

 (under development)