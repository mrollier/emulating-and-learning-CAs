# %% import packages

# classic
import numpy as np
import matplotlib.pyplot as plt

# custom
import sys
sys.path.insert(0, '..') # TODO: this is probably not the right way to do this
from src.nn.eca import EcaEmulator
from src.train.train import Train1D
from src.visual.histories import History1D

%load_ext autoreload
%autoreload 2

# choose appropriate directory
dir_figs = '../figures/eca/'
dir_files = '../files/models/eca/'

# %% set params

N = 32
N_train = 2**12
N_val = 2**12
timesteps = 1
# Note: persistent problems with i.a. rule 1
rules = [1] # list(range(256))

# BEWARE: this is an intensive computation
PRETRAIN = True
TRAIN = True

SAVE_FIG = True

# pretraining params
N_pt = 50
batch_size_pt = 128
min_pretrain_loss = 0.1

# training params
batch_size = 64
epochs = 40
learning_rate = 0.005
loss = 'mse'
stopping_patience = None # 20
stopping_delta = None # 0.0001
wab_callback = True
train_verbose=False

# bookkeeping
lr_str = str(learning_rate).replace('.','p')

# %% optimise eca emulator for all 256 rules

for rule in rules:
    print("")
    print("="*32)
    print(f"Working on rule {rule}.  ")

    activation = None
    # model with perfect weights and biases
    train_triplet_id = False
    ECA = EcaEmulator(N, rule=rule, timesteps=timesteps,
                    activation=activation, train_triplet_id=train_triplet_id)
    model_perfect = ECA.model()

    # model with random weights and biases, and activation function (for training)
    ECA.rule = None
    ECA.activation = 'tanh' # a modified sigmoid may be better
    ECA.train_triplet_id = True
    model = ECA.model()

    ## define inputs and outputs

    # inputs
    x_train = np.random.randint(2,size=(N_train, N, 1),
                                dtype=np.int8)
    x_val = np.random.randint(2, size=(N_val, N, 1),
                            dtype=np.int8)

    # desired outputs
    verbose = False
    r_train = model_perfect.predict(x_train, batch_size=len(x_train), verbose=verbose)
    r_val = model_perfect.predict(x_val, batch_size=len(x_val), verbose=verbose)

    ## pretraining and training

    if PRETRAIN:
        losses = []
        best_loss = np.infty
        # models = [ECA.model() for _ in range(N_pt)]
        # for i in range(N_pt):
        i = 0
        while best_loss > min_pretrain_loss:
            # print(f'Working on pretraining {i+1}/{N_pt}. Best loss: {best_loss}.         ', end='\r')
            print(f'Working on pretraining {i+1}. Best loss: {best_loss}.         ', end='\r')
            # current_model = models[i]
            current_model = ECA.model()
            tr = Train1D(current_model, x_train, r_train, x_val, r_val,
                        batch_size=batch_size_pt, epochs=1,
                        learning_rate=learning_rate, loss=loss)
            history = tr.train(verbose=False)
            current_loss = history.history['loss'][0]
            losses += [current_loss]
            if current_loss < best_loss:
                best_loss = current_loss
                best_model = current_model
            i += 1
        print('')
        model = best_model

    # save pretraining loss histogram
    if SAVE_FIG:
        plt.hist(losses, bins=np.arange(0,1.1,0.1))
        hist_title = f'Pretraining losses rule {rule}, {N_pt} weight initialisations'
        hist_savename = f'pretraining_loss_hist_ECA_{N}cells_rule{rule}_single-epochs_bs{batch_size_pt}_lr{lr_str}.pdf'
        plt.title(hist_title)
        plt.savefig(dir_figs+hist_savename, bbox_inches='tight')
        print(f"SAVED {hist_savename}.")
        plt.close()

    if TRAIN:
        tr = Train1D(model, x_train, r_train, x_val, r_val,
                    batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, loss=loss,stopping_patience=stopping_patience, stopping_delta=stopping_delta, wab_callback=wab_callback)
        if wab_callback:
            history, wab_history = tr.train(verbose=train_verbose)
        else:
            history = tr.train(verbose=train_verbose)

    # save model
    modelname = f"ECA_{N}cells_rule{rule}_{epochs}epochs_bs{batch_size}_lr{lr_str}.h5"
    model.save(dir_files+modelname)

    ## visualisation

    idx_ex = np.random.randint(N_train)
    input_example = x_train[idx_ex]
    output_example = r_train[idx_ex]

    hist = History1D(history, model=model, wab_history=wab_history)
    fig, ax = hist.plot_configs(input_example, output_example, rule, timesteps)

    if SAVE_FIG:
        savename = f"plot_configs_ECA_{N}cells_rule{rule}_{epochs}epochs_bs{batch_size}_lr{lr_str}.pdf"
        plt.savefig(dir_figs+savename, bbox_inches='tight')
        print(f"SAVED {savename}.")
        plt.close()

    final_loss = history.history['val_loss'][0]
    print(f"Finalised rule {rule} with val_loss {final_loss}.")
# %%
