"""C7: the emulator architecture can be trained from random weights (Fig. 3).

The seeded 2024 recipe (ca_emulators.training) is run for rule 54, as in
figures/fig3_training.py. Starting from random weights, the network ends up
exact after thresholding at 0.5, with a small mean squared error. The recipe
relies on its pretraining loop (re-initialising until the one-epoch loss
drops below 0.1), which is why experiments/training studies better recipes.
"""
import numpy as np
import pytest

from ca_emulators.reference import eca_step
from ca_emulators.rules import de_bruijn_configuration
from ca_emulators.training import train_2024_recipe


@pytest.mark.slow
def test_2024_recipe_learns_rule_54(published):
    fig3 = published["fig3_training_example_rule54"]
    run = train_2024_recipe(54, N=32, seed=2024, example=fig3["x"])
    assert len(run.weights_history) == 40
    assert min(run.pretrain_losses) < 0.1
    losses = run.history.history["val_loss"]
    assert losses[-1] < 1e-3 and losses[-1] < losses[0]
    x = de_bruijn_configuration(32)
    y = run.model.predict(x[None, :, None].astype(np.float32), verbose=0)[0, :, 0]
    assert np.array_equal(y > 0.5, eca_step(x, 54) == 1)
