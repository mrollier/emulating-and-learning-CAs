"""C7: the emulator architecture can be trained from random weights (Fig. 3).

The seeded 2024 recipe (ca_emulators.training) is run for rule 54 exactly as
figures/fig3_training.py runs it, in a subprocess: TensorFlow's op
determinism is process-global, so it is switched on there and not in the test
process. Starting from random weights, the network ends up exact after
thresholding at 0.5, with a small mean squared error. The recipe relies on
its pretraining loop (re-initialising until the one-epoch loss drops below
0.1), which is why experiments/training studies better recipes.
"""
import json
import os
import subprocess
import sys

import pytest

from conftest import ROOT


@pytest.mark.slow
def test_2024_recipe_learns_rule_54(tmp_path):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "TF_CPP_MIN_LOG_LEVEL": "2"}
    subprocess.run([sys.executable, str(ROOT / "figures" / "fig3_training.py"), "--no-usetex",
                    "--out", str(tmp_path)], check=True, cwd=ROOT, env=env)
    summary = json.loads((tmp_path / "fig3_training.json").read_text())
    assert summary["seed"] == 2024
    assert summary["pretraining_threshold_reached"]
    assert summary["final_val_loss"] < 1e-3
    assert summary["exact_after_thresholding"]
