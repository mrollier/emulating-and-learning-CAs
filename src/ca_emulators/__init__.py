"""Exact emulation of elementary and non-uniform cellular automata by CNNs.

Companion package of M. Rollier, A. J. Daly, O. M. Bruno and J. M. Baetens,
"Efficient Simulation of Non-uniform Cellular Automata with a Convolutional
Neural Network", ACRI 2024, LNCS 14978, pp. 121-131 (arXiv:2409.02722), and of
Ch. 7 of M. Rollier's PhD thesis (Ghent University, 2026).

>>> from ca_emulators import EcaEmulator
>>> model = EcaEmulator(32, rule=54).model()   # 40 fixed weights, exact
"""
from __future__ import annotations

import tensorflow as tf

__version__ = "1.0.0.dev0"

_keras_version = getattr(tf.keras, "__version__", "2")
if int(_keras_version.split(".")[0]) >= 3:
    raise ImportError(
        f"ca_emulators needs Keras 2 (found Keras {_keras_version}): Keras 3 no longer "
        "provides LocallyConnected1D. Install the pinned versions, e.g. "
        "`pip install -r requirements.txt` (tensorflow==2.14.0).")

from . import reference, rules, simulate, verify, weights  # noqa: E402
from .eca import EcaEmulator  # noqa: E402
from .layers import PeriodicPadding1D  # noqa: E402
from .nuca import NucaEmulator  # noqa: E402

__all__ = ["EcaEmulator", "NucaEmulator", "PeriodicPadding1D", "reference", "rules",
           "simulate", "verify", "weights", "__version__"]
