"""The benchmark scenarios of the ACRI 2024 paper and the format of their data.

Tab. 1 of the paper (thesis Tab. 7.2) varies one parameter at a time: the
number of rules N_R, of time steps T, of cells N and of samples S. The
timings are stored per scenario in one ``.npy`` file holding four arrays saved
in sequence: the x-values, then the CellPyLib, locally connected CNN and
densely connected CNN times, each of shape (repeats, number of x-values).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

METHODS = ("cpl", "lc", "dense")


@dataclass(frozen=True)
class Scenario:
    stem: str        # file name without extension, as in 2024
    fixed: dict      # the parameters kept fixed (also used in the plot title)
    varied: str      # the parameter that varies: "Nrules", "T", "N" or "S"
    values: tuple    # its values


SCENARIOS = {
    "Nrules": Scenario("nuca-comparison-Nrules-N256_T32_S32_avg-from-10",
                       dict(N=256, T=32, S=32), "Nrules", tuple(2**k for k in range(9))),
    "T": Scenario("nuca-comparison-T-N64_Nrules4_S32_avg-from-10",
                  dict(N=64, Nrules=4, S=32), "T", tuple(range(10, 101, 10))),
    "N": Scenario("nuca-comparison-N-T32_Nrules4_S32_avg-from-10",
                  dict(T=32, Nrules=4, S=32), "N", tuple(range(32, 257, 32))),
    "S": Scenario("nuca-comparison-S-N32_T32_Nrules4_avg-from-10",
                  dict(N=32, T=32, Nrules=4), "S", tuple(2**k for k in range(11))),
}


def load_timings(path) -> tuple[np.ndarray, list[np.ndarray]]:
    """The x-values and the three (repeats, n) timing arrays of one scenario."""
    with open(Path(path), "rb") as f:
        return np.load(f), [np.load(f) for _ in METHODS]


def save_timings(path, x, times) -> None:
    """Write x-values and three timing arrays (CellPyLib, LC, dense) in sequence."""
    with open(Path(path), "wb") as f:
        for arr in (np.asarray(x), *times):
            np.save(f, np.asarray(arr))
