"""Shared fixtures for the verification suite.

The claims C1-C8 checked here are listed in docs/provenance.md. Tests marked
``slow`` are skipped by ``reproduce.py quick``; run them with
``pytest -m slow`` (or everything with plain ``pytest``).
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ca_emulators import EcaEmulator  # noqa: E402
from ca_emulators.simulate import compiled_step, to_states  # noqa: E402
from ca_emulators.weights import rule_table_kernel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@pytest.fixture(scope="session")
def golden() -> dict[str, np.ndarray]:
    """Outputs of the 2024 code (scripts/capture_golden_2024.py)."""
    with np.load(DATA / "golden" / "golden_2024.npz") as f:
        return {k: f[k] for k in f.files}


@pytest.fixture(scope="session")
def published() -> dict[str, dict[str, np.ndarray]]:
    """Inputs decoded from the published figures (scripts/extract_published_inputs.py)."""
    out = {}
    for path in sorted((DATA / "published_inputs").glob("*.npz")):
        with np.load(path) as f:
            out[path.stem] = {k: f[k] for k in f.files}
    return out


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260929)


class EcaRunner:
    """One exact ECA model per N, with the rule table swapped in place.

    Building a Keras model per rule would make the 256-rule checks slow; the
    rule-table weights are frozen variables, so swapping them keeps the traced
    step function valid.
    """

    def __init__(self):
        self._cache = {}

    def __call__(self, n_cells: int, rule: int, x0, n_updates: int = 1) -> np.ndarray:
        if n_cells not in self._cache:
            model = EcaEmulator(n_cells, 0).model()
            self._cache[n_cells] = (model, compiled_step(model))
        model, step = self._cache[n_cells]
        model.get_layer("rule_tables").set_weights([rule_table_kernel([rule])])
        x = np.asarray(x0, dtype=np.float32)[:, :, np.newaxis]
        frames = [to_states(x)]
        for _ in range(n_updates):
            x = step(x).numpy()
            frames.append(to_states(x))
        return np.stack(frames, axis=1)  # (S, n_updates + 1, N)


@pytest.fixture(scope="session")
def eca_runner() -> EcaRunner:
    return EcaRunner()
