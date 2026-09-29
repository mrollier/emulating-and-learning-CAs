"""Analytic weights of the exact emulators (thesis Tab. 7.1, Eqs. 7.1 and 7.2).

Plain numpy functions; the emulators copy them into their layers with
``set_weights`` after building the model.

Layer 1, the eight neighbourhood detectors: channel i responds to the
neighbourhood (s_-, s_o, s_+) whose binary representation is i. Its three
weights are +1 where that neighbourhood holds a 1 and -omega where it holds a
0, and its bias is 1 - h, with h the number of ones. The weighted sum is 1
for the matching neighbourhood and at most 0 for every other one as soon as
omega >= 1 (for omega = 1 it equals 1 minus the Hamming distance), so after
the ReLU the eight channels form an exact one-hot encoding of the
neighbourhood. The index of the active channel is the integer
4 s_- + 2 s_o + s_+ that the (4, 2, 1) description of the ACRI 2024 paper
refers to.

Layer 2, the rule tables: a 1x1 convolution whose weight i is the state the
rule assigns to neighbourhood i; with N_R rules it has N_R output channels.

Layer 3 (nuCAs only), the selector: per cell, the one-hot vector of its
allocated rule, either as a locally connected layer or as a dense layer.
"""
from __future__ import annotations

import numpy as np

from .rules import check_alloc, check_rules, neighbourhood_patterns, rule_table

DEFAULT_OMEGA = 5.0
ENCODINGS = ("01", "pm1")


def detector_kernel(omega: float = DEFAULT_OMEGA, encoding: str = "01") -> np.ndarray:
    """Kernel of layer 1, shape (3, 1, 8): kernel[k, 0, i] weights position k for channel i.

    With states fed as 0/1 (``encoding="01"``, the paper's network) the
    weights are +1 and -omega. With states fed as -1/+1 (``"pm1"``) they are
    the signs of the pattern, +1 and -1, and omega plays no role.
    """
    patterns = neighbourhood_patterns()
    if encoding == "01":
        weights = np.where(patterns == 1, 1.0, -float(omega))  # (8, 3)
    elif encoding == "pm1":
        weights = 2.0 * patterns - 1.0
    else:
        raise ValueError(f"encoding must be one of {ENCODINGS}")
    return weights.T[:, np.newaxis, :].astype(np.float32)


def detector_bias(encoding: str = "01") -> np.ndarray:
    """Bias of layer 1, shape (8,).

    ``"01"``: 1 - h for the neighbourhood of each channel (h its number of
    ones). ``"pm1"``: -2, since the weighted sum of a +-1 neighbourhood with
    the sign pattern is 3 minus twice the Hamming distance.
    """
    if encoding == "pm1":
        return np.full(8, -2.0, dtype=np.float32)
    if encoding != "01":
        raise ValueError(f"encoding must be one of {ENCODINGS}")
    n_ones = neighbourhood_patterns().astype(np.int64).sum(axis=1)  # not uint8: 1 - 2 must be -1
    return (1 - n_ones).astype(np.float32)


def rule_table_kernel(rules) -> np.ndarray:
    """Kernel of layer 2, shape (1, 8, N_R): column j is the rule table of rules[j]."""
    tables = np.stack([rule_table(r) for r in check_rules(rules)], axis=1)
    return tables[np.newaxis].astype(np.float32)


def allocation_one_hot(alloc, n_rules: int) -> np.ndarray:
    """(N, N_R) float32 matrix with a single 1 per row at the allocated rule."""
    alloc = check_alloc(alloc, np.shape(alloc)[-1] if np.ndim(alloc) else 1, n_rules)
    return np.eye(n_rules, dtype=np.float32)[alloc]


def selector_kernel_lc(alloc, n_rules: int, implementation: int = 1) -> np.ndarray:
    """Kernel of the ``LocallyConnected1D`` selector (kernel size 1, one filter).

    Keras stores this kernel differently in its three implementation modes:

    1. shape (N, N_R, 1): one weight vector per cell (a loop over cells);
    2. shape (N, N_R, N, 1): a masked dense kernel (one matrix product);
    3. shape (N * N_R,): the non-zero entries of a sparse kernel, ordered by
       (output cell, input index), i.e. the flattened one-hot allocation.
    """
    one_hot = allocation_one_hot(alloc, n_rules)
    n_cells = len(one_hot)
    if implementation == 1:
        return one_hot[:, :, np.newaxis]
    if implementation == 2:
        kernel = np.zeros((n_cells, n_rules, n_cells, 1), dtype=np.float32)
        kernel[np.arange(n_cells), :, np.arange(n_cells), 0] = one_hot
        return kernel
    if implementation == 3:
        return one_hot.reshape(-1)
    raise ValueError("implementation must be 1, 2 or 3")


def selector_kernel_dense(alloc, n_rules: int) -> np.ndarray:
    """Kernel of the dense selector, shape (N_R * N, N).

    The N_R candidate configurations are flattened rule by rule (input index
    r * N + i), and entry (r * N + i, j) is 1 exactly when i = j and cell j is
    allocated rule r: a stack of N_R diagonal matrices.
    """
    one_hot = allocation_one_hot(alloc, n_rules)  # (N, N_R)
    n_cells = len(one_hot)
    kernel = np.zeros((n_rules, n_cells, n_cells), dtype=np.float32)
    kernel[:, np.arange(n_cells), np.arange(n_cells)] = one_hot.T
    return kernel.reshape(n_rules * n_cells, n_cells)


def parameter_count(n_cells: int, n_rules: int = 1, variant: str = "eca") -> int:
    """Number of parameters of the exact emulators (zero entries included).

    ``"eca"``: (3 + 1) * 8 + 8 = 40. ``"lc"``: Eq. (7.1),
    (3 + 1) * 8 + 8 N_R + N_R N. ``"dense"``: Eq. (7.2),
    (3 + 1) * 8 + 8 N_R + N_R N^2. The count for ``"lc"`` holds for Keras's
    implementation modes 1 and 3; mode 2 stores N_R N^2 selector weights, like
    the dense variant.
    """
    base = (3 + 1) * 8 + 8 * n_rules
    if variant == "eca":
        if n_rules != 1:
            raise ValueError("an ECA has one rule")
        return base
    if variant == "lc":
        return base + n_rules * n_cells
    if variant == "dense":
        return base + n_rules * n_cells**2
    raise ValueError("variant must be 'eca', 'lc' or 'dense'")
