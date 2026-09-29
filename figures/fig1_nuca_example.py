"""Fig. 1 (thesis Fig. 7.1): a nuCA with rules 30 and 90, uniform in time.

The published initial configuration and allocation (decoded from the paper's
PDF) are evolved by the nuCA emulator; the script checks that the result is
the published spacetime diagram before drawing it.
"""
import numpy as np

import _style
from ca_emulators import NucaEmulator
from ca_emulators.plotting import plot_nuca_example
from ca_emulators.reference import evolve_cellpylib

STEM = "spacetime_diagram-nuca-30_90-vertical"


def main(argv=None):
    args = _style.parser(__doc__.splitlines()[0]).parse_args(argv)
    _style.apply(args)
    fig1 = _style.load_npz(_style.DATA / "published_inputs" / "fig1_nuca_rules30_90.npz")
    rules, alloc, published = fig1["rules"], fig1["alloc"], fig1["diagram"]
    n_rows, n_cells = published.shape

    diagram = NucaEmulator(n_cells, rules, rule_alloc=alloc).simulate(published[0], n_rows - 1)
    assert np.array_equal(diagram, published), "the emulator does not reproduce Fig. 1"
    assert np.array_equal(evolve_cellpylib(published[0], rules, n_rows - 1, alloc), published)

    fig, _ = plot_nuca_example(np.tile(alloc, (n_rows, 1)), diagram, rules)
    _style.save(fig, STEM, args)


if __name__ == "__main__":
    main()
