"""Fig. 4 (thesis Fig. 7.4): an 8-rule nuCA simulated with CellPyLib.

As published, the allocation shifts by one cell per time step (so this nuCA
is non-uniform in time as well as in space; the text of the paper otherwise
considers spatial non-uniformity only). The published rules, allocation and
initial configuration are evolved with CellPyLib and checked against the
published diagram and against the numpy reference.
"""
import numpy as np

import _style
from ca_emulators.plotting import plot_nuca_cellpylib
from ca_emulators.reference import evolve, evolve_cellpylib

STEM = "cellpylib-spacetime_diagram-Nrules8"


def main(argv=None):
    args = _style.parser(__doc__.splitlines()[0]).parse_args(argv)
    _style.apply(args)
    fig4 = _style.load_npz(_style.DATA / "published_inputs" / "fig4_nuca_cellpylib_8rules.npz")
    rules, alloc, published = fig4["rules"], fig4["alloc"], fig4["diagram"]
    n_updates = len(published) - 1

    diagram = evolve_cellpylib(published[0], rules, n_updates, alloc[:n_updates])
    assert np.array_equal(diagram, published), "CellPyLib does not reproduce Fig. 4"
    assert np.array_equal(evolve(published[0], rules, n_updates, alloc[:n_updates]), published)

    fig, _ = plot_nuca_cellpylib(alloc, diagram, rules)
    _style.save(fig, STEM, args)


if __name__ == "__main__":
    main()
