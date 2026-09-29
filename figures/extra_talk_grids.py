"""Grids of example ECAs and two-rule nuCAs (used in talks, not in the paper).

Each panel is 16 time steps of 32 cells from a random initial configuration,
computed by the exact emulators. Replaces tests/generate-eca.py and
old/generate-nuca.py of 2024; unlike those, the draws are seeded.
"""
import matplotlib.pyplot as plt
import numpy as np

import _style
from ca_emulators import EcaEmulator, NucaEmulator

SEED = 2023  # the seed printed in the 2024 scripts (which did not use it for these draws)


def grid(title_and_diagram, rows: int, cols: int):
    fig, axs = plt.subplots(rows, cols, figsize=(9, 6))
    for ax, (title, diagram) in zip(axs.flat, title_and_diagram):
        ax.imshow(diagram, cmap="Greys")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    return fig


def main(argv=None):
    p = _style.parser(__doc__.splitlines()[0])
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--cols", type=int, default=6)
    args = p.parse_args(argv)
    _style.apply(args)
    rng = np.random.default_rng(SEED)
    n_cells, n_rows, n_panels = 32, 16, args.rows * args.cols

    ecas = []
    for rule in rng.integers(256, size=n_panels):
        x0 = rng.integers(2, size=n_cells)
        ecas.append((f"rule {rule}", EcaEmulator(n_cells, int(rule)).simulate(x0, n_rows - 1)))
    _style.save(grid(ecas, args.rows, args.cols), f"examples-of-ecas_{args.rows}x{args.cols}", args)

    nucas = []
    for _ in range(n_panels):
        rules = rng.integers(256, size=2)
        x0, alloc = rng.integers(2, size=n_cells), rng.integers(2, size=n_cells)
        diagram = NucaEmulator(n_cells, rules, rule_alloc=alloc).simulate(x0, n_rows - 1)
        nucas.append((f"{rules[0]} and {rules[1]}", diagram))
    _style.save(grid(nucas, args.rows, args.cols), f"examples-of-nucas_{args.rows}x{args.cols}", args)


if __name__ == "__main__":
    main()
