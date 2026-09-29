"""Fig. 5 (thesis Fig. 7.5): computation times of CellPyLib and the two CNNs.

Redrawn from the archived 2024 measurements in ``data/benchmarks_2024/``.
With ``--overlay 2026`` the re-run of the published protocol
(``scripts/benchmark_published.py``, results in ``data/benchmarks_2026/``) is
drawn on top in grey.
"""
import numpy as np

import _style
from ca_emulators.plotting import plot_benchmark

#: scenario -> (file stem, title parameters), as in Tab. 7.2
SCENARIOS = {
    "Nrules": ("nuca-comparison-Nrules-N256_T32_S32_avg-from-10", dict(S=32, N=256, T=32)),
    "T": ("nuca-comparison-T-N64_Nrules4_S32_avg-from-10", dict(S=32, N=64, Nrules=4)),
    "N": ("nuca-comparison-N-T32_Nrules4_S32_avg-from-10", dict(S=32, Nrules=4, T=32)),
    "S": ("nuca-comparison-S-N32_T32_Nrules4_avg-from-10", dict(N=32, Nrules=4, T=32)),
}


def load(path):
    """x-values and the three (repeats, n) timing arrays, stored in sequence."""
    with open(path, "rb") as f:
        return np.load(f), [np.load(f) for _ in range(3)]


def main(argv=None):
    p = _style.parser(__doc__.splitlines()[0])
    p.add_argument("--overlay", choices=["2026"], help="also draw the re-run of the protocol")
    args = p.parse_args(argv)
    _style.apply(args)
    for scenario, (stem, params) in SCENARIOS.items():
        x, times = load(_style.DATA / "benchmarks_2024" / f"{stem}.npy")
        overlay = None
        if args.overlay:
            x_new, overlay = load(_style.DATA / f"benchmarks_{args.overlay}" / f"{stem}.npy")
            assert np.array_equal(x_new, x), f"{scenario}: the re-run used other x-values"
        fig, _ = plot_benchmark(scenario, x, times, params, overlay=overlay)
        _style.save(fig, stem + (f"-overlay{args.overlay}" if args.overlay else ""), args)


if __name__ == "__main__":
    main()
