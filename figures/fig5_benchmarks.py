"""Fig. 5 (thesis Fig. 7.5): computation times of CellPyLib and the two CNNs.

Redrawn from the archived 2024 measurements in ``data/benchmarks_2024/``.
With ``--overlay 2026`` the re-run of the published protocol
(``scripts/benchmark_published.py``, results in ``data/benchmarks_2026/``) is
drawn on top in grey.
"""
import numpy as np

import _style
from ca_emulators.benchmarks import SCENARIOS, load_timings
from ca_emulators.plotting import plot_benchmark


def main(argv=None):
    p = _style.parser(__doc__.splitlines()[0])
    p.add_argument("--overlay", choices=["2026"], help="also draw the re-run of the protocol")
    args = p.parse_args(argv)
    _style.apply(args)
    for name, scenario in SCENARIOS.items():
        x, times = load_timings(_style.DATA / "benchmarks_2024" / f"{scenario.stem}.npy")
        overlay = None
        if args.overlay:
            x_new, overlay = load_timings(_style.DATA / f"benchmarks_{args.overlay}" / f"{scenario.stem}.npy")
            assert np.array_equal(x_new, x), f"{name}: the re-run used other x-values"
        fig, _ = plot_benchmark(name, x, times, scenario.fixed, overlay=overlay,
                                overlay_label=f"re-run {args.overlay} (grey)")
        _style.save(fig, scenario.stem + (f"-overlay{args.overlay}" if args.overlay else ""), args)


if __name__ == "__main__":
    main()
