"""Figures and summary tables from a benchmark results file (see README.md).

    python experiments/benchmarks/plot.py                 # results/benchmark_full.csv
    python experiments/benchmarks/plot.py --smoke         # results/raw/smoke/benchmark_smoke.csv
    python experiments/benchmarks/plot.py --csv PATH [--formats png,pdf]

Writes, next to the CSV, ``<tag>_overview``, ``<tag>_drivers_lc1``,
``<tag>_drivers_dense``, ``<tag>_selectors`` and ``<tag>_cold`` (PNG and PDF;
rows: thread setting, columns: scenario, the Tab. 7.2 range shaded), and
``<tag>_summary.md`` with the numbers REPORT.md asks for. Only timings with
status "ok" (validated against the reference) are shown.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import run

NUMERIC = ("value", "N", "n_rules", "T", "n_updates", "S", "import_s", "build_s", "first_call_s",
           "cold_s", "warm_median_s", "warm_q1_s", "warm_q3_s", "warm_min_s", "warm_max_s",
           "repeats", "number", "selector_mb", "peak_memory_mb", "wall_s")

# Reference categorical palette (dataviz skill, light mode), in its fixed order;
# markers are the secondary encoding, so identity never rests on colour alone.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, INK_2, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#f0efec"

DRIVER_STYLE = {  # colour and marker per driver
    "predict": (BLUE, "o"), "eager": (ORANGE, "s"), "compiled": (AQUA, "^"), "xla": (YELLOW, "v"),
    "while_loop": (MAGENTA, "D"), "while_loop_xla": (GREEN, "P"), "unrolled": (VIOLET, "*")}
SELECTOR_STYLE = {"lc1": (BLUE, "o"), "lc2": (ORANGE, "s"), "lc3": (AQUA, "^"),
                  "dense": (YELLOW, "v"), "elem": (MAGENTA, "D")}
REFERENCE_STYLE = {
    "cellpylib": dict(color=RED, marker="X", label="CellPyLib"),
    "numpy": dict(color=INK, marker="o", label="numpy (reference.evolve)"),
    "numpy_lut": dict(color=INK_2, marker="x", label="numpy, per-cell lookup table"),
}
DRIVER_LABEL = {"predict": "predict per update", "eager": "eager call per update",
                "compiled": "tf.function per update", "xla": "tf.function + XLA per update",
                "while_loop": "tf.while_loop", "while_loop_xla": "tf.while_loop + XLA",
                "unrolled": "unrolled model"}
SELECTOR_LABEL = {"lc1": "LocallyConnected1D (impl. 1)", "lc2": "LocallyConnected1D (impl. 2)",
                  "lc3": "LocallyConnected1D (impl. 3)", "dense": "dense", "elem": "elementwise"}
THREAD_LABEL = {"default": "default threads", "single": "single thread"}


@dataclass(frozen=True)
class Series:
    method: str
    label: str
    color: str
    marker: str
    linestyle: str = "-"
    filled: bool = True


def reference_series(names=("cellpylib", "numpy", "numpy_lut")):
    return [Series(n, REFERENCE_STYLE[n]["label"], REFERENCE_STYLE[n]["color"],
                   REFERENCE_STYLE[n]["marker"]) for n in names]


def driver_series(selector):
    return [Series(f"{d}:{selector}", DRIVER_LABEL[d], *DRIVER_STYLE[d]) for d in DRIVER_STYLE]


def selector_series():
    out = []
    for driver, ls, filled in (("compiled", "-", True), ("xla", "--", False)):
        for sel, (color, marker) in SELECTOR_STYLE.items():
            if (driver, sel) not in {("xla", "lc3")}:
                out.append(Series(f"{driver}:{sel}", f"{SELECTOR_LABEL[sel]}, "
                                  f"{'tf.function' if driver == 'compiled' else '+ XLA'}",
                                  color, marker, ls, filled))
    return out


def overview_series():
    return [*reference_series(),
            Series("predict:lc1", "CNN, LocallyConnected1D, predict (as 2024)", BLUE, "o"),
            Series("predict:dense", "CNN, dense, predict (as 2024)", BLUE, "o", "--", False),
            Series("compiled:lc1", "CNN, LocallyConnected1D, tf.function", AQUA, "^"),
            Series("xla:elem", "CNN, elementwise, tf.function + XLA", YELLOW, "v"),
            Series("while_loop_xla:elem", "CNN, elementwise, tf.while_loop + XLA", GREEN, "P")]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key in NUMERIC:
                row[key] = float(row[key]) if row.get(key) not in (None, "") else None
            rows.append(row)
    return rows


def scenarios_for(meta_path: Path, smoke: bool) -> dict:
    """Scenario definitions of the run (from its metadata when available)."""
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {k: run.Scenario(**{**v, "core": tuple(v["core"]), "extended": tuple(v["extended"])})
                for k, v in meta["scenarios"].items()}
    return run.SMOKE_SCENARIOS if smoke else run.SCENARIOS


def select(rows, scenario, threads, method, field="warm_median_s"):
    """Sorted ok points (x, y, q1, q3) of one method; thread-insensitive methods reuse 'default'."""
    if method in run.THREAD_INSENSITIVE:
        threads = "default"
    pts = sorted((r["value"], r[field], r.get("warm_q1_s"), r.get("warm_q3_s")) for r in rows
                 if r["scenario"] == scenario and r["threads"] == threads and r["method"] == method
                 and r["status"] == "ok" and r[field] is not None)
    return pts


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _style(plt):
    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 8,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True, "savefig.bbox": "tight", "pdf.fonttype": 42})


def figure(rows, scenarios, series, title, path_stem: Path, formats, field="warm_median_s",
           ylabel="time per diagram (s)", import_line=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)

    threads = [t for t in run.THREAD_SETTINGS if any(r["threads"] == t for r in rows)]
    fig, axes = plt.subplots(len(threads), len(scenarios), figsize=(3.3 * len(scenarios), 2.6 * len(threads)),
                             squeeze=False, sharey=True)
    handles = {}
    for i, thread in enumerate(threads):
        for j, scenario in enumerate(scenarios.values()):
            ax = axes[i, j]
            if scenario.core:
                ax.axvspan(min(scenario.core), max(scenario.core), color=BAND, zorder=0, lw=0)
            for s in series:
                pts = select(rows, scenario.name, thread, s.method, field)
                if not pts:
                    continue
                x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
                kw = dict(color=s.color, marker=s.marker, linestyle=s.linestyle, linewidth=1.4,
                          markersize=4.5, markerfacecolor=s.color if s.filled else "white",
                          markeredgewidth=1.0, label=s.label)
                if field == "warm_median_s":
                    lo = np.array([p[2] for p in pts])
                    hi = np.array([p[3] for p in pts])
                    h = ax.errorbar(x, y, yerr=[y - lo, hi - y], elinewidth=0.8, capsize=0, **kw)
                else:
                    h = ax.plot(x, y, **kw)[0]
                handles.setdefault(s.label, h)
            if import_line:
                imp = [r["import_s"] for r in rows if r["threads"] == thread and r["import_s"]
                       and r["method"] not in ("numpy", "numpy_lut", "cellpylib")]
                if imp:
                    h = ax.axhline(float(np.median(imp)), color=MUTED, lw=1.0, zorder=1,
                                   label="import TensorFlow + package (median)")
                    handles.setdefault(h.get_label(), h)
            ax.set_xscale("log", base=10 if scenario.vary == "T" else 2)
            ax.set_yscale("log")
            if i == len(threads) - 1:
                ax.set_xlabel(scenario.label)
            if j == 0:
                ax.set_ylabel(f"{THREAD_LABEL[thread]}\n{ylabel}")
            if i == 0:
                fixed = ", ".join(f"{k.replace('n_rules', 'N_R')}={v}" for k, v in scenario.fixed.items())
                ax.set_title(f"vary {scenario.vary.replace('n_rules', 'N_R')} ({fixed})")
    note = "shaded: range of thesis Tab. 7.2" + (
        "; error bars: interquartile range" if field == "warm_median_s" else "; single measurement")
    fig.suptitle(f"{title}  ({note})", fontsize=9.5, x=0.01, ha="left")
    fig.legend(list(handles.values()), list(handles.keys()), loc="lower center", frameon=False,
               ncol=min(4, len(handles)), bbox_to_anchor=(0.5, -0.02 - 0.035 * ((len(handles) - 1) // 4)))
    fig.tight_layout(rect=(0, 0.04 + 0.035 * ((len(handles) - 1) // 4), 1, 0.97))
    written = []
    for fmt in formats:
        out = path_stem.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=110 if fmt == "png" else None)
        written.append(out)
    plt.close(fig)
    return written


# --------------------------------------------------------------------------
# Summary tables
# --------------------------------------------------------------------------

def fmt(s):
    if s is None:
        return "–"
    return f"{s * 1e6:.0f} µs" if s < 1e-3 else f"{s * 1e3:.3g} ms" if s < 1 else f"{s:.3g} s"


def cell(rows, scenario, value, threads, method, field="warm_median_s"):
    if method in run.THREAD_INSENSITIVE:
        threads = "default"
    for r in rows:
        if (r["scenario"], r["value"], r["threads"], r["method"]) == (scenario, value, threads, method):
            if r["status"] == "ok":
                return fmt(r[field])
            return {"skipped_memory": "mem.", "skipped_predicted": "skip", "skipped_after_failure": "skip",
                    "over_budget": "budget", "timeout": "timeout"}.get(r["status"], r["status"])
    return ""


def best_neural(rows, scenario, value, threads):
    cands = [r for r in rows if r["scenario"] == scenario and r["value"] == value
             and r["threads"] == threads and r["status"] == "ok" and r["driver"] not in
             ("cellpylib", "numpy", "numpy_lut")]
    return min(cands, key=lambda r: r["warm_median_s"]) if cands else None


def summary(rows, scenarios, tag) -> str:
    methods = list(dict.fromkeys(r["method"] for r in rows))
    threads = [t for t in run.THREAD_SETTINGS if any(r["threads"] == t for r in rows)]
    lines = [f"# Benchmark summary ({tag})", "",
             "Generated by `plot.py`. Times are warm medians per complete diagram unless stated; "
             "'skip' = skipped (budget predicted or an earlier point failed), 'mem.' = selector "
             "kernel above the memory cap, 'budget' = over the time budget.", ""]

    # A. Fixed costs where computation is negligible.
    small = min(scenarios.values(), key=lambda s: s.case(min(s.points())[0]).N * s.case(min(s.points())[0]).S)
    v0 = min(small.points())[0]
    case0 = small.case(v0)
    lines += [f"## A. Fixed costs at the smallest case ({small.vary}={v0}: N={case0.N}, "
              f"N_R={case0.n_rules}, T={case0.T}, S={case0.S})", "",
              "| method | threads | import | build | first call | cold (build + first) | warm | warm per update |",
              "|---|---|---|---|---|---|---|---|"]
    for m in methods:
        for t in threads:
            r = next((r for r in rows if r["scenario"] == small.name and r["value"] == v0
                      and r["threads"] == t and r["method"] == m and r["status"] == "ok"), None)
            if r:
                lines.append(f"| {m} | {t} | {fmt(r['import_s'])} | {fmt(r['build_s'])} | "
                             f"{fmt(r['first_call_s'])} | {fmt(r['cold_s'])} | {fmt(r['warm_median_s'])} | "
                             f"{fmt(r['warm_median_s'] / r['n_updates'])} |")
    lines.append("")

    # B. Warm times at selected points of every scenario.
    lines += ["## B. Warm time per diagram at selected points", ""]
    for s in scenarios.values():
        values = [v for v, _ in s.points()]
        picks = sorted({values[0], max(s.core) if s.core else values[-1], values[-1]})
        header = [f"{s.vary}={int(v)} ({t})" for v in picks for t in threads]
        lines += [f"### Scenario {s.name} ({', '.join(f'{k}={v}' for k, v in s.fixed.items())})", "",
                  "| method | " + " | ".join(header) + " |", "|---" * (len(header) + 1) + "|"]
        for m in methods:
            cells = [cell(rows, s.name, float(v), t, m) for v in picks for t in threads]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")

    # C. numpy against the fastest neural emulator.
    lines += ["## C. numpy against the fastest exact CNN (per point)", "",
              "| scenario | threads | points | numpy faster | numpy_lut faster | "
              "time(best CNN) / time(numpy): min – max | best CNN at the largest point |",
              "|---|---|---|---|---|---|---|"]
    for s in scenarios.values():
        for t in threads:
            n = faster = faster_lut = 0
            ratios, last = [], None
            for v, _ in s.points():
                best = best_neural(rows, s.name, float(v), t)
                ref = {m: next((r for r in rows if r["scenario"] == s.name and r["value"] == float(v)
                                and r["threads"] == t and r["method"] == m and r["status"] == "ok"), None)
                       for m in ("numpy", "numpy_lut")}
                if best is None or ref["numpy"] is None:
                    continue
                n += 1
                faster += ref["numpy"]["warm_median_s"] < best["warm_median_s"]
                if ref["numpy_lut"] is not None:
                    faster_lut += ref["numpy_lut"]["warm_median_s"] < best["warm_median_s"]
                ratios.append(best["warm_median_s"] / ref["numpy"]["warm_median_s"])
                last = (v, best["method"])
            if n:
                lines.append(f"| {s.name} | {t} | {n} | {faster} | {faster_lut} | "
                             f"{min(ratios):.3g} – {max(ratios):.3g} | {last[1]} ({s.vary}={last[0]}) |")
    lines.append("")

    # D. Status counts.
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    lines += ["## D. Job statuses", "", "| status | jobs |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(counts.items())]
    bad = [r for r in rows if r["status"] in ("mismatch", "error")]
    if bad:
        lines += ["", "Failures:", ""] + [f"- {r['method']} {r['threads']} {r['scenario']} "
                                          f"{r['vary']}={r['value']:g}: {r['status']} {r['note']}" for r in bad]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, help="results file (default: results/benchmark_full.csv)")
    p.add_argument("--smoke", action="store_true", help="use results/raw/smoke/benchmark_smoke.csv")
    p.add_argument("--formats", default="png,pdf")
    args = p.parse_args(argv)
    csv_path = args.csv or (run.RESULTS / "raw" / "smoke" / "benchmark_smoke.csv" if args.smoke
                            else run.RESULTS / "benchmark_full.csv")
    tag = csv_path.stem.removeprefix("benchmark_")
    rows = load(csv_path)
    scenarios = scenarios_for(csv_path.with_name(f"{csv_path.stem}_meta.json"), args.smoke)
    scenarios = {k: s for k, s in scenarios.items() if any(r["scenario"] == k for r in rows)}
    formats = args.formats.split(",")
    out = csv_path.parent / tag
    written = []
    written += figure(rows, scenarios, overview_series(), "Exact nuCA simulation: overview",
                      out.with_name(f"{tag}_overview"), formats)
    for sel in ("lc1", "dense"):
        written += figure(rows, scenarios, [*driver_series(sel), *reference_series(("numpy",))],
                          f"Drivers of the CNN with the {SELECTOR_LABEL[sel]} selector",
                          out.with_name(f"{tag}_drivers_{sel}"), formats)
    written += figure(rows, scenarios, [*selector_series(), *reference_series(("numpy",))],
                      "Selectors of the CNN (solid: tf.function, dashed: + XLA)",
                      out.with_name(f"{tag}_selectors"), formats)
    cold_series = [*driver_series("lc1"),
                   Series("while_loop_xla:elem", "elementwise, tf.while_loop + XLA", GREEN, "P", "--", False),
                   *reference_series(("numpy",))]
    written += figure(rows, scenarios, cold_series,
                      "Cold time: set-up + first call (LocallyConnected1D selector unless stated)",
                      out.with_name(f"{tag}_cold"), formats, field="cold_s",
                      ylabel="cold time (s)", import_line=True)
    md = out.with_name(f"{tag}_summary.md")
    md.write_text(summary(rows, scenarios, tag), encoding="utf-8")
    written.append(md)
    for w in written:
        print(w)


if __name__ == "__main__":
    main()
