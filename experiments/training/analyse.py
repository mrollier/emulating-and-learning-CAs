"""Summarise sweeps: tables (CSV/Markdown) and figures in results/<sweep>/.

    python analyse.py onestep_ablations                 # one sweep
    python analyse.py onestep_ablations minimal_grid    # several
    python analyse.py onestep_ablations --smoke         # the --smoke version

Reads the raw per-member CSVs under results/raw/<sweep>/ and writes, per sweep:

- ``summary.csv`` / ``summary.md``: one row per configuration (success rates
  with Wilson 95% intervals, by f(000), closed-loop exactness, certificate,
  time to exactness, dead units, template recovery, ...);
- ``per_rule_success.csv``: success rate of every rule under every configuration;
- ``families.csv``: success per rule family (f(000)/f(111), linear
  separability, affine rules, Wolfram class);
- ``h1_complement_pairs.csv``: the complement-pair test of the dead-origin hypothesis;
- figures (PNG).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RAW = RESULTS / "raw"

# reference palette (dataviz skill): categorical slots in fixed order, chart chrome
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", \
    "#fcfcfb"

BOOL_COLS = {"exact_init", "exact_final", "exact_ever", "persist", "cl_exact", "template_strict"}

# Wolfram classes of the 88 representatives as commonly cited (Wolfram 2002; the
# class-4 set follows Martinez 2013). Used only as a coarse, indicative grouping.
WOLFRAM_1 = {0, 8, 32, 40, 128, 136, 160, 168}
WOLFRAM_3 = {18, 22, 30, 45, 60, 90, 105, 122, 126, 146, 150}
WOLFRAM_4 = {41, 54, 106, 110}


# --------------------------------------------------------------------------- rule descriptors

@lru_cache(maxsize=None)
def rule_families() -> dict:
    """Per-rule descriptors (arrays indexed by rule number)."""
    from ca_emulators import rules as R

    pats = R.neighbourhood_patterns().astype(int)
    tables = np.stack([R.rule_table(r) for r in range(256)]).astype(int)
    separable = np.zeros(256, bool)
    for w in itertools.product(range(-2, 3), repeat=3):
        s = pats @ np.array(w)
        for theta in np.arange(-4.5, 5.0, 1.0):
            separable[R.rule_number((s > theta).astype(int))] = True
    assert separable.sum() == 104, "3-input threshold functions: expected 104"
    affine = np.zeros(256, bool)
    for a in itertools.product((0, 1), repeat=3):
        for c in (0, 1):
            affine[R.rule_number((pats @ np.array(a) + c) % 2)] = True
    rep = np.array([R.equivalence_class(r)[0] for r in range(256)])
    wolfram = np.array([1 if x in WOLFRAM_1 else 3 if x in WOLFRAM_3 else 4 if x in WOLFRAM_4
                        else 2 for x in rep])
    return {"f000": tables[:, 0], "f111": tables[:, 7], "n_ones": tables.sum(axis=1),
            "separable": separable, "affine": affine, "rep": rep, "wolfram": wolfram,
            "complement": np.array([R.complement(r) for r in range(256)]),
            "reflect": np.array([R.reflect(r) for r in range(256)])}


@lru_cache(maxsize=None)
def t_step_map(rule: int, T: int) -> np.ndarray:
    """Output of T updates on every window of 2T + 1 cells (the T-step local map)."""
    from ca_emulators.rules import rule_table

    width = 2 * T + 1
    windows = ((np.arange(1 << width)[:, None] >> np.arange(width)[::-1]) & 1).astype(np.int64)
    table = rule_table(rule).astype(np.int64)
    x = windows
    for _ in range(T):
        x = table[4 * x[:, :-2] + 2 * x[:, 1:-1] + x[:, 2:]]
    return x[:, 0]


def t_equivalent(learned: int, target: int, T: int) -> bool:
    return learned == target or bool(np.array_equal(t_step_map(learned, T), t_step_map(target, T)))


# --------------------------------------------------------------------------- loading

def _convert(col: str, values: list[str]) -> np.ndarray:
    if col in BOOL_COLS:
        return np.array([v == "True" for v in values])
    if col == "config":
        return np.array(values)
    try:
        return np.array([int(v) for v in values])
    except ValueError:
        return np.array([float(v) if v != "" else np.nan for v in values])


def load_sweep(name: str) -> tuple[dict, dict]:
    """{config: {column: array}} and {config: meta} for all parts found."""
    root = RAW / name
    if not root.exists():
        raise FileNotFoundError(f"no raw results for sweep {name!r} in {RAW}")
    data, metas = {}, {}
    for cfg_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        parts = sorted(cfg_dir.glob("part_*.csv"))
        if not parts:
            continue
        meta = json.loads((cfg_dir / "meta.json").read_text(encoding="utf-8"))
        rows = []
        for part in parts:
            with open(part, encoding="utf-8") as fh:
                rows += list(csv.DictReader(fh))
        cols = {c: _convert(c, [r[c] for r in rows]) for c in rows[0]}
        cols["n_parts_done"] = len(parts)
        data[cfg_dir.name] = cols
        metas[cfg_dir.name] = meta
    order = [c["config"]["name"] for c in _sweep_order(name, metas)]
    return {k: data[k] for k in order}, {k: metas[k] for k in order}


def _sweep_order(name: str, metas: dict) -> list:
    """Configurations in the order of the sweep file when it can be found."""
    base = name.split("-")[0]  # strip -smoke, -ws, ... tags
    path = HERE / "configs" / f"{base}.json"
    if path.exists():
        import sys

        sys.path.insert(0, str(HERE))
        from sweep import expand_configs

        order = [c.name for c in expand_configs(json.loads(path.read_text(encoding="utf-8")))]
        known = [n for n in order if n in metas] + [n for n in metas if n not in order]
        return [metas[n] for n in known]
    return list(metas.values())


# --------------------------------------------------------------------------- statistics

def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def per_rule(cols: dict, key: str = "exact_final") -> np.ndarray:
    """Success rate of each of the 256 rules (nan for rules not in the sweep)."""
    out = np.full(256, np.nan)
    rules = cols["rule"]
    for r in np.unique(rules):
        out[r] = cols[key][rules == r].mean()
    return out


def summarise(name: str, cols: dict, meta: dict) -> dict:
    cfg = meta["config"]
    fam = rule_families()
    ok = cols["exact_final"]
    n = len(ok)
    lo, hi = wilson(ok.sum(), n)
    rules = cols["rule"]
    odd = fam["f000"][rules] == 1
    pr = per_rule(cols)
    pr = pr[~np.isnan(pr)]
    s = {"config": name, "n": n, "rules": len(pr), "seeds": len(np.unique(cols["seed"])),
         "width": cfg["width"], "depth": cfg["depth"], "params": 4 * cfg["width"]
         + cfg["depth"] * (cfg["width"] ** 2 + cfg["width"]) + cfg["width"] + 1,
         "success": ok.mean(), "ci_lo": lo, "ci_hi": hi,
         "success_f000_0": ok[~odd].mean() if (~odd).any() else math.nan,
         "success_f000_1": ok[odd].mean() if odd.any() else math.nan,
         "exact_ever": cols["exact_ever"].mean(),
         "persist": cols["persist"].mean(),
         "rules_all_seeds": (pr == 1).mean(), "rules_ge90": (pr >= 0.9).mean(),
         "rules_zero": (pr == 0).mean(), "min_rule_success": pr.min(),
         "cl_exact": cols["cl_exact"].mean(),
         "cl_given_exact": cols["cl_exact"][ok].mean() if ok.any() else math.nan,
         "cert_given_exact": (cols["cl_cert_eps"][ok] > 0).mean() if ok.any() else math.nan,
         "median_first_exact": float(np.median(cols["first_exact_step"][ok])) if ok.any()
         else math.nan,
         "p90_first_exact": float(np.percentile(cols["first_exact_step"][ok], 90)) if ok.any()
         else math.nan,
         "margin_exact": float(np.median(cols["margin_final"][ok])) if ok.any() else math.nan,
         "dead_l1_init": cols["dead_l1_init"].mean(), "dead_l1_final": cols["dead_l1_final"].mean(),
         "stuck_out_final": cols["stuck_out_final"].mean(),
         "err000_given_fail_odd": ((cols["err_mask"] & 1) > 0)[odd & ~ok].mean()
         if (odd & ~ok).any() else math.nan}
    if cfg["width"] == 8 and cfg["depth"] == 0:
        s["template_given_exact"] = cols["template_strict"][ok].mean() if ok.any() else math.nan
        s["cover_pure_exact"] = cols["cover_pure"][ok].mean() if ok.any() else math.nan
    else:
        s["template_given_exact"] = math.nan
        s["cover_pure_exact"] = cols["cover_pure"][ok].mean() if ok.any() else math.nan
    if cfg.get("pretrain"):
        s["pretrain_restarts"] = np.nanmean(cols["pretrain_restarts"])
        s["pretrain_none_passed"] = (cols["pretrain_pass"] == 0).mean()
    if cfg.get("T", 1) > 1:
        T = cfg["T"]
        eq = np.array([t_equivalent(int(a), int(b), T) for a, b in zip(cols["learned_rule"], rules)])
        s["t_equiv"] = eq.mean()
    else:
        s["t_equiv"] = ok.mean()
    if "seconds_per_part" in cols:
        # every row of a part carries that part's wall time
        secs = sum(np.unique(cols["seconds_per_part"][cols["seed"] == sd]).sum()
                   for sd in np.unique(cols["seed"]))
        s["cpu_s_per_1k"] = 1000 * secs / n
    for key in ("T", "frames", "curriculum", "ste", "head", "enc", "act", "bias_init", "lr",
                "data", "batch", "steps", "pretrain"):
        s[key] = cfg.get(key)
    return s


def family_table(data: dict) -> list[dict]:
    fam = rule_families()
    groups = {
        "f000=0,f111=0": lambda r: (fam["f000"][r] == 0) & (fam["f111"][r] == 0),
        "f000=0,f111=1": lambda r: (fam["f000"][r] == 0) & (fam["f111"][r] == 1),
        "f000=1,f111=0": lambda r: (fam["f000"][r] == 1) & (fam["f111"][r] == 0),
        "f000=1,f111=1": lambda r: (fam["f000"][r] == 1) & (fam["f111"][r] == 1),
        "linearly separable": lambda r: fam["separable"][r],
        "not separable": lambda r: ~fam["separable"][r],
        "affine (XOR-type)": lambda r: fam["affine"][r],
        "Wolfram 1": lambda r: fam["wolfram"][r] == 1,
        "Wolfram 2": lambda r: fam["wolfram"][r] == 2,
        "Wolfram 3": lambda r: fam["wolfram"][r] == 3,
        "Wolfram 4": lambda r: fam["wolfram"][r] == 4,
    }
    out = []
    for name, cols in data.items():
        row = {"config": name}
        for g, sel in groups.items():
            mask = sel(cols["rule"])
            row[g] = cols["exact_final"][mask].mean() if mask.any() else math.nan
        out.append(row)
    return out


def h1_pairs(data: dict) -> list[dict]:
    """Complement pairs that swap f(000): R with f(000)=f(111)=1 vs c(R) with both 0.

    Complementation exchanges the roles of the two states, so R and c(R) have
    the same dynamics; only the encoding treats them differently.
    """
    fam = rule_families()
    odd_rules = [r for r in range(256) if fam["f000"][r] == 1 and fam["f111"][r] == 1]
    refl = [(r, int(fam["reflect"][r])) for r in range(256) if r < fam["reflect"][r]]
    out = []
    for name, cols in data.items():
        pr = per_rule(cols)
        a = np.array([pr[r] for r in odd_rules])
        b = np.array([pr[fam["complement"][r]] for r in odd_rules])
        keep = ~np.isnan(a) & ~np.isnan(b)
        ra = np.array([pr[x] for x, _ in refl])
        rb = np.array([pr[y] for _, y in refl])
        rk = ~np.isnan(ra) & ~np.isnan(rb)
        n_seeds = len(np.unique(cols["seed"]))
        out.append({"config": name, "pairs": int(keep.sum()),
                    "success_f000_1": a[keep].mean() if keep.any() else math.nan,
                    "success_complement_f000_0": b[keep].mean() if keep.any() else math.nan,
                    "mean_diff": (a - b)[keep].mean() if keep.any() else math.nan,
                    "pairs_odd_worse": int((a < b)[keep].sum()),
                    "pairs_odd_better": int((a > b)[keep].sum()),
                    "reflection_pairs": int(rk.sum()),
                    "reflection_mean_abs_diff": np.abs(ra - rb)[rk].mean() if rk.any()
                    else math.nan,
                    "binomial_expected_abs_diff": float(np.mean(
                        np.sqrt(2 * ra[rk] * (1 - ra[rk]) / n_seeds) * math.sqrt(2 / math.pi)))
                    if rk.any() else math.nan})
    return out


# --------------------------------------------------------------------------- output

def _fmt(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        if math.isnan(v):
            return ""
        return f"{v:.4g}"
    return str(v)


def write_table(rows: list[dict], path: Path, md_cols: list[str] | None = None) -> None:
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows([{k: _fmt(r.get(k, "")) for k in cols} for r in rows])
    if md_cols:
        lines = ["| " + " | ".join(md_cols) + " |", "|" + "---|" * len(md_cols)]
        for r in rows:
            lines.append("| " + " | ".join(_fmt(r.get(c, "")) for c in md_cols) + " |")
        path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_success_bars(summary: list[dict], path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    names = [s["config"] for s in summary]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(names) + 1.3), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    h = 0.38
    for k, (key, label) in enumerate([("success_f000_0", "rules with f(000) = 0 (even)"),
                                      ("success_f000_1", "rules with f(000) = 1 (odd)")]):
        vals = np.array([s[key] for s in summary], float)
        ax.barh(y + (k - 0.5) * h, vals, height=h - 0.04, color=SERIES[k], label=label)
    all_ = np.array([s["success"] for s in summary])
    ax.scatter(all_, y, marker="|", s=120, color=INK, linewidths=1.5, label="all rules", zorder=3)
    ax.set_yticks(y, names, fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of runs exact after training (exact@1)", color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left", pad=24)
    ax.legend(fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=3,
              borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_rule_grids(data: dict, names: list[str], path: Path, key: str = "exact_final") -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n + 0.8, 3.5), facecolor=SURFACE,
                             squeeze=False)
    for ax, name in zip(axes[0], names):
        grid = per_rule(data[name], key).reshape(16, 16)
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(name, fontsize=9, color=INK)
        ax.set_xticks([0, 5, 10, 15])
        ax.set_yticks([0, 5, 10, 15], [0, 80, 160, 240])
        ax.tick_params(labelsize=7, colors=INK2)
        ax.set_xlabel("rule mod 16", fontsize=8, color=INK2)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[0][0].set_ylabel("rule - rule mod 16", fontsize=8, color=INK2)
    cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02)
    cb.set_label("success rate per rule", fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.outline.set_visible(False)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_h1(data: dict, names: list[str], path: Path) -> None:
    import matplotlib.pyplot as plt

    fam = rule_families()
    odd_rules = [r for r in range(256) if fam["f000"][r] == 1 and fam["f111"][r] == 1]
    fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 3.2), facecolor=SURFACE,
                             squeeze=False)
    for ax, name in zip(axes[0], names):
        _style(ax)
        pr = per_rule(data[name])
        a = np.array([pr[r] for r in odd_rules])
        b = np.array([pr[fam["complement"][r]] for r in odd_rules])
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
        ax.scatter(b, a, s=16, color=SERIES[0], edgecolors=SURFACE, linewidths=0.8, zorder=3)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(name, fontsize=9, color=INK)
        ax.set_xlabel("complement c(R): f(000) = 0", fontsize=8, color=INK2)
    axes[0][0].set_ylabel("rule R: f(000) = f(111) = 1", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_time_to_exact(data: dict, metas: dict, names: list[str], path: Path) -> None:
    """Fraction of runs exact (and staying exact) by training step."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.6), facecolor=SURFACE)
    _style(ax)
    for k, name in enumerate(names[:8]):
        cols = data[name]
        steps = metas[name]["config"]["steps"]
        grid = np.linspace(0, steps, 201)
        first = cols["first_exact_step"]
        stays = cols["persist"]
        frac = [(stays & (first >= 0) & (first <= g)).mean() for g in grid]
        ax.plot(grid, frac, color=SERIES[k], linewidth=2, label=name)
    ax.set_ylim(0, 1)
    ax.set_xlabel("optimisation step", fontsize=9, color=INK2)
    ax.set_ylabel("fraction exact from then on", fontsize=9, color=INK2)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_width_depth(summary: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    heads = list(dict.fromkeys(s["head"] for s in summary))
    fig, axes = plt.subplots(1, len(heads), figsize=(3.6 * len(heads), 3.3), facecolor=SURFACE,
                             squeeze=False, sharey=True)
    for ax, head in zip(axes[0], heads):
        _style(ax)
        rows = [s for s in summary if s["head"] == head]
        for k, d in enumerate(sorted({s["depth"] for s in rows})):
            pts = sorted((s["width"], s["success"]) for s in rows if s["depth"] == d)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", markersize=5,
                    color=SERIES[k], linewidth=2, label=f"D = {d} extra 1x1 layers")
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32, 64], ["8", "16", "32", "64"])
        ax.set_ylim(0, 1.02)
        ax.set_title(head, fontsize=9, color=INK)
        ax.set_xlabel("width H", fontsize=8, color=INK2)
    axes[0][0].set_ylabel("fraction exact (exact@1)", fontsize=8, color=INK2)
    axes[0][-1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_spacetime(summary: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    """Three panels (one-step exact, T-step map exact, closed loop), one bar per configuration."""
    keys = [("success", "one-step rule exact"), ("t_equiv", "T-step map exact"),
            ("cl_exact", "closed loop exact (100 steps)")]
    names = [s["config"] for s in summary]
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 0.3 * len(names) + 1.3), facecolor=SURFACE,
                             sharey=True)
    for ax, (key, title) in zip(axes, keys):
        _style(ax)
        ax.grid(axis="y", visible=False)
        vals = np.array([s[key] for s in summary], float)
        ax.barh(y, vals, height=0.7, color=SERIES[0])
        for yi, v in zip(y, vals):
            ax.text(min(v, 1.0) + 0.02, yi, f"{100 * v:.0f}%", va="center", fontsize=6.5,
                    color=INK2)
        ax.set_xlim(0, 1.18)
        ax.set_xticks([0, 0.5, 1])
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    axes[0].set_yticks(y, names, fontsize=7.5, color=INK)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_mechanism(data: dict, path: Path, names=("nopre", "bias0.1", "baseline_2024")) -> None:
    """H2: success vs stuck neighbourhoods at initialisation; H1: success by f(000) and #ones."""
    import matplotlib.pyplot as plt

    names = [n for n in names if n in data]
    fam = rule_families()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURFACE)
    ax = axes[0]
    _style(ax)
    width = 0.8 / len(names)
    for k, name in enumerate(names):
        c = data[name]
        stuck = c["stuck_out_init"]
        xs = np.arange(0, 6)
        ys = [c["exact_final"][stuck == x].mean() if (stuck == x).sum() >= 20 else np.nan for x in xs]
        ax.bar(xs + (k - (len(names) - 1) / 2) * width, ys, width=width - 0.03, color=SERIES[k],
               label=name)
    ax.set_xlabel("target-1 neighbourhoods with output pre-activation <= 0 at initialisation",
                  fontsize=8, color=INK2)
    ax.set_ylabel("fraction exact after training", fontsize=8, color=INK2)
    ax.set_ylim(0, 1.02)
    ax.set_title("H2: the output ReLU of the 2024 head", fontsize=9, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    _style(ax)
    c = data[names[0]]
    r = c["rule"]
    for k, (f0, label) in enumerate([(0, "f(000) = 0"), (1, "f(000) = 1")]):
        xs = np.arange(1, 8)
        ys = [c["exact_final"][(fam["f000"][r] == f0) & (fam["n_ones"][r] == x)].mean() for x in xs]
        ax.plot(xs, ys, marker="o", markersize=5, color=SERIES[k], linewidth=2, label=label)
    ax.set_xlabel("number of neighbourhoods mapped to 1 (rule-table weight)", fontsize=8,
                  color=INK2)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"H1: the dead origin ({names[0]})", fontsize=9, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def compare_keras(data: dict, path: Path, name: str = "baseline_2024") -> None:
    """Per rule: the real Keras 2024 recipe (validate_2024.py) vs the ensemble's baseline."""
    keras_csv = RESULTS / "validate_2024.csv"
    if not keras_csv.exists() or name not in data:
        return
    with open(keras_csv, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    c = data[name]
    out = []
    for rule in sorted({int(r["rule"]) for r in rows}):
        kr = [r for r in rows if int(r["rule"]) == rule]
        k_ok = [r["exact_final"] == "True" for r in kr]
        k_cl = [r["cl_exact"] == "True" for r in kr if r["exact_final"] == "True"]
        m = c["rule"] == rule
        e_ok = c["exact_final"][m]
        out.append({"rule": rule, "keras_runs": len(kr), "keras_exact": sum(k_ok) / len(kr),
                    "keras_closed_loop_given_exact": (sum(k_cl) / len(k_cl)) if k_cl else math.nan,
                    "keras_mean_restarts": np.mean([float(r["pretrain_restarts"]) for r in kr]),
                    "ensemble_runs": int(m.sum()), "ensemble_exact": e_ok.mean(),
                    "ensemble_closed_loop_given_exact": c["cl_exact"][m & c["exact_final"]].mean()
                    if e_ok.any() else math.nan,
                    "ensemble_mean_restarts": np.nanmean(c["pretrain_restarts"][m])})
    write_table(out, path, list(out[0]))


MD_COLS = ["config", "n", "success", "ci_lo", "ci_hi", "success_f000_0", "success_f000_1",
           "rules_all_seeds", "min_rule_success", "persist", "cl_given_exact",
           "cert_given_exact", "median_first_exact", "margin_exact", "dead_l1_final",
           "template_given_exact"]


def analyse(name: str, grid_configs: list[str] | None = None) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    data, metas = load_sweep(name)
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    summary = [summarise(k, data[k], metas[k]) for k in data]
    extra = [c for c in ("T", "frames", "curriculum", "ste") if
             len({s.get(c) for s in summary}) > 1]
    if any((s.get("T") or 1) > 1 for s in summary):
        extra += ["t_equiv", "cl_exact"]
    write_table(summary, out / "summary", MD_COLS + extra)
    rows = [{"rule": r, **{k: _fmt(per_rule(data[k])[r]) for k in data}} for r in range(256)]
    write_table(rows, out / "per_rule_success")
    write_table(family_table(data), out / "families", list(family_table(data)[0]))
    h1 = h1_pairs(data)
    write_table(h1, out / "h1_complement_pairs", list(h1[0]))
    plot_success_bars(summary, out / "fig_success_by_config.png", name)
    names = list(data)
    shown = grid_configs or names[:3]
    shown = [s for s in shown if s in data]
    if shown and all(len(np.unique(data[s]["rule"])) == 256 for s in shown):
        plot_rule_grids(data, shown, out / "fig_rule_grid.png")
        plot_h1(data, shown, out / "fig_h1_complement_pairs.png")
    plot_time_to_exact(data, metas, shown or names[:4], out / "fig_time_to_exact.png")
    if len({s["width"] for s in summary}) > 1 and len({s["depth"] for s in summary}) > 1:
        plot_width_depth(summary, out / "fig_width_depth.png")
    if len({s["T"] for s in summary}) > 1:
        plot_spacetime(summary, out / "fig_spacetime.png")
    if "nopre" in data and "stuck_out_init" in data["nopre"]:
        plot_mechanism(data, out / "fig_mechanism.png")
    compare_keras(data, out / "keras_vs_ensemble_2024")
    print((out / "summary.md").read_text(encoding="utf-8"))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sweeps", nargs="+")
    parser.add_argument("--smoke", action="store_true", help="analyse the --smoke outputs")
    parser.add_argument("--grid", default="", help="configs for the per-rule panels (comma list)")
    args = parser.parse_args(argv)
    for name in args.sweeps:
        analyse(f"{name}-smoke" if args.smoke else name,
                [g for g in args.grid.split(",") if g] or None)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
