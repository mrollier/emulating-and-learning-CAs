"""Do trained minimal networks recover the analytic detector templates?

The analytic emulator's width-3 layer is a one-hot "detector" layer: unit i
fires on neighbourhood i only, so the 8 x 8 response matrix (neighbourhoods x
units) is the identity. This script retrains a sample of minimal networks
(same configuration names, hence the same seeds, as in the sweeps), keeps
their weights, and describes their response matrices:

- the size of each unit's firing set (neighbourhoods with pre-activation > 0;
  1 for every unit of the template),
- the rank of the 8 x 8 response matrix (8 for the template),
- how many units the rule-table layer actually uses,

and draws the response matrices of a few trained networks next to the template.

    python templates.py            # writes results/templates/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "templates"

CONFIGS = {
    # name in the sweeps -> Config overrides
    "baseline_2024": {"pretrain": True},
    "sigmoid_bce_01_b0.0_relu": {"head": "sigmoid_bce"},
    "recipe_minimal": {"head": "sigmoid_bce", "enc": "pm1", "act": "softplus"},
}
SEEDS = range(4)


def responses(params, cfg):
    import ensemble as E
    import tensorflow as tf

    u = tf.constant(E.encode(E.PATTERNS, cfg.enc))
    p = {k: tf.constant(v) for k, v in params.items()}
    pres, z = E.forward(p, u, cfg)
    pre = pres[0].numpy()                                   # (M, 8, H)
    act = E.activation_fn(cfg)(pres[0]).numpy()
    return pre, act, z.numpy()


def main() -> int:
    sys.path.insert(0, str(HERE))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import ensemble as E
    from analyse import BLUE_RAMP, INK, INK2, SURFACE, write_table
    from matplotlib.colors import LinearSegmentedColormap
    from ca_emulators import weights as W

    OUT.mkdir(parents=True, exist_ok=True)
    members = [(r, s) for s in SEEDS for r in range(256)]
    rows, examples = [], {}
    for name, overrides in CONFIGS.items():
        cfg = E.Config(name=name, **overrides)
        result, extra = E.run_members(cfg, members, log=lambda *_: None, keep_params=True)
        exact = np.array([r["exact_final"] for r in result])
        params = {k: v[exact] for k, v in extra["params"].items()}
        pre, act, _ = responses(params, cfg)
        fire = pre > 0
        set_size = fire.sum(axis=1)                         # (M_exact, H)
        used = np.abs(params["wo"]) * np.abs(act).max(axis=1) > 1e-3 * np.abs(
            params["wo"] * np.abs(act).max(axis=1)).max(axis=1, keepdims=True)
        ranks = np.array([np.linalg.matrix_rank(a, tol=1e-4 * max(1e-12, np.abs(a).max()))
                          for a in act])
        hist = np.bincount(set_size.ravel(), minlength=9) / set_size.size
        rows.append({"config": name, "exact_networks": int(exact.sum()),
                     "template_recovered": int((fire.sum(axis=1) == 1).all(axis=1).sum()),
                     "units_firing_on_1": hist[1], "units_firing_on_0": hist[0],
                     "units_firing_on_2_to_6": hist[2:7].sum(), "units_firing_on_7_or_8":
                     hist[7:].sum(), "mean_firing_set": set_size.mean(),
                     "median_rank": float(np.median(ranks)), "rank_8": (ranks == 8).mean(),
                     "mean_units_used": used.sum(axis=1).mean()})
        rules = np.array([r["rule"] for r in result])[exact]
        pick = [i for i in range(len(rules)) if rules[i] == 110][:3]
        examples[name] = [(act[i], cfg) for i in pick]
    write_table(rows, OUT / "summary", list(rows[0]))

    # figure: analytic template vs trained response matrices for rule 110
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    pats = E.PATTERNS
    analytic = np.maximum(0, pats @ W.detector_kernel()[:, 0, :] + W.detector_bias())
    panels = [("analytic template", analytic)] + [
        (f"{name}\nseed {k}", a) for name, ex in examples.items() for k, (a, _) in enumerate(ex[:2])]
    fig, axes = plt.subplots(1, len(panels), figsize=(1.9 * len(panels), 2.6),
                             facecolor=SURFACE)
    labels = [f"{i:03b}" for i in range(8)]
    for ax, (title, a) in zip(axes, panels):
        order = np.lexsort((-a.max(axis=0), a.argmax(axis=0)))  # sort units by preferred pattern
        m = a[:, order] / max(1e-12, a.max())
        ax.imshow(m, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=7, color=INK)
        ax.set_xticks([])
        ax.set_yticks(range(8), labels, fontsize=6, color=INK2)
        ax.set_xlabel("units", fontsize=7, color=INK2)
        for side in ax.spines.values():
            side.set_visible(False)
    fig.suptitle("Detector responses to the 8 neighbourhoods (rule 110; activation / max)",
                 fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_response_matrices.png", dpi=150, facecolor=SURFACE)
    print((OUT / "summary.csv").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
