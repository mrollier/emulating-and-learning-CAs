"""The drawing code of the published figures (ACRI 2024 Figs 1-5, thesis 7.1-7.5).

Each function reproduces the layout of the 2024 script it replaces. Nothing
here changes Matplotlib's global settings; ``figures/_style.py`` switches on
LaTeX rendering when it is available, as in the published figures.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .reference import neighbourhood_index
from .rules import rule_table

RULE_90_BLUE = (30 / 255, 100 / 255, 200 / 255)


def plot_nuca_example(alloc, diagram, rules, labelsize: int = 14):
    """Fig. 1: rule allocation (white/blue) next to the state evolution.

    ``alloc`` is the (T, N) allocation as drawn (row t governs update t + 1),
    ``diagram`` the (T, N) spacetime diagram, ``rules`` the two rules.
    """
    fig, axs = plt.subplots(1, 2, figsize=(5, 3))
    caxs = [make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1) for ax in axs]
    ra = axs[0].imshow(alloc, cmap=ListedColormap([(1, 1, 1), RULE_90_BLUE]))
    se = axs[1].imshow(diagram, cmap=plt.get_cmap("Greys", 2))
    cbar_ra = fig.colorbar(ra, ax=axs[0], cax=caxs[0], ticks=[1 / 4, 3 / 4])
    cbar_ra.set_ticklabels([str(r) for r in rules], size=labelsize)
    cbar_se = fig.colorbar(se, ax=axs[1], cax=caxs[1], ticks=[1 / 4, 3 / 4])
    cbar_se.set_ticklabels(["0", "1"], size=labelsize)
    axs[0].set_title("Rule allocation", size=labelsize)
    axs[1].set_title("State evolution", size=labelsize)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(r"$\leftarrow$ Time", size=labelsize)
    fig.subplots_adjust(wspace=0.45)
    return fig, axs


def plot_decomposition(x, rule: int, detector_output=None, output=None, title_size: int = 14):
    """Fig. 2: the (de)composition steps of one ECA update inside the CNN.

    ``detector_output`` (N, 8) and ``output`` (N,) are the activations of the
    detector layer and the network output; if omitted they are computed with
    numpy. The integer strip is the (4, 2, 1) neighbourhood encoding.
    """
    x = np.asarray(x)
    n_cells = len(x)
    index = neighbourhood_index(x)
    one_hot = (np.arange(8)[np.newaxis, :] == index[:, np.newaxis]).astype(float)
    if detector_output is not None:
        if not np.array_equal(np.asarray(detector_output), one_hot):
            raise ValueError("the detector activations are not the one-hot neighbourhoods")
    table = rule_table(rule).astype(float)
    update = table[np.newaxis, :] * one_hot
    if output is None:
        output = update.sum(axis=1)
    fig, axs = plt.subplots(5, 2, height_ratios=[1, 1, 3, 3, 1], width_ratios=[1, n_cells],
                            figsize=(5, 5))
    cmap = "Greys"
    axs[0, 1].set_title("Input configuration", size=title_size)
    axs[0, 1].imshow(x[np.newaxis, :], cmap=cmap)
    axs[1, 1].set_title("Integer neighbourhood encoding", size=title_size)
    axs[1, 1].imshow(index[np.newaxis, :], cmap=cmap)
    axs[2, 0].set_ylabel(f"Rule {rule}", size=title_size + 2)
    axs[2, 0].imshow(table[:, np.newaxis], cmap=cmap)
    axs[2, 1].set_title("One-hot neighbourhood encoding", size=title_size)
    axs[2, 1].imshow(one_hot.T, cmap=cmap)
    axs[3, 1].set_title("Output of neighbourhood after local update", size=title_size)
    axs[3, 1].imshow(update.T, cmap=cmap)
    axs[4, 1].set_title("Output configuration", size=title_size)
    axs[4, 1].imshow(np.asarray(output)[np.newaxis, :], cmap=cmap)
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in (axs[0, 0], axs[1, 0], axs[3, 0], axs[4, 0]):
        ax.remove()
    return fig, axs


def plot_training_history(model, weights_history, x_example, y_example, rule, n_updates: int = 1):
    """Fig. 3: convergence of a trained emulator on one training example.

    For the weights after every epoch, the network's output on ``x_example``
    is compared with the target ``y_example``: the absolute error per cell
    (log-scaled colours) and the mean squared error (right).
    """
    x_example = np.asarray(x_example, dtype=np.float32).reshape(1, -1, 1)
    y_example = np.asarray(y_example, dtype=np.float32).reshape(1, -1)
    fig, axs = plt.subplots(4, 2, figsize=(5, 3), width_ratios=[5, 1], height_ratios=[1, 7, 1, 1])
    for ax in (axs[0, 1], axs[2, 1], axs[3, 1]):
        ax.remove()
    axs[0, 0].imshow(x_example[0, :, 0][np.newaxis, :], cmap="Greys")
    axs[0, 0].set_title("Input configuration")

    predictions, mses = [], []
    for weights in weights_history:
        model.set_weights(weights)
        prediction = model.predict(x_example, verbose=0)[0, :, 0]
        predictions.append(prediction)
        mses.append(np.square(prediction - y_example[0]).mean())
    predictions, mses = np.array(predictions), np.array(mses)
    errors = np.abs(predictions - y_example)

    im = axs[1, 0].imshow(errors, cmap="RdYlGn_r", norm=SymLogNorm(1e-3, vmin=0, vmax=1),
                          interpolation="none")
    cbar = fig.colorbar(im, ax=axs[1, 0], orientation="horizontal", aspect=75)
    cbar.ax.set_xticks([0, 1e-3, 1e-2, 1e-1, 1], minor=False)
    cbar.ax.set_xticklabels([r"$0$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$1$"],
                            fontdict={"fontsize": 8})
    axs[1, 0].set_ylabel(r"$\leftarrow$ Epochs")
    axs[1, 0].set_title(f"CNN model convergence over {len(mses)} epochs")
    axs[1, 0].set_aspect("auto")

    axs[2, 0].imshow(predictions[-1][np.newaxis, :], cmap="Greys")
    axs[2, 0].set_title("CNN model final output")

    axs[1, 1].plot(mses, np.arange(len(mses), 0, -1), color="k", lw=1)
    axs[1, 1].set_xscale("log")
    axs[1, 1].set_title(r"MSE (log)")
    axs[1, 1].set_ylim([-0.4 * len(mses), len(mses) + 1])
    axs[1, 1].set_axis_off()

    axs[3, 0].imshow(y_example, cmap="Greys")
    axs[3, 0].set_title(f"Desired output, rule(s) {rule}, {n_updates} timestep(s)")
    for ax in (axs[0, 0], axs[1, 0], axs[2, 0], axs[3, 0], axs[1, 1]):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig, axs


def plot_nuca_cellpylib(alloc, diagram, rules, labelsize: int = 14):
    """Fig. 4: a nuCA's (time-varying) rule allocation and its spacetime diagram."""
    n_rules = len(rules)
    fig, axs = plt.subplots(1, 2, figsize=(5, 3))
    caxs = [make_axes_locatable(ax).append_axes("bottom", size="5%", pad=0.1) for ax in axs]
    alloc_map = axs[0].imshow(alloc, cmap="Set2", vmin=0, vmax=n_rules - 1)
    cb_alloc = fig.colorbar(alloc_map, ax=axs[0], cax=caxs[0], orientation="horizontal")
    state_map = axs[1].imshow(diagram, cmap=plt.get_cmap("Greys", 2))
    cb_state = fig.colorbar(state_map, cax=caxs[1], orientation="horizontal")
    cb_alloc.set_ticks(np.linspace(0, n_rules - 1, 2 * n_rules + 1)[1::2])
    cb_alloc.set_ticklabels([str(r) for r in rules], size=labelsize - 4, rotation=90)
    cb_state.set_ticks([1 / 4, 3 / 4])
    cb_state.set_ticklabels(["0", "1"], size=labelsize - 4, rotation=90)
    axs[0].set_title(f"Rule allocation\n{n_rules} rules", size=labelsize + 4)
    axs[1].set_title("Spacetime diagram\nof the $\\nu$CA", size=labelsize + 4)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    axs[0].set_ylabel(r"$\leftarrow$ Time", size=labelsize)
    fig.subplots_adjust(wspace=0.1)
    return fig, axs


#: Axis settings of the four panels of Fig. 5, as in the 2024 scripts.
BENCHMARK_PANELS = {
    "Nrules": dict(xlabel="Number of rules $N_R$", xscale=2, jitter=("mul", 1.05),
                   yticks=[0, 2, 4, 6, 8],
                   title="Computation time for {S} $\\nu$CAs of {N} cells and {T} timesteps"),
    "T": dict(xlabel="Number of timesteps $T$", jitter=("add", 1), yticks=[0, 2, 4, 6, 8],
              xlim=[0, 110], title="Computation time for {S} $\\nu$CAs of {N} cells and {Nrules} rules"),
    "N": dict(xlabel="Number of cells $N$", jitter=("add", 1), yticks=[0, 2, 4, 6, 8, 10],
              ylim=[0, 11], xticks=True,
              title="Computation time for {S} $\\nu$CAs with {Nrules} rules over {T} timesteps"),
    "S": dict(xlabel="Number of samples $S$", xscale=2, yscale="log", jitter=("mul", 1.05),
              xticks=True, ylim=[1e-2, 5e2], yticks=[1e-2, 1e-1, 1e0, 1e1, 1e2],
              title="Computation time of $\\nu$CAs of {N} cells, {Nrules} rules, {T} timesteps"),
}

METHOD_LABELS = ("CellPyLib", "CNN (locally connected)", "CNN (densely connected)")


def plot_benchmark(scenario: str, x, times, params: dict, overlay=None, labelsize: int = 14,
                   overlay_label: str = "re-run (grey: all three methods)"):
    """One panel of Fig. 5: mean and standard deviation over repeats.

    ``times`` holds three (repeats, n) arrays (CellPyLib, locally connected,
    dense). ``params`` fills the title (keys S, N, T, Nrules). ``overlay``,
    optional, holds three more arrays (e.g. a re-run) drawn in grey with the
    same markers.
    """
    panel = BENCHMARK_PANELS[scenario]
    x = np.asarray(x, dtype=float)
    kind, amount = panel["jitter"]
    if kind == "mul":
        xs = [x, x / amount, x * amount]  # as in 2024: CellPyLib centred
    else:
        xs = [x - amount, x, x + amount]
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    for xi, t, fmt, label in zip(xs, times, ("+", "o", "x"), METHOD_LABELS):
        t = np.asarray(t)
        ax.errorbar(xi, t.mean(axis=0), yerr=t.std(axis=0), fmt=fmt, capsize=5, label=label)
    if overlay is not None:
        for k, (xi, t, fmt) in enumerate(zip(xs, overlay, ("+", "o", "x"))):
            t = np.asarray(t)
            ax.errorbar(xi, t.mean(axis=0), yerr=t.std(axis=0), fmt=fmt, capsize=3,
                        color="0.6", alpha=0.8, label=overlay_label if k == 0 else None)
    ax.set_title(panel["title"].format(**params), size=labelsize + 2)
    if "xscale" in panel:
        ax.set_xscale("log", base=panel["xscale"])
    if panel.get("yscale"):
        ax.set_yscale(panel["yscale"])
    ax.legend(ncols=3)
    if panel.get("xticks"):
        ax.set_xticks(x)
    if "xlim" in panel:
        ax.set_xlim(panel["xlim"])
    if "ylim" in panel:
        ax.set_ylim(panel["ylim"])
    ax.set_yticks(panel["yticks"])
    ax.set_xlabel(panel["xlabel"], size=labelsize)
    ax.set_ylabel("Time to compute (s)", size=labelsize)
    ax.tick_params(axis="both", which="major", labelsize=labelsize - 2)
    return fig, ax
