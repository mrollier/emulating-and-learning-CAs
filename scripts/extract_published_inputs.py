"""Decode the inputs of the published figures from their PDF files.

Figs 1, 2 and 4 of the ACRI 2024 paper (thesis Figs 7.1, 7.2 and 7.4) were drawn
from unseeded random draws. Their inputs survive only as the raster images that
Matplotlib embedded in the figure PDFs. This script reads those rasters,
samples every cell at its centre, and stores the decoded states, rules and
allocations in ``data/published_inputs/``, so that ``figures/`` and
``verification/test_c5_published_figures.py`` can redraw and check the
published figures exactly. Fig. 3 (thesis Fig. 7.3) is decoded too: its example
input and target, for use in the seeded re-run of the training.

Every decoding is verified before anything is written: evolving the decoded
initial row with CellPyLib, under the decoded rules and allocation, must give
back the published diagram cell for cell.

Run once, from the repository root (needs ``pip install -e .[dev]``)::

    python scripts/extract_published_inputs.py \
        --source-dir "../../Submissions/arXiv/nuca-simulation/figures"

The source PDFs are the ones included by the LaTeX source of the paper; their
sha256 hashes are written to ``data/published_inputs/SOURCES.txt``.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pymupdf

from ca_emulators.reference import evolve_cellpylib, neighbourhood_index

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT.parent.parent / "Submissions" / "arXiv" / "nuca-simulation" / "figures"

FIG1 = "acri-example-rules30_90_vertical.pdf"
FIG2 = "eca-N32-rule54-decomposition.pdf"
FIG3 = "plot_configs_ECA_32cells_rule54_40epochs_bs64_lr0p005.pdf"
FIG4 = "cellpylib-spacetime_diagram-Nrules8.pdf"

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (30, 100, 200)  # colour of rule 90 in Fig. 1
# Matplotlib's Set2 colour map (8 colours), used for the allocation of Fig. 4
SET2 = [(102, 194, 165), (252, 141, 98), (141, 160, 203), (231, 138, 195),
        (166, 216, 84), (255, 217, 47), (229, 196, 148), (179, 179, 179)]


def raster(pdf: Path, xref: int) -> np.ndarray:
    """Return the embedded image with this xref as an (H, W, 3) uint8 array."""
    doc = pymupdf.open(pdf)
    pix = pymupdf.Pixmap(doc, xref)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]


def images(pdf: Path) -> list[tuple[int, tuple[int, int]]]:
    """(xref, (width, height)) of the images on page 1, in drawing order."""
    doc = pymupdf.open(pdf)
    out = []
    for info in doc[0].get_image_info(xrefs=True):
        pix = pymupdf.Pixmap(doc, info["xref"])
        out.append((info["xref"], (pix.width, pix.height)))
    return out


def cells(img: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """Sample an upsampled imshow raster at the centre of every data cell.

    Also checks that each cell is drawn as a uniform block, so that the
    sampling cannot straddle a cell boundary.
    """
    h, w = img.shape[:2]
    rows = ((np.arange(n_rows) + 0.5) * h / n_rows).astype(int)
    cols = ((np.arange(n_cols) + 0.5) * w / n_cols).astype(int)
    out = img[np.ix_(rows, cols)]
    # uniformity: the pixel a quarter cell away in each direction agrees
    for dr, dc in [(-0.25, 0), (0.25, 0), (0, -0.25), (0, 0.25)]:
        r2 = np.clip(((np.arange(n_rows) + 0.5 + dr) * h / n_rows).astype(int), 0, h - 1)
        c2 = np.clip(((np.arange(n_cols) + 0.5 + dc) * w / n_cols).astype(int), 0, w - 1)
        if not np.array_equal(out, img[np.ix_(r2, c2)]):
            raise AssertionError("raster cells are not uniform blocks; check the grid size")
    return out


def classify(rgb: np.ndarray, palette: list[tuple[int, int, int]]) -> np.ndarray:
    """Map each RGB triple to the index of the exactly matching palette colour."""
    flat = rgb.reshape(-1, 3)
    idx = np.full(len(flat), -1)
    for i, colour in enumerate(palette):
        idx[np.all(flat == colour, axis=1)] = i
    if np.any(idx < 0):
        unknown = np.unique(flat[idx < 0], axis=0)
        raise AssertionError(f"colours outside the palette: {unknown[:5]}")
    return idx.reshape(rgb.shape[:-1])


def cellpylib_diagram(x0: np.ndarray, rules, alloc: np.ndarray) -> np.ndarray:
    """CellPyLib diagram with as many rows as ``alloc`` (row t - 1 governs update t)."""
    n_updates = len(alloc) - 1
    return evolve_cellpylib(x0, rules, n_updates, alloc[:n_updates])


def eca_update(x: np.ndarray, rule: int) -> np.ndarray:
    return evolve_cellpylib(x, rule, 1)[1]


def fig1(pdf: Path) -> dict[str, np.ndarray]:
    (xa, _), (xs, _) = images(pdf)
    alloc = classify(cells(raster(pdf, xa), 32, 32), [WHITE, BLUE])
    diagram = classify(cells(raster(pdf, xs), 32, 32), [WHITE, BLACK])
    rules = np.array([30, 90])
    assert np.all(alloc == alloc[0]), "Fig. 1 allocation is not uniform in time"
    assert np.array_equal(cellpylib_diagram(diagram[0], rules, alloc), diagram)
    return {"rules": rules, "alloc": alloc[0], "diagram": diagram}


def fig2(pdf: Path) -> dict[str, np.ndarray]:
    imgs = images(pdf)
    strips = [x for x, (w, h) in imgs if h < 20]  # input, integer encoding, output
    x_in = classify(cells(raster(pdf, strips[0]), 1, 32), [WHITE, BLACK])[0]
    x_out = classify(cells(raster(pdf, strips[-1]), 1, 32), [WHITE, BLACK])[0]
    one_hot_xref = [x for x, (w, h) in imgs if w > 300 and h > 50][0]
    one_hot = classify(cells(raster(pdf, one_hot_xref), 8, 32), [WHITE, BLACK])
    index = neighbourhood_index(x_in)
    assert np.array_equal(one_hot, (np.arange(8)[:, None] == index[None, :]).astype(int))
    assert np.array_equal(eca_update(x_in, 54), x_out)
    return {"rule": np.array(54), "x": x_in, "y": x_out}


def fig3(pdf: Path) -> dict[str, np.ndarray]:
    strips = [x for x, (w, h) in images(pdf) if w > 300 and h > 8]  # input, final, target
    x_in = classify(cells(raster(pdf, strips[0]), 1, 32), [WHITE, BLACK])[0]
    target = classify(cells(raster(pdf, strips[-1]), 1, 32), [WHITE, BLACK])[0]
    assert np.array_equal(eca_update(x_in, 54), target)
    return {"rule": np.array(54), "x": x_in, "y": target}


def fig4(pdf: Path) -> dict[str, np.ndarray]:
    (xa, _), (xs, _) = images(pdf)
    alloc = classify(cells(raster(pdf, xa), 32, 32), SET2)
    diagram = classify(cells(raster(pdf, xs), 32, 32), [WHITE, BLACK])
    text = re.sub(r"\d+\s*rules", "", pymupdf.open(pdf)[0].get_text())  # drop the title's "8 rules"
    numbers = [int(t) for t in re.findall(r"\b\d+\b", text)]
    rules = np.array([n for n in numbers if n > 1])  # tick labels 0 and 1 are states
    assert len(rules) == 8 and np.all(np.diff(rules) > 0), f"unexpected rule labels {rules}"
    for t in range(32):  # the allocation is shifted by one cell per time step
        assert np.array_equal(alloc[t], np.roll(alloc[0], -t)), "Fig. 4 allocation is not a shift"
    assert np.array_equal(cellpylib_diagram(diagram[0], rules, alloc), diagram)
    return {"rules": rules, "alloc": alloc, "diagram": diagram}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "published_inputs")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = [("fig1_nuca_rules30_90", FIG1, fig1), ("fig2_decomposition_rule54", FIG2, fig2),
            ("fig3_training_example_rule54", FIG3, fig3), ("fig4_nuca_cellpylib_8rules", FIG4, fig4)]
    lines = ["Source PDFs of data/published_inputs (scripts/extract_published_inputs.py)", ""]
    for stem, name, decode in jobs:
        pdf = args.source_dir / name
        data = decode(pdf)
        np.savez_compressed(args.out / f"{stem}.npz", **{k: np.asarray(v) for k, v in data.items()})
        digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
        lines.append(f"{stem}.npz  <-  {name}  sha256 {digest}")
        print(f"{stem}: decoded and verified against the published diagram")
    (args.out / "SOURCES.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
