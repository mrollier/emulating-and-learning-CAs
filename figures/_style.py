"""Settings shared by the figure scripts.

The published figures were rendered with LaTeX text. The scripts do the same
when a ``latex`` executable is found, and fall back to Matplotlib's mathtext
otherwise (``--no-usetex`` forces the fallback). Output goes to ``output/``
under the stems used in the thesis source.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "output"


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--no-usetex", action="store_true", help="use mathtext instead of LaTeX")
    p.add_argument("--png", action="store_true", help="also write a PNG next to the PDF")
    p.add_argument("--out", type=Path, default=OUTPUT, help="output folder (default: output/)")
    return p


def apply(args) -> bool:
    """Apply the style; returns whether LaTeX is used."""
    usetex = not args.no_usetex and shutil.which("latex") is not None
    matplotlib.rcParams["text.usetex"] = usetex
    return usetex


def save(fig, stem: str, args) -> Path:
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{stem}.pdf"
    fig.savefig(path, bbox_inches="tight")
    if args.png:
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    return path


def load_npz(path: Path) -> dict:
    with __import__("numpy").load(path) as f:
        return {k: f[k] for k in f.files}
