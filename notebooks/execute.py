"""Execute the notebooks in place and fail if any cell (or assertion) fails.

    python notebooks/execute.py

Needs the ``notebook`` extra (``pip install -e .[notebook]``); without it the
script says so and exits with code 0, so ``reproduce.py`` can skip it.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOKS = ["walkthrough.ipynb"]


def main() -> int:
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError:
        print("nbformat/nbclient not installed (pip install -e .[notebook]); skipping the notebooks")
        return 0
    for name in NOTEBOOKS:
        path = HERE / name
        nb = nbformat.read(path, as_version=4)
        NotebookClient(nb, timeout=600, kernel_name="python3",
                       resources={"metadata": {"path": str(HERE)}}).execute()
        nbformat.write(nb, path)
        print(f"executed {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
