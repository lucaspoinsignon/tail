"""CSV / npy loading for return series (extracted from tailfm's fit_returns.py
so this package is self-contained)."""

from __future__ import annotations

import numpy as np


def load_returns(path: str, prices: bool = False) -> np.ndarray:
    """Load a (T, f) return matrix from .npy or CSV.

    Handles a header row and/or a leading non-numeric column (e.g. Date).
    With prices=True the input is a price series and log-returns are taken.
    """
    if path.endswith(".npy"):
        arr = np.load(path)
    else:
        try:
            arr = np.loadtxt(path, delimiter=",")
        except ValueError:                       # header row and/or Date column
            def _is_float(tok: str) -> bool:
                try:
                    float(tok)
                    return True
                except ValueError:
                    return False
            with open(path) as fh:
                rows = [ln.strip().split(",") for ln in fh if ln.strip()]
            if not all(_is_float(t) for t in rows[0]):
                rows = rows[1:]                  # drop header
            keep = [j for j, t in enumerate(rows[0]) if _is_float(t)]
            arr = np.array([[float(r[j]) for j in keep] for r in rows])
    arr = np.atleast_2d(np.asarray(arr, dtype=float))
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
    if prices:
        arr = np.diff(np.log(arr), axis=0)
    if not np.isfinite(arr).all():
        raise ValueError("Non-finite values in the return matrix; clean the data first.")
    return arr


def feature_names_from_csv(path: str, f: int) -> list[str]:
    """Use the CSV header for labels if one is present."""
    if path.endswith(".npy"):
        return [f"feat{j}" for j in range(f)]
    with open(path) as fh:
        first = fh.readline().strip().split(",")
    try:
        [float(v) for v in first if v]                    # no header
        return [f"feat{j}" for j in range(f)]
    except ValueError:
        names = [c for c in first if c.lower() != "date"]
        return names if len(names) == f else [f"feat{j}" for j in range(f)]
