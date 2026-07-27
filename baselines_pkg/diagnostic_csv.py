"""Locate the values that break np.log() in fit_returns.py / run_baselines.py.

Usage:  python diagnose_csv.py timeseries.csv
"""
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "timeseries.csv"


def is_float(tok):
    try:
        float(tok)
        return True
    except ValueError:
        return False


with open(path) as fh:
    rows = [ln.rstrip("\n").split(",") for ln in fh if ln.strip()]

header = None
if not all(is_float(t) for t in rows[0]):
    header, rows = rows[0], rows[1:]

ncol = len(rows[0])
keep = [j for j in range(ncol) if is_float(rows[0][j])]
dropped = [j for j in range(ncol) if j not in keep]
names = header if header else [f"col{j}" for j in range(ncol)]
date_col = dropped[0] if dropped else None

print(f"{path}: {len(rows)} data rows, {ncol} columns")
print(f"  used as series : {[names[j] for j in keep]}")
print(f"  dropped        : {[names[j] for j in dropped] or 'none'}")
if date_col is not None:
    print(f"  date column    : {names[date_col]}")
print()

ragged = [i for i, r in enumerate(rows) if len(r) != ncol]
if ragged:
    print(f"!! {len(ragged)} ragged rows, first at data row {ragged[0]}\n")

bad = []
for i, r in enumerate(rows):
    when = r[date_col] if date_col is not None and date_col < len(r) else f"row {i}"
    for j in keep:
        tok = r[j] if j < len(r) else ""
        try:
            v = float(tok)
        except ValueError:
            bad.append((when, i, names[j], tok, "unparseable"))
            continue
        if np.isnan(v):
            bad.append((when, i, names[j], tok, "NaN"))
        elif np.isinf(v):
            bad.append((when, i, names[j], tok, "inf"))
        elif v == 0.0:
            bad.append((when, i, names[j], tok, "exactly zero"))
        elif v < 0.0:
            bad.append((when, i, names[j], tok, "negative"))

if not bad:
    print("No non-positive / non-finite values. The log warning is not coming from here.")
    print("Check that you are not passing --prices on a file that already holds returns.")
else:
    print(f"{len(bad)} problem cells (log needs strictly positive prices):\n")
    print(f"  {'date':<14}{'row':>7}  {'column':<14}{'value':<16}issue")
    for when, i, col, tok, why in bad[:40]:
        print(f"  {str(when)[:13]:<14}{i:>7}  {col:<14}{repr(tok):<16}{why}")
    if len(bad) > 40:
        print(f"  ... and {len(bad) - 40} more")

    cols = sorted({b[2] for b in bad})
    print(f"\n  affected columns: {cols}")
    print("  If any of these is a volume/count/indicator column, drop it before fitting.")
