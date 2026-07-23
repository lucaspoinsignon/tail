"""Multi-portfolio backtest over saved generated windows (no retraining).

Motivation: a single equal-weight portfolio is a weak functional of the
copula -- a model with correct marginals but wrong tail dependence can pass
it.  Backtesting a SET of portfolios, including pairwise
long-short spreads, probes the joint tail structure directly: a spread
portfolio's loss tail depends on exactly one pairwise joint distribution.

Reuses gen_<model>.npy from a run_baselines.py run, evaluates empirical
h-step VaR per portfolio per model, and backtests against the held-out period.

    python backtest_portfolios.py --data returns.csv --n 24 --gen-dir baseline_out

Notes on interpretation printed with the results: all portfolios share the
same held-out days, so exceedances are correlated ACROSS portfolios; the
rejection fraction is a descriptive summary, not a family-wise test.
Empirical (not GPD-refined) VaR is used: with 50k generated windows the
0.5% tail still rests on ~250 order statistics, and it avoids mixing
estimators across hundreds of (model, portfolio) pairs.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataio import load_returns, feature_names_from_csv
from evaluation import kupiec_test, make_windows, portfolio_losses


def portfolio_set(f: int, names: list[str], n_long: int, n_ls: int,
                  seed: int) -> tuple[np.ndarray, list[str]]:
    """Equal weight, all pairwise spreads, random long-only (Dirichlet),
    random long-short (uniform, ||w||_1 = 1)."""
    rng = np.random.default_rng(seed)
    ws, labels = [np.full(f, 1.0 / f)], ["equal"]
    for i, j in itertools.combinations(range(f), 2):
        w = np.zeros(f)
        w[i], w[j] = 0.5, -0.5
        ws.append(w)
        labels.append(f"spread({names[i]}-{names[j]})")
    for k in range(n_long):
        ws.append(rng.dirichlet(np.ones(f)))
        labels.append(f"long{k}")
    for k in range(n_ls):
        w = rng.uniform(-1.0, 1.0, f)
        ws.append(w / np.abs(w).sum())
        labels.append(f"ls{k}")
    return np.array(ws), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--prices", action="store_true")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen-dir", type=str, default="baseline_out",
                    help="directory holding gen_<model>.npy files")
    ap.add_argument("--n-long", type=int, default=25,
                    help="# random long-only portfolios")
    ap.add_argument("--n-ls", type=int, default=25,
                    help="# random long-short portfolios")
    args = ap.parse_args()

    alphas = (0.95, 0.99, 0.995)
    r = load_returns(args.data, args.prices)
    T, f = r.shape
    names = feature_names_from_csv(args.data, f)
    split = int((1.0 - args.test_frac) * T)
    test_w = make_windows(r[split:], args.horizon, stride=args.horizon)
    N_test = test_w.shape[0]

    gens: dict[str, np.ndarray] = {}
    for path in sorted(glob.glob(f"{args.gen_dir}/gen_*.npy")):
        base = os.path.basename(path)[4:-4]
        if base.endswith("_evt"):
            continue                # recalibrated variants excluded by design
        gens[base] = np.load(path)
    assert gens, f"no gen_*.npy files found in {args.gen_dir}"
    for name, g in gens.items():
        assert g.shape[1] >= args.horizon and g.shape[2] == f, \
            f"{name}: window shape {g.shape[1:]} incompatible with horizon/f"

    W, labels = portfolio_set(f, names, args.n_long, args.n_ls, args.seed)
    P = W.shape[0]
    print(f"data: T={T}, f={f} ({', '.join(names)}) | held-out {N_test} "
          f"non-overlapping h={args.horizon} losses | {P} portfolios "
          f"(1 equal, {f*(f-1)//2} spreads, {args.n_long} long, {args.n_ls} "
          f"long-short)\nmodels: {', '.join(gens)}\n")

    # exceedances[model][alpha] -> array over portfolios; pvals likewise
    results = {m: {a: {"exc": np.zeros(P, int), "p": np.zeros(P)}
                   for a in alphas} for m in gens}
    L_test_all = np.stack([portfolio_losses(test_w, weights=w,
                                            horizon=args.horizon) for w in W])
    for m, g in gens.items():
        for pi, w in enumerate(W):
            Lg = portfolio_losses(g, weights=w, horizon=args.horizon)
            for a in alphas:
                var = float(np.quantile(Lg, a))
                k = kupiec_test(L_test_all[pi], var, a)
                results[m][a]["exc"][pi] = k["exceedances"]
                results[m][a]["p"][pi] = k["p_value"]

    # ------------------------------------------------------------- summaries
    print("=" * 78)
    print("MULTI-PORTFOLIO BACKTEST -- empirical h-step VaR from generated "
          "windows\nper cell: mean exceedance rate %% (nominal %%) | fraction "
          "of portfolios rejected (Kupiec p<0.05)")
    print("=" * 78)
    for m in gens:
        row = f"{m:>12s}"
        for a in alphas:
            exc = results[m][a]["exc"]
            rate = 100.0 * exc.mean() / N_test
            rej = float((results[m][a]["p"] < 0.05).mean())
            row += f" | {rate:5.2f}% ({100*(1-a):4.1f}%) rej {rej:4.0%}"
        print(row)
    print("(portfolios share test days -> exceedances correlated across "
          "portfolios;\n rejection fraction is descriptive, not family-wise)")

    # spreads only: the direct probe of pairwise joint tails
    spread_idx = [i for i, l in enumerate(labels) if l.startswith("spread")]
    print("\nPAIRWISE SPREAD PORTFOLIOS (long one asset, short another) -- "
          "exceedances / expected at each alpha")
    header = f"{'model':>12s}"
    for i in spread_idx:
        header += f" | {labels[i]:>18s}"
    print(header + "   (cells: exc@95/99/99.5, expected "
          f"{[round(N_test*(1-a),1) for a in alphas]})")
    for m in gens:
        row = f"{m:>12s}"
        for i in spread_idx:
            cell = "/".join(str(results[m][a]["exc"][i]) for a in alphas)
            row += f" | {cell:>18s}"
        print(row)

    # worst portfolio per model at 95%
    print("\nWORST PORTFOLIO per model (largest |exc - expected| at a=0.95):")
    a = 0.95
    for m in gens:
        exc = results[m][a]["exc"]
        i = int(np.argmax(np.abs(exc - N_test * (1 - a))))
        print(f"  {m:>12s}: {labels[i]:<22s} exc {exc[i]}/{N_test*(1-a):.1f} "
              f"p={results[m][a]['p'][i]:.3f}")


if __name__ == "__main__":
    main()
