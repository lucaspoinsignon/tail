"""Train the baseline generators (TimeVAE, TimeGAN, Tail-GAN) on a return
series and evaluate their tail behavior and risk calibration.

Usage:
    python run_baselines.py --data returns.csv --n 24 --gen 50000
    python run_baselines.py --data returns_prices.csv --prices --models timevae,tailgan
    python run_baselines.py --data returns.csv --quick            # CPU smoke test

Pipeline per baseline: temporal train/test split -> min-max/max-abs scaling
(each reference implementation's own convention, applied inside the baseline)
-> training -> sampling -> tail diagnostics (evaluation.print_report),
portfolio VaR/CVaR with bootstrap CIs (evaluation.estimate_risk) and a Kupiec
backtest on the held-out period -> comparison table + figures
(baseline_diagnostics.png, empirical_distributions.png).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataio import load_returns, feature_names_from_csv
from evaluation import (estimate_risk, kupiec_test, portfolio_losses,
                        make_windows, print_report, tail_dependence_report)
from baselines import BASELINES


def main():
    ap = argparse.ArgumentParser()
    # ---- data arguments: keep identical to fit_returns.py -------------------
    ap.add_argument("--data", required=True, help="CSV or .npy of shape (T, f)")
    ap.add_argument("--prices", action="store_true",
                    help="input is prices, not returns")
    ap.add_argument("--n", type=int, default=24, help="window length")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--gen", type=int, default=50_000, help="# generated windows")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--weights", type=str, default=None,
                    help="comma-separated portfolio weights (default: equal)")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="baseline_out")
    # ---- model selection ----------------------------------------------------
    ap.add_argument("--models", type=str, default="timevae,timegan,tailgan",
                    help="comma-separated subset of: timevae,timegan,tailgan")
    ap.add_argument("--quick", action="store_true",
                    help="tiny training budgets for a CPU smoke test")
    ap.add_argument("--reuse", action="store_true",
                    help="for each requested model, load {outdir}/gen_<model>.npy "
                         "if it exists instead of retraining (recover from a "
                         "crashed evaluation, or re-run the evaluation and "
                         "figures without paying for training again)")
    # ---- per-model budgets (reference defaults; see baselines/*.py) ---------
    ap.add_argument("--timevae-epochs", type=int, default=1000)
    ap.add_argument("--timevae-batch", type=int, default=16)
    ap.add_argument("--timevae-recon-wt", type=float, default=3.0,
                    help="reference default 3.0; raise (e.g. 100) to counteract "
                         "posterior collapse on near-i.i.d. return windows")
    ap.add_argument("--timevae-latent", type=int, default=8)
    ap.add_argument("--timegan-iters", type=int, default=10_000,
                    help="iterations PER PHASE (reference default 50000)")
    ap.add_argument("--timegan-batch", type=int, default=128)
    ap.add_argument("--tailgan-epochs", type=int, default=3000)
    ap.add_argument("--tailgan-batch", type=int, default=1000)
    ap.add_argument("--tailgan-lr-g", type=float, default=1e-6)
    ap.add_argument("--tailgan-lr-d", type=float, default=1e-7)
    ap.add_argument("--tailgan-alphas", type=str, default="0.05",
                    help="comma-separated PnL tail levels the score targets "
                         "(reference default 0.05). To align training with the "
                         "evaluation levels use 0.05,0.01,0.005; note the "
                         "alpha=0.005 tail has only ~batch_size*0.005 order "
                         "statistics per batch, so keep the batch large.")
    args = ap.parse_args()

    if args.quick:
        args.timevae_epochs = 30
        args.timegan_iters = 100
        args.tailgan_epochs = 30
        args.tailgan_batch = 128
        args.gen = min(args.gen, 2048)

    # Fail fast: validate paths BEFORE spending compute on training.
    if not os.path.exists(args.data):
        ap.error(f"--data file not found: {args.data}")

    os.makedirs(args.outdir, exist_ok=True)
    alphas = (0.95, 0.99, 0.995)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in BASELINES]
    assert not unknown, f"unknown model(s) {unknown}; choose from {list(BASELINES)}"

    # --------------------------------------------------------------- data
    r = load_returns(args.data, args.prices)
    T, f = r.shape
    names = feature_names_from_csv(args.data, f)
    split = int((1.0 - args.test_frac) * T)
    train_r, test_r = r[:split], r[split:]
    real = make_windows(train_r, args.n, args.stride)
    print(f"data: T={T}, f={f} ({', '.join(names)}) | "
          f"train windows {real.shape} | test T={test_r.shape[0]}")

    w = (np.array([float(v) for v in args.weights.split(",")])
         if args.weights else np.full(f, 1.0 / f))
    assert w.size == f, "--weights length must equal the number of features"
    L_test = portfolio_losses(make_windows(test_r, args.horizon,
                                           stride=args.horizon),
                              weights=w, horizon=args.horizon)

    hparams = {
        "timevae": dict(max_epochs=args.timevae_epochs,
                        batch_size=args.timevae_batch,
                        reconstruction_wt=args.timevae_recon_wt,
                        latent_dim=args.timevae_latent),
        "timegan": dict(iterations=args.timegan_iters,
                        batch_size=args.timegan_batch),
        "tailgan": dict(n_epochs=args.tailgan_epochs,
                        batch_size=args.tailgan_batch,
                        lr_G=args.tailgan_lr_g, lr_D=args.tailgan_lr_d,
                        alphas=tuple(float(a) for a in
                                     args.tailgan_alphas.split(","))),
    }

    # ------------------------------------------------ train, generate, score
    gens: dict[str, np.ndarray] = {}
    for name in models:
        cache = f"{args.outdir}/gen_{name}.npy"
        if args.reuse and os.path.exists(cache):
            gen = np.load(cache)
            assert gen.shape[1:] == (args.n, f), \
                f"cached {cache} has window shape {gen.shape[1:]}, expected " \
                f"({args.n}, {f}); delete it or drop --reuse to retrain"
            print(f"\n################ {name} (reusing {cache}, "
                  f"{gen.shape[0]} windows) ################")
        else:
            print(f"\n################ {name} ################")
            gen = BASELINES[name](real, args.gen, seed=args.seed,
                                  device=args.device, **hparams[name])
            np.save(cache, gen)
        gens[name] = gen

    summary: dict[str, dict] = {}
    for name, gen in gens.items():
        print(f"\n=== Diagnostics: {name} (vs real train windows) ===")
        print_report(real, gen, feature_names=names)
        report = estimate_risk(gen, alphas=alphas, weights=w,
                               horizon=args.horizon, n_boot=200, seed=args.seed)
        summary[name] = {}
        print(f"\n=== {name}: portfolio risk (h={args.horizon}) and Kupiec "
              f"backtest (held-out N={L_test.size}) ===")
        for a in alphas:
            rp = report[a]
            k = kupiec_test(L_test, rp["var_gpd"], a)
            summary[name][a] = (rp["var_gpd"], rp["cvar_gpd"],
                                k["exceedances"], k["expected"], k["p_value"])
            print(f"a={a:5.3f}: VaR {rp['var_gpd']:.5f} "
                  f"[{rp['var_ci'][0]:.5f},{rp['var_ci'][1]:.5f}]  "
                  f"CVaR {rp['cvar_gpd']:.5f} "
                  f"[{rp['cvar_ci'][0]:.5f},{rp['cvar_ci'][1]:.5f}]  | "
                  f"exceed {k['exceedances']}/{k['expected']:.1f}  "
                  f"p={k['p_value']:.3f}")

    # ------------------------------------------------------ comparison table
    print("\n" + "=" * 78)
    print("MODEL COMPARISON -- portfolio VaR/CVaR (GPD-refined) and Kupiec "
          "p-value on held-out data")
    print("=" * 78)
    header = f"{'model':>10s}" + "".join(
        f" | a={a:.3f}: VaR    CVaR    exc    p " for a in alphas)
    print(header)
    for name, per_a in summary.items():
        row = f"{name:>10s}"
        for a in alphas:
            v, c, exc, expd, p = per_a[a]
            row += f" | {v:7.4f} {c:7.4f} {exc:3d}/{expd:4.1f} {p:5.3f}"
        print(row)
    print("(Kupiec: p > 0.05 means the VaR level is not rejected; exc/exp = "
          "observed vs expected exceedances)")

    # ------------------------------------------------------------------ figure
    colors = {m: f"C{i}" for i, m in enumerate(gens)}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    R = real.reshape(-1, f)
    ql = np.linspace(0.001, 0.05, 150)
    rq = np.quantile(R, ql)
    for name, gen in gens.items():
        G = gen.reshape(-1, f)
        axes[0].plot(rq, np.quantile(G, ql), ".", ms=3, color=colors[name],
                     label=name)
    lim = [min(rq.min(), *[np.quantile(g, 0.001) for g in gens.values()]), 0]
    axes[0].plot(lim, lim, "k--", lw=1)
    axes[0].set_title("Lower-tail QQ, pooled features (0.1%-5%)")
    axes[0].set_xlabel("real quantile"); axes[0].set_ylabel("generated quantile")
    axes[0].legend(fontsize=8)

    q_grid = np.linspace(0.005, 0.10, 20)
    if f >= 2:
        td_real = tail_dependence_report(real, real, q_grid)[(0, 1)][0]
        axes[1].plot(q_grid, td_real, "k-", lw=2, label="real")
        for name, gen in gens.items():
            td = tail_dependence_report(real, gen, q_grid)[(0, 1)][1]
            axes[1].plot(q_grid, td, "--", color=colors[name], label=name)
        axes[1].set_ylim(0, 1); axes[1].legend(fontsize=7)
        axes[1].set_title(rf"$\hat\lambda_L(q)$ pair ({names[0]},{names[1]})")
        axes[1].set_xlabel("q")

    L_real = np.sort(portfolio_losses(real, weights=w, horizon=args.horizon))
    axes[2].semilogy(L_real, 1 - np.arange(1, L_real.size + 1) / (L_real.size + 1),
                     "k", lw=2, label="real (train)")
    for name, gen in gens.items():
        Lg = np.sort(portfolio_losses(gen, weights=w, horizon=args.horizon))
        axes[2].semilogy(Lg, 1 - np.arange(1, Lg.size + 1) / (Lg.size + 1),
                         color=colors[name], label=name)
    axes[2].set_title(f"Portfolio loss survival (h={args.horizon})")
    axes[2].set_xlabel("loss"); axes[2].set_ylabel("P(L > l)")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{args.outdir}/baseline_diagnostics.png", dpi=140)

    # ------------------- empirical distributions: bulk + tails per feature ---
    fig2, axes2 = plt.subplots(2, f, figsize=(5.4 * f, 8.2), squeeze=False)
    Gflat = {name: gen.reshape(-1, f) for name, gen in gens.items()}
    for j in range(f):
        # row 1: pooled 1-step empirical density on a log scale (both tails)
        ax = axes2[0][j]
        lo = min(R[:, j].min(), *[G[:, j].min() for G in Gflat.values()])
        hi = max(R[:, j].max(), *[G[:, j].max() for G in Gflat.values()])
        bins = np.linspace(lo, hi, 120)
        ax.hist(R[:, j], bins=bins, density=True, histtype="step",
                color="k", lw=2, label="real")
        for name, G in Gflat.items():
            ax.hist(G[:, j], bins=bins, density=True, histtype="step",
                    color=colors[name], label=name)
        ax.set_yscale("log")
        ax.set_title(f"{names[j]}: empirical density (log scale)")
        ax.set_xlabel("1-step return")
        if j == 0:
            ax.set_ylabel("density")
        ax.legend(fontsize=7)
        # row 2: 1-step loss survival P(-X > x), the lower tail head-on
        ax = axes2[1][j]
        Lr = np.sort(-R[:, j])
        ax.semilogy(Lr, 1 - np.arange(1, Lr.size + 1) / (Lr.size + 1),
                    "k", lw=2, label="real")
        xmax = Lr[-1]
        for name, G in Gflat.items():
            Lg = np.sort(-G[:, j])
            ax.semilogy(Lg, 1 - np.arange(1, Lg.size + 1) / (Lg.size + 1),
                        color=colors[name], label=name)
            xmax = max(xmax, Lg[-1])
        ax.set_xlim(0, 1.02 * xmax)
        ax.set_ylim(1e-5, 1)
        ax.set_title(f"{names[j]}: loss survival P(-X > x)")
        ax.set_xlabel("loss x")
        if j == 0:
            ax.set_ylabel("P(-X > x)")
        ax.legend(fontsize=7)
    fig2.tight_layout()
    fig2.savefig(f"{args.outdir}/empirical_distributions.png", dpi=140)
    print(f"\nSaved: {args.outdir}/{{gen_<model>.npy, baseline_diagnostics.png, "
          f"empirical_distributions.png}}")


if __name__ == "__main__":
    main()
