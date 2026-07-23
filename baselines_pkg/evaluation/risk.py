"""VaR / CVaR estimation from generated scenarios -- the end product of the pipeline.

Loss convention: L = -(portfolio log-return over the horizon). With log-returns, the
h-step portfolio return of window x in R^{n x f} under weights w is approximately
sum_{t=1..h} w^T x_t (exact per-asset, first-order across assets), so

    L(x) = - sum_{t<=h} sum_j w_j x_{t,j}.

Definitions (alpha close to 1):

    VaR_a(L)  = inf { l : P(L <= l) >= a }                       (a-quantile of the loss)
    CVaR_a(L) = E[ L | L >= VaR_a(L) ]
              = min_c { c + E[(L - c)_+] / (1 - a) }             (Rockafellar-Uryasev)

Estimators from M generated windows:
  * empirical: order statistic + tail mean (consistent, but high variance for a -> 1);
  * GPD-refined: fit a GPD to exceedances of the generated losses over their 90% quantile
    and use the closed forms of evt.py -- a variance-reduced extrapolation that lets a
    exceed the empirical resolution 1 - 1/M.

Uncertainty: nonparametric bootstrap over windows. Validation: Kupiec (1995)
unconditional-coverage likelihood-ratio test of the exceedance frequency of held-out
real losses against the model VaR:  x ~ Bin(N, 1-a) under H0,

    LR_uc = -2 [ log( (1-p)^{N-x} p^x ) - log( (1-pi)^{N-x} pi^x ) ] ~ chi^2_1,

with p = 1 - a and pi = x/N. Use non-overlapping horizon blocks for the held-out losses
so the Bernoulli-independence assumption is not violated by construction.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def portfolio_losses(windows: np.ndarray, weights: np.ndarray | None = None,
                     horizon: int | None = None) -> np.ndarray:
    """(M, n, f) windows -> (M,) losses L = -sum_{t<=h} w^T x_t."""
    windows = np.asarray(windows, dtype=float)
    M, n, f = windows.shape
    w = np.full(f, 1.0 / f) if weights is None else np.asarray(weights, dtype=float)
    h = n if horizon is None else min(horizon, n)
    return -(windows[:, :h, :] @ w).sum(axis=1)


def var_cvar_empirical(losses: np.ndarray, alpha: float) -> tuple[float, float]:
    losses = np.asarray(losses, dtype=float)
    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    return float(var), float(tail.mean())


def var_cvar_gpd(losses: np.ndarray, alpha: float,
                 q_thresh: float = 0.90) -> tuple[float, float]:
    """GPD-refined VaR/ES on the loss sample (POT above its q_thresh quantile).

    Falls back to the empirical estimator when the POT fit is not identified:
    fewer than 10 strict exceedances above the threshold (possible for
    degenerate generators whose losses are massively tied, e.g. a collapsed
    VAE) or a numerically failing GPD MLE.
    """
    losses = np.asarray(losses, dtype=float)
    u = np.quantile(losses, q_thresh)
    exc = losses[losses > u] - u
    if exc.size < 10:
        return var_cvar_empirical(losses, alpha)
    try:
        xi, _, beta = stats.genpareto.fit(exc, floc=0.0)
    except (ValueError, RuntimeError):
        return var_cvar_empirical(losses, alpha)
    xi = min(xi, 0.95)  # guard: keep ES finite
    p_u = 1.0 - q_thresh
    ratio = (1.0 - alpha) / p_u
    if abs(xi) > 1e-8:
        var = u + (beta / xi) * (ratio ** (-xi) - 1.0)
    else:
        var = u - beta * np.log(ratio)
    es = var / (1.0 - xi) + (beta - xi * u) / (1.0 - xi)
    return float(var), float(es)


def estimate_risk(gen_windows: np.ndarray, alphas=(0.95, 0.99, 0.995),
                  weights: np.ndarray | None = None, horizon: int | None = None,
                  n_boot: int = 200, seed: int = 0) -> dict:
    """Full risk report from generated windows, with bootstrap percentile CIs."""
    rng = np.random.default_rng(seed)
    L = portfolio_losses(gen_windows, weights, horizon)
    M = L.size
    report = {}
    for a in alphas:
        v_e, c_e = var_cvar_empirical(L, a)
        v_g, c_g = var_cvar_gpd(L, a)
        boot = np.empty((n_boot, 2))
        for b in range(n_boot):
            Lb = L[rng.integers(0, M, M)]
            boot[b] = var_cvar_gpd(Lb, a)
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        report[a] = dict(var_emp=v_e, cvar_emp=c_e, var_gpd=v_g, cvar_gpd=c_g,
                         var_ci=(lo[0], hi[0]), cvar_ci=(lo[1], hi[1]))
    return report


def kupiec_test(real_losses: np.ndarray, var_level: float, alpha: float) -> dict:
    """Unconditional-coverage LR test of P(L > VaR) = 1 - alpha on held-out losses."""
    L = np.asarray(real_losses, dtype=float)
    N, x = L.size, int((L > var_level).sum())
    p, pi = 1.0 - alpha, max(x, 1e-10) / N
    ll0 = (N - x) * np.log(1.0 - p) + x * np.log(p)
    ll1 = (N - x) * np.log(max(1.0 - pi, 1e-300)) + x * np.log(pi)
    lr = -2.0 * (ll0 - ll1)
    return dict(n=N, exceedances=x, expected=N * p, observed_rate=x / N,
                LR_uc=float(lr), p_value=float(stats.chi2.sf(lr, df=1)))
