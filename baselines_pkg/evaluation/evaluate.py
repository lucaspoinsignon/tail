"""Tail-focused evaluation of generated vs. real windows.

Diagnostics (all computed on both real and generated samples, pooled over windows):

  * Hill tail index per feature and tail: does the generator reproduce tail *thickness*?
    A vanilla Gaussian-base flow typically shows alpha_gen >> alpha_real (too light).

  * Lower/upper tail-dependence curves per feature pair,

        lambda_L(q) = P(U_i < q, U_j < q) / q,   U = rank / (N+1),

    whose limit q -> 0 is the tail-dependence coefficient. This is the direct test of
    the "crash of one implies crash of the other" requirement: for a tail-dependent
    pair the curve plateaus at lambda > 0; for tail-independent features it decays
    to 0. The generator must match *both* behaviors, pair by pair.

  * Marginal VaR/CVaR (loss = -x, per feature, 1-step) real vs. generated.

  * ACF of returns (should be ~0) and of squared returns (volatility clustering).
"""

from __future__ import annotations

import numpy as np

from .risk import var_cvar_empirical

_EPS = 1e-12


def hill_estimator(x, k_frac: float = 0.05, tail: str = "lower") -> float:
    """Hill estimator of the tail index alpha (heavier tail <=> smaller alpha).

        alpha_hat^{-1} = (1/k) * sum_{i=1}^{k} log( X_(n-i+1) / X_(n-k) )

    computed on the positive exceedances of the requested tail (x -> -x for
    'lower'). Returns np.inf if the tail has too few positive observations.
    (Copied verbatim from tailfm/evt.py so this package needs no model code.)
    """
    y = -np.asarray(x, dtype=float) if tail == "lower" else np.asarray(x, dtype=float)
    y = np.sort(y[y > 0.0])
    k = max(10, int(k_frac * y.size))
    if y.size < k + 1:
        return np.inf
    top, x_k = y[-k:], y[-k - 1]
    inv_alpha = np.mean(np.log(top / x_k))
    return 1.0 / max(inv_alpha, _EPS)



def _pool(windows: np.ndarray) -> np.ndarray:
    """(M, n, f) -> (M*n, f)."""
    w = np.asarray(windows, dtype=float)
    return w.reshape(-1, w.shape[-1])


def hill_table(real: np.ndarray, gen: np.ndarray, k_frac: float = 0.02) -> dict:
    R, G = _pool(real), _pool(gen)
    out = {}
    for j in range(R.shape[1]):
        out[j] = {t: (hill_estimator(R[:, j], k_frac, t),
                      hill_estimator(G[:, j], k_frac, t))
                  for t in ("lower", "upper")}
    return out


def tail_dependence_curve(x: np.ndarray, i: int, j: int,
                          q_grid: np.ndarray, tail: str = "lower") -> np.ndarray:
    """Empirical lambda(q) for features (i, j) of pooled data x: (N, f)."""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    u = (np.argsort(np.argsort(x, axis=0), axis=0) + 1.0) / (N + 1.0)
    ui, uj = (u[:, i], u[:, j]) if tail == "lower" else (1.0 - u[:, i], 1.0 - u[:, j])
    return np.array([np.mean((ui < q) & (uj < q)) / q for q in q_grid])


def tail_dependence_report(real: np.ndarray, gen: np.ndarray,
                           q_grid: np.ndarray | None = None,
                           tail: str = "lower") -> dict:
    if q_grid is None:
        q_grid = np.linspace(0.01, 0.10, 10)
    R, G = _pool(real), _pool(gen)
    f = R.shape[1]
    out = {"q_grid": q_grid}
    for i in range(f):
        for j in range(i + 1, f):
            out[(i, j)] = (tail_dependence_curve(R, i, j, q_grid, tail),
                           tail_dependence_curve(G, i, j, q_grid, tail))
    return out


def marginal_risk_table(real: np.ndarray, gen: np.ndarray,
                        alphas=(0.95, 0.99, 0.995)) -> dict:
    """Per-feature 1-step VaR/CVaR of the loss -x, real vs. generated."""
    R, G = _pool(real), _pool(gen)
    out = {}
    for j in range(R.shape[1]):
        out[j] = {a: (var_cvar_empirical(-R[:, j], a), var_cvar_empirical(-G[:, j], a))
                  for a in alphas}
    return out


def acf(x: np.ndarray, max_lag: int = 10) -> np.ndarray:
    """Mean-over-windows autocorrelation of a (M, n) array, lags 1..max_lag."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean(axis=1, keepdims=True)
    denom = (x ** 2).sum(axis=1)
    return np.array([((x[:, :-k] * x[:, k:]).sum(axis=1) / (denom + 1e-12)).mean()
                     for k in range(1, max_lag + 1)])


def print_report(real: np.ndarray, gen: np.ndarray, feature_names=None) -> None:
    f = real.shape[-1]
    names = feature_names or [f"feat{j}" for j in range(f)]

    print("\n=== Hill tail index (smaller = heavier; gen should match real) ===")
    for j, d in hill_table(real, gen).items():
        for t in ("lower", "upper"):
            r, g = d[t]
            print(f"  {names[j]:>8s} {t:>5s}:  real {r:6.2f}   gen {g:6.2f}")

    print("\n=== Lower tail dependence lambda_L(q=0.02) per pair ===")
    td = tail_dependence_report(real, gen, q_grid=np.array([0.02]))
    for key, val in td.items():
        if key == "q_grid":
            continue
        i, j = key
        print(f"  ({names[i]},{names[j]}):  real {val[0][0]:.3f}   gen {val[1][0]:.3f}")

    print("\n=== Marginal 1-step VaR / CVaR of loss (-x) ===")
    for j, d in marginal_risk_table(real, gen).items():
        for a, ((vr, cr), (vg, cg)) in d.items():
            print(f"  {names[j]:>8s} a={a:5.3f}:  VaR real {vr:8.4f} gen {vg:8.4f}"
                  f"  |  CVaR real {cr:8.4f} gen {cg:8.4f}")

    print("\n=== ACF (lag 1..5), squared series: volatility clustering ===")
    for j in range(f):
        ar = acf(real[:, :, j] ** 2, 5); ag = acf(gen[:, :, j] ** 2, 5)
        print(f"  {names[j]:>8s} sq-ACF real {np.round(ar, 2)}  gen {np.round(ag, 2)}")
