"""Windowing utilities and a synthetic market with KNOWN ground-truth tail structure.

The synthetic process is designed so that every claim the pipeline must get right is
verifiable against the data-generating process:

    log-vol (common factor):  h_t = mu + phi (h_{t-1} - mu) + s_eta eta_t,  sigma_t = exp(h_t)

    shocks (all t_4 unless stated):
        eps_0 = sqrt(rho) S + sqrt(1-rho) e_0     }  shared heavy shock S ~ t_4:
        eps_1 = sqrt(rho) S + sqrt(1-rho) e_1     }  (0,1) are TAIL-DEPENDENT
        eps_2 = e_2  (t_4)                           heavy-tailed, tail-independent of 0,1
        eps_3 = e_3  (N(0,1))                        light-tailed control

    r_{t,j} = scale_j * sigma_t * eps_{t,j}

so a correct generator must simultaneously produce (i) heavy marginals for features 0-2,
(ii) a light marginal for feature 3, (iii) joint crashes for the pair (0,1) and NOT for
other pairs beyond what the common volatility induces, and (iv) volatility clustering.
"""

from __future__ import annotations

import numpy as np


def make_windows(series: np.ndarray, n: int, stride: int = 1) -> np.ndarray:
    """(T, f) series -> (N, n, f) overlapping windows."""
    series = np.asarray(series, dtype=float)
    T = series.shape[0]
    starts = np.arange(0, T - n + 1, stride)
    return np.stack([series[s:s + n] for s in starts], axis=0)


def synthetic_market(T: int, rho: float = 0.7, df: float = 4.0,
                     seed: int = 0) -> np.ndarray:
    from scipy.signal import lfilter
    rng = np.random.default_rng(seed)
    mu, phi, s_eta = np.log(0.01), 0.95, 0.15
    eta = rng.standard_normal(T)
    # AR(1) recursion y_t = phi y_{t-1} + s_eta eta_t, vectorized via a linear filter
    h = mu + lfilter([s_eta], [1.0, -phi], eta)
    sigma = np.exp(h)

    S = rng.standard_t(df, T)
    e = rng.standard_t(df, (T, 3))
    g = rng.standard_normal(T)
    eps = np.stack([np.sqrt(rho) * S + np.sqrt(1 - rho) * e[:, 0],
                    np.sqrt(rho) * S + np.sqrt(1 - rho) * e[:, 1],
                    e[:, 2],
                    g], axis=1)
    scale = np.array([1.0, 1.2, 0.8, 1.0])
    return sigma[:, None] * scale[None, :] * eps
