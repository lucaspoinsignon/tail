"""Evaluation utilities for the baselines-only package.

These are the tail diagnostics and risk estimators from the tailfm project
(tailfm/{data,risk,evaluate}.py), shipped WITHOUT any tailfm model code so
the baselines can be trained and assessed on their own: windowing, Hill /
tail-dependence / marginal-risk diagnostics, GPD-refined portfolio VaR/CVaR
with bootstrap CIs, and the Kupiec backtest.
"""

from .data import make_windows, synthetic_market
from .risk import (estimate_risk, kupiec_test, portfolio_losses,
                   var_cvar_empirical, var_cvar_gpd)
from .evaluate import (acf, hill_estimator, hill_table, marginal_risk_table,
                       print_report, tail_dependence_report)
