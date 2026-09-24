"""Value-at-Risk and Expected Shortfall by full-revaluation Monte Carlo.

The underlying's log-return over the horizon is simulated day by day under

* ``"normal"`` - Gaussian daily returns with constant volatility;
* ``"student_t"`` - Student-t daily returns rescaled to unit variance, so the
  volatility is preserved and only the tails get fatter;
* ``"garch"`` - GARCH(1,1) with variance targeting: the long-run daily
  variance equals the market volatility squared / 252 and
  ``omega = (1 - alpha - beta) * long_run_variance``. The same shocks drive
  the returns and the variance recursion, as the model requires.

Optionally the implied volatility moves too (log-normal, correlated with the
spot return), which captures vega risk. The position is then re-priced leg
by leg at the horizon, with the options' remaining life reduced accordingly,
and VaR/ES are read off the simulated P&L distribution:

* ``VaR_a`` = the ``a``-quantile of the loss distribution (loss = -P&L);
* ``ES_a``  = the mean loss in the worst ``1 - a`` tail (Acerbi-Tasche
  estimator), the measure used by Basel FRTB at ``a = 97.5 %``.

Horizons are in trading days (252 per year) because volatility scales with
trading time; a horizon of ``h`` days removes ``h / 252`` years from each
option's life.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri

from .strategies import Market, Strategy, position_greeks, position_value

__all__ = ["DISTRIBUTIONS", "VaRConfig", "VaRResult", "simulate_horizon", "value_at_risk"]

TRADING_DAYS = 252
DISTRIBUTIONS = {"normal": "Normal", "student_t": "Student-t (fat tails)", "garch": "GARCH(1,1)"}


@dataclass(frozen=True)
class VaRConfig:
    confidence: float = 0.99
    horizon_days: int = 1
    n_paths: int = 50_000
    distribution: str = "normal"
    #: Degrees of freedom of the Student-t innovations (also used by GARCH when ``garch_t``).
    dof: float = 5.0
    # Typical daily equity-index estimates (e.g. Zivot: S&P 500 alpha 0.09, beta 0.90).
    garch_alpha: float = 0.09
    garch_beta: float = 0.90
    #: Use Student-t innovations inside GARCH.
    garch_t: bool = False
    #: Annual expected return of the underlying (0 = no drift, common for short horizons).
    drift: float = 0.0
    #: Annualised volatility of implied volatility (0 = implied vol stays constant).
    vol_of_vol: float = 0.0
    #: Correlation between spot returns and implied-volatility changes.
    spot_vol_correlation: float = -0.7
    seed: int | None = 7

    def __post_init__(self):
        if not 0.5 < self.confidence < 1.0:
            raise ValueError("confidence must be between 0.5 and 1")
        if int(self.horizon_days) < 1:
            raise ValueError("horizon must be at least one trading day")
        if int(self.n_paths) < 1_000:
            raise ValueError("use at least 1,000 simulations")
        if self.distribution not in DISTRIBUTIONS:
            raise ValueError(f"distribution must be one of {sorted(DISTRIBUTIONS)}")
        if (self.distribution == "student_t" or self.garch_t) and not self.dof > 2:
            raise ValueError("Student-t degrees of freedom must exceed 2 for a finite variance")
        garch_invalid = self.garch_alpha < 0 or self.garch_beta < 0 or self.garch_alpha + self.garch_beta >= 1
        if self.distribution == "garch" and garch_invalid:
            raise ValueError("GARCH needs alpha, beta >= 0 and alpha + beta < 1 (stationarity)")
        if not -1.0 <= self.spot_vol_correlation <= 1.0:
            raise ValueError("correlation must be between -1 and 1")
        if self.vol_of_vol < 0:
            raise ValueError("vol of vol must be non-negative")


@dataclass(frozen=True)
class VaRResult:
    var: float
    expected_shortfall: float
    var_ci95: tuple[float, float]
    delta_normal_var: float
    pnl: np.ndarray
    mean: float
    std: float
    skewness: float
    excess_kurtosis: float
    worst: float
    best: float
    config: VaRConfig


def _unit_variance_t(rng: np.random.Generator, dof: float, size) -> np.ndarray:
    return rng.standard_t(dof, size=size) * np.sqrt((dof - 2.0) / dof)


def simulate_horizon(market: Market, config: VaRConfig) -> tuple[np.ndarray, np.ndarray]:
    """Simulate ``(spot, vol_multiplier)`` at the horizon under the configured model."""
    rng = np.random.default_rng(config.seed)
    n = int(config.n_paths)
    h = int(config.horizon_days)
    daily_var = market.volatility**2 / TRADING_DAYS
    daily_drift = (config.drift - market.dividend_yield) / TRADING_DAYS

    if config.distribution == "normal":
        z = rng.standard_normal(n)
        log_return = h * (daily_drift - 0.5 * daily_var) + np.sqrt(h * daily_var) * z
    elif config.distribution == "student_t":
        shocks = _unit_variance_t(rng, config.dof, (h, n))
        log_return = h * (daily_drift - 0.5 * daily_var) + np.sqrt(daily_var) * shocks.sum(axis=0)
    else:  # GARCH(1,1) with variance targeting
        alpha, beta = config.garch_alpha, config.garch_beta
        omega = (1.0 - alpha - beta) * daily_var
        variance = np.full(n, daily_var)
        log_return = np.zeros(n)
        for _ in range(h):
            z = _unit_variance_t(rng, config.dof, n) if config.garch_t else rng.standard_normal(n)
            eps = np.sqrt(variance) * z
            log_return += daily_drift - 0.5 * variance + eps
            variance = omega + alpha * eps**2 + beta * variance

    spot = market.spot * np.exp(log_return)

    if config.vol_of_vol > 0:
        # Standardised spot shock, then a correlated shock for log implied vol.
        std = log_return.std()
        z_spot = (log_return - log_return.mean()) / std if std > 0 else np.zeros(n)
        rho = config.spot_vol_correlation
        z_vol = rho * z_spot + np.sqrt(1.0 - rho**2) * rng.standard_normal(n)
        eta_t = config.vol_of_vol * np.sqrt(h / TRADING_DAYS)
        vol_multiplier = np.exp(eta_t * z_vol - 0.5 * eta_t**2)
    else:
        vol_multiplier = np.ones(n)
    return spot, vol_multiplier


def _var_es(pnl: np.ndarray, confidence: float) -> tuple[float, float, tuple[float, float]]:
    """Acerbi-Tasche (2002) natural estimators on the sorted P&L.

    With ``w = floor(n (1 - confidence))`` worst outcomes, VaR is minus the
    w-th worst P&L and ES is minus the average of the w worst P&Ls (which
    makes the estimator coherent and guarantees ``ES >= VaR``).
    """
    ordered = np.sort(pnl)
    n = ordered.size
    w = max(int(np.floor(n * (1.0 - confidence) + 1e-9)), 1)
    var = float(-ordered[w - 1])
    es = float(-ordered[:w].mean())
    # Distribution-free 95% confidence interval for the quantile (order statistics).
    half = 1.959963984540054 * np.sqrt(n * confidence * (1.0 - confidence))
    lo = int(np.clip(np.floor(w - half), 1, n)) - 1
    hi = int(np.clip(np.ceil(w + half), 1, n)) - 1
    return var, es, (float(-ordered[hi]), float(-ordered[lo]))


def value_at_risk(strategy: Strategy, market: Market, config: VaRConfig, multiplier: float = 1.0) -> VaRResult:
    """Full-revaluation Monte Carlo VaR/ES of ``strategy`` (in $ after ``multiplier``)."""
    spot, vol_multiplier = simulate_horizon(market, config)
    horizon_years = int(config.horizon_days) / TRADING_DAYS
    today = float(position_value(strategy, market))
    later = position_value(strategy, market, spot, horizon_years, vol_multiplier=vol_multiplier)
    pnl = multiplier * (np.asarray(later, dtype=float) - today)
    var, es, ci = _var_es(pnl, config.confidence)

    delta = position_greeks(strategy, market)["delta"]
    sigma_h = market.volatility * np.sqrt(horizon_years)
    delta_normal = float(abs(multiplier * delta * market.spot) * sigma_h * ndtri(config.confidence))

    mean = float(pnl.mean())
    std = float(pnl.std(ddof=1))
    centred = pnl - mean
    skew = float(np.mean(centred**3) / std**3) if std > 0 else 0.0
    kurt = float(np.mean(centred**4) / std**4 - 3.0) if std > 0 else 0.0
    return VaRResult(
        var=var,
        expected_shortfall=es,
        var_ci95=ci,
        delta_normal_var=delta_normal,
        pnl=pnl,
        mean=mean,
        std=std,
        skewness=skew,
        excess_kurtosis=kurt,
        worst=float(pnl.min()),
        best=float(pnl.max()),
        config=config,
    )
