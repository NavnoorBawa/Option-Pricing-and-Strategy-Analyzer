"""Value-at-Risk and Expected Shortfall by full-revaluation Monte Carlo.

The underlying's log-return is simulated day by day under

* ``"normal"`` - Gaussian daily returns with constant volatility;
* ``"student_t"`` - Student-t daily returns rescaled to unit variance, so the
  volatility is preserved and only the tails get fatter;
* ``"garch"`` - GARCH(1,1) with variance targeting: the long-run daily
  variance equals the market volatility squared / 252 and
  ``omega = (1 - alpha - beta) * long_run_variance``. The same shocks drive
  the returns and the variance recursion, as the model requires.

Optionally the implied volatility moves too: a log-normal shock correlated
with the spot return through a Gaussian copula (the spot return is mapped to
exact normal scores by rank, so the vol multiplier keeps mean one whatever
the return distribution) and damped for longer tenors like the stress tests.
This captures vega risk.

The position is then re-priced leg by leg at the horizon, with each option's
remaining life reduced accordingly. A leg that expires *before* the horizon is
settled at its intrinsic value on the simulated price at the close of its
expiry day, and that cash then earns the risk-free rate until the horizon.
VaR and ES are read off the simulated P&L distribution:

* ``VaR_a`` = the ``a``-quantile of the loss distribution (loss = -P&L);
* ``ES_a``  = the mean loss in the worst ``1 - a`` tail (Acerbi-Tasche
  estimator), the measure used by Basel FRTB at ``a = 97.5 %``.

Horizons are in trading days (252 per year) because volatility scales with
trading time; a horizon of ``h`` days removes ``h / 252`` years from each
option's life.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri

from .scenarios import DEFAULT_VOL_TENOR_DECAY
from .strategies import Market, Strategy, position_greeks, position_value

__all__ = ["DISTRIBUTIONS", "VaRConfig", "VaRResult", "simulate_horizon", "value_at_risk"]

TRADING_DAYS = 252
DISTRIBUTIONS = {"normal": "Normal", "student_t": "Student-t (fat tails)", "garch": "GARCH(1,1)"}
#: Fewer worst outcomes than this make the tail estimates unreliable.
MIN_TAIL_OBSERVATIONS = 10


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
    #: Annualised volatility of one-month implied volatility (0 = implied vol stays constant).
    vol_of_vol: float = 0.0
    #: Correlation between spot returns and implied-volatility changes.
    spot_vol_correlation: float = -0.7
    #: Damping of implied-vol moves for tenors beyond one month: ``(tau / 1M) ** -decay``.
    vol_tenor_decay: float = DEFAULT_VOL_TENOR_DECAY
    seed: int | None = 7

    def __post_init__(self):
        numbers = {
            "confidence": self.confidence,
            "dof": self.dof,
            "garch_alpha": self.garch_alpha,
            "garch_beta": self.garch_beta,
            "drift": self.drift,
            "vol_of_vol": self.vol_of_vol,
            "spot_vol_correlation": self.spot_vol_correlation,
            "vol_tenor_decay": self.vol_tenor_decay,
        }
        for name, value in numbers.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0.5 < self.confidence < 1.0:
            raise ValueError("confidence must be between 0.5 and 1")
        if self.horizon_days != int(self.horizon_days) or int(self.horizon_days) < 1:
            raise ValueError("horizon must be a whole number of trading days (at least one)")
        if self.n_paths != int(self.n_paths) or int(self.n_paths) < 1_000:
            raise ValueError("use a whole number of at least 1,000 simulations")
        if int(self.n_paths) * (1.0 - self.confidence) < MIN_TAIL_OBSERVATIONS:
            raise ValueError(
                f"too few simulations for this confidence: the tail would hold fewer than "
                f"{MIN_TAIL_OBSERVATIONS} outcomes; increase the number of simulations"
            )
        if self.distribution not in DISTRIBUTIONS:
            raise ValueError(f"distribution must be one of {sorted(DISTRIBUTIONS)}")
        if (self.distribution == "student_t" or self.garch_t) and not self.dof > 2:
            raise ValueError("Student-t degrees of freedom must exceed 2 for a finite variance")
        garch_invalid = self.garch_alpha < 0 or self.garch_beta < 0 or self.garch_alpha + self.garch_beta >= 1
        if self.distribution == "garch" and garch_invalid:
            raise ValueError("GARCH needs alpha, beta >= 0 and alpha + beta < 1 (stationarity)")
        if not -1.0 <= self.spot_vol_correlation <= 1.0:
            raise ValueError("correlation must be between -1 and 1")
        if self.vol_of_vol < 0 or self.vol_tenor_decay < 0:
            raise ValueError("vol of vol and tenor decay must be non-negative")


@dataclass(frozen=True)
class VaRResult:
    var: float
    expected_shortfall: float
    var_ci95: tuple[float, float]
    delta_normal_var: float
    pnl: np.ndarray
    mean: float
    std: float
    #: NaN when the P&L is (numerically) riskless and higher moments are meaningless.
    skewness: float
    excess_kurtosis: float
    worst: float
    best: float
    #: Number of option legs that expire before the horizon and are settled on their expiry day.
    settled_legs: int
    config: VaRConfig


def _unit_variance_t(rng: np.random.Generator, dof: float, size) -> np.ndarray:
    return rng.standard_t(dof, size=size) * np.sqrt((dof - 2.0) / dof)


def _simulate(market: Market, config: VaRConfig, checkpoints: list[int]):
    """Simulate log-returns day by day, recording ``spot`` at each checkpoint day.

    Returns ``(spot_at, spot_h, log_return_h)`` where ``spot_at`` maps a day to
    the simulated prices at the close of that day.
    """
    rng = np.random.default_rng(config.seed)
    n = int(config.n_paths)
    h = int(config.horizon_days)
    daily_var = market.volatility**2 / TRADING_DAYS
    daily_drift = (config.drift - market.dividend_yield) / TRADING_DAYS
    marks = sorted({d for d in checkpoints if 0 < d < h})
    spot_at: dict[int, np.ndarray] = {}
    log_return = np.zeros(n)

    if config.distribution == "normal":
        # Exact Gaussian increments between consecutive checkpoints.
        previous = 0
        for day in [*marks, h]:
            span = day - previous
            log_return += span * (daily_drift - 0.5 * daily_var) + np.sqrt(span * daily_var) * rng.standard_normal(n)
            if day != h:
                spot_at[day] = market.spot * np.exp(log_return)
            previous = day
    else:
        garch = config.distribution == "garch"
        fat_tails = config.distribution == "student_t" or config.garch_t
        alpha, beta = config.garch_alpha, config.garch_beta
        omega = (1.0 - alpha - beta) * daily_var
        variance = np.full(n, daily_var)
        for day in range(1, h + 1):
            z = _unit_variance_t(rng, config.dof, n) if fat_tails else rng.standard_normal(n)
            if garch:
                eps = np.sqrt(variance) * z
                log_return += daily_drift - 0.5 * variance + eps
                variance = omega + alpha * eps**2 + beta * variance
            else:
                log_return += daily_drift - 0.5 * daily_var + np.sqrt(daily_var) * z
            if day in marks:
                spot_at[day] = market.spot * np.exp(log_return)
    return spot_at, market.spot * np.exp(log_return), log_return, rng


def _vol_multiplier(log_return: np.ndarray, config: VaRConfig, rng: np.random.Generator) -> np.ndarray:
    """Log-normal one-month implied-vol multiplier with mean one, correlated with the spot return."""
    n = log_return.size
    if config.vol_of_vol <= 0:
        return np.ones(n)
    # Gaussian copula: exact normal scores of the spot return (by rank) keep the marginal N(0, 1).
    ranks = np.empty(n)
    ranks[np.argsort(log_return, kind="stable")] = np.arange(n)
    z_spot = ndtri((ranks + 0.5) / n)
    rho = config.spot_vol_correlation
    z_vol = rho * z_spot + np.sqrt(1.0 - rho**2) * rng.standard_normal(n)
    eta_t = config.vol_of_vol * np.sqrt(int(config.horizon_days) / TRADING_DAYS)
    return np.exp(eta_t * z_vol - 0.5 * eta_t**2)


def simulate_horizon(market: Market, config: VaRConfig) -> tuple[np.ndarray, np.ndarray]:
    """Simulate ``(spot, vol_multiplier)`` at the horizon under the configured model."""
    _, spot, log_return, rng = _simulate(market, config, [])
    return spot, _vol_multiplier(log_return, config, rng)


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


def _settlement_days(strategy: Strategy, market: Market, horizon_days: int) -> dict[int, int]:
    """Index of every option leg expiring before the horizon -> trading day it settles on.

    A leg settles at the close of the first trading day on or after its expiry, which is the horizon itself for a
    leg that expires during the horizon's last day (a 30-day option at a 21-day horizon: 20.7 trading days).
    """
    days = {}
    for i, leg in enumerate(strategy.legs):
        if leg.is_option:
            expiry = market.expiry if leg.expiry is None else leg.expiry
            expiry_days = expiry * TRADING_DAYS
            if expiry_days < horizon_days - 1e-9:
                days[i] = int(min(max(math.ceil(expiry_days - 1e-9), 1), horizon_days))
    return days


def value_at_risk(strategy: Strategy, market: Market, config: VaRConfig, multiplier: float = 1.0) -> VaRResult:
    """Full-revaluation Monte Carlo VaR/ES of ``strategy`` (in $ after ``multiplier``)."""
    h = int(config.horizon_days)
    settle = _settlement_days(strategy, market, h)
    spot_at, spot, log_return, rng = _simulate(market, config, sorted({d for d in settle.values() if d < h}))
    vol_multiplier = _vol_multiplier(log_return, config, rng)
    horizon_years = h / TRADING_DAYS

    later = np.zeros_like(spot)
    alive = tuple(leg for i, leg in enumerate(strategy.legs) if i not in settle)
    if alive:
        later = later + np.asarray(
            position_value(
                Strategy("alive", alive),
                market,
                spot,
                horizon_years,
                vol_multiplier=vol_multiplier,
                vol_tenor_decay=config.vol_tenor_decay if config.vol_of_vol > 0 else 0.0,
            ),
            dtype=float,
        )
    for i, day in settle.items():
        leg = strategy.legs[i]
        phi = 1.0 if leg.kind == "call" else -1.0
        payoff = np.maximum(phi * ((spot_at[day] if day < h else spot) - leg.strike), 0.0)
        later = later + leg.quantity * payoff * np.exp(market.rate * (h - day) / TRADING_DAYS)

    today = float(position_value(strategy, market))
    pnl = multiplier * (later - today)
    var, es, ci = _var_es(pnl, config.confidence)

    delta = position_greeks(strategy, market)["delta"]
    sigma_h = market.volatility * np.sqrt(horizon_years)
    delta_normal = float(abs(multiplier * delta * market.spot) * sigma_h * ndtri(config.confidence))

    mean = float(pnl.mean())
    std = float(pnl.std(ddof=1))
    centred = pnl - mean
    # Rounding noise on a riskless position (box, conversion) is not a distribution.
    degenerate = std <= 1e-9 * max(abs(mean), abs(multiplier * today), 1e-12)
    skew = np.nan if degenerate else float(np.mean(centred**3) / std**3)
    kurt = np.nan if degenerate else float(np.mean(centred**4) / std**4 - 3.0)
    return VaRResult(
        var=var,
        expected_shortfall=es,
        var_ci95=ci,
        delta_normal_var=delta_normal,
        pnl=pnl,
        mean=mean,
        std=0.0 if degenerate else std,
        skewness=skew,
        excess_kurtosis=kurt,
        worst=float(pnl.min()),
        best=float(pnl.max()),
        settled_legs=len(settle),
        config=config,
    )
