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

Beside the simulation, :func:`delta_gamma_var` gives the exact VaR and ES of
the position's delta-gamma-theta P&L, quadratic in a Normal spot move: the
analytic comparator that :func:`value_at_risk` stores on :class:`VaRResult`.
Its gap to the full revaluation is what the quadratic leaves out: the spot
terms beyond the second order, the change in time decay and gamma as expiry
nears (theta enters to first order only, so beyond a few days this can
dominate: over 20 trading days of a 30-day long straddle the comparator's VaR
is 40 % below the full revaluation's), legs that expire within the horizon,
implied-vol moves and the fatter tails of the Student-t and GARCH models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri

from .scenarios import DEFAULT_VOL_TENOR_DECAY
from .strategies import Market, Strategy, position_greeks, position_value

__all__ = [
    "DISTRIBUTIONS",
    "DeltaGammaVaR",
    "VaRConfig",
    "VaRResult",
    "delta_gamma_var",
    "simulate_horizon",
    "value_at_risk",
]

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
class DeltaGammaVaR:
    """Exact VaR and ES of the delta-gamma-theta P&L, from :func:`delta_gamma_var`.

    Both are losses in $ (for the Greeks of a whole position): positive when the position loses money, like
    :attr:`VaRResult.var`, and negative when even the worst ``1 - confidence`` of outcomes is a gain (long gamma
    with little time decay).

    Attributes
    ----------
    var : float
        Minus the ``1 - confidence`` quantile of the quadratic P&L ($).
    expected_shortfall : float
        Minus the mean P&L over the worst ``1 - confidence`` of outcomes ($); at least ``var``.
    es_error : float
        QUADPACK's estimate of the error of the numerical integration in ``expected_shortfall`` ($). Against mpmath,
        the ES's whole error (this integration's, the quantile's that bounds the loss region, and rounding) was below
        3e-13 of the P&L's size (see :func:`delta_gamma_var`).
    """

    var: float
    expected_shortfall: float
    es_error: float


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
    #: Exact VaR and ES of the delta-gamma-theta P&L (in $ after ``multiplier``, positive = loss), the analytic
    #: comparator of ``var`` and ``expected_shortfall``: :func:`delta_gamma_var` of today's position Greeks with the
    #: market volatility and the configured horizon and confidence, so a Normal spot move with implied vol frozen
    #: whatever the configured return model and vol of vol. None if a position Greek is not a finite number.
    delta_gamma: DeltaGammaVaR | None = None


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


#: Share of the tail probability that :func:`delta_gamma_var` may leave out on the far side of the parabola's vertex
#: when it reads the quantile off the Normal distribution: 2^-56, below the rounding of the tail probability itself.
_FAR_TAIL = 2.0**-56
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _finite(name: str, value: float) -> float:
    """``value`` as a float; ``ValueError`` naming the argument ``name`` unless it is a single finite number."""
    if np.ndim(value) != 0:
        raise ValueError(f"{name} must be a single number, got {value!r}")
    try:
        finite = math.isfinite(value)
    except TypeError:  # a string, None, ...
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not finite:
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def delta_gamma_var(
    delta: float,
    gamma: float,
    theta: float,
    spot: float,
    vol: float,
    horizon_days: float,
    confidence: float,
    *,
    es_rtol: float = 1e-10,
) -> DeltaGammaVaR:
    """Exact VaR and ES of the delta-gamma-theta P&L under a Normal spot move.

    Over a horizon of ``h = horizon_days`` trading days the position's P&L is taken to second order in the spot move
    ``x`` and to first order in time,

        ``P&L = θ h/252 + Δ x + ½ Γ x²``,  ``x ~ N(0, s²)``,  ``s = S σ √(h/252)``,

    the analytic comparator of the full revaluation in :func:`value_at_risk`, which stores it on
    :attr:`VaRResult.delta_gamma`. With ``x = s z`` and ``δ = Δ / (Γ s)``, completing the square gives

        ``P&L = θ h/252 − Δ²/(2Γ) + ½ Γ s² Y``,  ``Y = (z + δ)² ~ χ'²(1, λ = δ²)``,

    a shifted and scaled non-central chi-square with one degree of freedom. VaR is minus the P&L at the quantile of
    ``Y`` from ``scipy.stats.ncx2``: its lower tail (``ppf``) when Γ > 0, since long gamma loses most near the
    vertex of the parabola, and its upper tail (``isf``) when Γ < 0.

    As Γ → 0 the vertex moves out without bound (λ → ∞), where ``ncx2`` loses digits in ``Y − λ`` (5e-13 of
    the P&L at λ = 1e8) and fails beyond about λ = 1e12 (NaN in SciPy 1.18; RuntimeWarnings, 0.25 s and an error of
    5e-7 in 1.11; measured). So once the loss region's part beyond the vertex holds less than 2^-56 of the tail
    probability (``|δ| ≥ (z_α + z_f) / 2`` with ``Φ(−z_f) = 2^-56 (1 − α)``: |δ| above 5.7 at 99 %, λ above 32),
    the P&L is monotone wherever the probability is and VaR is read off the Normal quantile ``z_α`` directly:
    ``VaR = |Δ| s z_α − ½ Γ s² z_α² − θ h/252``. That is the limit of the chi-square form, so VaR is continuous in
    Γ, and at Γ = 0 it is the delta-normal VaR minus the time decay ``θ h/252``.

    ES is minus the mean P&L over the loss region (an interval around the vertex when Γ > 0, the two tails beyond
    it when Γ < 0, one tail in the monotone case): the same distribution, integrated against the Normal density of
    ``z`` by ``scipy.integrate.quad``, whose error estimate is returned as ``es_error`` (at most 1.6e-9 of the
    P&L's size on 20,000 random positions with the default ``es_rtol``).

    Against 50-digit mpmath (bisection on the Normal probability of the loss region for the quantile, the closed form
    of the ES integral), the VaR is within about 2e-14 and the ES within 3e-13 of the P&L's size
    ``|θ| h/252 + |Δ| s + ½ |Γ| s²`` (worst 1.9e-14 and 2.7e-13, the latter the integration's, within ``es_error``)
    on 3,000 random positions and 400 extreme ones (delta or gamma down to 1e-300, either side of the switch to the
    Normal quantile, confidence up to 99.9999 %), in SciPy 1.11 and 1.18. A call takes 0.1 to 0.2 ms, most of it in
    ``ncx2``, after the first, which imports ``scipy.stats`` (0.3 s).

    Parameters
    ----------
    delta : float
        Position delta dV/dS, in shares.
    gamma : float
        Position gamma d²V/dS², in shares per $1 of spot.
    theta : float
        Position theta dV/dt, in $ per year: the raw units of :func:`~.strategies.position_greeks` (positive for a
        short option). The horizon takes ``horizon_days / 252`` years of it, as :func:`value_at_risk` removes that
        much of each option's life: one trading day is 365/252 of a calendar day's theta.
    spot : float
        Spot price ($ per share), positive.
    vol : float
        Annualised volatility of the spot (0.2 = 20 %), positive.
    horizon_days : float
        Horizon in trading days (252 a year), positive; fractions are allowed.
    confidence : float
        Confidence level α, strictly between 0.5 and 1.
    es_rtol : float, default 1e-10
        Accuracy asked of the ES integration, relative to the larger of the integral and the P&L's size; positive.

    Returns
    -------
    DeltaGammaVaR
        VaR and ES in $ (positive = loss) and the integration's error estimate for the ES.

    Raises
    ------
    ValueError
        If an argument is not a single finite number, ``spot``, ``vol``, ``horizon_days`` or ``es_rtol`` is not
        positive, or ``confidence`` is not between 0.5 and 1.

    Examples
    --------
    One contract (100 shares) of a 30-day short straddle at the money, spot 100, volatility 20 %, rate 4 %: delta
    −6.85 shares, gamma −13.86 shares per $1, theta +$2,782 a year. Its one-day 99 % delta-normal VaR is $20.09;
    with gamma and theta the VaR is $70.55, where full revaluation (:func:`value_at_risk`, 400,000 paths, seed 7)
    gives $70.29.

    >>> from options_analyzer.strategies import Market, build_template, position_greeks
    >>> greeks = position_greeks(build_template("Short Straddle", 100, 5), Market(100, 0.04, 0.2, 30 / 365))
    >>> delta, gamma, theta = (100 * greeks[name] for name in ("delta", "gamma", "theta"))
    >>> dg = delta_gamma_var(delta, gamma, theta, 100, 0.2, 1, 0.99)
    >>> round(dg.var, 2), round(dg.expected_shortfall, 2)
    (70.55, 92.79)

    Without gamma it is the delta-normal VaR, |Δ| S σ √(h/252) z_α = $293.09 for 100 shares, less the time decay:

    >>> round(delta_gamma_var(100, 0, 0, 100, 0.2, 1, 0.99).var, 2)
    293.09
    >>> round(delta_gamma_var(100, 0, 252, 100, 0.2, 1, 0.99).var, 2)  # theta $252 a year: $1 a trading day
    292.09
    """
    delta, gamma, theta = _finite("delta", delta), _finite("gamma", gamma), _finite("theta", theta)
    positive = {"spot": spot, "vol": vol, "horizon_days": horizon_days, "es_rtol": es_rtol}
    for name, value in positive.items():
        if not _finite(name, value) > 0:
            raise ValueError(f"{name} must be positive, got {value!r}")
    spot, vol, horizon_days, es_rtol = (float(value) for value in positive.values())
    confidence = _finite("confidence", confidence)
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1")
    # Imported here: scipy.stats takes 0.3-0.4 s to import (measured), which the app would otherwise pay at every
    # start, on every tab; only the Risk tab's VaR calls this.
    from scipy.integrate import quad
    from scipy.stats import ncx2

    tau = horizon_days / TRADING_DAYS
    s = spot * vol * math.sqrt(tau)
    # P&L(z) = c + a z + b z², z ~ N(0, 1).
    c, a, b = theta * tau, delta * s, 0.5 * gamma * s * s
    if not math.isfinite(abs(c) + abs(a) + abs(b)):
        raise ValueError("the delta, gamma and theta P&L over the horizon overflows; check the units of the Greeks")
    tail = 1.0 - confidence
    z_alpha = float(ndtri(confidence))
    # Beyond z_far the Normal tail holds 2^-56 of the tail probability. When 2 |δ| = |a / b| is at least
    # z_alpha + z_far, the part of the loss region on the far side of the vertex z = -δ starts beyond z_far.
    z_far = -float(ndtri(_FAR_TAIL * tail))
    if abs(a) >= (z_alpha + z_far) * abs(b):
        # Monotone where the probability is: the loss tail is the Normal tail on the side delta points away from
        # (the lower one for a constant P&L, a = b = 0).
        z_loss = -z_alpha if a >= 0 else z_alpha
        quantile = c + a * z_loss + b * z_loss * z_loss
        # The ES integral runs over t = z, where the P&L is c0 + a0 t + b t².
        shift, c0, a0 = 0.0, c, a
        regions = [(-math.inf, z_loss)] if a >= 0 else [(z_loss, math.inf)]
    else:
        d = a / (2.0 * b)  # δ: below (z_alpha + z_far) / 2 in size here, 6.6 at 99.99 %, so λ = δ² is moderate
        lam = d * d
        # A non-centrality below 1e-30 moves the quantile by less than 1e-30 of itself, but a subnormal one throws
        # ncx2.isf off by up to tens of percent of it (74 % at λ = 3.5e-323 and a 30 % tail, 1.5 % at λ = 2e-321 and
        # 25 %; SciPy 1.11 and 1.18, measured): read the central quantile.
        nc = lam if lam >= 1e-30 else 0.0
        y = float(ncx2.ppf(tail, 1, nc) if b > 0 else ncx2.isf(tail, 1, nc))
        quantile = c + b * (y - lam)
        # The ES integral runs over t = z + δ = ±√Y, where the P&L is c - b λ + b t²: the bounds of the loss region
        # are then ±√y exactly, without the cancellation in its bounds -δ ± √y in z.
        shift, c0, a0 = d, c - b * lam, 0.0
        u = math.sqrt(y)
        regions = [(-u, u)] if b > 0 else [(-math.inf, -u), (u, math.inf)]

    def weighted_pnl(t: float) -> float:
        z = t - shift
        # The Normal density underflows to zero beyond |z| = 38.6; stopping at 40 keeps a P&L that overflows far out
        # from making inf * 0 = NaN.
        return (a0 * t + b * t * t) * math.exp(-0.5 * z * z) * _INV_SQRT_2PI if abs(z) < 40.0 else 0.0

    # The loss region holds the tail probability by construction, so its constant term c0 adds c0 (1 - α) exactly;
    # integrating it too cost up to 2e-11 of the ES for a P&L of theta alone (measured).
    size = abs(c) + abs(a) + abs(b)  # the P&L's size: each term at one standard deviation of the spot move
    total, error = c0 * tail, 0.0
    for lo, hi in regions:
        value, err = quad(weighted_pnl, lo, hi, epsabs=es_rtol * size * tail, epsrel=es_rtol, limit=200)
        total += value
        error += err
    return DeltaGammaVaR(var=-quantile, expected_shortfall=-total / tail, es_error=error / tail)


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

    greeks = position_greeks(strategy, market)
    sigma_h = market.volatility * np.sqrt(horizon_years)
    delta_normal = float(abs(multiplier * greeks["delta"] * market.spot) * sigma_h * ndtri(config.confidence))
    # The analytic comparator from the same Greeks, volatility, horizon and confidence.
    dollar_greeks = [float(multiplier * greeks[name]) for name in ("delta", "gamma", "theta")]
    delta_gamma = (
        delta_gamma_var(*dollar_greeks, market.spot, market.volatility, h, config.confidence)
        if all(math.isfinite(greek) for greek in dollar_greeks)
        else None
    )

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
        delta_gamma=delta_gamma,
    )
