"""VaR and ES backtests, forecast comparison, their power, and a synthetic market history to run them on.

A VaR model is right when its exceptions (days whose loss exceeds the forecast VaR) come at the forecast rate and
independently of one another; an ES model is right when, in addition, the losses beyond the VaR average the forecast
ES. Each test here is written from its paper:

* :func:`kupiec_pof` - the proportion-of-failures likelihood ratio of Kupiec (1995), with its chi-square(1) p-value,
  the exact p-value of the same statistic under the binomial, and the binomial tail ``P(X >= x)``;
* :func:`christoffersen` - the Markov test of independence of Christoffersen (1998) and his joint test of
  conditional coverage;
* :func:`traffic_light` - the Basel Committee's traffic light: the green, amber and red zones and the capital
  multiplier of MAR32.9 Table 1 at 250 days and 99 %, and the zones of BCBS (1996) for other sample sizes;
* :func:`acerbi_szekely_z2` - the Z2 test of ES of Acerbi and Szekely (2014), with critical values simulated under
  the model from a fixed seed;
* :func:`fz0_loss` and :func:`diebold_mariano` - the FZ0 loss of Patton, Ziegel and Chen (2019), a consistent scoring
  function for VaR and ES jointly, and the test of equal expected loss of Diebold and Mariano (1995) with the
  Newey and West (1987) long-run variance, to rank two models;
* :func:`pla_test` - the P&L attribution test of MAR32.34 to MAR32.42: the Spearman correlation and the
  Kolmogorov-Smirnov distance between hypothetical and risk-theoretical P&L, and their zone;
* :func:`power_kupiec`, :func:`power_traffic_light`, :func:`power_christoffersen` and :func:`power_z2` - how often a
  test rejects a wrong model at a given sample size, and (with the true model) its actual size.

The power is the reason for the last group. At 250 days and 99 %, Kupiec's test at a nominal 5 % rejects a correct
model 9.5 % of the time, because the binomial is discrete, and a model whose true exception rate is 2 %, twice the
forecast, only 24 % of the time; the traffic light leaves that model green 44 % of the time (BCBS 1996, Table 1).
One year of exceptions cannot tell those two models apart; ten years (2,520 days) reject the second one 98.7 % of the
time.

:func:`synthetic_history` generates a daily market history (closes and an implied-vol index) labelled
:data:`~options_analyzer.market_data.synthetic.SYNTHETIC_LABEL`, for demonstrations and the tests.
:func:`rolling_backtest` runs VaR models over a history: each day it builds a position at the close, lets every model
forecast the position's one-day P&L distribution from the history up to that close, and compares the forecasts with
the hypothetical P&L, the position held unchanged to the next close.

Conventions: P&L is signed (a loss is negative); VaR and ES are positive losses, the Acerbi-Tasche estimators of
:mod:`options_analyzer.risk` for a model given as simulated P&L. A day is an exception when its loss exceeds the VaR,
``pnl < -var``. A confidence ``c`` has the exception probability ``p = 1 - c``; the ES tail probability of Z2 and FZ0
is called ``alpha`` as in the papers (2.5 % for FRTB's 97.5 % ES).

References
----------
* Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*
  3(2), 73-84. doi:10.3905/jod.1995.407942
* Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review* 39(4), 841-862.
  doi:10.2307/2527341
* Basel Committee on Banking Supervision (1996). Supervisory framework for the use of "backtesting" in conjunction
  with the internal models approach to market risk capital requirements. https://www.bis.org/publ/bcbs22.htm
* Basel Committee on Banking Supervision (2020). MAR32, Backtesting and P&L attribution test requirements, version
  of 27 March 2020, in force from 1 January 2023. https://www.bis.org/basel_framework/chapter/MAR/32.htm
* Acerbi, C. and Szekely, B. (2014). Backtesting expected shortfall. MSCI research paper, December 2014.
  https://www.msci.com/documents/10199/22aa9922-f874-4060-b77a-0f0e267a489b
* Patton, A. J., Ziegel, J. F. and Chen, R. (2019). Dynamic semiparametric models for expected shortfall (and
  Value-at-Risk). *Journal of Econometrics* 211(2), 388-413. doi:10.1016/j.jeconom.2018.10.008
* Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic
  Statistics* 13(3), 253-263. doi:10.1080/07350015.1995.10524599
* Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation
  consistent covariance matrix. *Econometrica* 55(3), 703-708. doi:10.2307/1913610
* Newey, W. K. and West, K. D. (1994). Automatic lag selection in covariance matrix estimation. *Review of Economic
  Studies* 61(4), 631-653. doi:10.2307/2297912
* Hull, J. and White, A. (1998). Incorporating volatility updating into the historical simulation method for
  value-at-risk. *Journal of Risk* 1(1), 5-19. doi:10.21314/JOR.1998.001
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import signal, stats
from scipy.special import kolmogorov, ndtri, xlogy

from .market_data.synthetic import SYNTHETIC_LABEL
from .risk import TRADING_DAYS, VaRConfig, _var_es, value_at_risk
from .scenarios import DEFAULT_VOL_TENOR_DECAY
from .strategies import Market, Strategy, position_value

__all__ = [
    "MAR32_MULTIPLIERS",
    "ChristoffersenPower",
    "ChristoffersenResult",
    "DieboldMarianoResult",
    "KupiecResult",
    "PLAResult",
    "RollingBacktest",
    "TrafficLight",
    "TrafficLightPower",
    "VaRModel",
    "Z2Result",
    "acerbi_szekely_z2",
    "christoffersen",
    "diebold_mariano",
    "fz0_loss",
    "historical_model",
    "kupiec_pof",
    "monte_carlo_model",
    "newey_west_variance",
    "pla_test",
    "pla_zone",
    "power_christoffersen",
    "power_kupiec",
    "power_traffic_light",
    "power_z2",
    "rolling_backtest",
    "synthetic_history",
    "traffic_light",
    "z2_statistic",
]

#: One trading day in years: the backtest's horizon, as in :mod:`options_analyzer.risk` (a horizon of ``h`` trading
#: days removes ``h / 252`` years from every option's life).
ONE_TRADING_DAY = 1.0 / TRADING_DAYS
#: MAR32.9 Table 1: the backtesting-dependent multiplier for 0 to 9 exceptions in 250 days at 99 % (green 0 to 4,
#: amber 5 to 9); 10 or more exceptions are red, at :data:`RED_MULTIPLIER`.
MAR32_MULTIPLIERS = (1.50, 1.50, 1.50, 1.50, 1.50, 1.70, 1.76, 1.83, 1.88, 1.92)
RED_MULTIPLIER = 2.00
#: BCBS (1996): the amber (there "yellow") zone begins at the number of exceptions whose cumulative probability under
#: the model reaches 95 %, the red zone where it reaches 99.99 %. At 250 days and 99 % that is 5 and 10 exceptions.
AMBER_FROM = 0.95
RED_FROM = 0.9999
#: MAR32.42: a desk is green with a Spearman correlation above 0.80 and a KS distance below 0.09, red with a
#: correlation below 0.70 or a KS distance above 0.12, and amber otherwise.
PLA_GREEN = (0.80, 0.09)
PLA_RED = (0.70, 0.12)


def _check_probability(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1, got {value!r}")
    return value


def _check_count(name: str, value: int, least: int = 0) -> int:
    if isinstance(value, bool) or value != int(value) or int(value) < least:
        raise ValueError(f"{name} must be a whole number of at least {least}, got {value!r}")
    return int(value)


# --- Kupiec (1995) ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class KupiecResult:
    exceptions: int
    observations: int
    #: The model's exception probability (1 - VaR confidence).
    p: float
    #: The likelihood ratio LR_POF, asymptotically chi-square(1) when the model is right.
    lr: float
    #: P(chi-square(1) >= lr).
    p_value: float
    #: The exact p-value of the same statistic: the binomial probability of an LR at least as large.
    exact_p_value: float
    #: P(X >= exceptions) under the model's binomial: the exact one-sided p-value for too many exceptions.
    binomial_tail: float

    @property
    def expected(self) -> float:
        return self.observations * self.p


def _kupiec_lr(exceptions: ArrayLike, observations: int, p: float) -> np.ndarray:
    """LR_POF for every number of exceptions in ``exceptions`` (0 log 0 = 0, so no exception at all is finite)."""
    x = np.asarray(exceptions, dtype=float)
    rate = x / observations
    no = observations - x
    lr = 2.0 * (xlogy(no, 1.0 - rate) + xlogy(x, rate) - xlogy(no, 1.0 - p) - xlogy(x, p))
    # At x = n p exactly the two log-likelihoods cancel; rounding must not leave a negative statistic.
    return np.maximum(lr, 0.0)


def kupiec_pof(exceptions: int, observations: int, p: float = 0.01) -> KupiecResult:
    """Kupiec's (1995) proportion-of-failures test of the exception rate.

    With ``x`` exceptions in ``n`` days and the observed rate ``x / n``, the likelihood ratio

    ``LR = -2 ln[(1 - p)^(n - x) p^x] + 2 ln[(1 - x/n)^(n - x) (x/n)^x]``

    is asymptotically chi-square with one degree of freedom when the model's exception probability ``p`` is right.
    With 250 days the binomial is too discrete for that: the exact p-value (the binomial probability of a statistic
    at least as large) is returned beside it, with the binomial tail ``P(X >= x)``, which only counts too many
    exceptions (what a supervisor cares about).

    Parameters
    ----------
    exceptions : int
        Days whose loss exceeded the VaR.
    observations : int
        Days in the backtest (at least one).
    p : float
        The model's exception probability, ``1 - confidence`` (0.01 for a 99 % VaR).

    Returns
    -------
    KupiecResult

    Examples
    --------
    Ten exceptions in a year at 99 %: Basel's red zone, and a clear rejection however it is computed.

    >>> res = kupiec_pof(10, 250, 0.01)
    >>> round(res.lr, 4), round(res.p_value, 4), round(res.exact_p_value, 4), round(res.binomial_tail, 4)
    (12.9555, 0.0003, 0.0003, 0.0003)

    No exception at all scores 5.0252, which the chi-square approximation rejects at 5 % but the exact p-value does
    not: a correct 99 % model has no exception in 8.1 % of years (and MAR32 puts that year in the green zone). This
    is the discreteness that makes the test's actual size 9.5 % at 250 days (:func:`power_kupiec`).

    >>> res = kupiec_pof(0, 250, 0.01)
    >>> round(res.lr, 4), round(res.p_value, 4), round(res.exact_p_value, 4)
    (5.0252, 0.025, 0.0948)
    """
    n = _check_count("observations", observations, 1)
    x = _check_count("exceptions", exceptions)
    if x > n:
        raise ValueError(f"exceptions ({x}) cannot exceed observations ({n})")
    p = _check_probability("p", p)
    every = np.arange(n + 1)
    lr_all = _kupiec_lr(every, n, p)
    lr = float(lr_all[x])
    pmf = stats.binom.pmf(every, n, p)
    # Statistics equal to the observed one up to rounding count as "at least as large".
    exact = float(min(1.0, pmf[lr_all >= lr * (1.0 - 1e-12)].sum()))
    return KupiecResult(
        exceptions=x,
        observations=n,
        p=p,
        lr=lr,
        p_value=float(stats.chi2.sf(lr, 1)),
        exact_p_value=exact,
        binomial_tail=float(stats.binom.sf(x - 1, n, p)),
    )


def power_kupiec(observations: int, p_true: float, p_model: float = 0.01, *, size: float = 0.05) -> float:
    """Probability that Kupiec's test at the nominal ``size`` rejects a model whose true exception rate is ``p_true``.

    Exact: the binomial probability of the numbers of exceptions whose LR exceeds the chi-square(1) critical value.
    With ``p_true == p_model`` it is the test's actual size, which at 250 days differs from the nominal one.

    Examples
    --------
    >>> round(power_kupiec(250, 0.01, 0.01), 4)  # the actual size of a nominal 5 % test
    0.0948
    >>> round(power_kupiec(250, 0.02, 0.01), 4)  # twice as many exceptions as forecast
    0.2427
    """
    n = _check_count("observations", observations, 1)
    p_true, p_model = _check_probability("p_true", p_true), _check_probability("p_model", p_model)
    size = _check_probability("size", size)
    every = np.arange(n + 1)
    reject = _kupiec_lr(every, n, p_model) > stats.chi2.isf(size, 1)
    return float(stats.binom.pmf(every[reject], n, p_true).sum())


# --- Christoffersen (1998) ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChristoffersenResult:
    #: Transitions between consecutive days: ``n01`` counts a day without an exception followed by one with.
    n00: int
    n01: int
    n10: int
    n11: int
    #: LR_ind, asymptotically chi-square(1) when exceptions are independent, and its p-value.
    lr_ind: float
    p_ind: float
    #: LR_cc, asymptotically chi-square(2) when exceptions are independent at the rate ``p``, and its p-value.
    lr_cc: float
    p_cc: float


def _markov_lrs(n00, n01, n10, n11, p: float) -> tuple[np.ndarray, np.ndarray]:
    """``(LR_ind, LR_cc)`` from transition counts (arrays broadcast; 0 log 0 = 0)."""
    n00, n01, n10, n11 = (np.asarray(c, dtype=float) for c in (n00, n01, n10, n11))
    after_quiet, after_exception = n00 + n01, n10 + n11
    pi01 = np.divide(n01, after_quiet, out=np.zeros_like(n01), where=after_quiet > 0)
    pi11 = np.divide(n11, after_exception, out=np.zeros_like(n11), where=after_exception > 0)
    pi = (n01 + n11) / (after_quiet + after_exception)
    markov = xlogy(n00, 1.0 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1.0 - pi11) + xlogy(n11, pi11)
    iid = xlogy(n00 + n10, 1.0 - pi) + xlogy(n01 + n11, pi)
    null = xlogy(n00 + n10, 1.0 - p) + xlogy(n01 + n11, p)
    return np.maximum(2.0 * (markov - iid), 0.0), np.maximum(2.0 * (markov - null), 0.0)


def christoffersen(hits: ArrayLike, p: float = 0.01) -> ChristoffersenResult:
    """Christoffersen's (1998) tests of independence and of conditional coverage of a sequence of exceptions.

    The exceptions are modelled as a two-state Markov chain with ``pi01 = n01 / (n00 + n01)``, the probability of an
    exception after a day without one, and ``pi11 = n11 / (n10 + n11)``, after a day with one. LR_ind compares that
    chain with independent exceptions at the pooled rate ``(n01 + n11) / (n - 1)``; LR_cc compares it with independent
    exceptions at the model's rate ``p``. Both condition on the first day, as in the paper, so
    ``LR_cc = LR_uc + LR_ind`` with LR_uc Kupiec's statistic on the last ``n - 1`` days. (Some software adds Kupiec's
    statistic on all ``n`` days instead; the two differ by the first day's term.)

    Parameters
    ----------
    hits : array_like of bool
        Whether each day, in order, was an exception (at least two days).
    p : float
        The model's exception probability.

    Examples
    --------
    Five exceptions in a row, on days 100 to 104 of 250: the right number for 99 % (2.5 expected), all at once.

    >>> hits = np.zeros(250, dtype=bool)
    >>> hits[99:104] = True
    >>> res = christoffersen(hits, 0.01)
    >>> (res.n00, res.n01, res.n10, res.n11), round(res.lr_ind, 3), res.p_ind < 1e-7
    ((243, 1, 1, 4), 30.985, True)
    """
    h = np.asarray(hits)
    if h.ndim != 1 or h.size < 2:
        raise ValueError("hits must be a sequence of at least two days")
    if h.dtype != bool:
        if not np.all((h == 0) | (h == 1)):
            raise ValueError("hits must be booleans (or 0 and 1)")
        h = h.astype(bool)
    p = _check_probability("p", p)
    before, after = h[:-1], h[1:]
    n11 = int(np.count_nonzero(before & after))
    n10 = int(np.count_nonzero(before & ~after))
    n01 = int(np.count_nonzero(~before & after))
    n00 = int(before.size - n11 - n10 - n01)
    lr_ind, lr_cc = (float(v) for v in _markov_lrs(n00, n01, n10, n11, p))
    return ChristoffersenResult(
        n00, n01, n10, n11, lr_ind, float(stats.chi2.sf(lr_ind, 1)), lr_cc, float(stats.chi2.sf(lr_cc, 2))
    )


class ChristoffersenPower(NamedTuple):
    independence: float
    conditional_coverage: float


def power_christoffersen(
    observations: int,
    p_true: float,
    p_after_exception: float,
    p_model: float = 0.01,
    *,
    size: float = 0.05,
    n_sims: int = 10_000,
    seed: int = 1998,
) -> ChristoffersenPower:
    """Rejection rates of Christoffersen's tests against clustered exceptions, by simulation.

    The true exceptions follow a stationary two-state Markov chain whose unconditional rate is ``p_true`` and whose
    probability of an exception the day after one is ``p_after_exception`` (``p_true`` itself means independent
    exceptions, which gives the tests' actual size). Each test rejects at its chi-square critical value for ``size``.
    The Monte Carlo standard error of each rate is at most ``0.5 / sqrt(n_sims)``.

    Examples
    --------
    A year at 99 % in which an exception is followed by another a quarter of the time:

    >>> power = power_christoffersen(250, 0.01, 0.25)
    >>> round(power.independence, 2), round(power.conditional_coverage, 2)
    (0.34, 0.23)
    """
    n = _check_count("observations", observations, 2)
    p_true, p_model = _check_probability("p_true", p_true), _check_probability("p_model", p_model)
    p11 = _check_probability("p_after_exception", p_after_exception)
    size = _check_probability("size", size)
    sims = _check_count("n_sims", n_sims, 1)
    # Stationarity: p = p01 / (1 - p11 + p01).
    p01 = p_true * (1.0 - p11) / (1.0 - p_true)
    if not p01 < 1.0:
        raise ValueError("no Markov chain has that rate and that probability of an exception after one")
    rng = np.random.default_rng(seed)
    state = rng.random(sims) < p_true
    counts = np.zeros((4, sims))  # n00, n01, n10, n11
    columns = np.arange(sims)
    for _ in range(n - 1):
        following = rng.random(sims) < np.where(state, p11, p01)
        counts[2 * state + following, columns] += 1.0
        state = following
    lr_ind, lr_cc = _markov_lrs(*counts, p_model)
    return ChristoffersenPower(
        float(np.mean(lr_ind > stats.chi2.isf(size, 1))), float(np.mean(lr_cc > stats.chi2.isf(size, 2)))
    )


# --- The Basel traffic light ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrafficLight:
    #: "green", "amber" or "red".
    zone: str
    exceptions: int
    observations: int
    #: P(X <= exceptions) under the model's binomial.
    cumulative_probability: float
    #: The first numbers of exceptions in the amber and in the red zone at this sample size and coverage.
    amber_from: int
    red_from: int
    #: MAR32.9's backtesting-dependent multiplier; None away from 250 days at 99 %, where the table does not apply.
    multiplier: float | None


def _zone_bounds(observations: int, p: float) -> tuple[int, int]:
    cumulative = stats.binom.cdf(np.arange(observations + 1), observations, p)
    # The cdf reaches 1 at n exceptions (up to rounding), so both thresholds are crossed within the range.
    amber = int(np.argmax(cumulative >= AMBER_FROM - 1e-12))
    red = int(np.argmax(cumulative >= RED_FROM - 1e-12))
    return amber, red


def traffic_light(exceptions: int, observations: int = 250, *, coverage: float = 0.99) -> TrafficLight:
    """The Basel traffic light of a VaR backtest.

    MAR32.9 Table 1 puts 0 to 4 exceptions in 250 days at 99 % in the green zone (multiplier 1.50), 5 to 9 in the
    amber zone (1.70, 1.76, 1.83, 1.88, 1.92) and 10 or more in the red zone (2.00). Those bounds are where the
    cumulative binomial probability of the number of exceptions reaches 95 % and 99.99 % (BCBS 1996, which calls the
    amber zone "yellow" and gives that rule for other sample sizes); the zone is computed by that rule for any number
    of days and any coverage, which reproduces the table at 250 days and 99 %. The multiplier is given only there.

    Examples
    --------
    >>> light = traffic_light(5)
    >>> light.zone, light.multiplier, round(light.cumulative_probability, 4), (light.amber_from, light.red_from)
    ('amber', 1.7, 0.9588, (5, 10))
    >>> traffic_light(12, 1000).zone  # 10 expected in four years
    'green'
    """
    n = _check_count("observations", observations, 1)
    x = _check_count("exceptions", exceptions)
    if x > n:
        raise ValueError(f"exceptions ({x}) cannot exceed observations ({n})")
    coverage = _check_probability("coverage", coverage)
    p = 1.0 - coverage
    amber, red = _zone_bounds(n, p)
    zone = "red" if x >= red else "amber" if x >= amber else "green"
    multiplier = None
    if n == 250 and math.isclose(coverage, 0.99, rel_tol=0.0, abs_tol=1e-12):
        multiplier = MAR32_MULTIPLIERS[x] if x < len(MAR32_MULTIPLIERS) else RED_MULTIPLIER
    return TrafficLight(zone, x, n, float(stats.binom.cdf(x, n, p)), amber, red, multiplier)


class TrafficLightPower(NamedTuple):
    #: Probability of leaving the green zone.
    amber_or_red: float
    red: float


def power_traffic_light(observations: int, p_true: float, p_model: float = 0.01) -> TrafficLightPower:
    """Probability that a model with true exception rate ``p_true`` lands outside the green zone, and in the red one.

    Exact (binomial). With ``p_true == p_model`` these are the probabilities of wrongly flagging a correct model
    (BCBS 1996, Table 1, "type 1"); otherwise one minus them is its "type 2" error.

    Examples
    --------
    >>> [round(v, 3) for v in power_traffic_light(250, 0.01)]  # a correct model
    [0.108, 0.0]
    >>> [round(v, 3) for v in power_traffic_light(250, 0.02)]  # twice the forecast rate: green 43.9 % of the time
    [0.561, 0.03]
    """
    n = _check_count("observations", observations, 1)
    p_true, p_model = _check_probability("p_true", p_true), _check_probability("p_model", p_model)
    amber, red = _zone_bounds(n, p_model)
    return TrafficLightPower(float(stats.binom.sf(amber - 1, n, p_true)), float(stats.binom.sf(red - 1, n, p_true)))


# --- Acerbi and Szekely (2014) Z2 ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Z2Result:
    #: Z2 of the backtest: 0 on average under the model, negative when the tail losses exceed the forecast ES.
    statistic: float
    #: The ``size`` quantile of Z2 simulated under the model: the test rejects below it.
    critical_value: float
    #: Fraction of the simulated Z2 below the observed one.
    p_value: float
    size: float
    n_sims: int


def z2_statistic(pnl: ArrayLike, var: ArrayLike, es: ArrayLike, alpha: float) -> float | np.ndarray:
    """Acerbi and Szekely's ``Z2 = sum_t X_t I_t / (T alpha ES_t) + 1`` over the last axis of ``pnl``.

    ``I_t`` marks the days whose P&L ``X_t`` is at or below ``-VaR_t``; for a model given as a finite set of
    scenarios that includes the scenario at the VaR itself, which keeps Z2's mean at zero under the model. Leading
    axes of ``pnl`` (simulated histories) give one statistic each.
    """
    x = np.asarray(pnl, dtype=float)
    if x.ndim < 1 or x.shape[-1] < 1:
        raise ValueError("pnl must hold at least one day")
    v = np.broadcast_to(np.asarray(var, dtype=float), x.shape[-1:])
    e = np.broadcast_to(np.asarray(es, dtype=float), x.shape[-1:])
    if not np.all(e > 0):
        raise ValueError("ES must be positive (a loss) on every day: Z2 divides by it")
    alpha = _check_probability("alpha", alpha)
    tail = np.where(x <= -v, x / e, 0.0)
    out = tail.sum(axis=-1) / (x.shape[-1] * alpha) + 1.0
    return float(out) if np.ndim(out) == 0 else out


def _standard_tail(alpha: float, dof: float | None) -> tuple[float, float]:
    """VaR and ES, as positive numbers, of a standard Normal (``dof`` None) or Student-t P&L at tail ``alpha``."""
    if dof is None:
        q = float(-ndtri(alpha))
        return q, math.exp(-0.5 * q * q) / math.sqrt(2.0 * math.pi) / alpha
    q = float(stats.t.isf(alpha, dof))
    return q, float(stats.t.pdf(q, dof)) * (dof + q * q) / (dof - 1.0) / alpha


def _tail_draws(rng: np.random.Generator, size: int, tail_probability: float, dof: float | None) -> np.ndarray:
    """``size`` draws of a standard Normal or Student-t below its ``tail_probability`` quantile (inverse cdf)."""
    u = tail_probability * (1.0 - rng.random(size))  # in (0, tail_probability]: never the infinite quantile at 0
    return ndtri(u) if dof is None else stats.t.ppf(u, dof)


def _simulate_z2(
    rng: np.random.Generator,
    n_sims: int,
    days: int,
    alpha: float,
    dof: float | None,
    truth: tuple[float, float | None] | None = None,
) -> np.ndarray:
    """Z2 of ``n_sims`` histories of ``days`` days for a zero-mean Normal (``dof`` None) or Student-t model.

    The P&L is the model's own (``truth`` None, the null) or ``scale`` times a unit-variance Student-t with
    ``true_dof`` degrees of freedom, a Normal for None (``truth = (scale, true_dof)``). The model's Z2 does not depend
    on its scale, so standard draws suffice. Only tail days contribute, so each history draws its number of tail days
    from the binomial and then that many tail values by inverting the distribution below the VaR: exact in
    distribution, with about ``days alpha`` draws per history instead of ``days``.

    Both draws invert a cdf at uniforms, so a seed gives the same histories on every NumPy the project supports.
    ``Generator.binomial`` does not: where the expected count is above 30 it changed between NumPy 1.26 and 2.x, and
    from seed 2014 the 5 % critical value at 2,520 days and alpha = 2.5 % was -0.21149 on one and -0.21133 on the
    other (by inversion, -0.21036 on both). Up to 30 NumPy inverts the cdf too and draws the same counts from a seed.
    """
    q, e = _standard_tail(alpha, dof)
    if truth is None:
        factor, true_dof, tail = 1.0, dof, alpha
    else:
        scale, true_dof = truth
        factor = scale * (1.0 if true_dof is None else math.sqrt((true_dof - 2.0) / true_dof))
        tail = float(stats.norm.cdf(-q / factor) if true_dof is None else stats.t.cdf(-q / factor, true_dof))
    # The binomial quantile of each uniform: the first count whose cdf reaches it (the running maximum guards the
    # search against a cdf that rounding leaves a last bit lower near 1; under 0.01 s for 100,000 histories of 2,520
    # days).
    cdf = np.maximum.accumulate(stats.binom.cdf(np.arange(days + 1), days, tail))
    counts = np.minimum(np.searchsorted(cdf, rng.random(n_sims)), days)
    values = factor * _tail_draws(rng, int(counts.sum()), tail, true_dof)
    sums = np.bincount(np.repeat(np.arange(n_sims), counts), weights=values, minlength=n_sims)
    return sums / (days * alpha * e) + 1.0


def acerbi_szekely_z2(
    pnl: ArrayLike,
    var: ArrayLike,
    es: ArrayLike,
    alpha: float = 0.025,
    *,
    null: str | ArrayLike = "normal",
    dof: float = 5.0,
    size: float = 0.05,
    n_sims: int = 100_000,
    seed: int = 2014,
) -> Z2Result:
    """Acerbi and Szekely's (2014) Z2 test of Expected Shortfall, with simulated critical values.

    ``Z2 = sum_t X_t I_t / (T alpha ES_t) + 1`` (:func:`z2_statistic`) has mean zero when the model's VaR and ES at
    tail probability ``alpha`` are right and is negative when the losses beyond the VaR are larger than forecast. Its
    distribution under the model is simulated (from ``seed``): the test rejects at ``size`` below the simulated
    ``size`` quantile, and the p-value is the fraction of simulated Z2 below the observed one.

    Parameters
    ----------
    pnl, var, es : array_like
        The realised P&L of each day and the model's VaR and ES for it at tail probability ``alpha`` (positive losses;
        ``var`` and ``es`` may be scalars).
    alpha : float
        The ES tail probability (0.025 for a 97.5 % ES).
    null : {"normal", "student_t"} or array_like
        The model's P&L distribution. "normal" and "student_t" (with ``dof``) are zero-mean scale families, for which
        Z2 under the model depends only on the number of days and ``alpha``; for any other model pass P&L simulated
        under it, an array of shape ``(n_sims, days)`` (one history per row), as the paper does.
    size : float
        The test's nominal size.

    Examples
    --------
    The paper's 5 % critical value for a year at alpha = 2.5 % is about -0.70 for a Normal model. It moves little for
    moderate tails (-0.74 for a Student-t with 5 degrees of freedom) but reaches -0.82 with 3 (its Table 4), so
    ``null`` should be the model's own distribution:

    >>> rng = np.random.default_rng(0)
    >>> q, es = 1.959964, 2.337803  # VaR and ES of N(0, 1) at alpha = 2.5 %
    >>> res = acerbi_szekely_z2(rng.standard_normal(250), q, es, 0.025)
    >>> round(res.critical_value, 2)
    -0.7
    """
    x = np.asarray(pnl, dtype=float)
    if x.ndim != 1 or x.size < 1:
        raise ValueError("pnl must be a non-empty one-dimensional series")
    v = np.broadcast_to(np.asarray(var, dtype=float), x.shape)
    e = np.broadcast_to(np.asarray(es, dtype=float), x.shape)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(v)) and np.all(np.isfinite(e))):
        raise ValueError("pnl, var and es must be finite")
    if np.any(e <= 0):
        raise ValueError("ES must be positive (a loss) on every day: Z2 divides by it")
    alpha = _check_probability("alpha", alpha)
    size = _check_probability("size", size)
    statistic = float(z2_statistic(x, v, e, alpha))
    if isinstance(null, str):
        if null not in ("normal", "student_t"):
            raise ValueError(f"null must be 'normal', 'student_t' or simulated P&L, got {null!r}")
        model_dof = None if null == "normal" else float(dof)
        if model_dof is not None and not model_dof > 1.0:
            raise ValueError("Student-t degrees of freedom must exceed 1 for a finite ES")
        sims = _check_count("n_sims", n_sims, 100)
        simulated = _simulate_z2(np.random.default_rng(seed), sims, x.size, alpha, model_dof)
    else:
        draws = np.asarray(null, dtype=float)
        if draws.ndim != 2 or draws.shape[1] != x.size or draws.shape[0] < 100:
            raise ValueError("simulated P&L must have shape (n_sims, days) with at least 100 histories")
        simulated = np.asarray(z2_statistic(draws, v, e, alpha))
    return Z2Result(
        statistic=statistic,
        critical_value=float(np.quantile(simulated, size)),
        p_value=float(np.mean(simulated < statistic)),
        size=size,
        n_sims=int(simulated.size),
    )


def power_z2(
    observations: int,
    alpha: float = 0.025,
    *,
    scale: float = 1.0,
    dof: float | None = None,
    size: float = 0.05,
    n_sims: int = 100_000,
    seed: int = 2014,
) -> float:
    """Rejection rate of Z2 for a standard Normal model when the true P&L is ``scale`` times a Normal or Student-t.

    The true P&L has standard deviation ``scale`` (a unit-variance Student-t with ``dof`` degrees of freedom, or a
    Normal for None); ``scale=1`` with ``dof=None`` gives the test's actual size, ``size`` up to the simulation error.
    Critical value and rejection rate are simulated from ``seed``.

    Examples
    --------
    A year at alpha = 2.5 % when the true volatility is 20 % above the model's, and when it is the model's but the
    tails are a Student-t's with 5 degrees of freedom:

    >>> round(power_z2(250, 0.025, scale=1.2), 2), round(power_z2(250, 0.025, dof=5), 2)
    (0.77, 0.16)
    """
    n = _check_count("observations", observations, 1)
    alpha, size = _check_probability("alpha", alpha), _check_probability("size", size)
    if not (math.isfinite(scale) and scale > 0):
        raise ValueError("scale must be a positive number")
    if dof is not None and not dof > 2.0:
        raise ValueError("Student-t degrees of freedom must exceed 2 for a finite variance")
    sims = _check_count("n_sims", n_sims, 100)
    rng = np.random.default_rng(seed)
    critical = np.quantile(_simulate_z2(rng, sims, n, alpha, None), size)
    alternative = _simulate_z2(rng, sims, n, alpha, None, truth=(scale, dof))
    return float(np.mean(alternative < critical))


# --- Ranking VaR and ES forecasts: FZ0 and Diebold-Mariano ----------------------------------------------------------


def fz0_loss(pnl: ArrayLike, var: ArrayLike, es: ArrayLike, alpha: float) -> np.ndarray:
    """The FZ0 loss of Patton, Ziegel and Chen (2019, eq. 6) of each day's VaR and ES forecast at tail ``alpha``.

    In the paper's notation, with ``v = -VaR`` and ``e = -ES`` the quantile and ES of the P&L ``Y``:
    ``L = -1{Y <= v} (v - Y) / (alpha e) + v / e + ln(-e) - 1``, here
    ``max(-Y - VaR, 0) / (alpha ES) + VaR / ES + ln(ES) - 1``. Its expectation is smallest at the true VaR and ES
    (it is strictly consistent for them jointly, and, when VaR and ES are both losses, the only loss of its family
    whose differences between models are homogeneous of degree zero), so averaging it over a backtest ranks models;
    :func:`diebold_mariano` tells whether a difference is significant. It needs a positive ES; the level of the loss
    depends on the currency unit, differences between models do not. The paper's Figure 2 puts the minimum of the
    expected loss of a standard Normal at (-1.64, -2.06) in its convention, as below.

    Examples
    --------
    For a standard Normal P&L at alpha = 5 % the expected loss is smallest at VaR 1.645 and ES 2.063:

    >>> y = ndtri((np.arange(100_000) + 0.5) / 100_000)  # the Normal's quantiles, an exact-enough sample
    >>> [round(float(fz0_loss(y, v, e, 0.05).mean()), 5) for v, e in [(1.645, 2.063), (1.5, 2.063), (1.645, 1.9)]]
    [0.72402, 0.73537, 0.72748]
    """
    y = np.asarray(pnl, dtype=float)
    v = np.asarray(var, dtype=float)
    e = np.asarray(es, dtype=float)
    alpha = _check_probability("alpha", alpha)
    if np.any(e <= 0):
        raise ValueError("ES must be positive (a loss): FZ0 takes its logarithm")
    return np.maximum(-y - v, 0.0) / (alpha * e) + v / e + np.log(e) - 1.0


def newey_west_variance(x: ArrayLike, lags: int) -> float:
    """Long-run variance of ``x`` with Bartlett weights (Newey and West 1987).

    ``gamma_0 + 2 sum_{k=1}^{lags} (1 - k / (lags + 1)) gamma_k`` with the autocovariances
    ``gamma_k = sum_t (x_t - mean)(x_{t-k} - mean) / T``; never negative.
    """
    d = np.asarray(x, dtype=float)
    lags = _check_count("lags", lags)
    if d.ndim != 1 or d.size < 2:
        raise ValueError("the long-run variance needs a series of at least two values")
    centred = d - d.mean()
    total = float(centred @ centred) / d.size
    for k in range(1, min(lags, d.size - 1) + 1):
        total += 2.0 * (1.0 - k / (lags + 1.0)) * float(centred[k:] @ centred[:-k]) / d.size
    return max(total, 0.0)


@dataclass(frozen=True)
class DieboldMarianoResult:
    #: ``mean(loss_a - loss_b) / sqrt(long-run variance / T)``: positive when model a's loss is larger.
    statistic: float
    #: Two-sided, from the standard Normal.
    p_value: float
    mean_difference: float
    lags: int


def diebold_mariano(loss_a: ArrayLike, loss_b: ArrayLike, *, lags: int | None = None) -> DieboldMarianoResult:
    """Diebold and Mariano's (1995) test that two forecasts have the same expected loss.

    The loss differential ``d_t = loss_a_t - loss_b_t`` is averaged and divided by its standard error from the
    Newey-West long-run variance (:func:`newey_west_variance`), which allows for autocorrelated losses (VaR losses
    cluster with volatility). The default number of lags is ``floor(4 (T / 100)^(2/9))``, 7 for 2,000 days: the
    first-step lag count of Newey and West's (1994) automatic selection for the Bartlett kernel (the smaller of the two
    they try), used here as a fixed rule. The statistic is asymptotically standard Normal under equal expected loss.

    Examples
    --------
    >>> rng = np.random.default_rng(3)
    >>> a = rng.standard_normal(1_000) ** 2
    >>> res = diebold_mariano(a + 0.2, a + rng.normal(0.0, 0.1, 1_000))
    >>> res.statistic > 10, res.p_value < 1e-10, res.lags
    (True, True, 6)
    """
    a, b = np.asarray(loss_a, dtype=float), np.asarray(loss_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        raise ValueError("the two loss series must be one-dimensional, of the same length, at least two days")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("losses must be finite")
    d = a - b
    lags = math.floor(4.0 * (d.size / 100.0) ** (2.0 / 9.0)) if lags is None else _check_count("lags", lags)
    mean = float(d.mean())
    variance = newey_west_variance(d, lags)
    if variance > 0.0:
        statistic = mean / math.sqrt(variance / d.size)
    else:  # one model's loss is the other's plus a constant
        statistic = 0.0 if mean == 0.0 else math.copysign(math.inf, mean)
    return DieboldMarianoResult(statistic, float(2.0 * stats.norm.sf(abs(statistic))), mean, lags)


# --- P&L attribution (MAR32.34 to MAR32.42) -------------------------------------------------------------------------


@dataclass(frozen=True)
class PLAResult:
    #: Spearman's rank correlation of the hypothetical and risk-theoretical P&L.
    spearman: float
    #: Kolmogorov-Smirnov distance between their empirical distributions.
    ks: float
    #: The asymptotic p-value of that distance (MAR32.42 quotes 0.264 at 0.09 and 0.055 at 0.12 for 250 days).
    ks_p_value: float
    zone: str


def pla_test(hpl: ArrayLike, rtpl: ArrayLike) -> PLAResult:
    """The P&L attribution test of MAR32: does the risk model's P&L (RTPL) track the desk's hypothetical P&L (HPL)?

    * Spearman correlation (MAR32.36 to MAR32.38): the Pearson correlation of the two series' ranks, the lowest P&L
      ranked 1 (ties share their average rank).
    * Kolmogorov-Smirnov distance (MAR32.39 to MAR32.41): the largest absolute difference between the two empirical
      distribution functions (each a step of ``1 / n`` per observation, 0.004 for MAR32's 250 days).
    * Zones (MAR32.42): green if the correlation is above 0.80 and the distance below 0.09; red if the correlation is
      below 0.70 or the distance above 0.12; amber otherwise.

    MAR32.35 asks for the latest 250 trading days; any equal-length pair of at least three days is accepted here.

    Examples
    --------
    >>> rng = np.random.default_rng(5)
    >>> hpl = rng.standard_normal(250)
    >>> res = pla_test(hpl, hpl + rng.normal(0.0, 0.3, 250))
    >>> res.zone, res.spearman > 0.9, res.ks < 0.09
    ('green', True, True)
    """
    h, r = np.asarray(hpl, dtype=float), np.asarray(rtpl, dtype=float)
    if h.shape != r.shape or h.ndim != 1 or h.size < 3:
        raise ValueError("HPL and RTPL must be one-dimensional series of the same length, at least three days")
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(r))):
        raise ValueError("HPL and RTPL must be finite")
    if np.ptp(h) == 0 or np.ptp(r) == 0:
        raise ValueError("a constant P&L series has no rank correlation")
    rank_h, rank_r = stats.rankdata(h), stats.rankdata(r)
    spearman = float(np.corrcoef(rank_h, rank_r)[0, 1])
    points = np.concatenate([h, r])
    cdf_h = np.searchsorted(np.sort(h), points, side="right") / h.size
    cdf_r = np.searchsorted(np.sort(r), points, side="right") / r.size
    ks = float(np.max(np.abs(cdf_h - cdf_r)))
    ks_p = float(kolmogorov(math.sqrt(h.size * r.size / (h.size + r.size)) * ks))
    return PLAResult(spearman, ks, ks_p, pla_zone(spearman, ks))


def pla_zone(spearman: float, ks: float) -> str:
    """MAR32.42's zone of a P&L attribution result: "green", "amber" or "red".

    Green needs a Spearman correlation above 0.80 and a KS distance below 0.09; a correlation below 0.70 or a
    distance above 0.12 is red; anything else is amber. (A value on a boundary is not "above" or "below" it.)

    Examples
    --------
    >>> pla_zone(0.85, 0.05), pla_zone(0.80, 0.05), pla_zone(0.85, 0.13)
    ('green', 'amber', 'red')
    """
    if spearman > PLA_GREEN[0] and ks < PLA_GREEN[1]:
        return "green"
    if spearman < PLA_RED[0] or ks > PLA_RED[1]:
        return "red"
    return "amber"


# --- A synthetic market history --------------------------------------------------------------------------------------


def _normal_scores(raw_t: np.ndarray, dof: float | None) -> np.ndarray:
    """Standard Normal scores ``Phi^-1(F(z))`` of standard Student-t draws (the draws themselves for a Normal)."""
    if dof is None:
        return raw_t
    # By symmetry from the upper tail, so that neither tail rounds its probability to 1.
    return np.sign(raw_t) * -ndtri(stats.t.sf(np.abs(raw_t), dof))


def synthetic_history(
    seed: int,
    n_days: int = 2520,
    *,
    spot: float = 5800.0,
    iv_level: float = 0.18,
    vol_premium: float = 1.2,
    vol_of_vol: float = 1.0,
    spot_vol_correlation: float = -0.7,
    iv_mean_reversion: float = 4.0,
    garch_alpha: float = 0.06,
    garch_beta: float = 0.90,
    dof: float | None = 5.0,
    drift: float = 0.0,
) -> pd.DataFrame:
    """A daily history of an index and of its one-month implied-vol index, labelled :data:`SYNTHETIC_LABEL`.

    Synthetic, not market data (``attrs["source"]`` is :data:`SYNTHETIC_LABEL`): a reproducible stand-in for an
    index and its VIX, to demonstrate and test backtests. Day ``t`` runs from close ``t - 1`` to close ``t``:

    * Implied vol: ``ln IV`` is an Ornstein-Uhlenbeck process, log-normal at every date,
      ``ln IV_t = ln IV_{t-1} + (k / 252) (ln iv_level - ln IV_{t-1}) + (eta / sqrt(252)) xi_t``, with ``k`` the
      ``iv_mean_reversion`` (a half-life of ``ln 2 / k`` years, 0.17 for k = 4), ``eta`` the ``vol_of_vol`` and
      ``iv_level`` its long-run median. The index starts at ``iv_level``; the stationary standard deviation of its
      log is ``eta / sqrt(2 k)`` (0.35 at the defaults).
    * Spot: daily log returns ``r_t = drift / 252 - h_t / 2 + sqrt(h_t) z_t`` with ``z_t`` a unit-variance Student-t
      with ``dof`` degrees of freedom (Normal for None) and the conditional variance ``h_t = (IV_{t-1} /
      vol_premium)^2 / 252 * g_t`` of two components: the slow one set by yesterday's implied vol (which is
      ``vol_premium`` times the next day's volatility when ``g`` is at its mean of 1) and a fast one, the GARCH(1,1)
      ``g_{t+1} = 1 - alpha - beta + alpha g_t z_t^2 + beta g_t`` (``alpha + beta`` = 0.96 by default, a half-life of
      17 days, since the slow component already moves with the index).
    * Spot-vol correlation: ``xi_t = rho s_t + sqrt(1 - rho^2) w_t``, with ``s_t`` the standard Normal score of
      ``z_t`` and ``w_t`` independent: a Gaussian copula, as :mod:`options_analyzer.risk` joins spot and vol moves.
      So the daily log change of the index is ``N(0, eta^2 / 252)`` up to the small mean-reversion term, and its
      correlation with the Normal score of the return's shock ``z_t`` (not of the return itself, whose scale varies)
      is ``rho`` (``spot_vol_correlation``).

    With ``vol_of_vol=0`` the index stays at ``iv_level`` and the returns are a GARCH(1,1) with Student-t shocks and
    long-run volatility ``iv_level / vol_premium``. The defaults are those of an equity index (an 18 % median
    implied vol, a 15 % baseline realised vol) and the Risk tab's index dynamics (vol of vol 100 %, correlation -0.7).

    Returns
    -------
    pandas.DataFrame
        ``n_days + 1`` rows indexed by ``day`` (0 to ``n_days``) with columns ``close``, ``log_return`` (NaN on day
        0), ``iv_index`` (annualised decimal) and ``true_vol``: the annualised standard deviation of the next day's
        return, ``sqrt(252 h_{t+1})``, known at that close (the truth that a model's volatility can be checked
        against). ``attrs`` holds ``source`` (:data:`SYNTHETIC_LABEL`) and the ``parameters``.

    Examples
    --------
    >>> history = synthetic_history(0, 500)
    >>> history.attrs["source"], len(history), list(history.columns)
    ('Synthetic — not market data', 501, ['close', 'log_return', 'iv_index', 'true_vol'])
    """
    n = _check_count("n_days", n_days, 1)
    numbers = {
        "spot": spot,
        "iv_level": iv_level,
        "vol_premium": vol_premium,
        "vol_of_vol": vol_of_vol,
        "spot_vol_correlation": spot_vol_correlation,
        "iv_mean_reversion": iv_mean_reversion,
        "garch_alpha": garch_alpha,
        "garch_beta": garch_beta,
        "drift": drift,
    }
    for name, value in numbers.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    if not (spot > 0 and iv_level > 0 and vol_premium > 0):
        raise ValueError("spot, iv_level and vol_premium must be positive")
    if vol_of_vol < 0:
        raise ValueError("vol_of_vol must be non-negative")
    if not 0 <= iv_mean_reversion < TRADING_DAYS:
        raise ValueError("iv_mean_reversion must be at least 0 and below 252 a year (it is applied day by day)")
    if not -1.0 <= spot_vol_correlation <= 1.0:
        raise ValueError("spot_vol_correlation must be between -1 and 1")
    if garch_alpha < 0 or garch_beta < 0 or garch_alpha + garch_beta >= 1:
        raise ValueError("GARCH needs alpha, beta >= 0 and alpha + beta < 1 (stationarity)")
    if dof is not None and not (math.isfinite(dof) and dof > 2):
        raise ValueError("Student-t degrees of freedom must exceed 2 for a finite variance")

    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(n) if dof is None else rng.standard_t(dof, n)
    z = raw if dof is None else raw * math.sqrt((dof - 2.0) / dof)
    rho = spot_vol_correlation
    xi = rho * _normal_scores(raw, dof) + math.sqrt(1.0 - rho * rho) * rng.standard_normal(n)

    # The index first (it does not depend on the spot's variance), then the spot day by day. The index's log distance
    # from its median, x_t = (1 - k / 252) x_{t-1} + (eta / sqrt(252)) xi_t from x_0 = 0, is a first-order filter of
    # the shocks; without vol of vol it stays 0 and the index is iv_level exactly.
    decay = 1.0 - iv_mean_reversion / TRADING_DAYS
    distance = signal.lfilter([vol_of_vol / math.sqrt(TRADING_DAYS)], [1.0, -decay], xi)
    iv = iv_level * np.exp(np.concatenate(([0.0], distance)))
    base_variance = (iv / vol_premium) ** 2 / TRADING_DAYS  # the slow component of the next day's variance

    alpha, beta = garch_alpha, garch_beta
    g = np.empty(n + 1)
    g[0] = 1.0
    for t in range(1, n + 1):
        g[t] = 1.0 - alpha - beta + (alpha * z[t - 1] ** 2 + beta) * g[t - 1]
    variance = base_variance * g  # h_{t+1}: the variance of day t + 1's return, known at close t
    log_return = np.full(n + 1, np.nan)
    log_return[1:] = drift / TRADING_DAYS - 0.5 * variance[:-1] + np.sqrt(variance[:-1]) * z
    close = spot * np.exp(np.concatenate(([0.0], np.cumsum(log_return[1:]))))

    frame = pd.DataFrame(
        {
            "close": close,
            "log_return": log_return,
            "iv_index": iv,
            "true_vol": np.sqrt(TRADING_DAYS * variance),
        },
        index=pd.RangeIndex(n + 1, name="day"),
    )
    frame.attrs["source"] = SYNTHETIC_LABEL
    frame.attrs["parameters"] = {"seed": seed, "n_days": n, **numbers, "dof": dof}
    return frame


# --- Rolling backtest ---------------------------------------------------------------------------------------------

#: A VaR model for :func:`rolling_backtest`: given the position, its market at a close, and the history it may use
#: (the last ``window`` daily log returns of the spot and log changes of the implied-vol index, oldest first), the
#: model's simulated one-day P&L of one unit of the position (scenarios, each equally likely).
VaRModel = Callable[[Strategy, Market, np.ndarray, np.ndarray], np.ndarray]


def monte_carlo_model(config: VaRConfig) -> VaRModel:
    """The app's VaR engine as a backtest model: :func:`options_analyzer.risk.value_at_risk` with ``config``.

    The model ignores the history: its volatility is the position's market volatility (the implied vol that
    ``build_position`` sets), its implied-vol moves those of ``config`` (none when ``vol_of_vol`` is 0). ``config``
    must have a one-day horizon; its confidence is not used (the backtest reads every confidence from the same
    simulated P&L), and its seed is used every day, as the app does.
    """
    if int(config.horizon_days) != 1:
        raise ValueError("a daily backtest needs a one-day horizon")

    def model(strategy: Strategy, market: Market, returns: np.ndarray, iv_changes: np.ndarray) -> np.ndarray:
        return value_at_risk(strategy, market, config).pnl

    return model


def _ewma_scaled(returns: np.ndarray, decay: float) -> np.ndarray:
    """Hull and White's (1998) volatility updating: each return times today's EWMA volatility over its day's.

    The EWMA variance known before day ``i`` is ``s2_i = decay s2_{i-1} + (1 - decay) r_{i-1}^2``, started at the
    window's mean square; today's is the one after the last day.
    """
    squares = returns**2
    start = float(squares.mean())
    if start == 0.0:
        return returns
    after = signal.lfilter([1.0 - decay], [1.0, -decay], squares, zi=[decay * start])[0]
    before = np.concatenate(([start], after[:-1]))
    return returns * np.sqrt(after[-1] / before)


def historical_model(
    *,
    scaling: str = "ewma",
    decay: float = 0.94,
    implied_vol: bool = True,
    vol_tenor_decay: float = DEFAULT_VOL_TENOR_DECAY,
) -> VaRModel:
    """Historical simulation by full revaluation on the window's joint spot and implied-vol moves.

    Each past day's log return moves today's spot and its log change of the implied-vol index multiplies every leg's
    implied vol (damped for longer tenors by ``vol_tenor_decay``, as in :mod:`options_analyzer.risk`; with
    ``implied_vol=False`` implied vol stays put), one trading day later. With ``scaling="ewma"`` each return is first
    rescaled by today's EWMA volatility over its own day's (Hull and White 1998, RiskMetrics decay 0.94); with
    ``"none"`` the returns are used as they were. Implied-vol changes are not rescaled.
    """
    if scaling not in ("ewma", "none"):
        raise ValueError(f"scaling must be 'ewma' or 'none', got {scaling!r}")
    decay = _check_probability("decay", decay)
    tenor_decay = vol_tenor_decay if implied_vol else 0.0

    def model(strategy: Strategy, market: Market, returns: np.ndarray, iv_changes: np.ndarray) -> np.ndarray:
        moves = _ewma_scaled(returns, decay) if scaling == "ewma" else returns
        vol_multiplier = np.exp(iv_changes) if implied_vol else 1.0
        later = position_value(
            strategy,
            market,
            market.spot * np.exp(moves),
            ONE_TRADING_DAY,
            vol_multiplier=vol_multiplier,
            vol_tenor_decay=tenor_decay,
        )
        return np.asarray(later, dtype=float) - float(position_value(strategy, market))

    return model


@dataclass(frozen=True)
class RollingBacktest:
    #: One row per model and confidence (see :func:`rolling_backtest`).
    summary: pd.DataFrame
    #: One row per day, model and confidence: ``day`` (the day of the P&L), ``model``, ``confidence``, ``pnl``,
    #: ``var``, ``es``, ``exception`` and ``fz0`` (the FZ0 loss of that day's VaR and ES).
    daily: pd.DataFrame
    #: What the history is (:data:`SYNTHETIC_LABEL` for :func:`synthetic_history`).
    source: str


def _series(name: str, values: ArrayLike) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not (np.all(np.isfinite(array)) and np.all(array > 0)):
        raise ValueError(f"{name} must be positive and finite on every day")
    return array


def rolling_backtest(
    build_position: Callable[[float, float], tuple[Strategy, Market]],
    closes: ArrayLike,
    iv_index: ArrayLike,
    models: Mapping[str, VaRModel],
    window: int = 500,
    confidences: Sequence[float] = (0.99, 0.975),
    *,
    multiplier: float = 1.0,
    vol_tenor_decay: float = DEFAULT_VOL_TENOR_DECAY,
    z2_sims: int = 2_000,
    seed: int = 2014,
    source: str | None = None,
) -> RollingBacktest:
    """Backtest VaR models on a static position's hypothetical P&L, day by day.

    At each close ``t`` from ``window`` on, ``build_position(close_t, iv_t)`` returns the position and its market
    (whose spot must be that close); every model forecasts the position's one-day P&L from the last ``window`` days
    (:data:`VaRModel`); and the hypothetical P&L is the position held unchanged to the next close: re-priced at the
    next close's spot, with every leg's implied vol multiplied by the index's change (damped for longer tenors by
    ``vol_tenor_decay``, as :mod:`options_analyzer.risk` treats a one-month vol move) and one trading day less to
    expiry, minus its value at the close. P&L, VaR and ES are in dollars after ``multiplier``.

    Parameters
    ----------
    build_position : callable
        ``(spot, implied_vol) -> (Strategy, Market)``, called once per day.
    closes, iv_index : array_like
        Daily closes of the underlying and of its implied-vol index (annualised decimal), the same days.
    models : mapping
        Model name to :data:`VaRModel`, e.g. :func:`monte_carlo_model` or :func:`historical_model`.
    window : int
        Days of history each model sees; the backtest runs over the ``len(closes) - 1 - window`` days after it.
    confidences : sequence of float
        VaR (and ES) confidences to test, read from the same scenarios.
    z2_sims, seed : int
        Histories simulated under each model for the critical value of Z2, by drawing every day's P&L from that
        day's scenarios (Acerbi and Szekely's procedure), and the seed of those draws.
    source : str, optional
        What the history is; by default ``closes.attrs["source"]`` when ``closes`` has one (the columns of
        :func:`synthetic_history` do), else empty.

    Returns
    -------
    RollingBacktest
        ``summary`` has one row per model and confidence: ``model``, ``confidence``, ``days``, ``exceptions``,
        ``expected``, ``kupiec_p`` (chi-square), ``binomial_tail`` (``P(X >= exceptions)``), ``independence_p`` and
        ``christoffersen_p`` (LR_ind and LR_cc), ``zone`` (the traffic light of all the days, at the model's
        confidence), ``last_250_exceptions`` and ``last_250_zone`` (MAR32's year), ``z2`` with its 5 %
        ``z2_critical`` and ``z2_p`` (at the ES tail ``1 - confidence``; NaN if an ES is not positive), ``fz0`` (the
        mean FZ0 loss) and ``history`` (the source). ``daily`` holds the series behind it.
    """
    spot_closes = _series("closes", closes)
    iv_closes = _series("iv_index", iv_index)
    if spot_closes.size != iv_closes.size:
        raise ValueError("closes and iv_index must cover the same days")
    window = _check_count("window", window, 2)
    days = spot_closes.size - 1 - window
    if days < 1:
        raise ValueError(f"the history ({spot_closes.size} closes) needs two closes more than the window ({window})")
    levels = [_check_probability("confidence", c) for c in confidences]
    if not levels or any(c <= 0.5 for c in levels):
        raise ValueError("confidences must be between 0.5 and 1")
    if not models:
        raise ValueError("no model to backtest")
    if not (math.isfinite(multiplier) and multiplier > 0):
        raise ValueError("multiplier must be a positive number")
    draws = _check_count("z2_sims", z2_sims, 100)
    if source is None:
        source = str(getattr(closes, "attrs", {}).get("source", ""))

    returns = np.diff(np.log(spot_closes))
    iv_changes = np.diff(np.log(iv_closes))
    names = list(models)
    # The highest confidence needs a scenario in its tail: with fewer, its VaR would be a lower confidence's.
    min_tail = 1.0 - max(levels) + 1e-12
    rng = np.random.default_rng(seed)
    pnl = np.empty(days)
    var = np.empty((len(names), len(levels), days))
    es = np.empty_like(var)
    # Z2 under each model, accumulated day by day: sum_t X I / ES for each simulated history.
    z2_sums = np.zeros((len(names), len(levels), draws))
    for i in range(days):
        t = window + i  # the close the forecast is made at
        strategy, market = build_position(float(spot_closes[t]), float(iv_closes[t]))
        if not math.isclose(market.spot, spot_closes[t], rel_tol=1e-12):
            raise ValueError("build_position must return the market at the close it is given")
        today = float(position_value(strategy, market))
        later = position_value(
            strategy,
            market,
            spot_closes[t + 1],
            ONE_TRADING_DAY,
            vol_multiplier=iv_closes[t + 1] / iv_closes[t],
            vol_tenor_decay=vol_tenor_decay,
        )
        pnl[i] = multiplier * (float(later) - today)
        past = slice(t - window, t)  # returns[k] is day k + 1's, so these are days t - window + 1 to t
        for m, name in enumerate(names):
            scenarios = multiplier * np.asarray(models[name](strategy, market, returns[past], iv_changes[past]), float)
            if scenarios.ndim != 1 or not np.all(np.isfinite(scenarios)) or scenarios.size * min_tail < 1:
                raise ValueError(
                    f"model {name!r} must return finite one-day P&L scenarios, at least {math.ceil(1 / min_tail)} "
                    f"for a {max(levels):.4g} VaR (it returned {scenarios.size})"
                )
            picks = scenarios[rng.integers(0, scenarios.size, draws)]
            for c, level in enumerate(levels):
                v, e, _ = _var_es(scenarios, level)
                var[m, c, i], es[m, c, i] = v, e
                if e > 0:
                    z2_sums[m, c] += np.where(picks <= -v, picks / e, 0.0)

    rows, frames = [], []
    for m, name in enumerate(names):
        for c, level in enumerate(levels):
            p = 1.0 - level
            hits = pnl < -var[m, c]
            exceptions = int(hits.sum())
            kupiec = kupiec_pof(exceptions, days, p)
            markov = christoffersen(hits, p) if days >= 2 else None
            last = hits[-250:]
            positive = bool(np.all(es[m, c] > 0))
            z2 = z2_critical = z2_p = math.nan
            fz0 = np.full(days, math.nan)
            if positive:
                z2 = float(z2_statistic(pnl, var[m, c], es[m, c], p))
                simulated = z2_sums[m, c] / (days * p) + 1.0
                z2_critical = float(np.quantile(simulated, 0.05))
                z2_p = float(np.mean(simulated < z2))
                fz0 = fz0_loss(pnl, var[m, c], es[m, c], p)
            rows.append(
                {
                    "model": name,
                    "confidence": level,
                    "days": days,
                    "exceptions": exceptions,
                    "expected": days * p,
                    "kupiec_p": kupiec.p_value,
                    "binomial_tail": kupiec.binomial_tail,
                    "independence_p": markov.p_ind if markov else math.nan,
                    "christoffersen_p": markov.p_cc if markov else math.nan,
                    "zone": traffic_light(exceptions, days, coverage=level).zone,
                    "last_250_exceptions": int(last.sum()),
                    "last_250_zone": traffic_light(int(last.sum()), last.size, coverage=level).zone,
                    "z2": z2,
                    "z2_critical": z2_critical,
                    "z2_p": z2_p,
                    "fz0": float(fz0.mean()) if positive else math.nan,
                    "history": source,
                }
            )
            frames.append(
                pd.DataFrame(
                    {
                        "day": np.arange(window + 1, window + 1 + days),
                        "model": name,
                        "confidence": level,
                        "pnl": pnl,
                        "var": var[m, c],
                        "es": es[m, c],
                        "exception": hits,
                        "fz0": fz0,
                    }
                )
            )
    summary = pd.DataFrame(rows)
    daily = pd.concat(frames, ignore_index=True)
    for frame in (summary, daily):
        frame.attrs["source"] = source
    return RollingBacktest(summary, daily, source)
