"""A realistic synthetic option chain: the default demo data, and a test oracle with known true values.

:func:`sample_chain` lists an SPX-like chain (``style="index"``: European, AM-settled monthlies under SPX and PM-settled
weeklies under SPXW) or an SPY-like one (``style="etf"``: American, PM-settled, all under SPY), and prices it from a
model whose every input is known:

* Expiries: the Friday weeklies of the next six weeks, the next six third-Friday monthlies and the March, June,
  September and December quarterlies out to two years. An expiry on an NYSE holiday moves to the trading day before.
* Strikes: near the money, every strike at the exchange's interval (:func:`strike_interval`: 5 points for an index
  from 1,000, like SPX; $1 for an ETF from $25, like SPY), over 1.5 ATM standard deviations each side (at least five
  strikes each side, at most 100 in all); further out, two coarser bands (to 80 and 60 strikes) whose step is the
  finest multiple of the interval (1, 2 or 5 times a power of ten) that keeps each band within its limit. Every
  strike is on the exchange's grid, and an expiry lists at most 243 strikes whatever the spot: with at most 18
  expiries (6 weeklies, 6 monthlies and 6 later quarterlies), a chain has at most 8,748 rows. Strikes from 100,000
  up, which an OCC symbol cannot carry, are not listed (a spot near 50,000 at a high rate reaches them).
* Time: years to the settlement time from :mod:`options_analyzer.clock` (minutes / 525,600).
* Volatility: an SSVI surface (:class:`options_analyzer.vol_surface.SSVIParams`) evaluated in variance time
  (:class:`options_analyzer.clock.VarianceClock`, weekends and holidays at a tenth of a trading day), so the total
  implied variance at forward log-moneyness ``k`` is ``w(k, tau)`` and the implied vol is ``sqrt(w / T)`` at the
  calendar time ``T``. SSVI with ``eta (1 + |rho|) <= 2`` (and ``gamma <= 1/2``) has no butterfly or calendar
  arbitrage, and a variance clock only re-times it, so the European values have no static arbitrage.
* Rates: discounting and forwards at a box rate, the Treasury zero rate plus 45 bp (SPX box spreads finance above
  Treasuries). ``rate_curve`` is a flat rate or a function of the time to expiry until a curve module exists.
* Dividends: a continuous yield, or cash dividends ``(ex-date, $ per share)`` that drop the spot at the open of the
  ex-date. Cash dividends use the escrowed-dividend model: the share price is a lognormal escrowed spot (the spot less
  the present value of the dividends paid before expiry) plus the present value of the dividends still to come. The
  model gets the forward exactly and its European values are Black-76 at that forward, as the surface states, but it
  puts the volatility on the escrowed spot rather than on the share price. A spot-drop model (the dividend a step down
  in the share price, whose volatility is the input) at the same vol values options higher: on the SPY-like chain of
  the tests, up to $0.18 more for the puts with a dividend before expiry (QuantLib's finite-difference engines), about
  as much as for the European ones, and more than half the quoted spread on 40% of them. The planned dividends engine
  (a finite-difference model with the dividend step on the spot) replaces the escrowed model.
* Prices: Black-76 for index options. ETF options are American. With a continuous yield, puts and calls come from
  the Leisen-Reimer binomial tree of :mod:`options_analyzer.binomial` with 201 steps. With cash dividends, puts come
  from a Leisen-Reimer tree of 201 steps on the escrowed spot that exercises at the full share price (Hull's
  escrowed-dividend tree), so a deep in-the-money put is worth its exercise value ``K - S`` where exercising is
  optimal. (Exercising at the escrowed spot instead, as the binomial engine would on that spot, overvalues such puts
  by up to the present value of the dividends to expiry: $3.56 at the June 2027 expiry of the SPY-like chain.) The
  tree agrees with QuantLib's escrowed-dividend finite-difference engine within $0.07 on the SPY-like chain of the
  tests, and within half the quoted spread for dividends of up to 1.7% of the spot a quarter: its 201 steps exercise
  up to a step after an ex-date, which costs most for deep puts at two years ($0.20 at $10 a quarter on a $600 ETF).
  Every American value is then floored at the value of the same option with a shorter expiry, whose exercise it can
  copy: that keeps the values rising with maturity and lifts a long-dated put towards its finer short-dated
  neighbour, by up to $0.21 in the tests (the largest error left is then $0.17). Calls with cash dividends use
  Black's approximation, the largest of the exercise value, the European value to expiry and the European value to
  just before each ex-date, with no tree needed because the escrowed spot pays nothing. The European to just before
  an ex-date is priced on the surface itself, at the forward and variance time of that moment, so it is the same for
  every expiry after the ex-date and the American calls gain value with maturity, as they must.
* Quotes: a half-spread of a few hundredths of a vol point times vega, widening with ``|k|`` (in ATM standard
  deviations) and with ``T``, plus 0.1% of the value, and at least one tick. Ticks are $0.05 below $3 and $0.10 from $3
  (SPX) or $0.01 (SPY is penny-quoted). Bids are rounded down and asks up, so the model value always lies inside the
  quote, and the quotes inherit the model's freedom from static arbitrage: no vertical, butterfly or (for the
  American ETF options, whose values rise with maturity) calendar spread is free. The tests check all three. Calls
  and puts are quoted at every strike; the far wings have a zero bid (an offer only).
* Noise (``noise=True``, the default): each spread is scaled by ``exp(0.2 Z)``, each mid moved by up to half the
  half-spread from the model value, and a third of the wing quotes cheaper than four ticks lose their bid, all from a
  generator seeded with ``seed``; the output is the same for the same inputs. ``noise=False`` quotes the model value
  as the mid.

Everything is labelled :data:`SYNTHETIC_LABEL`.
"""

from __future__ import annotations

import itertools
import math
import numbers
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .. import binomial
from ..clock import (
    MARKET_OPEN,
    SettlementStyle,
    VarianceClock,
    previous_trading_day,
    settlement_time,
    to_new_york,
    year_fraction,
)
from ..vol_surface import SSVIParams
from .schema import chain_frame, occ_symbol

__all__ = [
    "BOX_SPREAD",
    "DEFAULT_CLOCK",
    "DEFAULT_SSVI",
    "MAX_SPOT",
    "SYNTHETIC_LABEL",
    "SyntheticChain",
    "sample_chain",
    "strike_interval",
]

SYNTHETIC_LABEL = "Synthetic — not market data"
#: An equity-index smile in a calm market: ATM vol rising from 15% at the front to 19% in the long run, steep negative
#: skew. eta (1 + |rho|) = 1.87 <= 2, so the surface has no static arbitrage.
DEFAULT_SSVI = SSVIParams(atm_vol_short=0.15, atm_vol_long=0.19, mean_reversion=1.5, rho=-0.7, eta=1.1, gamma=0.4)
#: Weekends and holidays at a tenth of a trading day's variance. French and Roll (1986) found the volatility from a
#: Friday close to a Monday close only 10% to 20% above a weekday's: a variance 1.2 to 1.4 times a weekday's, which is
#: 0.1 to 0.2 per weekend day.
DEFAULT_CLOCK = VarianceClock(weekend_weight=0.1, holiday_weight=0.1)
#: Box rate minus the Treasury zero rate (45 bp).
BOX_SPREAD = 0.0045
TREE_STEPS = 201
WEEKLY_WEEKS = 6
MONTHLIES = 6
HORIZON_DAYS = 730
#: Valuation dates whose expiries (out to two years) fall in the NYSE calendar and OCC's two-digit years.
_FIRST_VALUATION, _LAST_VALUATION = date(2000, 1, 1), date(2097, 12, 31)
#: Half-spread in vol at the money for a short expiry, by style.
_SPREAD_VOL = {"index": 0.0005, "etf": 0.0002}
_SPREAD_VALUE = 0.001  # plus this fraction of the option value (deep in-the-money options quote wide in dollars)
_MIN_TICK = {"index": 0.05, "etf": 0.01}
#: The exchange's strike interval near the money, by style: (the lowest spot it applies from, interval), highest
#: first. Cboe lists index options no closer than 5 points apart (SPX; Rule 4.13), with $1 strikes for small indices
#: such as XSP (its product specification); SPY, QQQ and IWM list $1 strikes, and low-priced ETFs $0.50 ones. Below
#: the lowest spot the grid holds too few strikes to quote, so :func:`sample_chain` refuses it.
_INTERVALS: dict[str, tuple[tuple[float, float], ...]] = {
    "index": ((1_000.0, 5.0), (100.0, 1.0)),
    "etf": ((25.0, 1.0), (5.0, 0.5)),
}
#: Largest spot: at rates of a few percent the furthest strikes, about 1.6 times the spot, stay below OCC's limit.
MAX_SPOT = 50_000.0
#: OCC symbols write the strike in eight digits of thousandths: strikes below 100,000.
_OCC_STRIKE_LIMIT = 100_000.0
#: The near-the-money band lists at least this many strikes each side of the forward (for low-priced underlyings).
_NEAR_STRIKES = 5
#: Strike bands around the forward: (the most strikes the band may hold, its lower edge, its upper edge), each edge in
#: ln(K/F) as (a multiple of the ATM standard deviation sqrt(w), the smallest, the largest). The first band lists every
#: strike at the exchange's interval and is cut to its limit around the forward; each further band uses the finest
#: multiple of the previous band's step that keeps it within its limit (see :func:`_strikes`).
_BANDS = (
    (100, (1.5, 0.02, 0.04), (1.5, 0.02, 0.04)),
    (80, (4.0, 0.08, 0.35), (2.5, 0.05, 0.2)),
    (60, (8.0, 0.15, 1.0), (4.0, 0.08, 0.4)),
)

Style = Literal["index", "etf"]


@dataclass(frozen=True, eq=False)
class SyntheticChain:
    """A synthetic chain and the model values behind it.

    Attributes
    ----------
    quotes : pandas.DataFrame
        The chain in the canonical schema (:data:`options_analyzer.market_data.schema.CHAIN_COLUMNS`); its
        ``attrs["source"]`` is :data:`SYNTHETIC_LABEL`.
    theoretical : pandas.DataFrame
        Row by row with ``quotes`` (same index): ``value``, the model value ($ per share), and ``iv``, the model's
        implied vol at that strike (decimal; for American options, the volatility the tree used).
    expiries : pandas.DataFrame
        One row per (expiry, root): ``expiry``, ``root``, ``settlement``, ``settles_at`` (New York time),
        ``maturity`` (calendar years, minutes / 525,600), ``variance_time`` (years), ``rate`` (the box rate, a
        continuously compounded zero rate), ``discount_factor``, ``dividends_pv`` ($ per share, cash dividends paid
        before settlement), ``forward`` and ``atm_vol`` (the implied vol at the forward, decimal).
    spot, as_of, style
        The inputs.
    label : str
        :data:`SYNTHETIC_LABEL`.
    """

    quotes: pd.DataFrame
    theoretical: pd.DataFrame
    expiries: pd.DataFrame
    spot: float
    as_of: datetime
    style: Style
    label: str = SYNTHETIC_LABEL


def _third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(4 - first.weekday()) % 7 + 14)


def _schedule(today: date, style: Style) -> list[tuple[date, str, SettlementStyle]]:
    """``(expiry, root, settlement)`` of the listed expiries from ``today``, in date order."""
    months = [
        _third_friday(today.year + (today.month - 1 + i) // 12, (today.month - 1 + i) % 12 + 1) for i in range(26)
    ]
    months = [day for day in months if day >= today]
    horizon = today + timedelta(days=HORIZON_DAYS)
    monthly = set(months[:MONTHLIES]) | {day for day in months if day.month % 3 == 0 and day <= horizon}
    first_friday = today + timedelta(days=(4 - today.weekday()) % 7)
    weekly = {first_friday + timedelta(weeks=i) for i in range(WEEKLY_WEEKS)}
    listed: set[tuple[date, str, SettlementStyle]]
    if style == "index":  # SPX lists the third-Friday monthlies (AM), SPXW the other Fridays (PM)
        listed = {(previous_trading_day(day), "SPX", "AM") for day in monthly}
        listed |= {(previous_trading_day(day), "SPXW", "PM") for day in weekly - set(months)}
    else:
        listed = {(previous_trading_day(day), "SPY", "PM") for day in monthly | weekly}
    return sorted(listed)


def strike_interval(spot: float, style: Style = "index") -> float:
    """The exchange's strike interval near the money for an underlying at ``spot``, in $ (index points).

    5 for an index from 1,000 (SPX) and 1 below (XSP); $1 for an ETF from $25 (SPY) and $0.50 below.

    Parameters
    ----------
    spot : float
        Underlying price, $ (index points); at least 100 for an index and $5 for an ETF.
    style : {"index", "etf"}, default "index"

    Raises
    ------
    ValueError
        If ``spot`` is below the style's lowest spot or above :data:`MAX_SPOT`.

    Examples
    --------
    >>> strike_interval(5999.0), strike_interval(600.0), strike_interval(600.0, "etf"), strike_interval(20.0, "etf")
    (5.0, 1.0, 1.0, 0.5)
    """
    if style not in _INTERVALS:
        raise ValueError(f"style must be 'index' or 'etf', got {style!r}")
    spot = float(spot)
    tiers = _INTERVALS[style]
    lowest = tiers[-1][0]
    if not (math.isfinite(spot) and lowest <= spot <= MAX_SPOT):
        raise ValueError(
            f"spot must be from {lowest:,g} to {MAX_SPOT:,g} for the {style} style, got {spot!r}: below, the strike "
            "grid holds too few strikes to quote; above, the furthest strikes pass OCC's limit of 100,000"
        )
    return next(interval for start, interval in tiers if spot >= start)


def _multiples() -> Iterator[int]:
    """1, 2, 5, 10, 20, 50, 100, ..."""
    for exponent in itertools.count():
        for digit in (1, 2, 5):
            yield digit * 10**exponent


def _strikes(forward: float, interval: float, sd: float) -> np.ndarray:
    """Listed strikes around ``forward``: every strike at ``interval`` near the money, coarser further out.

    Each band of :data:`_BANDS` spans its edges in ``sd`` (the ATM standard deviation of ln K/F) and holds at most
    its limit of strikes (plus one where both edges fall on a strike): the first, widened to at least
    :data:`_NEAR_STRIKES` strikes each side of the forward, by keeping only the strikes nearest the forward, the others
    by stepping at the finest multiple of the previous band's step that fits. Each step divides the next, so a coarser
    band adds no strike between those of a finer one.
    """
    grids = []
    multiple = 1
    for band, (limit, (below, low_min, low_max), (above, high_min, high_max)) in enumerate(_BANDS):
        low = forward * math.exp(-min(max(below * sd, low_min), low_max))
        high = forward * math.exp(min(max(above * sd, high_min), high_max))
        if band == 0:  # at least _NEAR_STRIKES each side of the forward, at most `limit` in all
            low, high = min(low, forward - _NEAR_STRIKES * interval), max(high, forward + _NEAR_STRIKES * interval)
            half_width = 0.5 * limit * interval
            low, high = max(low, forward - half_width), min(high, forward + half_width)
        else:
            multiple = next(
                m for m in _multiples() if m % multiple == 0 and high - low <= limit * m * interval * (1 + 1e-12)
            )
        width = multiple * interval
        grids.append(np.arange(math.ceil(low / width - 1e-9), math.floor(high / width + 1e-9) + 1) * width)
    strikes = np.unique(np.round(np.concatenate(grids), 3))
    return strikes[strikes > 0]


def _black(forward: float, strike: np.ndarray, variance: np.ndarray, discount: float, call: bool) -> np.ndarray:
    """Black-76 values with total variance ``variance`` (vectorised over strikes)."""
    sd = np.sqrt(variance)
    d1 = np.log(forward / strike) / sd + 0.5 * sd
    if call:
        return discount * (forward * ndtr(d1) - strike * ndtr(d1 - sd))
    return discount * (strike * ndtr(sd - d1) - forward * ndtr(-d1))


def _round_down(price: np.ndarray, style: Style) -> np.ndarray:
    """Bids: the next tick strictly below ``price``, with the tick of the rounded price (index: $0.05 below $3)."""
    if style == "etf":
        bid = np.floor(price / 0.01 - 1e-9) * 0.01
    else:
        bid = np.floor(price / 0.05 - 1e-9) * 0.05
        bid = np.where(bid < 3.0 - 1e-9, bid, np.floor(price / 0.10 - 1e-9) * 0.10)
    return np.round(np.maximum(bid, 0.0), 2)


def _round_up(price: np.ndarray, style: Style) -> np.ndarray:
    """Asks: the next tick strictly above ``price``, with the tick of the rounded price (index: $0.10 from $3)."""
    if style == "etf":
        ask = np.ceil(price / 0.01 + 1e-9) * 0.01
    else:
        ask = np.ceil(price / 0.05 + 1e-9) * 0.05
        ask = np.where(ask < 3.0 - 1e-9, ask, np.ceil(price / 0.10 + 1e-9) * 0.10)
    return np.round(ask, 2)


def _rate_function(rate_curve: float | Callable[[float], float]) -> Callable[[float], float]:
    if callable(rate_curve):
        curve = rate_curve
    else:
        flat = float(rate_curve)
        curve = lambda T: flat  # noqa: E731

    def rate(T: float) -> float:
        value = float(curve(T))
        if not (math.isfinite(value) and abs(value) < 1.0):
            raise ValueError(f"the rate curve must give a finite rate below 100% in size, got {value!r} at T = {T:g}")
        return value

    return rate


def _cash_dividends(dividends: Iterable[tuple[date, float]], as_of: datetime) -> list[tuple[datetime, float]]:
    """Cash dividends after ``as_of``: (the open of the ex-date in New York, $ per share), in date order."""
    paid = []
    for item in dividends:
        try:
            ex_date, cash = item
        except (TypeError, ValueError):
            raise ValueError(f"each dividend must be an (ex-date, cash per share) pair, got {item!r}") from None
        if isinstance(ex_date, datetime):
            ex_date = ex_date.date()
        if not isinstance(ex_date, date):
            raise TypeError(f"a dividend ex-date must be a date, got {type(ex_date).__name__}")
        cash = float(cash)
        if not (math.isfinite(cash) and cash >= 0):
            raise ValueError(f"a cash dividend must be finite and non-negative, got {cash!r}")
        drop = to_new_york(datetime.combine(ex_date, MARKET_OPEN))
        if drop > as_of:
            paid.append((drop, cash))
    return sorted(paid)


class _ExDate(NamedTuple):
    """A cash dividend seen from the valuation time."""

    drop: datetime  # the open of the ex-date, New York time
    years: float  # calendar years to the ex-date (minutes / 525,600)
    pv: float  # present value of the cash, $ per share
    forward: float  # forward to just before the ex-date (the spot less the dividends paid before it), $
    discount_factor: float  # to the ex-date
    variance_time: float  # years of variance time to the ex-date


def _ex_dates(
    cash: list[tuple[datetime, float]],
    spot: float,
    as_of: datetime,
    rate: Callable[[float], float],
    box_spread: float,
    variance_clock: VarianceClock,
) -> list[_ExDate]:
    """Each cash dividend's present value, and the forward, discount factor and variance time just before it."""
    discounted = []
    for drop, amount in cash:
        t = year_fraction(as_of, drop)
        discount = math.exp(-(rate(t) + box_spread) * t)
        discounted.append((drop, t, amount * discount, discount))
    ex_dates = []
    for drop, t, pv, discount in discounted:
        before = spot - sum(other_pv for other, _, other_pv, _ in discounted if other < drop)
        ex_dates.append(_ExDate(drop, t, pv, before / discount, discount, variance_clock.variance_time(as_of, drop)))
    return ex_dates


def _peizer_pratt(z: np.ndarray, n: int) -> np.ndarray:
    """Peizer-Pratt method-2 inversion of the normal distribution, as in the Leisen-Reimer tree."""
    denom = n + 1.0 / 3.0 + 0.1 / (n + 1.0)
    return 0.5 + np.copysign(0.5, z) * np.sqrt(1.0 - np.exp(-((z / denom) ** 2) * (n + 1.0 / 6.0)))


def _escrowed_american_puts(
    escrowed: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    iv: np.ndarray,
    dividends: Sequence[tuple[float, float]],
    steps: int = TREE_STEPS,
) -> np.ndarray:
    """American puts in the escrowed-dividend model, exercised at the full share price (Hull's tree).

    The escrowed spot ``S*`` (the spot less the present value of the dividends paid before expiry) follows a
    Leisen-Reimer tree of ``steps`` (odd) steps at the rate ``r`` and each strike's vol, as in
    :func:`options_analyzer.binomial.price`, vectorised over the strikes (where its probabilities round to 0 or 1,
    the drift-centred lattice the binomial module falls back to). At a node at time ``t`` the share price is
    ``S*`` plus the dividends still to go ex after ``t``, valued at ``t``; exercise pays the strike less that share
    price, which at the root is ``K - S``. ``dividends`` holds ``(years to the ex-date, present value)`` pairs.
    """
    K, sigma = strikes[:, None], iv[:, None]
    dt = T / steps
    growth = math.exp(r * dt)
    sd = sigma * math.sqrt(T)
    d1 = (np.log(escrowed / K) + r * T) / sd + 0.5 * sd
    p, p_star = _peizer_pratt(d1 - sd, steps), _peizer_pratt(d1, steps)
    centred = np.minimum(np.minimum(p, 1.0 - p), np.minimum(p_star, 1.0 - p_star)) > 1e-12
    p_lr = np.where(centred, p, 0.5)
    u = np.where(centred, growth * p_star / p_lr, np.exp((r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt)))
    d = np.where(centred, (growth - p_lr * u) / (1.0 - p_lr), np.exp((r - 0.5 * sigma**2) * dt - sigma * math.sqrt(dt)))
    p = np.where(centred, p, (growth - d) / (u - d))
    log_u, log_d = np.log(u), np.log(d)
    ups = np.arange(steps + 1)[None, :]
    values = np.maximum(K - escrowed * np.exp(ups * log_u + (steps - ups) * log_d), 0.0)
    for i in range(steps - 1, -1, -1):
        values = (p * values[:, 1:] + (1.0 - p) * values[:, :-1]) / growth
        t = i * dt
        to_come = sum(pv for years, pv in dividends if years > t) * math.exp(r * t)
        share = escrowed * np.exp(ups[:, : i + 1] * log_u + (i - ups[:, : i + 1]) * log_d) + to_come
        values = np.maximum(values, K - share)
    return values[:, 0]


def sample_chain(
    spot: float,
    rate_curve: float | Callable[[float], float],
    dividends: float | Sequence[tuple[date, float]],
    *,
    as_of: datetime,
    style: Style = "index",
    seed: int = 0,
    noise: bool = True,
    ssvi: SSVIParams = DEFAULT_SSVI,
    variance_clock: VarianceClock = DEFAULT_CLOCK,
    box_spread: float = BOX_SPREAD,
) -> SyntheticChain:
    """A realistic synthetic option chain with known true values (see the module docstring for the model).

    Parameters
    ----------
    spot : float
        Underlying price, $ (index points): 100 to 50,000 for an index, $5 to $50,000 for an ETF (see
        :func:`strike_interval`).
    rate_curve : float or callable
        Treasury zero rate (continuously compounded, decimal): one rate for every expiry, or a function of the time
        to expiry in years.
    dividends : float or sequence of (date, float)
        A continuous dividend yield (decimal), or cash dividends as ``(ex-date, $ per share)``; those with an ex-date
        open at or before ``as_of`` are already paid and ignored.
    as_of : datetime
        Valuation time (a naive datetime is New York wall-clock time), from 2000 to 2097.
    style : {"index", "etf"}, default "index"
        ``"index"``: European, SPX (AM-settled monthlies) and SPXW (PM-settled weeklies), ticks of $0.05/$0.10.
        ``"etf"``: American, SPY (PM-settled), penny ticks.
    seed : int, default 0
        Seed of the noise generator.
    noise : bool, default True
        Move mids within the spread and jitter spreads; ``False`` quotes the model value as the mid.
    ssvi : SSVIParams, default DEFAULT_SSVI
        The volatility surface, in variance time.
    variance_clock : VarianceClock, default DEFAULT_CLOCK
        The clock that turns calendar time into variance time; without events (SSVI has no jumps).
    box_spread : float, default 0.0045
        Box rate minus Treasury rate (decimal).

    Returns
    -------
    SyntheticChain
        The quotes (canonical schema), the model value and vol of each, and the true forward, discount factor and
        ATM vol of each expiry.

    Raises
    ------
    ValueError
        If the spot, ``as_of``, a rate, the dividend yield, a cash dividend or ``box_spread`` is out of range, if the
        variance clock has events, if the dividends to an expiry are worth more than the spot, or if an expiry's
        forward is so high that fewer than five strikes above it are below OCC's strike limit of 100,000.
    TypeError
        If ``dividends`` is neither a number nor a sequence of pairs, or an ex-date is not a date.

    Examples
    --------
    An SPX-like chain at 10:00 ET on Monday 26 October 2026, 4% Treasury rate, 1.3% dividend yield:

    >>> chain = sample_chain(6000.0, 0.04, 0.013, as_of=datetime(2026, 10, 26, 10, 0))
    >>> chain.label
    'Synthetic — not market data'
    >>> len(chain.quotes), chain.expiries[["root", "settlement"]].value_counts().sort_index().to_dict()
    (5458, {('SPX', 'AM'): 12, ('SPXW', 'PM'): 5})
    >>> front = chain.expiries.iloc[0]
    >>> str(front["expiry"].date()), front["root"], round(float(front["forward"]), 2), round(float(front["atm_vol"]), 4)
    ('2026-10-30', 'SPXW', 6002.2, 0.1772)
    >>> chain.quotes.query("symbol == 'SPXW261030C06000000'")[["bid", "ask"]].values.tolist()
    [[46.8, 47.2]]
    """
    spot = float(spot)
    interval = strike_interval(spot, style)  # also checks the style and the spot
    box_spread = float(box_spread)
    if not math.isfinite(box_spread):
        raise ValueError(f"box_spread must be finite, got {box_spread!r}")
    if variance_clock.events:
        raise ValueError("the SSVI surface has no jumps: pass a variance clock without events")
    as_of = to_new_york(as_of)
    if not _FIRST_VALUATION <= as_of.date() <= _LAST_VALUATION:
        raise ValueError(
            f"as_of must be from {_FIRST_VALUATION} to {_LAST_VALUATION}, so that every expiry falls in the NYSE "
            f"calendar and in OCC's two-digit years (2000-2099), got {as_of:%Y-%m-%d}"
        )
    rate = _rate_function(rate_curve)
    if isinstance(dividends, numbers.Real) and not isinstance(dividends, bool):
        dividend_yield, cash = float(dividends), []
        if not math.isfinite(dividend_yield):
            raise ValueError(f"the dividend yield must be finite, got {dividend_yield!r}")
    elif isinstance(dividends, Iterable) and not isinstance(dividends, (str, bytes)):
        dividend_yield, cash = 0.0, _cash_dividends(dividends, as_of)
    else:
        raise TypeError(
            "dividends must be a continuous yield (a number) or a sequence of (ex-date, cash per share) pairs, got "
            f"{type(dividends).__name__}"
        )
    ex_dates = _ex_dates(cash, spot, as_of, rate, box_spread, variance_clock)

    rows, values, vols, half_spreads, expiry_rows = [], [], [], [], []
    for expiry, root, settlement in _schedule(as_of.date(), style):
        settles = settlement_time(expiry, settlement)
        T = year_fraction(as_of, settles)
        tau = variance_clock.variance_time(as_of, settles)
        if not (T > 0 and tau > 0):
            continue
        r = rate(T) + box_spread
        discount = math.exp(-r * T)
        paid = [ex_date for ex_date in ex_dates if ex_date.drop <= settles]
        pv = sum(ex_date.pv for ex_date in paid)
        escrowed = spot - pv
        if not escrowed > 0:
            raise ValueError(f"the dividends to {expiry} are worth more than the spot")
        forward = escrowed * math.exp(-dividend_yield * T) / discount
        theta = float(ssvi.theta(tau))
        strikes = _strikes(forward, interval, math.sqrt(theta))
        strikes = strikes[strikes < _OCC_STRIKE_LIMIT]
        if np.count_nonzero(strikes > forward) < _NEAR_STRIKES:
            raise ValueError(
                f"the {expiry} forward is {forward:,.2f} (spot {spot:,g}, box rate {r:.2%}): fewer than "
                f"{_NEAR_STRIKES} strikes above it are below OCC's strike limit of {_OCC_STRIKE_LIMIT:,.0f}; lower "
                "the spot or the rate"
            )
        k = np.log(strikes / forward)
        w = ssvi.total_variance(k, tau)[0]
        iv = np.sqrt(w / T)
        d1 = -k / np.sqrt(w) + 0.5 * np.sqrt(w)
        vega = discount * forward * np.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi) * math.sqrt(T)
        widening = (1.0 + 0.25 * np.abs(k) / math.sqrt(theta)) * (1.0 + 0.5 * T)
        for call in (True, False):
            if style == "index":
                value = _black(forward, strikes, w, discount, call)
            elif not cash:  # a continuous yield: the binomial engine
                kind = "call" if call else "put"
                value = np.array(
                    [
                        binomial.price(escrowed, K, T, r, sigma, dividend_yield, kind, TREE_STEPS, True, "lr")
                        for K, sigma in zip(strikes, iv, strict=True)
                    ]
                )
            elif not call:
                value = _escrowed_american_puts(escrowed, strikes, T, r, iv, [(e.years, e.pv) for e in paid])
            else:  # Black's approximation: exercise now, just before an ex-date, or hold to expiry
                value = np.maximum(_black(forward, strikes, w, discount, True), spot - strikes)
                for ex_date in paid:
                    if ex_date.variance_time > 0:  # the European to just before it, on the surface at that moment
                        w_before = ssvi.total_variance(np.log(strikes / ex_date.forward), ex_date.variance_time)[0]
                        european = _black(ex_date.forward, strikes, w_before, ex_date.discount_factor, True)
                        value = np.maximum(value, european)
            half = _SPREAD_VOL[style] * widening * vega + _SPREAD_VALUE * value
            for strike, v, sigma, h in zip(strikes, value, iv, half, strict=True):
                rows.append((root, expiry, settlement, "call" if call else "put", strike))
                values.append(v)
                vols.append(sigma)
                half_spreads.append(h)
        expiry_rows.append(
            {
                "expiry": pd.Timestamp(expiry),
                "root": root,
                "settlement": settlement,
                "settles_at": settles,
                "maturity": T,
                "variance_time": tau,
                "rate": r,
                "discount_factor": discount,
                "dividends_pv": pv,
                "forward": forward,
                "atm_vol": math.sqrt(theta / T),
                "strikes": len(strikes),
            }
        )

    frame = pd.DataFrame(rows, columns=["root", "expiry", "settlement", "type", "strike"])
    value = np.maximum(np.array(values), 0.0)
    if style == "etf":  # an American option is worth at least a shorter one, whose exercise it can copy
        value = pd.Series(value).groupby([frame["root"], frame["type"], frame["strike"]]).cummax().to_numpy()
    tick = _MIN_TICK[style]
    half = np.maximum(np.array(half_spreads), tick)
    mid = value
    no_bid = np.zeros(len(frame), dtype=bool)
    if noise:
        rng = np.random.default_rng(seed)
        half = np.maximum(half * np.exp(0.2 * rng.standard_normal(len(frame))), tick)
        mid = value + half * np.clip(0.25 * rng.standard_normal(len(frame)), -0.5, 0.5)
        no_bid = (value < 4 * tick) & (rng.random(len(frame)) < 1 / 3)
    bid = np.where(no_bid, 0.0, _round_down(mid - half, style))
    ask = _round_up(mid + half, style)

    order = frame.assign(_kind=frame["type"].map({"call": 0, "put": 1})).sort_values(
        ["expiry", "root", "strike", "_kind"], kind="stable"
    )
    index = order.index.to_numpy()
    quotes = chain_frame(
        {
            "symbol": [
                occ_symbol(root, expiry, kind, strike)
                for root, expiry, kind, strike in order[["root", "expiry", "type", "strike"]].itertuples(index=False)
            ],
            "root": order["root"].to_numpy(),
            "expiry": pd.to_datetime(order["expiry"]).to_numpy(),
            "settlement": order["settlement"].to_numpy(),
            "type": order["type"].to_numpy(),
            "strike": order["strike"].to_numpy(),
            "bid": bid[index],
            "ask": ask[index],
        }
    )
    quotes.attrs["source"] = SYNTHETIC_LABEL
    theoretical = pd.DataFrame({"value": value[index], "iv": np.array(vols)[index]})
    expiries = pd.DataFrame(expiry_rows).sort_values(["expiry", "root"], kind="stable").reset_index(drop=True)
    return SyntheticChain(quotes, theoretical, expiries, spot, as_of, style)
