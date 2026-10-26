"""A realistic synthetic option chain: the default demo data, and a test oracle with known true values.

:func:`sample_chain` lists an SPX-like chain (``style="index"``: European, AM-settled monthlies under SPX and PM-settled
weeklies under SPXW) or an SPY-like one (``style="etf"``: American, PM-settled, all under SPY), and prices it from a
model whose every input is known:

* Expiries: the Friday weeklies of the next six weeks, the next six third-Friday monthlies and the March, June,
  September and December quarterlies out to two years. An expiry on an NYSE holiday moves to the trading day before.
* Time: years to the settlement time from :mod:`options_analyzer.clock` (minutes / 525,600).
* Volatility: an SSVI surface (:class:`options_analyzer.vol_surface.SSVIParams`) evaluated in variance time
  (:class:`options_analyzer.clock.VarianceClock`, weekends and holidays at a tenth of a trading day), so the total
  implied variance at forward log-moneyness ``k`` is ``w(k, tau)`` and the implied vol is ``sqrt(w / T)`` at the
  calendar time ``T``. SSVI with ``eta (1 + |rho|) <= 2`` has no static arbitrage, and a variance clock only
  re-times it, so the model prices have none either.
* Rates: discounting and forwards at a box rate, the Treasury zero rate plus 45 bp (SPX box spreads finance above
  Treasuries). ``rate_curve`` is a flat rate or a function of the time to expiry until a curve module exists.
* Dividends: a continuous yield, or cash dividends ``(ex-date, $ per share)`` that drop the spot at the open of the
  ex-date. Cash dividends use the escrowed-dividend approximation: the options are valued on the spot less the present
  value of the dividends paid before expiry, as if that were a lognormal asset. It gets the forward exactly but not the
  volatility of the stock just before an ex-date; the planned dividends engine (a finite-difference model with the
  dividend step on the spot) replaces it.
* Prices: Black-76 for index options. ETF options are American: puts (and calls with a continuous yield) come from the
  Leisen-Reimer binomial tree of :mod:`options_analyzer.binomial` with 201 steps; calls with cash dividends use Black's
  approximation, the larger of the European values to expiry and to just before each ex-date, with no tree needed
  because the escrowed spot pays nothing.
* Quotes: a half-spread of a few hundredths of a vol point times vega, widening with ``|k|`` (in ATM standard
  deviations) and with ``T``, plus 0.1% of the value, and at least one tick. Ticks are $0.05 below $3 and $0.10 from $3
  (SPX) or $0.01 (SPY is penny-quoted). Bids are rounded down and asks up, so the model value always lies inside the
  quote: the quotes have no static arbitrage either. Calls and puts are quoted at every strike; the far wings have a
  zero bid (an offer only).
* Noise (``noise=True``, the default): each spread is scaled by ``exp(0.2 Z)``, each mid moved by up to half the
  half-spread from the model value, and a third of the wing quotes cheaper than four ticks lose their bid, all from a
  generator seeded with ``seed``; the output is the same for the same inputs. ``noise=False`` quotes the model value
  as the mid.

Everything is labelled :data:`SYNTHETIC_LABEL`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal

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
    "SYNTHETIC_LABEL",
    "SyntheticChain",
    "sample_chain",
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
#: Half-spread in vol at the money for a short expiry, by style.
_SPREAD_VOL = {"index": 0.0005, "etf": 0.0002}
_SPREAD_VALUE = 0.001  # plus this fraction of the option value (deep in-the-money options quote wide in dollars)
_MIN_TICK = {"index": 0.05, "etf": 0.01}
#: Strike bands around the forward, in ln(K/F): (step as a multiple of the base step, then the lower and upper edge as
#: a multiple of the ATM standard deviation sqrt(w) with the smallest and largest width allowed). The base step is about
#: spot / 1,200 for an index (5 points for SPX) and spot / 600 for an ETF ($1 for SPY).
_BANDS = (
    (1, (1.5, 0.02, 0.04), (1.5, 0.02, 0.04)),
    (5, (4.0, 0.08, 0.35), (2.5, 0.05, 0.2)),
    (20, (8.0, 0.15, 1.0), (4.0, 0.08, 0.4)),
)
_NICE_STEPS = (0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0)

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


def _base_step(spot: float, style: Style) -> float:
    target = spot / (1_200.0 if style == "index" else 600.0)
    return max([step for step in _NICE_STEPS if step <= target], default=_NICE_STEPS[0])


def _strikes(forward: float, step: float, sd: float) -> np.ndarray:
    """Listed strikes around ``forward``: fine near the money, coarser further out (see :data:`_BANDS`)."""
    grids = []
    for multiple, (below, low_min, low_max), (above, high_min, high_max) in _BANDS:
        width = multiple * step
        low = forward * math.exp(-min(max(below * sd, low_min), low_max))
        high = forward * math.exp(min(max(above * sd, high_min), high_max))
        grids.append(np.arange(math.ceil(low / width), math.floor(high / width) + 1) * width)
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
    if style == "etf":
        bid = np.floor(price / 0.01 - 1e-9) * 0.01
    else:
        bid = np.where(price < 3.0, np.floor(price / 0.05 - 1e-9) * 0.05, np.floor(price / 0.10 - 1e-9) * 0.10)
    return np.round(np.maximum(bid, 0.0), 2)


def _round_up(price: np.ndarray, style: Style) -> np.ndarray:
    if style == "etf":
        ask = np.ceil(price / 0.01 + 1e-9) * 0.01
    else:
        ask = np.where(price <= 3.0, np.ceil(price / 0.05 + 1e-9) * 0.05, np.ceil(price / 0.10 + 1e-9) * 0.10)
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


def _cash_dividends(dividends: Sequence[tuple[date, float]], as_of: datetime) -> list[tuple[datetime, float]]:
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
        Underlying price, $.
    rate_curve : float or callable
        Treasury zero rate (continuously compounded, decimal): one rate for every expiry, or a function of the time
        to expiry in years.
    dividends : float or sequence of (date, float)
        A continuous dividend yield (decimal), or cash dividends as ``(ex-date, $ per share)``; those with an ex-date
        open at or before ``as_of`` are already paid and ignored.
    as_of : datetime
        Valuation time (a naive datetime is New York wall-clock time).
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
        The clock that turns calendar time into variance time.
    box_spread : float, default 0.0045
        Box rate minus Treasury rate (decimal).

    Returns
    -------
    SyntheticChain
        The quotes (canonical schema), the model value and vol of each, and the true forward, discount factor and
        ATM vol of each expiry.

    Examples
    --------
    An SPX-like chain at 10:00 ET on Monday 26 October 2026, 4% Treasury rate, 1.3% dividend yield:

    >>> chain = sample_chain(6000.0, 0.04, 0.013, as_of=datetime(2026, 10, 26, 10, 0))
    >>> chain.label
    'Synthetic — not market data'
    >>> len(chain.quotes), chain.expiries[["root", "settlement"]].value_counts().sort_index().to_dict()
    (6840, {('SPX', 'AM'): 12, ('SPXW', 'PM'): 5})
    >>> front = chain.expiries.iloc[0]
    >>> str(front["expiry"].date()), front["root"], round(float(front["forward"]), 2), round(float(front["atm_vol"]), 4)
    ('2026-10-30', 'SPXW', 6002.2, 0.1772)
    >>> chain.quotes.query("symbol == 'SPXW261030C06000000'")[["bid", "ask"]].values.tolist()
    [[46.7, 47.3]]
    """
    spot = float(spot)
    if not (math.isfinite(spot) and spot > 0):
        raise ValueError(f"spot must be finite and positive, got {spot!r}")
    if style not in _MIN_TICK:
        raise ValueError(f"style must be 'index' or 'etf', got {style!r}")
    box_spread = float(box_spread)
    if not math.isfinite(box_spread):
        raise ValueError(f"box_spread must be finite, got {box_spread!r}")
    as_of = to_new_york(as_of)
    rate = _rate_function(rate_curve)
    if isinstance(dividends, (int, float)):
        dividend_yield, cash = float(dividends), []
        if not math.isfinite(dividend_yield):
            raise ValueError(f"the dividend yield must be finite, got {dividend_yield!r}")
    else:
        dividend_yield, cash = 0.0, _cash_dividends(dividends, as_of)

    step = _base_step(spot, style)
    rows, values, vols, half_spreads, expiry_rows = [], [], [], [], []
    for expiry, root, settlement in _schedule(as_of.date(), style):
        settles = settlement_time(expiry, settlement)
        T = year_fraction(as_of, settles)
        tau = variance_clock.variance_time(as_of, settles)
        if not (T > 0 and tau > 0):
            continue
        r = rate(T) + box_spread
        discount = math.exp(-r * T)
        paid = [(year_fraction(as_of, drop), amount) for drop, amount in cash if drop <= settles]
        pv = sum(amount * math.exp(-(rate(t) + box_spread) * t) for t, amount in paid)
        escrowed = spot - pv
        if not escrowed > 0:
            raise ValueError(f"the dividends to {expiry} are worth more than the spot")
        forward = escrowed * math.exp(-dividend_yield * T) / discount
        theta = float(ssvi.theta(tau))
        strikes = _strikes(forward, step, math.sqrt(theta))
        k = np.log(strikes / forward)
        w = ssvi.total_variance(k, tau)[0]
        iv = np.sqrt(w / T)
        d1 = -k / np.sqrt(w) + 0.5 * np.sqrt(w)
        vega = discount * forward * np.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi) * math.sqrt(T)
        widening = (1.0 + 0.25 * np.abs(k) / math.sqrt(theta)) * (1.0 + 0.5 * T)
        for call in (True, False):
            if style == "index":
                value = _black(forward, strikes, w, discount, call)
            elif not call or not cash:
                kind = "call" if call else "put"
                value = np.array(
                    [
                        binomial.price(escrowed, K, T, r, sigma, dividend_yield, kind, TREE_STEPS, True, "lr")
                        for K, sigma in zip(strikes, iv, strict=True)
                    ]
                )
            else:  # Black's approximation: exercise just before an ex-date, or hold to expiry
                value = np.maximum(_black(forward, strikes, w, discount, True), spot - strikes)
                for t, _ in paid:
                    before = spot - sum(amount * math.exp(-(rate(s) + box_spread) * s) for s, amount in paid if s < t)
                    discount_t = math.exp(-(rate(t) + box_spread) * t)
                    value = np.maximum(value, _black(before / discount_t, strikes, iv * iv * t, discount_t, True))
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
