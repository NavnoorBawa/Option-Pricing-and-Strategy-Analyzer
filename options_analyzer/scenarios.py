"""Scenario analysis: mark-to-market P&L under spot, volatility, rate and time shocks.

Every number here comes from re-pricing each leg of the position with
Black-Scholes-Merton at the shocked inputs and the remaining time to expiry -
the same valuation used for the payoff diagram, so the tabs of the app are
mutually consistent. Scenario P&L is measured against the position's
*current* model value, so it isolates market risk (fill prices and fees do not
enter; they are part of the payoff analysis instead).

The risk tables (:func:`greek_ladder`, :func:`vega_buckets`) use stable
snake_case column and bucket names; display labels belong to the UI.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.special import ndtr

from . import black_scholes as bs
from .strategies import (
    VOL_SHOCK_REFERENCE_TENOR,
    Market,
    PositionRisk,
    Strategy,
    position_risk,
    position_value,
)

__all__ = [
    "DEFAULT_EXPIRY_EDGES_DAYS",
    "DEFAULT_LADDER_SHOCKS",
    "DEFAULT_MONEYNESS_EDGES",
    "DEFAULT_VOL_TENOR_DECAY",
    "DELTA_BUCKETS",
    "HISTORICAL_SCENARIOS",
    "HYPOTHETICAL_SCENARIOS",
    "VEGA_WEIGHTINGS",
    "VOL_MODES",
    "StressScenario",
    "greek_ladder",
    "pnl_grid",
    "run_stress_tests",
    "scenario_cube",
    "vega_buckets",
    "weighted_vega",
]

DAYS_PER_YEAR = 365.0


@dataclass(frozen=True)
class StressScenario:
    """An instantaneous shock, optionally spread over ``days`` of elapsed time."""

    name: str
    spot_change: float  # relative, e.g. -0.20 = -20 %
    vol_multiplier: float  # implied vol is multiplied by this factor
    rate_change: float  # absolute, e.g. -0.015 = -150 bp
    days: int = 0  # calendar days the episode lasted
    description: str = ""
    #: True when ``vol_multiplier`` is a one-month (VIX-like) shock, which is
    #: damped for longer tenors; False applies it literally to every leg.
    one_month_vol_shock: bool = False


#: Historical analogues, built from S&P 500 closes (Yardeni bull/bear tables,
#: closing basis), the VIX / VXO (FRED VIXCLS, VXOCLS) and the Fed funds target
#: (federalreserve.gov open-market archive) over each episode. The implied-vol
#: multiplier is the ratio of the VIX at the end of the window to its level at
#: the start, i.e. a shock to one-month implied vol; see :func:`run_stress_tests`
#: for how it is damped for longer tenors. Rate changes are policy-rate moves
#: over the window.
HISTORICAL_SCENARIOS: tuple[StressScenario, ...] = (
    StressScenario(
        "Black Monday (1987)",
        -0.205,
        4.1,
        -0.005,
        1,
        "19 Oct 1987: S&P 500 -20.5% in one session; VXO 36 -> 150.",
        one_month_vol_shock=True,
    ),
    StressScenario(
        "Dot-com bust (2000-02)",
        -0.491,
        1.8,
        -0.0425,
        929,
        "24 Mar 2000 - 9 Oct 2002: S&P 500 -49.1%; VIX ~23 -> ~42; Fed funds 6.00% -> 1.75%.",
        one_month_vol_shock=True,
    ),
    StressScenario(
        "Global Financial Crisis (2007-09)",
        -0.568,
        3.1,
        -0.046,
        517,
        "9 Oct 2007 - 9 Mar 2009: S&P 500 -56.8%; VIX 16 -> 50 (peak close 80.9 in Nov 2008); "
        "Fed funds 4.75% -> 0-0.25%.",
        one_month_vol_shock=True,
    ),
    StressScenario(
        "Volmageddon (2018)",
        -0.041,
        2.16,
        0.0,
        1,
        "5 Feb 2018: S&P 500 -4.1%; VIX 17.3 -> 37.3, its largest one-day jump at the time.",
        one_month_vol_shock=True,
    ),
    StressScenario(
        "COVID crash (2020)",
        -0.339,
        4.3,
        -0.015,
        33,
        "19 Feb - 23 Mar 2020: S&P 500 -33.9% in 23 sessions; VIX 14 -> 62 (record close 82.7 on 16 Mar); "
        "Fed cut 150 bp.",
        one_month_vol_shock=True,
    ),
    StressScenario(
        "2022 rate shock",
        -0.254,
        2.0,
        0.03,
        282,
        "3 Jan - 12 Oct 2022: S&P 500 -25.4%; VIX 17 -> 34; Fed funds +300 bp.",
        one_month_vol_shock=True,
    ),
    StressScenario(
        "Tariff shock (Apr 2025)",
        -0.121,
        2.4,
        0.0,
        6,
        "2 - 8 Apr 2025: S&P 500 -12.1% in four sessions; VIX 22 -> 52.",
        one_month_vol_shock=True,
    ),
)

HYPOTHETICAL_SCENARIOS: tuple[StressScenario, ...] = (
    StressScenario("Rally +15%, IV -30%", 0.15, 0.7, 0.0, 0, "Relief rally with implied volatility collapsing."),
    StressScenario(
        "Vol crush (IV -40%)",
        0.0,
        0.6,
        0.0,
        0,
        "Implied volatility falls 40% with spot unchanged (e.g. after earnings).",
    ),
    StressScenario("Vol spike (IV +50%)", 0.0, 1.5, 0.0, 0, "Implied volatility rises 50% with spot unchanged."),
    StressScenario("Rates +200 bp", 0.0, 1.0, 0.02, 0, "Parallel 2-point rise in the risk-free rate."),
    StressScenario(
        "Fed severely adverse (2026)",
        -0.58,
        4.0,
        -0.036,
        274,
        "Federal Reserve 2026 supervisory scenario (federalreserve.gov, Feb 2026): equities fall about 58% from "
        "Q4 2025 to Q3 2026, the VIX peaks at 72 (about 4x), the 3-month Treasury rate falls from 3.7% to 0.1%.",
        one_month_vol_shock=True,
    ),
    StressScenario("Gap down -10%", -0.10, 1.4, 0.0, 0, "Overnight gap lower with a volatility pickup."),
    StressScenario("Gap up +10%", 0.10, 0.9, 0.0, 0, "Overnight gap higher with a mild volatility drop."),
)


#: Exponent for damping implied-vol shocks with tenor. Crisis data put the
#: 1M-vs-3M decay of additive VIX shocks at about 0.20 (2008) to 0.25 (2020).
DEFAULT_VOL_TENOR_DECAY = 0.25


def run_stress_tests(
    strategy: Strategy,
    market: Market,
    scenarios: tuple[StressScenario, ...] | list[StressScenario],
    include_elapsed_time: bool = False,
    multiplier: float = 1.0,
    vol_tenor_decay: float = DEFAULT_VOL_TENOR_DECAY,
) -> list[dict]:
    """P&L of the position under each scenario (in $ after ``multiplier``).

    For scenarios whose IV multiplier is a one-month (VIX) shock,
    ``vol_tenor_decay > 0`` damps it for longer-dated legs (a 1-year option's
    IV moves roughly half as much as one-month IV in a crisis); hypothetical
    IV shocks apply literally to every leg. With ``include_elapsed_time`` the
    scenario's duration also passes, so options decay; a leg that expires
    during the episode is settled on the end-of-episode price (a stylised
    simplification, flagged in the ``Legs expired`` column).
    """
    base = float(position_value(strategy, market))
    rows = []
    for sc in scenarios:
        t = sc.days / DAYS_PER_YEAR if include_elapsed_time else 0.0
        decay = vol_tenor_decay if sc.one_month_vol_shock else 0.0
        value = position_value(
            strategy,
            market,
            spot=market.spot * (1.0 + sc.spot_change),
            t=t,
            vol_multiplier=sc.vol_multiplier,
            rate_shift=sc.rate_change,
            vol_tenor_decay=decay,
        )
        expired = sum(
            1 for leg in strategy.legs if leg.is_option and (market.expiry if leg.expiry is None else leg.expiry) < t
        )
        change = multiplier * (float(value) - base)
        rows.append(
            {
                "Scenario": sc.name,
                "Spot change": sc.spot_change,
                "IV multiplier": sc.vol_multiplier,
                "IV shock damped": bool(decay),
                "Rate change (bp)": sc.rate_change * 1e4,
                "Days elapsed": sc.days if include_elapsed_time else 0,
                "Legs expired": expired,
                "P&L": change,
                "Description": sc.description,
            }
        )
    return rows


#: How a volatility shock moves each leg's implied vol (see :func:`scenario_cube`).
VOL_MODES = ("relative", "absolute", "1m")


def _number(name: str, value: Any) -> float:
    """``value`` as a float; ``ValueError`` naming ``name`` unless it is a single finite number."""
    if np.ndim(value) != 0:
        raise ValueError(f"{name} must be a single number, got {value!r}")
    try:
        finite = math.isfinite(value)
    except TypeError:  # a string, None, ...
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not finite:
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _axis(name: str, values: ArrayLike, requirement: str = "finite", minimum: float = -math.inf) -> np.ndarray:
    """``values`` as a one-dimensional float array of finite numbers >= ``minimum``; ``ValueError`` naming ``name``."""
    try:
        raw = np.asarray(values)
        numeric = raw.dtype.kind in "biuf"  # not strings, None, ...
    except ValueError:  # a ragged list
        numeric = False
    if not numeric:
        raise ValueError(f"{name} must be numbers, got {values!r}")
    array = np.atleast_1d(raw.astype(float))
    if array.ndim != 1:
        raise ValueError(f"{name} must be a number or a one-dimensional array")
    bad = array[~(np.isfinite(array) & (array >= minimum))]
    if bad.size:
        raise ValueError(f"{name} must be {requirement}, got {bad[0]:g}")
    return array


def _spot_axis(spot_changes: ArrayLike) -> np.ndarray:
    return _axis("spot_changes", spot_changes, "finite and at least -1 (a -100% move takes spot to zero)", -1.0)


def _days_axis(days: ArrayLike) -> np.ndarray:
    return _axis("days", days, "finite and non-negative (calendar days elapsed)", 0.0)


def _check_vol_mode(vol_mode: str, vol_tenor_decay: float) -> None:
    if vol_mode not in VOL_MODES:
        raise ValueError(f"vol_mode must be one of {VOL_MODES}, got {vol_mode!r}")
    _number("vol_tenor_decay", vol_tenor_decay)


def _vol_shock(shocks: np.ndarray, vol_mode: str, vol_tenor_decay: float) -> dict:
    """Keyword arguments of :func:`position_value` that apply ``shocks`` in ``vol_mode`` (checked by the caller)."""
    if vol_mode == "relative":
        return {"vol_multiplier": 1.0 + shocks}
    if vol_mode == "absolute":
        return {"vol_shift": shocks}
    return {"vol_multiplier": 1.0 + shocks, "vol_tenor_decay": vol_tenor_decay}


def pnl_grid(
    strategy: Strategy,
    market: Market,
    spot_changes: np.ndarray,
    vol_changes: np.ndarray | None = None,
    days: np.ndarray | float = 0.0,
    multiplier: float = 1.0,
    *,
    vol_mode: str = "relative",
    vol_tenor_decay: float = DEFAULT_VOL_TENOR_DECAY,
) -> np.ndarray:
    """P&L on a grid.

    * with ``vol_changes`` (relative IV changes, e.g. 0.25 = +25 %) the result
      has shape ``(len(vol_changes), len(spot_changes))`` at a fixed ``days``;
    * otherwise the result has shape ``(len(days), len(spot_changes))`` at
      unchanged implied volatility (a single day count gives one row).

    ``vol_mode`` sets how ``vol_changes`` move each leg's IV: ``"relative"``
    (the default), ``"absolute"`` (0.05 = +5 vol points) or ``"1m"`` (a
    relative shock to one-month IV, damped for longer tenors by
    ``vol_tenor_decay``); see :func:`scenario_cube`.

    Spot changes must be finite and at least -1, volatility changes finite and
    days finite and non-negative; anything else raises ``ValueError`` naming
    the argument.
    """
    _check_vol_mode(vol_mode, vol_tenor_decay)
    spots = market.spot * (1.0 + _spot_axis(spot_changes))[None, :]
    if vol_changes is not None:
        if np.ndim(days) != 0:
            raise ValueError("with vol_changes, days must be a single number (the grid is volatility x spot)")
        shocks = _axis("vol_changes", vol_changes)[:, None]
        t = float(_days_axis(days)[0]) / DAYS_PER_YEAR
        values = position_value(strategy, market, spots, t, **_vol_shock(shocks, vol_mode, vol_tenor_decay))
        shape = (shocks.shape[0], spots.shape[1])
    else:
        times = _days_axis(days)[:, None] / DAYS_PER_YEAR
        values = position_value(strategy, market, spots, times)
        shape = (times.shape[0], spots.shape[1])
    base = float(position_value(strategy, market))
    # Shares do not depend on implied vol: a stock-only position still fills every row of the grid.
    return multiplier * (np.broadcast_to(np.asarray(values, dtype=float), shape) - base)


def scenario_cube(
    strategy: Strategy,
    market: Market,
    spot_changes: ArrayLike,
    vol_shocks: ArrayLike,
    days: ArrayLike = 0.0,
    *,
    vol_mode: str = "relative",
    vol_tenor_decay: float = DEFAULT_VOL_TENOR_DECAY,
    multiplier: float = 1.0,
) -> np.ndarray:
    """Full-revaluation P&L over elapsed days x volatility shocks x spot changes.

    Every cell re-prices each leg at the shocked spot and implied vol with its remaining time to expiry, against
    the position's model value today (as :func:`pnl_grid` does): ``cube[d, v, s]`` is the P&L after ``days[d]``
    days when spot moves by ``spot_changes[s]`` and implied vol by ``vol_shocks[v]``.

    Parameters
    ----------
    strategy, market : Strategy, Market
        The position and today's market.
    spot_changes : array_like
        Relative spot moves, e.g. -0.10 = -10%.
    vol_shocks : array_like
        Volatility shocks, applied to each leg's own IV ``sigma`` according to ``vol_mode``:

        * ``"relative"``: ``sigma (1 + shock)``, so 0.25 raises IV by a quarter of itself;
        * ``"absolute"``: ``sigma + shock``, so 0.05 = +5 vol points;
        * ``"1m"``: a relative shock to one-month IV, damped for a leg with ``tau`` years left to
          ``sigma (1 + shock min(1, (tau / 1M)^-vol_tenor_decay))``, as in :func:`run_stress_tests`.

        A shocked IV is floored at 1e-8.
    days : array_like, default 0.0
        Calendar days elapsed, non-negative (fractions allowed). A leg that has expired is worth its intrinsic
        value.
    vol_mode : {"relative", "absolute", "1m"}, default "relative"
        See ``vol_shocks``.
    vol_tenor_decay : float, default DEFAULT_VOL_TENOR_DECAY
        Exponent of the tenor damping in ``"1m"`` mode.
    multiplier : float, default 1.0
        Dollar scaling, e.g. contracts x shares per contract.

    Returns
    -------
    numpy.ndarray
        P&L ($) of shape ``(len(days), len(vol_shocks), len(spot_changes))``.

    Raises
    ------
    ValueError
        Naming the argument, if a spot change is below -1, a day count is negative or any value is not a finite
        number, or if ``vol_mode`` is unknown.

    Examples
    --------
    >>> from options_analyzer.strategies import build_template
    >>> market = Market(100, 0.05, 0.20, 30 / 365)
    >>> straddle = build_template("Long Straddle", 100, 5)
    >>> cube = scenario_cube(straddle, market, [-0.05, 0.0, 0.05], [0.0, 0.10], [0, 7])
    >>> cube.shape
    (2, 2, 3)
    >>> float(cube[0, 0, 1])
    0.0
    >>> round(float(cube[0, 1, 1]), 4)  # IV 20% -> 22%, nothing else changes
    0.4553
    """
    _check_vol_mode(vol_mode, vol_tenor_decay)
    spots = market.spot * (1.0 + _spot_axis(spot_changes))[None, None, :]
    shocks = _axis("vol_shocks", vol_shocks)[None, :, None]
    t = _days_axis(days)[:, None, None] / DAYS_PER_YEAR
    base = float(position_value(strategy, market))
    values = position_value(strategy, market, spots, t, **_vol_shock(shocks, vol_mode, vol_tenor_decay))
    shape = (t.shape[0], shocks.shape[1], spots.shape[2])
    return multiplier * (np.broadcast_to(np.asarray(values, dtype=float), shape) - base)


# ---------------------------------------------------------------------------
# Greek ladder
# ---------------------------------------------------------------------------

#: Default spot shocks of :func:`greek_ladder`, in percent.
DEFAULT_LADDER_SHOCKS: tuple[float, ...] = (-20, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 20)


def greek_ladder(
    strategy: Strategy,
    market: Market,
    shocks: Sequence[float] = DEFAULT_LADDER_SHOCKS,
    *,
    unit: str = "percent",
    sigma_days: float = 1.0,
    roll_days: float = 0.0,
    multiplier: float = 1.0,
) -> pd.DataFrame:
    """Spot ladder: full-revaluation P&L and the position's risk at each shocked spot.

    Each row re-prices the position at a shocked spot (implied vols unchanged) after ``roll_days`` and gives the
    P&L against today's model value with every field of
    :func:`~options_analyzer.strategies.position_risk` at that spot and date, so a desk sees how delta, gamma and
    vega change as spot moves.

    Parameters
    ----------
    strategy, market : Strategy, Market
        The position and today's market.
    shocks : sequence of float, default DEFAULT_LADDER_SHOCKS
        Spot shocks, in percent (``unit="percent"``: 5 = +5%) or in standard deviations (``unit="sigma"``: 2 =
        +2 sigma, one sigma being ``market.volatility * sqrt(sigma_days / 365)`` of spot).
    unit : {"percent", "sigma"}, default "percent"
        See ``shocks``.
    sigma_days : float, default 1.0
        Horizon of one standard deviation in ``"sigma"`` units, in calendar days.
    roll_days : float, default 0.0
        Calendar days that pass before the shock (fractions allowed), so the P&L includes the time decay.
    multiplier : float, default 1.0
        Dollar scaling, e.g. contracts x shares per contract.

    Returns
    -------
    pandas.DataFrame
        One row per shock with the columns ``shock`` (as given), ``spot_change`` (relative), ``spot``, ``pnl``
        ($) and the remaining :class:`~options_analyzer.strategies.PositionRisk` fields (``value``, ``delta``,
        ``dollar_delta``, ``gamma``, ``gamma_1pct``, ``dollar_gamma_1pct``, ``vega``, ``theta``, ``rho``,
        ``vanna``, ``volga``, ``charm``, ``veta``) in the units documented there.

    Raises
    ------
    ValueError
        Naming the argument, if a shock, ``roll_days`` or ``sigma_days`` is not a single finite number, if
        ``roll_days`` is negative or ``sigma_days`` not positive, or if a shock takes spot to zero or below.

    Examples
    --------
    >>> from options_analyzer.strategies import build_template
    >>> market = Market(100, 0.05, 0.20, 30 / 365)
    >>> strangle = build_template("Short Strangle", 100, 5)
    >>> ladder = greek_ladder(strangle, market, (-5, 0, 5), multiplier=100)
    >>> ladder[["shock", "spot", "pnl", "delta"]].round(2).to_dict("list")
    {'shock': [-5, 0, 5], 'spot': [95.0, 100.0, 105.0], 'pnl': [-86.45, 0.0, -147.02], 'delta': [41.01, -6.66, -50.75]}
    """
    if unit not in ("percent", "sigma"):
        raise ValueError(f"unit must be 'percent' or 'sigma', got {unit!r}")
    roll, horizon = _number("roll_days", roll_days), _number("sigma_days", sigma_days)
    if roll < 0:
        raise ValueError(f"roll_days must be a non-negative number of days, got {roll_days!r}")
    if not horizon > 0:
        raise ValueError(f"sigma_days must be a positive number of days, got {sigma_days!r}")
    try:
        shocks = list(shocks)
    except TypeError:
        raise ValueError(f"shocks must be a sequence of numbers, got {shocks!r}") from None
    step = 0.01 if unit == "percent" else market.volatility * math.sqrt(horizon / DAYS_PER_YEAR)
    t = roll / DAYS_PER_YEAR
    base = multiplier * float(position_value(strategy, market, market.spot, 0.0))
    rows = []
    for shock in shocks:
        change = _number("shock", shock) * step
        if not change > -1.0:
            raise ValueError(f"a shock of {shock} ({unit}) takes spot to zero or below")
        risk = asdict(position_risk(strategy, market, market.spot * (1.0 + change), t, multiplier=multiplier))
        rows.append({"shock": shock, "spot_change": change, "spot": risk["spot"], "pnl": risk["value"] - base, **risk})
    risk_columns = [f.name for f in fields(PositionRisk) if f.name != "spot"]
    return pd.DataFrame(rows, columns=["shock", "spot_change", "spot", "pnl", *risk_columns])


# ---------------------------------------------------------------------------
# Vega buckets and 1M-equivalent vega
# ---------------------------------------------------------------------------

#: Upper edges (calendar days, inclusive) of the expiry buckets of :func:`vega_buckets`; a last bucket holds the rest.
DEFAULT_EXPIRY_EDGES_DAYS: tuple[float, ...] = (7, 14, 30, 60, 91, 182, 365, 730)
#: Strike buckets of the delta axis, from low to high strike, and the undiscounted put delta ``N(-d1)`` each is
#: centred on. A strike goes to the nearest centre, so the 10-delta buckets also hold the strikes further out.
DELTA_BUCKETS: dict[str, float] = {"10d_put": 0.10, "25d_put": 0.25, "atm": 0.50, "25d_call": 0.75, "10d_call": 0.90}
#: Upper edges (inclusive) of the strike buckets of the moneyness axis, as strike / spot.
DEFAULT_MONEYNESS_EDGES: tuple[float, ...] = (0.80, 0.90, 0.95, 1.05, 1.10, 1.20)


def _bucket_labels(edges: np.ndarray, unit: str) -> list[str]:
    """``up_to_<e0>``, ``<e0>_to_<e1>``, ..., ``over_<last>`` for right-closed buckets with these upper edges."""
    text = [f"{e:g}{unit}" for e in edges]
    return [f"up_to_{text[0]}", *(f"{a}_to_{b}" for a, b in pairwise(text)), f"over_{text[-1]}"]


def _edges(name: str, edges: Sequence[float]) -> np.ndarray:
    values = np.asarray(edges, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"{name} must be a non-empty sequence of positive numbers, got {edges!r}")
    if np.any(np.diff(values) <= 0):
        raise ValueError(f"{name} must increase strictly, got {tuple(edges)}")
    return values


def _option_legs(strategy: Strategy, market: Market):
    """``(leg, tau, sigma, raw BSM vega)`` of every option leg at inception, at the leg's own expiry and IV."""
    for leg in strategy.legs:
        if leg.is_option:
            tau = market.expiry if leg.expiry is None else leg.expiry
            vol = market.volatility if leg.volatility is None else leg.volatility
            g = bs.greeks(market.spot, leg.strike, tau, market.rate, vol, market.dividend_yield, leg.kind)
            yield leg, tau, vol, float(g["vega"])


def vega_buckets(
    strategy: Strategy,
    market: Market,
    expiry_edges: Sequence[float] = DEFAULT_EXPIRY_EDGES_DAYS,
    strike_axis: str = "delta",
    *,
    moneyness_edges: Sequence[float] = DEFAULT_MONEYNESS_EDGES,
    multiplier: float = 1.0,
) -> pd.DataFrame:
    """Vega ($ per vol point) by expiry bucket x strike bucket, with row and column totals.

    Each option leg's vega (Black-Scholes-Merton at its own IV: ``quantity x vega / 100 x multiplier``, as in
    :func:`~options_analyzer.strategies.position_risk`) goes into exactly one cell, so the cells add up to the
    position's vega. Expiry buckets are right-closed: with the default edges a 30-day leg is in ``14d_to_30d``
    (days are rounded to 1e-6 first, so a leg entered as ``30 / 365`` years sits on the 30-day edge). Shares have
    no vega.

    Parameters
    ----------
    strategy, market : Strategy, Market
        The position and the market it is valued in.
    expiry_edges : sequence of float, default DEFAULT_EXPIRY_EDGES_DAYS
        Upper edges of the expiry buckets in calendar days, increasing; a last bucket holds the longer expiries.
    strike_axis : {"delta", "moneyness"}, default "delta"
        ``"delta"`` puts a strike in the :data:`DELTA_BUCKETS` bucket nearest to its undiscounted put delta
        ``N(-d1)``, computed with the leg's own IV and expiry (a call and a put at the same strike and IV share a
        bucket); ``"moneyness"`` buckets strike / spot by ``moneyness_edges`` (right-closed).
    moneyness_edges : sequence of float, default DEFAULT_MONEYNESS_EDGES
        Upper edges of the moneyness buckets (strike / spot), increasing.
    multiplier : float, default 1.0
        Dollar scaling, e.g. contracts x shares per contract.

    Returns
    -------
    pandas.DataFrame
        Index ``expiry_bucket`` (``up_to_7d``, ``7d_to_14d``, ..., ``over_730d``, then ``total``) and columns
        ``strike_bucket`` (low to high strike, then ``total``). Empty buckets are kept, so the shape depends only
        on the edges.

    Examples
    --------
    >>> from options_analyzer.strategies import Leg
    >>> market = Market(100, 0.05, 0.20, 30 / 365)
    >>> book = Strategy("book", (Leg("call", 10, 100), Leg("put", -5, 90, 91 / 365)))
    >>> vega_buckets(book, market, expiry_edges=(30, 91)).round(2)  # doctest: +NORMALIZE_WHITESPACE
    strike_bucket  10d_put  25d_put   atm  25d_call  10d_call  total
    expiry_bucket
    up_to_30d         0.00      0.0  1.14       0.0       0.0   1.14
    30d_to_91d       -0.47      0.0  0.00       0.0       0.0  -0.47
    over_91d          0.00      0.0  0.00       0.0       0.0   0.00
    total            -0.47      0.0  1.14       0.0       0.0   0.67
    """
    expiry = _edges("expiry_edges", expiry_edges)
    rows = _bucket_labels(expiry, "d")
    if strike_axis == "delta":
        columns = list(DELTA_BUCKETS)
        centres = np.fromiter(DELTA_BUCKETS.values(), dtype=float)
        bounds = 0.5 * (centres[1:] + centres[:-1])
    elif strike_axis == "moneyness":
        bounds = _edges("moneyness_edges", moneyness_edges)
        columns = _bucket_labels(bounds * 100.0, "pct")
    else:
        raise ValueError(f"strike_axis must be 'delta' or 'moneyness', got {strike_axis!r}")
    cells = np.zeros((len(rows), len(columns)))
    s, carry = market.spot, market.rate - market.dividend_yield
    for leg, tau, vol, vega in _option_legs(strategy, market):
        if strike_axis == "delta":
            v = vol * math.sqrt(tau)
            coordinate = float(ndtr(-((math.log(s / leg.strike) + carry * tau) / v + 0.5 * v)))
        else:
            coordinate = round(leg.strike / s, 12)
        i = int(np.searchsorted(expiry, round(tau * DAYS_PER_YEAR, 6), side="left"))
        j = int(np.searchsorted(bounds, coordinate, side="left"))
        cells[i, j] += multiplier * leg.quantity * vega * bs.GREEKS["vega"].scale
    table = pd.DataFrame(cells, index=pd.Index(rows, name="expiry_bucket"), columns=pd.Index(columns))
    table["total"] = table.sum(axis=1)
    table.loc["total"] = table.sum(axis=0)
    table.columns.name = "strike_bucket"
    return table


#: Weighting schemes of :func:`weighted_vega`.
VEGA_WEIGHTINGS = ("relative", "sqrt_time")


def weighted_vega(
    strategy: Strategy,
    market: Market,
    *,
    decay: float = DEFAULT_VOL_TENOR_DECAY,
    scheme: str = "relative",
    multiplier: float = 1.0,
) -> float:
    """One-month-equivalent vega: the position's vega with each expiry weighted by how much its IV moves.

    Raw vega adds a vol point of a one-week option to a vol point of a two-year option, although long-dated
    implied vol moves much less. Two weightings, both referred to the one-month (1M = 30 days) tenor, with
    ``tau`` each option leg's time to expiry:

    * ``"relative"`` (default): ``sum(q x vega x sigma x min(1, (tau / 1M)^-decay)) / 100``, the first-order P&L
      ($) of a 1% relative rise in one-month IV damped for longer tenors exactly as a one-month stress scenario is
      (see :func:`run_stress_tests` and ``StressScenario(..., one_month_vol_shock=True)``);
    * ``"sqrt_time"``: ``sum(q x vega x sqrt(1M / tau)) / 100``, the first-order P&L ($) of a one-vol-point rise
      in one-month IV under the square-root-of-time rule, where a tenor-``tau`` IV moves ``sqrt(1M / tau)``
      points. It is not capped: a one-week option counts about twice its vega, sqrt(30/7) = 2.07; ``decay`` is
      ignored.

    First-order means the shock times the derivative of the position's value: the symmetric stress
    ``(P&L(+1%) - P&L(-1%)) / 2`` equals it up to third-order terms. A one-sided stress adds the volga term
    ``sum(q x volga x dsigma^2) / 2``, which is not small for short-dated wings: for a long 7-day 95/105 strangle
    at 20% IV the P&L of the +1% one-month stress is 1.6% above the weighted vega.

    Parameters
    ----------
    strategy, market : Strategy, Market
        The position and the market it is valued in.
    decay : float, default DEFAULT_VOL_TENOR_DECAY
        Tenor exponent of the ``"relative"`` weighting.
    scheme : {"relative", "sqrt_time"}, default "relative"
        See above.
    multiplier : float, default 1.0
        Dollar scaling, e.g. contracts x shares per contract.

    Returns
    -------
    float
        Weighted vega ($).

    Examples
    --------
    A two-year at-the-money call has 4.4 times the raw vega of a one-month one, but only twice its exposure to a
    one-month vol move:

    >>> from options_analyzer.strategies import Leg, position_risk
    >>> market = Market(100, 0.05, 0.20, 30 / 365)
    >>> front = Strategy("1M", (Leg("call", 1, 100),))
    >>> back = Strategy("2Y", (Leg("call", 1, 100, 2.0),))
    >>> [round(position_risk(s, market).vega, 4) for s in (front, back)]
    [0.1138, 0.4991]
    >>> [round(weighted_vega(s, market), 4) for s in (front, back)]
    [0.0228, 0.0449]
    """
    if scheme not in VEGA_WEIGHTINGS:
        raise ValueError(f"scheme must be one of {VEGA_WEIGHTINGS}, got {scheme!r}")
    decay = _number("decay", decay)
    total = 0.0
    for leg, tau, vol, vega in _option_legs(strategy, market):
        # Same floor and ratio as the damping in strategies.position_value, so the two agree to first order.
        ratio = max(tau, 1e-8) / VOL_SHOCK_REFERENCE_TENOR
        weight = vol * min(1.0, ratio**-decay) if scheme == "relative" else ratio**-0.5
        total += leg.quantity * vega * weight * bs.GREEKS["vega"].scale
    return multiplier * total
