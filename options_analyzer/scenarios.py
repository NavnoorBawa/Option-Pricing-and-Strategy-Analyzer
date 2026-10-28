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
    _number,
    position_risk,
    position_value,
)

__all__ = [
    "DEFAULT_EXPIRY_EDGES_DAYS",
    "DEFAULT_LADDER_SHOCKS",
    "DEFAULT_MONEYNESS_EDGES",
    "DEFAULT_SIGMA_SHOCKS",
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
    """An instantaneous shock, optionally spread over ``days`` of elapsed time.

    A sourced scenario (the historical analogues and the Fed scenario, built with :meth:`sourced`) also keeps the
    volatility index it was calibrated on and that index's levels at the start and end of the window. Its
    ``vol_multiplier`` is the ratio ``vol_end / vol_start``, so the shock cannot drift from the levels that its
    description quotes: a multiplier more than 1e-12 (relative) away from the ratio raises ``ValueError``, as do
    missing, non-positive or non-finite levels and a sourced shock that is not a one-month shock.
    """

    name: str
    spot_change: float  # relative, e.g. -0.20 = -20 %
    vol_multiplier: float  # implied vol is multiplied by this factor
    rate_change: float  # absolute, e.g. -0.015 = -150 bp
    days: int = 0  # calendar days the episode lasted
    description: str = ""
    #: True when ``vol_multiplier`` is a one-month (VIX-like) shock, which is
    #: damped for longer tenors; False applies it literally to every leg.
    one_month_vol_shock: bool = False
    #: A sourced scenario's one-month volatility index ("VIX", or "VXO" before 1990) and its levels in vol points at
    #: the start and end of the window; empty for a hypothetical shock.
    vol_index: str = ""
    vol_start: float | None = None
    vol_end: float | None = None

    def __post_init__(self) -> None:
        if not self.vol_index and self.vol_start is None and self.vol_end is None:
            return  # a hypothetical shock
        problem = self._sourcing_problem()
        if problem:
            raise ValueError(f"{self.name}: {problem}")

    def _sourcing_problem(self) -> str:
        """What is wrong with a sourced scenario's levels and shock, or "" when they agree."""
        if not self.vol_index:
            return "vol_start and vol_end need the vol_index they are levels of"
        try:
            start, end = _number("vol_start", self.vol_start), _number("vol_end", self.vol_end)
        except ValueError as exc:
            return str(exc)
        if not (start > 0 and end > 0):
            return f"vol_start and vol_end must be positive, got {start:g} and {end:g}"
        if not math.isclose(self.vol_multiplier, end / start, rel_tol=1e-12):
            return f"vol_multiplier {self.vol_multiplier!r} is not vol_end / vol_start = {end / start!r}"
        if not self.one_month_vol_shock:
            return f"a shock calibrated on the {self.vol_index} is a one-month vol shock"
        return ""

    @classmethod
    def sourced(
        cls,
        name: str,
        spot_change: float,
        rate_change: float,
        days: int,
        description: str,
        *,
        vol_index: str,
        vol_start: float,
        vol_end: float,
    ) -> StressScenario:
        """A one-month vol shock calibrated on a volatility index: the multiplier is ``vol_end / vol_start``."""
        try:
            multiplier = vol_end / vol_start
        except (TypeError, ZeroDivisionError):
            multiplier = math.nan  # __post_init__ names the bad level
        return cls(name, spot_change, multiplier, rate_change, days, description, True, vol_index, vol_start, vol_end)


#: Historical analogues, each pinned to its sources by tests/test_scenarios.py. Between two closes: the S&P 500
#: change (S&P Dow Jones Indices; FRED series SP500 for the last ten years), quoted and applied to 0.1 %; the
#: implied-vol multiplier, the ratio of the VIX closes (FRED VIXCLS; the VXO, FRED VXOCLS, for 1987, before the VIX
#: begins), a shock to one-month implied vol that :func:`run_stress_tests` damps for longer tenors; and the change
#: in the Fed funds target (FRED DFEDTAR, and from 16 Dec 2008 the midpoint of the target range, DFEDTARL to
#: DFEDTARU). ``days`` counts the calendar days between the two closes, so a one-session move from a Friday close to
#: a Monday close lasts 3 days.
HISTORICAL_SCENARIOS: tuple[StressScenario, ...] = (
    StressScenario.sourced(
        "Black Monday (1987)",
        spot_change=-0.205,
        rate_change=0.0,
        days=3,
        vol_index="VXO",
        vol_start=36.37,
        vol_end=150.19,
        description="16–19 Oct 1987: S&P 500 282.70 → 224.84 (−20.5%) in one session; VXO 36.37 → 150.19; "
        "Fed funds target unchanged at 7.3125% until 4 Nov.",
    ),
    StressScenario.sourced(
        "Dot-com bust (2000-02)",
        spot_change=-0.491,
        rate_change=-0.0425,
        days=929,
        vol_index="VIX",
        vol_start=23.31,
        vol_end=42.13,
        description="24 Mar 2000 – 9 Oct 2002: S&P 500 1,527.46 → 776.76 (−49.1%); VIX 23.31 → 42.13; "
        "Fed funds target 6.00% → 1.75%.",
    ),
    StressScenario.sourced(
        "Global Financial Crisis (2007-09)",
        spot_change=-0.568,
        rate_change=-0.04625,
        days=517,
        vol_index="VIX",
        vol_start=16.12,
        vol_end=49.68,
        description="9 Oct 2007 – 9 Mar 2009: S&P 500 1,565.15 → 676.53 (−56.8%); VIX 16.12 → 49.68 "
        "(peak close 80.86 on 20 Nov 2008); Fed funds target 4.75% → 0–0.25% (midpoint 0.125%).",
    ),
    StressScenario.sourced(
        "Volmageddon (2018)",
        spot_change=-0.041,
        rate_change=0.0,
        days=3,
        vol_index="VIX",
        vol_start=17.31,
        vol_end=37.32,
        description="2–5 Feb 2018: S&P 500 2,762.13 → 2,648.94 (−4.1%) in one session; VIX 17.31 → 37.32, "
        "its largest one-day rise at the time; Fed funds target unchanged at 1.25–1.50%.",
    ),
    StressScenario.sourced(
        "COVID crash (2020)",
        spot_change=-0.339,
        rate_change=-0.015,
        days=33,
        vol_index="VIX",
        vol_start=14.38,
        vol_end=61.59,
        description="19 Feb – 23 Mar 2020: S&P 500 3,386.15 → 2,237.40 (−33.9%) in 23 sessions; VIX 14.38 → 61.59 "
        "(record close 82.69 on 16 Mar); Fed funds target 1.50–1.75% → 0–0.25%.",
    ),
    StressScenario.sourced(
        "2022 rate shock",
        spot_change=-0.254,
        rate_change=0.03,
        days=282,
        vol_index="VIX",
        vol_start=16.60,
        vol_end=33.57,
        description="3 Jan – 12 Oct 2022: S&P 500 4,796.56 → 3,577.03 (−25.4%); VIX 16.60 → 33.57; "
        "Fed funds target 0–0.25% → 3.00–3.25%.",
    ),
    StressScenario.sourced(
        "Tariff shock (Apr 2025)",
        spot_change=-0.121,
        rate_change=0.0,
        days=6,
        vol_index="VIX",
        vol_start=21.51,
        vol_end=52.33,
        description="2–8 Apr 2025: S&P 500 5,670.97 → 4,982.77 (−12.1%) in four sessions; VIX 21.51 → 52.33; "
        "Fed funds target unchanged at 4.25–4.50%.",
    ),
)

HYPOTHETICAL_SCENARIOS: tuple[StressScenario, ...] = (
    StressScenario("Rally +15%, IV −30%", 0.15, 0.7, 0.0, 0, "Relief rally with implied volatility collapsing."),
    StressScenario(
        "Vol crush (IV −40%)",
        0.0,
        0.6,
        0.0,
        0,
        "Implied volatility falls 40% with spot unchanged (e.g. after earnings).",
    ),
    StressScenario("Vol spike (IV +50%)", 0.0, 1.5, 0.0, 0, "Implied volatility rises 50% with spot unchanged."),
    StressScenario("Rates +200 bp", 0.0, 1.0, 0.02, 0, "Parallel 2-point rise in the risk-free rate."),
    # Table 2.A of the Fed's document gives the Q4 2025 jump-off and table 4.A the scenario's quarters. Its VIX is
    # the highest close of each quarter: the 26.4 jump-off is the close of 26.42 on 20 Nov 2025 (FRED VIXCLS); the
    # narrative rounds it to 26, which would give 2.77 instead of 2.73. The equity index is an end-of-quarter
    # value, so the window runs from 31 Dec 2025 to 30 Sep 2026. The VIX peaks in Q2 2026 and equities bottom in
    # Q3 2026, when the Fed's VIX is 70.9: applying both at once puts a vol shock 1.6 % larger than the Fed's
    # (72.0 / 70.9) on the equity trough.
    StressScenario.sourced(
        "Fed severely adverse (2026)",
        spot_change=-0.578,
        rate_change=-0.036,
        days=273,
        vol_index="VIX",
        vol_start=26.4,
        vol_end=72.0,
        description="Federal Reserve 2026 supervisory scenario (federalreserve.gov, Feb 2026), from the Q4 2025 "
        "jump-off: the Dow Jones Total Stock Market Index falls 67,502 → 28,490 (−57.8%) to its Q3 2026 trough; "
        "the VIX (highest close of the quarter) rises 26.4 → 72.0 to its Q2 2026 peak; the 3-month Treasury rate "
        "falls 3.7% → 0.1%. The VIX peak and the equity trough come a quarter apart in the Fed's paths; this "
        "scenario applies both at once.",
    ),
    StressScenario("Gap down −10%", -0.10, 1.4, 0.0, 0, "Overnight gap lower with a volatility pickup."),
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
    simplification, flagged in the ``Legs expired`` column). A ``multiplier`` that is not a finite number raises
    ``ValueError``.
    """
    multiplier = _number("multiplier", multiplier)
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

    Spot changes must be finite and at least -1, volatility changes and the
    multiplier finite and days finite and non-negative; anything else raises
    ``ValueError`` naming the argument.
    """
    _check_vol_mode(vol_mode, vol_tenor_decay)
    multiplier = _number("multiplier", multiplier)
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
        Dollar scaling, e.g. contracts x shares per contract; finite.

    Returns
    -------
    numpy.ndarray
        P&L ($) of shape ``(len(days), len(vol_shocks), len(spot_changes))``.

    Raises
    ------
    ValueError
        Naming the argument, if a spot change is below -1, a day count is negative or any value (the multiplier
        too) is not a finite number, or if ``vol_mode`` is unknown.

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
    multiplier = _number("multiplier", multiplier)
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

#: Default spot shocks of :func:`greek_ladder` in percent (``unit="percent"``).
DEFAULT_LADDER_SHOCKS: tuple[float, ...] = (-20, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 20)
#: Default spot shocks of :func:`greek_ladder` in standard deviations (``unit="sigma"``).
DEFAULT_SIGMA_SHOCKS: tuple[float, ...] = (-3, -2, -1, 0, 1, 2, 3)


def greek_ladder(
    strategy: Strategy,
    market: Market,
    shocks: Sequence[float] | None = None,
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
    shocks : sequence of float, optional
        Spot shocks in ``unit``; the default is :data:`DEFAULT_LADDER_SHOCKS` (-20% to +20%) in percent and
        :data:`DEFAULT_SIGMA_SHOCKS` (-3 to +3) in standard deviations.
    unit : {"percent", "sigma"}, default "percent"
        ``"percent"``: a shock of 5 moves spot by +5%, to ``spot x 1.05``. ``"sigma"``: a shock of ``k`` is a move
        of ``k`` standard deviations of the log price over ``sigma_days``, to ``spot x exp(k sigma sqrt(sigma_days
        / 365))`` with ``sigma = market.volatility`` (the convention of the Risk tab's price rows), so a downward
        shock never takes spot to zero.
    sigma_days : float, default 1.0
        Horizon of one standard deviation in ``"sigma"`` units, in calendar days.
    roll_days : float, default 0.0
        Calendar days that pass before the shock (fractions allowed), so the P&L includes the time decay.
    multiplier : float, default 1.0
        Dollar scaling, e.g. contracts x shares per contract; finite.

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
        Naming the argument, if a shock, ``roll_days``, ``sigma_days`` or ``multiplier`` is not a single finite
        number, if ``roll_days`` is negative or ``sigma_days`` not positive, or if a shock takes spot to zero or
        below (a percent shock of -100 or less) or out of floating-point range.

    Examples
    --------
    >>> from options_analyzer.strategies import build_template
    >>> market = Market(100, 0.05, 0.20, 30 / 365)
    >>> strangle = build_template("Short Strangle", 100, 5)
    >>> ladder = greek_ladder(strangle, market, (-5, 0, 5), multiplier=100)
    >>> ladder[["shock", "spot", "pnl", "delta"]].round(2).to_dict("list")
    {'shock': [-5, 0, 5], 'spot': [95.0, 100.0, 105.0], 'pnl': [-86.45, 0.0, -147.02], 'delta': [41.01, -6.66, -50.75]}

    In standard deviations of the 5-day log return, 20% x sqrt(5 / 365) = 2.34%:

    >>> greek_ladder(strangle, market, unit="sigma", sigma_days=5)["spot"].round(2).tolist()
    [93.22, 95.43, 97.69, 100.0, 102.37, 104.79, 107.27]
    """
    if unit not in ("percent", "sigma"):
        raise ValueError(f"unit must be 'percent' or 'sigma', got {unit!r}")
    roll, horizon = _number("roll_days", roll_days), _number("sigma_days", sigma_days)
    multiplier = _number("multiplier", multiplier)
    if roll < 0:
        raise ValueError(f"roll_days must be a non-negative number of days, got {roll_days!r}")
    if not horizon > 0:
        raise ValueError(f"sigma_days must be a positive number of days, got {sigma_days!r}")
    if shocks is None:
        shocks = DEFAULT_LADDER_SHOCKS if unit == "percent" else DEFAULT_SIGMA_SHOCKS
    try:
        shocks = list(shocks)
    except TypeError:
        raise ValueError(f"shocks must be a sequence of numbers, got {shocks!r}") from None
    sigma = market.volatility * math.sqrt(horizon / DAYS_PER_YEAR)  # one standard deviation of the log price
    t = roll / DAYS_PER_YEAR
    base = multiplier * float(position_value(strategy, market, market.spot, 0.0))
    rows = []
    for shock in shocks:
        k = _number("shock", shock)
        if unit == "percent":
            change = 0.01 * k
        else:
            try:
                change = math.expm1(k * sigma)  # exp(k sigma) - 1 without cancellation for small moves
            except OverflowError:
                change = math.inf
        spot = market.spot * (1.0 + change)
        if not spot > 0:
            raise ValueError(f"a shock of {shock} ({unit}) takes spot to zero or below")
        if not math.isfinite(spot):
            raise ValueError(f"a shock of {shock} ({unit}) takes spot out of floating-point range")
        risk = asdict(position_risk(strategy, market, spot, t, multiplier=multiplier))
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
    values = _axis(name, edges)
    if values.size == 0 or np.any(values <= 0):
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
        Dollar scaling, e.g. contracts x shares per contract; finite.

    Returns
    -------
    pandas.DataFrame
        Index ``expiry_bucket`` (``up_to_7d``, ``7d_to_14d``, ..., ``over_730d``, then ``total``) and columns
        ``strike_bucket`` (low to high strike, then ``total``). Empty buckets are kept, so the shape depends only
        on the edges.

    Raises
    ------
    ValueError
        Naming the argument, if the edges are not positive, finite and increasing, ``strike_axis`` is unknown or
        ``multiplier`` is not a finite number.

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
    multiplier = _number("multiplier", multiplier)
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
        Dollar scaling, e.g. contracts x shares per contract; finite.

    Returns
    -------
    float
        Weighted vega ($).

    Raises
    ------
    ValueError
        Naming the argument, if ``scheme`` is unknown or ``decay`` or ``multiplier`` is not a finite number.

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
    decay, multiplier = _number("decay", decay), _number("multiplier", multiplier)
    total = 0.0
    for leg, tau, vol, vega in _option_legs(strategy, market):
        # Same floor and ratio as the damping in strategies.position_value, so the two agree to first order.
        ratio = max(tau, 1e-8) / VOL_SHOCK_REFERENCE_TENOR
        weight = vol * min(1.0, ratio**-decay) if scheme == "relative" else ratio**-0.5
        total += leg.quantity * vega * weight * bs.GREEKS["vega"].scale
    return multiplier * total
