"""P&L explain: where a position's profit or loss came from, by Greeks and by factor.

Two attributions of the same full-revaluation P&L, the change in the position's value when the market moves from
today's state (spot ``S0``, each leg's implied vol, the rate, now) to a new one (spot ``S1``, the vols plus a shift,
the rate plus a shift, ``days`` later):

* :func:`taylor_explain`, the desk's Greek waterfall: delta, gamma, vega, theta, rho, vanna, volga, charm and veta
  terms from the Greeks today, and the unexplained P&L that they leave. It is second order in spot and vol and
  first order in the rate; the theta term is the time decay by repricing (exact for the passage of time alone, see
  the function), and a quality figure says how much of the P&L the terms miss.
* :func:`shapley_explain`, an exact split over the four factors spot, vol, time and rate: the Shapley value of the
  16 revaluations with each subset of the factors moved. The four parts add up to the P&L (no residual) and do not
  depend on an order of the factors, unlike a sequential attribution, which credits the interactions to whichever
  factor is moved last.

:func:`taylor_explain_by_leg` and :func:`shapley_explain_by_leg` give the same explains leg by leg; the position's
are their sums. :func:`pla_metrics` computes the two statistics of the Basel P&L attribution test (MAR32.34 to
32.41), the Spearman correlation and the Kolmogorov-Smirnov distance between hypothetical and risk-theoretical P&L,
for a backtest to put into zones.

Conventions
-----------
* Every leg is valued as :func:`~options_analyzer.strategies.position_value` values it (Black-Scholes-Merton at the
  leg's own implied vol and remaining time; a share with its accrued dividends), and each leg's P&L is the
  difference of its own two values, so a small P&L is not lost in the rounding of a large total value.
* A vol shift is absolute, in decimals (0.06 = +6 vol points), one number for every leg or one per leg; the legs'
  new vols (``vols``) can be given instead. A rate shift is in decimals (0.0025 = +25 bp) and moves the rate of the
  option legs, as in ``position_value``; ``days`` are calendar days (ACT/365) and may be fractional.
* Results are in dollars after ``multiplier`` (contracts x shares per contract): ``multiplier=100`` gives the P&L of
  one listed contract per unit of leg quantity.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import rankdata

from .strategies import Leg, Market, Strategy, _number, position_risk, position_value

__all__ = [
    "QUALITY_FLOOR",
    "SHAPLEY_FACTORS",
    "TAYLOR_TERMS",
    "PLAMetrics",
    "ShapleyExplain",
    "TaylorExplain",
    "pla_metrics",
    "shapley_explain",
    "shapley_explain_by_leg",
    "taylor_explain",
    "taylor_explain_by_leg",
]

DAYS_PER_YEAR = 365.0
#: The terms of :func:`taylor_explain`, in the order of its waterfall: first order, then second order.
TAYLOR_TERMS = ("delta", "gamma", "vega", "theta", "rho", "vanna", "volga", "charm", "veta")
#: The factors of :func:`shapley_explain`.
SHAPLEY_FACTORS = ("spot", "vol", "time", "rate")
#: The default floor of the quality measure, as a fraction of the gross of the nine terms (the sum of their absolute
#: values): a tenth. See :class:`TaylorExplain`.
QUALITY_FLOOR = 0.1


@dataclass(frozen=True)
class TaylorExplain:
    """A P&L explained by Greeks, in dollars (after the ``multiplier`` of :func:`taylor_explain`).

    With ``dS = S1 - S0`` ($ per share), ``dsigma`` the vol shift in vol points, ``dr`` the rate shift in percent and
    ``days`` the calendar days elapsed, and the Greeks of today in the desk units of
    :class:`~options_analyzer.strategies.PositionRisk` (a term with ``dsigma`` sums over the legs, each with its own
    Greek and its own shift, when the shifts differ):

    Attributes
    ----------
    delta : float
        ``delta x dS``.
    gamma : float
        ``gamma x dS^2 / 2``.
    vega : float
        ``vega x dsigma`` (vega in $ per vol point).
    theta : float
        Change in value over ``days`` with spot, implied vols and rate unchanged, by repricing: for one day, the
        one-day decay of :func:`~options_analyzer.strategies.time_decay`.
    rho : float
        ``rho x dr`` (rho in $ per 1% rate move).
    vanna : float
        ``vanna x dS x dsigma`` (vanna in shares of delta per vol point).
    volga : float
        ``volga x dsigma^2 / 2`` (volga in $ of vega per vol point).
    charm : float
        ``charm x dS x days`` (charm in shares of delta per calendar day).
    veta : float
        ``veta x dsigma x days`` (veta in $ of vega per calendar day).
    unexplained : float
        ``total`` minus the sum of the nine terms.
    total : float
        Full-revaluation P&L: the value at the new spot, vols and rate ``days`` later minus the value today.
    quality : float
        ``|unexplained| / max(|total|, floor)``: 0 for a perfect explain, 0.1 when a tenth of the P&L is unexplained.
        With the default floor it is never above ``1 + 1 / 0.1 = 11``.
    floor : float
        Smallest P&L ($) the quality is measured against, so that a P&L close to zero does not turn a small
        unexplained amount into a large ratio. By default a tenth of the gross of the nine terms, ``0.1 x
        sum(|term|)``. A P&L below it is a small difference of larger effects that offset (a day's time decay
        against the gamma of a spot move, say), and the share of it the Greeks miss says little; small moves need no
        floor, since what they leave unexplained is third order in them. The floor grows with the position's own
        P&L, which a floor in the notional does not: one basis point of the notional of a 7-day index iron condor
        (4 legs x 5,800 x 100 shares) is $232, more than the condor's whole day of time decay, $151.97.
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    vanna: float
    volga: float
    charm: float
    veta: float
    unexplained: float
    total: float
    quality: float
    floor: float

    @property
    def terms(self) -> dict[str, float]:
        """The nine terms, by name, in the order of :data:`TAYLOR_TERMS`."""
        return {name: getattr(self, name) for name in TAYLOR_TERMS}

    @property
    def explained(self) -> float:
        """Sum of the nine terms ($)."""
        return math.fsum(self.terms.values())


@dataclass(frozen=True)
class ShapleyExplain:
    """A P&L split exactly over the factors spot, vol, time and rate, in dollars.

    Each factor's part is its Shapley value: the average, over the 24 orders in which the four factors can be
    moved from today's market to the new one, of the change in value when that factor moves. The parts add up to
    ``total`` up to rounding (a few units in the last place of the legs' values).

    Attributes
    ----------
    spot, vol, time, rate : float
        The parts of the P&L credited to the move in spot, the implied-vol shift, the time elapsed and the rate
        shift ($).
    total : float
        Full-revaluation P&L ($), as in :class:`TaylorExplain`.
    """

    spot: float
    vol: float
    time: float
    rate: float
    total: float

    @property
    def terms(self) -> dict[str, float]:
        """The four parts, by factor, in the order of :data:`SHAPLEY_FACTORS`."""
        return {name: getattr(self, name) for name in SHAPLEY_FACTORS}


@dataclass(frozen=True)
class _Move:
    """The checked move from today's market to the new state, per unit (before the multiplier)."""

    spot: float
    #: Vol shift of each leg (decimal), in the order of ``strategy.legs``.
    vol_shifts: tuple[float, ...]
    rate_shift: float
    days: float
    multiplier: float


def _per_leg(name: str, values: float | Sequence[float], legs: int, noun: str) -> tuple[float, ...]:
    """``values`` as one number per leg: a single number repeated, or a sequence with one ``noun`` per leg."""
    if np.ndim(values) == 0:
        return (_number(name, values),) * legs
    numbers = tuple(_number(name, value) for value in cast(Sequence[float], values))
    if len(numbers) != legs:
        raise ValueError(f"{name} needs one {noun} per leg ({legs}), got {len(numbers)}")
    return numbers


def _vol_shifts(
    legs: Sequence[Leg], market: Market, vol_shift: float | Sequence[float] | None, vols: float | Sequence[float] | None
) -> tuple[float, ...]:
    """Each leg's vol shift, from ``vol_shift`` or from the new ``vols``; a stock leg's is 0."""
    if vol_shift is not None and vols is not None:
        raise ValueError("pass vol_shift or vols, not both")
    today = [market.volatility if leg.volatility is None else leg.volatility for leg in legs]
    if vols is not None:
        new = _per_leg("vols", vols, len(legs), "vol")
        for number, (leg, vol) in enumerate(zip(legs, new, strict=True), start=1):
            if leg.is_option and not vol > 0:
                raise ValueError(f"vols must be positive: leg {number} ({leg.label()}) got {vol:g}")
        # The shift that position_value adds back to today's vol: the new vol up to the rounding of one subtraction.
        return tuple(vol - old if leg.is_option else 0.0 for leg, vol, old in zip(legs, new, today, strict=True))
    shifts = _per_leg("vol_shift", 0.0 if vol_shift is None else vol_shift, len(legs), "shift")
    for number, (leg, shift, vol) in enumerate(zip(legs, shifts, today, strict=True), start=1):
        if leg.is_option and not vol + shift > 0:
            raise ValueError(
                f"vol_shift {shift:g} takes the implied vol of leg {number} ({leg.label()}, {vol:g}) to zero or below"
            )
    return shifts


def _check(
    strategy: Strategy,
    market: Market,
    spot: float | None,
    vol_shift: float | Sequence[float] | None,
    vols: float | Sequence[float] | None,
    rate_shift: float,
    days: float,
    multiplier: float,
) -> _Move:
    """The move, with every argument checked; ``ValueError`` naming the argument otherwise."""
    end = float(market.spot) if spot is None else _number("spot", spot)
    if end < 0:
        raise ValueError(f"spot must be a non-negative price, got {end:g}")
    elapsed = _number("days", days)
    if elapsed < 0:
        raise ValueError(f"days must be a non-negative number of calendar days, got {elapsed:g}")
    shifts = _vol_shifts(strategy.legs, market, vol_shift, vols)
    return _Move(end, shifts, _number("rate_shift", rate_shift), elapsed, _number("multiplier", multiplier))


def _check_floor(floor: float | None) -> float | None:
    """``floor`` checked: None (the default, from the terms) or a positive amount of P&L ($)."""
    if floor is None:
        return None
    value = _number("floor", floor)
    if not value > 0:
        raise ValueError(f"floor must be a positive amount of P&L, got {value:g}")
    return value


def _one(leg: Leg) -> Strategy:
    return Strategy(leg.label(), (leg,))


def _leg_taylor(leg: Leg, market: Market, move: _Move, shift: float, floor: float | None) -> TaylorExplain:
    """The Taylor explain of one leg (``floor`` checked, None for the default)."""
    position = _one(leg)
    risk = position_risk(position, market)  # desk units: vega per vol point, theta per day, rho per 1 %, ...
    d_spot = move.spot - market.spot
    d_vol = 100.0 * shift  # vol points
    d_rate = 100.0 * move.rate_shift  # percent
    t = move.days / DAYS_PER_YEAR
    # The leg today, after the time alone and after the whole move, in one revaluation.
    values = position_value(
        position,
        market,
        spot=np.array([market.spot, market.spot, move.spot]),
        t=np.array([0.0, t, t]),
        vol_shift=np.array([0.0, 0.0, shift]),
        rate_shift=np.array([0.0, 0.0, move.rate_shift]),
    )
    today, later, moved = np.broadcast_to(np.asarray(values, dtype=float), (3,)).tolist()
    m = move.multiplier
    terms = {
        "delta": risk.delta * d_spot,
        "gamma": 0.5 * risk.gamma * d_spot * d_spot,
        "vega": risk.vega * d_vol,
        # By repricing rather than theta x days, which misses (d2V/dt2) dt^2 / 2 and the higher orders in time: 0.8 %
        # of a 30-day at-the-money straddle's one-day decay, and half the decay of a 1-day option, whose value falls
        # as the square root of the time left (the dt^2 term is then a quarter of what is missed; see time_decay).
        "theta": later - today,
        "rho": risk.rho * d_rate,
        "vanna": risk.vanna * d_spot * d_vol,
        "volga": 0.5 * risk.volga * d_vol * d_vol,
        "charm": risk.charm * d_spot * move.days,
        "veta": risk.veta * d_vol * move.days,
    }
    terms = {name: m * value for name, value in terms.items()}
    total = m * (moved - today)
    unexplained = total - math.fsum(terms.values())
    return TaylorExplain(**terms, unexplained=unexplained, total=total, **_quality(terms, unexplained, total, floor))


def _quality(terms: Mapping[str, float], unexplained: float, total: float, floor: float | None) -> dict[str, float]:
    """The quality and its floor ($): ``floor`` if given, else a tenth of the gross of ``terms``."""
    if floor is None:
        floor = QUALITY_FLOOR * math.fsum(abs(value) for value in terms.values())
    # Nothing unexplained is a perfect explain, also with no P&L at all (no move, a zero multiplier): not 0 / 0.
    # Otherwise the scale is positive: a zero P&L with something unexplained has terms that sum to minus it, so a
    # positive gross (only an underflow of the tenth could leave a zero scale).
    scale = max(abs(total), floor)
    quality = 0.0 if unexplained == 0 else (abs(unexplained) / scale if scale > 0 else math.inf)
    return {"quality": quality, "floor": floor}


def _sum_taylor(parts: Sequence[TaylorExplain], floor: float | None) -> TaylorExplain:
    """The position's explain: the sum of its legs' (math.fsum, so the order of the legs does not matter).

    The default floor is a tenth of the gross of the position's terms, after the legs' terms have offset each other.
    """
    sums = {name: math.fsum(getattr(part, name) for part in parts) for name in (*TAYLOR_TERMS, "unexplained", "total")}
    terms = {name: sums[name] for name in TAYLOR_TERMS}
    return TaylorExplain(**sums, **_quality(terms, sums["unexplained"], sums["total"], floor))


def taylor_explain_by_leg(
    strategy: Strategy,
    market: Market,
    spot: float | None = None,
    *,
    vol_shift: float | Sequence[float] | None = None,
    vols: float | Sequence[float] | None = None,
    rate_shift: float = 0.0,
    days: float = 0.0,
    multiplier: float = 1.0,
    floor: float | None = None,
) -> tuple[TaylorExplain, ...]:
    """The Greek explain of each leg, in the order of ``strategy.legs``; they add up to :func:`taylor_explain`.

    Takes the arguments of :func:`taylor_explain`. Each leg's quality is measured against its own P&L, with the
    default floor of a tenth of the gross of the leg's own terms, or the ``floor`` given.

    Examples
    --------
    >>> from options_analyzer.strategies import build_template
    >>> market = Market(100, 0.04, 0.20, 30 / 365)
    >>> straddle = build_template("Short Straddle", 100, 5)
    >>> call, put = taylor_explain_by_leg(straddle, market, 96, vol_shift=0.06, days=1, multiplier=100)
    >>> round(call.delta, 2), round(put.delta, 2), round(call.delta + put.delta, 2)
    (213.71, -186.29, 27.42)
    """
    move = _check(strategy, market, spot, vol_shift, vols, rate_shift, days, multiplier)
    return _taylor_legs(strategy, market, move, _check_floor(floor))


def _taylor_legs(strategy: Strategy, market: Market, move: _Move, floor: float | None) -> tuple[TaylorExplain, ...]:
    return tuple(
        _leg_taylor(leg, market, move, shift, floor) for leg, shift in zip(strategy.legs, move.vol_shifts, strict=True)
    )


def taylor_explain(
    strategy: Strategy,
    market: Market,
    spot: float | None = None,
    *,
    vol_shift: float | Sequence[float] | None = None,
    vols: float | Sequence[float] | None = None,
    rate_shift: float = 0.0,
    days: float = 0.0,
    multiplier: float = 1.0,
    floor: float | None = None,
) -> TaylorExplain:
    """Explain the P&L of a market move by the Greeks of today: the desk's second-order waterfall.

    The full-revaluation P&L (``total``) is split into the terms of a Taylor expansion of each leg's value around
    today's market, with the Greeks of :func:`~options_analyzer.strategies.position_risk`::

        delta dS + gamma dS^2 / 2 + vega dsigma + theta + rho dr
            + vanna dS dsigma + volga dsigma^2 / 2 + charm dS dt + veta dsigma dt + unexplained

    with ``dS`` the spot move, ``dsigma`` each leg's vol shift, ``dr`` the rate shift and ``dt`` the time elapsed
    (:class:`TaylorExplain` gives each term's units). The theta term is the change in value over ``days`` with
    nothing else moving, by repricing: it is exact for the passage of time alone (a quiet day is explained in full,
    even the day before expiry), where ``theta x days`` leaves ``(d2V/dt2) dt^2 / 2`` and the higher orders in time
    unexplained: 6.5e-4 per unit of a 30-day at-the-money straddle over a day (0.8 % of its decay, 98 % of that the
    ``dt^2`` term), and half the decay of a 1-day at-the-money option, whose value falls as the square root of the
    time left, which no series in powers of the time follows (the ``dt^2`` term is only a quarter of that gap).
    Charm and veta then correct the delta and vega of today for the time elapsed. What the terms leave out is third
    order in the spot and vol moves, so halving those moves divides the unexplained P&L by about 8, plus the
    second-order terms of a rate move, mostly ``(d2V/dSdr) dS dr``: on the stress day below with a 25 bp rate rise,
    $1.14 per contract, against a rho term of -$0.05. A leg that expires within ``days`` is valued at its intrinsic
    value, which no Taylor expansion of today's value follows: the unexplained P&L, and the quality, show it.

    Parameters
    ----------
    strategy, market : Strategy, Market
        The position and today's market.
    spot : float, optional
        Spot price after the move ($ per share), finite and non-negative; defaults to ``market.spot`` (no move).
    vol_shift : float or sequence of float, optional
        Absolute change in implied vol (0.06 = +6 vol points), for every leg or one per leg in the order of
        ``strategy.legs`` (a shift for a stock leg is ignored); none by default. A shift must leave every option
        leg's vol positive.
    vols : float or sequence of float, optional
        The legs' new implied vols (0.26 = 26 %), in place of ``vol_shift``: one for every leg or one per leg (an
        entry for a stock leg is ignored), positive for an option leg. Each option leg's vol shift is then its new
        vol minus its vol today (the leg's own, or ``market.volatility``), as when re-marking the legs to a surface.
    rate_shift : float, default 0.0
        Change in the risk-free rate (0.0025 = +25 bp), finite.
    days : float, default 0.0
        Calendar days elapsed (fractions allowed), finite and non-negative.
    multiplier : float, default 1.0
        Dollar scaling, e.g. contracts x shares per contract; finite.
    floor : float, optional
        Floor of the quality measure ($), positive, e.g. a desk's own threshold; by default a tenth of the gross of
        the nine terms (see :class:`TaylorExplain`).

    Returns
    -------
    TaylorExplain
        The nine terms, the unexplained P&L, the full-revaluation P&L and the quality, in dollars.

    Raises
    ------
    ValueError
        Naming the argument, if a number is not finite, ``spot`` or ``days`` is negative, ``vol_shift`` or ``vols``
        has a number of entries other than the number of legs or leaves an option leg's implied vol at zero or below,
        both are given, or ``floor`` is not positive.

    Examples
    --------
    A short at-the-money straddle with 30 days to expiry (spot and strike 100, vol 20 %, rate 4 %), one contract, on
    a day when spot falls 4 % and implied vol rises 6 points: the Greeks explain all but $25.17 of the $188.36 loss.

    >>> from options_analyzer.strategies import build_template
    >>> market = Market(100, 0.04, 0.20, 30 / 365)
    >>> straddle = build_template("Short Straddle", 100, 5)
    >>> explain = taylor_explain(straddle, market, 96, vol_shift=0.06, days=1, multiplier=100)
    >>> {name: round(value, 2) for name, value in explain.terms.items()}  # doctest: +NORMALIZE_WHITESPACE
    {'delta': 27.42, 'gamma': -110.91, 'vega': -136.74, 'theta': 7.69, 'rho': 0.0, 'vanna': -2.73, 'volga': -0.05,
     'charm': -0.46, 'veta': 2.26}
    >>> round(explain.total, 2), round(explain.unexplained, 2), round(explain.quality, 3)
    (-188.36, 25.17, 0.134)
    """
    move = _check(strategy, market, spot, vol_shift, vols, rate_shift, days, multiplier)
    floor = _check_floor(floor)
    return _sum_taylor(_taylor_legs(strategy, market, move, floor), floor)


#: Weight of a marginal contribution in the Shapley value of four factors, by the number of factors already moved:
#: k! (3 - k)! / 4!, the share of the 24 orders of the factors in which exactly those k come first.
_SHAPLEY_WEIGHTS = tuple(math.factorial(k) * math.factorial(3 - k) / math.factorial(4) for k in range(4))
#: Which factors each of the 16 revaluations moves: row ``mask``, column ``i`` is bit ``i`` of ``mask``.
_SUBSETS = (np.arange(16)[:, None] >> np.arange(4)) & 1


def _leg_shapley(leg: Leg, market: Market, move: _Move, shift: float) -> ShapleyExplain:
    """The Shapley explain of one leg: its 16 revaluations in one call, then each factor's weighted contributions."""
    spot_moved, vol_moved, time_moved, rate_moved = (_SUBSETS[:, i].astype(bool) for i in range(4))
    values = position_value(
        _one(leg),
        market,
        spot=np.where(spot_moved, move.spot, market.spot),
        t=np.where(time_moved, move.days / DAYS_PER_YEAR, 0.0),
        vol_shift=np.where(vol_moved, shift, 0.0),
        rate_shift=np.where(rate_moved, move.rate_shift, 0.0),
    )
    values = np.broadcast_to(np.asarray(values, dtype=float), (16,)).tolist()
    parts = {}
    for i, name in enumerate(SHAPLEY_FACTORS):
        bit = 1 << i
        parts[name] = move.multiplier * math.fsum(
            _SHAPLEY_WEIGHTS[int(_SUBSETS[mask].sum())] * (values[mask | bit] - values[mask])
            for mask in range(16)
            if not mask & bit
        )
    return ShapleyExplain(**parts, total=move.multiplier * (values[15] - values[0]))


def shapley_explain_by_leg(
    strategy: Strategy,
    market: Market,
    spot: float | None = None,
    *,
    vol_shift: float | Sequence[float] | None = None,
    vols: float | Sequence[float] | None = None,
    rate_shift: float = 0.0,
    days: float = 0.0,
    multiplier: float = 1.0,
) -> tuple[ShapleyExplain, ...]:
    """The factor explain of each leg, in the order of ``strategy.legs``; they add up to :func:`shapley_explain`.

    Takes the arguments of :func:`shapley_explain`. The Shapley value is additive, so the legs' parts add up to the
    position's exactly (up to rounding).

    Examples
    --------
    >>> from options_analyzer.strategies import build_template
    >>> market = Market(100, 0.04, 0.20, 30 / 365)
    >>> call, put = shapley_explain_by_leg(build_template("Short Straddle", 100, 5), market, 96, multiplier=100)
    >>> round(call.spot, 2), round(put.spot, 2)
    (158.67, -241.33)
    """
    move = _check(strategy, market, spot, vol_shift, vols, rate_shift, days, multiplier)
    return tuple(
        _leg_shapley(leg, market, move, shift) for leg, shift in zip(strategy.legs, move.vol_shifts, strict=True)
    )


def shapley_explain(
    strategy: Strategy,
    market: Market,
    spot: float | None = None,
    *,
    vol_shift: float | Sequence[float] | None = None,
    vols: float | Sequence[float] | None = None,
    rate_shift: float = 0.0,
    days: float = 0.0,
    multiplier: float = 1.0,
) -> ShapleyExplain:
    """Split the P&L of a market move exactly over spot, vol, time and rate (Shapley attribution).

    Each leg is revalued with every subset of the four factors moved to its new value and the others at today's
    (16 revaluations); a factor's part is its Shapley value, the weighted sum of the change in value when it moves
    on top of each subset of the others::

        part(i) = sum over subsets A of the other factors of  |A|! (3 - |A|)! / 4!  x  (V(A + i) - V(A))

    which is the average of its contribution over the 24 orders in which the factors could be moved one after the
    other. The parts add up to the full-revaluation P&L, leave no residual and do not depend on an order of the
    factors: a sequential attribution (move spot, then vol, ...) credits each interaction, such as the vanna of a
    spot and a vol move, to whichever factor comes later: on the stress day below, the spot part ranges from
    -$84.87 to -$59.72 per contract and the vol part from -$136.77 to -$111.18 over the 24 orders.

    Parameters
    ----------
    strategy, market, spot, vol_shift, vols, rate_shift, days, multiplier
        As in :func:`taylor_explain`; the vol factor moves every leg's vol to its new value at once.

    Returns
    -------
    ShapleyExplain
        The parts of the spot, vol, time and rate moves and the full-revaluation P&L, in dollars.

    Raises
    ------
    ValueError
        As :func:`taylor_explain` (there is no floor).

    Examples
    --------
    The straddle's stress day of :func:`taylor_explain`, split by factor:

    >>> from options_analyzer.strategies import build_template
    >>> market = Market(100, 0.04, 0.20, 30 / 365)
    >>> straddle = build_template("Short Straddle", 100, 5)
    >>> explain = shapley_explain(straddle, market, 96, vol_shift=0.06, days=1, multiplier=100)
    >>> {name: round(value, 2) for name, value in explain.terms.items()}
    {'spot': -72.17, 'vol': -124.04, 'time': 7.85, 'rate': 0.0}
    >>> round(explain.total, 2)
    -188.36
    """
    parts = shapley_explain_by_leg(
        strategy,
        market,
        spot,
        vol_shift=vol_shift,
        vols=vols,
        rate_shift=rate_shift,
        days=days,
        multiplier=multiplier,
    )
    sums = {f.name: math.fsum(getattr(part, f.name) for part in parts) for f in fields(ShapleyExplain)}
    return ShapleyExplain(**sums)


# ---------------------------------------------------------------------------
# P&L attribution test statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PLAMetrics:
    """The two statistics of the Basel P&L attribution test (MAR32.34 to 32.41).

    Attributes
    ----------
    spearman : float
        Spearman rank correlation of the two P&L series, in [-1, 1]; NaN when either series is constant (it has no
        ranking).
    ks : float
        Kolmogorov-Smirnov distance between their empirical distributions, the largest absolute difference of the
        two empirical cumulative distribution functions, in [0, 1].
    observations : int
        Number of days in each series (MAR32.35 uses the latest 250).
    """

    spearman: float
    ks: float
    observations: int


def _series(name: str, values: ArrayLike) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a sequence of numbers") from None
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite numbers")
    return array


def pla_metrics(hpl: ArrayLike, rtpl: ArrayLike) -> PLAMetrics:
    """Spearman correlation and Kolmogorov-Smirnov distance between hypothetical and risk-theoretical P&L.

    The P&L attribution test of the Basel framework compares a desk's hypothetical P&L (HPL: the day's P&L of the
    positions held, revalued by the front-office pricing) with its risk-theoretical P&L (RTPL: the same, from the
    risk model's factors and pricing). As MAR32.36 to 32.41 define them:

    * the Spearman correlation is the correlation of the two series' ranks (rank 1 for the lowest P&L; tied values
      share their average rank, which MAR32 leaves unspecified);
    * the KS statistic is the largest absolute difference between the empirical cumulative distribution functions
      of the two series, ``F(x) = (number of observations <= x) / n``, over all P&L values.

    MAR32.42 puts a desk in the green zone when the correlation is above 0.80 and the KS statistic below 0.09, and
    in the red zone when the correlation is below 0.70 or the KS statistic above 0.12; this function only computes
    the two statistics.

    Parameters
    ----------
    hpl, rtpl : array_like
        The two P&L series, day by day, of equal length (at least 2) and finite.

    Returns
    -------
    PLAMetrics
        The Spearman correlation, the KS statistic and the number of days.

    Raises
    ------
    ValueError
        Naming the argument, if a series is not one-dimensional or holds a value that is not a finite number, or if
        the two have different lengths or fewer than 2 days.

    Examples
    --------
    >>> metrics = pla_metrics([1.0, -2.0, 0.5, 3.0], [0.9, -1.8, 0.7, 2.5])
    >>> metrics.spearman, metrics.ks
    (1.0, 0.25)
    """
    h, r = _series("hpl", hpl), _series("rtpl", rtpl)
    if h.size != r.size:
        raise ValueError(f"hpl and rtpl must cover the same days, got {h.size} and {r.size} observations")
    if h.size < 2:
        raise ValueError(f"the P&L attribution test needs at least 2 days, got {h.size}")
    rank_h = rankdata(h) - (h.size + 1) / 2.0  # centred ranks: the mean rank of n days is (n + 1) / 2
    rank_r = rankdata(r) - (r.size + 1) / 2.0
    spread = math.sqrt(float(rank_h @ rank_h) * float(rank_r @ rank_r))
    spearman = float(np.clip(float(rank_h @ rank_r) / spread, -1.0, 1.0)) if spread > 0 else math.nan
    pooled = np.concatenate([h, r])
    cdf_h = np.searchsorted(np.sort(h), pooled, side="right") / h.size
    cdf_r = np.searchsorted(np.sort(r), pooled, side="right") / r.size
    return PLAMetrics(spearman=spearman, ks=float(np.max(np.abs(cdf_h - cdf_r))), observations=int(h.size))
