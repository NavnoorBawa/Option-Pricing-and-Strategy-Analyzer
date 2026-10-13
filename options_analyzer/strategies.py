"""Leg-based option strategies: valuation, P&L, Greeks and payoff analytics.

A :class:`Strategy` is a list of :class:`Leg` objects (calls, puts and
stock). Everything else - the payoff diagram, the mark-to-market P&L at any
date, spot, volatility or rate, the position Greeks, breakevens, maximum
profit and loss, and the probability of profit - is derived from the legs, so
every tab of the app analyses exactly the same position.

Conventions
-----------
* ``quantity`` is signed: positive = long, negative = short. One unit is an
  option on one share, or one share. Multiply by the contract multiplier
  (typically 100) for the dollar P&L of listed contracts.
* Options are European and valued with Black-Scholes-Merton at each leg's
  own implied volatility (defaulting to the market volatility).
* A stock leg is worth its price plus the cash dividends it has paid,
  reinvested at the risk-free rate: ``S_t + S_0 (e^{rt} - e^{(r-q)t})``
  (the expected dividends at yield ``q``). Its delta is therefore exactly 1,
  a covered call's profit stays capped, and the whole position is a
  martingale after discounting, consistently with the option legs.
* Financing of the premium paid or received is ignored by default, as is
  conventional for payoff diagrams (see the ``carry`` option of :func:`pnl`).
* When legs are entered at their model value (no premium override) and
  without fees, the P&L "today" at the current spot is zero.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from . import black_scholes as bs

__all__ = [
    "TEMPLATES",
    "Leg",
    "Market",
    "Strategy",
    "StrategyMetrics",
    "StrategyTemplate",
    "analyze",
    "breakevens",
    "build_template",
    "entry_cost",
    "front_expiry",
    "leg_labels",
    "leg_table",
    "pnl",
    "position_greeks",
    "position_value",
    "strike_text",
    "unbounded_loss_from_dividends",
]

LEG_KINDS = ("call", "put", "stock")


@dataclass(frozen=True)
class Market:
    """Market state. ``expiry`` is the default time to expiry (years) for legs."""

    spot: float
    rate: float
    volatility: float
    expiry: float
    dividend_yield: float = 0.0

    def __post_init__(self):
        if not self.spot > 0:
            raise ValueError("spot must be positive")
        if not self.volatility > 0:
            raise ValueError("volatility must be positive")
        if not self.expiry > 0:
            raise ValueError("time to expiry must be positive")


@dataclass(frozen=True)
class Leg:
    kind: str
    quantity: float
    strike: float = 0.0
    #: Time to expiry in years; ``None`` uses :attr:`Market.expiry`.
    expiry: float | None = None
    #: Implied volatility of this leg; ``None`` uses :attr:`Market.volatility`.
    volatility: float | None = None
    #: Entry price per unit actually paid/received; ``None`` uses the model value.
    premium: float | None = None

    def __post_init__(self):
        kind = str(self.kind).lower()
        if kind not in LEG_KINDS:
            raise ValueError(f"leg kind must be one of {LEG_KINDS}, got {self.kind!r}")
        object.__setattr__(self, "kind", kind)
        if kind != "stock" and not self.strike > 0:
            raise ValueError("option legs need a positive strike")
        if self.expiry is not None and not self.expiry > 0:
            raise ValueError("leg expiry must be positive")
        if self.volatility is not None and not self.volatility > 0:
            raise ValueError("leg volatility must be positive")
        if self.premium is not None and not self.premium >= 0:
            raise ValueError("leg premium must be non-negative")

    @property
    def is_option(self) -> bool:
        return self.kind != "stock"

    def label(self) -> str:
        side = "Long" if self.quantity > 0 else "Short"
        qty = strike_text(abs(self.quantity))
        if self.kind == "stock":
            return f"{side} {qty} share{'s' if abs(self.quantity) != 1 else ''}"
        return f"{side} {qty} × {strike_text(self.strike)} {self.kind.capitalize()}"


def strike_text(value: float, most: int = 8) -> str:
    """A strike (or quantity) with every decimal it has, up to ``most``, and thousands separators.

    ``100``, ``102.5``, ``0.975``, ``12,345.67``: never rounded to a few significant digits or written in
    scientific notation, so a label shows the same strike as the legs table.
    """
    decimals = next(
        (d for d in range(most) if abs(round(value, d) - value) <= 1e-12 * max(1.0, abs(value))),
        most,
    )
    return f"{value:,.{decimals}f}"


@dataclass(frozen=True)
class Strategy:
    name: str
    legs: tuple[Leg, ...]
    description: str = ""
    #: Transaction costs per unit (e.g. commissions), added to the entry cost.
    fees: float = 0.0

    def __post_init__(self):
        legs = tuple(self.legs)
        if not legs:
            raise ValueError("a strategy needs at least one leg")
        object.__setattr__(self, "legs", tuple(leg for leg in legs if leg.quantity != 0) or legs)

    @property
    def strikes(self) -> list[float]:
        return sorted({leg.strike for leg in self.legs if leg.is_option})


def _leg_expiry(leg: Leg, market: Market) -> float:
    return market.expiry if leg.expiry is None else leg.expiry


def _leg_vol(leg: Leg, market: Market) -> float:
    return market.volatility if leg.volatility is None else leg.volatility


def front_expiry(strategy: Strategy, market: Market) -> float:
    """Earliest option expiry in the strategy (``market.expiry`` for stock only)."""
    expiries = [_leg_expiry(leg, market) for leg in strategy.legs if leg.is_option]
    return min(expiries) if expiries else market.expiry


def _days_text(years: float) -> str:
    days = years * 365.0
    return f"{round(days)}d" if abs(days - round(days)) < 1e-6 else f"{days:.1f}d"


def _price_text(price: float) -> str:
    text = f"{price:,.4f}".rstrip("0")
    return text + "0" * (2 - len(text.split(".")[1]))


def leg_labels(strategy: Strategy, market: Market) -> list[str]:
    """Display label of every leg, with the details that tell otherwise identical legs apart.

    Each option's expiry is added when the option legs expire on different
    dates (calendars, diagonals) or a leg's expiry differs from the market
    default, e.g. ``Short 1 × 100 Call (30d)``; an implied volatility other
    than the market's and a fill price entered by hand are added too.
    """
    options = [leg for leg in strategy.legs if leg.is_option]
    expiries = {round(_leg_expiry(leg, market) * 365.0, 6) for leg in options}
    show_expiry = len(expiries) > 1 or any(
        leg.expiry is not None and not np.isclose(leg.expiry, market.expiry, rtol=0, atol=1e-9) for leg in options
    )
    labels = []
    for leg in strategy.legs:
        details = []
        if leg.is_option and show_expiry:
            details.append(_days_text(_leg_expiry(leg, market)))
        if leg.is_option and leg.volatility is not None and not np.isclose(leg.volatility, market.volatility):
            details.append(f"IV {leg.volatility * 100:g}%")
        if leg.premium is not None:
            details.append(f"filled at ${_price_text(leg.premium)}")
        labels.append(leg.label() + (f" ({', '.join(details)})" if details else ""))
    return labels


#: Reference tenor for implied-volatility shocks (one month, like the VIX).
VOL_SHOCK_REFERENCE_TENOR = 30.0 / 365.0


def _dividends(market: Market, t):
    """Value at ``t`` of the expected cash dividends of one share, reinvested at ``r``."""
    r, q = market.rate, market.dividend_yield
    return market.spot * (np.exp(r * np.asarray(t, dtype=float)) - np.exp((r - q) * np.asarray(t, dtype=float)))


def _leg_value(leg, market, spot, t, vol_multiplier, vol_shift, rate_shift, vol_tenor_decay):
    q = market.dividend_yield
    if leg.kind == "stock":
        return spot + _dividends(market, t)
    tau = np.maximum(_leg_expiry(leg, market) - t, 0.0)
    multiplier = vol_multiplier
    if vol_tenor_decay:
        # Longer-dated implied vol moves less than one-month vol: scale the
        # shock by (tau / 1M)^(-decay), never amplifying it for short tenors.
        ratio = np.maximum(tau, 1e-8) / VOL_SHOCK_REFERENCE_TENOR
        multiplier = 1.0 + (np.asarray(vol_multiplier) - 1.0) * np.minimum(1.0, ratio ** (-vol_tenor_decay))
    vol = np.maximum(_leg_vol(leg, market) * multiplier + vol_shift, 1e-8)
    return bs.price(spot, leg.strike, tau, market.rate + rate_shift, vol, q, leg.kind)


def position_value(
    strategy: Strategy,
    market: Market,
    spot=None,
    t=0.0,
    vol_multiplier=1.0,
    vol_shift=0.0,
    rate_shift=0.0,
    vol_tenor_decay: float = 0.0,
):
    """Value of the position (per unit) after ``t`` years.

    ``spot``, ``t``, ``vol_multiplier`` (scales every leg's IV),
    ``vol_shift`` (added to every leg's IV) and ``rate_shift`` broadcast
    against each other, so whole scenario grids are valued in one call.
    With ``vol_tenor_decay > 0`` the IV multiplier is interpreted as the
    shock to one-month implied vol and damped for longer tenors by
    ``(tau / 1M) ** -vol_tenor_decay`` (history suggests about 0.25).
    Legs past their expiry are worth their intrinsic value.
    """
    spot = market.spot if spot is None else np.asarray(spot, dtype=float)
    total = 0.0
    for leg in strategy.legs:
        total = total + leg.quantity * _leg_value(
            leg, market, spot, t, vol_multiplier, vol_shift, rate_shift, vol_tenor_decay
        )
    return total


def entry_cost(strategy: Strategy, market: Market) -> float:
    """Net premium: positive for a debit (cash paid), negative for a credit.

    Legs with an explicit ``premium`` use it; the others use their model value.
    Transaction costs (``strategy.fees``) are included.
    """
    total = strategy.fees
    for leg in strategy.legs:
        if leg.premium is not None:
            total += leg.quantity * leg.premium
        else:
            total += leg.quantity * float(_leg_value(leg, market, market.spot, 0.0, 1.0, 0.0, 0.0, 0.0))
    return float(total)


def pnl(strategy: Strategy, market: Market, spot=None, t=0.0, carry: bool = False, **shocks):
    """Mark-to-market P&L per unit relative to the entry cost.

    With ``carry=True`` the entry cost is financed at the risk-free rate until
    ``t`` (a debit is borrowed, a credit earns interest). The risk-neutral
    expected P&L is then zero whenever every leg is entered at its model value
    at the market volatility and there are no fees; any premium override, leg
    IV different from the market volatility or fee shows up as the edge.
    """
    cost = entry_cost(strategy, market)
    if carry:
        cost = cost * np.exp(market.rate * np.asarray(t, dtype=float))
    return position_value(strategy, market, spot, t, **shocks) - cost


def position_greeks(strategy: Strategy, market: Market, spot=None, t: float = 0.0) -> dict:
    """First-order Greeks of the whole position (per unit, raw model units)."""
    spot = market.spot if spot is None else spot
    totals = dict.fromkeys(("delta", "gamma", "theta", "vega", "rho"), 0.0)
    r, q = market.rate, market.dividend_yield
    for leg in strategy.legs:
        if leg.kind == "stock":
            totals["delta"] = totals["delta"] + leg.quantity
            # Dividend accrual: d/dt of S0 (e^{rt} - e^{(r-q)t}), and its sensitivity to r.
            totals["theta"] = totals["theta"] + leg.quantity * market.spot * (
                r * np.exp(r * t) - (r - q) * np.exp((r - q) * t)
            )
            totals["rho"] = totals["rho"] + leg.quantity * market.spot * t * (np.exp(r * t) - np.exp((r - q) * t))
            continue
        tau = max(_leg_expiry(leg, market) - t, 0.0)
        g = bs.greeks(spot, leg.strike, tau, market.rate, _leg_vol(leg, market), market.dividend_yield, leg.kind)
        for name in totals:
            totals[name] = totals[name] + leg.quantity * g[name]
    return totals


def leg_table(strategy: Strategy, market: Market) -> list[dict]:
    """Per-leg premium and Greeks at inception (per unit, display units)."""
    rows = []
    for leg in strategy.legs:
        if leg.kind == "stock":
            unit_price = market.spot
            # Theta of a share is its dividend accrual, q * S per year.
            g = {"delta": 1.0, "gamma": 0.0, "theta": market.dividend_yield * market.spot, "vega": 0.0}
        else:
            tau = _leg_expiry(leg, market)
            vol = _leg_vol(leg, market)
            unit_price = bs.price(market.spot, leg.strike, tau, market.rate, vol, market.dividend_yield, leg.kind)
            g = bs.greeks(market.spot, leg.strike, tau, market.rate, vol, market.dividend_yield, leg.kind)
        entry = unit_price if leg.premium is None else leg.premium
        rows.append(
            {
                "Leg": leg.label(),
                "Type": leg.kind.capitalize(),
                "Quantity": leg.quantity,
                "Strike": leg.strike if leg.is_option else None,
                "Days to expiry": round(_leg_expiry(leg, market) * 365) if leg.is_option else None,
                "IV": _leg_vol(leg, market) if leg.is_option else None,
                "Model value": unit_price,
                "Entry price": entry,
                "Cost": leg.quantity * entry,
                "Delta": leg.quantity * g["delta"],
                "Gamma": leg.quantity * g["gamma"],
                "Theta / day": leg.quantity * g["theta"] / 365.0,
                "Vega / vol pt": leg.quantity * g["vega"] / 100.0,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Payoff analytics
# ---------------------------------------------------------------------------


def _live_vol_horizon(strategy: Strategy, market: Market, horizon: float) -> float:
    """Largest volatility x sqrt(time) relevant at ``horizon``: the price distribution and any live leg."""
    spread = market.volatility * np.sqrt(max(horizon, 1e-4))
    for leg in strategy.legs:
        if leg.is_option:
            remaining = _leg_expiry(leg, market) - horizon
            if remaining > 0:
                spread = max(spread, _leg_vol(leg, market) * np.sqrt(remaining))
    return spread


#: The analysis grid ends at most this many log-units above max(spot, strikes) (a factor of a million). Further
#: out the P&L is the difference of terms so large that rounding hides it, and its behaviour is known
#: analytically (upside slope and S -> infinity limit, see :func:`_tail_root`).
_MAX_GRID_WIDTH = float(np.log(1e6))


#: The analysis grid starts at most this many log-units below the spot (e^-650 is about 5e-283, so the grid stays
#: within the normal float range).
_MAX_GRID_DEPTH = 650.0


def _analysis_grid(strategy: Strategy, market: Market, horizon: float) -> np.ndarray:
    """Spot grid containing every kink and spanning ~8 standard deviations of every relevant distribution.

    Downwards it reaches below the bulk of the price distribution at ``horizon``, whose median lies
    sigma^2 h / 2 (plus the dividend yield) below the spot in log terms: at extreme sigma sqrt(h) nearly all the
    probability sits under 1e-6 x spot, and a breakeven or the edge of a zero-P&L region must not hide in a single
    gap between S = 0 and the first grid point there.
    """
    width = min(8.0 * _live_vol_horizon(strategy, market, horizon), _MAX_GRID_WIDTH)
    hi = max([market.spot, *strategy.strikes]) * max(np.exp(width), 3.0)
    lo = 1e-6 * market.spot
    vol_t = market.volatility * np.sqrt(max(horizon, 1e-4))
    depth = min(8.0 * vol_t + 0.5 * vol_t**2 + market.dividend_yield * horizon, _MAX_GRID_DEPTH)
    bottom = market.spot * np.exp(-depth)
    grid = np.concatenate(
        [
            np.linspace(0.0, hi, 4001),
            np.geomspace(bottom, lo, 1001) if 0.0 < bottom < lo else [],
            np.geomspace(lo, hi, 4001),
            np.asarray(strategy.strikes, dtype=float),
            [market.spot],
        ]
    )
    return np.unique(grid)


def _pnl_and_scale(strategy: Strategy, market: Market, spots, horizon: float, carry: bool):
    """P&L at ``horizon`` and the magnitude of the terms summed into it (for rounding tolerances)."""
    cost = entry_cost(strategy, market)
    if carry:
        cost *= float(np.exp(market.rate * horizon))
    spots = np.asarray(spots, dtype=float)
    total = np.zeros_like(spots)
    scale = np.full_like(spots, abs(cost))
    for leg in strategy.legs:
        value = leg.quantity * _leg_value(leg, market, spots, horizon, 1.0, 0.0, 0.0, 0.0)
        total = total + value
        scale = scale + np.abs(value)
    return total - cost, scale


def _sign_classes(values: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """+1 / -1 / 0 per point, treating only rounding-level P&L (relative to its terms) as zero."""
    tol = 1e-13 * scale
    return np.where(values > tol, 1, np.where(values < -tol, -1, 0))


def _upside_slope(strategy: Strategy, market: Market, horizon: float) -> float:
    """d(P&L)/dS as S -> infinity at ``horizon`` (per unit)."""
    slope = 0.0
    q = market.dividend_yield
    for leg in strategy.legs:
        if leg.kind == "stock":
            slope += leg.quantity
        elif leg.kind == "call":
            tau = _leg_expiry(leg, market) - horizon
            slope += leg.quantity * (np.exp(-q * tau) if tau > 0 else 1.0)
    return slope


def unbounded_loss_from_dividends(strategy: Strategy, market: Market, horizon: float | None = None) -> bool:
    """True when the loss is unbounded only because of the dividend yield.

    A European call expiring after ``horizon`` gains ``e^{-q tau} < 1`` per
    unit rise of the stock, so a long-dated call that "covers" a short
    front-month call (calendar, diagonal) leaves a small negative upside slope.
    """
    horizon = front_expiry(strategy, market) if horizon is None else horizon
    tol = _slope_tolerance(strategy)
    without = replace(market, dividend_yield=0.0)
    return _upside_slope(strategy, market, horizon) < -tol <= _upside_slope(strategy, without, horizon)


def _pnl_at_infinity(strategy: Strategy, market: Market, horizon: float, carry: bool) -> tuple[float, float]:
    """Limit of the P&L as S -> infinity when the upside slope is zero, and the size of the terms summed."""
    total = magnitude = 0.0
    for leg in strategy.legs:
        term = 0.0
        if leg.kind == "call":
            tau = _leg_expiry(leg, market) - horizon
            term = -leg.quantity * leg.strike * (np.exp(-market.rate * tau) if tau > 0 else 1.0)
        elif leg.kind == "stock":
            term = leg.quantity * float(_dividends(market, horizon))
        total += term
        magnitude += abs(term)
    cost = entry_cost(strategy, market) * (float(np.exp(market.rate * horizon)) if carry else 1.0)
    return float(total - cost), float(magnitude + abs(cost))


def _asymptotic_sign(strategy: Strategy, market: Market, horizon: float, carry: bool) -> int:
    """Sign of the P&L as S -> infinity (0 when it tends to zero within rounding)."""
    slope = _upside_slope(strategy, market, horizon)
    if abs(slope) > _slope_tolerance(strategy):
        return int(np.sign(slope))
    limit, magnitude = _pnl_at_infinity(strategy, market, horizon, carry)
    return int(np.sign(limit)) if abs(limit) > 1e-13 * magnitude else 0


def _transitions(strategy, market, grid, classes, horizon, carry) -> dict[int, float]:
    """Location of every change of sign class between grid points ``i`` and ``i + 1``.

    A strict sign change is refined with Brent's method; at the edge of a
    zero-P&L region the transition sits on the region's last zero point.
    The root tolerance is relative only: an absolute one (such as 1e-12 x spot)
    let a tiny positive root in the first bracket [0, grid[1]] collapse onto
    S = 0, which put the whole probability below the root on the wrong side.
    """

    def f(s):
        return float(pnl(strategy, market, s, horizon, carry))

    out = {}
    for i in np.flatnonzero(classes[:-1] != classes[1:]):
        a, b = classes[i], classes[i + 1]
        if a != 0 and b != 0:
            out[int(i)] = float(brentq(f, grid[i], grid[i + 1], xtol=1e-300, rtol=1e-12, maxiter=500))
        else:
            out[int(i)] = float(grid[i] if a == 0 else grid[i + 1])
    return out


def _slope_tolerance(strategy: Strategy) -> float:
    return 1e-9 * max(1.0, sum(abs(leg.quantity) for leg in strategy.legs))


def _tail_root(strategy, market, horizon, carry, end_spot: float, end_class: int) -> float | None:
    """Breakeven beyond the end of the analysis grid, if the P&L must still change sign out there.

    That happens when the sign of the P&L at the end of the grid differs from
    its asymptotic sign (the sign of the upside slope, or of the S -> infinity
    limit when the slope is zero), e.g. a diagonal spread whose dividend-paying
    long call gains a little less than the short call loses.
    """
    target = _asymptotic_sign(strategy, market, horizon, carry)
    if target == 0 or end_class == 0 or end_class == target:
        return None
    lo = hi = end_spot
    for _ in range(200):  # up to 2^200 times the end of the grid
        lo, hi = hi, 2.0 * hi
        value, scale = _pnl_and_scale(strategy, market, np.array([hi]), horizon, carry)
        cls = int(_sign_classes(value, scale)[0])
        if cls == target:
            return float(brentq(lambda s: float(pnl(strategy, market, s, horizon, carry)), lo, hi, xtol=1e-12 * hi))
        if cls == 0:  # rounding hides the sign before it flips: no resolvable root
            return None
    return None


@dataclass(frozen=True)
class _SignProfile:
    grid: np.ndarray
    values: np.ndarray
    classes: np.ndarray
    transitions: dict[int, float]
    tail: float | None  # breakeven beyond the grid, see _tail_root

    @property
    def breakevens(self) -> list[float]:
        return sorted([*self.transitions.values(), *([self.tail] if self.tail is not None else [])])


def _sign_profile(strategy: Strategy, market: Market, horizon: float, carry: bool) -> _SignProfile:
    grid = _analysis_grid(strategy, market, horizon)
    values, scale = _pnl_and_scale(strategy, market, grid, horizon, carry)
    classes = _sign_classes(values, scale)
    if classes[-1] == 0:
        # Far out, the P&L is a small difference of huge terms and can drop below the rounding tolerance
        # (e.g. a butterfly's tiny debit at extreme volatility). A zero run reaching the end of the grid
        # takes the exact asymptotic sign; it is a genuine zero plateau only if the asymptote is zero.
        target = _asymptotic_sign(strategy, market, horizon, carry)
        if target != 0:
            nonzero = np.flatnonzero(classes != 0)
            classes[(nonzero[-1] + 1) if nonzero.size else 0 :] = target
    transitions = _transitions(strategy, market, grid, classes, horizon, carry)
    tail = _tail_root(strategy, market, horizon, carry, float(grid[-1]), int(classes[-1]))
    return _SignProfile(grid, values, classes, transitions, tail)


def _merge(points: list[float], market: Market) -> list[float]:
    merged: list[float] = []
    for point in sorted(points):
        if not merged or point - merged[-1] > 1e-9 * max(market.spot, 1.0):
            merged.append(point)
    return merged


def breakevens(strategy: Strategy, market: Market, horizon: float | None = None, carry: bool = False) -> list[float]:
    """Spot prices at which the P&L at ``horizon`` (default: front expiry) changes between loss, zero and profit.

    A zero-P&L plateau (e.g. a zero-cost risk reversal between its strikes)
    contributes both of its endpoints.
    """
    horizon = front_expiry(strategy, market) if horizon is None else horizon
    return _merge(_sign_profile(strategy, market, horizon, carry).breakevens, market)


@dataclass(frozen=True)
class StrategyMetrics:
    """Payoff analytics per unit at the front expiry (multiply by the multiplier for $).

    ``max_profit`` is the largest P&L and ``max_loss`` minus the smallest, so
    ``max_loss < 0`` means the position cannot lose (a guaranteed minimum
    profit) and ``max_profit < 0`` that it cannot make money.
    """

    entry_cost: float
    max_profit: float  # np.inf when unlimited
    max_loss: float  # np.inf when unlimited; negative = guaranteed profit
    max_profit_spot: float | None  # None when unlimited or reached as S -> infinity
    max_loss_spot: float | None
    breakevens: list[float]
    probability_of_profit: float
    expected_pnl: float
    pnl_std: float
    horizon: float
    drift: float
    extra: dict = field(default_factory=dict)

    @property
    def reward_to_risk(self) -> float | None:
        if not np.isfinite(self.max_loss) or self.max_loss <= 0 or self.max_profit <= 0:
            return None
        return self.max_profit / self.max_loss


def _terminal_quadrature(market: Market, horizon: float, drift: float, kinks: Sequence[float]):
    """Nodes/weights for E[f(S_h)] with log-normal S_h, split at kinks (Gauss-Legendre).

    The z-window extends to ``12 + 2 sigma sqrt(h)`` on the upside so that the
    mass driving E[S] and E[S^2] (centred near z = sigma sqrt(h) and
    2 sigma sqrt(h)) is always inside it, even for extreme volatilities.
    """
    vol_t = market.volatility * np.sqrt(horizon)
    mean = np.log(market.spot) + (drift - market.dividend_yield - 0.5 * market.volatility**2) * horizon
    lo, hi = -12.0, 12.0 + 2.0 * vol_t
    z_kinks = sorted({float((np.log(k) - mean) / vol_t) for k in kinks if k > 0})
    edges = [lo, *(z for z in z_kinks if lo < z < hi), hi]
    x, w = np.polynomial.legendre.leggauss(96)
    nodes, weights = [], []
    for a, b in pairwise(edges):
        if b - a <= 0:
            continue
        z = 0.5 * (b - a) * x + 0.5 * (a + b)
        nodes.append(z)
        weights.append(0.5 * (b - a) * w * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi))
    z = np.concatenate(nodes)
    return np.exp(mean + vol_t * z), np.concatenate(weights), mean, vol_t


def analyze(strategy: Strategy, market: Market, drift: float | None = None, carry: bool = False) -> StrategyMetrics:
    """Key payoff statistics at the front expiry.

    ``drift`` is the annual expected return of the underlying used for the
    probability of profit and the expected P&L; the default is the risk-free
    rate (the risk-neutral measure). Probabilities assume a log-normal
    terminal price with the market volatility. ``carry`` finances the entry
    cost at the risk-free rate (see :func:`pnl`).
    """
    horizon = front_expiry(strategy, market)
    drift = market.rate if drift is None else drift
    profile = _sign_profile(strategy, market, horizon, carry)
    grid, values, classes, transitions = profile.grid, profile.values, profile.classes, profile.transitions
    slope = _upside_slope(strategy, market, horizon)
    slope_tol = _slope_tolerance(strategy)

    i_max, i_min = int(np.argmax(values)), int(np.argmin(values))
    max_profit, max_loss = float(values[i_max]), float(-values[i_min])
    max_profit_spot, max_loss_spot = float(grid[i_max]), float(grid[i_min])
    if slope > slope_tol:
        max_profit, max_profit_spot = np.inf, None
    elif slope < -slope_tol:
        max_loss, max_loss_spot = np.inf, None
    else:
        # Bounded upside: the extreme may only be reached as S -> infinity.
        limit, _ = _pnl_at_infinity(strategy, market, horizon, carry)
        if limit > max_profit:
            max_profit, max_profit_spot = limit, None
        if -limit > max_loss:
            max_loss, max_loss_spot = -limit, None

    be = _merge(profile.breakevens, market)

    # Probability of profit: log-normal mass of every run of grid points with P&L > 0.
    vol_t = market.volatility * np.sqrt(horizon)
    mean = np.log(market.spot) + (drift - market.dividend_yield - 0.5 * market.volatility**2) * horizon

    def cdf(x: float) -> float:
        if x <= 0:
            return 0.0
        return 1.0 if np.isinf(x) else float(ndtr((np.log(x) - mean) / vol_t))

    pop = 0.0
    positive = np.flatnonzero(classes == 1)
    if positive.size:
        breaks = np.flatnonzero(np.diff(positive) > 1)
        starts = np.concatenate([[positive[0]], positive[breaks + 1]])
        ends = np.concatenate([positive[breaks], [positive[-1]]])
        last = len(grid) - 1
        upper_end = np.inf if profile.tail is None else profile.tail  # a profitable run reaching the grid's end
        for i0, i1 in zip(starts, ends, strict=True):
            a = 0.0 if i0 == 0 else transitions[int(i0) - 1]
            b = upper_end if i1 == last else transitions[int(i1)]
            pop += cdf(b) - cdf(a)

    spots, weights, _, _ = _terminal_quadrature(market, horizon, drift, [*strategy.strikes, *be])
    outcomes = pnl(strategy, market, spots, horizon, carry)
    expected = float(np.dot(weights, outcomes))
    with np.errstate(over="ignore", invalid="ignore"):
        # Two-pass variance: E[X^2] - E[X]^2 cancels catastrophically for (nearly) riskless positions.
        variance = float(np.dot(weights, (outcomes - expected) ** 2))
    # At extreme sigma * sqrt(h) the second moment exceeds float range: report it as infinite.
    std = float(np.sqrt(variance)) if np.isfinite(variance) else np.inf

    return StrategyMetrics(
        entry_cost=entry_cost(strategy, market),
        max_profit=max_profit,
        max_loss=max_loss,
        max_profit_spot=max_profit_spot,
        max_loss_spot=max_loss_spot,
        breakevens=be,
        probability_of_profit=float(min(max(pop, 0.0), 1.0)),
        expected_pnl=expected,
        pnl_std=std,
        horizon=horizon,
        drift=drift,
    )


# ---------------------------------------------------------------------------
# Strategy templates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyTemplate:
    name: str
    category: str
    outlook: str
    description: str
    #: ``build(strike, width, expiry)`` -> legs; ``expiry`` is the default time to expiry in years.
    build: Callable[[float, float, float], tuple[Leg, ...]]
    uses_width: bool = True
    #: Human-readable leg layout, e.g. "+1 call @ K, -1 call @ K+w".
    structure: str = ""
    #: Multiple of ``width`` below the strike used by the lowest leg (for validation).
    lowest_offset: float = 0.0


def _t(name, category, outlook, description, structure, build, uses_width=True, lowest_offset=0.0):
    return StrategyTemplate(name, category, outlook, description, build, uses_width, structure, lowest_offset)


def C(q, k, T=None):
    return Leg("call", q, k, T)


def P(q, k, T=None):
    return Leg("put", q, k, T)


STOCK = Leg("stock", 1)

TEMPLATES: dict[str, StrategyTemplate] = {
    t.name: t
    for t in [
        # Single options
        _t(
            "Long Call",
            "Single option",
            "Bullish",
            "Buy a call: profits from a rise in the underlying with the loss limited to the premium paid.",
            "+1 call @ K",
            lambda k, w, T: (C(1, k),),
            False,
        ),
        _t(
            "Short Call",
            "Single option",
            "Bearish / neutral",
            "Sell a call: collects the premium but has unlimited loss if the underlying rallies.",
            "-1 call @ K",
            lambda k, w, T: (C(-1, k),),
            False,
        ),
        _t(
            "Long Put",
            "Single option",
            "Bearish",
            "Buy a put: profits from a fall in the underlying with the loss limited to the premium paid.",
            "+1 put @ K",
            lambda k, w, T: (P(1, k),),
            False,
        ),
        _t(
            "Short Put",
            "Single option",
            "Bullish / neutral",
            "Sell a put (cash-secured when backed by cash): collects the premium; loses below the breakeven.",
            "-1 put @ K",
            lambda k, w, T: (P(-1, k),),
            False,
        ),
        # Stock + option
        _t(
            "Covered Call",
            "Stock + option",
            "Neutral / mildly bullish",
            "Own the stock and sell a call against it: earns premium income but caps the upside at the strike.",
            "+1 share, -1 call @ K",
            lambda k, w, T: (STOCK, C(-1, k)),
            False,
        ),
        _t(
            "Protective Put",
            "Stock + option",
            "Bullish with protection",
            "Own the stock and buy a put: keeps the upside while flooring the loss at the strike.",
            "+1 share, +1 put @ K",
            lambda k, w, T: (STOCK, P(1, k)),
            False,
        ),
        _t(
            "Collar",
            "Stock + option",
            "Neutral, hedged",
            "Own the stock, buy a downside put and finance it by selling an upside call.",
            "+1 share, +1 put @ K-w, -1 call @ K+w",
            lambda k, w, T: (STOCK, P(1, k - w), C(-1, k + w)),
            lowest_offset=1,
        ),
        # Vertical spreads
        _t(
            "Bull Call Spread",
            "Vertical spread",
            "Moderately bullish",
            "Buy a call and sell a higher-strike call: a cheaper bullish bet with capped profit and loss.",
            "+1 call @ K, -1 call @ K+w",
            lambda k, w, T: (C(1, k), C(-1, k + w)),
        ),
        _t(
            "Bear Call Spread",
            "Vertical spread",
            "Moderately bearish",
            "Sell a call and buy a higher-strike call: a credit spread that profits if the underlying stays below K.",
            "-1 call @ K, +1 call @ K+w",
            lambda k, w, T: (C(-1, k), C(1, k + w)),
        ),
        _t(
            "Bull Put Spread",
            "Vertical spread",
            "Moderately bullish",
            "Sell a put and buy a lower-strike put: a credit spread that profits if the underlying stays above K.",
            "-1 put @ K, +1 put @ K-w",
            lambda k, w, T: (P(-1, k), P(1, k - w)),
            lowest_offset=1,
        ),
        _t(
            "Bear Put Spread",
            "Vertical spread",
            "Moderately bearish",
            "Buy a put and sell a lower-strike put: a cheaper bearish bet with capped profit and loss.",
            "+1 put @ K, -1 put @ K-w",
            lambda k, w, T: (P(1, k), P(-1, k - w)),
            lowest_offset=1,
        ),
        # Volatility
        _t(
            "Long Straddle",
            "Volatility",
            "Large move either way",
            "Buy a call and a put at the same strike: profits from a large move in either direction.",
            "+1 call @ K, +1 put @ K",
            lambda k, w, T: (C(1, k), P(1, k)),
            False,
        ),
        _t(
            "Short Straddle",
            "Volatility",
            "Range-bound",
            "Sell a call and a put at the same strike: profits if the underlying stays near K; unlimited risk.",
            "-1 call @ K, -1 put @ K",
            lambda k, w, T: (C(-1, k), P(-1, k)),
            False,
        ),
        _t(
            "Long Strangle",
            "Volatility",
            "Large move either way",
            "Buy an out-of-the-money put and call: cheaper than a straddle but needs a bigger move.",
            "+1 put @ K-w, +1 call @ K+w",
            lambda k, w, T: (P(1, k - w), C(1, k + w)),
            lowest_offset=1,
        ),
        _t(
            "Short Strangle",
            "Volatility",
            "Range-bound",
            "Sell an out-of-the-money put and call: profits inside the strikes; unlimited risk outside.",
            "-1 put @ K-w, -1 call @ K+w",
            lambda k, w, T: (P(-1, k - w), C(-1, k + w)),
            lowest_offset=1,
        ),
        _t(
            "Call Ratio Backspread",
            "Volatility",
            "Sharp rally",
            "Sell one call and buy two higher-strike calls: small risk if flat, unlimited profit on a big rally.",
            "-1 call @ K, +2 call @ K+w",
            lambda k, w, T: (C(-1, k), C(2, k + w)),
        ),
        _t(
            "Put Ratio Backspread",
            "Volatility",
            "Sharp sell-off",
            "Sell one put and buy two lower-strike puts: profits from a crash, limited loss if the market drifts.",
            "-1 put @ K, +2 put @ K-w",
            lambda k, w, T: (P(-1, k), P(2, k - w)),
            lowest_offset=1,
        ),
        # Range-bound
        _t(
            "Iron Butterfly",
            "Range-bound",
            "Pinned near K",
            "Sell an at-the-money straddle and buy wings w away: a defined-risk short straddle.",
            "+1 put @ K-w, -1 put @ K, -1 call @ K, +1 call @ K+w",
            lambda k, w, T: (P(1, k - w), P(-1, k), C(-1, k), C(1, k + w)),
            lowest_offset=1,
        ),
        _t(
            "Iron Condor",
            "Range-bound",
            "Range-bound",
            "Sell a put spread and a call spread around K: profits while the underlying stays between the short "
            "strikes.",
            "+1 put @ K-2w, -1 put @ K-w, -1 call @ K+w, +1 call @ K+2w",
            lambda k, w, T: (P(1, k - 2 * w), P(-1, k - w), C(-1, k + w), C(1, k + 2 * w)),
            lowest_offset=2,
        ),
        _t(
            "Long Call Butterfly",
            "Range-bound",
            "Pinned near K",
            "Buy a K-w call, sell two K calls and buy a K+w call: a cheap bet that the underlying expires near K.",
            "+1 call @ K-w, -2 call @ K, +1 call @ K+w",
            lambda k, w, T: (C(1, k - w), C(-2, k), C(1, k + w)),
            lowest_offset=1,
        ),
        _t(
            "Long Put Butterfly",
            "Range-bound",
            "Pinned near K",
            "Buy a K+w put, sell two K puts and buy a K-w put: the put version of the butterfly.",
            "+1 put @ K-w, -2 put @ K, +1 put @ K+w",
            lambda k, w, T: (P(1, k - w), P(-2, k), P(1, k + w)),
            lowest_offset=1,
        ),
        _t(
            "Jade Lizard",
            "Range-bound",
            "Neutral / mildly bullish",
            "Sell a put and a call spread: no upside risk when the total credit exceeds the call-spread width.",
            "-1 put @ K-w, -1 call @ K+w, +1 call @ K+2w",
            lambda k, w, T: (P(-1, k - w), C(-1, k + w), C(1, k + 2 * w)),
            lowest_offset=1,
        ),
        # Synthetic & directional combinations
        _t(
            "Synthetic Long Stock",
            "Synthetic",
            "Bullish",
            "Buy a call and sell a put at the same strike: replicates owning the stock (put-call parity).",
            "+1 call @ K, -1 put @ K",
            lambda k, w, T: (C(1, k), P(-1, k)),
            False,
        ),
        _t(
            "Risk Reversal",
            "Synthetic",
            "Bullish",
            "Sell an out-of-the-money put to buy an out-of-the-money call: cheap upside with downside exposure.",
            "-1 put @ K-w, +1 call @ K+w",
            lambda k, w, T: (P(-1, k - w), C(1, k + w)),
            lowest_offset=1,
        ),
        # Time spreads (legs with different expiries)
        _t(
            "Long Call Calendar",
            "Time spread",
            "Neutral near K, rising vol",
            "Sell a call expiring at T and buy the same-strike call expiring at 2T: profits from faster decay of the "
            "front option if the underlying stays near K.",
            "-1 call @ K (T), +1 call @ K (2T)",
            lambda k, w, T: (C(-1, k, T), C(1, k, 2 * T)),
            False,
        ),
        _t(
            "Diagonal (Poor Man's Covered Call)",
            "Time spread",
            "Mildly bullish",
            "Buy a deep in-the-money long-dated call as a stock substitute and sell a short-dated out-of-the-money "
            "call.",
            "+1 call @ K-w (2T), -1 call @ K+w (T)",
            lambda k, w, T: (C(1, k - w, 2 * T), C(-1, k + w, T)),
            lowest_offset=1,
        ),
    ]
}

TEMPLATE_CATEGORIES = tuple(dict.fromkeys(t.category for t in TEMPLATES.values()))


def build_template(name: str, strike: float, width: float, expiry: float | None = None) -> Strategy:
    """Instantiate a template around ``strike`` with strike spacing ``width``.

    ``expiry`` (years) is required by time spreads, whose legs carry their own
    expiries (``expiry`` and ``2 * expiry``); other templates leave leg
    expiries to the market default.
    """
    template = TEMPLATES[name]
    if template.uses_width:
        if not width > 0:
            raise ValueError("strike width must be positive")
        lowest = strike - template.lowest_offset * width
        if lowest <= 0:
            raise ValueError(
                f"strike width {width:g} is too wide for strike {strike:g}: the lowest strike would be {lowest:g}"
            )
    if template.category == "Time spread":
        if expiry is None or not expiry > 0:
            raise ValueError(f"{name} needs the front-month expiry (years)")
    elif expiry is None:
        expiry = 1.0  # unused: these legs take the market expiry
    return Strategy(name, template.build(strike, width, expiry), template.description)
