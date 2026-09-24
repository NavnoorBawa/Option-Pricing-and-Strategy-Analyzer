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
* Stock legs are valued on a total-return basis (dividends at yield ``q``
  are reinvested), which keeps the stock and option legs consistent with
  put-call parity. Financing of the premium paid or received is ignored,
  as is conventional for payoff diagrams.
* The entry cost of every option is its model value at inception, so the
  P&L "today" at the current spot is exactly zero.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
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
    "leg_table",
    "pnl",
    "position_greeks",
    "position_value",
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
        qty = f"{abs(self.quantity):g}"
        if self.kind == "stock":
            return f"{side} {qty} share{'s' if abs(self.quantity) != 1 else ''}"
        return f"{side} {qty} × {self.strike:g} {self.kind.capitalize()}"


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


#: Reference tenor for implied-volatility shocks (one month, like the VIX).
VOL_SHOCK_REFERENCE_TENOR = 30.0 / 365.0


def _leg_value(leg, market, spot, t, vol_multiplier, vol_shift, rate_shift, vol_tenor_decay):
    q = market.dividend_yield
    if leg.kind == "stock":
        return spot * np.exp(q * t)
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
    ``t`` (a debit is borrowed, a credit earns interest), so that under the
    risk-neutral measure the expected P&L is exactly zero.
    """
    cost = entry_cost(strategy, market)
    if carry:
        cost = cost * np.exp(market.rate * np.asarray(t, dtype=float))
    return position_value(strategy, market, spot, t, **shocks) - cost


def position_greeks(strategy: Strategy, market: Market, spot=None, t: float = 0.0) -> dict:
    """First-order Greeks of the whole position (per unit, raw model units)."""
    spot = market.spot if spot is None else spot
    totals = dict.fromkeys(("delta", "gamma", "theta", "vega", "rho"), 0.0)
    for leg in strategy.legs:
        if leg.kind == "stock":
            totals["delta"] = totals["delta"] + leg.quantity * np.exp(market.dividend_yield * t)
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
            g = {"delta": 1.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
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


def _analysis_grid(strategy: Strategy, market: Market, horizon: float) -> np.ndarray:
    """Spot grid covering every kink and ~8 standard deviations either side."""
    width = 8.0 * market.volatility * np.sqrt(max(horizon, 1e-4))
    hi = max([market.spot, *strategy.strikes]) * max(np.exp(width), 3.0)
    grid = np.concatenate(
        [
            np.linspace(0.0, hi, 4001),
            np.geomspace(1e-6 * market.spot, hi, 2001),
            np.asarray(strategy.strikes, dtype=float),
            [market.spot],
        ]
    )
    return np.unique(grid)


def _upside_slope(strategy: Strategy, market: Market, horizon: float) -> float:
    """d(P&L)/dS as S -> infinity at ``horizon`` (per unit)."""
    slope = 0.0
    q = market.dividend_yield
    for leg in strategy.legs:
        if leg.kind == "stock":
            slope += leg.quantity * np.exp(q * horizon)
        elif leg.kind == "call":
            tau = _leg_expiry(leg, market) - horizon
            slope += leg.quantity * (np.exp(-q * tau) if tau > 0 else 1.0)
    return slope


def breakevens(strategy: Strategy, market: Market, horizon: float | None = None, carry: bool = False) -> list[float]:
    """Spot prices at which the P&L at ``horizon`` (default: front expiry) is zero."""
    horizon = front_expiry(strategy, market) if horizon is None else horizon
    grid = _analysis_grid(strategy, market, horizon)
    values = pnl(strategy, market, grid, horizon, carry)
    tol = 1e-9 * max(market.spot, 1.0)
    sign = np.sign(np.where(np.abs(values) <= tol, 0.0, values))

    def f(s):
        return float(pnl(strategy, market, s, horizon, carry))

    # Walk the non-zero grid points; a sign flip between neighbours brackets a
    # root, a flip across a run of (numerically) zero points is a root there.
    roots: list[float] = []
    nonzero = np.flatnonzero(sign)
    for i, j in pairwise(nonzero):
        if sign[i] == sign[j]:
            continue
        if j == i + 1:
            roots.append(float(brentq(f, grid[i], grid[j], xtol=1e-10 * max(market.spot, 1.0))))
        else:
            roots.append(float(0.5 * (grid[i + 1] + grid[j - 1])))
    # Merge numerically identical roots.
    merged: list[float] = []
    for root in sorted(roots):
        if not merged or root - merged[-1] > 1e-7 * max(market.spot, 1.0):
            merged.append(root)
    return merged


@dataclass(frozen=True)
class StrategyMetrics:
    """Payoff analytics per unit at the front expiry (multiply by the multiplier for $)."""

    entry_cost: float
    max_profit: float  # np.inf when unlimited
    max_loss: float  # positive number; np.inf when unlimited
    max_profit_spot: float | None
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
        if not np.isfinite(self.max_loss) or self.max_loss <= 0:
            return None
        return self.max_profit / self.max_loss


def _terminal_quadrature(market: Market, horizon: float, drift: float, kinks: Sequence[float]):
    """Nodes/weights for E[f(S_h)] with log-normal S_h, split at kinks (Gauss-Legendre)."""
    vol_t = market.volatility * np.sqrt(horizon)
    mean = np.log(market.spot) + (drift - market.dividend_yield - 0.5 * market.volatility**2) * horizon
    z_kinks = sorted(
        {float((np.log(k) - mean) / vol_t) for k in kinks if k > 0 and abs((np.log(k) - mean) / vol_t) < 12}
    )
    edges = [-12.0, *z_kinks, 12.0]
    x, w = np.polynomial.legendre.leggauss(64)
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
    grid = _analysis_grid(strategy, market, horizon)
    values = pnl(strategy, market, grid, horizon, carry)
    slope = _upside_slope(strategy, market, horizon)
    slope_tol = 1e-9

    i_max, i_min = int(np.argmax(values)), int(np.argmin(values))
    max_profit = np.inf if slope > slope_tol else float(values[i_max])
    max_loss = np.inf if slope < -slope_tol else float(-values[i_min])
    be = breakevens(strategy, market, horizon, carry)

    # Probability of profit: sum the log-normal mass of the intervals where P&L > 0.
    vol_t = market.volatility * np.sqrt(horizon)
    mean = np.log(market.spot) + (drift - market.dividend_yield - 0.5 * market.volatility**2) * horizon
    edges = [0.0, *be, np.inf]
    pop = 0.0
    for a, b in pairwise(edges):
        if np.isinf(b):
            probe = (a if a > 0 else market.spot) * 2.0 + 1.0
        elif a == 0:
            probe = 0.5 * b
        else:
            probe = 0.5 * (a + b)
        if float(pnl(strategy, market, probe, horizon, carry)) > 0:
            cdf_b = 1.0 if np.isinf(b) else float(ndtr((np.log(b) - mean) / vol_t))
            cdf_a = 0.0 if a == 0 else float(ndtr((np.log(a) - mean) / vol_t))
            pop += cdf_b - cdf_a

    spots, weights, _, _ = _terminal_quadrature(market, horizon, drift, [*strategy.strikes, *be])
    outcomes = pnl(strategy, market, spots, horizon, carry)
    expected = float(np.dot(weights, outcomes))
    second = float(np.dot(weights, outcomes**2))
    std = float(np.sqrt(max(second - expected**2, 0.0)))

    return StrategyMetrics(
        entry_cost=entry_cost(strategy, market),
        max_profit=max_profit,
        max_loss=max_loss,
        max_profit_spot=None if np.isinf(max_profit) else float(grid[i_max]),
        max_loss_spot=None if np.isinf(max_loss) else float(grid[i_min]),
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

    ``expiry`` (years) is only needed by time spreads, whose legs carry their
    own expiries; other templates leave leg expiries to the market default.
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
    if expiry is None:
        expiry = 1.0
    return Strategy(name, template.build(strike, width, expiry), template.description)
