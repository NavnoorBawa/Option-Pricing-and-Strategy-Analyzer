"""The Pricing tab in the desk layout (UI spec 3.1; roadmap PR-7, UI-1 and UI-7).

It prices the sidebar's contract, call and put, with the sidebar's engine, and shows, top to bottom:

1. a context line: the contract, its last trading day and its settlement (and the what-if note while a shift is set);
2. six headline tiles (UI-1: the shell draws no price cards any more): CALL, PUT, STRADDLE, VOL, FORWARD (with the
   forward implied by this strike's quoted pair) and TIME;
3. the contract table: the chain's bid, ask and mid with their implied vols, the theo, Mid vs fit in vol points with
   an ``in band`` flag (the edge convention of UI spec 2.5), and the Greeks in desk units;
4. the profile: a value or Greek against spot, vol or time, its controls and chart in one fragment;
5. the chain ladder (UI-7): calls and puts mirrored about the strike, over a delta or risk-neutral window, in its own
   fragment; a click on a row prices that strike (:func:`on_chain_select`);
6. the Greeks table to third order, with P(ITM) from the smile (the published surface's risk-neutral CDF);
7. Engine diagnostics, computed only while open: the trees and Monte Carlo against the reference engine, with each
   simulated price's interval and its reliability flag, the trees' convergence and their own Greeks and one-day
   decay, and the American engine's exercise boundary;
8. a footer line naming the engine and the conventions.

Under the textbook market there are no quotes: the market slots of the tiles read ``–`` with the reason in their
help, the tables leave out their market columns, and P(ITM) is N(d₂) at the flat vol (``flat vol``).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import ndtr, ndtri

from .. import american, binomial, clock
from .. import black_scholes as bs
from ..american import AmericanPrice
from ..monte_carlo import RELIABLE_PATHS, RELIABLE_SAMPLES, MonteCarloResult
from ..strategies import ONE_DAY
from ..vol_surface.published import Expiry, PublishedSurface, Quote
from . import cache, charts, charts_pricing, surface_state, tab_methodology, whatif
from . import market as mk
from .components import card_row, cents_places, current_key, table
from .format import (
    MINUS,
    MISSING,
    NB_HYPHEN,
    ROW_HEIGHT,
    bp,
    decay_label,
    decay_leads,
    level,
    md,
    money,
    number,
    pct,
    premium,
    probability_text,
    signed,
    signed_sci,
    strike,
    time_left,
    vol,
    vol_change,
)
from .format import date as short_date
from .format import moneyness as fmt_moneyness
from .inputs import Inputs
from .model_config import PATH_CHOICES, ModelConfig

#: Session-state key of the Engine diagnostics expander (open or closed), bound to the URL. A version-2 link to a
#: tree or to Monte Carlo opens it (``app._read_link``).
DIAGNOSTICS = "pricing_diag"
#: Session-state key: the chain ladder moved the strike, so its fragment reruns the whole app (the sidebar, tiles and
#: tables read the new strike only in a full run).
_CHAIN_MOVED = "pricing_chain_moved"
#: Session-state key: why a click on the chain ladder did not move the strike (a row beyond the surface's reach).
_CHAIN_NOTE = "pricing_chain_note"
#: Session-state key: the clicks the chain ladder has handled, which its widget keys hold. A handled click starts a
#: new grid, whose selection is empty (Streamlit cannot clear a table's selection otherwise), and nothing else does:
#: with the strike in the key, every strike step remounted both grids.
_CHAIN_CLICKS = "pricing_chain_clicks"
#: The headline tiles' row and its three pairs: 6 across on a desktop, 4 + 2 on a laptop with the sidebar open and
#: 2 + 2 + 2 on a phone. (Groups of three, UI spec 2.9's 3 + 3, cannot hold three cards on a 390 px phone: their
#: values and deltas, "5,813.91" over "pair −0.29 bp", need about 130 px a card, so the groups went 2 + 1 twice, four
#: lines of tiles above the contract table.)
TILES = ("pricing_tiles", ("pricing_tiles_price", "pricing_tiles_vol", "pricing_tiles_market"))
#: The profile's controls (UI spec 3.1 item 4).
METRICS = ("Value", "Δ", "Γ", "Vega", "Θ")
AXES = ("Spot", "Vol", "Time")
RANGES = ("±5%", "±10%", "±20%", "RN 95%", "Auto")
#: Points of a profile curve: against spot 121 (the spot axis must resolve gamma's peak a few days out), against vol
#: or time 41 (under the American engine each point solves its own boundary).
SPOT_POINTS, OTHER_POINTS = 121, 41
#: The chain ladder's window and stride (UI spec 3.1 item 5), and the rows that "Auto" keeps it to before the
#: Pricing strike's row is added: 14, so that the default ladder fits the 15 rows a table shows before it scrolls
#: (UI spec 2.8) with the strike's row in view. (The spec's 30 gave 28 rows at 25 points, 5,425 to 6,075: the
#: strike's row was the 17th, below the fold of a 15-row frame, which Streamlit cannot scroll to. A 50-point stride
#: gives 14 rows over the same window; a finer stride or "All" scrolls inside the table.)
WINDOWS = ("10Δ–10Δ", "25Δ–25Δ", "RN 95%", "All")
STRIDES = ("Auto", 5, 10, 25, 50, 100)
CHAIN_ROWS = 14
#: Orders of the Greeks table.
ORDERS = ("1st", "+2nd", "+3rd")
#: The fixed trees of Engine diagnostics: (lattice, steps).
DIAGNOSTIC_TREES = (("lr", 501), ("lr", 2001), ("crr", 2001))
#: What the CALL and PUT tiles compare the American engine with: an American Leisen-Reimer tree of this many steps.
TREE_CHECK_STEPS = 2001
#: The confidence interval of a simulated price, in no-break spaces: it never splits over two lines.
_CI_FORMULA = "95% interval = price ± 1.96 SE".replace(" ", "\xa0")
#: What a heavy-tailed Monte Carlo sample does instead of the usual estimator (see ``monte_carlo``).
HEAVY_TAIL_NOTE = (
    "Heavy-tailed sample (large σ√T): the put is averaged directly and the call follows from put-call parity"
)
#: Why the American engine values an option at its European value. ``american.exercise_regime`` finds early exercise
#: never optimal for a call with ``q <= 0`` and ``q <= r`` and for a put with ``r <= 0`` and ``r <= q``; the sidebar's
#: dividend yield is never negative, so that is a call without dividends at a rate of at least zero, and a put at a
#: rate of at most zero.
NEVER_EARLY = {
    "call": (
        "with no dividend yield, exercising a call early never pays (it would pay the strike early and gain no "
        "dividends)"
    ),
    "put": "at a rate of zero or below, exercising a put early never pays (the strike would earn no interest)",
}
#: Why the American engine hands an option to the tree (``AmericanPrice.fallback_reason``).
FALLBACK_REASONS = {
    "double boundary": "its exercise region lies between two boundaries, which the engine does not solve",
    "not converged": "the engine's iteration for the exercise boundary did not converge at these inputs",
}
#: How the American engine valued an option (``AmericanPrice.method``), for the engine table.
ALO_METHODS = {
    "alo": "Spectral collocation",
    "european": "European value (never exercised early)",
    "tree": "Leisen-Reimer trees (fallback)",
    "exact": "Closed form",
    "intrinsic": "Intrinsic value",
}
#: The engine's Greeks (names of ``black_scholes.GREEKS``): the American engine gives first order only.
ALO_GREEKS = ("delta", "gamma", "theta", "vega", "rho", "epsilon")
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


# --- The engine's prices ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Prices:
    """The sidebar engine's values of the call and the put ($ per share), with a note on each (the tiles' help).
    Under the American engine also its results (``american.price_details``) and Greeks (``american.greeks``)."""

    call: float
    put: float
    call_note: str = ""
    put_note: str = ""
    alo: dict | None = None
    alo_greeks: dict | None = None


def compute_prices(inputs: Inputs) -> Prices:
    """The sidebar engine's values of the call and the put: the closed form, or the American engine.

    Raises ``ValueError`` where the American engine cannot value the contract (``app.run`` then falls back to the
    closed form and says so)."""
    S, K, T, r, sigma, q = _args(inputs)
    if inputs.model.kind == "alo":
        details, greeks = cache.alo_results(S, K, T, r, sigma, q, inputs.model.scheme)
        notes = {kind: _alo_note(kind, res) for kind, res in details.items()}
        return Prices(
            details["call"].value, details["put"].value, notes["call"], notes["put"], alo=details, alo_greeks=greeks
        )
    note = "European exercise" + (", Black-76 on the expiry's parity forward" if inputs.context is not None else "")
    return Prices(bs.price(S, K, T, r, sigma, q, "call"), bs.price(S, K, T, r, sigma, q, "put"), note, note)


def _args(inputs: Inputs) -> tuple[float, float, float, float, float, float]:
    """``(S, K, T, r, σ, q)`` of the sidebar's contract."""
    return inputs.spot, inputs.strike, inputs.expiry, inputs.rate, inputs.volatility, inputs.dividend_yield


def _alo(inputs: Inputs) -> bool:
    return inputs.model.kind == "alo"


def engine_name(inputs: Inputs) -> str:
    """``ALO`` or ``BSM``: the short name the tables' headers give the sidebar engine."""
    return "ALO" if _alo(inputs) else "BSM"


def mc_interval(res: MonteCarloResult, dollars: bool = True, short: bool = False) -> str:
    """The 95% interval of a simulated price, ``$10.4237 – $10.4582, 95%`` (``10.4237 – 10.4582`` without
    ``dollars``, for a table of plain amounts), or, when the result is not reliable, why it is not shown:
    ``Interval unreliable: 12 of 100,000 paths finish in the money`` in the diagnostics table, and ``short``,
    ``Interval unreliable: 12 ITM paths``, where room is tight."""
    if res.reliable:
        low, high = (money(bound, 4).replace("-", MINUS) for bound in res.ci95)
        return f"{low} – {high}, 95%" if dollars else f"{low} – {high}".replace("$", "")
    card, sentence, _ = _withheld(res)
    return card if short else sentence


def _withheld(res: MonteCarloResult) -> tuple[str, str, str]:
    """Why an unreliable result's interval is not shown (``MonteCarloResult.reliable``): the short line, the full
    sentence and the reason, with the coverage measured below the threshold (``monte_carlo.RELIABLE_PATHS`` and
    ``RELIABLE_SAMPLES``)."""
    if res.n_samples < RELIABLE_SAMPLES:  # never in the app: its fewest paths, 10,000, are 5,000 antithetic pairs
        needed = RELIABLE_SAMPLES * (2 if res.antithetic else 1)
        return (
            "Interval unreliable: too few paths",
            f"Interval unreliable: {number(res.n_paths, 0)} paths, fewer than {number(needed, 0)}",
            f"with fewer than {number(RELIABLE_SAMPLES, 0)} independent samples ({number(needed, 0)} paths"
            + (" in antithetic pairs" if res.antithetic else "")
            + ") the standard error is itself too uncertain, and the interval covers the value in less than 95% of "
            "runs (93 to 94% with 500 antithetic pairs and the control variate)",
        )
    count, side = _scarce_side(res)
    return (
        f"Interval unreliable: {number(count, 0)} {'ITM' if side == 'in' else 'OTM'} paths",
        f"Interval unreliable: {number(count, 0)} of {number(res.n_paths, 0)} paths finish {side} the money",
        f"with fewer than {RELIABLE_PATHS} paths finishing {side} the money it covers the value in less than 95% of "
        "runs (88% with 10 such paths, 92% with 30, 94% with 100)",
    )


def _scarce_side(res: MonteCarloResult) -> tuple[int, str]:
    """The simulated prices on the scarcer side of the strike, and which side: ``(12, "in")`` or ``(12, "out of")``.

    It is the side an unreliable result's flag counts (``MonteCarloResult.reliable``) whenever there are at least
    ``RELIABLE_SAMPLES`` independent samples, so at least 5,000 paths: a payoff averaged as it is fails on its
    in-the-money paths only when fewer than 300 of them, so fewer than half, finish in the money."""
    out = res.n_paths - res.n_itm
    return (res.n_itm, "in") if res.n_itm <= out else (out, "out of")


def _mc_note(res: MonteCarloResult) -> str:
    """A simulated price's standard error, and why its interval is not shown if it is not."""
    if not np.isfinite(res.std_error):
        return "Every simulated price finished on one side of the strike: the error cannot be estimated"
    if res.reliable:
        return f"Standard error {money(res.std_error, 4)}; {_CI_FORMULA}"
    _, sentence, reason = _withheld(res)
    return (
        f"{sentence}. Standard error {money(res.std_error, 4)}. The 95% interval, price ± 1.96 SE, is not shown: "
        f"{reason}."
    )


def _alo_note(kind: str, res: AmericanPrice) -> str:
    """The help of a price tile under the American engine: its early-exercise premium, or why it has none, or why
    the engine fell back to the tree."""
    if res.fallback:
        return f"Valued on Leisen-Reimer trees, not by the engine: {FALLBACK_REASONS[res.fallback_reason]}"
    if res.method == "european":
        return f"Equals the Black-Scholes-Merton value: {NEVER_EARLY[kind]}"
    return (
        f"Early-exercise premium {money(res.premium, 4, signed=True)} over the European value {money(res.european, 4)}"
    )


def alo_fallbacks(prices: Prices) -> list[str]:
    """Plain statements of the options the American engine handed to the tree, and why (empty when none)."""
    notes = []
    for kind, res in (prices.alo or {}).items():
        if res.fallback:
            detail = ""
            if res.fallback_reason == "not converged":
                detail = (
                    " (its last step still moved the logarithm of the boundary by "
                    f"{signed_sci(res.last_step).lstrip('+')}, where at most "
                    "0.01 passes)"
                )
            notes.append(
                f"The {kind} is valued on Leisen-Reimer trees (2,001 and 4,003 steps, extrapolated) instead of the "
                f"American engine: {FALLBACK_REASONS[res.fallback_reason]}{detail}."
            )
    return notes


# --- One-day decay --------------------------------------------------------------------------------------------
# Theta / 365 is the instantaneous rate of decay. Inside a few days of expiry it is far from what a day costs: a
# 1-day at-the-money SPX put (5,800, 18 % volatility, rate 4.3 %, dividend yield 1.3 %) worth $21.56 shows theta
# -$10.66 a day, and loses all $21.56 by expiry. The decay below reprices the option instead.


def to_expiry(expiry: float) -> bool:
    """Whether an option with ``expiry`` years left expires within the next calendar day, so that its one-day decay
    is its decay to expiry."""
    return expiry <= ONE_DAY


def closed_form_decay(S, K, T, r, sigma, q, option_type: str):
    """Black-Scholes-Merton one-day decay ($ per share): the value with one calendar day less to expiry, all else
    equal, minus the value now. With a day or less to expiry, the value at expiry (the intrinsic value
    ``max(±(S − K), 0)``) minus the value now. Broadcasts like :func:`black_scholes.price`."""
    later = bs.price(S, K, np.maximum(np.asarray(T, dtype=float) - ONE_DAY, 0.0), r, sigma, q, option_type)
    return later - bs.price(S, K, T, r, sigma, q, option_type)


def later_trees(steps: int, expiry: float, method: str) -> tuple[tuple[int, float], ...]:
    """The trees that value an option one calendar day later, as ``(steps, weight)`` pairs whose weights add up to
    one (``steps`` and ``expiry`` are the current tree's, ``expiry`` more than a day).

    Measured on European options against the closed-form decay (spot 100, strikes 80 to 120 in steps of 0.5, rate
    5 %, no dividends, 2 to 60 days and 13 expiries from 75 to 3,650 days, 501 steps), on every one-day decay of at
    least 1 cent per share, at 10 %, 20 % and 50 % volatility:

    * Leisen-Reimer keeps its steps: its error is smooth in the expiry, so the two trees' errors cancel, to within
      1.6e-6, 4.0e-6 and 1.2e-5 of the decay (medians 3e-7 or less).
    * The Cox-Ross-Rubinstein error swings with the strike's place among the final nodes. With the same steps the
      two errors did not cancel: up to 4.4 %, 8.5 % and 6.0 % of the decay. The shorter tree therefore keeps the
      time step, which keeps the final nodes where they were: its value is interpolated linearly in the step count
      between the two counts with the current tree's parity (the final nodes of an odd and an even tree interleave)
      that bracket ``steps · (expiry − 1 day) / expiry``, whose time steps bracket the current tree's. Within 0.55 %,
      1.5 % and 0.66 % of the decay (99 % of cases within 0.08 %, 0.12 % and 0.19 %), where the next count up with
      the parity alone was off by up to 2.1 %, 3.0 % and 2.9 %, and the nearest count by up to 1.1 %, 2.3 % and
      2.2 %.
    * Both scale with the steps. At 20 % volatility: with 101 steps, Cox-Ross-Rubinstein up to 11 % (99 % of cases
      within 1.2 %) and Leisen-Reimer 9.7e-5; with 2,000 steps (strikes in steps of 2), 0.062 % and 1.7e-7. With a
      3 % dividend yield and 501 steps, 0.64 % and 4.1e-6.

    The lower count's time step is slightly longer than the current tree's, so its lattice can fail to be
    arbitrage-free where the current one only just is (p lies in (0, 1) exactly while |r − q| Δt < σ√Δt); the
    caller then uses the upper count alone (:func:`tree_decay`), whose time step is never longer. The lower count is
    also left out when it is below one step or carries no weight (a whole number of steps keeps the time step
    exactly).
    """
    if method != "crr":
        return ((steps, 1.0),)
    exact = steps * (expiry - ONE_DAY) / expiry
    lower = math.floor(exact + 1e-9)  # a whole number is not rounded down
    lower -= (lower - steps) % 2
    weight = max(exact - lower, 0.0) / 2.0  # of the upper count, two steps above
    if lower < 1:
        return ((lower + 2, 1.0),)
    if weight < 1e-9:
        return ((lower, 1.0),)
    return ((lower, 1.0 - weight), (lower + 2, weight))


def greeks_decay(inputs: Inputs, prices: Prices) -> dict[str, float]:
    """One-day decay of the call and the put ($ per share) by the sidebar engine: the American engine's value one
    calendar day later minus its value now under that engine, the closed form's otherwise."""
    S, K, T, r, sigma, q = _args(inputs)
    if not _alo(inputs):
        return {kind: float(closed_form_decay(S, K, T, r, sigma, q, kind)) for kind in cache.KINDS}
    decay = {}
    for kind, res in prices.alo.items():
        if to_expiry(T):
            later = max(bs.option_sign(kind) * (S - K), 0.0)
        else:
            later = cache.alo_value(S, K, T - ONE_DAY, r, sigma, q, kind, inputs.model.scheme)
        decay[kind] = later - res.value
    return decay


def tree_decay(
    inputs: Inputs, model: ModelConfig, tree: dict
) -> tuple[dict[str, float], tuple[tuple[int, float], ...]]:
    """One-day decay of the call and the put ($ per share) on a diagnostics tree (``model``, whose results are
    ``tree``), and the shorter trees it used.

    The value on trees one calendar day shorter (:func:`later_trees`, weighted) minus the value on the tree now,
    with the tree's exercise style; with a day or less to expiry, the intrinsic value minus the value now (and no
    trees). Where the lower of two Cox-Ross-Rubinstein counts has no arbitrage-free lattice, the upper one alone.
    """
    S, K, T, r, sigma, q = _args(inputs)
    if to_expiry(T):
        later = {kind: max(bs.option_sign(kind) * (S - K), 0.0) for kind in cache.KINDS}
        return {kind: later[kind] - tree[kind].price for kind in cache.KINDS}, ()

    def value(steps: int) -> dict[str, float]:
        return cache.tree_prices(S, K, T - ONE_DAY, r, sigma, q, steps, model.american, model.tree_method)

    trees = later_trees(tree["call"].steps, T, model.tree_method)
    try:
        values = [(value(steps), weight) for steps, weight in trees]
    except ValueError:  # the lower count's longer time step has no arbitrage-free lattice
        trees = ((trees[-1][0], 1.0),)
        values = [(value(trees[0][0]), 1.0)]
    decay = {
        kind: math.fsum(weight * later[kind] for later, weight in values) - tree[kind].price for kind in cache.KINDS
    }
    return decay, trees


# --- The contract's market --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Contract:
    """The contract's market: under the index preset the published surface, the expiry's key in it, its forward,
    discount factor and strike interval, the chain's quotes of the call and the put (None when not listed) and the
    surface's vol at the strike (the fit, ``fit_vol``). Under the textbook market only the forward (S·e^{(r−q)T}),
    the discount factor e^{−rT} and a strike increment of a cent."""

    surface: PublishedSurface | None
    expiry: Expiry | None
    forward: float
    discount: float
    step: float
    quotes: dict[str, Quote | None]
    fit_vol: float | None

    @property
    def listed(self) -> bool:
        """Whether the market has quotes (the index preset)."""
        return self.surface is not None


def contract(inputs: Inputs) -> Contract:
    """The sidebar contract's market (see :class:`Contract`)."""
    published = surface_state.current(inputs)
    if published is None or inputs.expiry_date is None or inputs.settlement is None:
        T = inputs.expiry
        return Contract(None, None, inputs.forward, math.exp(-inputs.rate * T), 0.01, {"call": None, "put": None}, None)
    key = Expiry(inputs.expiry_date, inputs.settlement)
    quotes: dict[str, Quote | None] = {}
    for kind in cache.KINDS:
        try:
            quotes[kind] = published.quote(key, inputs.strike, kind)
        except KeyError:  # a typed strike that is not listed at this expiry
            quotes[kind] = None
    assert inputs.context is not None
    return Contract(
        published,
        key,
        published.forward(key),
        published.discount_factor(key),
        inputs.context.strike_step,
        quotes,
        inputs.surface_vol,
    )


def _day(day: date) -> str:
    """``Thu 19‑Nov`` (non-breaking hyphen)."""
    return f"{_WEEKDAYS[day.weekday()]} {short_date(day)[:6]}".replace("-", NB_HYPHEN)


def last_trade(expiry: date, settlement: str) -> str:
    """When the contract last trades, and how it settles: an AM monthly stops trading on the business day before
    its expiry and settles at the special opening quotation (SOQ); a PM weekly trades to the close on its expiry
    day, 16:00 ET or 13:00 on an early close, and settles at the close (Cboe SPX and SPXW specifications).

    >>> last_trade(date(2026, 11, 20), "AM"), last_trade(date(2026, 11, 25), "PM")
    ('last trade Thu 19‑Nov · settles at the SOQ', 'last trade Wed 25‑Nov 16:00 ET · settles at the close')
    >>> last_trade(date(2026, 11, 27), "PM")  # the day after Thanksgiving closes early
    'last trade Fri 27‑Nov 13:00 ET · settles at the close'
    """
    if settlement == "AM":
        return f"last trade {_day(clock.previous_trading_day(expiry - timedelta(days=1)))} · settles at the SOQ"
    day = clock.previous_trading_day(expiry)
    return f"last trade {_day(day)} {clock.market_close(day):%H:%M} ET · settles at the close"


def context_line(inputs: Inputs) -> str:
    """The Pricing and IV tabs' first line: the contract, and the what-if note while a shift is set (those tabs show
    the market: the shifts apply on Strategy and Risk only, UI spec 2.0).

    ``SPX 20‑Nov‑26 AM 5,815 · last trade Thu 19‑Nov · settles at the SOQ`` under the index;
    ``Textbook contract · K 100.00 · 365 days · flat vol, no quotes`` under the textbook market."""
    if inputs.expiry_date is not None and inputs.settlement is not None and inputs.context is not None:
        day = short_date(inputs.expiry_date).replace("-", NB_HYPHEN)
        text = (
            f"{mk.INDEX_ROOT} {day} {inputs.settlement} {strike(inputs.strike, inputs.context.strike_step)} · "
            f"{last_trade(inputs.expiry_date, inputs.settlement)}"
        )
    else:
        days = f"{number(inputs.days, 0 if float(inputs.days).is_integer() else 1)} days"
        text = f"Textbook contract · K {level(inputs.strike)} · {days} · flat vol, no quotes"
    note = whatif.note(whatif.current())
    return f"{text} · {note}" if note else text


# --- Greeks in desk units ---------------------------------------------------------------------------------------


def engine_greeks(inputs: Inputs, prices: Prices) -> dict[str, dict[str, float]]:
    """The sidebar engine's Greeks of the call and the put in raw model units (``black_scholes.greeks``): the
    closed form's to third order, or the American engine's first order (``american.greeks``)."""
    if _alo(inputs):
        return {kind: {name: float(prices.alo_greeks[kind][name]) for name in ALO_GREEKS} for kind in cache.KINDS}
    return {kind: {k: float(v) for k, v in bs.greeks(*_args(inputs), kind).items()} for kind in cache.KINDS}


def desk_units(greeks: dict[str, float], spot: float, rate: float, dividend_yield: float, T: float) -> dict:
    """First-order Greeks in desk units, per share (UI spec 2.5): Δ (spot), Δ fwd = ∂V/∂F = Δ·e^{−(r−q)T}, Γ 1% =
    ΓS/100 (the change in Δ for a 1% move), vega $ per vol point, Θ $ per calendar day, ρ and ε $ per 1%."""
    return {
        "delta": greeks["delta"],
        "delta_fwd": greeks["delta"] * math.exp(-(rate - dividend_yield) * T),
        "gamma": greeks["gamma"] * spot / 100.0,
        "vega": greeks["vega"] / 100.0,
        "theta": greeks["theta"] / 365.0,
        "rho": greeks["rho"] / 100.0,
        "epsilon": greeks["epsilon"] / 100.0,
    }


def _sig(x: float, digits: int = 3) -> str:
    """``x`` signed, with ``digits`` significant digits (U+2212; scientific notation below 1e−6)."""
    if not np.isfinite(x):
        return MISSING
    if x == 0:
        return "0"
    if abs(x) < 1e-6:
        return signed_sci(x, digits)
    return signed(x, max(0, digits - 1 - math.floor(math.log10(abs(x)))))


# --- The tiles ------------------------------------------------------------------------------------------------


def pair_forward(contract: Contract, K: float) -> float:
    """The forward implied by one strike's quoted pair, K + (C_mid − P_mid) / DF (put-call parity); NaN when either
    side has no two-sided quote."""
    call, put = contract.quotes["call"], contract.quotes["put"]
    if call is None or put is None or not (call.bid > 0 and put.bid > 0):
        return math.nan
    return K + (call.mid - put.mid) / contract.discount


def _tiles(inputs: Inputs, prices: Prices, market: Contract) -> None:
    S, K, T, r, sigma, q = _args(inputs)
    alo = _alo(inputs)
    off = {"delta_color": "off", "delta_arrow": "off"}
    checks = {"call": "", "put": ""}
    if alo:
        # The engine against an independent lattice: an American Leisen-Reimer tree of 2,001 steps (cached).
        try:
            tree = cache.tree_prices(S, K, T, r, sigma, q, TREE_CHECK_STEPS, True, "lr")
            lattice = f"ALO vs LR {number(TREE_CHECK_STEPS, 0)}"
            checks = {k: f" {lattice}: {signed_sci(prices.alo[k].value - tree[k])}." for k in tree}
        except ValueError:
            pass
    engine = inputs.model.label
    price_tiles = []
    for kind, value, note in (("call", prices.call, prices.call_note), ("put", prices.put, prices.put_note)):
        price_tiles.append(
            (
                kind.capitalize(),
                premium(value),
                {
                    "delta": f"{pct(value / S, 2)} of S",
                    **off,
                    "help": md(f"{engine}: {premium(value, 6)} per share. {note}.{checks[kind]}"),
                },
            )
        )
    straddle = prices.call + prices.put
    # The straddle-implied move is the premium over spot, the desk's usual reading. It is not the move to either
    # breakeven: those are K ± premium, which sit asymmetrically about spot unless K = S (at the default index,
    # −3.13% and +3.65% against the tile's ±3.39%), so the help gives both.
    low, high = K - straddle, K + straddle
    tiles = [
        *price_tiles,
        (
            "Straddle",
            premium(straddle),
            {
                "delta": f"±{pct(straddle / S, 2)} of S",
                **off,
                "help": md(
                    f"Call + put = {premium(straddle, 6)}; as a share of spot, the straddle-implied move to expiry. "
                    f"The long straddle breaks even at expiry at K ± premium: {level(low)} "
                    f"({signed((low / S - 1) * 100, 2)}% of S) and {level(high)} ({signed((high / S - 1) * 100, 2)}% "
                    "of S), before the premium's financing."
                ),
            },
        ),
    ]
    # Vol: the surface's or a typed override (index), or the textbook's one vol.
    if market.listed and inputs.surface_vol is not None:
        moneyness = f"K/F {fmt_moneyness(K, market.forward)}"
        if inputs.vol_overridden:
            source = f"override {vol_change(sigma - inputs.surface_vol)}"
            help_text = f"Typed over the surface's {vol(inputs.surface_vol)} at {moneyness}."
        else:
            source = "surface"
            help_text = f"The published surface's vol at {moneyness} (arbitrage-free SVI fitted to the chain)."
    else:
        source, help_text = "flat", "Typed in the sidebar: one vol for every strike (the textbook market has no smile)."
    # Labelled Vol, not σ: the tiles' labels are set in capitals (desk.css), and a capital sigma reads as a sum.
    tiles.append(("Vol", vol(sigma), {"delta": source, **off, "help": help_text}))
    # Forward: the parity forward of the expiry, and the forward this strike's own pair implies, as a market check.
    implied = pair_forward(market, K)
    if market.listed:
        gap = (implied - market.forward) / market.forward
        pair = f"pair {bp(gap)}"
        forward_help = (
            "The expiry's forward from put-call parity across its quotes "
            f"(DF {number(market.discount, 5)}). This strike's pair implies K + (C mid − P mid)/DF = {level(implied)}."
            if np.isfinite(implied)
            else "The expiry's forward from put-call parity across its quotes. "
            + (
                "This strike is not listed at this expiry, so no pair of its own implies a forward."
                if market.quotes["call"] is None
                else "This strike's call or put has no bid, so its pair implies no forward."
            )
        )
    else:
        pair = "from r and q"  # no quotes, so no pair: the forward is S·e^((r−q)T) (an en dash read as missing)
        forward_help = (
            f"F = S·e^((r−q)T), discount factor e^(−rT) = {number(market.discount, 6)}. No pair-implied forward: the "
            "textbook market has no quotes."
        )
    tiles.append(("Forward", level(market.forward), {"delta": pair, **off, "help": md(forward_help)}))
    # Time: to the settlement (index) or days / 365 (textbook).
    if inputs.expiry_date is not None and inputs.context is not None:
        expiry_day = inputs.context.as_of.date() == inputs.expiry_date
        settles = clock.settlement_time(inputs.expiry_date, inputs.settlement or "PM")
        tiles.append(
            (
                "Time",
                time_left(T * 525_600, expiry_day=expiry_day),
                {
                    "delta": f"{inputs.settlement} settle",
                    **off,
                    "help": f"T = {number(T, 6)} years to the settlement at {_day(settles.date())} {settles:%H:%M} ET "
                    "(minutes / 525,600).",
                },
            )
        )
    else:
        tiles.append(
            (
                "Time",
                time_left(inputs.days * 1440),
                {"delta": "calendar days", **off, "help": f"T = days / 365 = {number(T, 6)} years."},
            )
        )
    card_row(*TILES, tiles)
    for note in alo_fallbacks(prices):
        st.warning(note, icon=":material/warning:")


# --- The contract table -----------------------------------------------------------------------------------------


def contract_frame(inputs: Inputs, prices: Prices, market: Contract) -> tuple[pd.DataFrame, dict, dict, dict]:
    """The contract table's data, kinds, help and headers (UI spec 3.1 item 3): one row per option.

    Market columns (under the index preset): bid, ask, mid, theo, the implied vols of the bid and the ask, and Mid vs
    fit (σ_mid − σ_fit, vp) with ``in band`` when the fit lies inside [σ_bid, σ_ask]; with an override, Buy edge
    (σ_theo − σ_ask) and Sell edge (σ_bid − σ_theo) instead, ``in band`` when both are at most zero. Then the
    engine's Greeks in desk units and, under the American engine, the early-exercise premium."""
    S, _, T, r, sigma, q = _args(inputs)
    greeks = engine_greeks(inputs, prices)
    engine = engine_name(inputs)
    rows = []
    for kind, theo in (("call", prices.call), ("put", prices.put)):
        row: dict[str, object] = {"Type": kind.capitalize()}
        if market.listed:
            quote = market.quotes[kind]
            bid, ask, mid, iv_bid, iv_ask, iv_mid = (np.nan,) * 6 if quote is None else quote
            if quote is not None and not bid > 0:  # an offer only: no bid, so no mid either
                bid, mid = np.nan, np.nan
            row |= {"Bid": bid, "Ask": ask, "Mid": mid, "Theo": theo, "IV bid": iv_bid * 100, "IV ask": iv_ask * 100}
            if inputs.vol_overridden:
                buy, sell = (sigma - iv_ask) * 100, (iv_bid - sigma) * 100
                row |= {"Buy edge": buy, "Sell edge": sell, "Band": _band(buy, sell)}
            else:
                fit = market.fit_vol if market.fit_vol is not None else sigma
                row |= {"Mid vs fit": (iv_mid - fit) * 100, "Band": _band((fit - iv_ask) * 100, (iv_bid - fit) * 100)}
        else:
            row["Theo"] = theo
        units = desk_units(greeks[kind], S, r, q, T)
        row |= {"Δ": units["delta"], "Γ 1%": units["gamma"], "Vega": units["vega"], "Θ": units["theta"]}
        row["ρ"] = units["rho"]
        if prices.alo is not None:
            row["Early-ex"] = prices.alo[kind].premium
        rows.append(row)
    frame = pd.DataFrame(rows)
    unlisted = "Not a listed contract in the source." if market.listed and market.quotes["call"] is None else ""
    kinds = {
        "Type": "text", "Bid": "premium", "Ask": "premium", "Mid": "premium", "Theo": "premium", "IV bid": "vol",
        "IV ask": "vol", "Mid vs fit": "vol_change", "Buy edge": "vol_change", "Sell edge": "vol_change",
        "Band": "text", "Δ": "delta", "Γ 1%": "gamma", "Vega": "vega_share", "Θ": "theta_share", "ρ": "rho_share",
        "Early-ex": "premium",
    }  # fmt: skip
    market_help = f"The synthetic chain's quote, $ per share. {unlisted}".strip()
    help_text = {
        "Type": "The option.",
        "Bid": market_help,
        "Ask": market_help,
        "Mid": f"Half way between bid and ask; none for a one-sided quote. {unlisted}".strip(),
        "Theo": f"The {engine} value at the vol used ({vol(sigma)}), $ per share.",
        "IV bid": f"Black-76 implied vol of the bid at the expiry's parity forward. {unlisted}".strip(),
        "IV ask": f"Black-76 implied vol of the ask at the expiry's parity forward. {unlisted}".strip(),
        "Mid vs fit": "σ mid − σ fit, vol points: positive means the market is rich against the fit. A residual, "
        "not an edge: the fit is made from these mids.",
        "Buy edge": "σ theo − σ ask, vol points: positive means buying at the ask is cheap against your vol.",
        "Sell edge": "σ bid − σ theo, vol points: positive means selling at the bid is rich against your vol.",
        "Band": "in band: the theo vol lies inside [σ bid, σ ask], so neither side is an edge. outside: it does not.",
        "Δ": f"{engine} delta, spot convention: per 1 point of S.",
        "Γ 1%": "Change in delta for a 1% move in spot: Γ·S/100.",
        "Vega": "$ per share per vol point.",
        "Θ": "$ per share per calendar day of elapsed time, ∂V/∂t ÷ 365 at fixed vol; the one-day decay by "
        "repricing is in the Greeks table.",
        "ρ": "$ per share per 1% rise in the rate, spot and dividend yield held.",
        "Early-ex": "American value minus the closed-form European value, $ per share.",
    }
    labels = {"Θ": "Θ", "ρ": "ρ", "Early-ex": "Early-ex ($)", "Mid vs fit": "Mid vs fit"}
    return frame, {c: kinds[c] for c in frame}, {c: help_text[c] for c in frame}, labels


def _band(buy: float, sell: float) -> str:
    """``in band`` when both edges are at most zero, ``outside`` when either is positive, ``–`` without a market."""
    if not (np.isfinite(buy) and np.isfinite(sell)):
        return MISSING
    return "in band" if buy <= 0 and sell <= 0 else "outside"


# --- The profile ------------------------------------------------------------------------------------------------


def _rn_range(inputs: Inputs, market: Contract, level_: float = 0.95) -> tuple[float, float]:
    """The central risk-neutral range of S_T: from the published surface's smile (index), or lognormal at the flat
    vol (textbook)."""
    if market.surface is not None and market.expiry is not None:
        return market.surface.rn_range(market.expiry, level_)
    _, _, T, _, sigma, _ = _args(inputs)
    z = ndtri(0.5 + level_ / 2.0) * sigma * math.sqrt(T)
    centre = market.forward * math.exp(-0.5 * sigma * sigma * T)
    return centre * math.exp(-z), centre * math.exp(z)


def spot_grid(inputs: Inputs, market: Contract, choice: str) -> np.ndarray:
    """The spot axis of the profile for a Range choice: ±5, ±10 or ±20% of S, the 95% risk-neutral range (from the
    smile; lognormal under the textbook market), or Auto (half the lower of S and K to one and a half times the
    higher, version 2's axis)."""
    S, K = inputs.spot, inputs.strike
    if choice == "RN 95%":
        lo, hi = _rn_range(inputs, market)
    elif choice == "Auto":
        lo, hi = 0.5 * min(S, K), 1.5 * max(S, K)
    else:
        half = float(choice.strip("±%")) / 100.0
        lo, hi = S * (1 - half), S * (1 + half)
    return np.linspace(lo, hi, SPOT_POINTS)


def _bsm_metric(metric: str, S, K, T, r, sigma, q, kind: str) -> np.ndarray:
    """A profile metric of the closed form, per share, in desk units (Γ as Γ 1%)."""
    if metric == "Value":
        return np.asarray(bs.price(S, K, T, r, sigma, q, kind), dtype=float)
    g = bs.greeks(S, K, T, r, sigma, q, kind)
    if metric == "Δ":
        return np.asarray(g["delta"], dtype=float)
    if metric == "Γ":
        return np.asarray(g["gamma"], dtype=float) * np.asarray(S, dtype=float) / 100.0
    if metric == "Vega":
        return np.asarray(g["vega"], dtype=float) / 100.0
    return np.asarray(g["theta"], dtype=float) / 365.0


def _alo_metric(metric: str, S, K, T, r, sigma, q, kind: str, scheme: str) -> np.ndarray:
    """A profile metric of the American engine, per share, in desk units: the value, and the Greeks by central
    differences of the value with the bumps of ``american.GREEK_BUMPS`` (spot and expiry by ±0.01%, the vol by
    ±0.01 vol point), the scheme held fixed, as ``american.greeks`` takes them. Arrays broadcast; the engine solves one
    boundary per distinct (K, T, r, σ, q)."""
    S, T, sigma = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (S, T, sigma)))

    def value(s=S, t=T, v=sigma) -> np.ndarray:
        return cache.alo_values(s, K, t, r, v, q, kind, scheme)

    if metric == "Value":
        return value()
    if metric in ("Δ", "Γ"):
        h = american.GREEK_BUMPS["spot"] * S
        up, down = value(s=S + h), value(s=S - h)
        if metric == "Δ":
            return (up - down) / (2 * h)
        return (up - 2 * value() + down) / (h * h) * S / 100.0
    if metric == "Vega":
        h = american.GREEK_BUMPS["volatility"]
        return (value(v=sigma + h) - value(v=np.maximum(sigma - h, 1e-8))) / (2 * h) / 100.0
    h = american.GREEK_BUMPS["expiry"] * T
    return -(value(t=T + h) - value(t=T - h)) / (2 * h) / 365.0


def _alo_decay(S, K, T, r, sigma, q, kind: str, scheme: str) -> np.ndarray:
    """The American engine's one-day decay over arrays: the value one calendar day later minus the value now (the
    intrinsic value at expiry within a day)."""
    S, T, sigma = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (S, T, sigma)))
    now = cache.alo_values(S, K, T, r, sigma, q, kind, scheme)
    later = cache.alo_values(S, K, np.maximum(T - ONE_DAY, 1e-12), r, sigma, q, kind, scheme)
    intrinsic = np.maximum(bs.option_sign(kind) * (S - K), 0.0)
    return np.where(T <= ONE_DAY, intrinsic, later) - now


def profile_lines(inputs: Inputs, metric: str, axis: str, x: np.ndarray) -> list[charts_pricing.Line]:
    """The profile's curves: the engine's call and put (solid); under the American engine the closed form as ``BSM
    ref`` (dotted); and beside theta the one-day decay (dashed), as Milestone 1 put it beside theta / 365."""
    S, K, T, r, sigma, q = _args(inputs)
    if axis == "Spot":
        args = (x, K, T, r, sigma, q)
    elif axis == "Vol":
        args = (S, K, T, r, x / 100.0, q)
    else:
        args = (S, K, x / 365.0, r, sigma, q)
    alo = _alo(inputs)
    lines = []
    for kind in cache.KINDS:
        name = kind.capitalize()
        if alo:
            s_, k_, t_, r_, v_, q_ = args
            lines.append(
                charts_pricing.Line(name, _alo_metric(metric, s_, k_, t_, r_, v_, q_, kind, inputs.model.scheme), kind)
            )
        else:
            lines.append(charts_pricing.Line(name, _bsm_metric(metric, *args, kind), kind))
    if alo:
        for kind in cache.KINDS:
            lines.append(
                charts_pricing.Line(f"{kind.capitalize()} BSM ref", _bsm_metric(metric, *args, kind), kind, "ref")
            )
    if metric == "Θ":
        decay = decay_label(axis != "Time" and to_expiry(T))
        for kind in cache.KINDS:
            if alo:
                s_, k_, t_, r_, v_, q_ = args
                y = _alo_decay(s_, k_, t_, r_, v_, q_, kind, inputs.model.scheme)
            else:
                y = closed_form_decay(*args, kind)
            lines.append(charts_pricing.Line(f"{kind.capitalize()} {decay.lower()}", y, kind, "decay"))
    return lines


_Y_TITLES = {
    "Value": "Value ($ per share)",
    "Δ": "Delta (spot)",
    "Γ": "Γ 1% (Δ per 1% move)",
    "Vega": "Vega ($ per vol point)",
    "Θ": "Θ and 1-day decay ($ per day)",
}


@st.fragment
def _profile(inputs: Inputs, market: Contract) -> None:
    """The profile's controls and chart (UI spec 3.1 item 4), in the fragment ``pricing_profile``: a control
    reruns this fragment only."""
    row = st.container(horizontal=True, key="pricing_profile_controls", vertical_alignment="bottom", gap="medium")
    metric = row.segmented_control(
        "Metric", METRICS, default="Value", required=True, key="pr_metric", bind="query-params", persist_state="session"
    )
    axis = row.segmented_control(
        "Against", AXES, default="Spot", required=True, key="pr_x", bind="query-params", persist_state="session"
    )
    choice = row.segmented_control(
        "Range",
        RANGES,
        default="±10%",
        required=True,
        key="pr_range",
        bind="query-params",
        persist_state="session",
        disabled=axis != "Spot",
        help="The spot axis: a share of spot, the 95% risk-neutral range to expiry (from the smile; lognormal at the "
        "flat vol in the textbook market), or half the lower of S and K to one and a half times the higher.",
    )
    S, K, _, _, sigma, _ = _args(inputs)
    if axis == "Spot":
        x = spot_grid(inputs, market, choice)
        x_title, current = "Spot", (S, f"S {level(S)}")
        references = [(K, f"K {strike(K, market.step)}", "strike")]
        if market.listed:
            references.append((market.forward, "F", "forward"))
    elif axis == "Vol":
        x = np.linspace(1.0, max(60.0, 2.5 * sigma * 100), OTHER_POINTS)
        x_title, current, references = "Vol (%)", (sigma * 100, f"σ {vol(sigma)}"), []
    else:
        x = np.linspace(min(1.0, inputs.days / 2), max(inputs.days * 1.5, 30.0), OTHER_POINTS)
        x_title, current, references = "Days to expiry", (inputs.days, f"{number(inputs.days, 1)}d"), []
    lines = profile_lines(inputs, metric, axis, x)
    fig = charts_pricing.profile_chart(
        x, lines, x_title=x_title, y_title=_Y_TITLES[metric], current=current, references=references
    )
    engine = engine_name(inputs)
    charts.show(
        fig,
        key="pricing_profile",
        alt=f"{engine} {metric} of the call and the put against {x_title.lower()}, at fixed vol (sticky strike)",
    )


# --- The chain ladder (UI-7) ------------------------------------------------------------------------------------


def on_chain_select(
    key: str, strikes: tuple[float, ...], priced: tuple[bool, ...] | None = None, step: float = 0.01
) -> None:
    """The chain ladder's ``on_select`` callback: price the clicked row's strike (UI spec 2.8, a selection is a
    click action). It writes the sidebar's ``strike`` (its keyboard route) and asks the chain's fragment to rerun
    the whole app, since the sidebar, the tiles and the tables read the strike only in a full run. A row beyond the
    surface's reach (``priced`` False: the "All" window lists such strikes with their quotes) has no vol to price
    it, so the strike stays and the ladder says why.

    Parameters
    ----------
    key : str
        The table's widget key, whose state holds the selection (``{"selection": {"rows": [i], ...}}``).
    strikes : tuple of float
        The strike of each row, in the table's order.
    priced : tuple of bool, optional
        Whether each row has a vol to price it; every row when omitted.
    step : float, default 0.01
        The strike interval, for the note's strike.
    """
    state = st.session_state
    selection = state.get(key) or {}
    rows = (selection.get("selection") or {}).get("rows") or []
    if not rows:
        return
    state[_CHAIN_CLICKS] = state.get(_CHAIN_CLICKS, 0) + 1  # the next run draws a fresh grid, with no selection
    row = int(rows[0])
    chosen = float(strikes[row])
    if priced is not None and not priced[row]:
        state[_CHAIN_NOTE] = (
            f"{strike(chosen, step)} lies beyond the surface's reach at this expiry: "
            "it has quotes but no vol to price it, so the strike stays."
        )
        return
    if chosen != state.get("strike"):
        state["strike"] = chosen
        state[_CHAIN_MOVED] = True


def _snap(values: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    """``(lo, hi)`` widened to the listed values that bracket them (the window's ends are listed strikes)."""
    below = values[values <= lo + 1e-9]
    above = values[values >= hi - 1e-9]
    return (float(below[-1]) if below.size else float(values[0]), float(above[0]) if above.size else float(values[-1]))


def chain_strikes(listed: np.ndarray, lo: float, hi: float, stride: int | str, current: float) -> np.ndarray:
    """The ladder's strikes: the listed ones in [lo, hi] on the stride's grid, and the Pricing strike.

    ``stride`` ``"Auto"`` is the smallest of 5, 10, 25, 50 and 100 points that gives at most :data:`CHAIN_ROWS` rows
    (before the Pricing strike is added); 100 when none does.

    The default window, 5,425 to 6,075: 50 points (13 strikes) and 5,815.

    >>> chain_strikes(np.arange(5400.0, 6101.0, 5.0), 5425.0, 6075.0, "Auto", 5815.0).size
    14
    """
    inside = listed[(listed >= lo - 1e-9) & (listed <= hi + 1e-9)]
    steps = [s for s in STRIDES if s != "Auto"]
    if stride == "Auto":
        stride = next((s for s in steps if np.count_nonzero(np.isclose(inside % s, 0)) <= CHAIN_ROWS), steps[-1])
    on_grid = inside[np.isclose(np.mod(inside, float(stride)), 0) | np.isclose(np.mod(inside, float(stride)), stride)]
    return np.unique(np.r_[on_grid, current])


def _textbook_strikes(inputs: Inputs, market: Contract, window: str) -> tuple[np.ndarray, float, float]:
    """A strike grid for the textbook market, which lists none: steps of 1, 2 or 5 times a power of ten, the
    finest that keeps the 10Δ to 10Δ window within :data:`CHAIN_ROWS` rows, and the window's ends."""
    lo, hi = _window(inputs, market, window)
    width = max(hi - lo, 1e-6)
    raw = width / CHAIN_ROWS
    power = 10.0 ** math.floor(math.log10(raw))
    step = next(m * power for m in (1, 2, 5, 10) if m * power >= raw)
    grid = np.arange(math.floor(lo / step) * step, hi + step / 2, step)
    return grid[grid > 0], lo, hi


def _flat_delta_strike(inputs: Inputs, delta: float, kind: str) -> float:
    """The strike whose closed-form spot delta at the flat vol is ``delta`` (textbook market)."""
    S, _, T, r, sigma, q = _args(inputs)
    carry = math.exp(-q * T)
    d1 = ndtri(delta / carry) if kind == "call" else -ndtri(-delta / carry)
    return S * math.exp((r - q + 0.5 * sigma * sigma) * T - d1 * sigma * math.sqrt(T))


def _window(inputs: Inputs, market: Contract, window: str) -> tuple[float, float]:
    """The ladder's strike range: 10Δ put to 10Δ call (the default), 25Δ to 25Δ, the 95% risk-neutral range, or
    every strike. Under the index the delta strikes are the listed ones at their own surface vols (RK-9a)."""
    if window == "RN 95%":
        return _rn_range(inputs, market)
    target = 0.25 if window.startswith("25") else 0.10
    if market.surface is not None and market.expiry is not None:
        if window == "All":
            listed = market.surface.strikes(market.expiry)
            return float(listed[0]), float(listed[-1])
        return (
            market.surface.delta_strike(market.expiry, -target, "put"),
            market.surface.delta_strike(market.expiry, target, "call"),
        )
    if window == "All":
        return 0.5 * inputs.spot, 1.5 * inputs.spot
    return _flat_delta_strike(inputs, -target, "put"), _flat_delta_strike(inputs, target, "call")


def chain_vols(inputs: Inputs, market: Contract, strikes: np.ndarray) -> np.ndarray:
    """The vol of each of the ladder's strikes: the published surface's (NaN beyond its reach) or the flat vol."""
    if market.surface is not None and market.expiry is not None:
        return np.array([market.surface.vol(k, market.expiry).vol for k in strikes], dtype=float)
    return np.full(strikes.shape, inputs.volatility)


def chain_frame(
    inputs: Inputs, market: Contract, strikes: np.ndarray, vols: np.ndarray | None = None, *, phone: bool = False
) -> pd.DataFrame:
    """The ladder's rows: calls (bid, ask, theo, mid IV, Δ), the strike and a mark on the Pricing strike's row, then
    puts mirrored. Theo is the engine's value at each strike's surface vol (index) or at the flat vol (textbook);
    without quotes (textbook) the bid, ask and IV columns are left out. ``vols``: :func:`chain_vols`, if read
    already.

    ``phone``: the phone's column set (:data:`PHONE_LADDER`), the strike and its mark first, then the call's mid, IV
    and Δ and the put's Δ, IV and mid (theo for mid without quotes). The mirrored ladder is about 770 px wide; on a
    390 px phone it showed five call columns and cut the strike at the edge, and a pinned strike would be drawn
    first on a desktop too."""
    S, K, T, r, _sigma, q = _args(inputs)
    vols = chain_vols(inputs, market, strikes) if vols is None else vols
    # Beyond the surface's reach the vol is NaN (on the default 20-Nov expiry the nine listed strikes 4,150 to 4,550,
    # which only the "All" window shows): those rows keep their quotes and read "–" for theo and delta.
    priced = np.isfinite(vols)
    sides = {}
    for kind in cache.KINDS:
        theo, delta = np.full(strikes.shape, np.nan), np.full(strikes.shape, np.nan)
        k, v = strikes[priced], vols[priced]
        if not priced.any():
            pass
        elif _alo(inputs):
            theo[priced] = cache.alo_values(S, k, T, r, v, q, kind, inputs.model.scheme)
            delta[priced] = _alo_metric("Δ", S, k, T, r, v, q, kind, inputs.model.scheme)
        else:
            theo[priced] = np.asarray(bs.price(S, k, T, r, v, q, kind), dtype=float)
            delta[priced] = np.asarray(bs.greeks(S, k, T, r, v, q, kind)["delta"], dtype=float)
        side = {"Theo": theo, "Δ": delta}
        if market.listed:
            quotes = []
            for k in strikes:
                try:
                    quotes.append(market.surface.quote(market.expiry, float(k), kind))
                except KeyError:
                    quotes.append(Quote(*(np.nan,) * 6))
            side |= {
                "Bid": np.array([qt.bid if qt.bid > 0 else np.nan for qt in quotes]),
                "Ask": np.array([qt.ask for qt in quotes]),
                "Mid": np.array([qt.mid for qt in quotes]),
                "IV": np.array([qt.iv_mid for qt in quotes]) * 100,
            }
        sides[kind] = side
    marks = ["←" if abs(k - K) < 1e-9 else "" for k in strikes]
    if phone:
        names = PHONE_LADDER if market.listed else ("Theo", "Δ")
        return pd.DataFrame(
            {
                "Strike": strikes,
                "Mark": marks,
                **{f"C {name}": sides["call"][name] for name in names},
                **{f"P {name}": sides["put"][name] for name in reversed(names)},
            }
        )
    order = ("Bid", "Ask", "Theo", "IV", "Δ") if market.listed else ("Theo", "Δ")
    data = {f"C {name}": sides["call"][name] for name in order}
    data["Strike"] = strikes
    # The Pricing strike's row: an arrow at its strike (UI spec 3.1 item 5 draws a left-pointing triangle, U+25C2,
    # which IBM Plex Sans lacks).
    data["Mark"] = marks
    # Mirrored the OMON way: the puts read outwards from the strike, Δ · IV · Theo · Bid · Ask.
    mirror = ("Δ", "IV", "Theo", "Bid", "Ask") if market.listed else ("Δ", "Theo")
    data |= {f"P {name}": sides["put"][name] for name in mirror}
    return pd.DataFrame(data)


_CHAIN_KINDS = {"Bid": "premium", "Ask": "premium", "Mid": "premium", "Theo": "premium", "IV": "vol", "Δ": "delta"}
#: The call's columns of the phone's ladder (:func:`chain_frame`), which the put's mirror: the market's mid and its
#: implied vol, and the delta, without the bid, ask and theo that pushed the strike off a 390 px screen.
PHONE_LADDER = ("Mid", "IV", "Δ")


def _ladder_columns(
    frame: pd.DataFrame, engine: str, listed: bool
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """The kinds, help and labels of a ladder frame's columns (:func:`chain_frame`), for :func:`.components.table`."""
    kinds, help_text, labels = {}, {}, {}
    for column in frame.columns:
        if column in ("Strike", "Mark"):
            continue
        side, name = column.split(" ", 1)
        option = "call" if side == "C" else "put"
        kinds[column] = _CHAIN_KINDS[name]
        # Each side's headers name it ("Call bid" to "Put ask"): the two halves share their column names.
        labels[column] = f"{option.capitalize()} {name if name in ('IV', 'Δ') else name.lower()}"
        help_text[column] = {
            "Bid": f"The chain's {option} bid, $ per share.",
            "Ask": f"The chain's {option} ask, $ per share.",
            "Mid": f"The chain's {option} mid, $ per share.",
            "Theo": f"{engine} {option} value at the strike's "
            + ("surface vol, $ per share. – beyond the surface's reach." if listed else "flat vol, $ per share."),
            "IV": f"Black-76 implied vol of the {option} mid. – for an offer only.",
            "Δ": f"{engine} {option} delta at the strike's "
            + ("surface vol. – beyond the surface's reach." if listed else "flat vol."),
        }[name]
    kinds |= {"Strike": "strike", "Mark": "text"}
    help_text |= {"Strike": "The strike. Click a row to price it.", "Mark": "← marks the Pricing strike."}
    labels |= {"Mark": " "}
    return kinds, help_text, labels


@st.fragment
def _chain(inputs: Inputs, market: Contract) -> None:
    """The chain ladder (UI-7), in the fragment ``pricing_chain``: its window and stride rerun this fragment only;
    a row click reruns the app at the new strike."""
    if st.session_state.pop(_CHAIN_MOVED, False):
        st.rerun()  # the click's callback moved the strike: the whole app reads it
    row = st.container(horizontal=True, key="pricing_chain_controls", vertical_alignment="bottom", gap="medium")
    window = row.segmented_control(
        "Window",
        WINDOWS,
        default=WINDOWS[0],
        required=True,
        key="pr_chain_win",
        bind="query-params",
        persist_state="session",
        help="Strikes from the 10-delta put to the 10-delta call (each strike's delta at its own vol), 25 to 25, "
        "the 95% risk-neutral range to expiry, or every listed strike.",
    )
    stride = row.selectbox(
        "Stride",
        STRIDES,
        key="pr_chain_stride",
        bind="query-params",
        persist_state="session",
        format_func=lambda s: s if s == "Auto" else f"{s} pts",
        width=120,
        help=f"Points between rows. Auto: the finest of 5, 10, 25, 50 and 100 that keeps at most {CHAIN_ROWS} rows.",
    )
    if market.listed:
        listed = market.surface.strikes(market.expiry)
        lo, hi = _snap(listed, *_window(inputs, market, window))
        strikes = chain_strikes(listed, lo, hi, stride, inputs.strike)
        step = market.step
    else:
        grid, lo, hi = _textbook_strikes(inputs, market, window)
        step = float(grid[1] - grid[0]) if grid.size > 1 else market.step
        if stride != "Auto":  # a fixed stride in points, as typed
            step = float(stride)
            grid = np.arange(math.ceil(lo / step) * step, hi + 1e-9, step)
        strikes = np.unique(np.r_[grid, inputs.strike])
    vols = chain_vols(inputs, market, strikes)
    engine = engine_name(inputs)
    # Two column sets of one ladder, of which the stylesheet shows one (static/desk.css): mirrored about the strike
    # from a 641 px screen, and on a phone the strike first, pinned, with mid, IV and Δ either side. Both run the same
    # callback.
    for family, phone in (("tbl_pricing_chain_", False), ("tbl_pricing_ladder_phone_", True)):
        frame = chain_frame(inputs, market, strikes, vols, phone=phone)
        key = current_key(family, f"{family}{st.session_state.get(_CHAIN_CLICKS, 0)}")
        kinds, help_text, labels = _ladder_columns(frame, engine, market.listed)
        with st.container(key="pricing_ladder_phone") if phone else st.container(key="pricing_ladder"):
            table(
                frame,
                kinds,
                help_text,
                labels=labels,
                key=key,
                pin=phone,  # a pinned column is drawn first: the mirrored ladder pins none
                selection_mode="single-row",
                on_select=functools.partial(
                    on_chain_select,
                    key=key,
                    strikes=tuple(float(k) for k in strikes),
                    priced=tuple(bool(v) for v in np.isfinite(vols)),
                    step=step,
                ),
            )
    if note := st.session_state.pop(_CHAIN_NOTE, None):
        st.warning(note, icon=":material/block:")
    span = f"{strike(float(strikes[0]), step)}–{strike(float(strikes[-1]), step)}"
    st.caption(
        f"{len(strikes)} strikes, {span}. Click a row to price its strike; the sidebar's strike is the keyboard route."
    )


# --- The Greeks table -------------------------------------------------------------------------------------------

#: Rows of the Greeks table after the first order: (order, name in black_scholes.GREEKS, label, unit). Each unit
#: names a row the table shows: the third order is in Γ 1%, the table's gamma, not the raw Γ per point of spot it
#: never shows (58 times smaller at the default index, so a zomma of −0.000105 read as a change in the Γ 1% of
#: +0.0937 above it was 58 times too small). :func:`higher_order` does the scaling.
_HIGHER = (
    (2, "vanna", "Vanna", "Δ per vol point"),
    (2, "vomma", "Volga", "vega per vol point"),
    (2, "charm", "Charm", "Δ per calendar day"),
    (2, "veta", "Veta", "vega per calendar day"),
    (3, "speed", "Speed", "Γ 1% per 1% move in S"),
    (3, "zomma", "Zomma", "Γ 1% per vol point"),
    (3, "color", "Color", "Γ 1% per calendar day"),
    (3, "ultima", "Ultima", "volga per vol point"),
)


def higher_order(name: str, greeks: Mapping[str, float], spot: float) -> float:
    """A second- or third-order Greek of ``black_scholes.greeks`` (raw model units) in the Greeks table's unit
    (:data:`_HIGHER`).

    The second order and ultima take ``black_scholes.GREEKS``' display scale. Zomma and color are derivatives of
    Γ 1% = Γ·S/100, so they take S/100 on top of it. Speed is the change in Γ 1% for a 1% move in S:
    (S/100)·∂(Γ·S/100)/∂S = S·(Γ + S·speed)/10⁴. At S = 100 zomma and color are unchanged and speed is
    speed + Γ/100.

    >>> raw = {"gamma": 0.02, "speed": -1e-4, "zomma": 0.05, "color": -0.01}
    >>> [round(higher_order(n, raw, 100.0), 8) for n in ("speed", "zomma", "color")]
    [0.0001, 0.0005, -2.74e-05]
    """
    value = float(greeks[name])
    if name == "speed":
        return spot * (float(greeks["gamma"]) + spot * value) / 1e4
    scaled = value * bs.GREEKS[name].scale
    return scaled * spot / 100.0 if name in ("zomma", "color") else scaled


def p_itm(inputs: Inputs, market: Contract) -> tuple[dict[str, float], dict[str, float], str]:
    """P(ITM) of the call and the put at expiry: from the smile, P(S_T > K) = 1 − CDF(K) and P(S_T < K) = CDF(K) with
    the published surface's Breeden-Litzenberger CDF (``RN smile``), or N(±d₂) at the flat vol under the textbook
    market (``flat vol``); and N(±d₂) at the contract's vol for comparison, and the basis."""
    _, K, T, _, sigma, _ = _args(inputs)
    root = sigma * math.sqrt(T)
    if root > 0:
        d2 = (math.log(market.forward / K) - 0.5 * root * root) / root
        flat = {"call": float(ndtr(d2)), "put": float(ndtr(-d2))}
    else:
        itm = float(market.forward > K) if market.forward != K else 0.5
        flat = {"call": itm, "put": 1.0 - itm}
    if market.surface is not None and market.expiry is not None:
        below = float(market.surface.cdf(K, market.expiry))
        return {"call": 1.0 - below, "put": below}, flat, "RN smile"
    return flat, flat, "flat vol"


def _money_greek(value: float, places: int) -> str:
    """A Greek in $ per share in the Greeks table: signed to ``places`` decimals, and an exact zero as ``0``."""
    return "0" if value == 0 else signed(value, places)


def greeks_frame(
    inputs: Inputs, prices: Prices, market: Contract, *, order: int, bsm_ref: bool
) -> tuple[pd.DataFrame, str]:
    """The Greeks table (UI spec 3.1 item 6) as text, per share in desk units, and its caption.

    Rows: Δ, Δ fwd, Γ 1%, vega, Θ, the one-day decay (before Θ when it leads: :func:`.format.decay_leads`), ρ, ε;
    from ``order`` 2 vanna, volga, charm and veta; from 3 speed, zomma, color and ultima; then P(ITM). Columns: the
    engine's call and put, and the closed form's beside them under the American engine with ``bsm_ref``. The
    American engine gives first-order Greeks only: its higher-order cells read ``–``."""
    S, K, T, r, sigma, q = _args(inputs)
    engine = engine_name(inputs)
    alo = _alo(inputs)
    raw = engine_greeks(inputs, prices)
    bsm = {kind: bs.greeks(*_args(inputs), kind) for kind in cache.KINDS}
    columns = [(f"Call ({engine})", raw, "call"), (f"Put ({engine})", raw, "put")]
    if alo and bsm_ref:
        columns += [("Call BSM ref", bsm, "call"), ("Put BSM ref", bsm, "put")]
    decay = greeks_decay(inputs, prices)
    expiring = to_expiry(T)
    first = (
        ("delta", "Δ", "spot, per 1 point of S", lambda u: signed(u["delta"], 3)),
        ("delta_fwd", "Δ fwd", "∂V/∂F, per 1 point of F", lambda u: signed(u["delta_fwd"], 3)),
        ("gamma", "Γ 1%", "Δ per 1% move in S", lambda u: _sig(u["gamma"])),
        ("vega", "Vega", "$ per vol point", None),
        ("theta", "Θ", "$ per calendar day", None),
        ("rho", "ρ", "$ per 1% rate", None),
        ("epsilon", "ε", "$ per 1% dividend yield", None),
    )
    units = {
        header: desk_units({k: float(v) for k, v in source[kind].items()}, S, r, q, T)
        for header, source, kind in columns
    }
    rows = []
    for name, label, unit, text in first:
        row = {"Greek": label, "Unit": unit}
        if text is None:
            # Money per share: cents, or two significant digits of the row's largest value below 10 cents (the
            # textbook put's Θ of -0.0045 read 0.00), one precision across the row; an exact zero (an option
            # exercised at once) reads 0, as Γ's does, not 0.00000 beside a call's -0.00068.
            places = cents_places(max((abs(u[name]) for u in units.values() if np.isfinite(u[name])), default=0.0))
            row |= {header: _money_greek(u[name], places) for header, u in units.items()}
        else:
            row |= {header: text(u) for header, u in units.items()}
        rows.append(row)
    # The one-day decay by repricing, beside theta: first when it leads (Milestone 1's rule).
    theta = {kind: raw[kind]["theta"] / 365.0 for kind in cache.KINDS}
    leads = any(decay_leads(theta[kind], decay[kind], expiring) for kind in cache.KINDS)
    decay_row = {"Greek": decay_label(expiring), "Unit": "$, repriced" + (" at expiry" if expiring else " a day on")}
    decays = {
        header: decay[kind] if source is raw else float(closed_form_decay(S, K, T, r, sigma, q, kind))
        for header, source, kind in columns
    }
    places = cents_places(max((abs(v) for v in decays.values() if np.isfinite(v)), default=0.0))
    decay_row |= {header: _money_greek(value, places) for header, value in decays.items()}
    at = next(i for i, row in enumerate(rows) if row["Greek"] == "Θ") + (0 if leads else 1)
    rows.insert(at, decay_row)
    for row_order, name, label, unit in _HIGHER:
        if row_order > order:
            continue
        row = {"Greek": label, "Unit": unit}
        for header, source, kind in columns:
            known = source is bsm or not alo
            row[header] = _sig(higher_order(name, source[kind], S)) if known else MISSING
        rows.append(row)
    rn, flat, basis = p_itm(inputs, market)
    row = {"Greek": "P(ITM)", "Unit": f"at expiry, {basis}"}
    for header, _, kind in columns:
        row[header] = probability_text(rn[kind])
    rows.append(row)
    parts = []
    if basis == "RN smile":
        parts.append(
            f"P(ITM) from the smile (Breeden-Litzenberger); N(d₂) at flat vol: {probability_text(flat['call'])} call, "
            f"{probability_text(flat['put'])} put."
        )
    if alo and order > 1:
        parts.append("ALO gives first-order Greeks; BSM ref shows the higher orders.")
    # The values next to their Greek and the unit last: on a 390 px phone the unit column took the room of the values.
    frame = pd.DataFrame(rows)
    return frame[["Greek", *(c for c in frame.columns if c not in ("Greek", "Unit")), "Unit"]], " ".join(parts)


def _greeks(inputs: Inputs, prices: Prices, market: Contract) -> None:
    row = st.container(horizontal=True, key="pricing_greeks_controls", vertical_alignment="bottom", gap="medium")
    order = row.segmented_control(
        "Order",
        ORDERS,
        default="1st",
        required=True,
        key="pr_greeks",
        bind="query-params",
        persist_state="session",
        help="First-order Greeks, then also vanna, volga, charm and veta, then also speed, zomma, color and ultima.",
    )
    bsm_ref = False
    if _alo(inputs):
        bsm_ref = row.toggle(
            "BSM ref",
            value=False,
            key="pr_bsm_ref",
            bind="query-params",
            persist_state="session",
            help="Show the closed-form European Greeks beside the American engine's.",
        )
    frame, caption = greeks_frame(inputs, prices, market, order=ORDERS.index(order) + 1, bsm_ref=bsm_ref)
    engine = f"{inputs.model.label}, per share."
    if _alo(inputs):
        engine = alo_greeks_note(inputs, prices)
        if any(prices.alo_greeks[k].basis != "differences" or prices.alo_greeks[k].one_sided for k in cache.KINDS):
            caption = f"{caption} Some ALO Greeks are exact or one-sided here: see the column's help.".strip()
    config = {
        # 120 px holds "Decay to expiry", which the automatic width cut to "1-day de".
        "Greek": st.column_config.TextColumn("Greek", help="The sensitivity.", pinned=True, width=120),
        "Unit": st.column_config.TextColumn("Unit", help="What one unit of the figure is, per share.", width="medium"),
    }
    for column in frame.columns[1:-1]:
        source = "Closed-form European (Black-Scholes-Merton), per share." if "BSM ref" in column else engine
        # 110 px: the automatic width fitted the widest value to the cell's edges (Ultima's −0.0000136 touched both
        # borders, and a clipped color read ")0.0000292"); this leaves its padding.
        config[column] = st.column_config.TextColumn(column, help=md(source), alignment="right", width=110)
    st.dataframe(
        frame,
        key="tbl_pricing_greeks",
        width="content",  # stretched over 1,000 px, its value columns read far from their labels
        hide_index=True,
        row_height=ROW_HEIGHT,
        height="content",
        column_config=config,
    )
    if caption:
        st.caption(caption)


#: How the American engine's Greeks are computed (american.GREEK_BUMPS).
ALO_GREEKS_NOTE = (
    "American Greeks of the sidebar engine, per share in the units shown: central differences of its value with the "
    "spot and the time to expiry moved by ±0.01%, the vol by ±0.01 vol point and the rate and dividend yield by ±1 "
    "bp, the scheme and the boundary equation held fixed."
)
#: At a rate and dividend yield of zero both options are European, at the edge of early exercise (american.greeks).
ALO_ZERO_CARRY_NOTE = (
    "At a rate and dividend yield of zero, a 1\xa0bp move one way makes early exercise pay (a lower rate or a higher "
    "dividend yield for the call, a higher rate or a lower dividend yield for the put), so rho and dividend rho are "
    "one-sided here: the slopes on the side where it does not."
)
#: The Greeks as the note names them.
_GREEK_WORDS = {"epsilon": "dividend rho"}


def _join(words: list[str]) -> str:
    return " and ".join(words) if len(words) < 3 else ", ".join(words[:-1]) + " and " + words[-1]


def alo_greeks_note(inputs: Inputs, prices: Prices) -> str:
    """How the American engine's Greeks are computed, and for which option each exception holds
    (``AmericanGreeks.basis`` and ``one_sided``, and the fallback to the tree): the help of the Greeks table's
    engine columns."""
    greeks, details = prices.alo_greeks, prices.alo
    notes = [ALO_GREEKS_NOTE]
    closed = [kind for kind in cache.KINDS if greeks[kind].basis == "closed form"]
    if len(closed) == 2:
        notes.append(
            "The call and the put equal their Black-Scholes-Merton values, so their Greeks are the closed-form ones."
        )
    elif closed:
        notes.append(f"The {closed[0]} equals its Black-Scholes-Merton value, so its Greeks are the closed-form ones.")
    if len(closed) == 2 and inputs.rate == 0.0 and inputs.dividend_yield == 0.0:
        notes.append(ALO_ZERO_CARRY_NOTE)
    for kind in cache.KINDS:
        if greeks[kind].basis == "exercise value":
            side, payoff, delta = ("above", "S − K", "1") if kind == "call" else ("below", "K − S", "−1")
            notes.append(
                f"The {kind} is exercised at once at this spot (at or {side} its critical spot), so its Greeks are "
                f"those of its exercise value {payoff}: delta {delta}, the others zero."
            )
        if sided := [_GREEK_WORDS.get(name, name) for name in greeks[kind].one_sided]:
            notes.append(
                f"The {kind}'s {_join(sided)} {'is' if len(sided) == 1 else 'are'} one-sided: a bump one way would "
                "change how the option is valued (a different exercise regime, the tree instead of the engine or "
                "the reverse, or exercise at once), so the difference is taken on the other side alone."
            )
        if details[kind].fallback:
            notes.append(
                f"The {kind}'s are differences of tree values (see the warning above), far less smooth than the "
                "engine's."
            )
    return " ".join(notes)


# --- Engine diagnostics -----------------------------------------------------------------------------------------


def _diagnostic_settings(inputs: Inputs) -> tuple[ModelConfig, ModelConfig]:
    """The diagnostics' own tree and Monte Carlo settings, in a popover: ``(tree, monte_carlo)``."""
    default = ModelConfig()
    with st.popover("Settings", icon=":material/tune:", key="pricing_diag_settings"):
        method = st.segmented_control(
            "Lattice",
            list(binomial.METHODS),
            default=default.tree_method,
            required=True,
            format_func=binomial.METHODS.get,
            key="diag_tree",
            persist_state="session",
        )
        steps = st.slider("Steps", 10, 2000, default.steps, step=1, key="diag_steps", persist_state="session")
        n_paths = st.select_slider(
            "Monte Carlo paths",
            PATH_CHOICES,
            value=default.n_paths,
            format_func=lambda n: number(n, 0),
            key="diag_paths",
            persist_state="session",
        )
        seed = st.number_input("Seed", 0, 2**32 - 1, default.seed, step=1, key="diag_seed", persist_state="session")
        antithetic = st.toggle("Antithetic variates", default.antithetic, key="diag_av", persist_state="session")
        control = st.toggle(
            "Control variate (discounted spot)", default.control_variate, key="diag_cv", persist_state="session"
        )
    american_ = _alo(inputs)
    tree = ModelConfig(kind="binomial", tree_method=method, steps=int(steps), american=american_)
    mc = ModelConfig(
        kind="monte_carlo", n_paths=int(n_paths), seed=int(seed), antithetic=antithetic, control_variate=control
    )
    return tree, mc


def engine_rows(inputs: Inputs, tree: ModelConfig, mc: ModelConfig) -> tuple[pd.DataFrame, str]:
    """The engine comparison (UI spec 3.1 item 7): each engine's call and put, their difference from the reference
    and the milliseconds they took. The reference is the closed form for European exercise and the American engine
    at its high scheme for American: on the 4,495 puts of Healy's grid it is within 5.4e−8 (RMSE 1.8e−9) of QuantLib's
    QdFp at its strict tanh-sinh scheme (``docs/methods/american_alo.md``)."""
    S, K, T, r, sigma, q = _args(inputs)
    american_ = _alo(inputs)
    rows = []
    bsm_values = {kind: float(bs.price(S, K, T, r, sigma, q, kind)) for kind in cache.KINDS}
    if american_:
        for scheme in ("fast", "accurate", "high"):
            values, ms = cache.alo_timed(S, K, T, r, sigma, q, scheme)
            rows.append({"Engine": f"ALO {scheme}", **values, "ms": ms, "interval": MISSING})
        reference, ref_name = rows[-1], "ALO high"
    style = "American" if american_ else "European"
    trees = list(DIAGNOSTIC_TREES)
    if (tree.tree_method, tree.steps) not in trees:
        trees.append((tree.tree_method, tree.steps))
    for method, steps in trees:
        try:
            values, ms = cache.tree_timed(S, K, T, r, sigma, q, steps, american_, method)
        except ValueError:  # CRR has no arbitrage-free lattice at this step count
            values, ms = dict.fromkeys(cache.KINDS, np.nan), np.nan
        name = "LR" if method == "lr" else "CRR"
        rows.append({"Engine": f"{name} {number(steps, 0)} · {style}", **values, "ms": ms, "interval": MISSING})
    bsm_row = {"Engine": "BSM · European", **bsm_values, "ms": np.nan, "interval": MISSING}
    if american_:
        rows.append(bsm_row)
    else:
        rows.insert(0, bsm_row)
        reference, ref_name = bsm_row, "BSM"
    results, ms = cache.mc_timed(S, K, T, r, sigma, q, mc.n_paths, mc.seed, mc.antithetic, mc.control_variate)
    flags = [mc_interval(results[kind], dollars=False) for kind in cache.KINDS]
    rows.append(
        {
            "Engine": f"MC {number(mc.n_paths, 0)} · European",
            **{kind: results[kind].price for kind in cache.KINDS},
            "ms": ms,
            "interval": f"call {flags[0]}; put {flags[1]}",
        }
    )
    frame = pd.DataFrame(rows)
    frame["Call − ref"] = [signed_sci(v) for v in frame["call"] - reference["call"]]
    frame["Put − ref"] = [signed_sci(v) for v in frame["put"] - reference["put"]]
    frame = frame.rename(columns={"call": "Call", "put": "Put", "interval": "MC 95% interval"})
    return frame[["Engine", "Call", "Put", "Call − ref", "Put − ref", "MC 95% interval", "ms"]], ref_name


def _convergence(inputs: Inputs, tree: ModelConfig, ref_name: str, reference: dict[str, float]) -> None:
    """|tree − reference| against the steps, log-log, for both lattices and both options. A lattice whose call and
    put errors are one (European exercise: the tree keeps put-call parity, so they differ by a constant) draws one
    line, ``LR call and put``: the put's line hid the call's."""
    S, K, T, r, sigma, q = _args(inputs)
    series = cache.tree_convergence(S, K, T, r, sigma, q, tree.steps, tree.american)
    lines = {}
    for name in series["call"]:
        short = "LR" if name.startswith("Leisen") else "CRR"
        dash = "solid" if short == "LR" else "dot"
        errors = {}
        for kind in cache.KINDS:
            steps, values = series[kind][name]
            errors[kind] = np.asarray(values, dtype=float) - reference[kind]
        # Equal to a part in 1e6 of the error, or to 1e-9 (a tenth of the engine table's last digit).
        if np.allclose(errors["call"], errors["put"], rtol=1e-6, atol=1e-9, equal_nan=True):
            lines[f"{short} call and put"] = (steps, errors["call"], "call", dash)
        else:
            lines |= {f"{short} {kind}": (steps, errors[kind], kind, dash) for kind in cache.KINDS}
    charts.show(
        charts_pricing.error_chart(lines, reference=ref_name, guides=(("CRR", 1), ("LR", 2))),
        key="pricing_convergence",
        alt=f"Absolute error of the trees against {ref_name} by number of steps, log-log",
    )
    st.caption(
        f"Up to the settings' {number(tree.steps, 0)} steps. CRR converges at first order, oscillating with the "
        "strike's place among the nodes; LR at second."
    )


def _mc_convergence(inputs: Inputs, mc: ModelConfig) -> None:
    """Monte Carlo's 95% interval width against the paths, with a 1/√N guide (UI spec 3.1 item 7), on the settings'
    seed and variance reduction, from 1,000 paths to the settings' paths."""
    S, K, T, r, sigma, q = _args(inputs)
    paths, widths = cache.mc_convergence(S, K, T, r, sigma, q, mc.n_paths, mc.seed, mc.antithetic, mc.control_variate)
    charts.show(
        charts_pricing.interval_chart(paths, widths),
        key="pricing_mc_convergence",
        alt="Width of the Monte Carlo 95% interval of the call and the put by number of paths, log-log",
    )
    st.caption(
        f"Monte Carlo, European, seed {mc.seed}: the interval narrows as 1/√N, by half for four times the paths "
        "(shown even where it is withheld)."
    )


def _tree_greeks(inputs: Inputs, tree: ModelConfig) -> None:
    """The settings tree's own Greeks, with its one-day decay (Milestone 1): on the lattice, so they include early
    exercise under the American engine."""
    S, K, T, r, sigma, q = _args(inputs)
    try:
        results = cache.tree_results(S, K, T, r, sigma, q, tree.steps, tree.american, tree.tree_method)
        decay, trees = tree_decay(inputs, tree, results)
    except ValueError:  # a CRR lattice that is not arbitrage-free at these steps
        st.warning(
            f"{tree.label}: the lattice is not arbitrage-free at these inputs; raise the steps or use Leisen-Reimer."
        )
        return
    expiring = to_expiry(T)
    label = decay_label(expiring)
    leads = any(decay_leads(res.theta / 365, decay[kind], expiring) for kind, res in results.items())
    rows = []
    for kind, res in results.items():
        time_columns = {"Θ": res.theta / 365, label: decay[kind]}
        rows.append(
            {"Option": kind.capitalize(), "Value": res.price, "Δ": res.delta, "Γ 1%": res.gamma * S / 100}
            | (dict(reversed(time_columns.items())) if leads else time_columns)
            | {"Vega": res.vega / 100, "ρ": res.rho / 100}
        )
    frame = pd.DataFrame(rows)
    kinds = {"Option": "text", "Value": "premium_diag", "Δ": "delta", "Γ 1%": "gamma", "Θ": "theta_share",
             label: "theta_share", "Vega": "vega_share", "ρ": "rho_share"}  # fmt: skip
    repricing = {
        (): "the intrinsic value at expiry minus the tree value",
    }.get(trees, "the value on " + " and ".join(number(n, 0) for n, _ in trees) + "-step trees a day shorter minus now")
    help_text = {
        "Option": "The option.",
        "Value": "The tree's value, $ per share.",
        "Δ": "From the tree's first two time steps.",
        "Γ 1%": "Change in delta for a 1% move, from the tree's first two time steps.",
        "Θ": "From the second time step (CRR) or by repricing with the expiry moved (LR), ÷ 365.",
        label: f"{label}: {repricing}.",
        "Vega": "Repriced on a tree with the same steps, vol ±0.5 vol point.",
        "ρ": "Repriced on a tree with the same steps, rate ±25 bp.",
    }
    labels = {label: f"{label} ($)"}
    st.markdown(f"**{tree.label}**")
    table(frame, kinds, help_text, labels=labels, key="tbl_pricing_tree_greeks")


def _alo_boundary(inputs: Inputs, prices: Prices) -> None:
    """The American engine's results and its early-exercise boundary (x-axis from day 0)."""
    K, T, r, sigma, q = inputs.strike, inputs.expiry, inputs.rate, inputs.volatility, inputs.dividend_yield
    boundaries, missing = {}, []
    for kind, res in prices.alo.items():
        if res.method == "alo":
            tau, spots = cache.alo_boundary(K, T, r, sigma, q, kind, inputs.model.scheme)
            boundaries[f"{kind.capitalize()} boundary"] = ((T - tau[::-1]) * 365, spots[::-1])
        elif res.method == "european":
            missing.append(f"No {kind} boundary: {NEVER_EARLY[kind]}.")
        elif res.fallback:
            missing.append(f"No {kind} boundary: {FALLBACK_REASONS[res.fallback_reason]}.")
        else:
            missing.append(f"No {kind} boundary: the {kind} is valued in closed form here.")
    rows = [
        {
            "Option": kind.capitalize(),
            "American": res.value,
            "European": res.european,
            "Premium": res.premium,
            "Critical spot": res.critical_spot if res.method == "alo" and np.isfinite(res.critical_spot) else np.nan,
            "Method": ALO_METHODS.get(res.method, res.method),
        }
        for kind, res in prices.alo.items()
    ]
    table(
        pd.DataFrame(rows),
        {
            "Option": "text",
            "American": "premium_diag",
            "European": "premium_diag",
            "Premium": "premium_diag",
            "Critical spot": "level",
            "Method": "text",
        },
        {
            "Option": "The option.",
            "American": "The engine's value, $ per share.",
            "European": "The closed-form European value, $ per share.",
            "Premium": "American minus European: the right to exercise early.",
            "Critical spot": "Exercise at once at or below it (put), at or above it (call). – where none applies.",
            "Method": "How the engine valued the option.",
        },
        key="tbl_pricing_alo",
    )
    if boundaries:
        fig = charts.exercise_boundary_chart(boundaries, K)
        fig.update_xaxes(range=[0.0, T * 365])
        charts.show(
            fig,
            key="pricing_boundary",
            alt="Early-exercise boundary of the American engine: the critical spot against days from today",
        )
    for line in missing:
        st.caption(line)


def _diagnostics(inputs: Inputs, prices: Prices) -> None:
    """The contents of Engine diagnostics (computed only while it is open)."""
    tree, mc = _diagnostic_settings(inputs)
    frame, ref_name = engine_rows(inputs, tree, mc)
    reference = frame.loc[frame["Engine"].str.startswith("ALO high" if ref_name == "ALO high" else "BSM")].iloc[0]
    table(
        frame,
        {
            "Engine": "text",
            "Call": "premium_diag",
            "Put": "premium_diag",
            "Call − ref": "difference",
            "Put − ref": "difference",
            "MC 95% interval": "text",
            "ms": "level",
        },
        {
            "Engine": "The engine, its steps or paths, and the exercise it values.",
            "Call": "$ per share.",
            "Put": "$ per share.",
            "Call − ref": f"The call minus {ref_name}'s, $ per share.",
            "Put − ref": f"The put minus {ref_name}'s, $ per share.",
            "MC 95% interval": f"Price ± 1.96 SE; withheld as unreliable below {RELIABLE_PATHS} paths on the scarce "
            "side of the strike.",
            "ms": "Server time of the first computation; later runs read a cache.",
        },
        labels={"ms": "Time (ms)"},
        key="tbl_pricing_engines",
    )
    st.caption(f"Reference: {ref_name}. Monte Carlo values European exercise only.")
    _convergence(inputs, tree, ref_name, {"call": float(reference["Call"]), "put": float(reference["Put"])})
    _mc_convergence(inputs, mc)
    _tree_greeks(inputs, tree)
    if prices.alo is not None:
        _alo_boundary(inputs, prices)


# --- The tab ----------------------------------------------------------------------------------------------------


def footer_text(inputs: Inputs, market: Contract) -> str:
    """The footer line: the engine, the time basis, the vol's source and the unit."""
    engine = "ALO · American" if _alo(inputs) else "BSM · European"
    if inputs.expiry_date is not None and inputs.settlement is not None:
        settles = clock.settlement_time(inputs.expiry_date, inputs.settlement)
        time_basis = f"T to {_day(settles.date())} {settles:%H:%M} ET = minutes/525,600"
    else:
        time_basis = "T = days/365"
    if market.listed:
        at = fmt_moneyness(inputs.strike, market.forward)
        source = "typed σ" if inputs.vol_overridden else f"σ from the surface at K/F {at}"
    else:
        source = "flat σ"
    return f"{engine} · {time_basis} · {source} · per share"


def render(inputs: Inputs, prices: Prices) -> None:
    """The Pricing tab (see the module docstring)."""
    market = contract(inputs)
    # The context line names the tiles under it: the two sit half the usual gap apart (7 px, not 14), which keeps the
    # tiles' top within UI spec 3.1's 180 px at 1440 x 900 under the two-line header (179 px, from 186).
    with st.container(key="pricing_headline", gap="xsmall"):
        st.caption(context_line(inputs))
        _tiles(inputs, prices, market)
    frame, kinds, help_text, labels = contract_frame(inputs, prices, market)
    table(frame, kinds, help_text, labels=labels, key="tbl_pricing_contract")
    # The profile and the Greeks table side by side where there is room (desk.css, Pricing tab), stacked otherwise.
    with st.container(key="pricing_views"), st.container(key="pricing_views_row"):
        with st.container(key="pricing_views_profile"):
            _profile(inputs, market)
        with st.container(key="pricing_views_greeks"):
            _greeks(inputs, prices, market)
    st.subheader("Chain")
    _chain(inputs, market)
    # Bound to the URL, so that a link from this view opens it again; the key is DIAGNOSTICS, written out for the
    # link test's reader (tests/test_app_header.py reads the keys of the widgets bound to the URL).
    diagnostics = st.expander("Engine diagnostics", key="pricing_diag", on_change="rerun", bind="query-params")
    if diagnostics.open:
        with diagnostics, st.spinner("Pricing on the trees and by simulation…"):
            _diagnostics(inputs, prices)
    foot = st.container(horizontal=True, key="pricing_footer", vertical_alignment="center", gap="small")
    foot.caption(footer_text(inputs, market))
    foot.button("Methodology", key="pricing_methodology", type="tertiary", on_click=tab_methodology.open_drawer)
