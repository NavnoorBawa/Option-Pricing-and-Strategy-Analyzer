"""Risk tab (UI spec 3.3, BK-1): what the position can lose and why.

The tab opens with the position line and the what-if strip (:mod:`.whatif`, shared with the Strategy tab), then six
pinned tiles at the what-if state ($Δ with its ES-equivalent contracts, Γ P&L 1%, vega with its one-month weighting,
Θ with the breakeven move against the implied move, the worst cell of the desk grid and the worst named stress), then
one report at a time, chosen by the report selector: Ladder, Scenarios, Stress, VaR, Vega and Explain. Only the
selected report computes. No tile shows a statistic of a simulation or a history (spec section 5 item 19): VaR and ES
live in their report.

Every number re-prices the position built on the Strategy tab leg by leg with Black-Scholes-Merton (European legs,
whatever the sidebar's engine), so the tab agrees with the Strategy tab to the cent.

The what-if state is the position revalued as if the scenario were now: spot, vols, rate and date moved by the strip
(:func:`.whatif.apply_position`). The tiles, the Ladder, Stress, VaR and Vega reports read it. The Scenarios grid and
the WORST GRID tile read today's market instead, because the grid is where a scenario is picked: a click on a cell
copies its spot, vol and date into the strip (UI-10, :func:`on_scenario_select`), and the strip's cell is outlined
in amber. The Explain report explains the move from today to a target: the what-if, a stress or the next session.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from .. import attribution, clock, margin, risk, scenarios
from .. import strategies as strat
from . import cache, charts, charts_risk, whatif
from . import market as mk
from .components import Card, card_row, table
from .format import (
    MINUS,
    MISSING,
    NB_HYPHEN,
    SHORT_SCALES,
    decay_label,
    decay_leads,
    futures_equiv,
    md,
    money,
    money_short,
    number,
    right_aligned,
    session_label,
    signed,
    time_left,
    vol,
)
from .format import date as date_text
from .inputs import Inputs
from .tab_strategy import Position

#: The reports, in the selector's order (``?report=VaR``).
REPORTS = ("Ladder", "Scenarios", "Stress", "VaR", "Vega", "Explain")
#: The VaR simulation's default implied-vol dynamics, an equity index's (VIX-like): one-month implied vol with a
#: volatility of 100% a year, moving against spot with a correlation of −0.7. With implied vol frozen instead, a
#: 45-day at-the-money short straddle on an index (spot 5,800, vol 16%, r 4.3%, q 1.3%, one 100-share contract)
#: shows a one-day 99% VaR of $2,833 (ES $3,683) against $3,862 ($4,737) with these dynamics (100,000 paths, seed
#: 7): 27% less, on the most common vol position.
INDEX_VOL_OF_VOL = 1.0
INDEX_SPOT_VOL_CORRELATION = -0.7
#: VaR at 99% and ES at 97.5% (FRTB's level, BCBS MAR33), read off one simulation.
VAR_CONFIDENCE, ES_CONFIDENCE = 0.99, 0.975
PATH_CHOICES = (10_000, 25_000, 50_000, 100_000, 250_000, 500_000)
#: The VaR report's models: the Monte Carlo return models of :mod:`options_analyzer.risk` (historical simulation and
#: filtered historical simulation, RK-5, are not on this branch).
#: Short names, so that the four fit one line of a 390 px phone ("GARCH-t MC" ran out of it); the tiles say MC.
VAR_MODELS: dict[str, tuple[str, bool]] = {
    "Normal": ("normal", False),
    "Student-t": ("student_t", False),
    "GARCH": ("garch", False),
    "GARCH-t": ("garch", True),
}

#: The desk grid (UI spec 3.3, Scenarios): spot moves in percent by vol shocks in each vol mode.
DESK_SPOT = (-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0)
DESK_VOL = {"vp": (-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0), "relative": (-50.0, -25.0, -10.0, 0.0, 10.0, 25.0, 50.0)}
DESK_VOL["1M-scaled"] = DESK_VOL["relative"]
#: The grid's vol modes, and the what-if strip's and the scenario library's names for each.
GRID_VOL_MODES = {
    "vp": ("absolute", "absolute"),
    "relative": ("relative", "relative"),
    "1M-scaled": ("1M-scaled", "1m"),
}
GRID_AXES = ("Spot × Vol", "Spot × Date")
GRIDS = ("Desk 9×7", "Portfolio margin")
#: The ladder's moves in daily standard deviations (``lad_axis`` σ) and its rolls.
LADDER_SIGMA = (-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
LADDER_ROLLS = ("Today", "Next session", "+1w")
LADDER_METRICS = {
    "P&L": ("pnl", "P&L ($)"),
    "$Δ": ("dollar_delta", "$Δ ($)"),
    "Γ P&L 1%": ("dollar_gamma_1pct", "Γ P&L 1% ($)"),
    "Vega": ("vega", "Vega ($/vp)"),
    "Θ": ("theta", "Θ ($/day)"),
}
#: Labels of the vega buckets of ``scenarios.vega_buckets``.
DELTA_BUCKET_LABELS = {"10d_put": "10Δp", "25d_put": "25Δp", "atm": "ATM", "25d_call": "25Δc", "10d_call": "10Δc"}
#: The stress groups.
STRESS_GROUPS = ("Historical", "Hypothetical", "Custom")
ALL_STRESSES = (*scenarios.HISTORICAL_SCENARIOS, *scenarios.HYPOTHETICAL_SCENARIOS)
NEXT_SESSION = "Next session (time only)"
WHAT_IF = "What-if (the strip)"


def shock_label(x: float) -> str:
    """A relative shock as a percentage: whole percents when exact, otherwise one decimal; zero unsigned; U+2212 for
    minus."""
    pct = 100.0 * x
    if abs(pct) < 1e-9:
        return "0%"
    return signed(pct, 0 if abs(pct - round(pct)) < 1e-9 else 1, suffix="%")


def _minus(text: str) -> str:
    """``text`` with the minus sign U+2212 for the leading hyphen-minus that :func:`money` and :func:`number` write."""
    return MINUS + text[1:] if text.startswith("-") else text


def _signed_short(value: float) -> str:
    """:func:`money_short` with an explicit sign, the minus sign U+2212: ``+$1,029`` or ``−$18.40``."""
    return _minus(money_short(value, signed=True))


def dollars(value: float) -> str:
    """A position's dollar figure on a tile, signed (U+2212): whole dollars from $10 (``−$7,663``), cents below
    (``+$4.20``), and ``−$1.23M`` from a million, with the exact amount in the tile's help."""
    if value is None or not math.isfinite(value):
        return signed(value)
    if round(abs(value)) >= 1e6:
        return _signed_short(value)
    return signed(value, 0 if round(abs(value), 2) >= 10 else 2, "$")


def shown(value: float) -> float:
    """The amount :func:`dollars` shows for ``value``, as a number (−1,728.6 reads ``−$1,729``, so −1,729), so that a
    difference of two tiles adds up on screen."""
    if not math.isfinite(value):
        return value
    text = dollars(value).replace(MINUS, "-").replace("$", "").replace(",", "").lstrip("+")
    scale = {suffix: size for size, suffix in SHORT_SCALES}.get(text[-1])
    return float(text) if scale is None else float(text[:-1]) * scale


#: Scales of a grid's printed cells: the largest |P&L| from which cells read in thousands or millions of dollars.
_GRID_SCALES = ((1e6, 1e6, "$M"), (1e3, 1e3, "$k"), (0.0, 1.0, "$"))


def grid_scale(bound: float) -> tuple[float, str]:
    """``(divisor, unit)`` of a grid's printed cells for its largest |P&L| ``bound``: ``$k`` from $1,000, ``$M`` from
    a million, so that a cell reads at most 5 characters (``−3.6``, ``+216``) and nine spot columns stay legible in
    the 33 px cells of a 390 px phone. The axis title names the unit; the hover and the table give exact dollars.

    >>> grid_scale(4096.0), grid_scale(999.0), grid_scale(2.5e6)
    ((1000.0, '$k'), (1.0, '$'), (1000000.0, '$M'))
    """
    return next((divisor, unit) for floor, divisor, unit in _GRID_SCALES if bound >= floor)


def grid_cell_text(value: float, bound: float) -> str:
    """A grid cell in the unit of :func:`grid_scale`: one decimal below 100 of the unit (``−3.6``), none from 100
    (``+216``); cents below $10; signed with U+2212, ``0`` for a value that rounds to zero.

    >>> grid_cell_text(-3611.4, 4096.0), grid_cell_text(257.0, 4096.0), grid_cell_text(-4.2, 8.0)
    ('−3.6', '+0.3', '−4.20')
    """
    divisor, unit = grid_scale(bound)
    scaled, top = value / divisor, bound / divisor
    decimals = 2 if unit == "$" and bound < 10 else (0 if unit == "$" or top >= 100 else 1)
    text = signed(scaled, decimals)
    return "0" if text.strip("0.,") == "" else text


def level_label(value: float) -> str:
    """A spot level on a grid axis: whole points from 1,000 (``5,655``, ``5,426`` for 5,426.22) and when whole, else
    two decimals (``97.50``)."""
    return f"{value:,.0f}" if abs(value) >= 1000 or abs(value - round(value)) < 1e-9 else f"{value:,.2f}"


def spot_label(spot: float, change: float) -> str:
    """A grid's spot axis label, level and change: ``5,655 (−2.5%)`` (UI spec 2.7 item 7)."""
    return f"{level_label(spot * (1 + change))} ({shock_label(change)})"


def vol_label(shock: float, mode: str) -> str:
    """A grid's vol axis label: ``+2 vp`` or ``−25%`` (relative and 1M-scaled)."""
    return signed(shock, 0, suffix=" vp" if mode == "vp" else "%") if shock else ("0 vp" if mode == "vp" else "0%")


# --- the book: today's position and the what-if ---------------------------------------------------------------------


@dataclass(frozen=True)
class Book:
    """The position today and at the what-if state, with the market facts the tab reads.

    ``strategy`` and ``market`` are the what-if state (equal to today's while no shift is set); ``position`` holds
    today's. ``front`` is the front option leg's settlement time.
    """

    position: Position
    strategy: strat.Strategy
    market: strat.Market
    shifts: whatif.Shifts
    as_of: datetime
    front: datetime
    index: bool
    context: mk.MarketContext | None = None

    @property
    def units(self) -> float:
        return self.position.units

    def scenario_time(self) -> datetime:
        return whatif.scenario_time(self.as_of, self.shifts.sessions)

    def atm_vol(self, tau: float, market: strat.Market | None = None) -> float:
        """The at-the-money-forward vol at ``tau`` years: the published surface's under the index preset, else the
        market's one vol (textbook), before any what-if vol shift."""
        market = market or self.position.market
        if self.context is None:
            return market.volatility
        forward = market.spot * math.exp((market.rate - market.dividend_yield) * tau)
        return float(self.context.surface.implied_vol_at(forward / market.spot, max(tau, 1e-6), extrapolate=True))


def valuation_time(inputs: Inputs) -> datetime:
    """T+0 of the tab: the valuation time under the index preset; the last cash close under the textbook preset,
    which has no valuation time of its own (its times to expiry are in days)."""
    return inputs.context.as_of if inputs.context is not None else mk.last_close()


def _front(as_of: datetime, years: float) -> datetime:
    return as_of + timedelta(minutes=years * clock.MINUTES_PER_YEAR)


def next_session_days(start: datetime) -> float:
    """Calendar days from ``start`` to the same time of the next trading session (Friday 16:00 to Monday 16:00 is 3,
    as theta counts it).

    >>> next_session_days(datetime(2026, 10, 23, 16, 0)), next_session_days(datetime(2026, 11, 25, 16, 0))
    (3.0, 2.0)
    """
    later = whatif.scenario_time(start, 1)
    return round(clock.year_fraction(start, later) * clock.MINUTES_PER_YEAR, 6) / 1440.0  # whole minutes, exactly


def es_contract(as_of: datetime) -> date:
    """The front quarterly E-mini S&P 500 contract's expiry after ``as_of``: the third Friday of the next March, June,
    September or December (CME; the roll a week before is not modelled).

    >>> es_contract(datetime(2026, 10, 23, 16)), es_contract(datetime(2026, 12, 18, 16))
    (datetime.date(2026, 12, 18), datetime.date(2027, 3, 19))
    """
    year, month = as_of.year, as_of.month
    while True:
        if month % 3 == 0:
            first = date(year, month, 1)
            third_friday = first + timedelta(days=(4 - first.weekday()) % 7 + 14)
            if third_friday > as_of.date():
                return third_friday
        month += 1
        if month > 12:
            year, month = year + 1, 1


def es_carry(book: Book) -> tuple[float, date] | None:
    """``(F / S, expiry)`` of the front ES contract, from the published forward of the listed expiry nearest it; None
    without a chain (textbook)."""
    if book.context is None:
        return None
    expiry = es_contract(book.as_of)
    row = book.context.expiry_row(expiry)
    return float(row["forward"]) / book.context.spot, expiry


def make_book(position: Position, inputs: Inputs, shifts: whatif.Shifts) -> Book:
    """The book of a position under the shifts (pure: no widgets)."""
    as_of = valuation_time(inputs)
    front = _front(as_of, strat.front_expiry(position.strategy, position.market))
    moved, market = whatif.apply_position(position.strategy, position.market, shifts, as_of=as_of)
    return Book(position, moved, market, shifts, as_of, front, inputs.context is not None, inputs.context)


# --- the tiles ------------------------------------------------------------------------------------------------------


def desk_grid(book: Book, mode: str = "vp", sessions: int = 0) -> np.ndarray:
    """Today's desk grid: P&L ($) of 9 spot moves (columns) by 7 vol shocks (rows) after ``sessions`` trading
    sessions, against today's value, in vol ``mode`` (``vp``, ``relative`` or ``1M-scaled``)."""
    days = clock.year_fraction(book.as_of, whatif.scenario_time(book.as_of, sessions)) * 365.0 if sessions else 0.0
    shocks = np.asarray(DESK_VOL[mode]) / 100.0
    cube = scenarios.scenario_cube(
        book.position.strategy,
        book.position.market,
        np.asarray(DESK_SPOT) / 100.0,
        shocks,
        [days],
        vol_mode=GRID_VOL_MODES[mode][1],
        multiplier=book.units,
    )
    return cube[0]


def worst_stress(book: Book) -> dict:
    """The worst named stress at the what-if state: every historical and hypothetical scenario, instantaneous, one-
    month vol shocks damped by tenor."""
    rows = scenarios.run_stress_tests(book.strategy, book.market, ALL_STRESSES, False, book.units)
    return min(rows, key=lambda row: row["P&L"])


@dataclass(frozen=True)
class Tiles:
    """The six headline numbers and what their delta slots and help say (all $ for the whole position)."""

    dollar_delta: float
    delta: float
    gamma_pnl: float
    dollar_gamma: float
    vega: float
    vega_1m: float
    vega_relative: float
    theta: float
    next_session: float
    one_day: float
    expiring: bool
    breakeven_move: float | None
    implied_move: float
    worst_grid: float
    worst_grid_at: str
    worst_stress: float
    worst_stress_name: str


def tiles(book: Book) -> Tiles:
    """The tiles' numbers (pure: tested without the app)."""
    units = book.units
    pr = strat.position_risk(book.strategy, book.market, multiplier=units)
    theta, gamma_pnl = pr.theta, pr.dollar_gamma_1pct
    # The move a day of theta pays for: ½Γ(m·S)² = |Θ| when 1% × √(|Θ| / |Γ P&L 1%|) = m, defined when the two have
    # opposite signs (short gamma earns theta, long gamma pays it).
    be = 0.01 * math.sqrt(abs(theta) / abs(gamma_pnl)) if theta * gamma_pnl < 0 else None
    tau = strat.front_expiry(book.strategy, book.market)
    implied = whatif.leg_vol(book.atm_vol(tau), tau, book.shifts) / math.sqrt(365.0)
    days = next_session_days(book.scenario_time())
    grid = desk_grid(book)
    i, j = np.unravel_index(int(np.argmin(grid)), grid.shape)
    stress = worst_stress(book)
    return Tiles(
        dollar_delta=pr.dollar_delta,
        delta=pr.delta,
        gamma_pnl=gamma_pnl,
        dollar_gamma=pr.gamma * pr.spot**2 / 100.0,
        vega=pr.vega,
        vega_1m=scenarios.weighted_vega(book.strategy, book.market, scheme="sqrt_time", multiplier=units),
        vega_relative=scenarios.weighted_vega(book.strategy, book.market, multiplier=units),
        theta=theta,
        next_session=strat.time_decay(book.strategy, book.market, days=days, multiplier=units),
        one_day=strat.time_decay(book.strategy, book.market, multiplier=units),
        expiring=tau <= strat.ONE_DAY,
        breakeven_move=be,
        implied_move=implied,
        worst_grid=float(grid[i, j]),
        worst_grid_at=f"{shock_label(DESK_SPOT[j] / 100)} · {vol_label(DESK_VOL['vp'][i], 'vp')}",
        worst_stress=float(stress["P&L"]),
        worst_stress_name=str(stress["Scenario"]),
    )


def _card(label: str, value: str, help: str, delta: str | None = None) -> Card:
    options: dict[str, Any] = {"help": md(help)}
    if delta:
        options |= {"delta": md(delta), "delta_color": "off", "delta_arrow": "off"}
    return label, value, options


def _tile_cards(book: Book, t: Tiles) -> list[Card]:
    carry = es_carry(book)
    if carry is not None:
        factor, expiry = carry
        units_line = futures_equiv(t.delta, carry_factor=factor)
        units_help = (
            f" In index futures: {units_line}, {futures_equiv(t.delta, carry_factor=factor, contract='MES')}, at "
            f"the {date_text(expiry)} contract's forward over spot, {factor:.5f} ($50 and $5 a point)."
        )
    else:
        units_line = f"{signed(t.delta, 0 if abs(t.delta) >= 100 else 2)} sh"
        units_help = ""
    be = MISSING if t.breakeven_move is None else f"{t.breakeven_move:.2%}"
    ns = whatif.scenario_time(book.scenario_time(), 1)
    return [
        _card(
            "$Δ",
            dollars(t.dollar_delta),
            f"Δ × S: {money(t.dollar_delta, 2, True)}. Δ {signed(t.delta, 2)} index units (shares), sticky strike: "
            f"leg vols fixed by strike.{units_help}",
            units_line,
        ),
        _card(
            "Γ P&L 1%",
            dollars(t.gamma_pnl),
            f"½Γ(0.01·S)², the convexity P&L of a 1% move either way: {money(t.gamma_pnl, 2, True)}. The change in $Δ "
            f"per 1% move, ΓS²/100, is {money(t.dollar_gamma, 2, True)}.",
        ),
        _card(
            "Vega ($/vp)",
            dollars(t.vega),
            f"$ per vol point on every leg: {money(t.vega, 2, True)}. 1M: each expiry weighted √(1M/τ), the P&L of a "
            f"one-point rise in one-month vol under the square-root-of-time rule. Relative-weighted (1% of one-month "
            f"vol, damped by tenor): {money(t.vega_relative, 2, True)}.",
            f"1M {dollars(t.vega_1m)}",
        ),
        _card(
            "Θ ($/day)",
            dollars(t.theta),
            f"∂V/∂t ÷ 365 at fixed vols: {money(t.theta, 2, True)} a calendar day. BE: the breakeven daily move that "
            f"½Γ pays for, 1% × √(|Θ| ÷ |Γ P&L 1%|) = {be}, against the implied daily move σ ATMF ÷ √365 = "
            f"{t.implied_move:.2%} ({MISSING} when Θ and Γ have one sign). To the next session ({date_text(ns.date())} "
            f"{ns:%H:%M}) by full revaluation: {money(t.next_session, 2, True)}; over one calendar day: "
            f"{money(t.one_day, 2, True)}.",
            # "BE", not the spec's "BE move": with it the six tiles need 1,162 px and wrap 3 + 3 in the 1,038 px main
            # column of a 1,440 px window with the sidebar open; with "BE" they need 964 px (components._card_width).
            # Where Θ misstates a day's decay by more than 10% (within days of expiry, Milestone 1), the slot gives
            # the day's decay by repricing instead.
            f"{decay_label(t.expiring)} {dollars(t.one_day)}"
            if decay_leads(t.theta, t.one_day, t.expiring)
            else f"BE {be} vs {t.implied_move:.2%}",
        ),
        _card(
            "Worst grid",
            dollars(t.worst_grid),
            f"The worst cell of today's desk grid (9 spot moves × 7 vol shocks in vp, now): "
            f"{money(t.worst_grid, 2, True)}. Today's market, whatever the what-if: the grid is where a scenario is "
            "picked (Scenarios report).",
            t.worst_grid_at,
        ),
        _card(
            "Worst stress",
            dollars(t.worst_stress),
            f"The worst of {len(ALL_STRESSES)} named stresses at the what-if state, instantaneous, one-month vol "
            f"shocks damped by tenor: {t.worst_stress_name}, {money(t.worst_stress, 2, True)} (Stress report).",
            t.worst_stress_name.split(" (")[0],
        ),
    ]


# --- the position line ----------------------------------------------------------------------------------------------


def expiry_text(book: Book, inputs: Inputs) -> str:
    """The legs' expiries: ``20‑Nov‑26 AM (27d 17h)`` for the listed expiry and each typed one by its date (index);
    the time left alone under the textbook preset, whose expiries are days (``365d``)."""
    seen: dict[float, str] = {}
    for leg in book.position.strategy.legs:
        if not leg.is_option:
            continue
        years = book.position.market.expiry if leg.expiry is None else leg.expiry
        key = round(years * 365.0, 6)
        if key in seen:
            continue
        when = _front(book.as_of, years)
        left = time_left(years * clock.MINUTES_PER_YEAR, expiry_day=when.date() == book.as_of.date())
        if not book.index:  # the textbook market counts days, not dates
            seen[key] = left
            continue
        settle = f" {inputs.settlement}" if leg.expiry is None and inputs.settlement else ""
        day = inputs.expiry_date if leg.expiry is None and inputs.expiry_date else when.date()
        seen[key] = f"{date_text(day).replace('-', NB_HYPHEN)}{settle} ({left})"
    return ", ".join(seen[k] for k in sorted(seen))


def position_line(book: Book, inputs: Inputs, max_loss: float) -> str:
    """The context line (UI spec 3.3 item 1): the position, its legs' expiries, its size and its maximum loss at the
    front expiry; with a scenario, the what-if P&L against today."""
    loss = "Unlimited" if math.isinf(max_loss) else dollars(-max_loss) if max_loss > 0 else "none"
    parts = [book.position.describe(), expiry_text(book, inputs), md(f"max loss {loss}")]
    if book.shifts.active:
        today = float(strat.position_value(book.position.strategy, book.position.market))
        now = float(strat.position_value(book.strategy, book.market))
        parts.append(md(f"what-if P&L {dollars(book.units * (now - today))}"))
    return " · ".join(part for part in parts if part)


# --- UI-10: a scenario cell into the what-if strip ------------------------------------------------------------------


@dataclass(frozen=True)
class GridAxes:
    """What a grid's cells mean, for :func:`on_scenario_select`: the spot move (%) of each column, the vol shock and
    the T+N of each row, the vol mode (the strip's name) and the column labels of the table view."""

    spots: tuple[float, ...]
    rows: tuple[tuple[float, int], ...]
    mode: str = "absolute"
    columns: tuple[str, ...] = ()


def on_scenario_select(event: Mapping | Any, axes: GridAxes) -> None:
    """Copy the selected cell's spot move, vol shock and date into the what-if strip (UI-10).

    Two payloads: a point of the heatmap's scatter overlay (``selection.points``, by its ``point_index``, which runs
    row by row) and a cell of the table view (``selection.cells``, as ``(row, column name)``). An empty selection
    (the cell clicked again) changes nothing.
    """
    selection = (event or {}).get("selection", {}) if isinstance(event, Mapping) else getattr(event, "selection", {})
    points = list(selection.get("points", []) or [])
    cells = list(selection.get("cells", []) or [])
    if points:
        index = points[0].get("point_index", points[0].get("point_number"))
        if index is None:
            return
        row, column = divmod(int(index), len(axes.spots))
    elif cells:
        row, name = cells[0]
        if name not in axes.columns:
            return  # the row label's column
        column = axes.columns.index(name)
    else:
        return
    if not (0 <= row < len(axes.rows) and 0 <= column < len(axes.spots)):
        return
    shock, sessions = axes.rows[row]
    state = st.session_state
    state["wvmode"] = axes.mode
    state["ws"] = float(axes.spots[column])
    state["wv"] = float(shock)
    state["wd"] = int(sessions)


def _current_cell(book: Book, axes: GridAxes) -> tuple[int, int] | None:
    """The cell the strip holds, if the grid has it (the rate shift must be zero: the grid has no rate)."""
    s = book.shifts
    if not s.active or s.rate or (s.mode != axes.mode and s.vol):
        return None
    for i, (shock, sessions) in enumerate(axes.rows):
        for j, spot in enumerate(axes.spots):
            if math.isclose(spot, s.spot) and math.isclose(shock, s.vol, abs_tol=1e-9) and sessions == s.sessions:
                return i, j
    return None


# --- the reports ----------------------------------------------------------------------------------------------------


def _controls(key: str) -> Any:
    return st.container(horizontal=True, gap="small", vertical_alignment="bottom", key=key)


def _segmented(where: Any, label: str, options, key: str, help: str, **kwargs: Any) -> Any:
    return where.segmented_control(
        label,
        list(options),
        default=next(iter(options)),
        required=True,
        key=key,
        bind="query-params",
        persist_state="session",
        help=help,
        **kwargs,
    )


def ladder_frame(book: Book, axis: str = "%", roll: str = "Today", second: bool = False) -> pd.DataFrame:
    """The ladder (``scenarios.greek_ladder``) at the what-if state, in the table's columns and units."""
    roll_days = {"Today": 0.0, "Next session": next_session_days(book.scenario_time()), "+1w": 7.0}[roll]
    sigma = axis == "σ"
    frame = scenarios.greek_ladder(
        book.strategy,
        book.market,
        LADDER_SIGMA if sigma else None,
        unit="sigma" if sigma else "percent",
        sigma_days=1.0,
        roll_days=roll_days,
        multiplier=book.units,
    )
    moves = []
    for shock, change in zip(frame["shock"], frame["spot_change"], strict=True):
        text = f"{signed(shock, 1 if shock % 1 else 0)}σ · {signed(100 * change, 2)}%" if sigma else shock_label(change)
        moves.append("0 · now" if not shock else text)
    table_frame = pd.DataFrame(
        {
            "Spot": frame["spot"],
            "Move": moves,
            "P&L": frame["pnl"],
            "$Δ": frame["dollar_delta"],
            "Γ P&L 1%": frame["dollar_gamma_1pct"],
            "Vega": frame["vega"],
            "Θ": frame["theta"],
        }
    )
    if second:
        table_frame = table_frame.assign(Vanna=frame["vanna"], Volga=frame["volga"], Charm=frame["charm"])
    return table_frame


LADDER_KINDS = {
    "Spot": "level",
    "Move": "text",
    "P&L": "pnl",
    "$Δ": "dollar_delta",
    "Γ P&L 1%": "gamma_pnl",
    "Vega": "vega",
    "Θ": "theta",
    "Vanna": "greek",
    "Volga": "greek",
    "Charm": "greek",
}
LADDER_HELP = {
    "Spot": "Spot after the move.",
    "Move": "The spot move: percent, or daily standard deviations σ·√(1/365) at the leg vol. 0 · now: no move.",
    "P&L": "Full revaluation against the value now (at the what-if state), after the roll.",
    "$Δ": "Δ × S at the moved spot ($).",
    "Γ P&L 1%": "½Γ(0.01·S)² at the moved spot ($).",
    "Vega": "$ per vol point on every leg.",
    "Θ": "$ per calendar day at fixed vols.",
    "Vanna": "Change in Δ (index units) per vol point.",
    "Volga": "Change in vega ($/vp) per vol point.",
    "Charm": "Change in Δ (index units) per calendar day.",
}


#: Units of the whole-dollar kinds, for :func:`small_dollars`.
_DOLLAR_UNITS = {"dollar_delta": "$", "gamma_pnl": "$", "vega": "$/vp", "theta": "$/day"}


def small_dollars(frame: pd.DataFrame, kinds: Mapping[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """``(kinds, labels)`` for :func:`.components.table` with cents in a whole-dollar column (the registry's $Δ, Γ P&L,
    vega and Θ) whose amounts all stay below $10 and are not all zero, as the tiles write them (:func:`dollars`): on
    the textbook market's single contract the Γ P&L 1% column (−$0.05) and Θ (−$0.14) read 0 to the dollar. Such a
    column takes the P&L kind (cents below $1,000) under a label that keeps its unit.

    >>> frame = pd.DataFrame({"Θ": [-0.14, -0.2], "Vega": [-12.0, 3.0]})
    >>> small_dollars(frame, {"Θ": "theta", "Vega": "vega"})
    ({'Θ': 'pnl', 'Vega': 'vega'}, {'Θ': 'Θ ($/day)'})
    """
    kinds, labels = dict(kinds), {}
    for name, kind in kinds.items():
        if kind not in _DOLLAR_UNITS or name not in frame:
            continue
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
        values = values[np.isfinite(values)]
        if values.size and 0 < round(float(np.max(np.abs(values))), 2) < 10:  # an empty bucket stays ``0``
            kinds[name], labels[name] = "pnl", f"{name} ({_DOLLAR_UNITS[kind]})"
    return kinds, labels


def _ladder(book: Book) -> None:
    row = _controls("ladder_controls")
    axis = _segmented(row, "Steps", ("%", "σ"), "lad_axis", "Spot moves in percent, or in daily σ at the leg vol.")
    roll = _segmented(
        row, "Roll", LADDER_ROLLS, "lad_roll", "Time that passes before the moves: none, to the next session, a week."
    )
    order = _segmented(row, "Greeks", ("1st", "+2nd"), "lad_order", "+2nd adds vanna, volga and charm.")
    frame = ladder_frame(book, axis, roll, order == "+2nd")
    left, right = st.columns([1.45, 1], gap="medium")
    with left:
        kinds, labels = small_dollars(frame, {name: LADDER_KINDS[name] for name in frame.columns})
        table(frame, kinds, LADDER_HELP, labels=labels, key="tbl_ladder")
    with right:
        metric = _segmented(
            st, "Chart", LADDER_METRICS, "lad_metric", "The ladder column the chart draws.", width="stretch"
        )
        column, title = LADDER_METRICS[metric]
        source = {"pnl": "P&L", "dollar_delta": "$Δ", "dollar_gamma_1pct": "Γ P&L 1%", "vega": "Vega", "theta": "Θ"}
        charts.show(
            charts_risk.ladder_chart(
                frame["Spot"], frame[source[column]], title, spot=book.market.spot, labels=frame["Move"]
            ),
            key="ladder_chart",
            alt=f"The position's {metric} at each ladder spot",
        )


#: Steps (sessions) of the Spot × Date grid's rows before the last week: the first that leaves at most 5 rows.
_DATE_STEPS = (5, 10, 20, 25, 50, 100, 250, 500, 1000, 2500)


def date_rows(last: int) -> list[int]:
    """The sessions of the Spot × Date grid's rows for a front expiry ``last`` sessions away: today, at most five
    rows a round number of sessions apart, then each of the last five sessions, at most 11 rows whatever the
    horizon (a 365-day textbook option drew 55 rows of 5 sessions, their values overprinted).

    >>> date_rows(20), date_rows(248), date_rows(3)
    ([0, 5, 10, 15, 16, 17, 18, 19, 20], [0, 50, 100, 150, 200, 244, 245, 246, 247, 248], [0, 1, 2, 3])
    """
    head = max(last - 4, 1)
    step = next((s for s in _DATE_STEPS if (head - 1) // s <= 5), _DATE_STEPS[-1])
    return sorted({0, *range(step, head, step), *range(head, last + 1)})


def _grid_axes(book: Book, axes: str, mode: str, sessions: int) -> tuple[GridAxes, list[str], list[float]]:
    """A grid's axes: the cells' meaning, the row labels and the rows' calendar days (Spot × Date)."""
    strip_mode = GRID_VOL_MODES[mode][0]
    if axes == GRID_AXES[0]:
        rows = tuple((float(v), sessions) for v in DESK_VOL[mode])
        labels = [vol_label(v, mode) for v in DESK_VOL[mode]]
        return GridAxes(DESK_SPOT, rows, strip_mode), labels, []
    last = max(1, whatif.sessions_to(book.as_of.date(), book.front.date()))
    picks = date_rows(last)
    days = whatif.session_dates(book.as_of.date(), last)
    same_year = days[last].year == book.as_of.year  # else the labels keep their year: Fri 30-Oct-26, Wed 27-Oct-27
    labels = []
    for n in picks:
        day = days[n]
        label = session_label(n, day)
        label = label.replace(f"{NB_HYPHEN}{day:%y}", "") if same_year else label
        labels.append(label + (" exp" if n == last and book.front.date() == day else ""))
    elapsed = [clock.year_fraction(book.as_of, whatif.scenario_time(book.as_of, n)) * 365.0 for n in picks]
    return GridAxes(DESK_SPOT, tuple((0.0, n) for n in picks), "absolute"), labels, elapsed


def _scenarios(book: Book) -> None:
    row = _controls("scenario_controls")
    grid = _segmented(
        row,
        "Grid",
        GRIDS,
        "sc_grid",
        "Desk 9×7: spot ±10% by vol shocks. Portfolio margin: FINRA 4210(g)'s ten points, an illustrative requirement.",
    )
    position, market = book.position.strategy, book.position.market
    spot = market.spot
    if grid == GRIDS[1]:
        product = "high_cap_index" if book.index else "equity"
        result = margin.portfolio_margin(
            position,
            market,
            product_class=product,
            contracts=book.position.contracts,
            multiplier=book.position.multiplier,
        )
        labels = [spot_label(spot, x) for x in result.points]
        charts.show(
            charts_risk.scenario_heatmap(
                labels,
                ["Vol unchanged"],
                result.pnl[None, :],
                f"Spot · cells P&L ({grid_scale(float(np.max(np.abs(result.pnl))))[1]})",
                "",
                text=grid_cell_text,
                height=170,
            ),
            key="sc_heatmap_margin",
            alt="The position's P&L ($) at the ten valuation points of FINRA 4210(g)",
        )
        low, high = margin.PRODUCT_CLASSES[product]
        at_worst = f"{dollars(result.worst)} at {shock_label(result.worst_point)}"
        loss = (
            f"the largest loss over the {margin.VALUATION_POINTS} points, {at_worst},"
            if result.worst < 0
            else f"no loss at any of the {margin.VALUATION_POINTS} points (the least gain {at_worst}),"
        )
        st.markdown(
            md(
                f"Requirement (illustrative, not a broker's): {money(result.requirement, 0)}, the greater of {loss} "
                f"and the minimum of $0.375 × {book.position.multiplier:g} a contract, {money(result.minimum, 2)} "
                f"(FINRA 4210(g)(7)(B))."
            )
        )
        st.caption(
            f"{margin.CLASS_NAMES[product].capitalize()}: {shock_label(low)} to {shock_label(high)} (4210(g)(2)(F)), "
            "vols unchanged, no group offsets."
        )
        return
    axes_name = _segmented(row, "Axes", GRID_AXES, "sc_axes", "Spot by vol shock at a date, or spot by date.")
    mode, sessions = "vp", 0
    if axes_name == GRID_AXES[0]:
        mode = _segmented(
            row,
            "Vol",
            tuple(DESK_VOL),
            "sc_volmode",
            "vp: vol points added to every leg. Relative: percent of each leg's vol. 1M-scaled: percent of one-month "
            "vol, damped by tenor.",
        )
        last = max(1, whatif.sessions_to(book.as_of.date(), book.front.date()))
        if not 0 <= st.session_state.get("sc_date", 0) <= last:
            st.session_state["sc_date"] = 0  # a nearer front expiry than the date set
        at = int(st.session_state.get("sc_date", 0))
        sessions = row.slider(
            session_label(at, whatif.session_dates(book.as_of.date(), last)[at]),
            0,
            last,
            0,
            format="T+%d",
            key="sc_date",
            persist_state="session",
            width=230,
            help="The grid's date in trading sessions after the valuation date, as the strip counts them.",
        )
    view = _segmented(row, "View", ("Chart", "Table"), "sc_view", "The grid as a heatmap or as a table.")
    axes, row_labels, elapsed = _grid_axes(book, axes_name, mode, int(sessions))
    spots = np.asarray(DESK_SPOT) / 100.0
    if axes_name == GRID_AXES[0]:
        z = desk_grid(book, mode, int(sessions))
        y_title = "Vol shock" + (" (vp)" if mode == "vp" else f" ({mode})")
    else:
        z = scenarios.pnl_grid(position, market, spots, days=np.asarray(elapsed), multiplier=book.units)
        y_title = "Date"
    x_labels = [spot_label(spot, x) for x in spots]
    current = _current_cell(book, axes)
    if view == "Chart":
        key = f"sc_heatmap_{axes_name == GRID_AXES[0]}"
        custom = np.array([[[x, shock, n] for x in DESK_SPOT] for shock, n in axes.rows], dtype=float)
        charts.show(
            charts_risk.scenario_heatmap(
                x_labels,
                row_labels,
                z,
                f"Spot · cells P&L ({grid_scale(float(np.max(np.abs(z))))[1]})",
                y_title,
                text=grid_cell_text,
                customdata=custom,
                current=current,
            ),
            key=key,
            alt=f"Heatmap of the position's P&L ($) by spot and {y_title.lower()}; click a cell to apply it",
            on_select=lambda: on_scenario_select(st.session_state.get(key), axes),
            selection_mode="points",
        )
    else:
        columns = [shock_label(x) for x in spots]
        frame = pd.DataFrame(z, columns=columns)
        frame.insert(0, y_title, row_labels)
        axes = dataclasses.replace(axes, columns=tuple(columns))
        kinds = {y_title: "text"} | dict.fromkeys(columns, "pnl")
        helps = {y_title: "The row's vol shock or date."} | {
            c: f"P&L ($) at spot {level_label(spot * (1 + x))} ({c})." for c, x in zip(columns, spots, strict=True)
        }
        table(
            frame,
            kinds,
            helps,
            key="tbl_scenarios",
            selection_mode="single-cell",
            on_select=lambda: on_scenario_select(st.session_state.get("tbl_scenarios"), axes),
        )
    st.caption("Today's position, full revaluation against its value now. Click a cell to set it in the what-if strip.")


def stress_frame(
    book: Book, groups, elapsed: bool, damp: bool, custom: scenarios.StressScenario | None
) -> pd.DataFrame:
    """The stress table at the what-if state, sorted worst first."""
    chosen = []
    if "Historical" in groups:
        chosen += list(scenarios.HISTORICAL_SCENARIOS)
    if "Hypothetical" in groups:
        chosen += list(scenarios.HYPOTHETICAL_SCENARIOS)
    decay = scenarios.DEFAULT_VOL_TENOR_DECAY if damp else 0.0
    rows = scenarios.run_stress_tests(book.strategy, book.market, chosen, elapsed, book.units, decay)
    if custom is not None:
        rows += scenarios.run_stress_tests(book.strategy, book.market, [custom], True, book.units, decay)
        chosen.append(custom)
    one_month = book.atm_vol(strat.VOL_SHOCK_REFERENCE_TENOR, book.market)
    records = []
    for sc, row in zip(chosen, rows, strict=True):
        source = (
            f"{sc.vol_index} {sc.vol_start:g}→{sc.vol_end:g}"
            if sc.vol_index
            else ("custom" if sc is custom else "hypothetical")
        )
        records.append(
            {
                "Scenario": sc.name,
                "Source": source,
                "Spot": 100 * sc.spot_change,
                "1M vol": 100 * (sc.vol_multiplier - 1.0) * one_month,
                "Rate": 1e4 * sc.rate_change,
                "Days": row["Days elapsed"],
                "P&L": row["P&L"],
            }
        )
    frame = pd.DataFrame(records, columns=["Scenario", "Source", "Spot", "1M vol", "Rate", "Days", "P&L"])
    return frame.sort_values("P&L", kind="stable").reset_index(drop=True)


STRESS_KINDS = {
    "Scenario": "text",
    "Source": "text",
    "Spot": "share",
    "1M vol": "vol_change",
    "Rate": "bp",
    "Days": "days",
    "P&L": "pnl",
}


def _stress(book: Book) -> None:
    row = _controls("stress_controls")
    groups = row.pills(
        "Groups",
        STRESS_GROUPS,
        selection_mode="multi",
        default=list(STRESS_GROUPS[:2]),
        key="stress_groups",
        persist_state="session",
        help="Historical analogues (S&P 500 move, VIX ratio, Fed funds change), hypothetical shocks, your own.",
    )
    elapsed = row.toggle(
        "Episode length",
        value=False,
        key="stress_elapsed",
        persist_state="session",
        help="Let the episode's calendar days pass too (options decay or expire).",
    )
    custom = None
    if "Custom" in (groups or []):
        with row.popover("Custom shock"):
            spot = st.number_input("Spot (%)", -99.0, 500.0, -15.0, 1.0, key="custom_spot", persist_state="session")
            mult = st.number_input("IV × (multiplier)", 0.1, 10.0, 1.5, 0.1, key="custom_vol", persist_state="session")
            rate = st.number_input("Rate (bp)", -1000.0, 1000.0, 0.0, 25.0, key="custom_rate", persist_state="session")
            days = st.number_input("Days", 0, 3650, 0, 1, key="custom_days", persist_state="session")
        custom = scenarios.StressScenario("Custom", spot / 100, mult, rate / 1e4, int(days), "User-defined shock.")
    if not groups:
        st.info("Pick at least one group of scenarios.")
        return
    damp = st.session_state.get("stress_damp", True)
    frame = stress_frame(book, groups, elapsed, damp, custom)
    charts.show(
        charts_risk.stress_bars(frame["Scenario"], frame["P&L"]),
        key="stress_chart",
        alt="The position's P&L ($) in each stress scenario, worst first, as horizontal bars",
    )
    table(
        frame,
        STRESS_KINDS,
        {
            "Scenario": "The named stress.",
            "Source": "The vol index and its two closes the vol shock is calibrated on; hypothetical or custom.",
            "Spot": "Spot move (%).",
            "1M vol": "The shock to one-month ATM vol in vol points (the scenario's vol multiplier on today's 1M vol).",
            "Rate": "Rate change (bp).",
            "Days": "Calendar days elapsed (0 unless Episode length is on).",
            "P&L": "Full revaluation at the what-if state ($).",
        },
        labels={"Spot": "Spot", "1M vol": "1M vol", "Rate": "Rate"},
        key="tbl_stress",
    )
    st.toggle(
        f"IV shocks damped by tenor, (τ/1M)^{MINUS}{scenarios.DEFAULT_VOL_TENOR_DECAY:g}",
        value=True,
        key="stress_damp",
        persist_state="session",
        help="Historical and Fed shocks move one-month (VIX-like) vol; longer legs move less. Hypothetical shocks "
        "apply to every leg as written.",
    )


def _var_card(label: str, value: float | str, help_text: str, delta: str | None = None) -> Card:
    return _card(label, value if isinstance(value, str) else dollars(value), help_text, delta)


def _var(book: Book) -> None:
    row = _controls("var_controls")
    model = _segmented(
        row,
        "Model",
        VAR_MODELS,
        "var_model",
        "The Monte Carlo model of spot returns: Normal, Student-t, GARCH(1,1) with Normal or Student-t shocks.",
    )
    with row.popover("Vol dynamics"):
        moves = st.toggle(
            "Simulate implied-vol moves",
            value=True,
            key="var_volmoves",
            persist_state="session",
            help="Log implied vol moves with the vol of vol, correlated with spot. Off: implied vol frozen.",
        )
        vov = st.number_input(
            "Vol of vol (% a year)", 0.0, 500.0, 100 * INDEX_VOL_OF_VOL, 10.0, key="var_vov", persist_state="session"
        )
        corr = st.slider(
            "Spot–vol correlation", -1.0, 1.0, INDEX_SPOT_VOL_CORRELATION, 0.05, key="var_corr", persist_state="session"
        )
    distribution, garch_t = VAR_MODELS[model]
    with row.popover("Advanced"):
        horizon = st.number_input("Horizon (trading days)", 1, 60, 1, 1, key="var_horizon", persist_state="session")
        paths = st.select_slider(
            "Simulations", PATH_CHOICES, value=100_000, format_func=lambda n: f"{n:,}", key="var_paths",
            persist_state="session",
        )  # fmt: skip
        seed = st.number_input("Random seed", 0, 2**32 - 1, 7, 1, key="var_seed", persist_state="session")
        dof = st.slider("Student-t ν", 3, 30, 5, key="var_dof", persist_state="session", help="Lower: fatter tails.")
        alpha = st.number_input(
            "GARCH α", 0.0, 0.5, 0.09, 0.01, format="%.3f", key="garch_alpha", persist_state="session"
        )
        beta = st.number_input(
            "GARCH β", 0.0, 0.999, 0.90, 0.01, format="%.3f", key="garch_beta", persist_state="session"
        )
    try:
        config = risk.VaRConfig(
            confidence=VAR_CONFIDENCE,
            horizon_days=int(horizon),
            n_paths=int(paths),
            distribution=distribution,
            dof=float(dof),
            garch_alpha=float(alpha),
            garch_beta=float(beta),
            garch_t=garch_t,
            vol_of_vol=float(vov) / 100 if moves else 0.0,
            spot_vol_correlation=float(corr),
            seed=int(seed),
        )
    except ValueError as exc:
        st.error(str(exc))
        return
    # The spot-only run reprices the same spot paths with implied vol frozen: the simulation draws every spot shock
    # before any vol shock, so with one seed the two P&Ls differ path by path by the vol moves alone. Its correlation
    # is fixed, so a change of the vol dynamics finds it in the cache.
    frozen = dataclasses.replace(config, vol_of_vol=0.0, spot_vol_correlation=INDEX_SPOT_VOL_CORRELATION)
    with_vol = config.vol_of_vol > 0
    with st.spinner("Simulating…"):
        spot_only = cache.value_at_risk(book.strategy, book.market, frozen, book.units)
        result = cache.value_at_risk(book.strategy, book.market, config, book.units) if with_vol else spot_only
    h = f"{int(horizon)}D"
    var, es = risk.tail_measures(result.pnl, VAR_CONFIDENCE)[0], risk.tail_measures(result.pnl, ES_CONFIDENCE)[1]
    spot_var, spot_es = spot_only.var, risk.tail_measures(spot_only.pnl, ES_CONFIDENCE)[1]
    greeks = strat.position_greeks(book.strategy, book.market)
    dg = result.delta_gamma
    dg_es = None
    if dg is not None:
        dg_es = risk.delta_gamma_var(
            *(book.units * float(greeks[name]) for name in ("delta", "gamma", "theta")),
            book.market.spot,
            book.market.volatility,
            int(horizon),
            ES_CONFIDENCE,
        ).expected_shortfall
    basis = f"{model} Monte Carlo, {int(paths):,} paths, seed {int(seed)}"
    vol_text = "spot and implied vol moving" if with_vol else "implied vol frozen"
    cards = [
        _var_card(
            f"VaR 99% {h} · MC",
            -var,
            f"The 1% worst P&L over {h} by full revaluation, {vol_text} ({basis}): {money(-var, 2, True)}; 95% "
            f"simulation interval {money(-result.var_ci95[0], 0, True)} to {money(-result.var_ci95[1], 0, True)}.",
            f"{dollars(shown(-var) - shown(-spot_var))} vs spot only" if with_vol else None,
        ),
        _var_card(
            f"ES 97.5% {h} · MC",
            -es,
            f"The mean P&L of the worst 2.5% (FRTB's level), {vol_text}: {money(-es, 2, True)}.",
            f"{dollars(shown(-es) - shown(-spot_es))} vs spot only" if with_vol else None,
        ),
    ]
    if with_vol:
        cards += [
            _var_card(
                f"Spot-only VaR 99% {h}",
                -spot_var,
                "The same spot paths with implied vol frozen, which leaves out the vega risk: "
                f"{money(-spot_var, 2, True)}.",
            ),
            _var_card(f"Spot-only ES 97.5% {h}", -spot_es, f"Implied vol frozen: {money(-spot_es, 2, True)}."),
        ]
    # The comparator's like: the delta-gamma-theta VaR moves spot alone, Normally, at the leg vol, so its gap is taken
    # to the spot-only run (the full VaR when implied vol is frozen), which differs from it by the full revaluation
    # alone (and by fatter tails under a Student-t or GARCH model). Against the spot-plus-vol VaR the gap would mix
    # in the vega risk: on the default call spread at the pinned valuation time (ASOF in tests/_apptest.py), +$248
    # against the full VaR (−$1,729, implied vol moving) but +$56 against the spot-only one (−$1,922; DGT −$1,978).
    # The amounts shown are subtracted, so that the tile adds up on screen.
    gap = None if dg is None else shown(-spot_var) - shown(-dg.var)
    comparator = f"Spot-only VaR {MINUS} DGT VaR" if with_vol else f"VaR {MINUS} DGT VaR"
    cards += [
        _var_card(
            f"DGT VaR 99% {h}",
            MISSING if dg is None else -dg.var,
            "Exact VaR of Δ·x + ½Γ·x² + Θ·h for a Normal spot move at the leg vol, implied vol frozen (the analytic "
            f"comparator): {MISSING if dg is None else money(-dg.var, 2, True)}; its ES 97.5%: "
            f"{MISSING if dg_es is None else money(-dg_es, 2, True)}.",
        ),
        _var_card(
            "Non-linearity",
            MISSING if gap is None else gap,
            f"{comparator}, both as P&L and both with implied vol frozen: negative when the full revaluation loses "
            "more than the second-order expansion (spot beyond gamma, decay as expiry nears, legs that expire within "
            "the horizon, the model's fatter tails).",
            f"spot only {MINUS} DGT" if with_vol else None,
        ),
    ]
    # Pairs, VaR beside its ES: the full simulation, its spot-only twin (with implied vol moving), then the analytic
    # comparator and the gap. Two groups of three went 2 + 1 and 2 + 1 on a phone.
    pairs = ("var_cards_full", "var_cards_spot", "var_cards_comparator")
    card_row("var_cards", pairs if with_vol else pairs[::2], cards)
    left, right = st.columns([1, 2], gap="medium")
    stats = pd.DataFrame(
        [
            ("Mean P&L", money(result.mean, 0, True)),
            ("Std dev", money(result.std, 0)),
            ("Skewness", MISSING if np.isnan(result.skewness) else number(result.skewness, 2)),
            ("Excess kurtosis", MISSING if np.isnan(result.excess_kurtosis) else number(result.excess_kurtosis, 2)),
            ("Worst path", money(result.worst, 0, True)),
            ("Delta-normal VaR 99%", money(-result.delta_normal_var, 0, True)),
        ],
        columns=["Statistic", "Value"],
    )
    stats["Value"] = stats["Value"].map(_minus)
    left.table(right_aligned(stats, ["Statistic"]).hide(axis="index"), border="horizontal")
    fig = charts.histogram_chart(
        result.pnl,
        f"P&L over {h} ($)",
        [(-var, "VaR 99%", charts.LOSS), (-es, "ES 97.5%", charts.ACCENT)],
        value_text=lambda v: _minus(money(v, 0, True)),
        probability=True,
    )
    # The tail beyond VaR, shaded from the histogram's own left edge (its 0.05% quantile or the ES line): a shape that
    # reached the worst path would stretch the axis.
    fig.update_xaxes(tickformat=",.0f")
    tail_start = min(float(np.percentile(result.pnl, 0.05)), -es)
    fig.add_vrect(x0=tail_start, x1=-var, fillcolor=charts.LOSS, opacity=0.08, line_width=0, layer="below")
    charts.show(
        fig, key="var_hist", alt=f"Histogram of the simulated P&L ($) over {h} with the VaR and ES", container=right
    )
    settled = f" {result.settled_legs} leg(s) settle before the horizon." if result.settled_legs else ""
    st.caption(f"{basis}; vol of vol {100 * config.vol_of_vol:g}%, correlation {signed(corr)}.{settled}")
    # No statistic here comes from a history: the models simulate from the leg vol. Their backtests on the RK-4
    # synthetic history are in the Methodology tab, tagged as such.
    st.caption("Backtests of these models on a synthetic history (SYN HIST): Methodology tab, Risk, Backtesting.")


def vega_frame(book: Book, axis: str = "Δ") -> pd.DataFrame:
    """Vega ($/vp) by expiry bucket and strike bucket at the what-if state, with totals, in display labels."""
    raw = scenarios.vega_buckets(
        book.strategy, book.market, strike_axis="delta" if axis == "Δ" else "moneyness", multiplier=book.units
    )
    rows = {}
    for name in raw.index:
        if name == "total":
            rows[name] = "Total"
        elif name.startswith("up_to_"):
            rows[name] = "≤ " + name.removeprefix("up_to_")
        elif name.startswith("over_"):
            rows[name] = "> " + name.removeprefix("over_")
        else:
            rows[name] = name.replace("_to_", "–").replace("d–", "–")
    columns = {}
    for name in raw.columns:
        if name in DELTA_BUCKET_LABELS:
            columns[name] = DELTA_BUCKET_LABELS[name]
        elif name == "total":
            columns[name] = "Total"
        else:
            columns[name] = name.replace("pct", "%").replace("up_to_", "≤ ").replace("over_", "> ").replace("_to_", "–")
    frame = raw.rename(index=rows, columns=columns)
    frame.index.name = "Expiry"
    return frame.reset_index()


def _vega(book: Book) -> None:
    row = _controls("vega_controls")
    axis = _segmented(row, "Strikes by", ("Δ", "Moneyness"), "vg_axis", "Strike buckets by delta or by K/S.")
    frame = vega_frame(book, axis)
    raw = strat.position_risk(book.strategy, book.market, multiplier=book.units).vega
    card_row(
        "vega_cards",
        ("vega_cards_a", "vega_cards_b", "vega_cards_c"),
        [
            _card("Raw vega", dollars(raw), "$ per vol point on every leg.", "$/vp"),
            _card(
                "1M relative",
                dollars(scenarios.weighted_vega(book.strategy, book.market, multiplier=book.units)),
                "The P&L of a 1% relative rise in one-month vol, damped for longer tenors as the stresses are.",
                "1% of 1M vol",
            ),
            _card(
                "1M √T-weighted",
                dollars(scenarios.weighted_vega(book.strategy, book.market, scheme="sqrt_time", multiplier=book.units)),
                "The P&L of a one-point rise in one-month vol, each expiry moving √(1M/τ) points.",
                "1 vp of 1M vol",
            ),
        ],
    )
    cells = frame.set_index("Expiry")
    shown = cells.loc[[i for i in cells.index if i != "Total" and (cells.loc[i].abs() > 1e-9).any()]]
    shown = shown.drop(columns="Total")
    if not shown.empty:
        charts.show(
            charts_risk.scenario_heatmap(
                list(shown.columns),
                list(shown.index),
                shown.to_numpy(float),
                "Strike bucket",
                "Expiry",
                text=lambda v, bound: signed(v, 0 if bound >= 100 else 2) if abs(v) > 1e-9 else "",
                hover="Vega",
                worst=False,
                height=max(140, 60 + 44 * len(shown)),
            ),
            key="vega_heatmap",
            alt="Vega ($ per vol point) by expiry bucket and strike bucket",
        )
    kinds = {"Expiry": "text"} | dict.fromkeys(frame.columns[1:], "vega")
    helps = {"Expiry": "Expiry bucket by calendar days to expiry (right-closed)."} | {
        c: f"Vega ($/vp) of the legs in the {c} strike bucket." for c in frame.columns[1:]
    }
    helps["Total"] = "Vega ($/vp) of the expiry bucket."
    kinds, labels = small_dollars(frame, kinds)
    table(frame, kinds, helps, labels=labels, key="tbl_vega")


def explain_target(book: Book, target: str) -> dict | None:
    """The move from today to ``target`` as ``attribution`` takes it: spot, the legs' new vols, rate shift and days;
    None for the what-if while no shift is set."""
    position, market = book.position.strategy, book.position.market
    if target == WHAT_IF:
        if not book.shifts.active:
            return None
        vols = [leg.volatility if leg.is_option else market.volatility for leg in book.strategy.legs]
        days = clock.year_fraction(book.as_of, book.scenario_time()) * 365.0
        return {"spot": book.market.spot, "vols": vols, "rate_shift": book.shifts.rate / 1e4, "days": days}
    if target == NEXT_SESSION:
        return {"spot": market.spot, "days": next_session_days(book.as_of)}
    sc = next(s for s in ALL_STRESSES if s.name == target)
    vols = []
    for leg in position.legs:
        sigma = market.volatility if leg.volatility is None else leg.volatility
        tau = market.expiry if leg.expiry is None else leg.expiry
        damp = min(1.0, (max(tau, 1e-8) / strat.VOL_SHOCK_REFERENCE_TENOR) ** -scenarios.DEFAULT_VOL_TENOR_DECAY)
        vols.append(sigma * (1 + (sc.vol_multiplier - 1) * (damp if sc.one_month_vol_shock else 1.0)))
    return {"spot": market.spot * (1 + sc.spot_change), "vols": vols, "rate_shift": sc.rate_change, "days": 0.0}


TAYLOR_NAMES = {
    "delta": "Δ",
    "gamma": "Γ",
    "vega": "Vega",
    "theta": "Θ",
    "rho": "ρ",
    "vanna": "Vanna",
    "volga": "Volga",
    "charm": "Charm",
    "veta": "Veta",
}
SHAPLEY_NAMES = {"spot": "Spot", "vol": "Vol", "time": "Time", "rate": "Rate"}


def _explain(book: Book) -> None:
    row = _controls("explain_controls")
    target = row.selectbox(
        "Explain the move to",
        [WHAT_IF, NEXT_SESSION, *(s.name for s in ALL_STRESSES)],
        key="ex_to",
        persist_state="session",
        width=280,
        help="From today's market to the what-if state, to the next session's close with nothing else moving, or to a "
        "named stress (instantaneous).",
    )
    method = _segmented(row, "Method", ("Taylor", "Shapley"), "ex_method", "Greek waterfall, or an exact split.")
    move = explain_target(book, target)
    if move is None:
        # The what-if with every shift at zero has no move to explain: the report explains the next session instead
        # of opening empty, and says so.
        move = explain_target(book, NEXT_SESSION)
        st.caption("No what-if shift is set: this explains the move to the next session's close, nothing else moving.")
    args = (book.position.strategy, book.position.market)
    taylor = attribution.taylor_explain(*args, multiplier=book.units, **move)
    shapley = attribution.shapley_explain(*args, multiplier=book.units, **move)
    if method == "Taylor":
        names = [TAYLOR_NAMES[k] for k in taylor.terms] + ["Unexplained"]
        values = [*taylor.terms.values(), taylor.unexplained]
    else:
        names = [SHAPLEY_NAMES[k] for k in shapley.terms]
        values = list(shapley.terms.values())
    left, right = st.columns([1, 2], gap="medium")
    with right:
        charts.show(
            charts_risk.waterfall(names, values, taylor.total),
            key="explain_waterfall",
            alt=f"Waterfall of the {method} explain of the P&L ($)",
        )
    with left:
        frame = pd.DataFrame({"Term": [*names, "Full revaluation"], "P&L": [*values, taylor.total]})
        table(
            frame,
            {"Term": "text", "P&L": "pnl"},
            {"Term": "Term of the explain.", "P&L": "Its P&L ($)."},
            widths={"Term": "medium"},  # "Full revaluation" comes last, after the Greek letters the grid sizes from
            key="tbl_explain",
        )
    # attribution's quality: |unexplained| over the P&L, floored at a tenth of the terms' gross (0 is a perfect fit).
    share = MISSING if not math.isfinite(taylor.quality) else f"{taylor.quality:.1%} of the P&L"
    st.caption(
        md(
            f"Taylor leaves {dollars(taylor.unexplained)} unexplained ({share}); Shapley splits "
            f"{dollars(shapley.total)} exactly, spot {dollars(shapley.spot)} against Δ + Γ "
            f"{dollars(taylor.delta + taylor.gamma)}."
        )
    )


REPORT_BODIES: dict[str, Callable[[Book], None]] = {
    "Ladder": _ladder,
    "Scenarios": _scenarios,
    "Stress": _stress,
    "VaR": _var,
    "Vega": _vega,
    "Explain": _explain,
}


def _open_strategy() -> None:
    st.session_state["tab"] = "Strategy"


@st.fragment
def render(position: Position | None, inputs: Inputs) -> None:
    """Draw the tab: the position line, the what-if strip, the six tiles, the report selector and the open report."""
    if position is None:
        st.info("The Strategy tab's legs are invalid or empty, so there is no position to analyse.")
        st.button("Open Strategy", key="risk_open_strategy", on_click=_open_strategy)
        return
    as_of = valuation_time(inputs)
    front_years = strat.front_expiry(position.strategy, position.market)
    line = st.container(key="risk_position").empty()  # the context line, filled once the strip has run
    today = make_book(position, inputs, whatif.Shifts())
    shifts = whatif.strip(
        spot=position.market.spot,
        atm_vol=today.atm_vol(front_years),
        tenor=front_years,
        rate=position.market.rate,
        as_of=as_of,
        horizon=_front(as_of, front_years),
    )
    book = make_book(position, inputs, shifts)
    metrics = strat.analyze(position.strategy, position.market)
    line.markdown(position_line(book, inputs, metrics.max_loss * position.units))
    # Three pairs: one line of six on a laptop with the sidebar open, 2 + 2 + 2 on a phone (two groups of three went
    # 2 + 1 + 2 + 1 there).
    card_row("risk_tiles", ("risk_tiles_delta", "risk_tiles_vol", "risk_tiles_losses"), _tile_cards(book, tiles(book)))
    report = st.pills(
        "Report",
        REPORTS,
        default=REPORTS[0],
        required=True,
        key="report",
        bind="query-params",
        persist_state="session",
        label_visibility="collapsed",
    )
    REPORT_BODIES[report or REPORTS[0]](book)
    vols = "the contract's surface vol" if inputs.context is not None else "the sidebar vol"
    st.caption(
        f"Full revaluation · BSM European legs at {vol(position.market.volatility)} ({vols}) · sticky strike · "
        f"{len(position.strategy.legs)} legs · {position.contracts} × {position.multiplier:g}"
    )
    if inputs.model.kind != "bsm":  # a tree, Monte Carlo or the American engine prices the cards above
        st.caption("Every number on this tab is BSM on European legs, whatever the sidebar pricing model.")
