"""The Strategy tab's legs, without Streamlit: listed expiries and strikes, templates by smile delta, the structure
steppers, and the legs priced against the published surface or the textbook market (UI spec 3.2, roadmap items BK-2,
BK-3, RK-9a and UI-8).

The legs are a table with one row per leg and the columns of :data:`COLUMNS`, the table the blotter's Edit mode
shows. Under the synthetic index a leg holds a listed expiry, a date and a settlement written ``18‑Dec‑26 AM``
(:func:`expiry_text`), which every run converts to years with ``clock.time_to_expiry`` at the valuation time (spec
Appendix A item 14): the legs keep their dates when the valuation time moves. Its vol comes from the published surface
at its strike (``IV src`` ``surface``; ``extrap.`` beyond the expiry's quotes; ``unlisted`` for a date no longer
listed, read from the surface in total variance), unless the row types an override, and its bid and ask from the
chain. Under the textbook market a leg keeps days to expiry and the flat sidebar vol, as in version 2.

:func:`index_template` places a template's strikes the way a desk does (RK-9a): with strikes by delta, the short
strikes are the listed strikes whose own delta, at their own surface vol, is nearest the target
(``PublishedSurface.delta_strike``), and the wings sit a number of index points beyond them, snapped to listed strikes;
a template with legs at the money (a butterfly, a vertical) centres on the listed strike nearest the forward and sets
its other strikes by the wing points.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from itertools import pairwise

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .. import black_scholes as bs
from .. import clock
from .. import strategies as strat
from ..vol_surface import PublishedSurface
from ..vol_surface.published import Expiry, Quote
from .format import NB_HYPHEN

#: The legs table's columns: On (in every number or not), the signed quantity, the type, the listed expiry (index) or
#: the days to expiry (textbook), the strike, an implied vol in % (an override under the index; blank = the surface,
#: or the sidebar vol under the textbook market) and the entry price per share (blank = the mid, or the model value
#: where there is no market).
ON, QTY, TYPE, EXPIRY, DAYS, STRIKE, IV, ENTRY = "On", "Qty", "Type", "Expiry", "Days", "Strike", "IV (%)", "Entry"
COLUMNS = (ON, QTY, TYPE, EXPIRY, DAYS, STRIKE, IV, ENTRY)
NUMERIC = (QTY, DAYS, STRIKE, IV, ENTRY)
#: The option kinds under the index: SPX itself cannot be traded, so the index takes no stock leg (spec 3.2 item 5).
INDEX_KINDS = ("call", "put")
#: Short-strike deltas offered by the builder (spot convention, absolute), and the default (spec 3.2 item 1).
SHORT_DELTAS = (10, 16, 20, 25, 30)
DEFAULT_SHORT_DELTA = 16
#: Wing widths in index points (spec 3.2 item 1: equal on both sides, snapped to listed strikes), and the default.
#: Fixed values rather than multiples of each expiry's local strike interval, so that a link keeps its wing: the
#: synthetic SPX lists every 50 points from about 6% out of the money (``synthetic._INTERVALS``), every 5 near it.
WING_POINTS = (25, 50, 100, 150, 200)
DEFAULT_WING = 100
#: The template expiry the index opens on: the AM monthly nearest this many calendar days (spec 3.2 item 1).
DEFAULT_DTE = 45
#: Expiry labels: day, English month and two-digit year with non-breaking hyphens, then the settlement.
_EXPIRY = re.compile(r"^\s*(\d{1,2})\W(\w{3})\W(\d{2})\s+(AM|PM)\s*$")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


# --- Listed expiries -------------------------------------------------------------------------------------------------


def expiry_text(day: date, settlement: str) -> str:
    """A listed expiry as a leg holds it: ``18‑Dec‑26 AM`` (non-breaking hyphens).

    >>> expiry_text(date(2026, 12, 18), "AM")
    '18‑Dec‑26 AM'
    """
    return f"{day.day:02d}{NB_HYPHEN}{_MONTHS[day.month - 1]}{NB_HYPHEN}{day.year % 100:02d} {settlement}"


def parse_expiry(text: object) -> Expiry | None:
    """The :class:`~options_analyzer.vol_surface.published.Expiry` of a label from :func:`expiry_text` (any hyphen),
    or None when it is not one.

    >>> parse_expiry("18-Dec-26 AM")
    Expiry(day=datetime.date(2026, 12, 18), settlement='AM')
    >>> parse_expiry("Dec 18") is None
    True
    """
    match = _EXPIRY.match(str(text)) if isinstance(text, str) else None
    if match is None or match.group(2).title() not in _MONTHS:
        return None
    try:
        day = date(2000 + int(match.group(3)), _MONTHS.index(match.group(2).title()) + 1, int(match.group(1)))
    except ValueError:
        return None
    return Expiry(day, match.group(4))


def listed(published: PublishedSurface) -> list[Expiry]:
    """The listed expiries in order of settlement (an AM before a PM on one date)."""
    table = published.expiries
    return [Expiry(day, str(style)) for day, style in zip(table["expiry"], table["settlement"], strict=True)]


def default_expiry(published: PublishedSurface) -> Expiry:
    """The AM monthly nearest :data:`DEFAULT_DTE` calendar days: 18‑Dec‑26 at 55.7 days from 23‑Oct‑26 16:00, against
    20‑Nov‑26 at 27.7 (spec 3.2 item 1). Without an AM expiry, the nearest of all."""
    table = published.expiries
    monthly = table[table["settlement"] == "AM"]
    pool = monthly if len(monthly) else table
    i = int(np.argmin(np.abs(pool["maturity"].to_numpy(float) * 365.0 - DEFAULT_DTE)))
    return Expiry(pool["expiry"].iloc[i], str(pool["settlement"].iloc[i]))


def snap(strikes: np.ndarray, k: float) -> float:
    """The listed strike nearest ``k`` (the lower of two equally near)."""
    return float(strikes[int(np.argmin(np.abs(strikes - k)))])


def interval(strikes: np.ndarray, k: float) -> float:
    """The listed strike interval at ``k``: the wider of the two gaps beside the listed strike nearest it, so that a
    move by it lands on a listed strike on either side (5 near the money, 50 in the synthetic SPX's wings)."""
    i = int(np.argmin(np.abs(strikes - k)))
    gaps = [strikes[j + 1] - strikes[j] for j in (i - 1, i) if 0 <= j < len(strikes) - 1]
    return float(max(gaps)) if gaps else 1.0


def _back(published: PublishedSurface, front: Expiry, ratio: float = 2.0) -> Expiry:
    """The later listed expiry of a time spread: of the front's settlement, the one nearest ``ratio`` × its maturity
    (an 18‑Dec AM front, 55.7 days, gets the 19‑Feb AM, 118.7 days)."""
    T = published.maturity(front)
    later = [e for e in listed(published) if e.settlement == front.settlement and published.maturity(e) > T]
    later = later or [e for e in listed(published) if published.maturity(e) > T]
    if not later:
        raise ValueError(f"no expiry is listed after {expiry_text(*front)} for the time spread's back leg")
    return min(later, key=lambda e: abs(published.maturity(e) - ratio * T))


# --- Templates --------------------------------------------------------------------------------------------------------


def _offsets(template: strat.StrategyTemplate) -> list[tuple[strat.Leg, int, float | None]]:
    """Each leg of ``template`` with its strike offset n (strike K + n·w) and its expiry as a multiple of the
    template's (None: the template's expiry; 2.0: twice it)."""
    probe = template.build(1000.0, 1.0, 1.0)
    return [(leg, round(leg.strike - 1000.0) if leg.is_option else 0, leg.expiry) for leg in probe]


def _row(kind: str, quantity: float, strike: float | None, *, expiry: str | None = None, days=None) -> dict:
    return {ON: True, QTY: float(quantity), TYPE: kind, EXPIRY: expiry, DAYS: days, STRIKE: strike, IV: None,
            ENTRY: None}  # fmt: skip


def index_templates() -> list[str]:
    """The templates offered under the index: those with option legs only."""
    return [name for name, t in strat.TEMPLATES.items() if all(leg.is_option for leg, _, _ in _offsets(t))]


def index_template(
    name: str,
    published: PublishedSurface,
    expiry: Expiry,
    *,
    strikes_by: str = "delta",
    short_delta: float = DEFAULT_SHORT_DELTA / 100,
    wing: float = DEFAULT_WING,
    width: float | None = None,
) -> pd.DataFrame:
    """A template's legs on the listed chain (RK-9a).

    Strike offsets n of the template (K + n·w) are placed so:

    * by delta, a template with legs at the money (n = 0: straddles, butterflies, verticals) centres on the listed
      strike nearest the expiry's forward and puts the strike n steps out at n × ``wing`` points from it; one without
      (strangles, condors, risk reversals) puts its n = ±1 strikes at the listed put and call whose own delta at their
      own surface vol is nearest ``short_delta`` (``PublishedSurface.delta_strike``) and its n = ±2 strikes ``wing``
      points beyond them;
    * by width, every strike is the listed one nearest ATMF + n × ``width``.

    Every strike is snapped to the strikes listed on its leg's expiry; a time spread's back leg is on the listed expiry
    nearest twice the front's maturity (:func:`_back`).

    Parameters
    ----------
    name : str
        A template of :data:`options_analyzer.strategies.TEMPLATES` with option legs only (:func:`index_templates`).
    published : PublishedSurface
        The published surface and its chain.
    expiry : Expiry
        The template's (front) expiry.
    strikes_by : {"delta", "width"}
    short_delta : float
        The short strikes' delta, absolute (0.16).
    wing : float
        Wing width in index points.
    width : float, optional
        The strike spacing for ``strikes_by="width"``, in index points.

    Returns
    -------
    pandas.DataFrame
        The legs table (:data:`COLUMNS`).

    Raises
    ------
    ValueError
        If the template holds stock, or its strikes cannot be placed.
    """
    template = strat.TEMPLATES[name]
    shape = _offsets(template)
    if not all(leg.is_option for leg, _, _ in shape):
        raise ValueError(f"{name} holds stock, which the index cannot trade")
    if strikes_by not in ("delta", "width"):
        raise ValueError(f"strikes_by must be 'delta' or 'width', got {strikes_by!r}")
    if strikes_by == "width" and not (width is not None and width > 0):
        raise ValueError("strikes by width need a positive width")
    expiries = {None: expiry, 1.0: expiry}
    if any(multiple not in (None, 1.0) for _, _, multiple in shape):
        expiries[2.0] = _back(published, expiry)
    atm = snap(published.strikes(expiry), published.forward(expiry))
    centred = any(n == 0 for _, n, _ in shape)
    body: dict[int, float] = {}
    if strikes_by == "delta" and not centred:
        body[-1] = published.delta_strike(expiry, -short_delta, "put")
        body[1] = published.delta_strike(expiry, short_delta, "call")
    rows = []
    for leg, n, multiple in shape:
        on = expiries[multiple]
        grid = published.strikes(on)
        if strikes_by == "width":
            target = atm + n * float(width)  # type: ignore[arg-type]
        elif centred:
            target = atm + n * wing
        else:
            sign = 1 if n > 0 else -1
            target = body[sign] + sign * (abs(n) - 1) * wing
        rows.append(_row(leg.kind, leg.quantity, snap(grid, target), expiry=expiry_text(*on)))
    return frame(rows)


def textbook_template(name: str, strike: float, width: float, expiry: float) -> pd.DataFrame:
    """A template's legs in the textbook market: strikes K + n·w, days only where a leg has its own expiry (a time
    spread); blank days, vols and entries follow the sidebar."""
    strategy = strat.build_template(name, strike, width, expiry)
    rows = [
        _row(
            leg.kind,
            leg.quantity,
            float(leg.strike) if leg.is_option else None,
            days=None if leg.expiry is None else round(leg.expiry * 365),
        )
        for leg in strategy.legs
    ]
    return frame(rows)


def frame(rows: Sequence[dict]) -> pd.DataFrame:
    """The legs table of ``rows`` in its normal form: the columns of :data:`COLUMNS`, numbers as floats, ``On`` as a
    boolean and blank text as None."""
    out = pd.DataFrame(list(rows), columns=list(COLUMNS)).reset_index(drop=True)
    for column in (TYPE, EXPIRY):
        out[column] = out[column].astype(object).where(out[column].notna(), None)
    out[ON] = out[ON].astype(object).where(out[ON].notna(), True).astype(bool)
    return out.astype(dict.fromkeys(NUMERIC, float))


def cells(table: pd.DataFrame) -> list[tuple]:
    """The legs' cells row by row (blanks None, numbers floats, ``On`` a boolean), so that equal legs give equal cells
    whatever dtypes the editor returned. Plain Python: for a few rows it is several times faster than :func:`frame`."""
    out = []
    for values in table.reindex(columns=list(COLUMNS)).itertuples(index=False, name=None):
        row = []
        for column, x in zip(COLUMNS, values, strict=True):
            if column == ON:
                row.append(True if _blank(x) else bool(x))
            elif _blank(x):
                row.append(None)
            elif column in NUMERIC:
                row.append(float(x))
            else:
                row.append(str(x))
        out.append(tuple(row))
    return out


def _blank(x) -> bool:
    try:
        return x is None or bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


# --- Structure steppers (UI-8) ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    """Where strikes and expiries may move: the listed chain (index) or, in the textbook market, a fixed step and no
    listed expiries."""

    published: PublishedSurface | None = None
    step: float = 1.0  # the textbook market's strike step

    def strikes(self, label: object) -> np.ndarray | None:
        if self.published is None:
            return None
        key = parse_expiry(label)
        try:
            return None if key is None else self.published.strikes(key)
        except (KeyError, ValueError):
            return None

    def interval(self, label: object, k: float) -> float:
        strikes = self.strikes(label)
        return self.step if strikes is None else interval(strikes, k)

    def snap(self, label: object, k: float) -> float:
        strikes = self.strikes(label)
        return round(k, 8) if strikes is None else snap(strikes, k)


def _options(table: pd.DataFrame) -> pd.Series:
    """Which rows are option legs with a strike."""
    return table[TYPE].isin(("call", "put")) & table[STRIKE].notna()


def flip(table: pd.DataFrame, rows: Sequence[int] | None = None) -> pd.DataFrame:
    """Every quantity (or those of ``rows``) negated: long becomes short."""
    out = frame(table.to_dict("records"))
    index = list(range(len(out))) if rows is None else list(rows)
    out.loc[index, QTY] = -out.loc[index, QTY]
    return out


def shift(table: pd.DataFrame, direction: int, grid: Grid) -> pd.DataFrame:
    """Every strike moved one interval up (``direction`` +1) or down (−1): one common step, the widest listed interval
    at the legs' strikes, so that the structure keeps its shape, then snapped to each leg's listed strikes."""
    out = frame(table.to_dict("records"))
    options = _options(out)
    if not options.any():
        return out
    step = max(grid.interval(out.at[i, EXPIRY], out.at[i, STRIKE]) for i in out.index[options])
    for i in out.index[options]:
        moved = out.at[i, STRIKE] + direction * step
        if moved > 0:
            out.at[i, STRIKE] = grid.snap(out.at[i, EXPIRY], moved)
    return out


def widen(table: pd.DataFrame, direction: int, grid: Grid, forward: float) -> pd.DataFrame:
    """The wings moved out one interval (``direction`` +1) or in (−1), never onto or across the body.

    With three or more distinct strikes the wings are the lowest and the highest; with two, the strike further from
    ``forward`` (a vertical's short leg, say). A wing moving in stops one interval short of the next strike inside it.
    """
    out = frame(table.to_dict("records"))
    options = _options(out)
    strikes = sorted(set(out.loc[options, STRIKE]))
    if len(strikes) < 2:
        return out
    if len(strikes) == 2:
        far = max(strikes, key=lambda k: abs(k - forward))
        wings = {far: 1 if far == strikes[-1] else -1}
    else:
        wings = {strikes[0]: -1, strikes[-1]: 1}
    for wing, side in wings.items():
        inner = strikes[-2] if side > 0 else strikes[1]
        for i in out.index[options & (out[STRIKE] == wing)]:
            label = out.at[i, EXPIRY]
            moved = grid.snap(label, wing + side * direction * grid.interval(label, wing))
            # Moving in: never onto or past the next strike inside the wing.
            if direction < 0 and (moved - inner) * side <= 0:
                continue
            if moved > 0:
                out.at[i, STRIKE] = moved
    return out


def roll(table: pd.DataFrame, direction: int, grid: Grid, rows: Sequence[int] | None = None) -> pd.DataFrame:
    """Every option leg (or those of ``rows``) moved to the next listed expiry (``direction`` +1) or the previous
    one (−1), by settlement, its strike snapped to the strikes listed there. A leg with no such expiry stays.
    The textbook market has no listed expiries: nothing moves."""
    out = frame(table.to_dict("records"))
    if grid.published is None:
        return out
    order = listed(grid.published)
    index = list(range(len(out))) if rows is None else list(rows)
    for i in index:
        key = parse_expiry(out.at[i, EXPIRY])
        if key is None or out.at[i, TYPE] not in INDEX_KINDS or key not in order:
            continue
        j = order.index(key) + direction
        if 0 <= j < len(order):
            label = expiry_text(*order[j])
            out.at[i, EXPIRY] = label
            if not _blank(out.at[i, STRIKE]):
                out.at[i, STRIKE] = grid.snap(label, out.at[i, STRIKE])
    return out


def duplicate(table: pd.DataFrame, row: int) -> pd.DataFrame:
    """The legs with a copy of ``row`` after it."""
    records = table.to_dict("records")
    return frame([*records[: row + 1], dict(records[row]), *records[row + 1 :]])


def remove(table: pd.DataFrame, row: int) -> pd.DataFrame:
    """The legs without ``row``."""
    records = table.to_dict("records")
    return frame(records[:row] + records[row + 1 :])


# --- Pricing ------------------------------------------------------------------------------------------------------


#: How each leg's vol was read (the blotter's ``IV src``).
SURFACE, OVERRIDE, EXTRAPOLATED, UNLISTED, FLAT, EXPIRED = (
    "surface",
    "override",
    "extrap.",
    "unlisted",
    "flat",
    "expired",
)


@dataclass(frozen=True)
class PricedLeg:
    """One row of the legs, priced.

    ``leg`` is the leg the analysis values (None when the row is off, expired or blank); ``maturity`` is in years to
    the settlement; ``vol`` is the vol it is valued at and ``fit`` the surface's (the sidebar vol under the textbook
    market); ``theo`` is the model value per share at ``fit`` and ``entry`` the entry price per share; ``quote`` is the
    chain's market (None without one).
    """

    row: int
    on: bool
    kind: str
    quantity: float
    strike: float | None
    label: str  # the expiry as shown: 18‑Dec‑26 AM, or 365d
    expiry: Expiry | None
    maturity: float
    vol: float
    fit: float
    src: str
    quote: Quote | None
    theo: float
    entry: float
    leg: strat.Leg | None

    @property
    def expired(self) -> bool:
        return self.src == EXPIRED


@dataclass(frozen=True, eq=False)
class Book:
    """The legs priced: each row, the market of the analysis (spot, the front expiry's carry and ATMF vol, the front
    maturity) and the strategy of the legs that are on and live."""

    legs: tuple[PricedLeg, ...]
    market: strat.Market
    strategy: strat.Strategy | None
    front: Expiry | None  # the front live expiry (index)
    errors: tuple[str, ...]

    @property
    def live(self) -> list[PricedLeg]:
        return [p for p in self.legs if p.leg is not None]


def _carry(published: PublishedSurface, key: Expiry, spot: float) -> tuple[float, float]:
    """r and q that put the expiry's parity forward and discount factor into Black-Scholes-Merton (``market.carry``)."""
    T = published.maturity(key)
    rate = -math.log(published.discount_factor(key)) / T
    return rate, rate - math.log(published.forward(key) / spot) / T


def price_index(
    table: pd.DataFrame,
    published: PublishedSurface,
    as_of: datetime,
    *,
    name: str = "Strategy",
    fees_per_contract: float = 0.0,
    multiplier: float = 100.0,
) -> Book:
    """The legs priced on the published surface at ``as_of`` (BK-3).

    Each row's expiry is converted to years with ``clock.time_to_expiry`` at ``as_of``; a leg that has settled is
    ``expired`` and left out of every number. Its vol is the override typed in the row or the surface's at its strike
    (``surface``, ``extrap.`` beyond the expiry's quotes, or ``unlisted`` for a date the chain no longer lists, read
    from the surface in total variance), its market the chain's quote, and its entry the typed price, else the mid
    (else the model value, where the chain has no two-sided quote). The analysis market carries the front live expiry's
    parity carry and ATMF vol: Black-Scholes-Merton with that carry is Black-76 on the front's forward, and later
    expiries' carry differs from it by a few basis points on the synthetic chain.
    """
    spot = float(published.spot)
    order = listed(published)
    priced, errors = [], []
    for i, row in enumerate(cells(table)):
        on, qty, kind, label, _days, k, iv, entry = row
        if kind is None or qty is None:
            continue  # a blank row being edited
        if qty == 0:
            errors.append(f"Row {i + 1}: the quantity is zero; type a signed whole number, or switch the leg off.")
            continue
        if kind not in INDEX_KINDS:
            errors.append(f"Row {i + 1}: the index takes calls and puts only (SPX itself cannot be traded).")
            continue
        key = parse_expiry(label)
        if key is None:
            errors.append(f"Row {i + 1}: pick a listed expiry ({expiry_text(*order[0])} to {expiry_text(*order[-1])}).")
            continue
        if k is None or not k > 0:
            errors.append(f"Row {i + 1}: the strike must be positive.")
            continue
        T = clock.time_to_expiry(as_of, key.day, style=key.settlement)
        is_listed = key in order
        quote = None
        if T <= 0:
            src, fit, vol = EXPIRED, math.nan, math.nan
        else:
            if is_listed:
                read = published.vol(k, key)
                fit, src = read.vol, (EXTRAPOLATED if read.extrapolated else SURFACE)
                try:
                    quote = published.quote(key, k, kind)
                except KeyError:
                    quote = None
            else:
                fit, src = float(published.surface.implied_vol_at(k / spot, T, extrapolate=True)), UNLISTED
            if not np.isfinite(fit):
                errors.append(f"Row {i + 1}: the surface has no vol at {k:,.0f} on {label}; type an IV override.")
                continue
            vol = fit
            if iv is not None:
                if not iv > 0:
                    errors.append(f"Row {i + 1}: the IV override must be positive.")
                    continue
                vol, src = iv / 100.0, OVERRIDE
        priced.append((i, bool(on), kind, qty, k, label, key, T, vol, fit, src, quote, entry))
    live = [p for p in priced if p[10] != EXPIRED and p[1]]
    front = min(live, key=lambda p: p[7]) if live else None
    front_key = front[6] if front is not None and front[6] in order else None
    if front_key is None:
        candidates = [e for e in order if published.maturity(e) > 0]
        front_key = min(candidates, key=lambda e: abs(published.maturity(e) - (front[7] if front else 0.0)))
    rate, q = _carry(published, front_key, spot)
    T_front = front[7] if front is not None else published.maturity(front_key)
    atm = published.vol(published.forward(front_key), front_key).vol
    market = strat.Market(spot, rate, atm, T_front, q)
    legs = []
    for i, on, kind, qty, k, label, key, T, vol, fit, src, quote, entry in priced:
        if src == EXPIRED:  # settled: its intrinsic value at today's spot
            theo = max(spot - k, 0.0) if kind == "call" else max(k - spot, 0.0)
        else:
            theo = float(bs.price(spot, k, T, rate, fit, q, kind))
        if entry is None:
            entry = quote.mid if quote is not None and quote.bid > 0 else theo
        leg = strat.Leg(kind, qty, k, T, vol, entry) if src != EXPIRED and on else None
        legs.append(PricedLeg(i, on, kind, qty, k, label, key, max(T, 0.0), vol, fit, src, quote, theo, entry, leg))
    return _book(legs, market, front_key, errors, name, fees_per_contract, multiplier)


def price_textbook(
    table: pd.DataFrame,
    market: strat.Market,
    default_strike: float,
    *,
    name: str = "Strategy",
    fees_per_contract: float = 0.0,
    multiplier: float = 100.0,
) -> Book:
    """The legs priced in the textbook market: blank days, vol and strike follow the sidebar; the entry is the typed
    price, else the model value (there is no market)."""
    priced, errors = [], []
    for i, row in enumerate(cells(table)):
        on, qty, kind, _label, days, k, iv, entry = row
        if kind is None or qty is None:
            continue
        if qty == 0:
            errors.append(f"Row {i + 1}: the quantity is zero; type a signed whole number, or switch the leg off.")
            continue
        try:
            leg = strat.Leg(
                kind=kind,
                quantity=qty,
                strike=0.0 if kind == "stock" else (default_strike if k is None else k),
                expiry=None if kind == "stock" or days is None else days / 365.0,
                volatility=None if kind == "stock" or iv is None else iv / 100.0,
                premium=entry,
            )
        except ValueError as exc:
            errors.append(f"Row {i + 1}: {exc}")
            continue
        T = market.expiry if leg.expiry is None else leg.expiry
        vol = market.volatility if leg.volatility is None else leg.volatility
        if leg.is_option:
            theo = float(bs.price(market.spot, leg.strike, T, market.rate, vol, market.dividend_yield, kind))
        else:
            theo = market.spot
        src = "" if not leg.is_option else (OVERRIDE if leg.volatility is not None else FLAT)
        label = "" if not leg.is_option else f"{T * 365:.0f}d"
        entered = theo if entry is None else entry
        priced.append(
            PricedLeg(i, bool(on), kind, qty, leg.strike if leg.is_option else None, label, None, T, vol,
                      market.volatility, src, None, theo, entered, leg if on else None)
        )  # fmt: skip
    return _book(priced, market, None, errors, name, fees_per_contract, multiplier)


def _book(legs, market, front, errors, name, fees_per_contract, multiplier) -> Book:
    live = [p.leg for p in legs if p.leg is not None]
    strategy = None
    if live:
        option_units = sum(abs(leg.quantity) for leg in live if leg.is_option)
        strategy = strat.Strategy(name, tuple(live), fees=fees_per_contract * option_units / multiplier)
    return Book(tuple(legs), market, strategy, front, tuple(errors))


# --- Book analytics -----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Package:
    """The structure's natural market per share: ``bid`` and ``ask`` in the structure's direction (a credit's are what
    selling it receives), ``theo`` at the fit, ``mid`` at the legs' mids, ``credit`` and the package vega ($ per vol
    point per share)."""

    bid: float
    ask: float
    mid: float
    theo: float
    credit: bool
    vega: float


def package(book: Book) -> Package | None:
    """The package market of the legs that are on and live (spec 3.2 item 2): a credit structure's bid is Σ shorts'
    bids − Σ longs' asks and its ask Σ shorts' asks − Σ longs' bids; a debit structure's the other way round, so that
    both read as positive amounts with a ``cr`` or ``db`` tag. None when a leg has no two-sided quote."""
    live = book.live
    if not live or any(p.quote is None or not p.quote.bid > 0 for p in live):
        return None
    mid = sum(p.quantity * p.quote.mid for p in live)  # type: ignore[union-attr]
    theo = sum(p.quantity * p.theo for p in live)
    credit = mid < 0
    sign = -1.0 if credit else 1.0  # the structure's direction: a credit is sold
    # Buying the package (debit) pays the asks of its longs and receives the bids of its shorts; selling it (credit)
    # receives the bids of what it sells (the shorts) and pays the asks of what it buys back (the longs).
    bid = sum((p.quantity * sign) * (p.quote.bid if p.quantity * sign > 0 else p.quote.ask) for p in live)  # type: ignore[union-attr]
    ask = sum((p.quantity * sign) * (p.quote.ask if p.quantity * sign > 0 else p.quote.bid) for p in live)  # type: ignore[union-attr]
    market = book.market
    vega = sum(
        p.quantity * float(bs.greeks(market.spot, p.strike, p.maturity, market.rate, p.fit, market.dividend_yield,
                                     p.kind)["vega"]) / 100.0
        for p in live
    )  # fmt: skip
    return Package(bid=bid, ask=ask, mid=sign * mid, theo=sign * theo, credit=credit, vega=vega)


def profit_intervals(
    pnl_at: Callable[[np.ndarray], np.ndarray], breakevens: Sequence[float]
) -> list[tuple[float, float]]:
    """The spot intervals where the P&L at expiry is positive, between consecutive breakevens (0 and ∞ at the ends),
    judged at each interval's midpoint (its geometric mean, the lower end halved and the upper doubled)."""
    edges = [0.0, *sorted(breakevens), math.inf]
    intervals = []
    for a, b in pairwise(edges):
        if a == 0.0 and math.isinf(b):
            probe = 1.0
        elif a == 0.0:
            probe = b / 2.0
        elif math.isinf(b):
            probe = a * 2.0
        else:
            probe = math.sqrt(a * b)
        if float(np.asarray(pnl_at(np.array([probe])))[0]) > 0:
            intervals.append((a, b))
    return intervals


def probability(cdf: Callable[[float], float], intervals: Sequence[tuple[float, float]]) -> float:
    """The probability mass of ``intervals`` under ``cdf`` (P(S ≤ x))."""
    total = 0.0
    for a, b in intervals:
        upper = 1.0 if math.isinf(b) else float(cdf(b))
        lower = 0.0 if a <= 0 else float(cdf(a))
        total += upper - lower
    return float(min(max(total, 0.0), 1.0))


def rn_moments(pnl_at: Callable[[np.ndarray], np.ndarray], quantile: Callable[[np.ndarray], np.ndarray],
               points: int = 4000) -> tuple[float, float]:  # fmt: skip
    """Mean and standard deviation of the P&L at expiry under a distribution given by its quantile function: the
    midpoint rule in probability, E[f(S)] = ∫₀¹ f(Q(p)) dp, on ``points`` equal steps (the tails beyond the first and
    last midpoints, 1/(2·points) each, are read at those midpoints)."""
    p = (np.arange(points) + 0.5) / points
    spots = np.asarray(quantile(p), dtype=float)
    keep = np.isfinite(spots)
    values = np.asarray(pnl_at(spots[keep]), dtype=float)
    mean = float(values.mean())
    return mean, float(np.sqrt(np.mean((values - mean) ** 2)))


def lognormal_cdf(spot: float, forward: float, vol: float, years: float) -> Callable[[float], float]:
    """P(S_T ≤ x) of a log-normal price with median F·e^{−σ²T/2} (the flat-vol basis)."""
    sd = vol * math.sqrt(years)
    centre = math.log(forward) - 0.5 * sd * sd
    return lambda x: float(ndtr((math.log(x) - centre) / sd)) if x > 0 else 0.0


def straddle_move(published: PublishedSurface, key: Expiry) -> tuple[float, float] | None:
    """The straddle-implied move to the expiry: the mid of the call and the put at the listed strike nearest the
    forward, over spot (UI-13), and that strike. None without a two-sided quote on both."""
    k = snap(published.strikes(key), published.forward(key))
    try:
        call, put = published.quote(key, k, "call"), published.quote(key, k, "put")
    except KeyError:
        return None
    if not (call.bid > 0 and put.bid > 0):
        return None
    return (call.mid + put.mid) / published.spot, k
