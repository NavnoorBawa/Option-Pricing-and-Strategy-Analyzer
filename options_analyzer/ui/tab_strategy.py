"""Strategy tab: build a multi-leg position from a template, read its premium, package market, maximum P&L,
breakevens and probability of profit, and its P&L across spot, time and vol (UI spec 3.2; roadmap BK-2, BK-3, RK-9a,
UI-4, UI-8, UI-9 and UI-13).

Top to bottom: the builder bar (template, strikes by delta or width, expiry, position settings), six headline tiles,
the what-if strip (:mod:`.whatif`, shared with Risk), the primary view (the P&L chart with its Greek strip in one
figure, or the date × spot P&L table), the blotter (View and Edit), and a footer line naming the engine, the dynamics,
the entry basis and the probability basis.

Under the synthetic index the legs hold listed expiries (date and settlement) and take their vols from the published
surface and their bid and ask from the chain (:mod:`.legs`); under the textbook market they keep days to expiry and
the flat sidebar vol, as in version 2. :func:`controls` draws the builder bar on every run (its widgets are bound to the
URL); :func:`build_position` builds the position without drawing anything, for the Risk tab; :func:`render` draws the
rest while the tab is open.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import ndtri

from .. import black_scholes as bs
from .. import clock
from .. import strategies as strat
from ..vol_surface.published import ONE_SD, TWO_SD, Expiry
from . import cache, charts, charts_strategy, components, exports, surface_state, whatif
from . import legs as lg
from . import market as mk
from .components import Card, card_row, link_match, table
from .format import (
    MINUS,
    MISSING,
    NB_HYPHEN,
    decay_label,
    md,
    md_code,
    money,
    money_short,
    number,
    pct,
    premium,
    probability_text,
    signed,
    strike,
    time_left,
    vol_change,
)
from .inputs import Inputs

CUSTOM = "Custom"
#: The template the tab opens on: under the index the spec's default state (an iron condor, 3.2), in the textbook
#: market version 2's (a bull call spread), which that market reproduces.
DEFAULT_TEMPLATE = "Iron Condor"
TEXTBOOK_TEMPLATE = "Bull Call Spread"


def default_template(index: bool) -> str:
    """The template the tab opens on under the index (``True``) or in the textbook market."""
    return DEFAULT_TEMPLATE if index else TEXTBOOK_TEMPLATE


DEFAULT_MULTIPLIER = 100
#: The ranges (smallest, largest) of the position-size inputs: contracts, shares per contract and commission ($).
CONTRACTS = (1, 100_000)
MULTIPLIER = (1, 10_000)
COMMISSION = (0.0, 100.0)
#: Strike width bounds in % of the strike. Below 50 % every template keeps its lowest strike (K - 2w) positive.
MIN_WIDTH_PCT = 0.1
MAX_WIDTH_PCT = 45.0
#: The automatic width never exceeds this, so very volatile or long-dated templates stay readable.
MAX_AUTO_WIDTH_PCT = 25.0
#: Strikes with as many decimals as they have (up to 10 significant digits): the textbook editor's strike format.
STRIKE_FORMAT = "%.10g"
#: The builder's choices, as their controls show them and a link holds them.
BY_DELTA, BY_WIDTH = "Δ", "Width"
VIEWS = ("Chart", "Table")
RANGES = ("±5%", "±10%", "±20%", "RN 95%", "Auto")
GREEKS = ("Δ", "Γ", "Vega", "Θ")
UNITS = ("$", "% of max loss", "% of premium")
BLOTTER_MODES = ("View", "Edit")
#: The dated curves, as fractions of the front expiry's trading sessions (spec 3.2 item 4: ⅓ and ⅔ by default).
DATED = {"Dated ⅓": 1 / 3, "Dated ⅔": 2 / 3}
#: The tiles' value size (px) while the six of them hold one line: 1.375rem of the theme's 14 px, a step below the
#: theme's 1.625rem, so that they fit beside the open sidebar at 1,440 px (spec 3.2's chart above the fold needs the
#: room); once they wrap (a laptop, a phone) they take the theme's size.
TILE_VALUE_PX = 19.25
#: Points of the P&L chart's spot grid.
GRID_POINTS = 600


# --- The position ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Position:
    """The tab's position: the strategy of the legs that are on and live, the market it is valued in, and its size.
    ``book`` holds every leg priced (the blotter's rows) and is left out of comparisons."""

    strategy: strat.Strategy
    market: strat.Market
    contracts: int
    multiplier: float
    book: lg.Book | None = field(default=None, compare=False)

    @property
    def units(self) -> float:
        """Dollar scaling: contracts × shares per contract."""
        return self.contracts * self.multiplier

    def describe(self) -> str:
        """The position in a few words, the start of the Risk tab's context line: under the index the legs as a desk
        writes them, each with its listed expiry only when the legs have more than one (the line that follows names
        every expiry with its time left: four copies of ``18‑Dec‑26 AM`` took an iron condor to three lines on a
        phone); under the textbook market the legs' labels."""
        if self.book is not None and self.book.front is not None:  # the index: listed expiries, legs as a desk writes
            live = self.book.live
            dated = len({p.expiry for p in live if p.kind != "stock"}) > 1
            parts = [_leg_short(p, dated) for p in live]
            size = f"{number(self.contracts, 0)} × {number(self.multiplier, 0)}"
            return md(f"**{self.strategy.name}** — {' / '.join(parts)} · {size}")
        legs = ", ".join(strat.leg_labels(self.strategy, self.market))
        size = f"{number(self.contracts, 0)} × {number(self.multiplier, 0)}"
        return md(f"**{self.strategy.name}** — {legs} · {size} multiplier")


def _leg_short(p: lg.PricedLeg, expiry: bool = True) -> str:
    """``+1 5,300P 18‑Dec‑26 AM`` (``+1 102.5C 365d`` in the textbook market, ``+100 shares``); ``+1 5,300P`` without
    the ``expiry``. The quantity and the strike are joined by a no-break space: a phone broke the Risk tab's line
    between them (``… / +1`` above ``6,250C · 1 × 100``)."""
    qty = signed(p.quantity, 0) if p.quantity else "0"
    if p.kind == "stock":
        return f"{qty} share{'s' if abs(p.quantity) != 1 else ''}"
    return f"{qty} {strike(p.strike, math.nan)}{p.kind[0].upper()}" + (f" {p.label}" if expiry else "")


# --- Templates and the builder's values --------------------------------------------------------------------------


def _label(name: str) -> str:
    """A template as the menu lists it: ``Range-bound · Iron Condor`` (``Custom`` alone)."""
    return name if name == CUSTOM else f"{strat.TEMPLATES[name].category} · {name}"


def _name(label: str) -> str:
    return label.split(" · ", 1)[-1]


def template_labels(index: bool) -> list[str]:
    """The menu's options in ``strategies.TEMPLATE_CATEGORIES`` order: under the index only the templates of option
    legs (SPX itself cannot be traded), then ``Custom``."""
    names = lg.index_templates() if index else list(strat.TEMPLATES)
    return [_label(n) for n in names] + [CUSTOM]


def link_selection(name: str, index: bool = True) -> dict[str, str] | None:
    """The URL parameter that selects strategy ``name`` in the template menu (``?strategy=Range-bound · Iron Condor``).

    A link may name a template alone (``?strategy=iron-condor``), matched without case or punctuation.

    Examples
    --------
    >>> link_selection("iron condor")
    {'strategy': 'Range-bound · Iron Condor'}
    >>> link_selection("Custom")
    {'strategy': 'Custom'}
    >>> link_selection("Iron Condr") is None
    True
    """
    names = [*(lg.index_templates() if index else strat.TEMPLATES), CUSTOM]
    found = link_match(name, names, aliases={_label(n): n for n in names})
    return None if found is None else {"strategy": _label(found)}


#: Parameters of version 3.0's two template menus (``strategy_category`` and one ``strategy_<category>`` per
#: category), which :func:`read_link` maps to ``strategy``.
LEGACY_PARAMS = ("strategy_category", *(f"strategy_{c}" for c in strat.TEMPLATE_CATEGORIES))


def read_link(params: dict[str, list[str]]) -> list[str]:
    """Rewrite a hand-typed link's template to the menu's parameter, in place, and return its notes.

    ``?strategy=<name>`` (any case or punctuation, with or without its category) and the two menus of earlier links
    (``strategy_category`` with ``strategy_<category>``) become ``strategy=<category> · <name>``. A template that
    holds stock is not offered under the index (SPX itself cannot be traded): it is dropped with a note.
    """
    index = params.get("preset", [mk.INDEX])[-1] == mk.INDEX
    notes = []
    legacy = {key: params.pop(key) for key in LEGACY_PARAMS if key in params}
    asked = params.pop("strategy", [None])[-1]
    if asked is None and legacy:
        category = legacy.get("strategy_category", [""])[-1]
        if category == CUSTOM:
            asked = CUSTOM
        elif category in strat.TEMPLATE_CATEGORIES:
            names = [t.name for t in strat.TEMPLATES.values() if t.category == category]
            chosen = legacy.get(f"strategy_{category}", [names[0]])[-1]
            asked = chosen if chosen in names else names[0]
    if asked is None:
        return notes
    selection = link_selection(asked, index)
    if selection is not None:
        params.update({key: [value] for key, value in selection.items()})
    elif link_selection(asked, False) is not None:
        notes.append(
            f"This link asks for {_name(link_selection(asked, False)['strategy'])}, which holds stock: "
            f"{mk.INDEX_ROOT} itself cannot be traded, so the {default_template(True)} is shown."
        )
    elif asked.strip():
        notes.append(
            f"This link asks for a strategy named {_shown(asked)}, which is not one of the templates, so it was "
            "ignored."
        )
    return notes


def _shown(value: str) -> str:
    """A link's value as a note quotes it: in a code span, at most 40 characters (as the app's other link notes)."""
    value = " ".join(value.split())
    return md_code(value if len(value) <= 40 else value[:39] + "…")


def _control(
    key: str,
    default: Any,
    parse: Callable[[str], Any],
    valid: Callable[[Any], bool],
    read: Callable[[str], str] = lambda key: st.query_params[key],
) -> Any:
    """A control's value in this run: its session state once :func:`controls` has drawn it; otherwise (a caller
    of :func:`build_position` that draws no controls) its URL parameter (``read``), read as the control reads it,
    when that is valid; otherwise the control's default."""
    if key in st.session_state:
        value = st.session_state[key]
        return value if value is None or valid(value) else default
    try:
        value = parse(read(key))
    except (KeyError, ValueError, OverflowError):
        return default
    return value if valid(value) else default


def _link_template(key: str) -> str:
    """The template a link asks for, as the app's link reading maps it (:func:`read_link`): a caller of
    :func:`build_position` without the controls sees the raw link, with a short name or the two menus of earlier
    links."""
    params = {name: st.query_params.get_all(name) for name in st.query_params}
    read_link(params)
    return params[key][-1]


def _within(bounds: tuple[float, float]) -> Callable[[float], bool]:
    return lambda value: bounds[0] <= value <= bounds[1]


def _whole(text: str) -> int:
    """A whole number from a URL value, as a whole-number input reads it (2.5 is 2)."""
    return int(float(text))


def _round_step(x: float) -> float:
    """The 1, 2, 2.5 or 5 x 10^n step closest to ``x`` in relative terms (the spacing of listed strikes)."""
    power = 10.0 ** np.floor(np.log10(x))
    steps = [m * power * scale for scale in (1.0, 10.0) for m in (1.0, 2.0, 2.5, 5.0)]
    return min(steps, key=lambda s: abs(np.log(s / x)))


def round_width(width: float) -> float:
    """A strike width on a price tick: whole cents, or two significant figures below ten cents."""
    decimals = max(2, 1 - int(np.floor(np.log10(width))))
    return round(float(width), decimals)


def auto_width(strike: float, volatility: float, expiry: float) -> float:
    """Default strike spacing ($): about half the one-standard-deviation move σ√T, on a round strike step."""
    step = _round_step(0.5 * volatility * np.sqrt(expiry) * strike)
    return round_width(float(np.clip(step, strike * MIN_WIDTH_PCT / 100, strike * MAX_AUTO_WIDTH_PCT / 100)))


def _expiry_param(key: Expiry) -> str:
    """An expiry as a link holds it: ``2026-12-18 AM``."""
    return f"{key.day.isoformat()} {key.settlement}"


def _parse_expiry_param(text: str) -> Expiry:
    day, style = str(text).split()
    if style not in ("AM", "PM"):
        raise ValueError(style)
    return Expiry(date.fromisoformat(day), style)


@dataclass(frozen=True)
class _Choice:
    """What the builder chose: the template (``None`` for a custom position), its legs, where they are stored, the
    position size and the position settings."""

    name: str
    label: str
    base: pd.DataFrame
    store: str
    expiry: Expiry | None  # the template's expiry (index)
    contracts: int
    multiplier: float
    commission: float
    risk_neutral: bool
    drift: float  # % a year, the user's view
    carry: bool


def _published(inputs: Inputs):
    return surface_state.current(inputs)


def _template_expiry(published) -> tuple[list[Expiry], Expiry]:
    """The listed expiries and the template's expiry: the session's, else the link's (``?strategy_expiry=2026-12-18
    AM``), else the AM monthly nearest 45 days. A date no longer listed moves to the nearest listed one."""
    order = lg.listed(published)
    default = lg.default_expiry(published)
    current = st.session_state.get("strategy_expiry")
    if current is None:
        try:
            current = _parse_expiry_param(st.query_params["strategy_expiry"])
        except (KeyError, ValueError):
            current = default
    if not isinstance(current, Expiry) or current not in order:
        if isinstance(current, Expiry):
            later = [e for e in order if e.day >= current.day]
            current = later[0] if later else default
        else:
            current = default
    return order, current


def _choice(inputs: Inputs) -> _Choice:
    """The builder's values, from the session state (:func:`controls` draws them earlier in the same run) or, when
    the controls are not drawn, from the URL and the controls' defaults.

    Raises ``ValueError`` when the template cannot be built at the sidebar inputs.
    """
    published = _published(inputs)
    index = published is not None
    labels = template_labels(index)
    label = _control("strategy", _label(default_template(index)), str, labels.__contains__, _link_template)
    name = _name(label)
    expiry = None
    if index:
        _, expiry = _template_expiry(published)
        by = _control("strategy_strikes_by", BY_DELTA, str, (BY_DELTA, BY_WIDTH).__contains__)
        delta = _control("strategy_short_delta", lg.DEFAULT_SHORT_DELTA, int, lg.SHORT_DELTAS.__contains__)
        wing = _control("strategy_wing_pts", lg.DEFAULT_WING, int, lg.WING_POINTS.__contains__)
        width_pct = _control("strategy_width_pct", None, float, _within((MIN_WIDTH_PCT, MAX_WIDTH_PCT)))
        if name == CUSTOM:
            atm = lg.snap(published.strikes(expiry), published.forward(expiry))
            base = lg.frame([{lg.ON: True, lg.QTY: 1.0, lg.TYPE: "call", lg.EXPIRY: lg.expiry_text(*expiry),
                              lg.STRIKE: atm}])  # fmt: skip
        else:
            atm = lg.snap(published.strikes(expiry), published.forward(expiry))
            T = published.maturity(expiry)
            width = (
                auto_width(atm, published.vol(atm, expiry).vol, T)
                if width_pct is None
                else round_width(atm * float(width_pct) / 100.0)
            )
            base = lg.index_template(
                name,
                published,
                expiry,
                strikes_by="delta" if by == BY_DELTA else "width",
                short_delta=delta / 100.0,
                wing=float(wing),
                width=width,
            )
    else:
        width_pct = _control("strategy_width_pct", None, float, _within((MIN_WIDTH_PCT, MAX_WIDTH_PCT)))
        if name == CUSTOM:
            base = lg.frame([{lg.ON: True, lg.QTY: 1.0, lg.TYPE: "call", lg.STRIKE: inputs.strike}])
        else:
            base = lg.textbook_template(name, inputs.strike, _width(inputs, width_pct), inputs.expiry)
    contracts = _control("contracts", CONTRACTS[0], _whole, _within(CONTRACTS))
    multiplier = _control("multiplier", DEFAULT_MULTIPLIER, _whole, _within(MULTIPLIER))
    commission = _control("commission", 0.0, float, _within(COMMISSION))
    risk_neutral = bool(st.session_state.get("risk_neutral", True))
    drift = float(st.session_state.get("drift", 8.0))
    carry = bool(st.session_state.get("carry", False))
    store = f"{inputs.preset}::{label}"
    return _Choice(name, label, base, store, expiry, int(contracts), float(multiplier), float(commission),
                   risk_neutral, drift, carry)  # fmt: skip


def _width(inputs: Inputs, width_pct: float | None) -> float:
    """The textbook strike width ($): automatic when ``width_pct`` is None, otherwise that share of the strike."""
    if width_pct is None:
        return auto_width(inputs.strike, inputs.volatility, inputs.expiry)
    return round_width(inputs.strike * float(width_pct) / 100.0)


# --- The builder bar (controls) -----------------------------------------------------------------------------------


def _days_left(as_of: datetime, day: date, years: float) -> str:
    """Time to an expiry as the tab writes it everywhere, floored as the what-if strip floors it (``55d`` for 55.73
    days, where rounding read 56d in the menu beside 55d in the chart)."""
    return time_left(years * clock.MINUTES_PER_YEAR, expiry_day=day == as_of.date())


def _settings_label() -> str:
    """The Position settings popover's label: ``1 × 100 · $0.00/ct · RN``."""
    contracts = _control("contracts", CONTRACTS[0], _whole, _within(CONTRACTS))
    multiplier = _control("multiplier", DEFAULT_MULTIPLIER, _whole, _within(MULTIPLIER))
    commission = _control("commission", 0.0, float, _within(COMMISSION))
    basis = "RN" if st.session_state.get("risk_neutral", True) else "view"
    return f"{number(contracts, 0)} × {number(multiplier, 0)} · {money(commission)}/ct · {basis}"


def _link_expiry(chosen: Expiry, default: Expiry) -> None:
    """Write the template's expiry to the link by hand (``?strategy_expiry=2026-12-18 AM``): the menu is not bound,
    because its labels count days, which a link must not hold. The link leaves it out at the default."""
    if chosen == default:
        st.query_params.pop("strategy_expiry", None)
    else:
        st.query_params["strategy_expiry"] = _expiry_param(chosen)


def controls(inputs: Inputs) -> None:
    """Draw the builder bar: template, strikes by delta or width, expiry, and the Position settings popover.

    The app draws it on every run, also while the tab is closed: its widgets are bound to the URL, and Streamlit
    removes the URL parameter of a widget it does not draw, so a link to any tab keeps the position. They are cheap:
    the analysis runs only in :func:`render`, while the tab is open. :func:`build_position` and :func:`render` read the
    values from the session state.

    Parameters
    ----------
    inputs : Inputs
        The sidebar's market: the index's listed expiries and strikes, or the textbook strike, vol and expiry.
    """
    published = _published(inputs)
    index = published is not None
    labels = template_labels(index)
    if st.session_state.get("strategy") not in (None, *labels):
        st.session_state["strategy"] = _label(default_template(index))  # a stock template under the index
    bar = st.container(horizontal=True, key="strategy_builder", wrap=True, gap="small", vertical_alignment="bottom")
    label = bar.selectbox(
        "Template",
        labels,
        index=labels.index(_label(default_template(index))),
        key="strategy",
        bind="query-params",
        width=270,
        help="The structure, by category. Its legs follow the builder's settings until you edit them in the blotter; "
        "Reset legs to template (Edit) rebuilds them.",
    )
    name = _name(label)
    uses_width = name != CUSTOM and strat.TEMPLATES[name].uses_width
    if index:
        # A menu rather than a segmented control: in a row of menus its label and box line up with theirs (a segmented
        # control is 6 px shorter, so its label sat 6 px lower in the bottom-aligned row).
        by = bar.selectbox(
            "Strikes by",
            [BY_DELTA, BY_WIDTH],
            key="strategy_strikes_by",
            bind="query-params",
            width=100,
            disabled=not uses_width,
            help="Δ: the short strikes are the listed strikes whose own delta, at their own surface vol, is nearest "
            "the target; wings sit the set number of points beyond them (structures with legs at the money centre "
            "on the listed strike nearest the forward). Width: strikes at ATMF ± n × the width.",
        )
        if by == BY_WIDTH:
            _width_input(bar, uses_width, "the listed strike nearest the forward")
        else:
            # Plain numbers, no format_func: a bound menu writes its options as shown to the link and reads them
            # back so, and the link should hold ?strategy_short_delta=16, as a hand-typed one would.
            bar.selectbox(
                "Shorts (Δ)",
                lg.SHORT_DELTAS,
                index=lg.SHORT_DELTAS.index(lg.DEFAULT_SHORT_DELTA),
                key="strategy_short_delta",
                bind="query-params",
                width=90,
                disabled=not uses_width,
                help="The short strikes' delta (spot convention, at each strike's own surface vol), snapped to the "
                "listed strike nearest it.",
            )
            bar.selectbox(
                "Wings (pts)",
                lg.WING_POINTS,
                index=lg.WING_POINTS.index(lg.DEFAULT_WING),
                key="strategy_wing_pts",
                bind="query-params",
                width=100,
                disabled=not uses_width,
                help="Wing width in index points, equal on both sides, snapped to listed strikes (SPX lists every 50 "
                "points this far out of the money).",
            )
        order, current = _template_expiry(published)
        st.session_state["strategy_expiry"] = current  # listed, before the menu reads it
        rows = {
            Expiry(r.expiry, str(r.settlement)): r.maturity for r in published.expiries.itertuples(index=False)
        }  # fmt: skip
        chosen = bar.selectbox(
            "Expiry",
            order,
            format_func=lambda e: f"{lg.expiry_text(*e)} · {_days_left(inputs.context.as_of, e.day, rows[e])}",
            key="strategy_expiry",
            width=200,
            help="The template's expiry (a time spread's front). SPX monthlies settle AM at the opening prints, SPXW "
            "weeklies PM at the close. The default is the monthly nearest 45 days.",
        )
        _link_expiry(chosen, lg.default_expiry(published))
    else:
        _width_input(bar, uses_width, "the sidebar strike")
    with bar.popover(_settings_label(), key="strategy_settings", help="Position size, commission and assumptions."):
        st.number_input(
            "Contracts", min_value=CONTRACTS[0], max_value=CONTRACTS[1], value=1, step=1, key="contracts",
            bind="query-params",
        )  # fmt: skip
        st.number_input(
            "Shares per contract",
            min_value=MULTIPLIER[0],
            max_value=MULTIPLIER[1],
            value=DEFAULT_MULTIPLIER,
            step=1,
            key="multiplier",
            bind="query-params",
            help="100 for SPX and US equity options. P&L is for the whole position.",
        )
        st.number_input(
            "Commission per contract ($)",
            min_value=COMMISSION[0],
            max_value=COMMISSION[1],
            value=0.0,
            step=0.05,
            format=components.input_format("premium"),
            key="commission",
            bind="query-params",
            help="Charged on every option contract when the position is opened.",
        )
        risk_neutral = st.toggle(
            "Risk-neutral probabilities",
            value=True,
            key="risk_neutral",
            help="On: P(profit) from the risk-neutral distribution (the published smile under the index; a "
            "log-normal at the flat vol under the textbook market). Off: your own drift at the flat ATMF vol (view).",
        )
        if not risk_neutral:
            st.number_input("Expected annual return (%)", -100.0, 200.0, 8.0, 0.5, key="drift")
        st.toggle(
            "Finance the premium",
            value=False,
            key="carry",
            help="Charge interest on a net debit (or earn it on a credit) at the risk-free rate. Payoff diagrams "
            "conventionally leave it off.",
        )


def _width_input(bar, enabled: bool, centre: str) -> None:
    bar.number_input(
        "Width (% of strike)",
        min_value=MIN_WIDTH_PCT,
        max_value=MAX_WIDTH_PCT,
        value=None,
        step=0.5,
        format=components.input_format("moneyness", 2),
        placeholder="Auto",
        key="strategy_width_pct",
        bind="query-params",
        disabled=not enabled,
        width=150,
        help=f"Distance w between adjacent strikes (K ± w, K ± 2w, centred on {centre}) as a percentage of it, "
        "rounded to the cent (to listed strikes under the index). Blank: about half the expected move σ√T to the "
        "expiry, on a strike step of 1, 2, 2.5 or 5 times a power of ten (at most 25% of the strike).",
    )


# --- The legs across reruns ---------------------------------------------------------------------------------------


def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return lg.cells(a) == lg.cells(b)


def _listing_tag(inputs: Inputs) -> str:
    """A short tag of the listed expiries, part of the editor's key: its Expiry column lists them, and a new list would
    give the editor a new identity under its old key, which drops the edits it holds."""
    published = _published(inputs)
    if published is None:
        return "tb"
    text = "|".join(_expiry_param(e) for e in lg.listed(published))
    return hashlib.blake2s(text.encode(), digest_size=4).hexdigest()


def _editor_key(store: str, state: dict, tag: str) -> str:
    return f"legs::{store}::{state['version']}::{tag}"


def _new_legs(base: pd.DataFrame, version: int) -> dict:
    return {"base": base, "seed": base, "edited": base, "template": base, "edits": None, "version": version}


def _legs_state(store: str, base: pd.DataFrame, tag: str) -> dict:
    """The legs of template ``store`` across reruns, kept in the session so that edits are never silently lost.

    ``seed`` is the frame the editor starts from and ``base`` the template it was built from; the editor holds the
    user's edits on top of the seed. ``edited``, ``template`` and ``edits`` are the legs, the template and the
    editor's raw edits at the end of the last run. Legs that matched the template follow it when its inputs change
    (the expiry, the deltas, the width, the valuation time); legs that differed from it (edited, or moved by a
    structure stepper) are kept until they match the template again or are reset.
    """
    legs_store = st.session_state.setdefault("strategy_legs", {})
    state = legs_store.get(store)
    if state is None:
        state = legs_store[store] = _new_legs(base, 0)
        return state
    key = _editor_key(store, state, tag)
    # An edit sent with this rerun (together with a builder change) is not in ``edited`` yet: keep the legs for it.
    editing = key in st.session_state and st.session_state[key] != state["edits"]
    if not editing and _same(state["edited"], state["template"]) and not _same(state["template"], base):
        state = legs_store[store] = _new_legs(base, state["version"] + 1)
    elif key not in st.session_state and not _same(state["seed"], state["edited"]):
        # The editor was not on screen in the last run (View mode, another template, or the tab was closed), so it
        # no longer holds the edits: show the edited legs as its new starting point.
        state["seed"] = state["edited"]
    return state


def _apply_edits(frame: pd.DataFrame, edits: dict) -> pd.DataFrame:
    """The legs ``frame`` with a legs editor's edits applied, as ``st.data_editor`` applies them: cell edits by row
    position, then the positions of deleted rows, then the added rows (values by column name)."""
    rows = frame.reset_index(drop=True).to_dict("records")
    for position, changes in edits.get("edited_rows", {}).items():
        rows[int(position)].update(changes)
    deleted = {int(position) for position in edits.get("deleted_rows", [])}
    rows = [row for i, row in enumerate(rows) if i not in deleted]
    rows += [dict.fromkeys(lg.COLUMNS) | {lg.ON: True} | added for added in edits.get("added_rows", [])]
    return lg.frame(rows)


def _current_legs(choice: _Choice, tag: str) -> tuple[dict, pd.DataFrame, str]:
    """The legs state, the legs as the editor shows them in this run (its edits applied, even where it is not drawn),
    and the editor's key."""
    state = _legs_state(choice.store, choice.base, tag)
    key = _editor_key(choice.store, state, tag)
    edits = st.session_state.get(key)
    edited = state["edited"] if edits is None or edits == state["edits"] else _apply_edits(state["seed"], edits)
    state.update(edited=edited, template=choice.base, edits=copy.deepcopy(edits))
    return state, edited, key


def _price(inputs: Inputs, choice: _Choice, frame: pd.DataFrame) -> lg.Book:
    published = _published(inputs)
    name = choice.name if _same(frame, choice.base) or choice.name == CUSTOM else f"{choice.name} (modified)"
    if published is not None:
        return lg.price_index(
            frame, published, inputs.context.as_of, name=name, fees_per_contract=choice.commission,
            multiplier=choice.multiplier,
        )  # fmt: skip
    return lg.price_textbook(
        frame, inputs.market, inputs.strike, name=name, fees_per_contract=choice.commission,
        multiplier=choice.multiplier,
    )  # fmt: skip


def build_position(inputs: Inputs) -> Position | None:
    """The tab's position, built without drawing anything: the Risk tab analyses it while this tab is closed.

    It reads the builder's values and the legs from the session state, as :func:`render` would show them: legs that
    match the template follow it, edited legs are kept, and an edit sent together with the switch to another tab is
    applied and kept in the session. The app draws the controls earlier in the same run (:func:`controls`); without
    them, each control's URL parameter is read where it is valid, and its default otherwise.

    Parameters
    ----------
    inputs : Inputs
        The sidebar inputs of this run: the market the legs are valued in.

    Returns
    -------
    Position or None
        ``None`` when the template cannot be built, the legs are invalid, or no leg is on and live.

    Examples
    --------
    Without the controls, from a link (the textbook market):

    >>> from streamlit.testing.v1 import AppTest
    >>> page = AppTest.from_string('''
    ... import streamlit as st
    ... from options_analyzer.ui import inputs, tab_strategy
    ... st.write(tab_strategy.build_position(inputs.sidebar()).describe())
    ... ''')
    >>> page.query_params.update(preset="Textbook", strategy="Long Straddle", contracts="2")
    >>> page.run().markdown[0].value
    '**Long Straddle** — Long 1 × 100 Call, Long 1 × 100 Put · 2 × 100 multiplier'
    """
    try:
        choice = _choice(inputs)
    except ValueError:
        return None
    _, frame, _ = _current_legs(choice, _listing_tag(inputs))
    book = _price(inputs, choice, frame)
    if book.strategy is None or book.errors:
        return None
    return Position(book.strategy, book.market, choice.contracts, choice.multiplier, book)


# --- Structure steppers and row actions (UI-8): callbacks on the legs state ---------------------------------------


def _set_legs(store: str, change: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
    """Apply a structure change to the legs of ``store``: a new editor (version) starting from the changed legs,
    which differ from the template and so are kept until reset."""
    legs_store = st.session_state.get("strategy_legs", {})
    state = legs_store.get(store)
    if state is None:
        return
    changed = change(state["edited"])
    legs_store[store] = {**state, "seed": changed, "edited": changed, "edits": None, "version": state["version"] + 1}


def _reset_legs(store: str) -> None:
    state = st.session_state.get("strategy_legs", {}).get(store)
    if state is not None:
        st.session_state["strategy_legs"][store] = _new_legs(state["template"], state["version"] + 1)


def _on_blotter_select(legs_count: int) -> None:
    """A row clicked in the blotter selects it in the action bar (the widget route of the same choice)."""
    rows = st.session_state.get("tbl_blotter", {}).get("selection", {}).get("rows", [])
    if rows and rows[0] < legs_count:
        st.session_state["blotter_row"] = int(rows[0])


# --- Display helpers ----------------------------------------------------------------------------------------------


def _signed_money(value: float) -> str:
    """Signed dollars for a tile: cents below $1,000, whole dollars from $1,000, U+2212 for minus (``−$7,765``)."""
    if not math.isfinite(value):
        return "Unlimited" if value > 0 else f"{MINUS}Unlimited"
    return money_short(value, signed=True).replace("-", MINUS)


def _money(value: float) -> str:
    """Unsigned dollars, cents below $1,000 and whole dollars from $1,000 (spec 2.5 money)."""
    return money(value, 2 if round(abs(value), 2) < 1000 else 0)


def _level_text(x: float, spot: float) -> str:
    """A spot level to 0.1% of spot or finer: whole points on an index from 1,000 (``5,378``), else as a breakeven's
    help writes it (:func:`breakeven_decimals`: at least cents, ``85.58``; three decimals on a $1 underlying,
    ``0.962``), so that a tile, a chart label and the help agree.

    Examples
    --------
    >>> _level_text(5377.65, 5800.0), _level_text(85.58, 100.0), _level_text(0.96155, 1.0), _level_text(10.384, 10.0)
    ('5,378', '85.58', '0.962', '10.38')
    """
    return f"{x:,.{0 if spot >= 1000 else breakeven_decimals(spot)}f}"


def _pct_text(x: float) -> str:
    return signed(x * 100, 1, suffix="%")


def breakeven_decimals(spot: float) -> int:
    """Decimals that resolve a breakeven as finely as its change from spot, shown to 0.1 %: four significant
    digits of the spot and at least cents ($96.21, $4.739, $0.9621)."""
    return min(8, max(2, 3 - math.floor(math.log10(spot))))


def payoff_range(spot: float, strikes, median: float, sigma_t: float) -> tuple[float, float]:
    """x-range of the payoff chart (Range ``Auto``): the likely prices at expiry plus a margin around the strikes.

    The probability range is the median price at expiry ± 2.5 standard deviations of the log price; at very high
    volatility its upper end is capped at three times the highest strike or spot so the strikes stay readable.
    """
    anchors = [spot, *strikes]
    low, high = min(anchors), max(anchors)
    margin = 0.25 * (high - low)
    lo = min(median * np.exp(-2.5 * sigma_t), low - margin)
    hi = max(median * np.exp(2.5 * sigma_t), high + margin)
    return max(lo, 0.0), min(hi, 3.0 * high)


def expires_within_a_day(strategy: strat.Strategy, market: strat.Market) -> bool:
    """Whether an option leg of the position expires within the next calendar day."""
    has_options = any(leg.is_option for leg in strategy.legs)
    return has_options and strat.front_expiry(strategy, market) <= strat.ONE_DAY


def _short_date(day: date) -> str:
    """``18‑Dec`` (non-breaking hyphen)."""
    return f"{day:%d}{NB_HYPHEN}{day:%b}"


# --- The analysis context -----------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _Context:
    """Everything the tab's views share in one run."""

    inputs: Inputs
    choice: _Choice
    book: lg.Book
    position: Position
    published: Any
    as_of: datetime
    front_at: datetime  # the front expiry's settlement (New York wall time)
    entered: strat.Strategy  # every leg with its entry price fixed (a P&L baseline that vol shifts leave alone)
    metrics: strat.StrategyMetrics

    @property
    def units(self) -> float:
        return self.position.units

    @property
    def market(self) -> strat.Market:
        return self.book.market

    @property
    def index(self) -> bool:
        return self.published is not None

    @property
    def front(self) -> Expiry | None:
        return self.book.front if self.index else None


def _entered(strategy: strat.Strategy, market: strat.Market) -> strat.Strategy:
    """The strategy with every leg's entry price explicit (its model value where none is given), so that a P&L under
    shifted vols or rates is measured against what was paid, not against a repriced entry."""
    legs = []
    for leg in strategy.legs:
        if leg.premium is None:
            value = float(strat.position_value(strat.Strategy("leg", (replace(leg, quantity=1.0),)), market))
            leg = replace(leg, premium=max(value, 0.0))
        legs.append(leg)
    return replace(strategy, legs=tuple(legs))


def _shifted(ctx: _Context, shifts: whatif.Shifts, t: float) -> tuple[strat.Strategy, strat.Market]:
    """The entered strategy and the market under the what-if vol and rate shifts at elapsed time ``t`` (years): each
    leg's vol by :func:`.whatif.leg_vol` with its own time left; the rate + wr with the dividend yield held."""
    legs = []
    for leg in ctx.entered.legs:
        if leg.is_option:
            tau = (ctx.market.expiry if leg.expiry is None else leg.expiry) - t
            sigma = ctx.market.volatility if leg.volatility is None else leg.volatility
            leg = replace(leg, volatility=whatif.leg_vol(sigma, max(tau, 1e-8), shifts))
        legs.append(leg)
    market = replace(ctx.market, rate=ctx.market.rate + shifts.rate / 1e4)
    return replace(ctx.entered, legs=tuple(legs)), market


def _elapsed(ctx: _Context, sessions: int) -> float:
    """Years from the valuation time to session T+``sessions`` (its wall-clock time), at most the front expiry."""
    if not sessions:
        return 0.0
    t = clock.year_fraction(ctx.as_of, whatif.scenario_time(ctx.as_of, sessions))
    return min(t, ctx.market.expiry)


# --- Tiles ----------------------------------------------------------------------------------------------------------

#: The spot range over which a loss that grows without bound only through the dividend yield is read (a long
#: calendar's): the central 99.9% of the price at the front expiry.
BOUNDED_LEVEL = 0.999


def worst_loss(ctx: _Context) -> tuple[float, tuple[float, float] | None]:
    """The position's maximum loss ($, positive) at the front expiry, and the spot range it was read over (None: every
    spot).

    A long calendar or diagonal loses without bound only because a dividend yield makes its long back call gain less
    than $1 for each $1 rise: on the default index calendar the loss passes the debit near 3.4 × spot. A desk reads its
    worst case as about the debit, so this loss is the worst P&L over the central :data:`BOUNDED_LEVEL` of the price
    at the front expiry (risk-neutral from the smile under the index, log-normal at the flat vol otherwise), on 2,001
    spots and every strike inside. Any other unbounded loss (a short call) stays infinite."""
    m = ctx.metrics
    if math.isfinite(m.max_loss) or not strat.unbounded_loss_from_dividends(ctx.entered, ctx.market):
        return m.max_loss * ctx.units, None
    market, T = ctx.market, m.horizon
    if ctx.index and ctx.front is not None:
        lo, hi = ctx.published.rn_range(ctx.front, BOUNDED_LEVEL)
    else:
        sd = market.volatility * math.sqrt(T)
        median = market.spot * math.exp((market.rate - market.dividend_yield) * T - 0.5 * sd * sd)
        z = float(ndtri(0.5 + BOUNDED_LEVEL / 2))
        lo, hi = median * math.exp(-z * sd), median * math.exp(z * sd)
    strikes = [k for k in ctx.entered.strikes if lo < k < hi]
    spots = np.unique(np.concatenate([np.linspace(lo, hi, 2001), strikes]))
    values = strat.pnl(ctx.entered, market, spots, T, ctx.choice.carry) * ctx.units
    return float(-np.min(values)), (float(lo), float(hi))


def _probability(ctx: _Context) -> tuple[float, str, float, float]:
    """P(profit) at the front expiry, its basis, and the expected P&L and its standard deviation ($) under it.

    ``RN smile``: the published surface's risk-neutral CDF at the front expiry (Breeden-Litzenberger), integrated over
    the spot intervals where the expiry P&L is positive; ``view``: the user's drift at the flat ATMF vol (log-normal);
    ``flat vol``: N(d₂) at the textbook market's vol.
    """
    m, units = ctx.metrics, ctx.units
    strategy, market, carry = ctx.entered, ctx.market, ctx.choice.carry
    if not ctx.choice.risk_neutral:
        view = strat.analyze(strategy, market, ctx.choice.drift / 100.0, carry)
        return view.probability_of_profit, "view", view.expected_pnl * units, view.pnl_std * units
    if ctx.index and ctx.front is not None:
        published, key, T = ctx.published, ctx.front, m.horizon

        def at_expiry(spots: np.ndarray) -> np.ndarray:
            return strat.pnl(strategy, market, spots, T, carry) * units

        intervals = lg.profit_intervals(at_expiry, m.breakevens)
        p = lg.probability(lambda x: published.cdf(x, key), intervals)
        mean, sd = lg.rn_moments(at_expiry, lambda q: published.quantile(q, key))
        return p, "RN smile", mean, sd
    return m.probability_of_profit, "flat vol", m.expected_pnl * units, m.pnl_std * units


def _tiles(ctx: _Context) -> None:
    m, units, market = ctx.metrics, ctx.units, ctx.market
    multi = len({p.label for p in ctx.book.live if p.kind != "stock"}) > 1
    cost = m.entry_cost * units
    theo_cost = sum(p.quantity * p.theo for p in ctx.book.live) * units + ctx.entered.fees * units
    t0 = float(strat.pnl(ctx.entered, market, market.spot, 0.0)) * units
    basis = "mid" if ctx.index else "model value"
    word = "credit" if cost < -0.005 else ("debit" if cost > 0.005 else "no premium")
    premium_tile: Card = (
        "Net premium",
        money_short(abs(cost)),
        {
            "delta": word,
            "delta_color": "off",
            "delta_arrow": "off",
            "help": md(
                f"{premium(abs(m.entry_cost))} per share at {basis} ({money(abs(cost))} for the position, "
                f"commissions included). The theo {'credit' if theo_cost < 0 else 'debit'} is "
                f"{money(abs(theo_cost))}, so entry at {basis} is {signed(t0, 2, '$')} against theo: the T+0 P&L at "
                "today's spot."
            ),
        },
    )
    pkg = lg.package(ctx.book) if ctx.index else None
    if pkg is not None:
        tag = "cr" if pkg.credit else "db"
        cross = (pkg.ask - pkg.bid) / 2 * units
        package_tile: Card = (
            "Package",
            f"{premium(pkg.bid)} / {premium(pkg.ask)}",
            {
                "delta": f"{tag} · theo {premium(pkg.theo)}",
                "delta_color": "off",
                "delta_arrow": "off",
                "help": md(
                    f"The structure's natural market per share, {'selling' if pkg.credit else 'buying'} it: "
                    f"{'Σ shorts’ bids − Σ longs’ asks' if pkg.credit else 'Σ longs’ bids − Σ shorts’ asks'} to "
                    f"{'Σ shorts’ asks − Σ longs’ bids' if pkg.credit else 'Σ longs’ asks − Σ shorts’ bids'}. "
                    f"Package mid {premium(pkg.mid)} against theo {premium(pkg.theo)}: mid vs fit "
                    f"{vol_change(pkg.mid_vs_fit / 100)} in the package vega ({_signed_money(pkg.vega * units)} per "
                    f"vp). Crossing half the spread costs {_money(cross)} "
                    "per side; closing at the market rather than the mid costs the same again."
                ),
            },
        )
    else:
        reason = "no two-sided quote on every leg" if ctx.index else "the textbook market has no quotes"
        theo_pkg = sum(p.quantity * p.theo for p in ctx.book.live)
        package_tile = (
            "Package",
            MISSING,
            {
                "delta": f"theo {premium(abs(theo_pkg))}",
                "delta_color": "off",
                "delta_arrow": "off",
                "help": f"No natural market: {reason}. The theo is the legs' model value per share.",
            },
        )
    basis_front = "front expiry"
    later = " Later expiries are valued at their surface vols then; their losses can exceed it." if multi else ""
    if m.max_profit < 0:
        profit_tile: Card = (
            "Max profit",
            "No profit",
            {"help": md(f"The position cannot profit at the {basis_front}.")},
        )
    else:
        profit_tile = (
            "Max profit",
            money_short(m.max_profit * units),
            {"delta": basis_front if multi else None, "delta_color": "off", "delta_arrow": "off",
             "help": md(f"The largest P&L at the {basis_front}: {money(m.max_profit * units)}.{later}")},
        )  # fmt: skip
    loss, within = worst_loss(ctx)
    if m.max_loss < 0:
        loss_tile: Card = ("Max loss", "No loss", {"help": md(f"The position cannot lose at the {basis_front}.")})
    elif within is not None:
        word = "risk-neutral" if ctx.index else "flat-vol"
        loss_tile = (
            "Max loss",
            _signed_money(-loss),
            {"delta": f"{'RN ' if ctx.index else ''}99.9% range", "delta_color": "off", "delta_arrow": "off",
             "help": md(
                 f"The worst P&L at the {basis_front} for spot within the {word} 99.9% range to it, "
                 f"{number(within[0], 0)}–{number(within[1], 0)}: {signed(-loss, 2, '$')}. Far above that range the "
                 "loss grows without bound: with a dividend yield the long back call gains less than $1 for each $1 "
                 f"rise.{later}"
             )},
        )  # fmt: skip
    else:
        worst = signed(-loss, 2, "$") if math.isfinite(loss) else "unbounded"
        loss_tile = (
            "Max loss",
            _signed_money(-loss) if math.isfinite(loss) else "Unlimited",
            {"delta": basis_front if multi else None, "delta_color": "off", "delta_arrow": "off",
             "help": md(f"The worst P&L at the {basis_front}: {worst}.{later}")},
        )  # fmt: skip
    be = sorted(m.breakevens)
    spot = market.spot
    if not be:
        be_tile: Card = ("Breakevens", "none", {"help": "The P&L at the front expiry never changes sign."})
    else:
        nearest = sorted(sorted(be, key=lambda b: abs(b - spot))[:2])
        more = len(be) - len(nearest)
        value = " / ".join(_level_text(b, spot) for b in nearest)
        delta = " / ".join(_pct_text(b / spot - 1) for b in nearest) + (f" · +{more} more" if more else "")
        listing = ", ".join(f"{b:,.{breakeven_decimals(spot)}f} ({_pct_text(b / spot - 1)})" for b in be)
        be_tile = (
            "Breakevens",
            value,
            {"delta": delta, "delta_color": "off", "delta_arrow": "off",
             "help": f"Where the P&L at the front expiry changes sign, with the change from today's spot: {listing}."},
        )  # fmt: skip
    p, basis, mean, sd = _probability(ctx)
    prob_tile: Card = (
        "P(profit)",
        probability_text(p, sure=m.max_loss < 0, never=m.max_profit <= 0),
        {
            "delta": basis,
            "delta_color": "off",
            "delta_arrow": "off",
            "help": md(
                f"Probability that the P&L at the front expiry is positive, {_basis_text(basis)}. Under it the "
                f"expected P&L is {signed(mean, 0, '$')} and its standard deviation {_money(sd)}."
            ),
        },
    )
    card_row(
        "strategy_tiles",
        ("strategy_tiles_a", "strategy_tiles_b", "strategy_tiles_c"),
        [premium_tile, package_tile, profit_tile, loss_tile, be_tile, prob_tile],
        one_line_px=TILE_VALUE_PX,
    )


def _basis_text(basis: str) -> str:
    return {
        "RN smile": "risk-neutral from the published smile (Breeden-Litzenberger: P(S ≤ K) = 1 + ∂C/∂K ÷ DF)",
        "view": "under your drift at the flat ATMF vol (log-normal)",
        "flat vol": "risk-neutral at the flat sidebar vol (log-normal, N(d₂))",
    }[basis]


def _empty_tiles() -> None:
    labels = ("Net premium", "Package", "Max profit", "Max loss", "Breakevens", "P(profit)")
    card_row(
        "strategy_tiles",
        ("strategy_tiles_a", "strategy_tiles_b", "strategy_tiles_c"),
        [(label, MISSING, {"help": "No leg is on."}) for label in labels],
        one_line_px=TILE_VALUE_PX,
    )


# --- The primary view -----------------------------------------------------------------------------------------------


def _bands(ctx: _Context) -> list[charts_strategy.Band]:
    """The 68% and 95% ranges of the price at the front expiry: from the published smile's CDF under the index
    (UI-13), log-normal at the flat vol under the textbook market."""
    out = []
    market, T = ctx.market, ctx.metrics.horizon
    if ctx.index and ctx.front is not None:
        to = _short_date(ctx.front.day)
        for level, word in ((ONE_SD, "68%"), (TWO_SD, "95%")):
            lo, hi = ctx.published.rn_range(ctx.front, level)
            out.append(charts_strategy.Band(lo, hi, f"RN {word} to {to}: {number(lo, 0)}–{number(hi, 0)} (smile)"))
        return out
    sd = market.volatility * math.sqrt(T)
    median = market.spot * math.exp((market.rate - market.dividend_yield) * T - 0.5 * sd * sd)
    for n, word in ((1.0, "68%"), (2.0, "95%")):
        lo, hi = median * math.exp(-n * sd), median * math.exp(n * sd)
        text = _level_text(lo, market.spot), _level_text(hi, market.spot)
        out.append(charts_strategy.Band(lo, hi, f"{word} at expiry: {text[0]}–{text[1]} (flat vol)"))
    return out


def _x_range(ctx: _Context, choice: str, bands: list[charts_strategy.Band]) -> tuple[float, float]:
    market, strategy = ctx.market, ctx.entered
    spot = market.spot
    if choice in ("±5%", "±10%", "±20%"):
        x = float(choice.strip("±%")) / 100
        return spot * (1 - x), spot * (1 + x)
    if choice == "RN 95%" and bands:  # the range itself (UI-12), the smile's under the index
        return bands[1].low, bands[1].high
    sigma_t = market.volatility * math.sqrt(ctx.metrics.horizon)
    forward = spot * math.exp((market.rate - market.dividend_yield) * ctx.metrics.horizon)
    return payoff_range(spot, strategy.strikes, forward * math.exp(-0.5 * sigma_t**2), sigma_t)


def _curve_name(shifts: whatif.Shifts, base: str) -> str:
    """A curve's legend name with the what-if vol and rate shifts it carries: ``T+0 · vol +3.00 vp``."""
    parts = [base]
    if shifts.vol:
        if shifts.mode == "absolute":
            parts.append(f"vol {vol_change(shifts.vol / 100)}")
        else:
            parts.append(f"vol {signed(shifts.vol, 1)}%{' 1M' if shifts.mode == '1M-scaled' else ''}")
    if shifts.rate:
        parts.append(f"rate {signed(shifts.rate, 0)} bp")
    return " · ".join(parts)


def _session_text(ctx: _Context, n: int) -> str:
    day = whatif.session_dates(ctx.as_of.date(), n)[-1]
    return f"T+{n} · {_short_date(day)}"


def _strip_values(ctx: _Context, kind: str, strategy, market, spots, t) -> tuple[np.ndarray, str]:
    raw = strat.position_greeks(strategy, market, spots, t)
    units = ctx.units
    if kind == "Δ":
        return np.asarray(raw["delta"]) * spots * units, "$Δ ($)"
    if kind == "Γ":
        return 0.5 * np.asarray(raw["gamma"]) * (0.01 * spots) ** 2 * units, "Γ P&L 1% ($)"
    if kind == "Vega":
        return np.broadcast_to(np.asarray(raw["vega"]) / 100 * units, spots.shape).copy(), "Vega ($/vp)"
    return np.broadcast_to(np.asarray(raw["theta"]) / 365 * units, spots.shape).copy(), "Θ ($/day)"


def _chart(ctx: _Context, shifts: whatif.Shifts, range_choice: str, dated: list[str], greek: str) -> pd.DataFrame:
    market, units, carry = ctx.market, ctx.units, ctx.choice.carry
    m = ctx.metrics
    bands = _bands(ctx)
    lo, hi = _x_range(ctx, range_choice, bands)
    spots = np.unique(
        np.concatenate([np.linspace(lo, hi, GRID_POINTS), [k for k in ctx.entered.strikes if lo <= k <= hi]])
    )
    T = m.horizon
    front_text = _short_date(ctx.front.day) if ctx.front is not None else f"{number(T * 365, 0)}d"
    curves = [
        charts_strategy.Curve(
            strat.pnl(ctx.entered, market, spots, T, carry) * units, f"Expiry {front_text}", "expiry",
            f"exp {front_text}",
        )
    ]  # fmt: skip
    t0 = _elapsed(ctx, shifts.sessions)
    shifted, shifted_market = _shifted(ctx, shifts, t0)
    t0_name = "T+0" if not shifts.sessions else _session_text(ctx, shifts.sessions)
    t0_values = strat.pnl(shifted, shifted_market, spots, t0, carry) * units
    curves.append(charts_strategy.Curve(t0_values, _curve_name(shifts, t0_name), "t0", t0_name.split(" · ")[0]))
    sessions = whatif.sessions_to(ctx.as_of.date(), ctx.front_at.date())
    for choice in dated:
        n = round(DATED[choice] * sessions)
        t = _elapsed(ctx, n)
        if n <= 0 or n == shifts.sessions or t >= T:
            continue
        values = strat.pnl(*_shifted(ctx, shifts, t), spots, t, carry) * units
        text = _session_text(ctx, n)
        curves.append(charts_strategy.Curve(values, _curve_name(shifts, text), "dated", text))
    strip = _strip_values(ctx, greek, shifted, shifted_market, spots, t0)
    spot = market.spot
    breakevens = [(b, f"BE {_level_text(b, spot)} ({_pct_text(b / spot - 1)})") for b in m.breakevens if lo <= b <= hi]
    straddle = None
    forward = None
    increment = math.nan  # a textbook strike shows the decimals it has
    if ctx.index and ctx.front is not None:
        forward = ctx.published.forward(ctx.front)
        increment = ctx.inputs.context.strike_step
        move = lg.straddle_move(ctx.published, ctx.front)
        if move is not None:
            straddle = (spot * (1 - move[0]), spot * (1 + move[0]), f"straddle ±{pct(move[0], 2)}")
    scenario = None
    if shifts.spot:
        x = spot * (1 + shifts.spot / 100)
        y = float(strat.pnl(shifted, shifted_market, x, t0, carry)) * units
        scenario = (x, y, f"Scenario {_level_text(x, spot)}: {_signed_money(y)}")
    note = " · ".join(
        part
        for part in (
            ctx.front and lg.expiry_text(*ctx.front),
            _days_left(ctx.as_of, ctx.front_at.date(), clock.year_fraction(ctx.as_of, ctx.front_at))
            if ctx.index
            else "",
            "RN smile" if ctx.index else "flat vol",
        )
        if part
    )
    fig = charts_strategy.payoff_figure(
        spots,
        curves,
        strip,
        spot=spot,
        forward=forward,
        strikes=ctx.entered.strikes,
        breakevens=breakevens,
        bands=bands,
        straddle=straddle,
        scenario=scenario,
        note=note,
        x_range=(lo, hi),
        strike_text=lambda k: strike(k, increment),
    )
    charts.show(
        fig,
        key="payoff_chart",
        alt="Profit and loss of the position against spot at the front expiry, today and on later dates, with its "
        "Greek strip, strikes, breakevens and risk-neutral ranges",
    )
    return pd.DataFrame({"Spot": spots} | {c.name: c.values for c in curves} | {strip[1]: strip[0]})


def _download(ctx: _Context, profile: pd.DataFrame, where) -> None:
    """The P&L profile's download (built on click), with its provenance (spec 2.13)."""
    provenance = exports.provenance(ctx.inputs, model="BSM · European legs")
    where.download_button(
        "P&L profile (CSV)",
        exports.builder(exports.csv, profile, provenance),
        provenance.file_name(f"{ctx.entered.name.lower().replace(' ', '_')}_pnl", "csv"),
        "text/csv",
        on_click="ignore",
        icon=":material/download:",
        key="dl_pnl",
        help="The chart's curves and Greek strip against spot, with the source and valuation time in its first lines.",
    )


def table_dates(ctx: _Context) -> list[tuple[int | None, date]]:
    """The table's columns: trading sessions from today to the front expiry, weekly (the last session of each calendar
    week: a Friday, or the session before a Friday holiday) then the last five daily, and the expiry itself (``None``
    sessions): ``(T+N, date)``. Every fifth session drifted to Mondays after the Thanksgiving holiday."""
    sessions = whatif.sessions_to(ctx.as_of.date(), ctx.front_at.date())
    days = whatif.session_dates(ctx.as_of.date(), sessions)
    last = [n for n in range(sessions + 1) if days[n] < ctx.front_at.date() or n < sessions]
    daily = last[-5:] if last else []
    week_end: dict[tuple[int, int], int] = {}
    for n in last:
        week_end[days[n].isocalendar()[:2]] = n  # the sessions are in order: the last one of each week stays
    weekly = [n for n in week_end.values() if not daily or n < daily[0]]
    picked = sorted(set(weekly) | set(daily) | {0})
    return [(n, days[n]) for n in picked if _elapsed(ctx, n) < ctx.metrics.horizon] + [(None, ctx.front_at.date())]


def table_spots(ctx: _Context, rows: int = 15) -> np.ndarray:
    """The table's spot levels: on a round step (1, 2, 2.5 or 5 × a power of ten, at least the strike interval) that
    spans the 95% range to the front expiry in about ``rows`` rows, through today's spot."""
    bands = _bands(ctx)
    spot = ctx.market.spot
    lo, hi = bands[1].low, bands[1].high
    step = _round_step((hi - lo) / (rows - 1))
    if ctx.index:
        step = max(step, ctx.inputs.context.strike_step)
    below = np.arange(np.floor(spot / step) * step, lo - step, -step)
    above = np.arange(np.ceil(spot / step) * step, hi + step, step)
    levels = np.unique(np.concatenate([below, above, [spot]]))
    return levels[levels > 0]


def _table(ctx: _Context, shifts: whatif.Shifts, unit: str) -> None:
    units, carry, market = ctx.units, ctx.choice.carry, ctx.market
    spots = table_spots(ctx)
    columns, values = [], []
    for n, day in table_dates(ctx):
        if n is None:
            column = strat.pnl(ctx.entered, market, spots, ctx.metrics.horizon, carry) * units
            columns.append(f"Exp<br>{_short_date(day)}")
        else:
            t = _elapsed(ctx, n)
            column = strat.pnl(*_shifted(ctx, shifts, t), spots, t, carry) * units
            columns.append(f"{day:%a}<br>{_short_date(day)}")
        values.append(column)
    grid = np.array(values).T  # rows: spots
    if unit == "% of max loss":
        scale = worst_loss(ctx)[0]
        grid = grid / scale * 100 if math.isfinite(scale) and scale > 0 else grid * np.nan
    elif unit == "% of premium":
        scale = abs(ctx.metrics.entry_cost) * units
        grid = grid / scale * 100 if scale > 0 else grid * np.nan
    spot = market.spot
    labels = [f"{_level_text(s, spot)} ({_pct_text(s / spot - 1)})" for s in spots]
    row = int(np.argmin(np.abs(spots - spot)))
    fig = charts_strategy.pnl_table_figure(columns, labels, grid, spot_row=row, unit=unit)
    charts.show(
        fig,
        key="chart_pnl_table",
        alt="Profit and loss of the position by date (columns) and spot (rows), from today to the front expiry",
    )


# --- The blotter --------------------------------------------------------------------------------------------------


def blotter_frame(ctx: _Context) -> pd.DataFrame:
    """The blotter's rows (spec 3.2 item 5): one per leg, a subtotal per expiry when the legs have more than one, and
    the Total. Greeks are the whole position's in desk units ($Δ, Γ P&L 1%, vega $/vp, Θ $/day) with the one-day
    decay by repricing beside theta; a leg that is off or expired shows its market but no Greeks."""
    market, units = ctx.market, ctx.units
    rows = []
    expiring = expires_within_a_day(ctx.entered, market)
    for p in ctx.book.legs:
        quote = p.quote
        greeks: dict[str, float] = dict.fromkeys(("$Δ", "Γ P&L 1%", "Vega", "Θ", "Decay"), math.nan)
        if p.leg is not None:
            single = strat.Strategy("leg", (p.leg,))
            risk = strat.position_risk(single, market, multiplier=units)
            greeks = {
                "$Δ": risk.dollar_delta,
                "Γ P&L 1%": risk.dollar_gamma_1pct,
                "Vega": risk.vega,
                "Θ": risk.theta,
                "Decay": strat.time_decay(single, market, multiplier=units),
            }
        mid_vs_fit = (quote.iv_mid - p.fit) * 100 if quote is not None and np.isfinite(quote.iv_mid) else math.nan
        rows.append(
            {
                "On": "on" if p.on and not p.expired else ("expired" if p.expired else "off"),
                "Qty": p.quantity,
                "Type": p.kind.capitalize(),
                "Expiry": p.label,
                "Strike": p.strike if p.strike is not None else math.nan,
                "Bid": quote.bid if quote is not None else math.nan,
                "Ask": quote.ask if quote is not None else math.nan,
                "Mid": quote.mid if quote is not None and quote.bid > 0 else (p.theo if p.expired else math.nan),
                "IV": p.vol * 100 if p.kind != "stock" and math.isfinite(p.vol) else math.nan,
                "IV src": p.src or MISSING,
                "Mid vs fit": mid_vs_fit,
                **greeks,
                "_label": p.label,
                "_live": p.leg is not None,
            }
        )
    frame = pd.DataFrame(rows)
    sums = ["$Δ", "Γ P&L 1%", "Vega", "Θ", "Decay"]
    extra = []
    live = frame[frame["_live"]] if len(frame) else frame
    expiries = [e for e in dict.fromkeys(live["_label"]) if e] if len(live) else []
    if len(expiries) > 1:
        for e in expiries:
            part = live[live["_label"] == e]
            extra.append({"On": f"Σ {e.rsplit(' ', 1)[0]}", **{c: part[c].sum() for c in sums}})
    extra.append({"On": "Total", **{c: (live[c].sum() if len(live) else math.nan) for c in sums}})
    out = pd.concat([frame, pd.DataFrame(extra)], ignore_index=True).drop(columns=["_label", "_live"])
    out.attrs["decay_label"] = decay_label(expiring)
    return out


#: The blotter's columns: kind (the column registry) and help.
BLOTTER_KINDS = {
    "On": "text", "Qty": "quantity", "Type": "text", "Expiry": "text", "Strike": "strike", "Bid": "premium",
    "Ask": "premium", "Mid": "premium", "IV": "vol", "IV src": "text", "Mid vs fit": "vol_change", "$Δ": "dollar_delta",
    "Γ P&L 1%": "gamma_pnl", "Vega": "vega", "Θ": "theta", "Decay": "gamma_pnl",
}  # fmt: skip
BLOTTER_HELP = {
    "On": "on: in every number. off: kept in the blotter but out of every number (Edit to switch). expired: settled at "
    "or before the valuation time, out of every number. Σ: a subtotal per expiry; Total: the position.",
    "Qty": "Signed contracts per position unit: positive long, negative short.",
    "Type": "Call or put (or shares under the textbook market).",
    "Expiry": "The listed expiry and its settlement (AM at the opening prints, PM at the close); days to expiry "
    "under the textbook market.",
    "Strike": "The leg's strike.",
    "Bid": "The chain's bid per share (synthetic, not market data).",
    "Ask": "The chain's ask per share (synthetic, not market data).",
    "Mid": "The chain's mid per share; an expired leg's intrinsic value at today's spot.",
    "IV": "The vol the leg is valued at: the published surface's at its strike, or your override.",
    "IV src": "surface: the published fit; extrap.: beyond the expiry's quotes (Lee-bounded wing); override: typed; "
    "unlisted: a date the chain no longer lists, from the surface in total variance; flat: the sidebar vol.",
    "Mid vs fit": "σ_mid − σ_fit in vol points: positive means the market is rich against the fit.",
    "$Δ": "Dollar delta of the whole leg: Δ × S × units (sticky strike).",
    "Γ P&L 1%": "Convexity P&L of a 1% move: ½ Γ (0.01 S)² × units.",
    "Vega": "$ per vol point, whole leg.",
    "Θ": "$ per calendar day: θ ÷ 365 at fixed vols, whole leg.",
    "Decay": "The change in value over the next calendar day by repricing, spot and vols unchanged (a leg expiring "
    "within the day at its intrinsic value): beside theta, which is the instantaneous rate.",
}


#: The blotter's dollar Greeks: whole dollars (spec 2.5: −$68, −$306, +$41) unless every value of the column is below
#: this many dollars, when they show cents (a one-lot on the textbook's $100 underlying has a theta of $2 a day).
CENTS_BELOW = 10.0
#: The blotter's columns that hold a quote; the textbook market has none, so it leaves them out.
QUOTE_COLUMNS = ("Bid", "Ask", "Mid", "Mid vs fit")


def strike_decimals(strikes: pd.Series, most: int = 6) -> int:
    """The fewest decimals (at most ``most``) that write every strike exactly: a textbook strike shows the decimals it
    has (``0.975``, ``1,234,567.89``), as the legs editor does.

    >>> strike_decimals(pd.Series([0.95, 0.975, 1.025, float("nan")])), strike_decimals(pd.Series([100.0, 110.0]))
    (3, 0)
    """
    values = pd.to_numeric(strikes, errors="coerce").dropna().to_numpy(float)
    for places in range(most + 1):
        if np.allclose(values, np.round(values, places), rtol=0, atol=1e-9):
            return places
    return most


def blotter_digits(frame: pd.DataFrame, index: bool) -> dict[str, int]:
    """The blotter's precisions other than its kinds': listed strikes whole under the index (textbook strikes with
    the decimals they have); each dollar Greek in cents where its largest value rounds below :data:`CENTS_BELOW`
    dollars, else whole dollars.

    Examples
    --------
    >>> frame = pd.DataFrame({"Strike": [5300.0, 6250.0], "$Δ": [-18870.7, 63.68], "Γ P&L 1%": [-0.64, 0.4],
    ...                       "Vega": [9.96, -9.2], "Θ": [float("nan"), 1.5], "Decay": [0.0, 0.0]})
    >>> blotter_digits(frame, index=True)
    {'Strike': 0, 'Γ P&L 1%': 2, 'Θ': 2, 'Decay': 2}
    """
    digits = {"Strike": 0} if index else {"Strike": strike_decimals(frame["Strike"])}
    for column in ("$Δ", "Γ P&L 1%", "Vega", "Θ", "Decay"):
        values = pd.to_numeric(frame[column], errors="coerce").abs()
        if not (values.round(0) >= CENTS_BELOW).any():
            digits[column] = 2
    return digits


def _blotter_view(ctx: _Context, frame: pd.DataFrame) -> None:
    decay = frame.attrs.get("decay_label", "1-day decay")
    labels = {"Decay": "1d decay ($)" if decay == "1-day decay" else "To expiry ($)"}
    if not ctx.index:
        frame = frame.drop(columns=list(QUOTE_COLUMNS))
    digits = blotter_digits(frame, ctx.index)
    table(
        frame,
        BLOTTER_KINDS,
        BLOTTER_HELP,
        labels=labels,
        digits=digits,
        key="tbl_blotter",
        on_select=lambda: _on_blotter_select(len(ctx.book.legs)),
        selection_mode="single-row",
    )


def _editor_config(index: bool, published) -> dict:
    """The legs editor's columns (spec 3.2 item 5, Edit). Static for a listing, so that the editor's identity does
    not change while the user edits."""
    config = {
        lg.ON: st.column_config.CheckboxColumn("On", help="Off: kept here but out of every number.", default=True),
        lg.QTY: components.col("quantity", "Qty", help="Signed: positive long, negative short.", step=1, required=True),
        # Put and Call, as the blotter's View writes them.
        lg.TYPE: st.column_config.SelectboxColumn(
            "Type",
            options=list(lg.INDEX_KINDS if index else strat.LEG_KINDS),
            required=True,
            format_func=str.capitalize,
        ),
        lg.ENTRY: components.col(
            "premium",
            "Entry",
            min_value=0.0,
            help="Your fill per share. Blank: the mid (the model value where there is no market).",
        ),
    }
    if index:
        config[lg.EXPIRY] = st.column_config.SelectboxColumn(
            "Expiry", options=[lg.expiry_text(*e) for e in lg.listed(published)], required=True,
            help="A listed expiry: AM settles at the opening prints, PM at the close.",
        )  # fmt: skip
        config[lg.STRIKE] = components.col(
            "strike",
            "Strike",
            min_value=1.0,
            step=5.0,
            digits=0,
            help="A listed strike: every 5 points near the money, every 50 further out. A strike the chain does not "
            "list has no quote: it takes the surface vol there and enters at its model value.",
        )
        config[lg.IV] = components.col(
            "vol", "IV override", min_value=0.1, max_value=500.0, help="Blank: the published surface."
        )
    else:
        config[lg.DAYS] = components.col(
            "days", "Days", min_value=1, max_value=3650, step=1, help="Blank: the sidebar days to expiry."
        )
        config[lg.STRIKE] = st.column_config.NumberColumn(
            "Strike", min_value=0.0001, format=STRIKE_FORMAT, help="Blank: the sidebar strike. Ignored for shares."
        )
        config[lg.IV] = components.col(
            "vol", "IV", min_value=0.1, max_value=500.0, help="Blank: the sidebar volatility."
        )
    return config


def _steppers(ctx: _Context, grid: lg.Grid, row, frame: pd.DataFrame) -> None:
    """The structure steppers (UI-8): Shift, Width, Roll and Flip, small buttons in the blotter's head row. The
    delta-hedge ticket waits for futures legs (roadmap RK-11)."""
    store = ctx.choice.store
    forward = ctx.published.forward(ctx.front) if ctx.index and ctx.front is not None else ctx.market.spot
    roll_help = "" if ctx.index else " The textbook market lists no expiries."
    no_wings = not lg.wings(frame, forward)
    width_help = " This structure has no wings to move (a straddle or a calendar has one strike)." if no_wings else ""
    for key, label, change, help_text, disabled in (
        ("step_shift_down", "Shift −", lambda f: lg.shift(f, -1, grid), "Every strike one listed interval lower.",
         False),
        ("step_shift_up", "Shift +", lambda f: lg.shift(f, 1, grid), "Every strike one listed interval higher.", False),
        ("step_width_in", "Width −", lambda f: lg.widen(f, -1, grid, forward),
         "The wings one listed interval in, never onto the body: the outer strikes held the other way from the "
         "strikes inside them (a strangle's two strikes; a vertical's far strike)." + width_help, no_wings),
        ("step_width_out", "Width +", lambda f: lg.widen(f, 1, grid, forward),
         "The wings one listed interval out." + width_help, no_wings),
        ("step_roll_back", "Roll ←", lambda f: lg.roll(f, -1, grid),
         "Every leg to the previous listed expiry." + roll_help, not ctx.index),
        ("step_roll_next", "Roll →", lambda f: lg.roll(f, 1, grid), "Every leg to the next listed expiry." + roll_help,
         not ctx.index),
        ("step_flip", "Flip", lg.flip, "Every quantity negated: long becomes short.", False),
    ):  # fmt: skip
        row.button(
            label,
            key=key,
            on_click=_set_legs,
            args=(store, change),
            help=help_text,
            disabled=disabled,
        )


def _row_actions(ctx: _Context, grid: lg.Grid) -> None:
    """The action bar over the selected row (spec 2.8): a leg menu, the widget route of a click on a blotter row, and
    Flip, Duplicate, Roll → and Remove."""
    legs = ctx.book.legs
    if not legs:
        return
    store = ctx.choice.store
    if st.session_state.get("blotter_row", 0) >= len(legs):
        st.session_state["blotter_row"] = 0
    bar = st.container(horizontal=True, key="strategy_row_actions", gap="small", vertical_alignment="bottom")
    chosen = bar.selectbox(
        "Selected leg",
        range(len(legs)),
        format_func=lambda i: f"{i + 1} · {_leg_short(legs[i])}",
        key="blotter_row",
        width=260,
        help="The leg the actions apply to; a click on a blotter row selects it too.",
    )
    i = legs[int(chosen)].row
    for key, label, change, help_text, disabled in (
        ("row_flip", "Flip leg", lambda f: lg.flip(f, [i]), "This leg's quantity negated.", False),
        ("row_duplicate", "Duplicate", lambda f: lg.duplicate(f, i), "A copy of this leg after it.", False),
        ("row_roll", "Roll leg →", lambda f: lg.roll(f, 1, grid, [i]), "This leg to the next listed expiry.",
         not ctx.index),
        ("row_remove", "Remove", lambda f: lg.remove(f, i), "This leg out of the position.", len(legs) < 2),
    ):  # fmt: skip
        bar.button(
            label,
            key=key,
            on_click=_set_legs,
            args=(store, change),
            help=help_text,
            disabled=disabled,
        )


# --- The tab ------------------------------------------------------------------------------------------------------


def render(inputs: Inputs) -> None:
    """Draw the rest of the tab, below :func:`controls`: tiles, what-if strip, primary view, blotter and footer."""
    try:
        choice = _choice(inputs)
    except ValueError as exc:
        st.error(str(exc))
        return
    tag = _listing_tag(inputs)
    state, frame, editor_key = _current_legs(choice, tag)
    book = _price(inputs, choice, frame)
    published = _published(inputs)
    index = published is not None
    as_of = inputs.context.as_of if index else mk.last_close()
    if book.strategy is None:
        _empty_tiles()
        st.info("No legs are on. Pick a template or switch a leg on in the blotter.")
        _blotter(None, inputs, choice, state, frame, editor_key, book, None)
        return
    position = Position(book.strategy, book.market, choice.contracts, choice.multiplier, book)
    entered = _entered(book.strategy, book.market)
    metrics = strat.analyze(entered, book.market, None, choice.carry)
    if index and book.front is not None:
        front_at = clock.to_new_york(clock.settlement_time(book.front.day, book.front.settlement)).replace(tzinfo=None)
    else:
        front_at = as_of + timedelta(days=metrics.horizon * 365)
    ctx = _Context(inputs, choice, book, position, published, as_of, front_at, entered, metrics)
    _tiles(ctx)
    atm = book.market.volatility
    shifts = whatif.strip(
        spot=book.market.spot,
        atm_vol=atm,
        tenor=metrics.horizon,
        rate=book.market.rate,
        as_of=as_of,
        horizon=front_at,
    )
    view_bar = st.container(horizontal=True, key="strategy_view", gap="small", vertical_alignment="bottom", wrap=True)
    view = view_bar.segmented_control(
        "View", VIEWS, default="Chart", required=True, key="st_view", bind="query-params", persist_state="session",
        label_visibility="collapsed",
    )  # fmt: skip
    if view == "Chart":
        range_choice = view_bar.segmented_control(
            "Range", RANGES, default="Auto", required=True, key="st_range", bind="query-params",
            persist_state="session", label_visibility="collapsed",
            help="The spot axis: a move either side of today's spot, the 95% risk-neutral range to the front expiry, "
            "or Auto (the likely prices at expiry and every strike).",
        )  # fmt: skip
        dated = view_bar.pills(
            "Dated curves", list(DATED), selection_mode="multi", default=list(DATED), key="st_dates",
            persist_state="session", label_visibility="collapsed",
            help="P&L curves at a third and two thirds of the trading sessions to the front expiry.",
        )  # fmt: skip
        greek = view_bar.segmented_control(
            "Greek strip", GREEKS, default="Δ", required=True, key="st_greek", bind="query-params",
            persist_state="session", label_visibility="collapsed",
            help="The Greek strip under the chart, at the T+0 date and shifts: $Δ, Γ P&L 1%, vega ($/vp) or Θ "
            "($/day), whole position.",
        )  # fmt: skip
        profile = _chart(ctx, shifts, str(range_choice), list(dated or []), str(greek))
    else:
        unit = view_bar.segmented_control(
            "Units", UNITS, default="$", required=True, key="st_units", bind="query-params", persist_state="session",
            label_visibility="collapsed",
            help="Cell values: dollars, percent of the maximum loss, or percent of the net premium.",
        )  # fmt: skip
        _table(ctx, shifts, str(unit))
        profile = None
    _blotter(ctx, inputs, choice, state, frame, editor_key, book, profile)
    _footer(ctx)


def _blotter(ctx, inputs, choice, state, frame, editor_key, book, profile) -> None:
    published = _published(inputs)
    index = published is not None
    grid = lg.Grid(published, step=_width(inputs, None) if not index else 1.0)
    head = st.container(horizontal=True, key="strategy_blotter_head", gap="small", vertical_alignment="bottom",
                        wrap=True)  # fmt: skip
    mode = head.segmented_control(
        "Blotter", BLOTTER_MODES, default="View", required=True, key="st_blotter", persist_state="session",
        label_visibility="collapsed",
    )  # fmt: skip
    if ctx is not None:
        _steppers(ctx, grid, head, frame)
        if profile is not None:
            _download(ctx, profile, head)
    for message in book.errors:
        st.error(message)
    if choice.name != CUSTOM and not _same(frame, choice.base):
        st.caption(
            "Edited legs: kept as they are while the builder or the market changes. Reset legs to template (Edit) "
            "rebuilds them."
        )
    if mode == "Edit" or ctx is None:
        columns = [lg.ON, lg.QTY, lg.TYPE, lg.EXPIRY if index else lg.DAYS, lg.STRIKE, lg.IV, lg.ENTRY]
        edited = st.data_editor(
            state["seed"],
            key=editor_key,
            num_rows="dynamic",
            hide_index=True,
            column_order=columns,
            column_config=_editor_config(index, published),
            row_height=32,
            placeholder="",  # a blank cell follows the surface, the mid or the sidebar (its help says which)
        )
        state.update(edited=lg.frame(edited.to_dict("records")), edits=copy.deepcopy(st.session_state.get(editor_key)))
        st.button(
            "Reset legs to template", key="reset_legs", on_click=_reset_legs, args=(choice.store,),
            icon=":material/restart_alt:",
        )  # fmt: skip
        return
    _row_actions(ctx, grid)
    _blotter_view(ctx, blotter_frame(ctx))


def _footer(ctx: _Context) -> None:
    if ctx.index:
        when = _short_date(ctx.front.day) if ctx.front is not None else "the front expiry"
        basis = "RN from the smile" if ctx.choice.risk_neutral else "view (your drift, flat ATMF vol)"
        parts = [
            "BSM European legs at surface IVs",
            "sticky strike",
            "entry at mid",
            f"P(profit) {basis} at {when}",
            "expired legs out of every number",
        ]
    else:
        basis = "flat vol (log-normal)" if ctx.choice.risk_neutral else "view (your drift)"
        parts = ["BSM European legs at the sidebar vol", "entry at model value", f"P(profit) {basis}"]
    if ctx.entered.fees:
        parts.append(f"commissions {_money(ctx.entered.fees * ctx.units)} included")
    if ctx.choice.carry:
        parts.append("premium financed at r")
    st.caption(" · ".join(parts))
    if strat.unbounded_loss_from_dividends(ctx.entered, ctx.market):
        st.caption(
            "With a dividend yield the long back call gains less than \\$1 for each \\$1 rise, so far above the "
            "strikes the loss is unbounded."
        )
    if not ctx.index:
        _valuation_note(ctx.inputs, ctx.entered)


def _valuation_note(inputs: Inputs, strategy: strat.Strategy) -> None:
    """Under an American engine in the sidebar (the textbook market's ALO, or a tree from a caller), what early
    exercise adds to the sidebar contract: the legs here are European Black-Scholes-Merton values. The ALO premium is
    the engine's American value less the closed form; a tree's is its American less its European price on one lattice,
    so that the tree's discretisation error cancels."""
    model = inputs.model
    if not model.is_american:
        return
    kinds = sorted(
        {
            leg.kind
            for leg in strategy.legs
            if leg.is_option
            and np.isclose(leg.strike, inputs.strike, rtol=1e-12, atol=0)
            and (leg.expiry is None or np.isclose(leg.expiry, inputs.expiry, rtol=1e-12, atol=0))
        }
    )
    S, K, T, r, sigma, q = (
        inputs.spot,
        inputs.strike,
        inputs.expiry,
        inputs.rate,
        inputs.volatility,
        inputs.dividend_yield,
    )
    if not kinds:
        st.caption(
            "The sidebar's American engine prices the Pricing and IV tabs only: early exercise is not modelled here."
        )
        return
    try:
        if model.kind == "alo":
            where = "with the American engine (ALO)"
            values = {
                k: (cache.alo_value(S, K, T, r, sigma, q, k, model.scheme), bs.price(S, K, T, r, sigma, q, k))
                for k in kinds
            }
        else:
            where = "on this tree"
            trees = {
                style: cache.tree_results(S, K, T, r, sigma, q, model.steps, style == "American", model.tree_method)
                for style in ("American", "European")
            }
            values = {k: (trees["American"][k].price, trees["European"][k].price) for k in kinds}
    except ValueError:
        return
    sentences = []
    for kind in kinds:
        american, european = values[kind]
        sentence = (
            f"For one {strat.strike_text(K)} {kind} (the sidebar contract), early exercise adds "
            f"{money(american - european, 4)} per share {where}: {money(american, 4)} American against "
            f"{money(european, 4)} European."
        )
        if model.kind != "alo":
            sentence += (
                f" The strategy values it at {money(bs.price(S, K, T, r, sigma, q, kind), 4)}, its European "
                "Black-Scholes-Merton value."
            )
        sentences.append(sentence)
    st.info(
        md("**Early exercise is not modelled here:** the legs are European BSM values. " + " ".join(sentences)),
        icon=":material/info:",
    )
