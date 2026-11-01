"""Sidebar: the market the app values against, with the pricing-model section from :mod:`.model_config`.

The Market menu (``preset``) picks one of two contexts (:mod:`.market`):

* **Synthetic index** (the default): a valuation time, then a contract on the synthetic SPX chain: a listed expiry,
  a strike on the listed interval and the volatility of the fitted surface (or an override). Spot, forward,
  discount factor and time to expiry are the chain's, read from its quotes, and are shown, not typed.
* **Textbook**: spot, strike, days to expiry, volatility, rate and dividend yield, typed (version 2's inputs).

Either way :func:`sidebar` returns one :class:`Inputs`, which every tab reads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime

import streamlit as st

from .. import clock
from ..market_data.synthetic import strike_interval
from ..strategies import Market
from . import market as mk
from . import surface_state
from .format import date as short_date
from .format import level, moneyness, pct
from .format import strike as strike_text
from .model_config import ModelConfig, model_sidebar  # code that imports ModelConfig from here keeps working

#: The valuation times the index preset accepts (New York wall time): the clock's calendar runs 2000 to 2100, and
#: the chain lists expiries up to three years out.
ASOF_RANGE = (datetime(2000, 1, 3, 0, 0), datetime(2097, 12, 31, 23, 55))
#: The textbook inputs (version 2's defaults), by widget key.
TEXTBOOK = {"spot": 100.0, "strike": 100.0, "days": 365, "vol": 20.0, "rate": 5.0, "div": 0.0}
#: The range each textbook input accepts, by widget key (the link reader checks a link's values against it).
TEXTBOOK_LIMITS = {
    "spot": (0.01, 1e7),
    "strike": (0.01, 1e7),
    "days": (1, 3650),
    "vol": (0.1, 500.0),
    "rate": (-5.0, 50.0),
    "div": (0.0, 50.0),
}
#: What a link note calls each textbook input.
TEXTBOOK_NAMES = {
    "spot": "spot price",
    "strike": "strike",
    "days": "days to expiry",
    "vol": "volatility",
    "rate": "risk-free rate",
    "div": "dividend yield",
}
#: Vol sources of the index preset's contract.
SURFACE, OVERRIDE = "Surface", "Override"
#: Session-state key: notes from the sidebar (an expiry moved to a listed one), shown once above the tabs.
_NOTES = "market_notes"


@dataclass(frozen=True)
class Inputs:
    """The market and contract that every tab values.

    Under the textbook preset the fields are the typed inputs and the time to expiry is ``days / 365``. Under the
    index preset ``spot`` is the chain's; ``rate`` and ``dividend_yield`` are the carry of the expiry's parity
    forward and discount factor (:func:`.market.carry`), so Black-Scholes-Merton with them is Black-76 on that
    forward; ``maturity`` runs to the expiry's settlement (minutes / 525,600) and ``days`` is ``maturity × 365``.
    """

    spot: float
    strike: float
    days: float  # calendar days to expiry
    volatility: float  # decimal
    rate: float  # decimal
    dividend_yield: float  # decimal
    model: ModelConfig
    preset: str = mk.TEXTBOOK
    context: mk.MarketContext | None = None
    expiry_date: date | None = None  # the listed expiry (index preset)
    settlement: str | None = None  # "AM" or "PM" (index preset)
    maturity: float | None = None  # years to the settlement (index preset)
    surface_vol: float | None = None  # the fitted surface's vol at the contract (index preset)

    @property
    def expiry(self) -> float:
        """Time to expiry in years: to the settlement under the index preset, else ACT/365 on the days."""
        return self.maturity if self.maturity is not None else self.days / 365.0

    @property
    def forward(self) -> float:
        """The forward of the expiry, S·e^{(r − q)T}."""
        return self.spot * math.exp((self.rate - self.dividend_yield) * self.expiry)

    @property
    def vol_overridden(self) -> bool:
        """Whether the contract's vol is typed over the surface's (index preset): by more than the 0.0001 vol point
        to which an override starts at the surface's vol, or at a strike where the surface gives none (NaN beyond its
        reach). With no vol at all (NaN on both sides: the surface's, beyond its reach) nothing was typed."""
        if self.surface_vol is None or not math.isfinite(self.volatility):
            return False
        return not math.isfinite(self.surface_vol) or not math.isclose(self.volatility, self.surface_vol, abs_tol=1e-6)

    @property
    def market(self) -> Market:
        return Market(self.spot, self.rate, self.volatility, self.expiry, self.dividend_yield)


def index_strike_limits() -> tuple[float, float]:
    """The index strike input's range: one listed interval to four times spot. The synthetic chain's spot is
    :data:`.market.INDEX_SPOT` at every valuation time, so the range is known before the chain is built (the link
    reader checks a link's strike against it)."""
    step = float(strike_interval(mk.INDEX_SPOT, "index"))
    return step, 4 * mk.INDEX_SPOT


def snap_strike(strike: float) -> float:
    """An index strike moved to the nearest strike on the listed interval near the money (5 points for SPX), as a
    listed contract's strike is: 5,817.37 is priced as 5,815, not as a contract that does not exist."""
    step = index_strike_limits()[0]
    return float(step * round(strike / step))


def _snap_typed_strike() -> None:
    """A strike typed off the listed grid moves to the nearest listed one, with a note (the input box shows whole
    points, so a typed 5,817.37 would read 5817 while it priced 5,817.37)."""
    state = st.session_state
    typed = state.get("strike")
    if typed is None:
        return
    step, listed = index_strike_limits()[0], snap_strike(float(typed))
    if listed != typed:
        state["strike"] = listed
        state[_NOTES] = [
            f"{strike_text(float(typed), step)} is not a listed strike (every {step:g} points near the money), so the "
            f"nearest listed one, {strike_text(listed, step)}, is used."
        ]


def _apply_preset() -> None:
    """Put the contract inputs at the new preset's defaults when the Market menu changes.

    The two presets share the ``strike`` and ``vol`` keys (and their URL parameters), whose values mean different
    things in each. A widget that keeps its key keeps its value, and one bound to the URL would read the old
    parameter back, so the callback sets them."""
    state = st.session_state
    if state.get("preset") == mk.TEXTBOOK:
        state.update(TEXTBOOK)
        return
    context = mk.index_market(_asof_value(), surface_state.settings())
    state["strike"] = context.atm_strike(context.default_expiry())
    state.pop("expiry", None)
    state["vol_src"] = SURFACE


def _asof_value() -> datetime:
    """The valuation time in the session (the widget's), else the default: the last cash close."""
    value = st.session_state.get("asof")
    return mk.snap(value) if isinstance(value, datetime) else mk.last_close()


def _set_now() -> None:
    st.session_state["asof"] = mk.snap(datetime.now(clock.NEW_YORK))


def _set_last_close() -> None:
    st.session_state["asof"] = mk.last_close()


def _snap_strike(context: mk.MarketContext) -> None:
    """The ATMF button: the listed strike nearest the focus expiry's forward."""
    st.session_state["strike"] = context.atm_strike(date.fromisoformat(st.session_state["expiry"]))


def _seed_override(surface_vol: float) -> None:
    """Start an override at the surface's vol, to 0.0001 vol point: switching to Override moves no price by a cent
    (a vega of $6.37 per vol point, at the money 28 days out, moves $0.0003)."""
    if st.session_state.get("vol_src") == OVERRIDE:
        st.session_state["vol"] = round(surface_vol * 100, 4)


def _focus_expiry(context: mk.MarketContext) -> list[str]:
    """The listed expiries (ISO dates), with a listed focus expiry put in the session.

    The menu is not bound to the URL, because its labels count the days from the valuation time: the link holds
    the ISO date (``?expiry=2026-11-20``) instead, which this reads on the session's first run and the sidebar
    writes back on every run. An expiry that is not listed (a link from another valuation time, or a later
    valuation time that let it expire) moves to the nearest listed one, with a note."""
    state = st.session_state
    listed = [day.isoformat() for day in context.expiries["expiry"]]
    current = state.get("expiry")
    if current in listed:
        return listed
    wanted = current if current is not None else st.query_params.get("expiry")
    choice = context.default_expiry()
    if wanted:
        try:
            asked = date.fromisoformat(str(wanted))
        except ValueError:
            state[_NOTES] = [
                f"The expiry {wanted!r} is not a date (2026-11-20), so the expiry nearest 30 days is used."
            ]
        else:
            choice = context.expiry_row(asked)["expiry"]
            if choice != asked:
                state[_NOTES] = [
                    f"{short_date(asked)} is not a listed expiry at this valuation time, so the nearest listed one, "
                    f"{short_date(choice)}, is used."
                ]
    state["expiry"] = choice.isoformat()
    return listed


def _index_market(sb, model: ModelConfig | None = None) -> Inputs:
    """The index preset's sidebar: valuation time, contract, and the chain's spot, forward and carry."""
    state = st.session_state
    default_asof = mk.last_close()
    asof = sb.datetime_input(
        "Valuation time (New York)",
        value=default_asof,
        min_value=ASOF_RANGE[0],
        max_value=ASOF_RANGE[1],
        step=mk.ASOF_STEP,
        format="YYYY-MM-DD",
        key="asof",
        bind="query-params",
        help="When the synthetic chain is quoted and every option is valued: New York wall time, on 5-minute steps. "
        "It starts at the last completed cash close (16:00, or 13:00 on an early close), so every visitor sees the "
        "same numbers all day. Time to expiry runs to each expiry's settlement, in minutes / 525,600.",
    )
    asof = mk.snap(asof) if asof is not None else default_asof
    left, right = sb.columns(2)
    left.button(
        "Last close", key="asof_close", on_click=_set_last_close, width="stretch", help="The last completed cash close."
    )
    right.button("Now", key="asof_now", on_click=_set_now, width="stretch", help="The current time, New York.")
    with st.spinner("Pricing the synthetic chain and fitting its surface…"):
        context = mk.index_market(asof, surface_state.settings())  # the Surface tab's fit settings
    sb.caption(f"{mk.session_note(asof).capitalize()} · {asof:%a %d %b %Y, %H:%M} ET")

    sb.subheader("Contract")
    listed = _focus_expiry(context)
    rows = {row["expiry"].isoformat(): row for _, row in context.expiries.iterrows()}
    focus = sb.selectbox(
        "Expiry",
        listed,
        format_func=lambda iso: mk.expiry_label(rows[iso]),
        key="expiry",
        help="The synthetic chain's listed expiries: SPX monthlies settle AM (third Friday, on the opening prints), "
        "SPXW weeklies PM (at the 16:00 close). The days count to the settlement.",
    )
    if focus == context.default_expiry().isoformat():
        st.query_params.pop("expiry", None)
    else:
        st.query_params["expiry"] = focus
    row = rows[focus]
    expiry_date = date.fromisoformat(focus)
    step = context.strike_step
    strike_col, atm_col = sb.columns([2, 1], vertical_alignment="bottom")
    strike = strike_col.number_input(
        "Strike",
        min_value=step,
        max_value=4 * context.spot,
        value=context.atm_strike(context.default_expiry()),
        step=step,
        format="%.0f",
        key="strike",
        bind="query-params",
        on_change=_snap_typed_strike,
        help=f"Listed every {step:g} points near the money; a strike typed between them moves to the nearest. ATMF "
        "snaps to the listed strike nearest the expiry's forward.",
    )
    atm_col.button(
        "ATMF",
        key="strike_atmf",
        on_click=_snap_strike,
        args=(context,),
        width="stretch",
        help=f"The listed strike nearest this expiry's forward: {level(context.atm_strike(expiry_date))}.",
    )
    maturity = float(row["maturity"])
    forward, discount = float(row["forward"]), float(row["discount_factor"])
    # implied_vol_at reads K/S and moves it to the expiry's forward itself.
    surface_vol = float(context.surface.implied_vol_at(float(strike) / context.spot, maturity, extrapolate=True))
    vol_src = sb.segmented_control(
        "Volatility",
        [SURFACE, OVERRIDE],
        default=SURFACE,
        required=True,
        key="vol_src",
        bind="query-params",
        on_change=_seed_override,
        args=(surface_vol,),
        help="Surface: the fitted surface's implied vol at this strike and expiry (an arbitrage-free, spread-weighted "
        "SVI smile through the chain's quotes). Override: a vol you type, marked in amber.",
    )
    volatility = surface_vol
    if vol_src == OVERRIDE:
        if "vol" not in state:
            state["vol"] = round(surface_vol * 100, 4)
        typed = sb.container(key="vol_override").number_input(
            "Override vol (%)", min_value=0.1, max_value=500.0, step=0.25, format="%.2f", key="vol", bind="query-params"
        )
        volatility = float(typed) / 100.0
        gap = (volatility - surface_vol) * 100
        sb.caption(f"{gap:+.2f} vp against the surface's {pct(surface_vol, 2)}".replace("-", "−"))
    else:
        sb.caption(f"σ {pct(surface_vol, 2)} from the surface at K/F {moneyness(float(strike), forward)}")
    rate, dividend_yield = mk.carry(forward, discount, context.spot, maturity)
    sb.caption(
        f"{mk.INDEX_ROOT} {level(context.spot)}, synthetic (not market data). From put-call parity on the expiry's "
        f"quotes: F {level(forward)}, DF {discount:.5f}, so r {pct(rate, 2)} and q {pct(dividend_yield, 2)}."
    )
    return Inputs(
        spot=context.spot,
        strike=float(strike),
        days=maturity * 365.0,
        volatility=volatility,
        rate=rate,
        dividend_yield=dividend_yield,
        model=model_sidebar(european_only=True),
        preset=mk.INDEX,
        context=context,
        expiry_date=expiry_date,
        settlement=str(row["settlement"]),
        maturity=maturity,
        surface_vol=surface_vol,
    )


def _textbook_market(sb) -> Inputs:
    """The textbook preset's sidebar: the six typed inputs of version 2."""
    st.query_params.pop("expiry", None)
    spot = sb.number_input(
        "Spot price",
        min_value=TEXTBOOK_LIMITS["spot"][0],
        max_value=TEXTBOOK_LIMITS["spot"][1],
        value=TEXTBOOK["spot"],
        step=1.0,
        format="%.2f",
        key="spot",
        bind="query-params",
        help="Current price of the underlying asset.",
    )
    strike = sb.number_input(
        "Strike price",
        min_value=TEXTBOOK_LIMITS["strike"][0],
        max_value=TEXTBOOK_LIMITS["strike"][1],
        value=TEXTBOOK["strike"],
        step=1.0,
        format="%.2f",
        key="strike",
        bind="query-params",
    )
    days = sb.number_input(
        "Days to expiry",
        min_value=TEXTBOOK_LIMITS["days"][0],
        max_value=TEXTBOOK_LIMITS["days"][1],
        value=TEXTBOOK["days"],
        step=1,
        key="days",
        bind="query-params",
        help="Calendar days; time to expiry T = days / 365.",
    )
    vol_pct = sb.number_input(
        "Volatility (%)",
        min_value=TEXTBOOK_LIMITS["vol"][0],
        max_value=TEXTBOOK_LIMITS["vol"][1],
        value=TEXTBOOK["vol"],
        step=0.5,
        format="%.2f",
        key="vol",
        bind="query-params",
        help="Annualised implied volatility σ.",
    )
    rate_pct = sb.number_input(
        "Risk-free rate (%)",
        min_value=TEXTBOOK_LIMITS["rate"][0],
        max_value=TEXTBOOK_LIMITS["rate"][1],
        value=TEXTBOOK["rate"],
        step=0.05,
        format="%.2f",
        key="rate",
        bind="query-params",
        help="Continuously compounded; negative rates are allowed.",
    )
    div_pct = sb.number_input(
        "Dividend yield (%)",
        min_value=TEXTBOOK_LIMITS["div"][0],
        max_value=TEXTBOOK_LIMITS["div"][1],
        value=TEXTBOOK["div"],
        step=0.1,
        format="%.2f",
        key="div",
        bind="query-params",
        help="Continuous dividend yield q (Merton's extension).",
    )
    sb.caption(f"T = {days / 365:.4f} years")
    return Inputs(
        spot=float(spot),
        strike=float(strike),
        days=int(days),
        volatility=float(vol_pct) / 100.0,
        rate=float(rate_pct) / 100.0,
        dividend_yield=float(div_pct) / 100.0,
        model=model_sidebar(),
    )


def sidebar() -> Inputs:
    sb = st.sidebar
    sb.header("Market")
    preset = sb.selectbox(
        "Market",
        mk.PRESETS,
        key="preset",
        bind="query-params",
        on_change=_apply_preset,
        label_visibility="collapsed",
        help="Synthetic index: an SPX-like chain priced at the valuation time and read like real quotes (put-call "
        "parity forwards, an arbitrage-free SVI surface); the contract is a listed expiry and strike. Textbook: "
        "S = K = 100, one year, σ 20%, r 5%, q 0, all typed.",
    )
    inputs = _index_market(sb) if preset == mk.INDEX else _textbook_market(sb)
    sb.divider()
    sb.markdown(
        "Built by **[Navnoor Bawa](https://www.linkedin.com/in/navnoorbawa/)** · "
        "[Source](https://github.com/NavnoorBawa/Option-Pricing-and-Strategy-Analyzer)"
    )
    sb.caption("Educational tool — not investment advice.")
    return inputs


def market_notes() -> list[str]:
    """Notes from this run's sidebar (an expiry moved to a listed one), once."""
    return st.session_state.pop(_NOTES, [])
