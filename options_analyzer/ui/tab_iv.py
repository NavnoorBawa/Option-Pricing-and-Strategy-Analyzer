"""The IV tab: a single-quote inverter for the sidebar's contract (UI spec 3.5; roadmap PR-7).

It inverts one market quote, a bid and an ask or a single price, to implied vols, with the no-arbitrage bounds and
the quote's distance from the published surface in view: a context line (the contract), an input row, six tiles
(IV BID, IV MID, IV ASK, SURFACE with mid vs fit in vol points, VEGA and BOUNDS), the model value against vol with the
quotes and their vols marked, and a footer line. Chain-scale inversion lives on the Surface tab.

European quotes are inverted with Jäckel's Let's Be Rational (``implied_vol.implied_volatility``); under the index
preset the sidebar's rate and dividend yield are the expiry's parity carry, so that is Black-76 on the parity forward
and discount factor, the published surface's own inversion. American quotes (textbook market only: SPX options are
European) are solved with the sidebar's American engine, or on a 501-step Leisen-Reimer tree under the closed form.

A price outside the bounds has no implied vol: its tile reads ``–`` with the bound in its help, the inputs get one
``st.error`` line naming the bound to the cent, and the chart stays, with the quote's line where it is. A quote with
no time value (within 1e−12 of the floor) is told apart, as Milestone 1 made the solvers do (``no_time_value``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import streamlit as st

from .. import black_scholes as bs
from ..implied_vol import (
    MAX_VOL,
    TIME_VALUE_MARGIN,
    ImpliedVolError,
    american_bounds,
    implied_volatility,
    lowest_tree_volatility,
    no_arbitrage_bounds,
)
from . import cache, charts, charts_pricing
from .components import card_row, current_key
from .format import MISSING, md, money, number, premium, vol, vol_change
from .inputs import Inputs
from .tab_pricing import DRAWER, context_line, contract

#: The tiles' row and its three pairs, which wrap as the Pricing tab's do (``tab_pricing.TILES``).
TILES = ("iv_tiles", ("iv_tiles_bid", "iv_tiles_ask", "iv_tiles_context"))
#: The index's tick (Cboe SPX: $0.05 below $3, $0.10 from $3); the textbook market's cent.
INDEX_TICKS = ((3.0, 0.05), (math.inf, 0.10))
TEXTBOOK_TICK = 0.01
#: The quote inputs' format: cents, as the chain quotes them.
PRICE_FORMAT = "%.2f"
#: Half the chart's vol range about the mid's implied vol (:func:`chart_window`): 1 vol point, where an SPX
#: month's 0.17-point spread puts the bid and the ask about 20 px apart on the 360 px chart.
WINDOW = 0.01
_TREE_LIMIT = "The 501-step tree cannot price this option at the vols the search needs"
_ENGINE_LIMIT = "The American engine cannot price this option at the vols the search needs"


def _small_amount(value: float) -> str:
    """A small positive amount in fixed point to two significant digits: ``0.00000000010`` for 1e-10.

    Not in scientific notation, whose exponent needs a minus sign the app's font lacks in superscript.
    """
    return f"{value:.{max(0, 1 - math.floor(math.log10(value)))}f}"


def _no_time_value_text(price: float, lower: float, scale: float, american: bool, option_type: str) -> str:
    """What the tab says for a quote the solvers reject with the reason ``no_time_value`` (Milestone 1).

    ``lower`` is the no-arbitrage lower bound, zero exactly when the option is worth nothing at zero vol (out of the
    money)."""
    margin = f"${_small_amount(TIME_VALUE_MARGIN * scale)} (one part in 10¹² of the larger of spot and strike)"
    if lower <= 0.0 and american:
        # The American engines' values near zero vol are accurate only in absolute terms.
        return md(
            f"No implied vol: the option is worth $0 at zero vol, and a quote of {money(price, 2)} is no more than "
            f"{margin} above that, closer than the American pricing engines can tell vols apart there."
        )
    if lower <= 0.0:
        return md(
            f"No implied vol: an out-of-the-money option is worth only its time value, and a quote of "
            f"{money(price, 2)} has none."
        )
    if american:
        where = (
            f"no more than {margin} above the option's value at zero vol (deep in the money, the value of exercising "
            "now)"
        )
    else:
        discounted = "S·e^(−qT) − K·e^(−rT)" if option_type == "call" else "K·e^(−rT) − S·e^(−qT)"
        where = f"within {margin} of the floor, the discounted intrinsic value {discounted} = {money(lower, 2)}"
    return md(
        f"No implied vol: this quote has no time value. It is {where}, and every vol from zero up to some level "
        "prices the option that close to it. For a vol at this strike, invert the out-of-the-money option: it is "
        "all time value."
    )


@dataclass(frozen=True)
class Solved:
    """One quoted price and its implied vol (NaN when it has none), with why not: ``"below"`` or ``"above"`` the
    bounds, ``"below_zero_vol"`` (inside them but below the American value at zero vol), ``"no_time_value"``, or
    another solver message."""

    label: str
    price: float
    iv: float
    problem: str = ""


def tick(price: float, listed: bool) -> float:
    """The quote's tick: the index's $0.05 below $3 and $0.10 from $3, or a cent in the textbook market.

    >>> tick(2.5, True), tick(98.2, True), tick(10.45, False)
    (0.05, 0.1, 0.01)
    """
    if not listed:
        return TEXTBOOK_TICK
    return next(step for limit, step in INDEX_TICKS if price < limit)


def render(inputs: Inputs) -> None:
    """The IV tab (see the module docstring)."""
    S, K, T, r, q, sigma = (
        inputs.spot,
        inputs.strike,
        inputs.expiry,
        inputs.rate,
        inputs.dividend_yield,
        inputs.volatility,
    )
    model = inputs.model
    alo = model.kind == "alo"
    market = contract(inputs)
    st.caption(context_line(inputs))

    row = st.container(horizontal=True, key="iv_inputs", vertical_alignment="bottom", gap="small")
    option_type = row.segmented_control(
        "Type",
        ["call", "put"],
        default="call",
        required=True,
        format_func=str.capitalize,
        key="iv_type",
        persist_state="session",
    )
    # The exercise style follows the sidebar's engine until the user picks one here (the key changes with the
    # engine); the index is European only.
    sidebar_style = "American" if model.is_american else "European"
    style = row.segmented_control(
        "Exercise",
        ["European", "American"],
        default=sidebar_style,
        required=True,
        key=current_key("iv_style_", f"iv_style_{'index' if market.listed else sidebar_style.lower()}"),
        persist_state="session",
        disabled=market.listed,
        help="SPX options are European-exercise."
        if market.listed
        else "European quotes are inverted with Let's Be Rational; American ones with "
        + ("the sidebar's American engine." if alo else "a 501-step Leisen-Reimer tree."),
    )
    american = style == "American" and not market.listed
    quote = market.quotes[option_type]
    # The quote belongs to the contract: its widgets' keys hold it, so a typed quote lasts until the contract changes.
    identity = f"{S!r}|{K!r}|{T!r}|{r!r}|{q!r}|{option_type}"
    single = row.toggle(
        "Single price",
        value=quote is None,
        key=current_key("iv_single|", f"iv_single|{identity}"),
        persist_state="session",
        help="Invert one price instead of a bid and an ask. The textbook market has no quotes, so it starts here.",
    )

    def model_price(v: float) -> float:
        if american and alo:
            return cache.alo_value(S, K, T, r, max(v, 0.0), q, option_type, model.scheme)
        if american:
            return cache.american_value(S, K, T, r, max(v, lowest), q, option_type)
        return float(bs.price(S, K, T, r, v, q, option_type))

    lowest = lowest_tree_volatility(T, r, q, *cache.DEFAULT_TREE) if american and not alo else 0.0
    try:
        model_value = model_price(sigma)
    except ValueError as exc:  # a tree too short for the sidebar vol
        st.error(f"{_TREE_LIMIT} ({exc}).")
        return
    listed = market.listed
    lower, upper = (
        american_bounds(S, K, T, r, q, option_type) if american else no_arbitrage_bounds(S, K, T, r, q, option_type)
    )
    # Without a quote the price starts at the model value to the cent, never below the floor in cents: a deep
    # in-the-money option with no time value (a call at 60 on 100 two days out, floor 40.0131) would otherwise start
    # at 40.01, below the bound, an error on opening the tab.
    # Nor at or above the cap: at 500% vol over five years the call is worth 99.999998 on a spot of 100, which would
    # round to the cap, 100.00.
    floor_cents = math.ceil(lower * 100 - 1e-9) / 100
    cap_cents = math.floor(upper * 100 - 1e-9) / 100

    def cents(value: float) -> float:
        return min(max(round(value, 2), floor_cents), cap_cents)

    if single:
        start = cents(model_value) if quote is None else quote.mid if quote.bid > 0 else quote.ask
        price = row.number_input(
            "Price",
            min_value=0.0,
            value=float(start),
            step=tick(start, listed),
            format=PRICE_FORMAT,
            key=current_key("iv_price|", f"iv_price|{identity}"),
            persist_state="session",
        )
        prices = [("Price", float(price))]
    else:
        bid0 = quote.bid if quote is not None else cents(model_value * 0.99)
        ask0 = quote.ask if quote is not None else max(cents(model_value * 1.01), bid0)
        bid = row.number_input(
            "Bid",
            min_value=0.0,
            value=float(bid0),
            step=tick(bid0, listed),
            format=PRICE_FORMAT,
            key=current_key("iv_bid|", f"iv_bid|{identity}"),
            persist_state="session",
        )
        ask = row.number_input(
            "Ask",
            min_value=0.0,
            value=float(ask0),
            step=tick(ask0, listed),
            format=PRICE_FORMAT,
            key=current_key("iv_ask|", f"iv_ask|{identity}"),
            persist_state="session",
        )
        prices = [("Bid", float(bid)), ("Mid", 0.5 * (float(bid) + float(ask))), ("Ask", float(ask))]

    # An American floor is the larger of the intrinsic value and the European floor (implied_vol.american_bounds):
    # the at-the-money call with a 5% rate and a 2% yield is floored at 2.90, its discounted intrinsic value.
    intrinsic = max(bs.option_sign(option_type) * (S - K), 0.0)
    floor_name = "the intrinsic value" if american and intrinsic >= lower else "the discounted intrinsic value"
    solved = [
        _solve(label, p, (S, K, T, r, q, option_type), american, alo, model.scheme, lower, upper) for label, p in prices
    ]
    errors = []
    for s in solved:
        if s.problem == "below":
            errors.append(
                f"{s.label} {premium(s.price)} is below the no-arbitrage floor {money(lower, 2)} ({floor_name})"
            )
        elif s.problem == "above":
            errors.append(f"{s.label} {premium(s.price)} is at or above the no-arbitrage cap {money(upper, 2)}")
        elif s.problem == "below_zero_vol":
            errors.append(
                f"{s.label} {premium(s.price)} is below {money(model_price(0.0), 2)}, the American value at zero vol, "
                "the least any vol gives"
            )
        elif s.problem == "no_time_value":
            st.warning(_no_time_value_text(s.price, lower, max(S, K), american, option_type))
        elif s.problem:
            errors.append(f"{s.label} {premium(s.price)}: {s.problem}")
    if not single and prices[0][1] > prices[2][1]:
        errors.append(f"the bid {premium(prices[0][1])} is above the ask {premium(prices[2][1])}")
    if errors:
        st.error(md("No implied vol: " + "; ".join(errors) + "."))

    by_label = {s.label: s for s in solved}
    mid = by_label.get("Mid", by_label.get("Price"))
    _tiles(inputs, market, by_label, mid, single, american, alo, (lower, upper), floor_name, option_type)

    low, high = chart_window([s.iv for s in solved], mid.iv, sigma)
    if american and alo:
        vols, curve = cache.alo_curve(S, K, T, r, q, option_type, low, high, model.scheme)
    elif american:
        vols, curve = cache.american_curve(S, K, T, r, q, option_type, low, high)
    else:
        vols = np.linspace(low, high, 201)
        curve = bs.price(S, K, T, r, vols, q, option_type)
    engine = ("ALO" if alo else "LR 501 tree") if american else "BSM"
    fig = charts_pricing.iv_price_chart(
        vols * 100,
        curve,
        quotes=[(s.label, s.price, s.iv * 100) for s in solved],
        floor=lower if lower > 0 else None,
        name=f"Model value ({engine})",
    )
    charts.show(
        fig,
        key="iv_price",
        alt=f"{style} {option_type} value against vol, with the quoted prices and their implied vols",
    )
    foot = st.container(horizontal=True, key="iv_footer", vertical_alignment="center", gap="small")
    method = (
        ("American engine (ALO), Brent's method" if alo else "501-step LR tree, Brent's method")
        if american
        else "Let's Be Rational (Jäckel 2015), Black-76 on the parity forward"
        if listed
        else "Let's Be Rational (Jäckel 2015)"
    )
    foot.caption(f"{method} · bounds checked first · per share")
    foot.button("Methodology", key="iv_methodology", type="tertiary", on_click=_open_methodology)


def _open_methodology() -> None:
    st.session_state[DRAWER] = True


def chart_window(ivs: list[float], mid: float, sigma: float) -> tuple[float, float]:
    """The chart's vol range (decimals): :data:`WINDOW` either side of the mid's implied vol (or the one finite
    one), widened to three times the quotes' spread in vol, so that the bid, mid and ask lines stand apart. On the SPX
    default the bid and ask are 0.17 vol points apart: over 0.5% to 60% they printed on top of each other. With no
    implied vol (every price outside the bounds) the whole range, 0.5% to the larger of 60% and twice ``sigma``,
    where the floor and the quote outside it show.

    >>> [round(v, 4) for v in chart_window([0.1534, 0.1543, 0.1551], 0.1543, 0.1544)]
    [0.1443, 0.1643]
    >>> chart_window([float("nan")], float("nan"), 0.2)
    (0.005, 0.6)
    """
    finite = [v for v in ivs if np.isfinite(v)]
    if not finite:
        return 0.005, max(0.6, 2.0 * sigma)
    centre = mid if np.isfinite(mid) else finite[0]
    half = max(WINDOW, 3.0 * (max(finite) - min(finite)))
    return max(0.005, centre - half), centre + half


def _solve(label, price, contract_args, american, alo, scheme, lower, upper) -> Solved:
    """Invert one price: NaN with the reason where it is outside the bounds or has no time value."""
    S, K, T, r, q, option_type = contract_args
    margin = 1e-12 * max(S, K)
    if price < lower - margin:
        return Solved(label, price, math.nan, "below")
    if price >= upper:
        return Solved(label, price, math.nan, "above")
    try:
        if american and alo:
            iv = cache.alo_iv(price, S, K, T, r, q, option_type, scheme)
        elif american:
            iv = cache.american_iv(price, S, K, T, r, q, option_type)
        else:
            iv = implied_volatility(price, S, K, T, r, q, option_type)
    except ImpliedVolError as exc:
        reason = exc.reason
        if reason == "below_lower_bound":
            # Above the floor but below the American value at zero vol: with a dividend yield above the rate, a long
            # put can be worth most exercised at a date before expiry (25 at zero vol, floor 23.25, 20 years out).
            return Solved(label, price, math.nan, "below" if price < lower else "below_zero_vol")
        if reason == "at_or_above_upper_bound":
            return Solved(label, price, math.nan, "above")
        # The solvers' own messages carry raw numbers; the tab names the reason in words (the number standard).
        words = {
            "no_time_value": "no_time_value",
            "above_max_vol": f"its implied vol would exceed {number(MAX_VOL * 100, 0)}%",
            "below_tree_floor": "it is not above the tree's value at the lowest vol the tree can represent",
        }
        return Solved(label, price, math.nan, words.get(reason, "it cannot be inverted"))
    except ValueError as exc:  # the bracket search reached a vol the engine cannot price
        return Solved(label, price, math.nan, f"{_ENGINE_LIMIT if alo else _TREE_LIMIT} ({exc})")
    return Solved(label, price, float(iv))


def _vega(inputs: Inputs, iv: float, option_type: str, american: bool, alo: bool) -> float:
    """$ per share per vol point at ``iv``: the closed form's, the American engine's own (``american.greeks``), or a
    difference of tree values."""
    S, K, T, r, q = inputs.spot, inputs.strike, inputs.expiry, inputs.rate, inputs.dividend_yield
    if american and alo:
        return float(cache.alo_option(S, K, T, r, iv, q, option_type, inputs.model.scheme)[1]["vega"]) / 100
    if american:
        lowest = lowest_tree_volatility(T, r, q, *cache.DEFAULT_TREE)
        up, down = iv + 0.005, max(iv - 0.005, lowest)
        try:
            return (
                cache.american_value(S, K, T, r, up, q, option_type)
                - cache.american_value(S, K, T, r, down, q, option_type)
            ) / ((up - down) * 100)
        except ValueError:
            return math.nan
    return float(bs.greeks(S, K, T, r, max(iv, 1e-6), q, option_type)["vega"]) / 100


def _tiles(inputs, market, by_label, mid, single, american, alo, bounds, floor_name, option_type) -> None:
    """IV BID, IV MID (IV in single-price mode), IV ASK, SURFACE, VEGA and BOUNDS."""
    lower, upper = bounds
    off = {"delta_color": "off", "delta_arrow": "off"}

    def iv_tile(label: str, solved) -> tuple:
        if solved is None:
            return (label, MISSING, {"help": "Single price: no bid or ask to invert."})
        if np.isfinite(solved.iv):
            return (label, vol(solved.iv), {"help": f"Implied vol of {premium(solved.price)}."})
        why = {
            "below": f"{premium(solved.price)} is below the no-arbitrage floor {money(lower, 2)} ({floor_name}).",
            "above": f"{premium(solved.price)} is at or above the no-arbitrage cap {money(upper, 2)}.",
            "no_time_value": f"{premium(solved.price)} has no time value.",
            "below_zero_vol": f"{premium(solved.price)} is below the American value at zero vol.",
        }.get(solved.problem, f"No implied vol: {solved.problem}.")
        return (label, MISSING, {"help": md(why)})

    tiles = [
        iv_tile("IV bid", None if single else by_label["Bid"]),
        iv_tile("IV" if single else "IV mid", mid),
        iv_tile("IV ask", None if single else by_label["Ask"]),
    ]
    if market.listed and market.fit_vol is not None:
        gap = mid.iv - market.fit_vol if np.isfinite(mid.iv) else math.nan
        tiles.append(
            (
                "Surface",
                vol(market.fit_vol),
                {
                    "delta": f"mid vs fit {vol_change(gap)}",
                    **off,
                    "help": "The published surface's vol at this strike. Mid vs fit = σ mid − σ fit: positive means "
                    "the quote is rich against the fit.",
                },
            )
        )
    else:
        tiles.append(
            (
                "Surface",
                MISSING,
                {
                    "delta": f"mid vs fit {MISSING}",
                    **off,
                    "help": f"The textbook market has no surface; the sidebar vol is {vol(inputs.volatility)}.",
                },
            )
        )
    vega = _vega(inputs, mid.iv, option_type, american, alo) if np.isfinite(mid.iv) and mid.iv > 0 else math.nan
    tiles.append(
        (
            "Vega",
            MISSING if not np.isfinite(vega) else premium(vega),
            {"delta": "$ per vol point", **off, "help": "Value change per vol point at the mid's implied vol."},
        )
    )
    problems = [s.problem for s in by_label.values() if s.problem in ("below", "above")]
    state = "below floor" if "below" in problems else "above cap" if "above" in problems else "inside"
    tiles.append(
        (
            "Bounds",
            state,
            {
                "delta": f"floor {premium(lower)}",
                **off,
                "help": md(f"No-arbitrage bounds: {money(lower, 2)} ({floor_name}) ≤ price < {money(upper, 2)}."),
            },
        )
    )
    card_row(*TILES, tiles)
    if np.isfinite(vega) and vega < 1e-6 * max(inputs.spot, inputs.strike):
        st.warning("Vega is tiny here, so the implied vol is poorly determined: a tick moves it a lot.")
