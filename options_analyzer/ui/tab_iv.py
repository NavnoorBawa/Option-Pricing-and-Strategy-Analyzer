"""IV tab: invert a market price to its implied volatility with a robust bracketed solver."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from .. import black_scholes as bs
from ..binomial import METHODS
from ..implied_vol import (
    TIME_VALUE_MARGIN,
    ImpliedVolError,
    american_bounds,
    implied_volatility,
    lowest_tree_volatility,
    no_arbitrage_bounds,
)
from . import cache, charts
from .components import card_row, current_key
from .format import md, money, pct
from .inputs import Inputs

_TREE_LIMIT = (
    "The sidebar's binomial tree has too few steps to price this option at the volatilities needed; "
    "increase the number of steps"
)
_ENGINE_LIMIT = "The American engine cannot price this option at the volatilities the search needs"


def _reset_quote() -> None:
    st.session_state["iv_quote_generation"] = st.session_state.get("iv_quote_generation", 0) + 1


def _small_amount(value: float) -> str:
    """A small positive amount in fixed point to two significant digits: ``0.00000000010`` for 1e-10.

    Not in scientific notation, whose exponent needs a minus sign the app's font lacks in superscript.
    """
    return f"{value:.{max(0, 1 - math.floor(math.log10(value)))}f}"


def _no_time_value_text(price: float, lower: float, scale: float, american: bool, model_quote: bool) -> str:
    """What the tab says for a quote the solvers reject with the reason ``no_time_value``.

    ``lower`` is the no-arbitrage lower bound, zero exactly when the option is worth nothing at zero volatility (out of
    the money); ``model_quote``: the quote is the model value at the sidebar volatility.
    """
    if lower <= 0.0:
        return md(
            f"No implied volatility: an out-of-the-money option is worth only its time value, and a quote of "
            f"{money(price, 4)} has none."
        )
    floor = (
        "value at zero volatility (deep in the money, the value of exercising now)"
        if american
        else "intrinsic value, the no-arbitrage lower bound"
    )
    why = (
        "The model value at the sidebar volatility is such a quote: the option is that deep in the money."
        if model_quote
        else "This is typical deep in the money close to expiry."
    )
    # The European solver's margin holds on both sides of the bound; the American solvers also take a quote up to
    # 1e-10·max(S, K) below the zero-volatility value (their engines' error there) as on it.
    margin = f"${_small_amount(TIME_VALUE_MARGIN * scale)} (one part in 10¹² of the larger of spot and strike)"
    where = f"no more than {margin} above" if american else f"within {margin} of"
    return md(
        f"No implied volatility: this quote has no time value. It is {where} the option's {floor}, and every "
        "volatility from zero up to some level prices the option that close to it, so none can be singled out. "
        f"{why} For a volatility at this strike, invert the out-of-the-money option's quote: it is all time value."
    )


@st.fragment
def render(inputs: Inputs) -> None:
    S, K, T, r, q, sigma = (
        inputs.spot,
        inputs.strike,
        inputs.expiry,
        inputs.rate,
        inputs.dividend_yield,
        inputs.volatility,
    )
    model = inputs.model
    # American quotes are inverted on the sidebar's tree when the binomial model is selected, and with the sidebar's
    # American engine when that is, so the headline American price reproduces the sidebar volatility here.
    alo = model.kind == "alo"
    steps, method = (model.steps, model.tree_method) if model.kind == "binomial" else cache.DEFAULT_TREE
    tree_name = f"{steps:,}-step {METHODS[method]} tree"
    engine_name = f"the sidebar's American engine ({model.scheme} scheme)"
    st.caption(
        f"Uses the sidebar contract: S = {S:,.2f}, K = {K:,.2f}, {inputs.days} days, r = {r:.2%}, q = {q:.2%}. "
        "The quote starts at the model value at the sidebar volatility and is kept when you change the volatility "
        "or the exercise style; it starts again when the contract changes."
    )
    c1, c2, c3 = st.columns([1, 1, 2])
    option_type = c1.segmented_control(
        "Option",
        ["call", "put"],
        default="call",
        required=True,
        format_func=str.capitalize,
        key="iv_type",
        persist_state="session",
    )
    # Follow the sidebar's exercise style until the user picks one here; the choice lasts until the sidebar
    # style changes again (the key, and so the widget, changes with it).
    sidebar_style = "American" if model.is_american else "European"
    style = c2.segmented_control(
        "Exercise",
        ["European", "American"],
        default=sidebar_style,
        required=True,
        key=current_key("iv_style_", f"iv_style_{sidebar_style.lower()}"),
        persist_state="session",
        help="Follows the sidebar's pricing model unless you choose here. European quotes are inverted with "
        "Black-Scholes-Merton; American quotes (listed single-stock options) "
        + (
            f"with {engine_name}."
            if alo
            else f"on a {tree_name}" + (" (the sidebar's tree)." if model.kind == "binomial" else ".")
        ),
    )
    american = style == "American"
    # The engine values zero volatility exactly; a tree has a lowest volatility it can represent.
    lowest = 0.0 if alo or not american else lowest_tree_volatility(T, r, q, steps, method)
    limit = _ENGINE_LIMIT if alo else _TREE_LIMIT

    def model_price(vol: float) -> float:
        if american and alo:
            return cache.alo_value(S, K, T, r, max(vol, 0.0), q, option_type, model.scheme)
        if american:
            return cache.american_value(S, K, T, r, max(vol, lowest), q, option_type, steps, method)
        return bs.price(S, K, T, r, vol, q, option_type)

    try:
        model_value = model_price(sigma)
    except ValueError as exc:  # a short tree cannot represent the sidebar volatility
        st.error(f"{limit} ({exc}).")
        return
    # The quote belongs to the contract: its widget key holds the contract (not the volatility or the exercise
    # style), so the typed quote survives those changes and the reset button starts a fresh widget.
    generation = st.session_state.get("iv_quote_generation", 0)
    quote_key = current_key("iv_quote|", f"iv_quote|{S!r}|{K!r}|{inputs.days}|{r!r}|{q!r}|{option_type}|{generation}")
    price = c3.number_input(
        "Market price",
        min_value=0.0,
        value=float(model_value),
        step=0.05,
        format="%.4f",
        key=quote_key,
        persist_state="session",
    )
    c3.button(
        "Reset to model value",
        on_click=_reset_quote,
        key="iv_quote_reset",
        help=f"Set the quote to the model value at the sidebar volatility ({money(model_value, 4)}).",
    )
    lower, upper = (
        american_bounds(S, K, T, r, q, option_type) if american else no_arbitrage_bounds(S, K, T, r, q, option_type)
    )
    st.caption(md(f"No-arbitrage bounds for this {style} {option_type}: {money(lower, 4)} ≤ price < {money(upper, 4)}"))

    try:
        if american and alo:
            with st.spinner("Inverting the American engine…"):
                iv = cache.alo_iv(price, S, K, T, r, q, option_type, model.scheme)
        elif american:
            with st.spinner("Inverting the binomial tree…"):
                iv = cache.american_iv(price, S, K, T, r, q, option_type, steps, method)
        else:
            iv = implied_volatility(price, S, K, T, r, q, option_type)
    except ImpliedVolError as exc:
        if exc.reason == "no_time_value":
            model_quote = abs(price - model_value) <= 1e-9 * max(S, K)
            st.warning(_no_time_value_text(price, lower, max(S, K), american, model_quote))
            return
        at_bound = min(abs(price - lower), abs(price - upper)) <= 1e-9 * max(S, K)
        if at_bound and abs(price - model_value) <= 1e-9 * max(S, K):
            st.info(
                "At these inputs the option value is numerically indistinguishable from a no-arbitrage bound, "
                "so the volatility cannot be recovered from the price (vega is effectively zero)."
            )
        else:
            st.error(f"No implied volatility exists: {exc}.")
        return
    except ValueError as exc:  # the bracket search reached a volatility the tree cannot represent
        st.error(f"{limit} ({exc}).")
        return

    def safe_price(vol: float) -> float:
        try:
            return model_price(vol)
        except ValueError:
            return np.nan

    if american and alo and iv > 0.0:
        # The engine's own vega (american.greeks): closed form for an option valued as European, otherwise a
        # difference with its scheme held fixed.
        vega = cache.alo_option(S, K, T, r, iv, q, option_type, model.scheme)[1]["vega"] / 100
    elif american:
        bump = 0.005
        low_vol = max(iv - bump, lowest)
        vega = (safe_price(iv + bump) - safe_price(low_vol)) / (iv + bump - low_vol) / 100
    else:
        vega = bs.greeks(S, K, T, r, max(iv, 1e-6), q, option_type)["vega"] / 100
    intrinsic = max(bs.option_sign(option_type) * (S - K), 0.0)
    gap = (iv - sigma) * 100
    check = safe_price(iv)
    # The quote's own four decimals on the card; the full value and its distance from the quote in the help.
    distance = "equal to the quote" if check == price else f"${abs(check - price):.1e} from the quote"
    check_detail = f": {money(check, 6)}, {distance}" if np.isfinite(check) else ""
    # The same wrapping pairs as the price cards, so no value is cut off at laptop or phone width.
    card_row(
        "iv_cards",
        ("iv_cards_solution", "iv_cards_option"),
        [
            (
                "Implied volatility",
                pct(iv, 3),
                {
                    "delta": f"{0.0 if abs(gap) < 0.005 else gap:+.2f} vol pts vs sidebar",
                    "delta_color": "off",
                    "delta_arrow": "off",
                },
            ),
            (
                "Model check",
                money(check, 4),
                {"help": md(f"Model value at the implied volatility (should equal the quote){check_detail}.")},
            ),
            ("Vega at IV", money(vega, 4), {"help": "Price change per 1 vol point at the solution."}),
            (
                "Time value",
                money(price - intrinsic, 4),
                {
                    "help": md(
                        f"Quote minus intrinsic value max({'S − K' if option_type == 'call' else 'K − S'}, 0) = "
                        f"{money(intrinsic, 4)}. A European option can trade below intrinsic (deep in-the-money "
                        "puts, calls on high-dividend stocks), so this can be negative. The quote is "
                        f"{money(price - lower, 4)} above the no-arbitrage lower bound."
                    )
                },
            ),
        ],
    )
    if vega < 1e-4 * max(S, K):
        st.warning(
            "Vega is tiny here, so the implied volatility is poorly determined: a small change in the quote "
            "moves it a lot. This is typical for deep in- or out-of-the-money options close to expiry."
        )

    max_vol = max(2.5 * iv, 2.0 * sigma, 0.6)
    if american and alo:
        vols, curve = cache.alo_curve(S, K, T, r, q, option_type, max_vol, model.scheme)
    elif american:
        vols, curve = cache.american_curve(S, K, T, r, q, option_type, max_vol, steps, method)
    else:
        vols = np.linspace(0.005, max_vol, 400)
        curve = bs.price(S, K, T, r, vols, q, option_type)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vols * 100, y=curve, name="Model value", line=dict(color=charts.CALL, width=2.5)))
    fig.add_hline(y=price, line=dict(color=charts.NEUTRAL, dash="dash"), annotation_text="market price")
    fig.add_trace(
        go.Scatter(
            x=[iv * 100],
            y=[price],
            mode="markers",
            name="Implied vol",
            marker=dict(size=12, color=charts.ACCENT, symbol="diamond"),
        )
    )
    if lower > 0:
        fig.add_hline(
            y=lower,
            line=dict(color=charts.LOSS, dash="dot", width=1),
            annotation_text="lower bound",
            annotation_position="bottom right",
        )
    charts.show(
        charts.apply_layout(fig, None, "Volatility (%)", "Option value ($)"),
        key="iv_curve",
        alt=f"{style} {option_type} value against volatility, with the market price and the implied volatility",
    )
    st.caption(
        "The quote is checked against the no-arbitrage bounds first. European quotes are converted to their "
        "out-of-the-money equivalent through put–call parity and inverted with Jäckel's Let's Be Rational; American "
        f"quotes are solved with Brent's method on {engine_name if alo else 'the ' + tree_name}. Option value "
        "increases with volatility, so a solution is unique whenever vega is not zero."
    )
