"""Implied Volatility tab: invert a market price with a robust bracketed solver."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from .. import black_scholes as bs
from ..implied_vol import ImpliedVolError, american_bounds, implied_volatility, no_arbitrage_bounds
from . import cache, charts
from .format import md, money, pct
from .inputs import Inputs


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
    st.caption(
        f"Uses the sidebar contract: S = {S:,.2f}, K = {K:,.2f}, {inputs.days} days, r = {r:.2%}, q = {q:.2%}. "
        "The default quote is the model value at the sidebar volatility."
    )
    c1, c2, c3 = st.columns([1, 1, 2])
    option_type = c1.segmented_control(
        "Option", ["call", "put"], default="call", required=True, format_func=str.capitalize, key="iv_type"
    )
    american_model = inputs.model.kind == "binomial" and inputs.model.american
    style = c2.segmented_control(
        "Exercise",
        ["European", "American"],
        default="American" if american_model else "European",
        required=True,
        key="iv_style",
        help="European quotes are inverted with Black-Scholes-Merton; American quotes (listed single-stock "
        "options) on a 501-step Leisen-Reimer tree.",
    )
    american = style == "American"

    def model_price(vol: float) -> float:
        if american:
            return cache.american_value(S, K, T, r, max(vol, 1e-6), q, option_type)
        return bs.price(S, K, T, r, vol, q, option_type)

    model_value = model_price(sigma)
    # Deliberately un-keyed: the quote resets to the model value when the contract changes.
    price = c3.number_input("Market price", min_value=0.0, value=float(model_value), step=0.05, format="%.4f")
    lower, upper = (
        american_bounds(S, K, T, r, q, option_type) if american else no_arbitrage_bounds(S, K, T, r, q, option_type)
    )
    st.caption(
        md(f"No-arbitrage bounds for this {style.lower()} {option_type}: {money(lower, 4)} ≤ price < {money(upper, 4)}")
    )

    try:
        if american:
            with st.spinner("Inverting the binomial tree…"):
                iv = cache.american_iv(price, S, K, T, r, q, option_type)
        else:
            iv = implied_volatility(price, S, K, T, r, q, option_type)
    except ImpliedVolError as exc:
        at_bound = min(abs(price - lower), abs(price - upper)) <= 1e-9 * max(S, K)
        if at_bound and abs(price - model_value) <= 1e-9 * max(S, K):
            st.info(
                "At these inputs the option value is numerically indistinguishable from a no-arbitrage bound, "
                "so the volatility cannot be recovered from the price (vega is effectively zero)."
            )
        else:
            st.error(f"No implied volatility exists: {exc}.")
        return

    if american:
        bump = 0.005
        vega = (model_price(iv + bump) - model_price(max(iv - bump, 1e-6))) / (iv + bump - max(iv - bump, 1e-6)) / 100
    else:
        vega = bs.greeks(S, K, T, r, max(iv, 1e-6), q, option_type)["vega"] / 100
    c = st.columns(4)
    c[0].metric(
        "Implied volatility",
        pct(iv, 3),
        f"{(iv - sigma) * 100:+.2f} vol pts vs sidebar",
        border=True,
        delta_color="off",
    )
    c[1].metric(
        "Model check",
        money(model_price(iv), 6),
        border=True,
        help="Model value at the implied volatility (should equal the quote).",
    )
    c[2].metric("Vega at IV", money(vega, 4), border=True, help="Price change per 1 vol point at the solution.")
    c[3].metric("Time value", money(price - lower, 4), border=True, help="Quote minus the lower bound.")
    if iv == 0.0:
        st.warning(
            "The quote equals the option's zero-volatility value (for an American option, typically the value of "
            "exercising now), so every volatility up to some level reproduces it and none can be singled out."
        )
    elif vega < 1e-4 * max(S, K):
        st.warning(
            "Vega is tiny here, so the implied volatility is poorly determined: a small change in the quote "
            "moves it a lot. This is typical for deep in- or out-of-the-money options close to expiry."
        )

    max_vol = max(2.5 * iv, 2.0 * sigma, 0.6)
    if american:
        vols, curve = cache.american_curve(S, K, T, r, q, option_type, max_vol)
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
    st.plotly_chart(charts.apply_layout(fig, None, "Volatility (%)", "Option value ($)"), key="iv_curve")
    st.caption(
        "The quote is checked against the no-arbitrage bounds first. European quotes are converted to their "
        "out-of-the-money equivalent through put-call parity and solved with Brent's method; American quotes are "
        "solved with Brent's method on the tree. Option value increases with volatility, so a solution is unique "
        "whenever vega is not zero."
    )
