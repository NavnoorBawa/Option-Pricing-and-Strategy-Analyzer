"""Implied Volatility tab: invert a market price with a robust bracketed solver."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from .. import black_scholes as bs
from ..implied_vol import ImpliedVolError, implied_volatility, no_arbitrage_bounds
from . import charts
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
        "The default quote is the Black-Scholes-Merton value at the sidebar volatility."
    )
    c1, c2 = st.columns([1, 2])
    option_type = c1.segmented_control(
        "Option", ["call", "put"], default="call", required=True, format_func=str.capitalize, key="iv_type"
    )
    model_value = bs.price(S, K, T, r, sigma, q, option_type)
    # Deliberately un-keyed: the quote resets to the model value when the contract changes.
    price = c2.number_input("Market price", min_value=0.0, value=float(model_value), step=0.05, format="%.4f")
    lower, upper = no_arbitrage_bounds(S, K, T, r, q, option_type)
    st.caption(md(f"No-arbitrage bounds for this {option_type}: {money(lower, 4)} ≤ price < {money(upper, 4)}"))

    try:
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
        money(bs.price(S, K, T, r, iv, q, option_type), 6),
        border=True,
        help="Black-Scholes-Merton value at the implied volatility (should equal the quote).",
    )
    c[2].metric("Vega at IV", money(vega, 4), border=True, help="Price change per 1 vol point at the solution.")
    time_value = price - lower
    c[3].metric("Time value", money(time_value, 4), border=True, help="Quote minus the lower no-arbitrage bound.")
    if vega < 1e-4 * max(S, K):
        st.warning(
            "Vega is tiny here, so the implied volatility is poorly determined: a small change in the quote "
            "moves it a lot. This is typical for deep in- or out-of-the-money options close to expiry."
        )

    vols = np.linspace(0.005, max(2.5 * iv, 2.0 * sigma, 0.6), 400)
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
        "Solved with Brent's method on the out-of-the-money equivalent (via put-call parity) after checking the "
        "no-arbitrage bounds; the price is strictly increasing in volatility, so the solution is unique."
    )
