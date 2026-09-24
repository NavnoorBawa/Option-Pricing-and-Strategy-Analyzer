"""Pricing & Greeks tab and the headline price cards."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from .. import black_scholes as bs
from . import cache, charts
from .format import md, money
from .inputs import Inputs


@dataclass(frozen=True)
class Prices:
    call: float
    put: float
    call_note: str = ""
    put_note: str = ""
    tree: dict | None = None
    mc: dict | None = None


def compute_prices(inputs: Inputs) -> Prices:
    """Model prices for the headline cards. Raises ``ValueError`` on invalid tree inputs."""
    S, K, T, r, sigma, q = (
        inputs.spot,
        inputs.strike,
        inputs.expiry,
        inputs.rate,
        inputs.volatility,
        inputs.dividend_yield,
    )
    model = inputs.model
    bs_call = bs.price(S, K, T, r, sigma, q, "call")
    bs_put = bs.price(S, K, T, r, sigma, q, "put")
    if model.kind == "binomial":
        tree = cache.tree_results(S, K, T, r, sigma, q, model.steps, model.american, model.tree_method)
        notes = {}
        for kind, ref in (("call", bs_call), ("put", bs_put)):
            diff = tree[kind].price - ref
            notes[kind] = (
                f"Early-exercise premium {money(diff, 4, signed=True)} vs European BSM"
                if model.american
                else f"Tree error vs BSM {money(diff, 4, signed=True)}"
            )
        return Prices(tree["call"].price, tree["put"].price, notes["call"], notes["put"], tree=tree)
    if model.kind == "monte_carlo":
        mc = cache.mc_results(S, K, T, r, sigma, q, model.n_paths, model.seed, model.antithetic, model.control_variate)
        notes = {
            kind: f"95% CI {money(res.ci95[0], 3)} – {money(res.ci95[1], 3)} · SE {money(res.std_error, 4)}"
            for kind, res in mc.items()
        }
        return Prices(mc["call"].price, mc["put"].price, notes["call"], notes["put"], mc=mc)
    return Prices(bs_call, bs_put, "Closed-form European value", "Closed-form European value")


def render_price_cards(inputs: Inputs, prices: Prices) -> None:
    S, K, T, r, q = inputs.spot, inputs.strike, inputs.expiry, inputs.rate, inputs.dividend_yield
    forward = S * np.exp((r - q) * T)
    parity = prices.call - prices.put - (S * np.exp(-q * T) - K * np.exp(-r * T))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Call value", money(prices.call, 4), border=True, help=md(prices.call_note))
    c1.caption(md(prices.call_note))
    c2.metric("Put value", money(prices.put, 4), border=True, help=md(prices.put_note))
    c2.caption(md(prices.put_note))
    c3.metric("Forward price", money(forward, 4), border=True, help="F = S·e^{(r−q)T}")
    c3.caption(f"Discount factor e^(−rT) = {np.exp(-r * T):.6f}")
    american = inputs.model.kind == "binomial" and inputs.model.american
    c4.metric(
        "Put-call parity gap",
        "n/a" if american else money(parity, 6),
        border=True,
        help="C − P − (S·e^{−qT} − K·e^{−rT}). Zero for European prices; American options only satisfy an inequality.",
    )
    c4.caption("American options: parity is an inequality" if american else "Should be ≈ 0 for European models")


def _greeks_table(inputs: Inputs, higher_order: bool) -> pd.DataFrame:
    args = (inputs.spot, inputs.strike, inputs.expiry, inputs.rate, inputs.volatility, inputs.dividend_yield)
    call = bs.greeks(*args, "call")
    put = bs.greeks(*args, "put")
    rows = []
    for name, info in bs.GREEKS.items():
        if info.order > 1 and not higher_order:
            continue
        rows.append(
            {
                "Greek": f"{info.label} {info.symbol}".strip(),
                "Call": call[name] * info.scale,
                "Put": put[name] * info.scale,
                "Unit": info.unit,
                "Definition": info.description,
                "Order": info.order,
            }
        )
    return pd.DataFrame(rows)


def _sensitivity(inputs: Inputs) -> None:
    quantities = {
        "Value": ("value", 1.0, "Option value ($)"),
        "Delta": ("delta", 1.0, "Delta"),
        "Gamma": ("gamma", 1.0, "Gamma"),
        "Theta / day": ("theta", 1 / 365, "Theta ($ per day)"),
        "Vega / vol pt": ("vega", 0.01, "Vega ($ per vol point)"),
        "Rho / 1%": ("rho", 0.01, "Rho ($ per 1% rate)"),
    }
    axes = {"Spot price": "spot", "Volatility": "vol", "Days to expiry": "days"}
    c1, c2 = st.columns(2)
    quantity = c1.selectbox("Quantity", list(quantities), key="sens_quantity")
    axis = c2.selectbox("Against", list(axes), key="sens_axis")
    name, scale, y_title = quantities[quantity]

    S, K, T, r, sigma, q = (
        inputs.spot,
        inputs.strike,
        inputs.expiry,
        inputs.rate,
        inputs.volatility,
        inputs.dividend_yield,
    )
    kind = axes[axis]
    if kind == "spot":
        lo, hi = 0.5 * min(S, K), 1.5 * max(S, K)
        x = np.linspace(lo, hi, 301)
        args, current, strike_x, x_title = (x, K, T, r, sigma, q), S, K, "Spot price"
    elif kind == "vol":
        x = np.linspace(0.01, max(1.0, 2.5 * sigma), 301)
        args, current, strike_x, x_title = (S, K, T, r, x, q), sigma * 100, None, "Volatility (%)"
    else:
        x = np.linspace(1, max(inputs.days * 1.5, 30), 301)
        args, current, strike_x, x_title = (S, K, x / 365.0, r, sigma, q), inputs.days, None, "Days to expiry"

    def curve(option_type):
        if name == "value":
            return bs.price(*args, option_type)
        return bs.greeks(*args, option_type)[name] * scale

    x_plot = x * 100 if kind == "vol" else x
    fig = charts.sensitivity_chart(x_plot, curve("call"), curve("put"), x_title, y_title, current, strike_x)
    st.plotly_chart(fig, key="sensitivity_chart")


def _heatmaps(inputs: Inputs) -> None:
    c1, c2 = st.columns(2)
    spot_span = c1.slider("Spot range (±%)", 5, 50, 20, 5, key="hm_spot") / 100
    vol_span = c2.slider("Volatility range (±%)", 10, 90, 50, 10, key="hm_vol") / 100
    spots = inputs.spot * np.linspace(1 - spot_span, 1 + spot_span, 9)
    vols = np.maximum(inputs.volatility * np.linspace(1 - vol_span, 1 + vol_span, 9), 1e-4)
    grid_s, grid_v = np.meshgrid(spots, vols)
    args = (inputs.strike, inputs.expiry, inputs.rate)
    left, right = st.columns(2)
    for col, kind in ((left, "call"), (right, "put")):
        values = bs.price(grid_s, args[0], args[1], args[2], grid_v, inputs.dividend_yield, kind)
        col.plotly_chart(
            charts.price_heatmap(
                spots, vols, values, f"{kind.capitalize()} value", "Blues" if kind == "call" else "Oranges"
            ),
            key=f"heatmap_{kind}",
        )


def _chain(inputs: Inputs) -> None:
    """Black-Scholes-Merton values and key Greeks across strikes at the sidebar expiry and volatility."""
    S, T, r, sigma, q = inputs.spot, inputs.expiry, inputs.rate, inputs.volatility, inputs.dividend_yield
    strikes = np.unique(np.round(inputs.strike * np.linspace(0.8, 1.2, 9), 2))
    call = bs.greeks(S, strikes, T, r, sigma, q, "call")
    put = bs.greeks(S, strikes, T, r, sigma, q, "put")
    chain = pd.DataFrame(
        {
            "Call value": bs.price(S, strikes, T, r, sigma, q, "call"),
            "Call delta": call["delta"],
            "Call theta / day": call["theta"] / 365,
            "Strike": strikes,
            "Put value": bs.price(S, strikes, T, r, sigma, q, "put"),
            "Put delta": put["delta"],
            "Put theta / day": put["theta"] / 365,
            "Gamma": call["gamma"],
            "Vega / vol pt": call["vega"] / 100,
        }
    )
    st.dataframe(
        chain,
        hide_index=True,
        column_config={
            "Strike": st.column_config.NumberColumn(format="%.2f"),
            "Call value": st.column_config.NumberColumn(format="%.4f"),
            "Put value": st.column_config.NumberColumn(format="%.4f"),
            "Call delta": st.column_config.NumberColumn(format="%.3f"),
            "Put delta": st.column_config.NumberColumn(format="%.3f"),
            "Call theta / day": st.column_config.NumberColumn(format="%.4f"),
            "Put theta / day": st.column_config.NumberColumn(format="%.4f"),
            "Gamma": st.column_config.NumberColumn(format="%.5f"),
            "Vega / vol pt": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    st.caption("Strikes from 80% to 120% of the sidebar strike, flat volatility (no smile), same expiry.")


def _model_diagnostics(inputs: Inputs, prices: Prices) -> None:
    S, K, T, r, sigma, q = (
        inputs.spot,
        inputs.strike,
        inputs.expiry,
        inputs.rate,
        inputs.volatility,
        inputs.dividend_yield,
    )
    model = inputs.model
    if model.kind == "binomial":
        tree = prices.tree
        st.markdown("**Tree Greeks** (read off the lattice, so they include the early-exercise feature)")
        rows = [
            {
                "Option": kind.capitalize(),
                "Value": res.price,
                "Delta": res.delta,
                "Gamma": res.gamma,
                "Theta / day": res.theta / 365,
                "Steps used": res.steps,
            }
            for kind, res in tree.items()
        ]
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            column_config={
                "Value": st.column_config.NumberColumn(format="%.4f"),
                "Delta": st.column_config.NumberColumn(format="%.4f"),
                "Gamma": st.column_config.NumberColumn(format="%.5f"),
                "Theta / day": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        st.markdown("**Convergence of the call value with the number of steps**")
        steps, series = cache.tree_convergence(S, K, T, r, sigma, q, model.steps, model.american)
        reference = None if model.american else bs.price(S, K, T, r, sigma, q, "call")
        st.plotly_chart(
            charts.convergence_chart(steps, series, reference, "Black-Scholes-Merton"), key="tree_convergence"
        )
        st.caption("CRR oscillates between odd and even step counts; Leisen-Reimer converges smoothly at ~O(1/n²).")
        if model.american:
            boundaries = {}
            for kind, res in tree.items():
                if res.exercise_boundary is not None and len(res.exercise_boundary[0]):
                    boundaries[f"{kind.capitalize()} boundary"] = (
                        res.exercise_boundary[0] * 365,
                        res.exercise_boundary[1],
                    )
            st.markdown("**Early-exercise boundary**")
            if boundaries:
                st.plotly_chart(charts.exercise_boundary_chart(boundaries, K), key="exercise_boundary")
                st.caption(
                    "Exercising immediately is optimal when the spot is beyond the boundary: below it for a put, "
                    "above it for a call. Without dividends an American call is never exercised early."
                )
            else:
                st.info("Early exercise is never optimal for these inputs (e.g. an American call without dividends).")
    elif model.kind == "monte_carlo":
        times, paths = cache.mc_paths(S, T, r, sigma, q, model.seed)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Sample paths** (200 of the simulated scenarios)")
            st.plotly_chart(charts.paths_chart(times * 365, paths[:60], K, paths.mean(axis=0)), key="mc_paths")
        with c2:
            st.markdown("**Terminal price distribution**")
            s_t = cache.mc_terminal(S, T, r, sigma, q, model.n_paths, model.seed, model.antithetic)
            grid = np.linspace(np.percentile(s_t, 0.05), np.percentile(s_t, 99.95), 400)
            mu = np.log(S) + (r - q - 0.5 * sigma**2) * T
            sd = sigma * np.sqrt(T)
            pdf = np.exp(-0.5 * ((np.log(grid) - mu) / sd) ** 2) / (grid * sd * np.sqrt(2 * np.pi))
            st.plotly_chart(
                charts.histogram_chart(
                    s_t, "Underlying price at expiry", [(K, "strike", charts.NEUTRAL)], pdf=(grid, pdf)
                ),
                key="mc_terminal",
            )
        mc = prices.mc
        rows = [
            {
                "Option": kind.capitalize(),
                "Monte Carlo": res.price,
                "Std. error": res.std_error,
                "95% CI low": res.ci95[0],
                "95% CI high": res.ci95[1],
                "BSM": bs.price(S, K, T, r, sigma, q, kind),
                "Error / SE": (res.price - bs.price(S, K, T, r, sigma, q, kind)) / res.std_error
                if res.std_error
                else 0.0,
            }
            for kind, res in mc.items()
        ]
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            column_config={
                c: st.column_config.NumberColumn(format="%.4f")
                for c in ("Monte Carlo", "Std. error", "95% CI low", "95% CI high", "BSM")
            }
            | {"Error / SE": st.column_config.NumberColumn(format="%.2f")},
        )
        st.caption(
            "Terminal prices are sampled exactly from the log-normal distribution (no time-stepping bias). "
            "|Error / SE| below about 2 is consistent with an unbiased estimator."
        )
    else:
        st.info(
            "Black-Scholes-Merton is exact for European options under its assumptions. Switch the sidebar model to a "
            "binomial tree or Monte Carlo to see convergence diagnostics, early exercise and simulated paths."
        )


def render(inputs: Inputs, prices: Prices) -> None:
    st.subheader("Greeks")
    higher = st.toggle("Show second- and third-order Greeks", value=False, key="higher_greeks")
    table = _greeks_table(inputs, higher)
    st.dataframe(
        table.drop(columns="Order"),
        hide_index=True,
        column_config={
            "Call": st.column_config.NumberColumn(format="%.6f"),
            "Put": st.column_config.NumberColumn(format="%.6f"),
        },
    )
    st.caption(
        "Analytical Black-Scholes-Merton Greeks for European exercise, in the units shown "
        "(theta and the other time Greeks per calendar day of elapsed time). "
        + (
            "Tree Greeks for the American option are under *Model diagnostics* below."
            if inputs.model.american and inputs.model.kind == "binomial"
            else ""
        )
    )

    st.subheader("Sensitivity explorer")
    _sensitivity(inputs)

    with st.expander("Price heatmaps (spot × volatility)"):
        _heatmaps(inputs)

    with st.expander("Theoretical option chain"):
        _chain(inputs)

    st.subheader("Model diagnostics")
    st.caption(inputs.model.label)
    _model_diagnostics(inputs, prices)
