"""Pricing tab and the headline price cards."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import ndtr

from .. import black_scholes as bs
from . import cache, charts
from .components import card_row
from .format import md, money, pct
from .inputs import Inputs

#: Monte Carlo scenarios drawn on the sample-paths chart.
SAMPLE_PATHS = 60
#: The confidence interval of a simulated price, in no-break spaces: it never splits over two lines (on a phone "SE"
#: stood alone on the second line of the caption under the price cards).
_CI_FORMULA = "95% CI = price ± 1.96 SE".replace(" ", "\xa0")
#: What a heavy-tailed Monte Carlo sample does instead of the usual estimator (see ``monte_carlo``).
HEAVY_TAIL_NOTE = (
    "Heavy-tailed sample (large σ√T): the put is averaged directly and the call follows from put-call parity"
)


@dataclass(frozen=True)
class Prices:
    call: float
    put: float
    call_note: str = ""
    put_note: str = ""
    tree: dict | None = None
    mc: dict | None = None
    #: Values on the same tree with European exercise (American tree only).
    european_tree: dict | None = None


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
        # The early-exercise premium is measured on the same lattice, so the tree's discretisation error cancels.
        european = (
            cache.tree_prices(S, K, T, r, sigma, q, model.steps, False, model.tree_method) if model.american else None
        )
        notes = {}
        for kind, ref in (("call", bs_call), ("put", bs_put)):
            if model.american:
                premium = tree[kind].price - european[kind]
                notes[kind] = f"Early-exercise premium {money(premium, 4, signed=True)} (same tree, European exercise)"
            else:
                notes[kind] = f"Tree error vs BSM {money(tree[kind].price - ref, 6, signed=True)}"
        return Prices(
            tree["call"].price, tree["put"].price, notes["call"], notes["put"], tree=tree, european_tree=european
        )
    if model.kind == "monte_carlo":
        mc = cache.mc_results(S, K, T, r, sigma, q, model.n_paths, model.seed, model.antithetic, model.control_variate)
        notes = {
            kind: (f"{HEAVY_TAIL_NOTE}. " if res.heavy_tailed else "")
            + (
                f"95% CI {money(res.ci95[0], 3)} – {money(res.ci95[1], 3)} · SE {money(res.std_error, 4)}"
                if np.isfinite(res.std_error)
                else "Every simulated price finished on one side of the strike: the error cannot be estimated"
            )
            for kind, res in mc.items()
        }
        return Prices(mc["call"].price, mc["put"].price, notes["call"], notes["put"], mc=mc)
    return Prices(bs_call, bs_put, "Closed-form European value", "Closed-form European value")


def _model_card(inputs: Inputs, prices: Prices) -> tuple[str, str, str, str]:
    """``(label, value, caption, help)`` of the fourth headline card: what the sidebar model adds or how far it
    is from the exact answer. (Put-call parity would read zero for every European model by construction.) The
    caption under the cards gives the call's and the put's figures where the card shows the larger of the two; it
    is empty for the closed form, whose card holds the call's figure and its tooltip the put's."""
    S, K, T, r, sigma, q = (
        inputs.spot,
        inputs.strike,
        inputs.expiry,
        inputs.rate,
        inputs.volatility,
        inputs.dividend_yield,
    )
    model = inputs.model
    if model.kind == "binomial" and model.american:
        premium = {kind: prices.tree[kind].price - prices.european_tree[kind] for kind in ("call", "put")}
        larger = max(premium, key=lambda kind: abs(premium[kind]))
        return (
            "Exercise premium",
            money(premium[larger], 4),
            f"Early-exercise premium: call {money(premium['call'], 4, signed=True)}, "
            f"put {money(premium['put'], 4, signed=True)} (against the same tree with European exercise)",
            "American value minus the value on the same tree with European exercise (the larger of the call's and "
            "the put's is shown). Measured on one lattice, so the tree's discretisation error cancels.",
        )
    if model.kind == "binomial":
        error = {kind: prices.tree[kind].price - bs.price(S, K, T, r, sigma, q, kind) for kind in ("call", "put")}
        return (
            "Tree error vs BSM",
            money(max(abs(e) for e in error.values()), 6),
            f"Error against the closed-form value: call {money(error['call'], 6, signed=True)}, "
            f"put {money(error['put'], 6, signed=True)}",
            "Tree value minus the closed-form Black-Scholes-Merton value, the exact answer for European exercise "
            "(the larger of the call's and the put's is shown). It shrinks as the number of steps grows.",
        )
    if model.kind == "monte_carlo":
        se = {kind: prices.mc[kind].std_error for kind in ("call", "put")}
        finite = [v for v in se.values() if np.isfinite(v)]
        heavy = any(res.heavy_tailed for res in prices.mc.values())
        return (
            "Std. error",
            money(max(finite), 4) if finite else "–",
            f"Standard error: call {money(se['call'], 4)}, put {money(se['put'], 4)}; {_CI_FORMULA}"
            + (f". {HEAVY_TAIL_NOTE}." if heavy else ""),
            "Standard error of the simulated prices, after any variance reduction in use (the header's badge names "
            "it: AV for antithetic variates, CV for a control variate). The larger of the call's and the put's is "
            "shown; it falls as one over the square root of the number of paths.",
        )
    vst = sigma * np.sqrt(T)
    if vst > 0:
        itm = float(ndtr((np.log(S / K) + (r - q - 0.5 * sigma**2) * T) / vst))
    else:
        itm = float(np.sign(S * np.exp((r - q) * T) - K) + 1.0) / 2.0  # certain, or 1/2 exactly at the strike
    return (
        "Call ITM probability",
        pct(itm, 2),
        "",
        "Probability that the call finishes in the money under the pricing (risk-neutral) measure, N(d₂); the "
        f"put's is 1 − N(d₂) = {pct(1.0 - itm, 2)}. It uses the risk-free drift, so it is not a real-world forecast.",
    )


#: Keys of the headline price-card row and of its two pairs.
PRICE_CARDS = ("price_cards", ("price_cards_prices", "price_cards_model"))


def render_price_cards(inputs: Inputs, prices: Prices) -> None:
    S, T, r, q = inputs.spot, inputs.expiry, inputs.rate, inputs.dividend_yield
    forward = S * np.exp((r - q) * T)
    label, value, caption, help_text = _model_card(inputs, prices)
    # The per-option notes are in the tooltips so the cards stay short on narrow screens.
    cards = [
        ("Call value", money(prices.call, 4), md(prices.call_note)),
        ("Put value", money(prices.put, 4), md(prices.put_note)),
        ("Forward price", money(forward, 4), f"F = S·e^((r−q)T). Discount factor e^(−rT) = {np.exp(-r * T):.6f}."),
        (label, value, md(help_text)),
    ]
    card_row(*PRICE_CARDS, [(name, text, {"help": help_text}) for name, text, help_text in cards])
    # The header's badge names the model, so the caption only adds what the cards do not show.
    if caption:
        st.caption(md(caption))


def _european_note(inputs: Inputs) -> str:
    """Caption for views that always use Black-Scholes-Merton, whatever the sidebar model."""
    model = inputs.model
    if model.kind == "bsm":
        return ""
    early = " Early exercise is ignored here." if model.kind == "binomial" and model.american else ""
    return (
        f"Closed-form Black-Scholes-Merton values for European exercise, not the sidebar's {model.label} "
        f"(used for the price cards and the model diagnostics).{early}"
    )


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
    quantity = c1.selectbox("Quantity", list(quantities), key="sens_quantity", persist_state="session")
    axis = c2.selectbox("Against", list(axes), key="sens_axis", persist_state="session")
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
    charts.show(fig, key="sensitivity_chart", alt=f"Call and put {y_title.lower()} against {x_title.lower()}")
    if note := _european_note(inputs):
        st.caption(note)


def _heatmaps(inputs: Inputs) -> None:
    c1, c2 = st.columns(2)
    spot_span = c1.slider("Spot range (±%)", 5, 50, 20, 5, key="hm_spot", persist_state="session") / 100
    vol_span = c2.slider("Volatility range (±%)", 10, 90, 50, 10, key="hm_vol", persist_state="session") / 100
    spots = inputs.spot * np.linspace(1 - spot_span, 1 + spot_span, 9)
    vols = np.maximum(inputs.volatility * np.linspace(1 - vol_span, 1 + vol_span, 9), 1e-4)
    grid_s, grid_v = np.meshgrid(spots, vols)
    args = (inputs.strike, inputs.expiry, inputs.rate)
    # One above the other at full width: side by side, each cell was too narrow for its value (5 to 9 px text).
    for kind in ("call", "put"):
        values = bs.price(grid_s, args[0], args[1], args[2], grid_v, inputs.dividend_yield, kind)
        charts.show(
            charts.price_heatmap(spots, vols, values, f"{kind.capitalize()} value ($)"),
            key=f"heatmap_{kind}",
            alt=f"Heatmap of the {kind} value ($) over spot price and volatility",
        )
    if note := _european_note(inputs):
        st.caption(note)


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
    st.caption(
        "Strikes from 80% to 120% of the sidebar strike, flat volatility (no smile), same expiry. "
        + _european_note(inputs)
    )


def _tree_diagnostics(inputs: Inputs, prices: Prices) -> None:
    S, K, T, r, sigma, q = (
        inputs.spot,
        inputs.strike,
        inputs.expiry,
        inputs.rate,
        inputs.volatility,
        inputs.dividend_yield,
    )
    model = inputs.model
    st.markdown("**Tree Greeks** (on the lattice, so they include the early-exercise feature)")
    rows = [
        {
            "Option": kind.capitalize(),
            "Value": res.price,
            "Delta": res.delta,
            "Gamma": res.gamma,
            "Theta / day": res.theta / 365,
            "Vega / vol pt": res.vega / 100,
            "Rho / 1%": res.rho / 100,
            "Div. rho / 1%": res.epsilon / 100,
        }
        for kind, res in prices.tree.items()
    ]
    steps_used = prices.tree["call"].steps
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        column_config={
            "Value": st.column_config.NumberColumn(format="%.4f"),
            "Delta": st.column_config.NumberColumn(format="%.4f"),
            "Gamma": st.column_config.NumberColumn(format="%.5f"),
            "Theta / day": st.column_config.NumberColumn(format="%.4f"),
            "Vega / vol pt": st.column_config.NumberColumn(format="%.4f"),
            "Rho / 1%": st.column_config.NumberColumn(format="%.4f"),
            "Div. rho / 1%": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    st.caption(
        f"Dollars per share on a {steps_used:,}-step tree. Delta and gamma come from its first two time steps, "
        "theta from the second step (CRR) or from repricing with the expiry moved (Leisen-Reimer). Vega, rho and "
        "dividend rho (per 1% move in the dividend yield) reprice on a tree with the same steps, with the "
        "volatility moved ±0.5 vol pt and the rates ±25 bp."
    )

    st.markdown("**Convergence with the number of steps**")
    series, references = cache.tree_convergence(S, K, T, r, sigma, q, model.steps, model.american)
    reference_label = "Converged" if model.american else "Black-Scholes-Merton"
    reference = "converged American value" if model.american else "Black-Scholes-Merton value"
    for col, kind in zip(st.columns(2), ("call", "put"), strict=True):
        charts.show(
            charts.convergence_chart(series[kind], references[kind], reference_label, kind.capitalize()),
            key=f"tree_convergence_{kind}",
            alt=f"{kind.capitalize()} value on each lattice against the number of tree steps, with the {reference}",
            container=col,
        )
    st.caption(
        f"Step counts from 10 to the sidebar's {model.steps:,}. CRR oscillates between odd and even step counts; "
        "Leisen-Reimer converges smoothly, "
        + (
            "at first order with early exercise. The converged value extrapolates Leisen-Reimer trees with 2,001 "
            "and 4,003 steps (Richardson), kept within the no-arbitrage bounds of an American option."
            if model.american
            else "at roughly O(1/n²), to the closed-form value."
        )
    )
    if model.american:
        boundaries = {}
        for kind, res in prices.tree.items():
            if res.exercise_boundary is not None and len(res.exercise_boundary[0]):
                boundaries[f"{kind.capitalize()} boundary"] = (
                    res.exercise_boundary[0] * 365,
                    res.exercise_boundary[1],
                )
        st.markdown("**Early-exercise boundary**")
        if boundaries:
            charts.show(
                charts.exercise_boundary_chart(boundaries, K),
                key="exercise_boundary",
                alt="Early-exercise boundary: the critical spot price against days from today, with the strike",
            )
            st.caption(
                "Exercising immediately is optimal when the spot is beyond the boundary: below it for a put, "
                "above it for a call. With a non-negative rate and no dividends an American call is never "
                "exercised early. The critical price at each step is interpolated between the last exercised "
                "node and the next one, and consecutive steps are averaged (their nodes interleave)."
            )
        else:
            st.info(
                "Early exercise is never optimal for these inputs, or the spot is already in the exercise region "
                "at every step the tree resolves (e.g. a deep in-the-money put worth its intrinsic value)."
            )


def _monte_carlo_diagnostics(inputs: Inputs, prices: Prices) -> None:
    S, K, T, r, sigma, q = (
        inputs.spot,
        inputs.strike,
        inputs.expiry,
        inputs.rate,
        inputs.volatility,
        inputs.dividend_yield,
    )
    model = inputs.model
    times, paths = cache.mc_paths(S, T, r, sigma, q, model.n_paths, model.seed, model.antithetic, SAMPLE_PATHS)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Sample paths** ({len(paths)} of the {model.n_paths:,} scenarios behind the price)")
        charts.show(
            charts.paths_chart(times * 365, paths, K, S * np.exp((r - q) * times), "Risk-neutral mean (forward)"),
            key="mc_paths",
            alt=f"{len(paths)} simulated price paths and their risk-neutral mean (the forward) against days from "
            "today, with the strike",
        )
        st.caption(
            "The price samples each scenario's price at expiry exactly; its path from today is filled in with a "
            "Brownian bridge. The purple line is the forward S·e^((r−q)t), the expected price of a scenario at each "
            "date under the pricing measure; the average of the simulated scenarios differs from it by sampling noise."
        )
    with c2:
        st.markdown("**Terminal price distribution**")
        s_t = cache.mc_terminal(S, T, r, sigma, q, model.n_paths, model.seed, model.antithetic)
        grid = np.linspace(np.percentile(s_t, 0.05), np.percentile(s_t, 99.95), 400)
        mu = np.log(S) + (r - q - 0.5 * sigma**2) * T
        sd = sigma * np.sqrt(T)
        pdf = np.exp(-0.5 * ((np.log(grid) - mu) / sd) ** 2) / (grid * sd * np.sqrt(2 * np.pi))
        charts.show(
            charts.histogram_chart(s_t, "Underlying price at expiry", [(K, "strike", charts.NEUTRAL)], pdf=(grid, pdf)),
            key="mc_terminal",
            alt="Histogram of the simulated underlying prices at expiry, with the log-normal density and the strike",
        )
    rows = [
        {
            "Option": kind.capitalize(),
            "Monte Carlo": res.price,
            "Std. error": res.std_error,
            "95% CI low": res.ci95[0],
            "95% CI high": res.ci95[1],
            "BSM": bs.price(S, K, T, r, sigma, q, kind),
            "Error / SE": (res.price - bs.price(S, K, T, r, sigma, q, kind)) / res.std_error
            if res.std_error > 0
            else np.nan,
            "In the money": res.itm_fraction,
        }
        for kind, res in prices.mc.items()
    ]
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        placeholder="",  # undefined values are left blank, as the caption says (Streamlit would write "None")
        column_config={
            c: st.column_config.NumberColumn(format="%.4f")
            for c in ("Monte Carlo", "Std. error", "95% CI low", "95% CI high", "BSM")
        }
        | {
            "Error / SE": st.column_config.NumberColumn(format="%.2f"),
            "In the money": st.column_config.NumberColumn(format="percent"),
        },
    )
    st.caption(
        "Terminal prices are sampled exactly from the log-normal distribution (no time-stepping bias). "
        "|Error / SE| below about 2 is consistent with an unbiased estimator. The standard error is left blank "
        "when the sample cannot estimate it: every price finished out of the money, or every one in the money "
        "with the control variate on. At very large σ√T (hundreds of percent of volatility over years) the mean of "
        "the price at expiry rests on scenarios too rare to draw: when the simulated discounted prices miss their "
        "known mean S·e^(−qT) by more than 5 standard errors, the put, whose payoff is at most its strike, is "
        "averaged directly (no control variate) and the call follows from put-call parity. Without the control "
        "variate the call's own average fails much sooner: from σ√T = 2 its 95% interval misses the value more "
        "often than 1 run in 20 (about 1 in 5 at σ√T = 3), so from σ√T = 2 on the call follows from the put by "
        "parity as well. Every estimate is kept within the no-arbitrage bounds of a European option."
        + (
            " This sample is heavy-tailed: the prices above come from the put's direct average and parity."
            if any(res.heavy_tailed for res in prices.mc.values())
            else ""
        )
    )


def _model_diagnostics(inputs: Inputs, prices: Prices) -> None:
    if inputs.model.kind == "binomial":
        _tree_diagnostics(inputs, prices)
    elif inputs.model.kind == "monte_carlo":
        _monte_carlo_diagnostics(inputs, prices)
    else:
        st.info(
            "Black-Scholes-Merton is exact for European options under its assumptions. Switch the sidebar model to a "
            "binomial tree or Monte Carlo to see convergence diagnostics, early exercise and simulated paths."
        )


def render(inputs: Inputs, prices: Prices) -> None:
    st.subheader("Greeks")
    higher = st.toggle("Show second- and third-order Greeks", value=False, key="higher_greeks", persist_state="session")
    table = _greeks_table(inputs, higher)
    st.dataframe(
        table.drop(columns="Order"),
        hide_index=True,
        column_config={
            # A fixed width: the grid sizes columns from the first rows it draws (and drops the longest name as an
            # outlier), which cut "Epsilon (dividend rho) ε" and, after the toggle, "Vomma (volga)" on phones.
            "Greek": st.column_config.TextColumn(width="medium"),
            "Call": st.column_config.NumberColumn(format="%.6f"),
            "Put": st.column_config.NumberColumn(format="%.6f"),
        },
    )
    st.caption(
        "Analytical Black-Scholes-Merton Greeks for European exercise, in the units shown "
        "(theta and the other time Greeks per calendar day of elapsed time). "
        + (
            "Tree Greeks for the American option, including vega and rho, are under *Model diagnostics* below."
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
