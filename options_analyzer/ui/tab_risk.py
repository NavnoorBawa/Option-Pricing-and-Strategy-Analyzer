"""Risk & Scenarios tab: scenario grids, time decay, stress tests and VaR/ES.

Everything here analyses the position built in the Strategy Builder tab and
re-prices it leg by leg, so the numbers are consistent across the app.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .. import risk, scenarios
from .. import strategies as strat
from . import cache, charts
from .format import md, money
from .tab_strategy import Position

CONFIDENCE_LEVELS = (0.95, 0.975, 0.99, 0.995, 0.999)
PATH_CHOICES = (10_000, 25_000, 50_000, 100_000, 250_000, 500_000)


def _scenario_grid(position: Position, front_days: float) -> None:
    c1, c2 = st.columns(2)
    last_day = round(front_days)
    horizon = 0
    if last_day >= 1:
        horizon = c1.slider(
            "Days from today", 0, last_day, 0, key="grid_days", help="Time that passes before the shock is applied."
        )
    span = c2.slider("Spot range (±%)", 10, 60, 30, 5, key="grid_span")
    spot_changes = np.linspace(-span, span, 13) / 100
    vol_changes = np.array([-0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
    grid = scenarios.pnl_grid(position.strategy, position.market, spot_changes, vol_changes, horizon, position.units)
    st.plotly_chart(
        charts.pnl_heatmap(
            [f"{x:+.0%}" for x in spot_changes],
            [f"IV {x:+.0%}" for x in vol_changes],
            grid,
            "Spot change",
            "Implied-vol change",
        ),
        key="risk_heatmap",
    )
    st.caption(
        "Every cell re-prices all legs at the shocked spot and implied volatility (relative change) "
        f"after {horizon} day(s); values are P&L of the whole position versus its current model value."
    )


def _time_decay(position: Position, front_days: float) -> None:
    days = np.linspace(0, front_days, int(np.clip(round(front_days) + 1, 2, 200)))
    moves = np.array([-0.10, -0.05, 0.0, 0.05, 0.10])
    grid = scenarios.pnl_grid(position.strategy, position.market, moves, days=days, multiplier=position.units)
    series = {f"Spot {m:+.0%}" if m else "Spot unchanged": grid[:, i] for i, m in enumerate(moves)}
    st.plotly_chart(charts.lines_chart(days, series, "Days from today", "P&L ($)"), key="time_decay")
    theta = strat.position_greeks(position.strategy, position.market)["theta"] * position.units / 365
    st.caption(
        md(
            "Mark-to-market P&L as time passes with implied volatility unchanged. "
            f"Current theta: {money(theta, signed=True)} per day."
        )
    )


def _stress_tests(position: Position) -> None:
    c1, c2, c3 = st.columns(3)
    show_hist = c1.toggle("Historical analogues", value=True, key="stress_hist")
    show_hypo = c1.toggle("Hypothetical shocks", value=True, key="stress_hypo")
    elapsed = c2.toggle(
        "Include episode duration",
        value=False,
        key="stress_elapsed",
        help="Let the episode's calendar time pass too (options decay or expire).",
    )
    damp = c2.toggle(
        "Damp IV shock for longer tenors",
        value=True,
        key="stress_damp",
        help="Historical and Fed scenarios shock one-month (VIX-like) implied vol; longer-dated IV moves less, "
        f"so the shock is scaled by (T / 1 month)^-{scenarios.DEFAULT_VOL_TENOR_DECAY}. Hypothetical IV shocks "
        "always apply literally to every leg, like the heatmap above.",
    )
    with c3.popover("Custom scenario", icon=":material/tune:"):
        custom_on = st.toggle("Include custom scenario", value=False, key="stress_custom")
        spot = st.number_input("Spot change (%)", -99.0, 500.0, -15.0, 1.0, key="custom_spot")
        vol = st.number_input("IV multiplier", 0.1, 10.0, 1.5, 0.1, key="custom_vol")
        rate = st.number_input("Rate change (bp)", -1000.0, 1000.0, 0.0, 25.0, key="custom_rate")
        days = st.number_input("Days elapsed", 0, 3650, 0, 1, key="custom_days")

    presets = []
    if show_hist:
        presets += list(scenarios.HISTORICAL_SCENARIOS)
    if show_hypo:
        presets += list(scenarios.HYPOTHETICAL_SCENARIOS)
    decay = scenarios.DEFAULT_VOL_TENOR_DECAY if damp else 0.0
    rows = scenarios.run_stress_tests(position.strategy, position.market, presets, elapsed, position.units, decay)
    chosen = list(presets)
    if custom_on:
        custom = scenarios.StressScenario("Custom", spot / 100, vol, rate / 1e4, int(days), "User-defined shock.")
        # The custom scenario always applies its own elapsed time.
        rows += scenarios.run_stress_tests(position.strategy, position.market, [custom], True, position.units, decay)
        chosen.append(custom)
    if not rows:
        st.info("Select at least one group of scenarios.")
        return
    table = pd.DataFrame(rows)
    left, right = st.columns([1.1, 1])
    left.plotly_chart(charts.bar_chart(table["Scenario"], table["P&L"], "P&L ($)"), key="stress_chart")
    columns = ["Scenario", "P&L", "Spot change", "IV multiplier", "IV shock damped", "Rate change (bp)"]
    if elapsed or custom_on:
        columns += ["Days elapsed", "Legs expired"]
    right.dataframe(
        table[columns],
        hide_index=True,
        height=35 * (len(table) + 1) + 3,  # show every scenario without an inner scrollbar
        column_config={
            "Spot change": st.column_config.NumberColumn(format="percent"),
            "IV multiplier": st.column_config.NumberColumn(format="%.2fx", help="Multiplier on implied vol."),
            "IV shock damped": st.column_config.CheckboxColumn(
                help="One-month shock, damped for longer-dated legs (see the toggle above)."
            ),
            "Rate change (bp)": st.column_config.NumberColumn(format="%+.0f"),
            "P&L": st.column_config.NumberColumn(format="dollar"),
            "Legs expired": st.column_config.NumberColumn(
                help="Option legs that expire during the episode; they are settled on the end-of-episode price."
            ),
        },
    )
    with st.expander("Scenario sources and definitions"):
        for sc in chosen:
            st.markdown(f"- **{sc.name}** — {sc.description}")
        st.caption(
            "Historical analogues apply the episode's S&P 500 move, its VIX ratio (as a shock to one-month "
            "implied vol) and its policy-rate change to your position instantly, unless durations are "
            "included. They are stylised: single stocks can move far more than the index."
        )


def _var(position: Position) -> None:
    c1, c2, c3, c4 = st.columns(4)
    confidence = c1.selectbox(
        "Confidence",
        CONFIDENCE_LEVELS,
        index=2,
        format_func=lambda c: f"{c:.1%}",
        key="var_conf",
        help="97.5% is the Basel FRTB Expected Shortfall level; 99% is the classic VaR level.",
    )
    horizon = c2.number_input("Horizon (trading days)", 1, 60, 1, 1, key="var_horizon")
    n_paths = c3.select_slider(
        "Simulations", PATH_CHOICES, value=100_000, format_func=lambda n: f"{n:,}", key="var_paths"
    )
    distribution = c4.selectbox(
        "Return model", list(risk.DISTRIBUTIONS), format_func=risk.DISTRIBUTIONS.get, key="var_dist"
    )

    dof, alpha, beta, garch_t = 5.0, 0.09, 0.90, False
    if distribution == "student_t":
        c1, c2 = st.columns([1, 3])
        dof = c1.slider(
            "Degrees of freedom ν",
            3,
            30,
            5,
            key="var_dof",
            help="Lower = fatter tails. Daily equity returns are typically 3–6.",
        )
    elif distribution == "garch":
        c1, c2, c3, c4 = st.columns(4)
        alpha = c1.number_input("GARCH α", 0.0, 0.5, 0.09, 0.01, format="%.3f", key="garch_alpha")
        beta = c2.number_input("GARCH β", 0.0, 0.999, 0.90, 0.01, format="%.3f", key="garch_beta")
        garch_t = c3.toggle("Student-t innovations", value=False, key="garch_t")
        if garch_t:
            dof = c4.slider(
                "Degrees of freedom ν",
                3,
                30,
                6,
                key="var_dof_garch",
                help="GARCH-t fits to equity indices typically give ν ≈ 6–8.",
            )
        if alpha + beta >= 1:
            st.error("GARCH needs α + β < 1 for a finite long-run variance.")
            return
        persistence = alpha + beta
        half_life = f" (half-life {np.log(0.5) / np.log(persistence):.0f} trading days)" if persistence > 0 else ""
        st.caption(
            f"Persistence α + β = {persistence:.3f}{half_life}. "
            "Variance targeting: the long-run volatility equals the sidebar volatility."
        )

    c1, c2, c3 = st.columns(3)
    vol_dynamics = c1.toggle(
        "Simulate implied-vol moves",
        value=False,
        key="var_volmoves",
        help="Adds vega risk: log implied vol moves with the chosen vol-of-vol, correlated with spot.",
    )
    vov, corr = 0.0, -0.7
    if vol_dynamics:
        vov = (
            c2.number_input(
                "Vol of vol (% p.a.)",
                0.0,
                500.0,
                100.0,
                10.0,
                key="var_vov",
                help="Volatility of one-month implied vol; about 100% matches index (VIX) dynamics. Longer-dated "
                "legs move less, damped by tenor like the stress tests.",
            )
            / 100
        )
        corr = c3.slider(
            "Spot–vol correlation",
            -1.0,
            1.0,
            -0.7,
            0.05,
            key="var_corr",
            help="Equity indices: about −0.7 (spot down, implied vol up).",
        )
    seed = int(st.session_state.get("var_seed", 7))

    try:
        config = risk.VaRConfig(
            confidence=float(confidence),
            horizon_days=int(horizon),
            n_paths=int(n_paths),
            distribution=distribution,
            dof=float(dof),
            garch_alpha=float(alpha),
            garch_beta=float(beta),
            garch_t=garch_t,
            vol_of_vol=float(vov),
            spot_vol_correlation=float(corr),
            seed=seed,
        )
    except ValueError as exc:
        st.error(str(exc))
        return
    with st.spinner("Simulating…"):
        result = cache.value_at_risk(position.strategy, position.market, config, position.units)

    c = st.columns(4)
    c[0].metric(
        f"VaR {confidence:.1%}",
        money(result.var),
        border=True,
        help=f"Loss not exceeded with {confidence:.1%} probability over {horizon} trading day(s).",
    )
    c[1].metric(
        f"Expected Shortfall {confidence:.1%}",
        money(result.expected_shortfall),
        border=True,
        help="Average loss in the worst tail beyond VaR.",
    )
    c[2].metric(
        "VaR 95% CI (simulation)",
        md(f"{money(result.var_ci95[0], 0)} – {money(result.var_ci95[1], 0)}"),
        border=True,
        help="Monte Carlo uncertainty of the VaR estimate (order-statistic interval).",
    )
    c[3].metric(
        "Delta-normal VaR",
        money(result.delta_normal_var),
        border=True,
        help="Linear approximation |Δ|·S·σ·√(h/252)·z. Differences from full revaluation show the "
        "position's non-linearity (gamma, vega).",
    )
    st.plotly_chart(
        charts.histogram_chart(
            result.pnl,
            f"P&L over {horizon} trading day(s) ($)",
            [(-result.var, f"VaR {confidence:.1%}", charts.LOSS), (-result.expected_shortfall, "ES", charts.ACCENT)],
        ),
        key="var_hist",
    )
    stats = pd.DataFrame(
        [
            {"Statistic": "Mean P&L", "Value": money(result.mean, signed=True)},
            {"Statistic": "Standard deviation", "Value": money(result.std)},
            {
                "Statistic": "Skewness",
                "Value": "n/a (riskless)" if np.isnan(result.skewness) else f"{result.skewness:.3f}",
            },
            {
                "Statistic": "Excess kurtosis",
                "Value": "n/a (riskless)" if np.isnan(result.excess_kurtosis) else f"{result.excess_kurtosis:.3f}",
            },
            {"Statistic": "Worst simulated P&L", "Value": money(result.worst, signed=True)},
            {"Statistic": "Best simulated P&L", "Value": money(result.best, signed=True)},
        ]
    )
    c1, c2 = st.columns([1, 1.4])
    c1.dataframe(stats, hide_index=True)
    c2.caption(
        f"Full revaluation of every leg on {n_paths:,} simulated {risk.DISTRIBUTIONS[distribution]} paths "
        f"(zero drift, volatility {position.market.volatility:.1%}), with {horizon} trading day(s) of time decay. "
        "VaR and ES use the Acerbi–Tasche estimators on the sorted P&L. Excess kurtosis is unstable for ν ≤ 4."
        + (
            f" {result.settled_legs} option leg(s) expire before the horizon and are settled on the simulated price "
            "at the close of their expiry day."
            if result.settled_legs
            else ""
        )
    )
    c2.number_input("Random seed", 0, 2**32 - 1, 7, 1, key="var_seed")


@st.fragment
def render(position: Position | None) -> None:
    if position is None:
        st.info("Build a valid position in the **Strategy Builder** tab to analyse its risk.")
        return
    st.markdown(f"Analysing {position.describe()}")
    front_days = strat.front_expiry(position.strategy, position.market) * 365

    st.subheader("Scenario P&L: spot × implied volatility")
    _scenario_grid(position, front_days)
    st.subheader("Time decay")
    _time_decay(position, front_days)
    st.subheader("Stress tests")
    _stress_tests(position)
    st.subheader("Value at Risk & Expected Shortfall")
    _var(position)
