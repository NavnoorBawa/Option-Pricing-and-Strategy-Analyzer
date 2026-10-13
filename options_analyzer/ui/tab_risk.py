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
from .tab_strategy import Position, money_short

CONFIDENCE_LEVELS = (0.95, 0.975, 0.99, 0.995, 0.999)
PATH_CHOICES = (10_000, 25_000, 50_000, 100_000, 250_000, 500_000)


def shock_label(x: float) -> str:
    """A relative shock as a percentage: whole percents when exact, otherwise one decimal; zero unsigned."""
    pct = 100.0 * x
    if abs(pct) < 1e-9:
        return "0%"
    if abs(pct - round(pct)) < 1e-9:
        return f"{pct:+.0f}%"
    return f"{pct:+.1f}%"


def price_labels(prices) -> list[str]:
    """Price labels with enough decimals to tell neighbouring grid columns apart."""
    prices = np.asarray(prices, dtype=float)
    step = float(np.min(np.diff(np.sort(prices)))) if prices.size > 1 else 0.0
    decimals = int(np.clip(np.ceil(-np.log10(step)) + 1, 2, 6)) if step > 0 else 2
    return [f"{p:,.{decimals}f}" for p in prices]


def _sigma_t(position: Position, front_days: float) -> float:
    """Standard deviation of the log price at the front expiry, at the sidebar volatility."""
    return position.market.volatility * np.sqrt(front_days / 365.0)


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
            [shock_label(x) for x in spot_changes],
            [f"IV {shock_label(x)}" for x in vol_changes],
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


def _date_grid(position: Position, front_days: float) -> None:
    sigma_t = _sigma_t(position, front_days)
    reach = st.select_slider(
        "Spot range (standard deviations of the price at the front expiry)",
        (1.0, 1.5, 2.0, 2.5, 3.0),
        value=2.0,
        format_func=lambda z: f"±{z:g}σ",
        key="date_grid_reach",
        help="Columns are spaced evenly in standard deviations σ√T of the log price at the front expiry, so the "
        "range follows the volatility and the time left.",
    )
    z = np.linspace(-reach, reach, 13)
    spot_changes = np.exp(z * sigma_t) - 1.0
    days = np.unique(np.round(np.linspace(0.0, front_days, 8)))
    grid = scenarios.pnl_grid(position.strategy, position.market, spot_changes, days=days, multiplier=position.units)
    rows = ["Today" if d == 0 else f"Day {d:.0f}" for d in days]
    rows[-1] = f"Expiry (day {days[-1]:.0f})" if np.isclose(days[-1], front_days) else rows[-1]
    st.plotly_chart(
        # Reversed so that today is the top row.
        charts.pnl_heatmap(
            price_labels(position.market.spot * (1.0 + spot_changes)), rows[::-1], grid[::-1], "Underlying price", ""
        ),
        key="date_heatmap",
    )
    st.caption(
        f"Columns run from {-reach:g}σ to +{reach:g}σ in 13 even steps of the log price, where σ√T = {sigma_t:.1%} "
        "is one standard deviation of the price at the front expiry (sidebar volatility); rows are days from "
        "today. Every cell re-prices all legs at the spot and date shown with implied volatility unchanged; values "
        "are P&L of the whole position versus its current model value."
    )


def _time_decay(position: Position, front_days: float) -> None:
    days = np.linspace(0, front_days, int(np.clip(round(front_days) + 1, 2, 200)))
    sigma_t = _sigma_t(position, front_days)
    zs = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    moves = np.exp(zs * sigma_t) - 1.0
    grid = scenarios.pnl_grid(position.strategy, position.market, moves, days=days, multiplier=position.units)
    series = {
        (f"Spot {z:+g}σ ({shock_label(m)})" if z else "Spot unchanged"): grid[:, i]
        for i, (z, m) in enumerate(zip(zs, moves, strict=True))
    }
    st.plotly_chart(charts.lines_chart(days, series, "Days from today", "P&L ($)"), key="time_decay")
    theta = strat.position_greeks(position.strategy, position.market)["theta"] * position.units / 365
    st.caption(
        md(
            "Mark-to-market P&L as time passes with implied volatility unchanged, for spot moves of half and one "
            f"standard deviation of the price at the front expiry (σ√T = {sigma_t:.1%}). "
            f"Current theta: {money(theta, signed=True)} per day."
        )
    )


GREEK_VIEWS = {
    "Delta": ("delta", 1.0, "Delta (shares)"),
    "Gamma": ("gamma", 1.0, "Gamma (delta per $1)"),
    "Theta": ("theta", 1.0 / 365.0, "Theta ($ per day)"),
    "Vega": ("vega", 0.01, "Vega ($ per vol point)"),
}


def _greek_profiles(position: Position, front_days: float) -> None:
    market = position.market
    greek = st.segmented_control("Greek", list(GREEK_VIEWS), default="Delta", key="risk_greek", required=True)
    name, scale, title = GREEK_VIEWS[greek]
    spots = market.spot * np.exp(np.linspace(-3.0, 3.0, 241) * _sigma_t(position, front_days))
    dates = [d for d in dict.fromkeys([0, round(front_days / 2), round(front_days) - 1]) if 0 <= d < front_days]
    series = {}
    for d in dates:
        values = strat.position_greeks(position.strategy, market, spots, d / 365.0)[name]
        series["Today" if d == 0 else f"In {d} days"] = np.broadcast_to(values * scale * position.units, spots.shape)
    st.plotly_chart(
        charts.lines_chart(spots, series, "Underlying price", title, vline=(market.spot, "spot")),
        key="greek_profiles",
    )
    st.caption(
        "Position Greeks of the whole position across ±3 standard deviations of the price at the front expiry, "
        "today and at later dates with implied volatility unchanged (analytical Black-Scholes-Merton Greeks)."
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
    # Chart and table are stacked at full width: side by side, the table hid its right-hand columns.
    st.plotly_chart(charts.bar_chart(table["Scenario"], table["P&L"], "P&L ($)"), key="stress_chart")
    columns = ["Scenario", "P&L", "Spot change", "IV multiplier", "IV shock damped", "Rate change (bp)"]
    if elapsed or custom_on:
        columns += ["Days elapsed", "Legs expired"]
    st.dataframe(
        table[columns],
        hide_index=True,
        height=35 * (len(table) + 1) + 3,  # show every scenario without an inner scrollbar
        column_config={
            "Scenario": st.column_config.TextColumn(width="medium"),
            "Spot change": st.column_config.NumberColumn(format="percent", width="small"),
            "IV multiplier": st.column_config.NumberColumn(
                format="%.2fx", width="small", help="Multiplier on implied vol."
            ),
            "IV shock damped": st.column_config.CheckboxColumn(
                width="small", help="One-month shock, damped for longer-dated legs (see the toggle above)."
            ),
            "Rate change (bp)": st.column_config.NumberColumn(format="%+.0f", width="small"),
            "P&L": st.column_config.NumberColumn(format="dollar", width="small"),
            "Days elapsed": st.column_config.NumberColumn(width="small"),
            "Legs expired": st.column_config.NumberColumn(
                width="small",
                help="Option legs that expire during the episode; they are settled on the end-of-episode price.",
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

    ci = md(f"{money(result.var_ci95[0])} to {money(result.var_ci95[1])}")
    c = st.columns(3)
    c[0].metric(
        f"VaR {confidence:.1%}",
        money_short(result.var),
        border=True,
        help=md(
            f"Loss not exceeded with {confidence:.1%} probability over {horizon} trading day(s). "
            f"Exact: {money(result.var)}; 95% simulation interval {money(result.var_ci95[0])} to "
            f"{money(result.var_ci95[1])}."
        ),
    )
    c[1].metric(
        f"Expected Shortfall {confidence:.1%}",
        money_short(result.expected_shortfall),
        border=True,
        help=md(f"Average loss in the worst tail beyond VaR. Exact: {money(result.expected_shortfall)}."),
    )
    c[2].metric(
        "Delta-normal VaR",
        money_short(result.delta_normal_var),
        border=True,
        help=md(
            "Linear approximation |Δ|·S·σ·√(h/252)·z. Differences from full revaluation show the "
            f"position's non-linearity (gamma, vega). Exact: {money(result.delta_normal_var)}."
        ),
    )
    st.caption(
        f"Simulation uncertainty: the 95% confidence interval of the VaR estimate is {ci} "
        f"(order statistics of the {n_paths:,} simulated outcomes)."
    )
    st.plotly_chart(
        charts.histogram_chart(
            result.pnl,
            f"P&L over {horizon} trading day(s) ($)",
            [(-result.var, f"VaR {confidence:.1%}", charts.LOSS), (-result.expected_shortfall, "ES", charts.ACCENT)],
            value_text=money,
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
        "VaR and ES use the Acerbi–Tasche estimators on the sorted P&L."
        + (" Excess kurtosis is unstable for ν ≤ 4." if distribution == "student_t" or garch_t else "")
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
    st.caption(
        "Every scenario, stress test and VaR path re-prices the legs as European options with "
        "Black-Scholes-Merton, whatever the sidebar pricing model."
    )
    front_days = strat.front_expiry(position.strategy, position.market) * 365

    st.subheader("Scenario P&L: spot × implied volatility")
    _scenario_grid(position, front_days)
    st.subheader("Scenario P&L: spot × date")
    _date_grid(position, front_days)
    st.subheader("Time decay")
    _time_decay(position, front_days)
    st.subheader("Position Greeks across spot")
    _greek_profiles(position, front_days)
    st.subheader("Stress tests")
    _stress_tests(position)
    st.subheader("Value at Risk & Expected Shortfall")
    _var(position)
