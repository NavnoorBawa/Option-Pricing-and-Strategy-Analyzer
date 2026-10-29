"""Risk tab: scenario grids, time decay, stress tests and VaR/ES.

Everything here analyses the position built in the Strategy tab and
re-prices it leg by leg, so the numbers are consistent across the app.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import streamlit as st

from .. import risk, scenarios
from .. import strategies as strat
from . import cache, charts
from .components import Card, card_row, current_key
from .format import MINUS, SHORT_SCALES, md, money, money_short, signed
from .tab_strategy import Position

CONFIDENCE_LEVELS = (0.95, 0.975, 0.99, 0.995, 0.999)
PATH_CHOICES = (10_000, 25_000, 50_000, 100_000, 250_000, 500_000)
STRESS_HEADERS = {"IV shock damped": "IV damped", "Rate change (bp)": "Rate Δ (bp)"}
#: The VaR simulation's default implied-vol dynamics, an equity index's (VIX-like): one-month implied vol with a
#: volatility of 100% a year, moving against spot with a correlation of −0.7. With implied vol frozen instead, a
#: 45-day at-the-money short straddle on an index (spot 5,800, vol 16%, r 4.3%, q 1.3%, one 100-share contract)
#: shows a one-day 99% VaR of $2,833 (ES $3,683) against $3,862 ($4,737) with these dynamics (100,000 paths, seed
#: 7): 27% less, on the most common vol position.
INDEX_VOL_OF_VOL = 1.0
INDEX_SPOT_VOL_CORRELATION = -0.7


def shock_label(x: float) -> str:
    """A relative shock as a percentage: whole percents when exact, otherwise one decimal; zero unsigned."""
    pct = 100.0 * x
    if abs(pct) < 1e-9:
        return "0%"
    if abs(pct - round(pct)) < 1e-9:
        return f"{pct:+.0f}%"
    return f"{pct:+.1f}%"


def price_labels(prices) -> list[str]:
    """Price labels that tell every grid point apart (a category axis merges equal labels).

    Ordinary grids get one fixed number of decimals, at least two; grids that need more than six decimals, such
    as prices spanning many orders of magnitude at extreme volatility, get as many significant digits as needed.
    """
    prices = np.asarray(prices, dtype=float)
    unique = np.unique(prices).size
    for decimals in range(2, 7):
        labels = [f"{p:,.{decimals}f}" for p in prices]
        if len(set(labels)) == unique:
            return labels
    for digits in range(3, 18):
        labels = [f"{p:,.{digits}g}" for p in prices]
        if len(set(labels)) == unique:
            break
    return labels


def _sigma_t(position: Position, front_days: float) -> float:
    """Standard deviation of the log price at the front expiry, at the sidebar volatility."""
    return position.market.volatility * np.sqrt(front_days / 365.0)


def _scenario_grid(position: Position, front_days: float) -> None:
    c1, c2 = st.columns(2)
    last_day = round(front_days)
    horizon = 0
    if last_day >= 1:
        horizon = c1.slider(
            "Days from today",
            0,
            last_day,
            0,
            key=current_key("grid_days_", f"grid_days_{last_day}"),  # with the range, as on the Strategy tab
            persist_state="session",
            help="Time that passes before the shock is applied.",
        )
    span = c2.slider("Spot range (±%)", 10, 60, 30, 5, key="grid_span", persist_state="session")
    spot_changes = np.linspace(-span, span, 13) / 100
    vol_changes = np.array([-0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
    grid = scenarios.pnl_grid(position.strategy, position.market, spot_changes, vol_changes, horizon, position.units)
    charts.show(
        # Implied-vol shocks (6) are the columns and spot shocks the rows, highest at the top, without the colour
        # scale (the cells say it), as in the spot × date grid: with 13 spot columns and a colour bar the cell values
        # shrank to 4 px on a phone.
        charts.pnl_heatmap(
            [f"IV {shock_label(x)}" for x in vol_changes],
            [shock_label(x) for x in spot_changes],
            grid.T,
            "Implied-vol change",
            "Spot change",
            colorbar=False,
        ),
        key="risk_heatmap",
        alt="Heatmap of the position's P&L ($) by spot change (rows) and implied-vol change (columns)",
    )
    st.caption(
        "Every cell re-prices all legs at the shocked spot and implied volatility (relative change) "
        f"after {horizon} day(s); values are P&L of the whole position versus its current model value, in "
        "thousands (k) or millions (M) of dollars when large (hover a cell for the exact amount)."
    )


def _date_grid(position: Position, front_days: float) -> None:
    sigma_t = _sigma_t(position, front_days)
    reach = st.select_slider(
        "Spot range (standard deviations of the price at the front expiry)",
        (1.0, 1.5, 2.0, 2.5, 3.0),
        value=2.0,
        format_func=lambda z: f"±{z:g}σ",
        key="date_grid_reach",
        persist_state="session",
        help="Rows are spaced evenly in standard deviations σ√T of the log price at the front expiry, so the "
        "range follows the volatility and the time left.",
    )
    z = np.linspace(-reach, reach, 13)
    spot_changes = np.exp(z * sigma_t) - 1.0
    days = np.unique(np.round(np.linspace(0.0, front_days, 8)))
    grid = scenarios.pnl_grid(position.strategy, position.market, spot_changes, days=days, multiplier=position.units)
    dates = ["Today" if d == 0 else f"Day {d:.0f}" for d in days]
    at_expiry = bool(np.isclose(days[-1], front_days))
    if at_expiry:
        dates[-1] = "Expiry"
    charts.show(
        # Dates (at most 8) are the columns and prices the rows, highest at the top, so that the cells stay wide
        # enough to read on a phone.
        charts.pnl_heatmap(
            dates,
            price_labels(position.market.spot * (1.0 + spot_changes)),
            grid.T,
            "Date",
            "Underlying price",
            colorbar=False,
        ),
        key="date_heatmap",
        alt="Heatmap of the position's P&L ($) by underlying price (rows) and date (columns)",
    )
    expiry = f", up to the front expiry on day {days[-1]:.0f}" if at_expiry else ""
    st.caption(
        f"Rows run from {-reach:g}σ to +{reach:g}σ in 13 even steps of the log price, where σ√T = {sigma_t:.1%} is "
        "one standard deviation of the price at the front expiry (sidebar volatility); columns are days from "
        f"today{expiry}. Every cell re-prices all legs at the spot and date shown with implied volatility "
        "unchanged; values are P&L of the whole position versus its current model value, in thousands (k) or "
        "millions (M) of dollars when large (hover a cell for the exact amount)."
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
    fig = charts.lines_chart(days, series, "Days from today", "P&L ($)", styles=charts.scenario_styles(len(series)))
    charts.show(
        fig,
        key="time_decay",
        alt="The position's P&L ($) against days from today, for spot moves of up to one standard deviation",
    )
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
    greek = st.segmented_control(
        "Greek", list(GREEK_VIEWS), default="Delta", key="risk_greek", required=True, persist_state="session"
    )
    name, scale, title = GREEK_VIEWS[greek]
    spots = market.spot * np.exp(np.linspace(-3.0, 3.0, 241) * _sigma_t(position, front_days))
    dates = [d for d in dict.fromkeys([0, round(front_days / 2), round(front_days) - 1]) if 0 <= d < front_days]
    series = {}
    for d in dates:
        values = strat.position_greeks(position.strategy, market, spots, d / 365.0)[name]
        series["Today" if d == 0 else f"In {d} days"] = np.broadcast_to(values * scale * position.units, spots.shape)
    charts.show(
        charts.lines_chart(spots, series, "Underlying price", title, vline=(market.spot, "spot")),
        key="greek_profiles",
        alt=f"The position's {greek.lower()} against the underlying price, today and at later dates",
    )
    st.caption(
        "Position Greeks of the whole position across ±3 standard deviations of the price at the front expiry, "
        "today and at later dates with implied volatility unchanged (analytical Black-Scholes-Merton Greeks)."
    )


def _stress_tests(position: Position) -> None:
    c1, c2, c3 = st.columns(3)
    show_hist = c1.toggle("Historical analogues", value=True, key="stress_hist", persist_state="session")
    show_hypo = c1.toggle("Hypothetical shocks", value=True, key="stress_hypo", persist_state="session")
    elapsed = c2.toggle(
        "Include episode duration",
        value=False,
        key="stress_elapsed",
        persist_state="session",
        help="Let the episode's calendar time pass too (options decay or expire).",
    )
    damp = c2.toggle(
        "Damp IV shock for longer tenors",
        value=True,
        key="stress_damp",
        persist_state="session",
        help="Historical and Fed scenarios shock one-month (VIX-like) implied vol; longer-dated IV moves less, "
        f"so the shock is scaled by (T / 1 month)^-{scenarios.DEFAULT_VOL_TENOR_DECAY}. Hypothetical IV shocks "
        "always apply literally to every leg, like the heatmap above.",
    )
    with c3.popover("Custom scenario", icon=":material/tune:"):
        custom_on = st.toggle("Include custom scenario", value=False, key="stress_custom", persist_state="session")
        spot = st.number_input("Spot change (%)", -99.0, 500.0, -15.0, 1.0, key="custom_spot", persist_state="session")
        vol = st.number_input("IV multiplier", 0.1, 10.0, 1.5, 0.1, key="custom_vol", persist_state="session")
        rate = st.number_input(
            "Rate change (bp)", -1000.0, 1000.0, 0.0, 25.0, key="custom_rate", persist_state="session"
        )
        days = st.number_input("Days elapsed", 0, 3650, 0, 1, key="custom_days", persist_state="session")

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
    charts.show(
        charts.bar_chart(table["Scenario"], table["P&L"], "P&L ($)"),
        key="stress_chart",
        alt="The position's P&L ($) in each stress scenario, as horizontal bars",
    )
    columns = ["Scenario", "P&L", "Spot change", "IV multiplier", "IV shock damped", "Rate change (bp)"]
    if elapsed or custom_on:
        columns += ["Days elapsed", "Legs expired"]
    # A mark, not a checkbox: a CheckboxColumn drew toggles that looked editable in this read-only table.
    table["IV shock damped"] = table["IV shock damped"].map({True: "yes", False: "—"})
    st.dataframe(
        # Short headers so that every one fits its narrow column; the help gives the full name.
        table[columns].rename(columns=STRESS_HEADERS),
        hide_index=True,
        height=35 * (len(table) + 1) + 3,  # show every scenario without an inner scrollbar
        column_config={
            "Scenario": st.column_config.TextColumn(width="medium"),
            "Spot change": st.column_config.NumberColumn(format="percent", width="small"),
            "IV multiplier": st.column_config.NumberColumn(
                format="%.2fx", width="small", help="Multiplier on implied vol."
            ),
            "IV damped": st.column_config.TextColumn(
                width="small",
                help="IV shock damped: yes for a one-month shock, damped for longer-dated legs (see the toggle "
                "above); — for a shock applied as it is to every leg.",
            ),
            "Rate Δ (bp)": st.column_config.NumberColumn(
                format="%+.0f", width="small", help="Rate change in basis points (100 bp = 1 percentage point)."
            ),
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


def _signed_short(value: float) -> str:
    """:func:`money_short` with an explicit sign, the minus sign U+2212: ``+$1,029`` or ``−$18.40``."""
    text = money_short(value, signed=True)
    return MINUS + text[1:] if text.startswith("-") else text


def _as_shown(value: float) -> float:
    """The amount :func:`money_short` shows for ``value``, as a number: 54.29505 reads ``$54.30``, so 54.3.

    Read back from the text, so that it follows the cards' precision by construction (cents below $1,000, whole
    dollars below a million, then ``$1.23M``)."""
    if not math.isfinite(value):
        return value
    text = money_short(abs(value)).removeprefix("$").replace(",", "")
    scale = {suffix: size for size, suffix in SHORT_SCALES}.get(text[-1])
    amount = float(text) if scale is None else float(text[:-1]) * scale
    return math.copysign(amount, value)


def _var_help(result: risk.VaRResult, meaning: str) -> str:
    """Help of a VaR card: what it means, then the exact amount and its 95% simulation interval."""
    lo, hi = result.var_ci95
    return f"{meaning} Exact: {money(result.var)}; 95% simulation interval {money(lo)} to {money(hi)}."


def _es_help(result: risk.VaRResult, vol: str) -> str:
    """Help of an Expected Shortfall card; ``vol`` says what implied vol does in the simulation."""
    return f"Average loss in the worst tail beyond VaR, {vol}. Exact: {money(result.expected_shortfall)}."


def _risk_card(label: str, value: float, help_text: str, frozen: float | None = None) -> Card:
    """A VaR or ES card: the amount shortened, ``help_text`` as its help, and a delta line with what the implied-vol
    moves add to ``frozen``, the same measure with implied vol frozen (``+$1,029 vs spot only``). Without
    ``frozen`` the card's own amount has implied vol frozen, and the line says so.

    The delta is the difference of the two amounts as the cards show them, so that it adds up on screen: on the
    default page the cards read $54.30 and $54.87 and the line −$0.57, where the exact amounts (54.295 and 54.873)
    would give −$0.58."""
    if frozen is None:
        delta = "implied vol frozen"
    else:
        delta = md(f"{_signed_short(_as_shown(value) - _as_shown(frozen))} vs spot only")
    return (
        label,
        money_short(value),
        {"help": md(help_text), "delta": delta, "delta_color": "off", "delta_arrow": "off"},
    )


def _var(position: Position) -> None:
    c1, c2, c3, c4 = st.columns(4)
    confidence = c1.selectbox(
        "Confidence",
        CONFIDENCE_LEVELS,
        index=2,
        format_func=lambda c: f"{c:.1%}",
        key="var_conf",
        persist_state="session",
        help="97.5% is the Basel FRTB Expected Shortfall level; 99% is the classic VaR level.",
    )
    horizon = c2.number_input("Horizon (trading days)", 1, 60, 1, 1, key="var_horizon", persist_state="session")
    n_paths = c3.select_slider(
        "Simulations",
        PATH_CHOICES,
        value=100_000,
        format_func=lambda n: f"{n:,}",
        key="var_paths",
        persist_state="session",
    )
    distribution = c4.selectbox(
        "Return model",
        list(risk.DISTRIBUTIONS),
        format_func=risk.DISTRIBUTIONS.get,
        key="var_dist",
        persist_state="session",
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
            persist_state="session",
            help="Lower = fatter tails. Daily equity returns are typically 3–6.",
        )
    elif distribution == "garch":
        c1, c2, c3, c4 = st.columns(4)
        alpha = c1.number_input(
            "GARCH α", 0.0, 0.5, 0.09, 0.01, format="%.3f", key="garch_alpha", persist_state="session"
        )
        beta = c2.number_input(
            "GARCH β", 0.0, 0.999, 0.90, 0.01, format="%.3f", key="garch_beta", persist_state="session"
        )
        garch_t = c3.toggle("Student-t innovations", value=False, key="garch_t", persist_state="session")
        if garch_t:
            dof = c4.slider(
                "Degrees of freedom ν",
                3,
                30,
                6,
                key="var_dof_garch",
                persist_state="session",
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
        value=True,
        key="var_volmoves",
        persist_state="session",
        help="Adds vega risk: log implied vol moves with the chosen vol of vol, correlated with spot. On by default "
        "with index (VIX-like) dynamics: vol of vol 100% a year and spot–vol correlation −0.7. The spot-only VaR "
        "and ES, with implied vol frozen, are shown beside the result. Off, implied vol stays frozen.",
    )
    vov, corr = 0.0, INDEX_SPOT_VOL_CORRELATION
    if vol_dynamics:
        vov = (
            c2.number_input(
                "Vol of vol (% p.a.)",
                0.0,
                500.0,
                100 * INDEX_VOL_OF_VOL,
                10.0,
                key="var_vov",
                persist_state="session",
                help="Volatility of one-month implied vol. The default, 100%, matches index (VIX-like) dynamics. "
                "Longer-dated legs move less, damped by tenor like the stress tests.",
            )
            / 100
        )
        corr = c3.slider(
            "Spot–vol correlation",
            -1.0,
            1.0,
            INDEX_SPOT_VOL_CORRELATION,
            0.05,
            key="var_corr",
            persist_state="session",
            help="The default, −0.7, matches index (VIX-like) dynamics: spot down, implied vol up.",
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
    # The spot-only run re-prices the same spot paths with implied vol frozen: the simulation draws every spot shock
    # before any vol shock, and vol shocks only for a positive vol of vol, so with one seed the two P&Ls differ path
    # by path by the vol moves alone. Without vol moves the correlation does nothing, so it is fixed: a change of the
    # vol dynamics then finds the spot-only run in the cache, and the toggle switched off finds it too.
    frozen = dataclasses.replace(config, vol_of_vol=0.0, spot_vol_correlation=INDEX_SPOT_VOL_CORRELATION)
    with_vol = config.vol_of_vol > 0
    with st.spinner("Simulating…"):
        spot_only = cache.value_at_risk(position.strategy, position.market, frozen, position.units)
        result = spot_only
        if with_vol:
            result = cache.value_at_risk(position.strategy, position.market, config, position.units)

    level = f"{confidence:.1%}"
    meaning = f"Loss not exceeded with {level} probability over {horizon} trading day(s)"
    moving, still = "with spot and implied vol moving", "with implied vol frozen"
    if with_vol:
        # Side by side, so that the vega risk shows: what the implied-vol moves add to VaR and ES.
        groups = ("var_cards_full", "var_cards_spot")
        cards = [
            _risk_card(f"VaR {level}", result.var, _var_help(result, f"{meaning}, {moving}."), spot_only.var),
            _risk_card(
                f"Expected Shortfall {level}",
                result.expected_shortfall,
                _es_help(result, moving),
                spot_only.expected_shortfall,
            ),
            _risk_card(
                f"Spot-only VaR {level}",
                spot_only.var,
                _var_help(spot_only, f"{meaning} on the same spot paths {still}, which leaves out the vega risk."),
            ),
            _risk_card(f"Spot-only ES {level}", spot_only.expected_shortfall, _es_help(spot_only, still)),
        ]
    else:
        groups = ("var_cards_all",)
        cards = [
            _risk_card(f"VaR {level}", result.var, _var_help(result, f"{meaning}, {still} (no vega risk).")),
            _risk_card(f"Expected Shortfall {level}", result.expected_shortfall, _es_help(result, still)),
        ]
    card_row("var_cards", groups, cards)
    interval = f"{money(result.var_ci95[0])} to {money(result.var_ci95[1])}"
    if with_vol:  # two VaRs on the cards: name the one each interval belongs to
        lo, hi = spot_only.var_ci95
        uncertainty = (
            f"the 95% confidence intervals of the VaR estimates are {interval} with implied-vol moves and "
            f"{money(lo)} to {money(hi)} spot only (order statistics of the {n_paths:,} simulated outcomes of each run)"
        )
    else:
        uncertainty = (
            f"the 95% confidence interval of the VaR estimate is {interval} (order statistics of the {n_paths:,} "
            "simulated outcomes)"
        )
    st.caption(md(f"Simulation uncertainty: {uncertainty}."))
    charts.show(
        charts.histogram_chart(
            result.pnl,
            f"P&L over {horizon} trading day(s) ($)",
            [(-result.var, f"VaR {confidence:.1%}", charts.LOSS), (-result.expected_shortfall, "ES", charts.ACCENT)],
            value_text=money,
        ),
        key="var_hist",
        alt=f"Histogram of the simulated P&L ($) over {horizon} trading day(s), with the VaR and the Expected "
        "Shortfall",
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
            {"Statistic": f"Delta-normal VaR {level}", "Value": money(result.delta_normal_var)},
        ]
    )
    vol_moves = (
        f"implied vol moving (vol of vol {100 * config.vol_of_vol:g}% a year, correlation {signed(corr)} with spot)"
        if with_vol
        else "implied vol frozen"
    )
    # What delta-normal VaR leaves out, so what its gap to the full-revaluation VaR shows. One-day GARCH returns
    # with Normal shocks are Normal (the variance starts at its long-run level), so their tails are no fatter.
    left_out = "gamma, time decay and vega" if with_vol else "gamma and time decay"
    if distribution == "student_t" or garch_t or (distribution == "garch" and horizon > 1):
        left_out += ", and the fatter tails of the return model"
    c1, c2 = st.columns([1, 1.4])
    c1.dataframe(stats, hide_index=True)
    c2.caption(
        f"Full revaluation of every leg on {n_paths:,} simulated {risk.DISTRIBUTIONS[distribution]} paths "
        f"(zero drift, volatility {position.market.volatility:.1%}), with {horizon} trading day(s) of time decay and "
        f"{vol_moves}. VaR and ES use the Acerbi–Tasche estimators on the sorted P&L. Delta-normal VaR is the "
        f"linear approximation |Δ|·S·σ·√(h/252)·z with Normal returns: its gap to the full-revaluation VaR shows "
        f"the position's {left_out}."
        + (" Excess kurtosis is unstable for ν ≤ 4." if distribution == "student_t" or garch_t else "")
        + (
            f" {result.settled_legs} option leg(s) expire before the horizon and are settled on the simulated price "
            "at the close of their expiry day."
            if result.settled_legs
            else ""
        )
    )
    c2.number_input("Random seed", 0, 2**32 - 1, 7, 1, key="var_seed", persist_state="session")


@st.fragment
def render(position: Position | None) -> None:
    if position is None:
        st.info("Build a valid position in the **Strategy** tab to analyse its risk.")
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
