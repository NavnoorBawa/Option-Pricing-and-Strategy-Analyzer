"""Strategy Builder tab: templates, editable legs, payoff chart and analytics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from .. import strategies as strat
from . import charts
from .format import md, money, pct
from .inputs import Inputs

CUSTOM = "Custom"
LEG_COLUMNS = ["Type", "Quantity", "Strike", "Days", "IV (%)", "Premium"]


@dataclass(frozen=True)
class Position:
    strategy: strat.Strategy
    market: strat.Market
    contracts: int
    multiplier: float

    @property
    def units(self) -> float:
        """Dollar scaling: contracts × shares per contract."""
        return self.contracts * self.multiplier

    def describe(self) -> str:
        legs = ", ".join(leg.label() for leg in self.strategy.legs)
        return f"**{self.strategy.name}** — {legs} · {self.contracts} × {self.multiplier:g} multiplier"


def _template_frame(strategy: strat.Strategy) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Type": leg.kind,
                "Quantity": float(leg.quantity),
                "Strike": float(leg.strike) if leg.is_option else None,
                "Days": None if leg.expiry is None else round(leg.expiry * 365),
                "IV (%)": None if leg.volatility is None else leg.volatility * 100,
                "Premium": leg.premium,
            }
            for leg in strategy.legs
        ],
        columns=LEG_COLUMNS,
    ).astype({"Quantity": float, "Strike": float, "Days": float, "IV (%)": float, "Premium": float})


def _legs_from_frame(frame: pd.DataFrame) -> tuple[list[strat.Leg], list[str]]:
    legs, errors = [], []
    for i, row in frame.reset_index(drop=True).iterrows():
        kind, qty = row.get("Type"), row.get("Quantity")
        if kind is None or (isinstance(kind, float) and np.isnan(kind)) or pd.isna(qty):
            continue  # a blank row being edited
        if qty == 0:
            continue
        strike = row.get("Strike")
        days = row.get("Days")
        iv = row.get("IV (%)")
        premium = row.get("Premium")
        try:
            legs.append(
                strat.Leg(
                    kind=str(kind),
                    quantity=float(qty),
                    strike=0.0 if kind == "stock" or pd.isna(strike) else float(strike),
                    expiry=None if kind == "stock" or pd.isna(days) else float(days) / 365.0,
                    volatility=None if kind == "stock" or pd.isna(iv) else float(iv) / 100.0,
                    premium=None if pd.isna(premium) else float(premium),
                )
            )
        except ValueError as exc:
            errors.append(f"Row {i + 1}: {exc}")
    return legs, errors


def _selector(inputs: Inputs):
    c1, c2, c3 = st.columns([1.1, 1.4, 1])
    categories = [*strat.TEMPLATE_CATEGORIES, CUSTOM]
    category = c1.selectbox("Category", categories, index=categories.index("Vertical spread"), key="strategy_category")
    if category == CUSTOM:
        name = CUSTOM
        c2.selectbox("Strategy", [CUSTOM], key="strategy_custom", disabled=True)
        template = None
    else:
        names = [t.name for t in strat.TEMPLATES.values() if t.category == category]
        name = c2.selectbox("Strategy", names, key=f"strategy_{category}")
        template = strat.TEMPLATES[name]
    default_width = max(round(inputs.strike * 0.10, 2), 0.01)
    width = c3.number_input(
        "Strike width ($)",
        min_value=0.01,
        value=default_width,
        step=max(round(inputs.strike * 0.01, 2), 0.01),
        format="%.2f",
        key="strategy_width",
        disabled=template is None or not template.uses_width,
        help="Distance w between adjacent strikes in multi-strike templates (K ± w, K ± 2w).",
    )
    return name, template, float(width)


def render(inputs: Inputs) -> Position | None:
    """Render the tab and return the position (``None`` if the legs are invalid)."""
    name, template, width = _selector(inputs)
    if template is not None:
        st.caption(f"**Outlook:** {template.outlook} · **Structure:** {template.structure} — {template.description}")

    c1, c2, c3, c4 = st.columns(4)
    contracts = int(c1.number_input("Contracts", min_value=1, max_value=100_000, value=1, step=1, key="contracts"))
    multiplier = float(
        c2.number_input(
            "Shares per contract",
            min_value=1,
            max_value=10_000,
            value=100,
            step=1,
            key="multiplier",
            help="100 for US equity options. P&L below is for the whole position.",
        )
    )
    commission = float(
        c3.number_input(
            "Commission per contract ($)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.05,
            format="%.2f",
            key="commission",
            help="Charged on every option contract when the position is opened.",
        )
    )
    with c4.popover("Assumptions", icon=":material/tune:", width="stretch"):
        risk_neutral = st.toggle(
            "Risk-neutral probabilities",
            value=True,
            key="risk_neutral",
            help="Probabilities and expected P&L use drift = risk-free rate. Turn off to enter your own view.",
        )
        drift = inputs.rate
        if not risk_neutral:
            drift = st.number_input("Expected annual return (%)", -100.0, 200.0, 8.0, 0.5, key="drift") / 100.0
        carry = st.toggle(
            "Finance the premium",
            value=False,
            key="carry",
            help="Charge interest on a net debit (or earn it on a credit) at the risk-free rate. With this on, "
            "the risk-neutral expected P&L is zero for legs entered at model value with the sidebar IV and no "
            "fees. Payoff diagrams conventionally leave it off.",
        )

    try:
        base = (
            strat.build_template(name, inputs.strike, width, inputs.expiry)
            if template
            else strat.Strategy(CUSTOM, (strat.Leg("call", 1, inputs.strike),))
        )
    except ValueError as exc:
        st.error(str(exc))
        return None

    version = st.session_state.setdefault("legs_version", 0)
    st.markdown("**Legs** — edit, add or delete rows to customise the position.")
    edited = st.data_editor(
        _template_frame(base),
        key=f"legs::{name}::{inputs.strike:g}::{width:g}::{version}",
        num_rows="dynamic",
        hide_index=True,
        placeholder="auto",
        column_config={
            "Type": st.column_config.SelectboxColumn(options=list(strat.LEG_KINDS), required=True),
            "Quantity": st.column_config.NumberColumn(
                help="Positive = long, negative = short. One unit = one option on one share, or one share.",
                format="%.2f",
                required=True,
            ),
            "Strike": st.column_config.NumberColumn(min_value=0.01, format="%.2f", help="Ignored for stock."),
            "Days": st.column_config.NumberColumn(
                min_value=1,
                max_value=3650,
                step=1,
                format="%d",
                help=f"Days to expiry; blank = sidebar value ({inputs.days}).",
            ),
            "IV (%)": st.column_config.NumberColumn(
                min_value=0.1,
                max_value=500.0,
                format="%.2f",
                help=f"Implied volatility; blank = sidebar value ({inputs.volatility * 100:.2f}%).",
            ),
            "Premium": st.column_config.NumberColumn(
                min_value=0.0,
                format="%.4f",
                help="Your fill price per share; blank = model value. P&L is measured against it.",
            ),
        },
    )
    if st.button("Reset legs to template", key="reset_legs", icon=":material/restart_alt:"):
        st.session_state["legs_version"] = version + 1
        st.rerun()

    legs, errors = _legs_from_frame(edited)
    for message in errors:
        st.error(message)
    if errors or not legs:
        if not legs and not errors:
            st.warning("Add at least one leg with a non-zero quantity.")
        return None

    modified = tuple(legs) != tuple(base.legs)
    label = name if not modified else (f"{name} (modified)" if name != CUSTOM else CUSTOM)
    option_units = sum(abs(leg.quantity) for leg in legs if leg.is_option)
    fees = commission * option_units / multiplier
    strategy = strat.Strategy(label, tuple(legs), template.description if template else "", fees)
    market = inputs.market
    position = Position(strategy, market, contracts, multiplier)
    _analysis(position, drift, carry)
    return position


def _analysis(position: Position, drift: float, carry: bool) -> None:
    strategy, market, units = position.strategy, position.market, position.units
    metrics = strat.analyze(strategy, market, drift, carry)
    front_days = metrics.horizon * 365

    cost = metrics.entry_cost * units
    c = st.columns(5)
    c[0].metric(
        "Net premium",
        money(abs(cost)),
        "debit (paid)" if cost > 0 else "credit (received)",
        delta_color="off",
        border=True,
    )
    if metrics.max_profit < 0:
        c[1].metric("Max profit", "None", border=True, help=md(f"Loses at least {money(-metrics.max_profit * units)}."))
    else:
        c[1].metric("Max profit", money(metrics.max_profit * units), border=True)
    if metrics.max_loss < 0:
        c[2].metric(
            "Max loss",
            "None",
            border=True,
            help=md(f"Guaranteed profit of at least {money(-metrics.max_loss * units)}."),
        )
    else:
        c[2].metric("Max loss", money(metrics.max_loss * units), border=True)
    be = metrics.breakevens
    c[3].metric(
        "Breakeven" + ("s" if len(be) > 1 else ""), " · ".join(f"{b:,.2f}" for b in be) if be else "none", border=True
    )
    c[4].metric(
        "Probability of profit",
        pct(metrics.probability_of_profit, 1),
        border=True,
        help="At the front expiry, log-normal terminal price with the chosen drift and the sidebar volatility.",
    )

    greeks = strat.position_greeks(strategy, market)
    rr = metrics.reward_to_risk
    c = st.columns(5)
    c[0].metric(
        "Reward / risk",
        "Unlimited" if rr is not None and np.isinf(rr) else ("–" if rr is None else f"{rr:.2f}"),
        border=True,
        help="Max profit ÷ max loss at the front expiry.",
    )
    c[1].metric(
        "Delta (shares)",
        f"{greeks['delta'] * units:,.2f}",
        border=True,
        help="Share-equivalent exposure: P&L change for a $1 move in the underlying.",
    )
    c[2].metric("Gamma", f"{greeks['gamma'] * units:,.4f}", border=True, help="Change in delta per $1 move.")
    c[3].metric(
        "Theta / day",
        money(greeks["theta"] * units / 365, signed=True),
        border=True,
        help="P&L from one calendar day passing, all else equal.",
    )
    c[4].metric(
        "Vega / vol pt",
        money(greeks["vega"] * units / 100, signed=True),
        border=True,
        help="P&L for a 1-point rise in every leg's implied volatility.",
    )

    st.markdown(f"**Profit & loss** — front expiry in {front_days:.0f} days")
    last_day = round(front_days) - 1
    c1, c2 = st.columns(2)
    date = 0
    if last_day >= 1:
        date = c1.slider(
            "Also show P&L after (days from today)",
            0,
            last_day,
            last_day // 2 + 1 if last_day > 1 else 1,
            key="pnl_date",
        )
    iv_shift = (
        c2.slider(
            "Implied-vol shift for the curves before expiry (vol pts)",
            -50,
            100,
            0,
            1,
            key="pnl_iv_shift",
            help="Adds this many volatility points to every leg's IV in the 'Today' and dated curves. The "
            "at-expiry curve, breakevens and the metrics above use the legs' own IVs.",
        )
        / 100
    )
    sigma_t = market.volatility * np.sqrt(metrics.horizon)
    anchors = [market.spot, *strategy.strikes]
    lo = max(min(anchors) * np.exp(-3.5 * sigma_t), 0.0)
    hi = max(anchors) * np.exp(3.5 * sigma_t)
    spots = np.unique(np.concatenate([np.linspace(lo, hi, 600), strategy.strikes]))
    shift_label = f", IV {iv_shift * 100:+.0f} pts" if iv_shift else ""
    curves = [
        (
            strat.pnl(strategy, market, spots, metrics.horizon, carry) * units,
            f"At expiry ({front_days:.0f}d)",
            charts.CALL,
            "solid",
        ),
        (
            strat.pnl(strategy, market, spots, 0.0, carry, vol_shift=iv_shift) * units,
            f"Today{shift_label}",
            charts.PUT,
            "dash",
        ),
    ]
    if 0 < date < front_days:
        curves.append(
            (
                strat.pnl(strategy, market, spots, date / 365, carry, vol_shift=iv_shift) * units,
                f"In {date} days{shift_label}",
                charts.ACCENT,
                "dot",
            )
        )
    forward = market.spot * np.exp((metrics.drift - market.dividend_yield) * metrics.horizon)
    band = (forward * np.exp(-sigma_t), forward * np.exp(sigma_t))
    visible_be = [b for b in be if lo <= b <= hi]
    st.plotly_chart(
        charts.payoff_chart(spots, curves, market.spot, strategy.strikes, visible_be, band), key="payoff_chart"
    )
    profile = pd.DataFrame({"Underlying price": spots} | {name: values for values, name, _, _ in curves})
    st.download_button(
        "Download P&L profile (CSV)",
        profile.to_csv(index=False).encode(),
        f"{strategy.name.lower().replace(' ', '_')}_pnl.csv",
        "text/csv",
        on_click="ignore",
        icon=":material/download:",
        key="dl_pnl",
    )

    c = st.columns(3)
    c[0].metric(
        "Expected P&L at expiry",
        money(metrics.expected_pnl * units, signed=True),
        help="Mean P&L under the chosen drift. With the risk-neutral drift and legs at model value it equals "
        "the interest the premium would have earned (zero when the premium is financed); fills better or "
        "worse than model value, leg IVs and fees show up as the difference.",
    )
    c[1].metric(
        "P&L standard deviation",
        money(metrics.pnl_std * units) if np.isfinite(metrics.pnl_std) else "Too large",
        help="Dispersion of the P&L at the front expiry under the chosen drift.",
    )
    c[2].metric("Drift used", pct(metrics.drift, 2), help="Annual expected return of the underlying.")

    st.markdown("**Legs at inception**")
    rows = strat.leg_table(strategy, market)
    table = pd.DataFrame(rows)
    for col in ("Cost", "Delta", "Gamma", "Theta / day", "Vega / vol pt"):
        table[col] = table[col] * units
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "Strike": st.column_config.NumberColumn(format="%.2f"),
            "IV": st.column_config.NumberColumn(format="percent"),
            "Model value": st.column_config.NumberColumn(format="$%.4f", help="Black-Scholes-Merton value per share."),
            "Entry price": st.column_config.NumberColumn(format="$%.4f", help="Premium paid/received per share."),
            "Cost": st.column_config.NumberColumn(format="dollar", help="Whole position; negative = credit."),
            "Delta": st.column_config.NumberColumn(format="%.2f"),
            "Gamma": st.column_config.NumberColumn(format="%.4f"),
            "Theta / day": st.column_config.NumberColumn(format="dollar"),
            "Vega / vol pt": st.column_config.NumberColumn(format="dollar"),
        },
    )
    if strategy.fees:
        st.caption(
            md(f"Commissions of {money(strategy.fees * units)} are included in the net premium and every P&L figure.")
        )
    if strat.unbounded_loss_from_dividends(strategy, market):
        st.info(
            md(
                "With a dividend yield, a European long-dated call gains less than $1 for each $1 rise in the "
                "stock, so far above the strikes the short front-month call dominates and the loss is unbounded. "
                "An American call would be exercised early to avoid this."
            ),
            icon=":material/info:",
        )
    st.caption(
        "Options are valued with Black-Scholes-Merton (European exercise) at each leg's implied volatility; "
        "stock is valued at its price plus the dividends it pays. "
        + ("The premium is financed at the risk-free rate. " if carry else "Premium financing is ignored. ")
        + "Probabilities assume a log-normal price with the sidebar volatility; the shaded band is the ±1σ range "
        "of the price at the front expiry."
    )
