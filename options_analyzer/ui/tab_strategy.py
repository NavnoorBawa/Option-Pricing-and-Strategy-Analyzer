"""Strategy Builder tab: templates, editable legs, payoff chart and analytics."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from .. import black_scholes as bs
from .. import strategies as strat
from . import cache, charts
from .format import md, money
from .inputs import Inputs

CUSTOM = "Custom"
LEG_COLUMNS = ["Type", "Quantity", "Strike", "Days", "IV (%)", "Premium"]
NUMERIC_LEG_COLUMNS = LEG_COLUMNS[1:]

#: Strike width bounds in % of the strike. Below 50 % every template keeps its lowest strike (K - 2w) positive.
MIN_WIDTH_PCT = 0.1
MAX_WIDTH_PCT = 45.0
#: The automatic width never exceeds this, so very volatile or long-dated templates stay readable.
MAX_AUTO_WIDTH_PCT = 25.0

#: Strikes with as many decimals as they have (up to 8 significant digits): 100, 102.5, 0.975. Static, like the
#: rest of the editor's configuration.
STRIKE_FORMAT = "%.8g"

#: The column help is static on purpose: the editor's identity hashes its column configuration, so text that
#: changed with the sidebar would recreate the editor and throw away the user's edits.
LEG_COLUMN_CONFIG = {
    "Type": st.column_config.SelectboxColumn(options=list(strat.LEG_KINDS), required=True),
    "Quantity": st.column_config.NumberColumn(
        help="Positive = long, negative = short. One unit = one option on one share, or one share.",
        format="%.2f",
        required=True,
    ),
    "Strike": st.column_config.NumberColumn(
        min_value=0.0001, format=STRIKE_FORMAT, help="Blank = the sidebar strike. Ignored for stock."
    ),
    "Days": st.column_config.NumberColumn(
        min_value=1, max_value=3650, step=1, format="%d", help="Days to expiry. Blank = the sidebar days to expiry."
    ),
    "IV (%)": st.column_config.NumberColumn(
        min_value=0.1, max_value=500.0, format="%.2f", help="Implied volatility. Blank = the sidebar volatility."
    ),
    "Premium": st.column_config.NumberColumn(
        min_value=0.0,
        format="%.4f",
        help="Your fill price per share. Blank = model value. P&L is measured against it.",
    ),
}


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
        legs = ", ".join(strat.leg_labels(self.strategy, self.market))
        return md(f"**{self.strategy.name}** — {legs} · {self.contracts} × {self.multiplier:g} multiplier")


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
    ).astype(dict.fromkeys(NUMERIC_LEG_COLUMNS, float))


def _legs_from_frame(frame: pd.DataFrame, default_strike: float) -> tuple[list[strat.Leg], list[str]]:
    """Legs from the editor rows. A blank strike, expiry or IV means the sidebar value."""
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
                    strike=0.0 if kind == "stock" else (default_strike if pd.isna(strike) else float(strike)),
                    expiry=None if kind == "stock" or pd.isna(days) else float(days) / 365.0,
                    volatility=None if kind == "stock" or pd.isna(iv) else float(iv) / 100.0,
                    premium=None if pd.isna(premium) else float(premium),
                )
            )
        except ValueError as exc:
            errors.append(f"Row {i + 1}: {exc}")
    return legs, errors


def _round_step(x: float) -> float:
    """The 1, 2, 2.5 or 5 x 10^n step closest to ``x`` in relative terms (the spacing of listed strikes)."""
    power = 10.0 ** np.floor(np.log10(x))
    steps = [m * power * scale for scale in (1.0, 10.0) for m in (1.0, 2.0, 2.5, 5.0)]
    return min(steps, key=lambda s: abs(np.log(s / x)))


def round_width(width: float) -> float:
    """A strike width on a price tick: whole cents, or two significant figures below ten cents."""
    decimals = max(2, 1 - int(np.floor(np.log10(width))))
    return round(float(width), decimals)


def auto_width(strike: float, volatility: float, expiry: float) -> float:
    """Default strike spacing ($): about half the one-standard-deviation move σ√T, on a round strike step."""
    step = _round_step(0.5 * volatility * np.sqrt(expiry) * strike)
    return round_width(float(np.clip(step, strike * MIN_WIDTH_PCT / 100, strike * MAX_AUTO_WIDTH_PCT / 100)))


def _selector(inputs: Inputs):
    c1, c2, c3 = st.columns([1.1, 1.4, 1])
    categories = [*strat.TEMPLATE_CATEGORIES, CUSTOM]
    category = c1.selectbox(
        "Category",
        categories,
        index=categories.index("Vertical spread"),
        key="strategy_category",
        bind="query-params",
    )
    if category == CUSTOM:
        name = CUSTOM
        c2.selectbox("Strategy", [CUSTOM], key="strategy_custom", disabled=True)
        template = None
    else:
        names = [t.name for t in strat.TEMPLATES.values() if t.category == category]
        name = c2.selectbox("Strategy", names, key=f"strategy_{category}", bind="query-params", persist_state="session")
        template = strat.TEMPLATES[name]
    uses_width = template is not None and template.uses_width
    automatic = auto_width(inputs.strike, inputs.volatility, inputs.expiry)
    width_pct = c3.number_input(
        "Strike width (% of strike)",
        min_value=MIN_WIDTH_PCT,
        max_value=MAX_WIDTH_PCT,
        value=None,
        step=0.5,
        format="%.2f",
        placeholder=f"Auto: {100 * automatic / inputs.strike:.2f}%",
        key="strategy_width_pct",
        bind="query-params",
        disabled=not uses_width,
        help="Distance w between adjacent strikes (K ± w, K ± 2w) as a percentage of the sidebar strike, so it "
        "moves with the strike; it is rounded to the cent (two significant figures below ten cents). Leave blank "
        "for the automatic width: about half the expected move σ√T to expiry, rounded to a strike step of 1, 2, "
        "2.5 or 5 times a power of ten (at most 25% of the strike).",
    )
    width = automatic if width_pct is None else round_width(inputs.strike * float(width_pct) / 100.0)
    width_text = ""
    if uses_width:
        source = "automatic" if width_pct is None else "set by you"
        width_text = md(
            f" · **Width** w = {money(width, price_decimals(width))} ({100 * width / inputs.strike:.2f}% of the "
            f"strike, {source})"
        )
    return name, template, width, width_text


def _canonical(frame: pd.DataFrame) -> pd.DataFrame:
    """The legs frame in one normal form, so that equal legs compare equal whatever dtypes the editor returned."""
    out = frame.reindex(columns=LEG_COLUMNS).reset_index(drop=True)
    out["Type"] = out["Type"].astype(object).where(out["Type"].notna(), None)
    return out.astype(dict.fromkeys(NUMERIC_LEG_COLUMNS, float))


def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return _canonical(a).equals(_canonical(b))


def _editor_key(name: str, state: dict) -> str:
    return f"legs::{name}::{state['version']}"


def _new_legs(base: pd.DataFrame, version: int) -> dict:
    return {"base": base, "seed": base, "edited": base, "template": base, "edits": None, "version": version}


def _legs_state(name: str, base: pd.DataFrame) -> dict:
    """The legs of strategy ``name`` across reruns, kept in the session so that edits are never silently lost.

    ``seed`` is the frame the editor starts from and ``base`` the template it was built from; the editor holds
    the user's edits on top of the seed. ``edited``, ``template`` and ``edits`` are the legs, the template and the
    editor's raw edits at the end of the last run. Legs that matched the template on screen follow it when its
    inputs change (sidebar strike or expiry, strike width); legs that differed from it are kept until they match
    the template again or are reset, so a later rerun from an unrelated widget never moves them.
    """
    store = st.session_state.setdefault("strategy_legs", {})
    state = store.get(name)
    if state is None:
        state = store[name] = _new_legs(base, 0)
        return state
    key = _editor_key(name, state)
    # An edit sent with this rerun (together with a sidebar change) is not in ``edited`` yet: keep the legs for it.
    editing = key in st.session_state and st.session_state[key] != state["edits"]
    if not editing and _same(state["edited"], state["template"]) and not _same(state["template"], base):
        # A new editor key, so that nothing typed on the old legs can apply to the new ones.
        state = store[name] = _new_legs(base, state["version"] + 1)
    elif key not in st.session_state and not _same(state["seed"], state["edited"]):
        # The editor was not on screen in the last run (another strategy was selected), so it no longer holds
        # the edits: show the edited legs as its new starting point.
        state["seed"] = state["edited"]
    return state


def render(inputs: Inputs) -> Position | None:
    """Render the tab and return the position (``None`` if the legs are invalid)."""
    name, template, width, width_text = _selector(inputs)
    if template is not None:
        st.caption(
            md(f"**Outlook:** {template.outlook} · **Structure:** {template.structure} — {template.description}")
            + width_text
        )

    c1, c2, c3, c4 = st.columns(4)
    contracts = int(
        c1.number_input(
            "Contracts", min_value=1, max_value=100_000, value=1, step=1, key="contracts", bind="query-params"
        )
    )
    multiplier = float(
        c2.number_input(
            "Shares per contract",
            min_value=1,
            max_value=10_000,
            value=100,
            step=1,
            key="multiplier",
            bind="query-params",
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
            bind="query-params",
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

    base_frame = _template_frame(base)
    state = _legs_state(name, base_frame)
    st.markdown("**Legs** — edit, add or delete rows to customise the position.")
    key = _editor_key(name, state)
    edited = st.data_editor(
        state["seed"],
        key=key,
        num_rows="dynamic",
        hide_index=True,
        placeholder="auto",
        column_config=LEG_COLUMN_CONFIG,
    )
    state.update(edited=edited, template=base_frame, edits=copy.deepcopy(st.session_state.get(key)))
    if template is not None and not _same(state["base"], base_frame) and not _same(edited, base_frame):
        st.info(
            "These legs differ from the template because you edited them, so they were kept when the template's "
            "inputs changed: the sidebar strike or days to expiry, or the strike width (the automatic width "
            "follows the volatility and the expiry). **Reset legs to template** rebuilds them from the current "
            "inputs.",
            icon=":material/edit_note:",
        )
    if st.button("Reset legs to template", key="reset_legs", icon=":material/restart_alt:"):
        st.session_state["strategy_legs"][name] = _new_legs(base_frame, state["version"] + 1)
        st.rerun()

    legs, errors = _legs_from_frame(edited, inputs.strike)
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
    _valuation_note(inputs, strategy)
    _analysis(position, drift, carry)
    return position


def _is_sidebar_contract(leg: strat.Leg, inputs: Inputs) -> bool:
    return (
        leg.is_option
        and np.isclose(leg.strike, inputs.strike, rtol=1e-12, atol=0)
        and (leg.expiry is None or np.isclose(leg.expiry, inputs.expiry, rtol=1e-12, atol=0))
        and (leg.volatility is None or np.isclose(leg.volatility, inputs.volatility, rtol=1e-12, atol=0))
    )


def _valuation_note(inputs: Inputs, strategy: strat.Strategy) -> None:
    """State that legs are European BSM values and, under an American tree, compare with the sidebar contract."""
    model = inputs.model
    european = (
        "Legs are valued as European options with Black-Scholes-Merton at each leg's implied volatility, "
        "whatever the sidebar pricing model."
    )
    american = model.kind == "binomial" and model.american
    kinds = sorted({leg.kind for leg in strategy.legs if _is_sidebar_contract(leg, inputs)})
    if american and not kinds:
        european += " The sidebar's American tree applies to the price cards only: early exercise is not modelled here."
    if not (american and kinds):
        st.caption(european)
        return
    S, K, T, r, sigma, q = (
        inputs.spot,
        inputs.strike,
        inputs.expiry,
        inputs.rate,
        inputs.volatility,
        inputs.dividend_yield,
    )
    try:
        tree = cache.tree_results(S, K, T, r, sigma, q, model.steps, True, model.tree_method)
    except ValueError:
        st.caption(european)
        return
    comparisons = " and ".join(
        f"one {K:g} {kind} (the sidebar contract) is worth {money(tree[kind].price, 4)} with early exercise "
        f"against {money(bs.price(S, K, T, r, sigma, q, kind), 4)} European"
        for kind in kinds
    )
    st.info(
        md(
            f"**Early exercise is not modelled here.** The price cards use the sidebar's American tree "
            f"({model.label}), but strategy legs are valued as European options with Black-Scholes-Merton. "
            f"Per share, {comparisons}."
        ),
        icon=":material/info:",
    )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def money_short(value: float, signed: bool = False) -> str:
    """Dollar amount short enough for a metric card: no cents from $100,000 and millions as ``$1.23M``."""
    if value is None or not np.isfinite(value) or abs(value) < 1e5:
        return money(value, 2, signed)
    if abs(value) < 1e6:
        return money(value, 0, signed)
    sign = "-" if value < 0 else ("+" if signed else "")
    return f"{sign}${abs(value) / 1e6:,.2f}M"


def _money_help(value: float, text: str = "", signed: bool = False) -> str | None:
    """Metric help with the exact amount whenever the card shows a shortened one."""
    exact = money(value, 2, signed)
    if money_short(value, signed) != exact:
        text = f"{text} Exact: {exact}.".strip()
    return md(text) if text else None


def price_decimals(value: float, most: int = 8) -> int:
    """Decimals needed to show ``value`` exactly: at least two (cents), at most ``most``."""
    for decimals in range(2, most):
        if abs(round(value, decimals) - value) <= 1e-9 * max(1.0, abs(value)):
            return decimals
    return most


def number(value: float, decimals: int) -> str:
    """Fixed-point number without a negative zero (``-0.00`` reads as a short position)."""
    if abs(value) < 0.5 * 10**-decimals:
        value = 0.0
    return f"{value:,.{decimals}f}"


def probability_text(p: float, sure: bool = False, never: bool = False) -> str:
    """Probability of profit with one decimal, never rounded to a certainty the payoff does not have.

    ``sure`` says the position cannot lose and ``never`` that it cannot profit. Otherwise a probability that
    rounds (or saturates in floating point) to 100 % or 0 % is shown as ``>99.9%`` or ``<0.1%``.
    """
    if p >= 0.9995 and not sure:
        return ">99.9%"
    if p <= 0.0005 and not never:
        return "<0.1%"
    return f"{p:.1%}"


def reward_risk_text(metrics: strat.StrategyMetrics, units: float) -> str:
    rr = metrics.reward_to_risk
    if rr is not None and np.isinf(rr):
        return "Unlimited"
    if rr is None or metrics.max_loss * units < 0.005:
        return "–"  # no loss, no profit, or a maximum loss that shows as $0.00
    if rr > 1000:
        return "> 1,000 : 1"
    return f"{rr:,.2f} : 1"


#: Decimals the legs table shows per column (its column formats), used to write values that round to zero as 0.
LEGS_TABLE_DECIMALS = {
    "Model value": 4,
    "Entry price": 4,
    "Cost": 2,
    "Delta": 2,
    "Gamma": 4,
    "Theta/day": 2,
    "Vega/pt": 2,
}


def legs_table(strategy: strat.Strategy, market: strat.Market, units: float) -> pd.DataFrame:
    """Per-leg values and Greeks of the whole position, without negative zeros (``-$0.00`` reads as a credit)."""
    table = pd.DataFrame(strat.leg_table(strategy, market)).drop(columns="Leg")
    for col in ("Cost", "Delta", "Gamma", "Theta / day", "Vega / vol pt"):
        table[col] = table[col] * units
    table = table.rename(columns={"Days to expiry": "Days", "Theta / day": "Theta/day", "Vega / vol pt": "Vega/pt"})
    for col, decimals in LEGS_TABLE_DECIMALS.items():
        table[col] = table[col].mask(table[col].abs() < 0.5 * 10**-decimals, 0.0)
    return table


def _join(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def breakeven_text(breakevens: list[float], spot: float) -> str:
    if not breakevens:
        return "none (the P&L at expiry never changes sign)"
    decimals = 2 if spot >= 1 else 4
    return _join([f"${b:,.{decimals}f} ({b / spot - 1:+.1%})" for b in breakevens])


def payoff_range(spot: float, strikes, forward: float, sigma_t: float) -> tuple[float, float]:
    """x-range of the payoff chart: the likely prices at expiry plus a margin around the strikes.

    The probability range is the forward ± 2.5 standard deviations (in log terms); at very high volatility
    its upper end is capped at three times the highest strike or spot so the strikes stay readable.
    """
    anchors = [spot, *strikes]
    low, high = min(anchors), max(anchors)
    margin = 0.25 * (high - low)
    lo = min(forward * np.exp(-2.5 * sigma_t), low - margin)
    hi = max(forward * np.exp(2.5 * sigma_t), high + margin)
    return max(lo, 0.0), min(hi, 3.0 * high)


def _analysis(position: Position, drift: float, carry: bool) -> None:
    strategy, market, units = position.strategy, position.market, position.units
    metrics = strat.analyze(strategy, market, drift, carry)
    front_days = metrics.horizon * 365

    cost = metrics.entry_cost * units
    c = st.columns(4)
    c[0].metric(
        "Net premium",
        money_short(abs(cost)),
        "debit (paid)" if cost > 0 else "credit (received)",
        delta_color="off",
        delta_arrow="off",
        border=True,
        help=_money_help(abs(cost)),
    )
    profit, loss = metrics.max_profit * units, metrics.max_loss * units
    if metrics.max_profit < 0:
        c[1].metric(
            "Max profit",
            "No profit",
            border=True,
            help=md(f"No profit is possible: the position loses at least {money(-profit)} at the front expiry."),
        )
    else:
        c[1].metric("Max profit", money_short(profit), border=True, help=_money_help(profit))
    if metrics.max_loss < 0:
        c[2].metric(
            "Max loss",
            "Cannot lose",
            border=True,
            help=md(f"The position cannot lose: it makes at least {money(-loss)} at the front expiry."),
        )
    else:
        c[2].metric("Max loss", money_short(loss), border=True, help=_money_help(loss))
    c[3].metric(
        "Probability of profit",
        probability_text(metrics.probability_of_profit, sure=metrics.max_loss < 0, never=metrics.max_profit <= 0),
        border=True,
        help="At the front expiry, log-normal terminal price with the chosen drift and the sidebar volatility. "
        "Shown as >99.9% or <0.1% when it rounds to 100% or 0% but the position can still lose or profit.",
    )
    be = metrics.breakevens
    st.markdown(
        md(f"**Breakeven{'s' if len(be) > 1 else ''} at expiry:** {breakeven_text(be, market.spot)}")
        + f" · **Reward / risk:** {reward_risk_text(metrics, units)}",
        help="Breakevens are the prices where the P&L at the front expiry changes sign (% change from today's "
        "spot). Reward / risk is max profit ÷ max loss; it is not shown when the maximum loss rounds to $0.00.",
    )

    greeks = strat.position_greeks(strategy, market)
    c = st.columns(4)
    c[0].metric(
        "Delta (shares)",
        number(greeks["delta"] * units, 2),
        border=True,
        help="Share-equivalent exposure: P&L change for a $1 move in the underlying.",
    )
    c[1].metric("Gamma", number(greeks["gamma"] * units, 4), border=True, help="Change in delta per $1 move.")
    theta, vega = greeks["theta"] * units / 365, greeks["vega"] * units / 100
    c[2].metric(
        "Theta / day",
        money_short(theta, signed=True),
        border=True,
        help=_money_help(theta, "P&L from one calendar day passing, all else equal.", signed=True),
    )
    c[3].metric(
        "Vega / vol pt",
        money_short(vega, signed=True),
        border=True,
        help=_money_help(vega, "P&L for a 1-point rise in every leg's implied volatility.", signed=True),
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
    forward = market.spot * np.exp((metrics.drift - market.dividend_yield) * metrics.horizon)
    lo, hi = payoff_range(market.spot, strategy.strikes, forward, sigma_t)
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
    expected = metrics.expected_pnl * units
    c[0].metric(
        "Expected P&L at expiry",
        money_short(expected, signed=True),
        help=_money_help(
            expected,
            "Mean P&L under the chosen drift. With the risk-neutral drift and legs at model value it equals "
            "the interest the premium would have earned (zero when the premium is financed); fills better or "
            "worse than model value, leg IVs and fees show up as the difference.",
            signed=True,
        ),
    )
    std = metrics.pnl_std * units
    c[1].metric(
        "P&L standard deviation",
        money_short(std) if np.isfinite(std) else "Too large",
        help=_money_help(std, "Dispersion of the P&L at the front expiry under the chosen drift.")
        if np.isfinite(std)
        else "Dispersion of the P&L at the front expiry under the chosen drift.",
    )
    c[2].metric("Drift used", f"{metrics.drift:.2%}", help="Annual expected return of the underlying.")

    st.markdown("**Legs at inception**")
    st.dataframe(
        legs_table(strategy, market, units),
        hide_index=True,
        column_config={
            "Strike": st.column_config.NumberColumn(format=STRIKE_FORMAT),
            "Days": st.column_config.NumberColumn(format="%d", help="Days to expiry."),
            "IV": st.column_config.NumberColumn(format="percent"),
            "Model value": st.column_config.NumberColumn(
                format="$%.4f", help="Black-Scholes-Merton (European) value per share."
            ),
            "Entry price": st.column_config.NumberColumn(format="$%.4f", help="Premium paid/received per share."),
            "Cost": st.column_config.NumberColumn(format="dollar", help="Whole position; negative = credit."),
            "Delta": st.column_config.NumberColumn(format="%.2f", help="Whole position, in shares."),
            "Gamma": st.column_config.NumberColumn(format="%.4f", help="Whole position, per $1 move."),
            "Theta/day": st.column_config.NumberColumn(format="dollar", help="Whole position, per calendar day."),
            "Vega/pt": st.column_config.NumberColumn(format="dollar", help="Whole position, per vol point."),
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
