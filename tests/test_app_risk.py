"""The Risk tab in the desk layout (UI spec 3.3, BK-1, UI-10): the position line, the six pinned tiles, the report
selector and its six reports, on the synthetic index at the pinned valuation time and on the textbook market.

Numbers are checked against Black-Scholes-Merton written here from the textbook formulas (scipy's normal
distribution), at the market the sidebar publishes (the default expiry's parity forward and discount factor, the
surface vol at the strike): not against the package's pricer. The spec's example numbers are those of an iron
condor that the Strategy tab of this branch does not build (its default is the v2 Bull Call Spread, here 5,815 /
5,915 on 20-Nov-26 AM), so the goldens below are this branch's, each recomputed independently.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date, datetime

import numpy as np
import pytest
from _apptest import ASOF, make_app
from scipy.stats import norm
from streamlit.testing.v1 import AppTest

from options_analyzer import scenarios
from options_analyzer.ui import market as mk
from options_analyzer.ui import tab_risk
from options_analyzer.ui.tab_risk import GridAxes

#: The default Strategy position under the index preset at ASOF: 1 × 100 of a 5,815 / 5,915 call spread.
LEGS = ((1.0, 5815.0), (-1.0, 5915.0))
UNITS = 100.0


def _ok(at: AppTest) -> AppTest:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    return at


def _risk(**query) -> AppTest:
    """The Risk tab on the synthetic index at ASOF (or on the textbook market with ``preset="Textbook"``)."""
    query.setdefault("preset", None)
    if query["preset"] is None:
        query.setdefault("asof", ASOF)
    at = make_app(tab="Risk", **query)
    at.run(timeout=240)
    return _ok(at)


def _tiles(at: AppTest) -> dict:
    labels = ("$Δ", "Γ P&L 1%", "Vega ($/vp)", "Θ ($/day)", "Worst grid", "Worst stress")
    return {m.label: m for m in at.metric if m.label in labels}


def _chart(at: AppTest, key: str) -> dict:
    return next(json.loads(c.proto.spec) for c in at.get("plotly_chart") if c.proto.id.endswith(key))


def _frame(at: AppTest, column: str):
    return next(d.value for d in at.dataframe if column in d.value.columns)


# --- an independent Black-Scholes-Merton ------------------------------------------------------------------------------


class Index:
    """The sidebar's market at ASOF, read from the published context: spot, the 20-Nov-26 expiry's years to
    settlement, the carry of its parity forward and discount factor, and the surface vol at 5,815."""

    def __init__(self) -> None:
        context = mk.index_market(datetime.fromisoformat(ASOF))
        row = context.expiry_row(date(2026, 11, 20))
        self.context = context
        self.s = context.spot
        self.t = float(row["maturity"])
        self.r, self.q = mk.carry(float(row["forward"]), float(row["discount_factor"]), self.s, self.t)
        self.vol = float(context.surface.implied_vol_at(5815.0 / self.s, self.t, extrapolate=True))


@pytest.fixture(scope="module")
def index() -> Index:
    return Index()


def _d1(s, k, t, r, q, v):
    return (math.log(s / k) + (r - q + 0.5 * v * v) * t) / (v * math.sqrt(t))


def _call(s, k, t, r, q, v):
    if t <= 0:
        return max(s - k, 0.0)
    d1 = _d1(s, k, t, r, q, v)
    return s * math.exp(-q * t) * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d1 - v * math.sqrt(t))


def _greeks(s, k, t, r, q, v):
    """Delta, gamma, vega (per 1.00) and theta (∂V/∂t per year) of a call, from the closed forms."""
    d1 = _d1(s, k, t, r, q, v)
    d2 = d1 - v * math.sqrt(t)
    eq, er = math.exp(-q * t), math.exp(-r * t)
    delta = eq * norm.cdf(d1)
    gamma = eq * norm.pdf(d1) / (s * v * math.sqrt(t))
    vega = s * eq * norm.pdf(d1) * math.sqrt(t)
    theta = -s * eq * norm.pdf(d1) * v / (2 * math.sqrt(t)) - r * k * er * norm.cdf(d2) + q * s * eq * norm.cdf(d1)
    return delta, gamma, vega, theta


def _book(s, t, r, q, v, legs=LEGS):
    return UNITS * sum(n * _call(s, k, t, r, q, v) for n, k in legs)


def _book_greeks(s, t, r, q, v, legs=LEGS):
    totals = np.zeros(4)
    for n, k in legs:
        totals += n * np.array(_greeks(s, k, t, r, q, v))
    return UNITS * totals


# --- the report selector and what each report computes ----------------------------------------------------------------


def test_the_selector_has_six_reports_and_ladder_is_the_default():
    at = _risk()
    pills = at.get("button_group")
    report = next(p for p in pills if p.proto.id.endswith("-report") or "report" in p.proto.id)
    assert [o.content for o in report.proto.options] == list(tab_risk.REPORTS)
    assert _frame(at, "Move") is not None  # the Ladder's table


def test_only_the_selected_report_computes(monkeypatch):
    # ?report=VaR runs the two simulations (with implied vol moving and frozen, Milestone 1's spot-only VaR beside
    # the spot-plus-vol one) and no other report's engine: the ladder never, the scenario cube once (the WORST GRID
    # tile's grid), no backtest.
    from options_analyzer import backtest, risk
    from options_analyzer.ui import cache

    calls = {"greek_ladder": 0, "scenario_cube": 0, "value_at_risk": 0, "rolling_backtest": 0}

    def counted(module, name):
        real = getattr(module, name)

        def wrapper(*args, **kwargs):
            calls[name] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(module, name, wrapper)

    for module, name in ((scenarios, "greek_ladder"), (scenarios, "scenario_cube"), (risk, "value_at_risk")):
        counted(module, name)
    counted(backtest, "rolling_backtest")
    cache.value_at_risk.clear()
    _risk(report="VaR")
    assert calls == {"greek_ladder": 0, "scenario_cube": 1, "value_at_risk": 2, "rolling_backtest": 0}


# --- the tiles --------------------------------------------------------------------------------------------------------


def test_the_default_tiles_are_the_closed_form_values(index):
    # Measured goldens of this branch's default position, each recomputed here: $Δ = Δ·S, Γ P&L 1% = ½Γ(0.01S)²,
    # vega per vol point, Θ = ∂V/∂t ÷ 365. Tolerance: the tile's own rounding (whole dollars).
    at = _risk()
    tiles = _tiles(at)
    delta, gamma, vega, theta = _book_greeks(index.s, index.t, index.r, index.q, index.vol)
    expected = {
        "$Δ": delta * index.s,
        "Γ P&L 1%": 0.5 * gamma * (0.01 * index.s) ** 2,
        "Vega ($/vp)": vega / 100,
        "Θ ($/day)": theta / 365,
    }
    for label, value in expected.items():
        assert tiles[label].value == tab_risk.dollars(value), label
    assert [tiles[k].value for k in ("$Δ", "Γ P&L 1%", "Vega ($/vp)", "Θ ($/day)")] == [
        "+$90,506",
        "+$19",
        "+$45",
        "−$20",
    ]
    # No statistic of a simulation or a history in a pinned tile (spec section 5 item 19).
    assert not any("VaR" in m.label or "ES " in m.label for m in tiles.values())


def test_the_delta_tile_counts_es_contracts_at_the_december_forward(index):
    at = _risk()
    delta = _book_greeks(index.s, index.t, index.r, index.q, index.vol)[0]
    forward = float(index.context.expiry_row(date(2026, 12, 18))["forward"])
    contracts = delta / (50.0 * forward / index.s)
    assert _tiles(at)["$Δ"].delta == f"{contracts:+.2f} ES" == "+0.31 ES"
    assert tab_risk.es_contract(datetime(2026, 10, 23, 16)) == date(2026, 12, 18)


def test_the_theta_tile_gives_the_breakeven_move_against_the_implied_move(index):
    # BE = 1% × √(|Θ| ÷ |Γ P&L 1%|), defined since Θ < 0 < Γ; implied = σ ATMF ÷ √365 at the front expiry, both per
    # calendar day. The ATMF vol is the surface's at K = F.
    at = _risk()
    _, gamma, _, theta = _book_greeks(index.s, index.t, index.r, index.q, index.vol)
    be = 0.01 * math.sqrt(abs(theta / 365) / abs(0.5 * gamma * (0.01 * index.s) ** 2))
    forward = index.s * math.exp((index.r - index.q) * index.t)
    atm = float(index.context.surface.implied_vol_at(forward / index.s, index.t, extrapolate=True))
    assert _tiles(at)["Θ ($/day)"].delta == f"BE {be:.2%} vs {atm / math.sqrt(365):.2%}" == "BE 1.02% vs 0.81%"
    # Θ to the next session (Mon 26-Oct 16:00, three calendar days) by full revaluation, in the help.
    decay = _book(index.s, index.t - 3 / 365, index.r, index.q, index.vol) - _book(
        index.s, index.t, index.r, index.q, index.vol
    )
    assert f"{abs(decay):,.2f}" in _tiles(at)["Θ ($/day)"].help


def test_the_worst_grid_and_worst_stress_tiles(index):
    at = _risk()
    tiles = _tiles(at)
    today = _book(index.s, index.t, index.r, index.q, index.vol)
    grid = {
        (x, dv): _book(index.s * (1 + x / 100), index.t, index.r, index.q, index.vol + dv / 100) - today
        for x in tab_risk.DESK_SPOT
        for dv in tab_risk.DESK_VOL["vp"]
    }
    (x, dv), worst = min(grid.items(), key=lambda item: item[1])
    assert tiles["Worst grid"].value == tab_risk.dollars(worst) == "−$4,096"
    assert tiles["Worst grid"].delta == f"{tab_risk.shock_label(x / 100)} · {tab_risk.vol_label(dv, 'vp')}"
    stresses = {}
    for sc in tab_risk.ALL_STRESSES:
        damp = min(1.0, (index.t / (30 / 365)) ** -0.25) if sc.one_month_vol_shock else 1.0
        vol = index.vol * (1 + (sc.vol_multiplier - 1) * damp)
        stresses[sc.name] = (
            _book(index.s * (1 + sc.spot_change), index.t, index.r + sc.rate_change, index.q, vol) - today
        )
    name, loss = min(stresses.items(), key=lambda item: item[1])
    assert tiles["Worst stress"].value == tab_risk.dollars(loss)
    assert tiles["Worst stress"].delta == name.split(" (")[0]


def test_the_position_line_names_legs_expiry_size_and_max_loss(index):
    at = _risk()
    line = next(m.value for m in at.markdown if "max loss" in m.value)
    # The spread's maximum loss is its premium: 100 × (C(5,815) − C(5,915)).
    premium = _book(index.s, index.t, index.r, index.q, index.vol)
    assert line == (
        "**Bull Call Spread** — Long 1 × 5,815 Call, Short 1 × 5,915 Call · 1 × 100 multiplier · "
        f"20‑Nov‑26 AM (27d 17h) · max loss −\\${premium:,.0f}"
    )
    textbook = next(m.value for m in _risk(preset="Textbook").markdown if "max loss" in m.value)
    assert "· 365d ·" in textbook  # the textbook market counts days, not dates


def test_tiles_follow_the_what_if_state(index):
    # ?ws=-2&wv=3&wd=5: Friday 23-Oct 16:00 to Friday 30-Oct 16:00 is seven calendar days.
    at = _risk(ws=-2, wv=3, wd=5)
    s, t, v = index.s * 0.98, index.t - 7 / 365, index.vol + 0.03
    delta = _book_greeks(s, t, index.r, index.q, v)[0]
    assert _tiles(at)["$Δ"].value == tab_risk.dollars(delta * s)
    line = next(m.value for m in at.markdown if "max loss" in m.value)
    pnl = _book(s, t, index.r, index.q, v) - _book(index.s, index.t, index.r, index.q, index.vol)
    assert line.endswith(f"what-if P&L {tab_risk.dollars(pnl)}".replace("$", "\\$"))
    assert any("T+5 · Fri 30‑Oct‑26" in s.label for s in at.slider if s.key == "wd")


# --- the reports ------------------------------------------------------------------------------------------------------


def test_the_ladder_zero_row_equals_the_tiles_and_rolls_to_the_next_session(index):
    at = _risk()
    ladder = _frame(at, "Move")
    zero = ladder[ladder["Move"].str.startswith("0")].iloc[0]
    tiles = _tiles(at)
    assert zero["P&L"] == 0
    for column, label in (("$Δ", "$Δ"), ("Γ P&L 1%", "Γ P&L 1%"), ("Vega", "Vega ($/vp)"), ("Θ", "Θ ($/day)")):
        assert f"{zero[column]:,.0f}" == tiles[label].value.replace("$", "").lstrip("+").replace("−", "-"), column
    rolled = _frame(_risk(lad_roll="Next session"), "Move")
    zero = rolled[rolled["Move"].str.startswith("0")].iloc[0]
    decay = _book(index.s, index.t - 3 / 365, index.r, index.q, index.vol) - _book(
        index.s, index.t, index.r, index.q, index.vol
    )
    assert zero["P&L"] == pytest.approx(decay, abs=1.0)  # the column's whole dollars


def test_small_greek_columns_keep_their_cents_as_the_tiles_do():
    # The textbook market's single contract of a 100/110 call spread: Γ P&L 1% −$0.05 and Θ −$0.14 a day, which
    # whole dollars would show as 0 beside tiles that read −$0.05 and −$0.14.
    at = _risk(preset="Textbook")
    tiles = _tiles(at)
    ladder = next(d for d in at.dataframe if "Move" in d.value.columns)
    zero = ladder.value[ladder.value["Move"].str.startswith("0")].iloc[0]
    for column, label in (("Γ P&L 1%", "Γ P&L 1%"), ("Θ", "Θ ($/day)")):
        assert tab_risk.dollars(zero[column]) == tiles[label].value, column
    config = json.loads(ladder.proto.columns)
    # A column has one precision: vega reaches −$14 at +20%, so it stays in whole dollars (−2 beside −$2.05).
    assert ladder.value["Vega"].abs().max() >= 10 and config["Vega"]["type_config"]["format"] == "%,.0f"
    assert config["Γ P&L 1%"]["label"] == "Γ P&L 1% ($)" and config["Θ"]["label"] == "Θ ($/day)"
    assert config["Θ"]["type_config"]["format"] == "%,.2f" and config["$Δ"]["type_config"]["format"] == "%,.0f"
    # Under the index the same columns stay in whole dollars.
    index_ladder = next(d for d in _risk().dataframe if "Move" in d.value.columns)
    assert json.loads(index_ladder.proto.columns)["Θ"]["type_config"]["format"] == "%,.0f"


def test_the_ladder_in_sigma_and_second_order():
    ladder = _frame(_risk(lad_axis="σ", lad_order="+2nd"), "Move")
    assert len(ladder) == len(tab_risk.LADDER_SIGMA)
    assert {"Vanna", "Volga", "Charm"} <= set(ladder.columns)
    assert ladder["Move"].iloc[0].startswith("−3σ · −")


def test_the_stress_table_is_sorted_worst_first_with_readable_labels():
    at = _risk(report="Stress")
    table = _frame(at, "Scenario")
    assert table["P&L"].is_monotonic_increasing
    assert len(table) == len(tab_risk.ALL_STRESSES)
    assert "IV damped" not in table.columns and "IV shock damped" not in table.columns
    assert set(table.columns) == {"Scenario", "Source", "Spot", "1M vol", "Rate", "Days", "P&L"}
    bars = _chart(at, "stress_chart")["data"][0]
    assert list(bars["y"]) == table["Scenario"].tolist()
    # A label sits inside its bar only on a bar that spans a third of the axis or more; a short bar's label goes
    # outside it, in the text colour, rather than running out of the bar in the bar's label colour.
    pnl = table["P&L"].to_numpy(float)
    third = (pnl.max() - pnl.min()) / 3
    assert bars["textposition"] == ["inside" if abs(v) >= third else "outside" for v in pnl]
    assert "inside" in bars["textposition"] and "outside" in bars["textposition"]


def test_the_desk_grid_axes_and_its_zero_cell():
    at = _risk(report="Scenarios")
    heat = _chart(at, "sc_heatmap_True")["data"][0]
    assert all(label.endswith("vp") for label in heat["y"])
    assert len(heat["x"]) == 9 and len(heat["y"]) == 7
    assert all(re.fullmatch(r"[0-9,]+ \([+−]?[0-9.]+%\)", label) for label in heat["x"]), heat["x"]
    z = _array(heat["z"])
    assert z[heat["y"].index("0 vp"), heat["x"].index("5,800 (0%)")] == 0.0


def test_the_spot_by_date_grid_uses_trading_dates():
    heat = _chart(_risk(report="Scenarios", sc_axes="Spot × Date"), "sc_heatmap_False")["data"][0]
    assert heat["y"][0].startswith("T+0 · Fri 23‑Oct") and heat["y"][-1].endswith("exp")
    assert not any("Day " in label for label in heat["y"])
    # Thanksgiving (26-Nov) is no session, and the last rows are daily up to the 20-Nov expiry.
    assert heat["y"][-6:] == [
        "T+15 · Fri 13‑Nov",
        "T+16 · Mon 16‑Nov",
        "T+17 · Tue 17‑Nov",
        "T+18 · Wed 18‑Nov",
        "T+19 · Thu 19‑Nov",
        "T+20 · Fri 20‑Nov exp",
    ]


def test_the_finra_grid_and_its_illustrative_requirement(index):
    at = _risk(report="Scenarios", sc_grid="Portfolio margin")
    heat = _chart(at, "sc_heatmap_margin")["data"][0]
    assert len(heat["x"]) == 10
    assert heat["x"][0].endswith("(−8%)") and heat["x"][-1].endswith("(+6%)")
    today = _book(index.s, index.t, index.r, index.q, index.vol)
    worst = _book(index.s * 0.92, index.t, index.r, index.q, index.vol) - today
    line = next(m.value for m in at.markdown if "Requirement" in m.value)
    assert "illustrative" in line and "4210(g)(7)(B)" in line
    assert f"−\\${abs(worst):,.0f} at −8%" in line
    assert "−\\$3,947 at −8%" in line


def test_the_var_report(index):
    at = _risk(report="VaR")
    hist = _chart(at, "var_hist")
    assert hist["layout"]["yaxis"]["title"]["text"] == "Probability (%)"
    cards = {m.label: m for m in at.metric}
    for label in ("VaR 99% 1D · MC", "ES 97.5% 1D · MC", "Spot-only VaR 99% 1D", "Spot-only ES 97.5% 1D"):
        assert cards[label].value.startswith("−"), label
    # The comparator's gap is taken like for like, to the spot-only run (implied vol frozen in both), and from the
    # amounts shown, so that the three tiles add up on screen: −$1,922 − (−$1,978) = +$56.
    gap = tab_risk.shown(_amount(cards["Spot-only VaR 99% 1D"].value)) - _amount(cards["DGT VaR 99% 1D"].value)
    assert cards["Non-linearity"].value == tab_risk.dollars(gap)
    assert cards["Non-linearity"].delta == "spot only − DGT"
    # The models simulate from the leg vol: no statistic of a history, so none is tagged SYN HIST; the line under
    # the report points to the backtests on the synthetic history.
    assert not any("SYN HIST" in label for label in cards)
    assert any("SYN HIST" in c.value and "Methodology" in c.value for c in at.caption)
    # The VaR, ES and spot-only numbers are read off the cached simulations with the Milestone 1 defaults.
    assert "vs spot only" in cards["VaR 99% 1D · MC"].delta
    # The bars are shares of all outcomes in percent: they add up to the share inside the plotted range.
    bars = _bars(hist)
    assert 99.0 < bars.sum() <= 100.0 + 1e-9


def _amount(text: str) -> float:
    """A tile's dollar amount as a number: ``−$1,922`` is −1,922."""
    return float(text.replace("−", "-").replace("$", "").replace(",", ""))


def _bars(fig: dict) -> np.ndarray:
    return _array(fig["data"][0]["y"])


def _array(value) -> np.ndarray:
    """A trace's data array; Plotly sends numeric NumPy arrays as base64 ``bdata`` (with a shape for 2-D)."""
    import base64

    if isinstance(value, dict) and "bdata" in value:
        array = np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"])
        shape = value.get("shape")
        if shape is not None:
            array = array.reshape([int(n) for n in str(shape).split(",")] if isinstance(shape, str) else shape)
        return array
    return np.asarray(value, dtype=float)


def test_the_vega_report_buckets_add_up_to_the_vega_tile():
    at = _risk(report="Vega")
    table = _frame(at, "Expiry")
    total = table[table["Expiry"] == "Total"]["Total"].iloc[0]
    assert f"{total:+,.0f}" == _tiles(at)["Vega ($/vp)"].value.replace("$", "")
    assert table["Expiry"].tolist()[:3] == ["≤ 7d", "7–14d", "14–30d"]
    assert list(table.columns) == ["Expiry", "10Δp", "25Δp", "ATM", "25Δc", "10Δc", "Total"]
    # The 20-Nov legs (27.7 days) sit in 14-30d: the long 5,815 at the money, the short 5,915 in 25Δc.
    row = table[table["Expiry"] == "14–30d"].iloc[0]
    assert row["ATM"] > 0 > row["25Δc"]


def test_the_explain_report_explains_the_what_if_move(index):
    at = _risk(report="Explain", ws=-2, wv=3, wd=5)
    table = _frame(at, "Term")
    total = _book(index.s * 0.98, index.t - 7 / 365, index.r, index.q, index.vol + 0.03) - _book(
        index.s, index.t, index.r, index.q, index.vol
    )
    full = table[table["Term"] == "Full revaluation"]["P&L"].iloc[0]
    assert full == pytest.approx(total, abs=0.5)
    terms = table[table["Term"] != "Full revaluation"]["P&L"]
    assert terms.sum() == pytest.approx(full, abs=len(terms) * 0.5)  # each row rounded to the dollar
    shapley = _frame(_risk(report="Explain", ws=-2, wv=3, wd=5, ex_method="Shapley"), "Term")
    assert shapley["Term"].tolist() == ["Spot", "Vol", "Time", "Rate", "Full revaluation"]
    # No shift set: the what-if has nothing to explain, so the report explains the next session (Friday 16:00 to
    # Monday 16:00, three calendar days, nothing else moving) and says so.
    quiet = _risk(report="Explain")
    assert any(c.value.startswith("No what-if shift is set") for c in quiet.caption)
    decay = _book(index.s, index.t - 3 / 365, index.r, index.q, index.vol) - _book(
        index.s, index.t, index.r, index.q, index.vol
    )
    full = _frame(quiet, "Term").set_index("Term").loc["Full revaluation", "P&L"]
    assert full == pytest.approx(decay, abs=0.005)  # the column's cents


@pytest.mark.parametrize("report", tab_risk.REPORTS)
@pytest.mark.parametrize("preset", [None, "Textbook"])
def test_every_report_runs_with_short_captions(report, preset):
    at = _risk(report=report, preset=preset)
    captions = [c.value for c in at.tabs[2].caption]  # the Risk tab body (the share note is not its own)
    assert all(len(c) <= 140 for c in captions), [c for c in captions if len(c) > 140]
    # The removed views stay removed (spec 3.3 Removed).
    keys = {c.proto.id.rsplit("-", 1)[-1] for c in at.get("plotly_chart")}
    assert not keys & {"time_decay", "greek_profiles", "risk_heatmap", "date_heatmap"}


def test_without_a_position_the_tab_says_so_and_opens_strategy():
    page = AppTest.from_string(
        "from options_analyzer.ui import tab_risk\ntab_risk.render(None, None)", default_timeout=60
    ).run()
    assert page.info[0].value.startswith("The Strategy tab's legs are invalid")
    page.button(key="risk_open_strategy").click().run()
    assert page.session_state["tab"] == "Strategy"


# --- UI-10: a scenario cell into the what-if strip ------------------------------------------------------------------

AXES = GridAxes(
    tab_risk.DESK_SPOT,
    tuple((v, 3) for v in tab_risk.DESK_VOL["vp"]),
    "absolute",
    tuple(tab_risk.shock_label(x / 100) for x in tab_risk.DESK_SPOT),
)


@pytest.fixture
def state(monkeypatch):
    import streamlit as st

    store: dict = {}
    monkeypatch.setattr(st, "session_state", store)
    return store


def test_a_heatmap_point_sets_spot_vol_and_date(state):
    # The overlay's points run row by row: row 4 (+2 vp), column 6 (+5%) is point 4 × 9 + 6.
    tab_risk.on_scenario_select({"selection": {"points": [{"point_index": 42, "x": "6,090 (+5%)"}]}}, AXES)
    assert state == {"wvmode": "absolute", "ws": 5.0, "wv": 2.0, "wd": 3}


def test_a_table_cell_sets_spot_vol_and_date(state):
    tab_risk.on_scenario_select({"selection": {"cells": [(1, "+5%")]}}, AXES)
    assert state == {"wvmode": "absolute", "ws": 5.0, "wv": -5.0, "wd": 3}
    state.clear()
    tab_risk.on_scenario_select({"selection": {"cells": [(1, "Vol shock (vp)")]}}, AXES)  # the row label
    tab_risk.on_scenario_select({"selection": {"points": [], "cells": []}}, AXES)  # a cleared selection
    assert state == {}


def test_a_relative_grid_sets_the_relative_mode(state):
    axes = GridAxes(tab_risk.DESK_SPOT, tuple((v, 0) for v in tab_risk.DESK_VOL["relative"]), "relative")
    tab_risk.on_scenario_select({"selection": {"points": [{"point_index": 0}]}}, axes)
    assert state == {"wvmode": "relative", "ws": -10.0, "wv": -50.0, "wd": 0}


def test_the_strips_cell_is_outlined_in_amber():
    at = _risk(report="Scenarios", ws=5, wv=2)
    shapes = _chart(at, "sc_heatmap_True")["layout"]["shapes"]
    amber = [s for s in shapes if s["line"]["dash"] == "solid"]
    assert len(amber) == 1 and amber[0]["x0"] == 5.5 and amber[0]["y0"] == 3.5  # column +5%, row +2 vp
    resting = _chart(_risk(report="Scenarios"), "sc_heatmap_True")["layout"]["shapes"]
    assert all(s["line"]["dash"] == "dot" for s in resting)  # the worst cell only


# --- Milestone 1, kept: implied-vol moves in the VaR by default, beside the spot-only VaR; decay beside theta -------

#: The review panel's VaR case (market maker, finding 2): a 45-day at-the-money short straddle on an index, one
#: contract of 100 shares, on the textbook market's typed inputs.
SPX_STRADDLE = {
    "preset": "Textbook",
    "spot": 5800,
    "strike": 5800,
    "days": 45,
    "vol": 16,
    "rate": 4.3,
    "div": 1.3,
    "strategy": "Short Straddle",
}


def _var_configs(monkeypatch) -> list:
    """The configuration of every VaR simulation from now on, in order (the cache is cleared, so none hides there)."""
    from options_analyzer import risk
    from options_analyzer.ui import cache

    configs = []
    real = risk.value_at_risk
    monkeypatch.setattr(
        risk, "value_at_risk", lambda s, m, config, n=1.0: configs.append(config) or real(s, m, config, n)
    )
    cache.value_at_risk.clear()
    return configs


def test_var_simulates_index_vol_dynamics_beside_the_spot_only_var(monkeypatch):
    # The panel's numbers: VaR $2,833 with implied vol frozen, $3,862 with index dynamics (vol of vol 100%, spot-vol
    # correlation -0.7), 27% more. Both show, signed as P&L, and the headline cards say what the vol moves add.
    from dataclasses import replace

    from options_analyzer import risk
    from options_analyzer.strategies import Market, build_template

    configs = _var_configs(monkeypatch)
    at = _risk(report="VaR", **SPX_STRADDLE)
    moving = risk.VaRConfig(0.99, 1, 100_000, seed=7, vol_of_vol=1.0, spot_vol_correlation=-0.7)
    assert configs == [replace(moving, vol_of_vol=0.0), moving]
    cards = {m.label: m for m in at.metric}
    assert cards["VaR 99% 1D · MC"].value == "−$3,862" and cards["Spot-only VaR 99% 1D"].value == "−$2,833"
    assert cards["VaR 99% 1D · MC"].delta == r"−\$1,029 vs spot only"  # Markdown, as the delta renders it
    # ES at 97.5% (FRTB's level) off the same simulation: the mean of the 2,500 worst of 100,000.
    straddle = build_template("Short Straddle", 5800, 580, 45 / 365)
    run = risk.value_at_risk(straddle, Market(5800, 0.043, 0.16, 45 / 365, 0.013), moving, 100)
    es = -np.sort(run.pnl)[:2_500].mean()
    assert cards["ES 97.5% 1D · MC"].value == tab_risk.dollars(-es)
    assert at.toggle(key="var_volmoves").value is True
    assert at.number_input(key="var_vov").value == 100.0 and at.slider(key="var_corr").value == -0.7


def test_the_spot_only_run_keeps_every_setting_but_the_vol_moves(monkeypatch):
    from dataclasses import replace

    from options_analyzer import risk

    configs = _var_configs(monkeypatch)
    at = _risk(report="VaR")
    at.number_input(key="var_horizon").set_value(5)
    at.select_slider(key="var_paths").set_value(10_000)
    at.number_input(key="var_vov").set_value(150.0)
    at.slider(key="var_corr").set_value(-0.5)
    at.number_input(key="var_seed").set_value(11)
    at.slider(key="var_dof").set_value(4)
    at.segmented_control(key="var_model").set_value("Student-t MC").run()
    _ok(at)
    config = risk.VaRConfig(
        0.99, 5, 10_000, distribution="student_t", dof=4.0, seed=11, vol_of_vol=1.5, spot_vol_correlation=-0.5
    )
    assert configs[-2:] == [replace(config, vol_of_vol=0.0, spot_vol_correlation=-0.7), config]
    assert [m.label for m in at.metric if "VaR" in m.label or "ES" in m.label][:4] == [
        "VaR 99% 5D · MC",
        "ES 97.5% 5D · MC",
        "Spot-only VaR 99% 5D",
        "Spot-only ES 97.5% 5D",
    ]


def test_vol_moves_switched_off_show_the_spot_only_run_from_the_cache(monkeypatch):
    configs = _var_configs(monkeypatch)
    at = _risk(report="VaR", **SPX_STRADDLE)
    at.toggle(key="var_volmoves").set_value(False).run()
    _ok(at)
    assert len(configs) == 2  # the frozen run came from the cache
    labels = [m.label for m in at.metric if "VaR" in m.label or "ES" in m.label or m.label == "Non-linearity"]
    assert labels == ["VaR 99% 1D · MC", "ES 97.5% 1D · MC", "DGT VaR 99% 1D", "Non-linearity"]
    assert next(m for m in at.metric if m.label == "VaR 99% 1D · MC").value == "−$2,833"


def test_the_theta_tile_gives_the_decay_to_expiry_of_a_one_day_position():
    # A 1-day short straddle: repricing gains its whole value, $4,359.89, where theta / 365 says $2,179.76 a day.
    from options_analyzer import black_scholes as bs

    tile = _tiles(_risk(**(SPX_STRADDLE | {"days": 1, "vol": 18})))["Θ ($/day)"]
    value = 100 * sum(bs.price(5800, 5800, 1 / 365, 0.043, 0.18, 0.013, kind) for kind in ("call", "put"))
    assert value == pytest.approx(4359.89, abs=0.005)
    assert tile.value == "+$2,180" and tile.delta == r"Decay to expiry +\$4,360"
    assert _tiles(_risk(**SPX_STRADDLE))["Θ ($/day)"].delta.startswith("BE ")  # 45 days: theta leads


def test_every_risk_chart_has_the_desk_config():
    for query in ({}, {"report": "Scenarios"}, {"report": "Stress"}, {"report": "VaR"}, {"report": "Vega"},
                  {"report": "Explain", "ws": 1}, {"report": "Scenarios", "sc_grid": "Portfolio margin"}):  # fmt: skip
        for chart in _risk(**query).get("plotly_chart"):
            key = chart.proto.id.rsplit("-", 1)[-1]
            config = json.loads(chart.proto.config)
            assert config["toImageButtonOptions"]["filename"] == f"options-analyzer-{key}", key
            assert len(chart.proto.alt) >= 20 and not chart.proto.alt[0].islower(), (key, chart.proto.alt)
