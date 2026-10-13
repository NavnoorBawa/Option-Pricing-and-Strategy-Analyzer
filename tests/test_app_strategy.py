"""Strategy Builder and Risk & Scenarios tabs: app behaviour (streamlit.testing) and their display helpers."""

import base64
import json
import re
from pathlib import Path

import numpy as np
import pytest
from streamlit.proto.WidgetStates_pb2 import WidgetState
from streamlit.testing.v1 import AppTest

from options_analyzer import strategies as strat
from options_analyzer.ui import charts, tab_risk, tab_strategy

APP = str(Path(__file__).resolve().parents[1] / "main.py")
NO_EDITS = {"edited_rows": {}, "added_rows": [], "deleted_rows": []}


def _ok(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def _app(**query) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    for name, value in query.items():
        at.query_params[name] = str(value)
    at.run()
    _ok(at)
    return at


def _walk(node):
    yield node
    children = getattr(node, "children", None)
    for child in (children.values() if isinstance(children, dict) else children) or ():
        yield from _walk(child)


def _editor_id(at: AppTest) -> str:
    ids = [
        node.proto.id
        for node in _walk(at._tree)
        if type(getattr(node, "proto", None)).__name__ == "Dataframe" and "legs::" in node.proto.id
    ]
    assert len(ids) == 1, ids
    return ids[0]


class LegsEditor:
    """Edits the legs table the way the browser does (AppTest has no data-editor API).

    The browser keeps an editor's edits and re-sends them with every rerun for as long as the editor keeps its
    identity; when the app gives it a new identity, the edits are gone.
    """

    def __init__(self, at: AppTest):
        self.at, self.id, self.state = at, None, None

    def edit(self, edited_rows=None, added_rows=(), deleted_rows=()) -> AppTest:
        self.id = _editor_id(self.at)
        self.state = {
            "edited_rows": edited_rows or {},
            "added_rows": list(added_rows),
            "deleted_rows": list(deleted_rows),
        }
        return self.run()

    def run(self) -> AppTest:
        states = self.at._tree.get_widget_states()
        if self.id is not None and _editor_id(self.at) == self.id:
            states.widgets.append(WidgetState(id=self.id, string_value=json.dumps(self.state)))
        self.at = self.at._run(states)
        _ok(self.at)
        return self.at


def _position(at: AppTest) -> str:
    return next(m.value for m in at.markdown if m.value.startswith("Analysing"))


def _metric(at: AppTest, label: str):
    return next(m for m in at.metric if m.label == label)


def _summary_line(at: AppTest) -> str:
    return next(m.value for m in at.markdown if "Reward / risk" in m.value)


def _chart(at: AppTest, key: str) -> dict:
    return next(json.loads(c.proto.spec) for c in at.get("plotly_chart") if c.proto.id.endswith(key))


def _array(value) -> np.ndarray:
    """A trace's data array; Plotly sends numeric NumPy arrays as base64 ``bdata``."""
    if isinstance(value, dict) and "bdata" in value:
        array = np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"])
        shape = value.get("shape")
        if shape is not None:
            array = array.reshape([int(n) for n in str(shape).split(",")] if isinstance(shape, str) else shape)
        return array
    return np.asarray(value)


# --- Leg edits are never silently lost (review finding: the editor was recreated on sidebar changes) -------------


@pytest.mark.parametrize(
    ("key", "value"), [("vol", 25.0), ("vol", 30.0), ("days", 200), ("spot", 105.0), ("rate", 3.0)]
)
def test_leg_edits_survive_sidebar_changes(key, value):
    editor = LegsEditor(_app())
    editor.edit({"0": {"Quantity": 3}})
    assert "Long 3 × 100 Call" in _position(editor.at)
    editor.at.sidebar.number_input(key=key).set_value(value)
    at = editor.run()
    assert "**Bull Call Spread (modified)** — Long 3 × 100 Call, Short 1 × 110 Call" in _position(at)


def test_edited_template_is_kept_on_a_strike_change_until_reset():
    editor = LegsEditor(_app())
    editor.edit({"0": {"Quantity": 3}})
    editor.at.sidebar.number_input(key="strike").set_value(95.0)
    at = editor.run()
    assert "Long 3 × 100 Call, Short 1 × 110 Call" in _position(at)
    assert any("kept" in info.value for info in at.info)
    at.button(key="reset_legs").click()
    at = editor.run()
    assert "**Bull Call Spread** — Long 1 × 95 Call, Short 1 × 105 Call" in _position(at)
    assert not any("kept" in info.value for info in at.info)


def test_unedited_template_follows_the_sidebar_strike():
    at = _app()
    at.sidebar.number_input(key="strike").set_value(95.0).run()
    _ok(at)
    assert "**Bull Call Spread** — Long 1 × 95 Call, Short 1 × 105 Call" in _position(at)


def test_custom_position_survives_volatility_and_strike_changes():
    at = _app()
    at.selectbox(key="strategy_category").set_value("Custom").run()
    editor = LegsEditor(at)
    editor.edit(added_rows=[{"Type": "put", "Quantity": -1, "Strike": 90}, {"Type": "stock", "Quantity": 1}])
    built = "**Custom** — Long 1 × 100 Call, Short 1 × 90 Put, Long 1 share"
    assert built in _position(editor.at)
    for key, value in (("vol", 30.0), ("strike", 101.0)):
        editor.at.sidebar.number_input(key=key).set_value(value)
        assert built in _position(editor.run())


def test_edits_survive_switching_to_another_template_and_back():
    editor = LegsEditor(_app())
    editor.edit({"0": {"Quantity": 3}})
    editor.at.selectbox(key="strategy_Vertical spread").set_value("Bear Call Spread")
    assert "**Bear Call Spread** —" in _position(editor.run())
    editor.at.selectbox(key="strategy_Vertical spread").set_value("Bull Call Spread")
    assert "Long 3 × 100 Call, Short 1 × 110 Call" in _position(editor.run())


def test_blank_strike_means_the_sidebar_strike():
    editor = LegsEditor(_app())
    at = editor.edit(added_rows=[{"Type": "put", "Quantity": 1}])
    assert "Long 1 × 100 Put" in _position(at)
    # Like a blank expiry or IV, a blank strike follows the sidebar.
    editor.at.sidebar.number_input(key="strike").set_value(90.0)
    assert "Long 1 × 100 Call, Short 1 × 110 Call, Long 1 × 90 Put" in _position(editor.run())


def test_reverted_edit_on_kept_legs_is_not_moved_by_a_later_rerun():
    # Round-1 verifier: after a strike change kept the edited legs, editing them back to the old template values
    # left them on the old strikes, and the next rerun from any widget silently moved them to the new template.
    editor = LegsEditor(_app())
    editor.edit({"0": {"Quantity": 3}})
    editor.at.sidebar.number_input(key="strike").set_value(95.0)
    editor.run()
    editor.state = {**NO_EDITS, "edited_rows": {"0": {"Quantity": 1}}}
    at = editor.run()
    kept = "**Bull Call Spread (modified)** — Long 1 × 100 Call, Short 1 × 110 Call"
    assert kept in _position(at)
    assert any("kept" in info.value for info in at.info)
    at.number_input(key="contracts").set_value(2)
    at = editor.run()
    assert f"{kept} · 2 × 100 multiplier" in _position(at)
    # Once the legs match the template on screen again, they follow it.
    at.sidebar.number_input(key="strike").set_value(100.0)
    at = editor.run()
    assert "**Bull Call Spread** — Long 1 × 100 Call, Short 1 × 110 Call" in _position(at)
    assert not any("kept" in info.value for info in at.info)
    at.sidebar.number_input(key="strike").set_value(90.0)
    assert "**Bull Call Spread** — Long 1 × 90 Call, Short 1 × 100 Call" in _position(editor.run())


def test_an_edit_that_arrives_with_a_strike_change_is_kept():
    # The browser sends both in one rerun when the strike changes while the edit's own rerun is still running.
    editor = LegsEditor(_app())
    editor.at.sidebar.number_input(key="strike").set_value(95.0)
    at = editor.edit({"0": {"Quantity": 3}})
    assert "**Bull Call Spread (modified)** — Long 3 × 100 Call, Short 1 × 110 Call" in _position(at)
    assert any("kept" in info.value for info in at.info)


def test_editor_identity_does_not_depend_on_sidebar_values():
    # Root cause of the lost edits: the column help quoted the sidebar values and so changed the editor's identity.
    at = _app(strategy_width_pct=10)  # a fixed width keeps the template itself unchanged
    before = _editor_id(at)
    for key, value in (("vol", 30.0), ("days", 90), ("spot", 120.0)):
        at.sidebar.number_input(key=key).set_value(value).run()
        assert _editor_id(at) == before


# --- Strike width and defaults --------------------------------------------------------------------------------


def test_strike_width_follows_the_sidebar_strike():
    at = _app(spot=8, strike=8)
    at.selectbox(key="strategy_Vertical spread").set_value("Bull Put Spread").run()
    _ok(at)
    assert "Short 1 × 8 Put, Long 1 × 7 Put" in _position(at)
    at.number_input(key="strategy_width_pct").set_value(10.0).run()
    assert "Short 1 × 8 Put, Long 1 × 7.2 Put" in _position(at)
    at.sidebar.number_input(key="strike").set_value(500.0).run()
    _ok(at)
    assert "Short 1 × 500 Put, Long 1 × 450 Put" in _position(at)


@pytest.mark.parametrize(
    ("strike", "vol", "years", "width"),
    [
        (100, 0.2, 1.0, 10.0),
        (100, 0.2, 7 / 365, 1.0),
        (100, 0.3, 1.0, 20.0),
        (4500, 0.2, 1.0, 500.0),
        (8, 0.2, 1.0, 1.0),
        (100, 5.0, 10, 25.0),
    ],
)
def test_default_width_scales_with_the_expected_move(strike, vol, years, width):
    assert tab_strategy.auto_width(strike, vol, years) == pytest.approx(width)


@pytest.mark.parametrize(
    ("width", "rounded"), [(10.137, 10.14), (2.5, 2.5), (0.123, 0.12), (0.065, 0.065), (0.0123, 0.012), (0.025, 0.025)]
)
def test_round_width_keeps_strikes_on_a_price_tick(width, rounded):
    assert tab_strategy.round_width(width) == rounded


def _caption(at: AppTest, text: str) -> str:
    return next(c.value for c in at.caption if text in c.value)


def test_percent_width_is_rounded_to_the_cent_and_shown_as_used():
    # Round-1 verifier: 10% of a 101.37 strike gave a 111.507 call while the caption said w = $10.14.
    at = _app(spot=101.37, strike=101.37, strategy_width_pct=10)
    assert "Long 1 × 101.37 Call, Short 1 × 111.51 Call" in _position(at)
    assert "w = \\$10.14 (10.00% of the strike, set by you)" in _caption(at, "Width")


def test_sub_cent_strikes_are_shown_in_full():
    # Round-1 verifier: a penny condor's 0.975/1.025 strikes showed as 0.97/1.02 and its width as $0.03.
    at = _app(spot=1, strike=1, days=30, strategy_category="Range-bound")
    at.selectbox(key="strategy_Range-bound").set_value("Iron Condor").run()
    _ok(at)
    assert "Long 1 × 0.95 Put, Short 1 × 0.975 Put, Short 1 × 1.025 Call, Long 1 × 1.05 Call" in _position(at)
    assert "w = \\$0.025 (2.50% of the strike, automatic)" in _caption(at, "Width")
    editor = next(n.proto for n in _walk(at._tree) if "legs::" in getattr(getattr(n, "proto", None), "id", ""))
    assert json.loads(editor.columns)["Strike"]["type_config"]["format"] == tab_strategy.STRIKE_FORMAT
    # The format is printf-style and the browser applies it the way Python's % operator does.
    shown = [tab_strategy.STRIKE_FORMAT % k for k in (0.95, 0.975, 1.025, 1.05, 100.0, 111.51, 1 - 2 * 0.025)]
    assert shown == ["0.95", "0.975", "1.025", "1.05", "100", "111.51", "0.95"]


def test_short_dated_condor_is_not_degenerate_by_default():
    # Review finding: a 10%-of-strike width put a 7-day condor's short strikes 3.6 standard deviations away.
    at = _app(days=7, strategy_category="Range-bound")
    at.selectbox(key="strategy_Range-bound").set_value("Iron Condor").run()
    _ok(at)
    pop = float(_metric(at, "Probability of profit").value.rstrip("%")) / 100
    assert 0.2 < pop < 0.8
    at.number_input(key="strategy_width_pct").set_value(10.0).run()
    assert _metric(at, "Probability of profit").value == ">99.9%"


def test_probability_text_never_rounds_to_a_certainty():
    assert tab_strategy.probability_text(0.99962) == ">99.9%"
    assert tab_strategy.probability_text(0.0003) == "<0.1%"
    # Probabilities that saturate to exactly 1 or 0 in floating point are no certainty either...
    assert tab_strategy.probability_text(1.0) == ">99.9%"
    assert tab_strategy.probability_text(0.0) == "<0.1%"
    # ...unless the payoff itself cannot lose or cannot profit.
    assert tab_strategy.probability_text(1.0, sure=True) == "100.0%"
    assert tab_strategy.probability_text(0.0, never=True) == "0.0%"
    assert tab_strategy.probability_text(0.523) == "52.3%"


def test_probability_of_profit_is_not_a_certainty_when_the_position_can_lose_or_profit():
    # Round-1 verifier: a 7-day short 60 put has a probability of exactly 1.0 in floating point but can lose
    # $6,000, and a far out-of-the-money call spread showed 0.0% with a $200 maximum profit.
    at = _app(days=7, strike=60, strategy_category="Single option")
    at.selectbox(key="strategy_Single option").set_value("Short Put").run()
    _ok(at)
    assert _metric(at, "Max loss").value == "$6,000.00"
    assert _metric(at, "Probability of profit").value == ">99.9%"
    at = _app(days=7, strike=150)
    assert _metric(at, "Max profit").value.startswith("$")
    assert _metric(at, "Probability of profit").value == "<0.1%"


# --- Metrics -----------------------------------------------------------------------------------------------------


def test_reward_to_risk_is_never_an_absurd_number():
    # Review finding: a $0.00 maximum loss produced a 50-digit ratio.
    assert "**Reward / risk:** 1.27 : 1" in _summary_line(_app())
    at = _app(strike=150, days=7, strategy_width_pct=10)
    assert _metric(at, "Max loss").value == "$0.00"
    assert "**Reward / risk:** –" in _summary_line(at)
    at = _app(strike=125, days=30, strategy_width_pct=10)
    assert _metric(at, "Max loss").value == "$0.01"
    assert "**Reward / risk:** > 1,000 : 1" in _summary_line(at)


def test_both_breakevens_are_shown_in_full():
    # Review finding: the second breakeven was cut off in a narrow metric card.
    at = _app(strategy_category="Range-bound")
    at.selectbox(key="strategy_Range-bound").set_value("Iron Condor").run()
    assert re.search(
        r"Breakevens at expiry:\*\* \\\$85\.58 \(-14\.4%\) and \\\$114\.42 \(\+14\.4%\)", _summary_line(at)
    )


def test_large_amounts_are_shortened_with_the_exact_value_in_the_help():
    at = _app(spot=4500, strike=4500, contracts=10)
    premium = _metric(at, "Net premium")
    assert premium.value == "$215,715"
    assert "Exact: \\$215,715" in premium.proto.help
    assert tab_strategy.money_short(-2_345_678.9, signed=True) == "-$2.35M"
    assert tab_strategy.money_short(19_847.23) == "$19,847.23"


def test_cannot_lose_and_no_profit_wording_and_no_arrows():
    box = [
        {"Type": "call", "Quantity": -1, "Strike": 110},
        {"Type": "put", "Quantity": 1, "Strike": 110},
        {"Type": "put", "Quantity": -1, "Strike": 90},
    ]
    at = _app(strategy_category="Custom")
    at = LegsEditor(at).edit({"0": {"Strike": 90}}, added_rows=box)
    assert _metric(at, "Max loss").value == "Cannot lose"
    assert _metric(at, "Delta (shares)").value == "0.00"
    premium = _metric(at, "Net premium")
    assert premium.proto.delta == "debit (paid)"
    assert premium.proto.direction == premium.proto.MetricDirection.NONE  # no up arrow on a label
    short_box = [{**row, "Quantity": -row["Quantity"]} for row in box]
    at = LegsEditor(_app(strategy_category="Custom")).edit({"0": {"Strike": 90, "Quantity": -1}}, added_rows=short_box)
    assert _metric(at, "Max profit").value == "No profit"


def test_legs_table_has_no_negative_zeros():
    # Round-1 verifier: far out-of-the-money legs showed Cost -$0.00, Delta -0.00, Gamma -0.0000 and so on.
    at = _app(strike=150, days=7)
    table = next(df.value for df in at.dataframe if "Model value" in df.value.columns)
    rounded = 0
    shown_decimals = {
        "Model value": 4,
        "Entry price": 4,
        "Cost": 2,
        "Delta": 2,
        "Gamma": 4,
        "Theta/day": 2,
        "Vega/pt": 2,
    }
    for col, decimals in shown_decimals.items():
        values = table[col].to_numpy(dtype=float)
        shown = [f"{v:.{decimals}f}" for v in values]
        assert not [s for s in shown if re.fullmatch(r"-0\.0+", s)], (col, shown)
        rounded += int(np.sum(np.abs(values) < 0.5 * 10**-decimals))
    assert rounded >= 5  # the case has values that round to zero
    assert shown_decimals == tab_strategy.LEGS_TABLE_DECIMALS


def test_number_has_no_negative_zero():
    assert tab_strategy.number(-1e-12, 2) == "0.00"
    assert tab_strategy.number(-0.004, 2) == "0.00"
    assert tab_strategy.number(-0.006, 2) == "-0.01"


# --- Descriptions and valuation notes --------------------------------------------------------------------------


def test_calendar_is_described_with_its_expiries():
    at = _app(strategy_category="Time spread")
    assert "Short 1 × 100 Call (365d), Long 1 × 100 Call (730d)" in _position(at)


def test_american_sidebar_model_is_compared_with_the_european_leg_value():
    at = _app(div=3)
    at.sidebar.radio(key="model").set_value("binomial").run()
    at.sidebar.segmented_control(key="style").set_value("American").run()
    at.selectbox(key="strategy_category").set_value("Single option").run()
    at.selectbox(key="strategy_Single option").set_value("Long Put").run()
    _ok(at)
    american = _metric(at, "Put value").value
    note = next(i.value for i in at.info if "Early exercise is not modelled" in i.value)
    assert american.replace("$", "\\$") in note
    assert "\\$6.7309 European" in note  # the leg's Black-Scholes-Merton value, which the strategy uses
    assert _metric(at, "Net premium").value == "$673.09"
    assert any("whatever the sidebar pricing model" in c.value for c in at.caption)


# --- Risk tab ------------------------------------------------------------------------------------------------------


def test_risk_heatmap_labels_show_the_exact_spot_shocks():
    at = _app()
    at.slider(key="grid_span").set_value(25).run()
    spec = _chart(at, "risk_heatmap")["data"][0]
    assert spec["x"][:4] == ["-25%", "-20.8%", "-16.7%", "-12.5%"]
    assert spec["x"][6] == "0%"
    assert spec["y"][2] == "IV 0%"


def test_spot_date_grid_steps_in_standard_deviations():
    at = _app()
    heat = _chart(at, "date_heatmap")["data"][0]
    # Dates are the (at most 8) columns and prices the rows, lowest first so that the highest is at the top: the
    # 13 price columns were unreadable on a phone.
    prices = np.array([float(y.replace(",", "")) for y in heat["y"]])
    assert prices[0] == pytest.approx(100 * np.exp(-0.4), abs=0.005)
    assert prices[-1] == pytest.approx(100 * np.exp(0.4), abs=0.005)
    assert np.diff(np.log(prices)) == pytest.approx(np.full(12, 0.8 / 12), abs=1e-4)
    assert heat["x"] == ["Today", "Day 52", "Day 104", "Day 156", "Day 209", "Day 261", "Day 313", "Expiry"]
    z = _array(heat["z"])
    assert z.shape == (13, 8)
    # At expiry and the lowest price, the bull call spread has lost its premium versus today's value.
    assert z[0, -1] == pytest.approx(-441.05, abs=0.01)
    assert z[6, 0] == pytest.approx(0.0, abs=1e-9)  # today at today's spot
    assert any("front expiry on day 365" in c.value for c in at.caption)


@pytest.mark.parametrize("query", [{"vol": 500, "days": 3650}, {"spot": 0.01, "strike": 0.01, "days": 1, "vol": 0.1}])
def test_spot_date_grid_labels_stay_distinct_at_the_sidebar_limits(query):
    # Round-1 verifier: at these sidebar limits the price labels repeated and Plotly merged the rows.
    heat = _chart(_app(**query), "date_heatmap")["data"][0]
    assert len(set(heat["y"])) == len(heat["y"]) == 13


def test_time_decay_moves_scale_with_volatility():
    names = [trace["name"] for trace in _chart(_app(vol=40), "time_decay")["data"]]
    assert names == [
        "Spot -1σ (-33.0%)",
        "Spot -0.5σ (-18.1%)",
        "Spot unchanged",
        "Spot +0.5σ (+22.1%)",
        "Spot +1σ (+49.2%)",
    ]


def test_position_greek_profiles():
    at = _app()
    data = _chart(at, "greek_profiles")["data"]
    assert [trace["name"] for trace in data] == ["Today", "In 182 days", "In 364 days"]
    spots = _array(data[0]["x"])
    i = int(np.argmin(np.abs(spots - 100)))
    strategy = strat.build_template("Bull Call Spread", 100, 10)
    expected = strat.position_greeks(strategy, strat.Market(100, 0.05, 0.2, 1.0), spots[i])["delta"] * 100
    assert _array(data[0]["y"])[i] == pytest.approx(expected)
    at.segmented_control(key="risk_greek").set_value("Vega").run()
    _ok(at)
    assert _chart(at, "greek_profiles")["layout"]["yaxis"]["title"]["text"] == "Vega ($ per vol point)"


def test_stress_table_headers_are_short_with_the_full_name_in_the_help():
    # Round-1 verifier: in the narrow columns 'Rate change (bp)' was cut to 'Rate change (bp', losing its unit.
    at = _app()
    at.toggle(key="stress_elapsed").set_value(True).run()
    _ok(at)
    frame = next(df for df in at.dataframe if "Scenario" in df.value.columns)
    headers = list(frame.value.columns)
    assert headers == [
        "Scenario",
        "P&L",
        "Spot change",
        "IV multiplier",
        "IV damped",
        "Rate Δ (bp)",
        "Days elapsed",
        "Legs expired",
    ]
    config = json.loads(frame.proto.columns)
    assert "IV shock damped" in config["IV damped"]["help"]
    assert "basis points" in config["Rate Δ (bp)"]["help"]


def test_stress_table_and_var_cards():
    at = _app()
    labels = [m.label for m in at.metric]
    assert "VaR 95% CI (simulation)" not in labels
    assert any("95% confidence interval of the VaR estimate" in c.value for c in at.caption)
    full = next(c.value for c in at.caption if c.value.startswith("Full revaluation"))
    assert "ν ≤ 4" not in full  # Normal returns have no degrees of freedom
    at.selectbox(key="var_dist").set_value("student_t").run()
    full = next(c.value for c in at.caption if c.value.startswith("Full revaluation"))
    assert "ν ≤ 4" in full


# --- Chart and label helpers ---------------------------------------------------------------------------------


def _note(fig):
    return next((a for a in fig.layout.annotations if a.name == "outside"), None)


def test_histogram_does_not_pile_the_tails_into_the_end_bins():
    # Review finding: outcomes beyond the 0.05%/99.95% quantiles were clipped into spikes at both ends.
    rng = np.random.default_rng(3)
    values = np.concatenate([rng.standard_normal(200_000), [-60.0, 45.0]])
    fig = charts.histogram_chart(values, "P&L", [(-2.6, "VaR", charts.LOSS)])
    bars = fig.data[0]
    width = bars.x[1] - bars.x[0]
    lo, hi = bars.x[0] - width / 2, bars.x[-1] + width / 2
    inside = np.mean((values >= lo) & (values <= hi))
    assert np.sum(bars.y) * width == pytest.approx(inside)
    assert bars.y[0] < 2 * bars.y[3] and bars.y[-1] < 2 * bars.y[-4]
    below, above = int(np.sum(values < lo)), int(np.sum(values > hi))
    # One short line per side (round-1 verifier: a single line was wider than a phone-width or half-width chart).
    assert _note(fig).text.split("<br>") == [
        f"{below} outcomes below the plotted range (lowest -60.00)",
        f"{above} outcomes above the plotted range (highest 45.00)",
    ]
    # Above the marker labels, with room for both lines in the top margin.
    assert _note(fig).yanchor == "bottom" and _note(fig).yshift == charts.NOTE_SHIFT
    assert fig.layout.margin.t >= charts.NOTE_SHIFT + 2 * 15


def test_histogram_axis_includes_marker_lines():
    values = np.random.default_rng(4).standard_normal(50_000)
    fig = charts.histogram_chart(values, "P&L", [(-9.0, "ES", charts.ACCENT)])
    assert fig.data[0].x[0] < -9.0
    assert _note(fig) is None or "below" not in _note(fig).text


def test_histogram_labels_of_close_marker_lines_go_to_opposite_sides():
    values = np.random.default_rng(5).standard_normal(20_000)
    fig = charts.histogram_chart(values, "P&L", [(-2.3, "VaR", charts.LOSS), (-2.6, "ES", charts.ACCENT)])
    sides = {a.text: a.xanchor for a in fig.layout.annotations if a.name != "outside"}
    assert sides == {"ES": "right", "VaR": "left"}  # ES text ends at its line, VaR text starts at its line
    single = charts.histogram_chart(values, "Price", [(0.0, "strike", charts.NEUTRAL)])
    assert [a.xanchor for a in single.layout.annotations if a.name != "outside"] == ["center"]


def test_pnl_heatmap_cells_have_no_negative_zero():
    fig = charts.pnl_heatmap(["a", "b"], ["x", "y"], [[-0.2, 0.3], [-150.0, 200.0]], "spot", "vol")
    assert np.asarray(fig.data[0].text).tolist() == [["0", "0"], ["-150", "200"]]
    small = charts.pnl_heatmap(["a"], ["x"], [[-0.004]], "spot", "vol")
    assert np.asarray(small.data[0].text).tolist() == [["0.00"]]


@pytest.mark.parametrize(
    ("value", "bound", "text"),
    [
        (-4.414, 5.0, "-4.41"),
        (-44.14, 50.0, "-44.1"),
        (-4410.4, 5000.0, "-4,410"),
        (-21571.5, 25000.0, "-21.6k"),
        (-215715.2, 250000.0, "-216k"),
        (0.2, 250000.0, "0"),
        (-1234567.0, 2e6, "-1.23M"),
    ],
)
def test_pnl_cell_text_stays_short(value, bound, text):
    # Long cell values made Plotly shrink the heatmap text to about 6 px on a phone.
    assert charts.pnl_cell_text(value, bound) == text


def test_spot_date_heatmap_gives_the_colour_scale_width_to_its_cells():
    at = _app()
    assert _chart(at, "date_heatmap")["data"][0]["showscale"] is False  # the cell values already say it
    assert _chart(at, "risk_heatmap")["data"][0].get("showscale", True) is True


@pytest.mark.parametrize(("x", "label"), [(-0.125, "-12.5%"), (0.0, "0%"), (0.3, "+30%"), (-1 / 12, "-8.3%")])
def test_shock_label(x, label):
    assert tab_risk.shock_label(x) == label


def test_price_labels_tell_close_prices_apart():
    labels = tab_risk.price_labels(0.65 * np.exp(np.linspace(-0.02, 0.02, 13)))
    assert len(set(labels)) == 13
    assert tab_risk.price_labels([67.03, 100.0, 149.18]) == ["67.03", "100.00", "149.18"]
    # Beyond six decimals (prices over 27 orders of magnitude at 500% volatility over ten years, or a 0.01 spot at
    # 0.1% volatility over a day) the labels switch to significant digits.
    wide = tab_risk.price_labels(100 * np.exp(np.linspace(-31.6, 31.6, 13)))
    assert len(set(wide)) == 13 and max(map(len, wide)) <= 9
    narrow = tab_risk.price_labels(0.01 * np.exp(np.linspace(-1.05e-4, 1.05e-4, 13)))
    assert len(set(narrow)) == 13


def test_payoff_range_keeps_the_strikes_readable():
    # Review finding: at 80% volatility the axis ran to 1,800 with the strikes squashed into its first 6%.
    lo, hi = tab_strategy.payoff_range(100, [100, 110], 105.13, 0.8)
    assert 0 < lo < 100 and 110 < hi <= 330
    assert (100 - lo) / (hi - lo) > 0.2
    lo, hi = tab_strategy.payoff_range(100, [99.5, 100.5], 100.0, 0.005)
    assert lo < 99.5 and hi > 100.5 and hi - lo < 5
