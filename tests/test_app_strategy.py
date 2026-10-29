"""Strategy and Risk tabs: app behaviour (streamlit.testing) and their display helpers."""

import base64
import dataclasses
import json
import re
import urllib.parse
from decimal import Decimal

import numpy as np
import pytest
from _apptest import make_app
from scipy.stats import lognorm, norm
from streamlit.proto.WidgetStates_pb2 import WidgetState
from streamlit.testing.v1 import AppTest

from options_analyzer import black_scholes as bs
from options_analyzer import risk, scenarios
from options_analyzer import strategies as strat
from options_analyzer.ui import cache, charts, tab_risk, tab_strategy
from options_analyzer.ui.app import TABS
from options_analyzer.ui.format import MINUS, md, money

NO_EDITS = {"edited_rows": {}, "added_rows": [], "deleted_rows": []}
#: The Risk tab's default implied-vol dynamics in the VaR simulation, an index's (VIX-like).
INDEX_VOL = {"vol_of_vol": 1.0, "spot_vol_correlation": -0.7}


def _ok(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def _app(tab: str = "Strategy", **query) -> AppTest:
    """The app opened at ``?tab=<tab>`` and the other query parameters (only the open tab runs)."""
    at = make_app(tab=tab, **query)
    at.run()
    _ok(at)
    return at


def _nodes(node):
    for child in getattr(node, "children", {}).values():
        yield child
        yield from _nodes(child)


def _open(at: AppTest, tab: str, *also: WidgetState) -> AppTest:
    """Open ``tab`` as a click on it does: the browser sends the tabs' new value with the rerun (and ``also``, other
    widget values that arrive with it) and sets the URL's ``tab`` parameter (none for the first tab)."""
    container = next(node for node in _nodes(at._tree) if getattr(node, "type", None) == "tab_container")
    states = at._tree.get_widget_states()
    states.widgets.extend([WidgetState(id=container.proto.tab_container.id, string_value=tab), *also])
    if tab == TABS[0]:
        at.query_params.pop("tab", None)
    else:
        at.query_params["tab"] = tab
    at._run(states)
    _ok(at)
    return at


def _editor_id(at: AppTest) -> str | None:
    """The element id of the legs editor on the page, or None when it is not there (another tab is open)."""
    ids = [
        node.proto.id
        for node in _nodes(at._tree)
        if type(getattr(node, "proto", None)).__name__ == "Dataframe" and "legs::" in node.proto.id
    ]
    assert len(ids) <= 1, ids
    return ids[0] if ids else None


class LegsEditor:
    """Edits the legs table the way the browser does (AppTest has no data-editor API).

    The browser keeps an editor's edits and re-sends them with every rerun for as long as the editor stays on the
    page with the same identity. When a run leaves it out (another tab is open) or gives it a new identity, the
    edits are gone: every run of the app, also one that opens another tab, is watched for that.
    """

    def __init__(self, at: AppTest):
        self.at, self.id, self.state = at, None, None
        run = at._run

        def watched(*args, **kwargs) -> AppTest:
            result = run(*args, **kwargs)
            if self.id is not None and _editor_id(result) != self.id:
                self.id = None
            return result

        at._run = watched

    def edit(self, edited_rows=None, added_rows=(), deleted_rows=()) -> AppTest:
        self.id = _editor_id(self.at)
        assert self.id is not None, "the legs editor is not on the page"
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


def _analysed(at: AppTest) -> str:
    """The Risk tab's description of the position it analyses."""
    return next(m.value for m in at.markdown if m.value.startswith("Analysing"))


def _position(at: AppTest) -> str:
    """The position the Risk tab analyses. It opens the Risk tab, which builds the position from the Strategy tab's
    session state (tab_strategy.build_position), and goes back to the Strategy tab, as a user who checks the risk of
    the legs they just edited does."""
    text = _analysed(_open(at, "Risk"))
    _open(at, "Strategy")
    return text


def _metric(at: AppTest, label: str):
    return next(m for m in at.metric if m.label == label)


def _summary_line(at: AppTest, raw: bool = False) -> str:
    """The breakevens and reward / risk line, with its no-break spaces as plain spaces unless ``raw``."""
    line = next(m.value for m in at.markdown if "Reward / risk" in m.value)
    return line if raw else line.replace("\N{NO-BREAK SPACE}", " ")


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


# --- The Risk tab's position while the Strategy tab is closed (tab_strategy.build_position) ----------------------

CONDOR_LINK = {"strategy_category": "Range-bound", "strategy_Range-bound": "Iron Condor"}


def _legs_value(units: float, legs, spot: float, vol: float) -> float:
    """Black-Scholes-Merton value ($) of one-year option ``legs`` ``(kind, quantity, strike)`` at the default rate."""
    return units * sum(q * bs.price(spot, k, 1.0, 0.05, vol, 0.0, kind) for kind, q, k in legs)


def _assert_heatmap_of(at: AppTest, units: float, legs) -> None:
    """The Risk tab's spot x IV grid is the P&L of ``legs``, priced here leg by leg."""
    z = _array(_chart(at, "risk_heatmap")["data"][0]["z"])
    today = _legs_value(units, legs, 100, 0.2)
    for (row, col), spot, vol in (((0, 0), 70, 0.1), ((-1, -1), 130, 0.4), ((0, -1), 70, 0.4), ((6, 2), 100, 0.2)):
        assert z[row, col] == pytest.approx(_legs_value(units, legs, spot, vol) - today, abs=1e-6)


def test_a_link_to_the_risk_tab_analyses_the_linked_template():
    # A cold deep link: the Strategy tab never opens in this session, so its legs editor never runs and the Risk tab
    # builds the position from the link's parameters.
    at = _app("Risk", contracts=2, commission=0.65, **CONDOR_LINK)
    assert _analysed(at) == (
        "Analysing **Iron Condor** — Long 1 × 80 Put, Short 1 × 90 Put, Short 1 × 110 Call, Long 1 × 120 Call · 2 × "
        "100 multiplier"
    )
    assert _editor_id(at) is None and _chart(at, "var_hist")
    condor = [("put", 1, 80), ("put", -1, 90), ("call", -1, 110), ("call", 1, 120)]
    _assert_heatmap_of(at, 200, condor)
    strategy = strat.Strategy("Iron Condor", strat.build_template("Iron Condor", 100, 10, 1.0).legs, fees=4 * 0.0065)
    # The VaR with the default index implied-vol dynamics, and beside it the spot-only VaR (implied vol frozen).
    frozen = risk.VaRConfig(confidence=0.99, horizon_days=1, n_paths=100_000, seed=7)
    for config, var_label, es_label in (
        (dataclasses.replace(frozen, **INDEX_VOL), "VaR 99.0%", "Expected Shortfall 99.0%"),
        (frozen, "Spot-only VaR 99.0%", "Spot-only ES 99.0%"),
    ):
        var = risk.value_at_risk(strategy, strat.Market(100, 0.05, 0.2, 1.0), config, 200)
        assert _metric(at, var_label).value == tab_strategy.money_short(var.var)
        assert _metric(at, es_label).value == tab_strategy.money_short(var.expected_shortfall)


CONDOR = "**Iron Condor** — Long 1 × 80 Put, Short 1 × 90 Put, Short 1 × 110 Call, Long 1 × 120 Call"


def test_the_short_link_to_a_template_analyses_it_on_the_risk_tab():
    # The link of the acceptance criterion, exactly: AppTest sends its query parameters URL-encoded, as a browser does.
    query = {"tab": "Risk", "strategy": "Iron Condor"}
    assert urllib.parse.urlencode(query) == "tab=Risk&strategy=Iron+Condor"
    at = _app(**query)
    assert _analysed(at) == f"Analysing {CONDOR} · 1 × 100 multiplier"
    assert _editor_id(at) is None and _chart(at, "var_hist")
    _assert_heatmap_of(at, 100, [("put", 1, 80), ("put", -1, 90), ("call", -1, 110), ("call", 1, 120)])
    # The link now holds the parameters the Strategy tab's menus are bound to, so a copy of it reopens the same view.
    assert at.query_params == {"tab": "Risk", **CONDOR_LINK}
    assert not at.warning
    _open(at, "Strategy")
    assert at.selectbox(key="strategy_category").value == "Range-bound"
    assert at.selectbox(key="strategy_Range-bound").value == "Iron Condor"


@pytest.mark.parametrize(
    ("name", "query", "analysed"),
    [
        ("iron-condor", {}, f"{CONDOR} · 1 × 100 multiplier"),
        ("JADE LIZARD", {"contracts": 3}, "**Jade Lizard** — Short 1 × 90 Put, Short 1 × 110 Call, Long 1 × 120 Call"),
        ("custom", {}, "**Custom** — Long 1 × 100 Call · 1 × 100 multiplier"),
        # The short form wins over the two-parameter form, which it replaces.
        (
            "Iron Condor",
            {"strategy_category": "Vertical spread", "strategy_Vertical spread": "Bear Put Spread"},
            CONDOR,
        ),
    ],
)
def test_the_short_link_names_a_template_without_case_or_punctuation(name, query, analysed):
    at = _app("Risk", strategy=name, **query)
    assert analysed in _analysed(at)
    assert "strategy" not in at.query_params and "strategy_Vertical spread" not in at.query_params
    assert not at.warning


@pytest.mark.parametrize(
    ("name", "notes"),
    [
        ("Iron Condr", ["This link asks for a strategy named `Iron Condr`, which is not one of the templates, so it "
                        "was ignored."]),
        ("", []),  # asks for nothing
    ],
)  # fmt: skip
def test_a_short_link_to_a_strategy_that_is_not_a_template_is_ignored_with_a_note(name, notes):
    at = _app("Risk", strategy=name)
    assert (
        _analysed(at) == "Analysing **Bull Call Spread** — Long 1 × 100 Call, Short 1 × 110 Call · 1 × 100 multiplier"
    )
    assert at.query_params == {"tab": "Risk"}
    assert [w.value for w in at.warning] == notes
    at.number_input(key="contracts").set_value(2).run()  # the note is shown once
    assert not at.warning


#: A script that builds the position without drawing the Strategy tab's controls, as another page or a report would.
WITHOUT_CONTROLS = """
import streamlit as st
from options_analyzer.ui import inputs, tab_strategy
st.session_state["position"] = tab_strategy.build_position(inputs.sidebar())
"""


@pytest.mark.parametrize(
    "query",
    [
        {},
        {**CONDOR_LINK, "contracts": "2", "multiplier": "50", "commission": "0.65", "strategy_width_pct": "5"},
        {"strategy_category": "Custom", "contracts": "2.5"},
        {"strategy_category": "Time spread", "strike": "120", "days": "200"},
        # Values the controls would not take: each is replaced by the control's default.
        {"strategy_category": "Range-bound", "strategy_Range-bound": "Bull Call Spread", "contracts": "0"},
        {"strategy_category": "range-bound", "multiplier": "x", "commission": "-1", "strategy_width_pct": "50"},
        {"strategy_width_pct": "nan", "contracts": "1e9", "commission": "inf"},
    ],
    ids=["defaults", "condor", "custom", "calendar", "out of range", "invalid", "not finite"],
)
def test_build_position_reads_a_link_as_the_controls_do(monkeypatch, query):
    # Without controls() in the run, build_position reads the controls' URL parameters itself; the position must be
    # the one the app builds (with the controls drawn) for the same link.
    analysed = []
    real = tab_risk.render
    monkeypatch.setattr(tab_risk, "render", lambda position: analysed.append(position) or real(position))
    _app("Risk", **query)
    page = AppTest.from_string(WITHOUT_CONTROLS)
    page.query_params.update(query)
    page.run()
    assert not page.exception, [e.value for e in page.exception]
    assert not page.main.children  # nothing drawn in the main area
    assert page.session_state["position"] == analysed[-1]
    assert analysed[-1] is not None


def test_leg_edits_survive_a_visit_to_the_risk_tab():
    editor = LegsEditor(_app())
    at = editor.edit({"1": {"Strike": 115.0}}, added_rows=[{"Type": "put", "Quantity": -1, "Strike": 90}])
    cards = {m.label: m.value for m in at.metric}
    _open(at, "Risk")
    edited = "**Bull Call Spread (modified)** — Long 1 × 100 Call, Short 1 × 115 Call, Short 1 × 90 Put"
    assert _analysed(at) == f"Analysing {edited} · 1 × 100 multiplier"
    _assert_heatmap_of(at, 100, [("call", 1, 100), ("call", -1, 115), ("put", -1, 90)])
    _open(at, "Strategy")
    # The browser drew a new, empty editor; it starts from the edited legs, and the analysis is that of them.
    assert editor.id is None
    shown = next(d.value for d in at.dataframe if "legs::" in d.proto.id)
    assert shown["Type"].tolist() == ["call", "call", "put"] and shown["Strike"].tolist() == [100.0, 115.0, 90.0]
    assert {m.label: m.value for m in at.metric} == cards
    assert edited in _position(at)  # and again


@pytest.mark.parametrize("strike_change", [False, True], ids=["straight back", "after a strike change"])
@pytest.mark.parametrize("tab", [tab for tab in TABS if tab != "Strategy"])
def test_an_edit_sent_with_the_click_on_another_tab_is_kept(tab, strike_change):
    # The legs editor commits a cell edit when it loses the focus, as it does when the user clicks another tab, so
    # the edit arrives with the run that opens that tab, where the editor is not drawn to apply it, and Streamlit
    # drops the editor's state at the end of that run. Every tab keeps it, not only the Risk tab that analyses it;
    # once kept, the edited legs stay as they are when the sidebar strike changes (unedited legs would follow it).
    at = _app()
    edit = {**NO_EDITS, "edited_rows": {"0": {"Quantity": 3}}}
    _open(at, tab, WidgetState(id=_editor_id(at), string_value=json.dumps(edit)))
    assert _editor_id(at) is None
    modified = "**Bull Call Spread (modified)** — Long 3 × 100 Call, Short 1 × 110 Call"
    if tab == "Risk":
        assert modified in _analysed(at)
    if strike_change:
        at.sidebar.number_input(key="strike").set_value(105.0).run()
        _ok(at)
    _open(at, "Strategy")
    shown = next(d.value for d in at.dataframe if "legs::" in d.proto.id)
    assert shown["Quantity"].tolist() == [3.0, -1.0] and shown["Strike"].tolist() == [100.0, 110.0]
    # Net premium: 3 x 100 call - 1 x 110 call at their Black-Scholes-Merton values (one year, 5 %, 20 %), for 100
    # shares; the card shows whole dollars from $1,000 and the exact amount in its help.
    call = {strike: bs.price(100.0, strike, 1.0, 0.05, 0.2, 0.0, "call") for strike in (100.0, 110.0)}
    premium = _metric(at, "Net premium")
    assert premium.value == f"${100 * (3 * call[100.0] - call[110.0]):,.0f}" == "$2,531"
    assert f"${100 * (3 * call[100.0] - call[110.0]):,.2f}" in premium.help
    assert modified in _position(at)


@pytest.mark.parametrize(
    "edits",
    [
        {"edited_rows": {"0": {"Quantity": 2, "Strike": None}, "3": {"IV (%)": 25.0, "Premium": 1.5}}},
        {"deleted_rows": [1, 2]},
        {"added_rows": [{"Type": "stock", "Quantity": 1}, {"Type": "put", "Strike": 95.5, "Days": 30}]},
        {
            "edited_rows": {"2": {"Type": "put"}},
            "deleted_rows": [0, 3],
            "added_rows": [{"Type": "call", "Quantity": -2}],
        },
    ],
    ids=["cells", "deletions", "additions", "all"],
)
# Streamlit's data editor deletes rows with DataFrame.drop(inplace=True), which pandas 3.1 deprecates.
@pytest.mark.filterwarnings("ignore:The inplace keyword in DataFrame.drop is deprecated")
def test_edits_are_applied_as_the_legs_editor_applies_them(edits):
    # tab_strategy._apply_edits stands in for the editor when an edit arrives with a switch to another tab: it must
    # give the legs that st.data_editor itself returns for the same edits.
    editor = LegsEditor(_app(**CONDOR_LINK))
    editor.edit(edits.get("edited_rows"), edits.get("added_rows", ()), edits.get("deleted_rows", ()))
    legs = editor.at.session_state["strategy_legs"]["Iron Condor"]
    assert not tab_strategy._same(legs["edited"], legs["seed"])
    assert tab_strategy._same(tab_strategy._apply_edits(legs["seed"], {**NO_EDITS, **edits}), legs["edited"])


def _condor_frame():
    return tab_strategy._template_frame(strat.build_template("Iron Condor", 100.0, 10.0, 1.0))


def _changed(**cells):
    """The condor's legs with cells changed: ``Strike=(0, 95.0)`` sets the first row's strike."""
    frame = _condor_frame()
    for column, (row, value) in cells.items():
        frame.loc[row, column] = value
    return frame


@pytest.mark.parametrize(
    ("other", "same"),
    [
        (lambda f: f.astype(object), True),
        (lambda f: f.astype({"Quantity": int, "Days": "Int64", "Type": "string"}), True),
        (lambda f: f.set_axis([5, 6, 7, 8]), True),  # the editor's index does not matter
        (lambda f: f[["Type", "Quantity", "Strike"]], True),  # missing columns are blank, as Days, IV and Premium
        (lambda f: _changed(Type=(1, None)), False),
        (lambda f: _changed(Strike=(0, np.nan)), False),
        (lambda f: _changed(Premium=(2, 1.5)), False),
        (lambda f: f.iloc[:3], False),
    ],
    ids=["object", "other dtypes", "index", "missing blank columns", "blank type", "blank strike", "premium", "row"],
)
def test_legs_compare_equal_whatever_dtypes_the_editor_returns(other, same):
    # tab_strategy._same decides whether legs follow the template; it runs on every run of the app.
    frame = _condor_frame()
    assert tab_strategy._same(frame, other(frame)) is same


def test_blank_cells_and_signed_zeros_compare_equal():
    assert tab_strategy._same(_changed(Type=(1, None)), _changed(Type=(1, np.nan)))
    assert tab_strategy._same(_changed(Quantity=(0, -0.0)), _changed(Quantity=(0, 0.0)))


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
    editor = next(n.proto for n in _nodes(at._tree) if "legs::" in getattr(getattr(n, "proto", None), "id", ""))
    assert json.loads(editor.columns)["Strike"]["type_config"]["format"] == tab_strategy.STRIKE_FORMAT
    # The format is printf-style and the browser applies it the way Python's % operator does.
    shown = [tab_strategy.STRIKE_FORMAT % k for k in (0.95, 0.975, 1.025, 1.05, 100.0, 111.51, 1 - 2 * 0.025)]
    assert shown == ["0.95", "0.975", "1.025", "1.05", "100", "111.51", "0.95"]


def _legs_table(at: AppTest):
    return next(df for df in at.dataframe if "Model value" in df.value.columns)


def test_large_strikes_are_shown_in_full():
    # Round-2 verifier: labels used 6 significant digits, so the Risk header read 'Long 1 × 12345.7 Put' and
    # 'Long 1 × 1.23457e+06 Call', and the legs table cut 1234567.89 to 1234567.9.
    at = _app(spot=12345.67, strike=12345.67, strategy_category="Single option")
    at.selectbox(key="strategy_Single option").set_value("Long Put").run()
    _ok(at)
    assert "Long Put** — Long 1 × 12,345.67 Put ·" in _position(at)
    assert "one 12,345.67 put (the sidebar contract)" in _american_note(_american(at))
    at = _app(spot=1234567.89, strike=1234567.89)
    assert "Long 1 × 1,234,567.89 Call, Short 1 × 1,334,567.89 Call" in _position(at)
    table = _legs_table(at)
    assert json.loads(table.proto.columns)["Strike"]["type_config"]["format"] == tab_strategy.STRIKE_FORMAT
    shown = [tab_strategy.STRIKE_FORMAT % k for k in table.value["Strike"]]
    assert shown == ["1234567.89", "1334567.89"]
    assert [tab_strategy.STRIKE_FORMAT % k for k in (12345.67, 9999999.99, 9999999.99 * 1.9)] == [
        "12345.67",
        "9999999.99",
        "18999999.98",
    ]


def test_index_sized_legs_table_fits_a_laptop_screen():
    # Round-2 verifier: at 1,280 px the legs table of a 100-lot SPX iron condor was 13 px wider than the page and
    # cut its last column. Entry prices equal to the model values are no longer repeated, and dollar columns of
    # $10,000 or more show whole dollars, rounded rather than truncated by the browser.
    at = _app(spot=4500, strike=4500, contracts=100, strategy_category="Range-bound")
    at.selectbox(key="strategy_Range-bound").set_value("Iron Condor").run()
    _ok(at)
    table = _legs_table(at)
    assert "Entry price" not in table.value.columns
    config = json.loads(table.proto.columns)
    for col in ("Cost", "Vega/pt"):
        values = table.value[col].to_numpy(dtype=float)
        assert np.abs(values).max() >= 1e4
        assert config[col]["type_config"] == {"type": "number", "format": "dollar", "step": 1}
        assert np.array_equal(values, np.round(values))
    assert table.value["Vega/pt"].tolist() == [49391.0, -115530.0, -176740.0, 145020.0]
    assert config["Theta/day"]["type_config"].get("step") is None  # below $10,000: cents
    assert config["Delta"]["type_config"]["format"] == "%.2f" and config["Gamma"]["type_config"]["format"] == "%.4f"
    at = LegsEditor(at).edit({"0": {"Premium": 2.5}})
    table = _legs_table(at)
    assert table.value["Entry price"].tolist()[:2] == [2.5, pytest.approx(92.55996, abs=1e-5)]


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
    assert _metric(at, "Max loss").value == "$6,000"  # whole dollars from $1,000 on a card
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


def test_tiny_reward_to_risk_is_not_shown_as_zero():
    # Final review: a 1-day covered call's $42.45 maximum profit against a $9,958 maximum loss (a ratio of 0.0043)
    # read "0.00 : 1", as if the position could not profit.
    at = _app(strategy_category="Stock + option", days=1)
    assert _metric(at, "Max profit").value == "$42.45"
    assert _metric(at, "Max loss").value == "$9,958"
    assert "**Reward / risk:** < 0.01 : 1" in _summary_line(at)
    # No-break spaces keep the ratio on one line: on a phone it wrapped after the "<".
    nbsp = "\N{NO-BREAK SPACE}"
    assert f"**Reward / risk:** <{nbsp}0.01{nbsp}:{nbsp}1" in _summary_line(at, raw=True)
    assert "**Reward / risk:** 0.03 : 1" in _summary_line(_app(strategy_category="Stock + option", days=30))


def test_zero_net_premium_is_neither_a_debit_nor_a_credit():
    # Final review: an at-the-money synthetic at a zero rate costs exactly nothing (C - P = S - K e^{-rT} = 0) but its
    # card read "$0.00 credit (received)".
    at = _app(rate=0, strategy_category="Synthetic")
    premium = _metric(at, "Net premium")
    assert premium.value == "$0.00"
    assert premium.proto.delta == "no net premium"
    # The label follows the amount as shown: a sub-cent residue is $0.00 either way.
    assert tab_strategy.premium_label(0.004) == tab_strategy.premium_label(-0.004) == "no net premium"
    assert tab_strategy.premium_label(0.005) == "debit (paid)"
    assert tab_strategy.premium_label(-0.005) == "credit (received)"


@pytest.mark.parametrize("vol", [20, 100, 200])
def test_payoff_band_is_the_one_sigma_range_of_the_price_at_expiry(vol):
    # Final review: the band was centred on the forward, not on the median of the log-normal price the probabilities
    # use, so at 200% volatility over a year its lower "-1σ" edge was the median itself.
    spec = _chart(_app(vol=vol), "payoff_chart")
    sigma = vol / 100  # one year to expiry
    price = lognorm(s=sigma, scale=100 * np.exp(0.05 - 0.5 * sigma**2))  # risk-neutral price at expiry
    band = next(shape for shape in spec["layout"]["shapes"] if shape["type"] == "rect")
    assert band["x0"] == pytest.approx(price.ppf(norm.cdf(-1.0)), rel=1e-9)
    assert band["x1"] == pytest.approx(price.ppf(norm.cdf(1.0)), rel=1e-9)
    # The chart's probability range starts 2.5 standard deviations below the median (it ends capped at 3 x 110).
    spots = _array(next(t for t in spec["data"] if t.get("name", "").startswith("At expiry"))["x"])
    assert spots[0] == pytest.approx(price.ppf(norm.cdf(-2.5)), rel=1e-9)


@pytest.mark.parametrize("spot", [1.0, 1.5, 5.0, 100.0])
def test_breakeven_dollars_agree_with_their_percentage(spot):
    # Final review: at a $1 spot the condor's breakevens read "$0.96 (-3.8%)", although $0.96 is 4.0% below the spot.
    market = strat.Market(spot, 0.05, 0.2, 30 / 365)
    condor = strat.build_template("Iron Condor", spot, tab_strategy.auto_width(spot, 0.2, 30 / 365))
    text = tab_strategy.breakeven_text(strat.analyze(condor, market).breakevens, spot)
    pairs = re.findall(r"\$([\d,.]+) \(([+-][\d.]+%)\)", text)
    assert len(pairs) == 2, text
    for dollars, percent in pairs:
        assert f"{float(dollars.replace(',', '')) / spot - 1:+.1%}" == percent, text


@pytest.mark.parametrize("spot", [1e-4, 0.01, 0.5, 0.999, 1.0, 5.0, 9.99, 10.0, 100.0, 4500.0, 1e6])
def test_breakeven_decimals_resolve_a_tenth_of_a_percent(spot):
    decimals = tab_strategy.breakeven_decimals(spot)
    assert decimals >= 2  # at least cents
    assert 10.0**-decimals <= spot / 1000 * (1 + 1e-12)  # one unit of the last decimal is at most 0.1% of the spot
    assert decimals == 2 or 10.0 ** -(decimals - 1) > spot / 1000  # and no decimal more than that needs


def test_breakevens_of_a_one_dollar_underlying_show_a_tenth_of_a_cent():
    at = _app(spot=1, strike=1, days=30, strategy_category="Range-bound")
    at.selectbox(key="strategy_Range-bound").set_value("Iron Condor").run()
    _ok(at)
    assert "Breakevens at expiry:** \\$0.962 (-3.8%) and \\$1.038 (+3.8%)" in _summary_line(at)
    marker = next(t for t in _chart(at, "payoff_chart")["data"] if t.get("name") == "Breakeven")
    assert marker["hovertemplate"].startswith("Breakeven %{x:$,.3f}")


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


#: Advance widths (px) of the metric-card value font (the theme's IBM Plex Sans Medium at 22.75 px), measured in
#: Chromium and rounded up. Digits, ``$``, ``+``, ``<`` and ``>`` are equally wide (``$`` is 0.02 px narrower); ``-``,
#: ``,`` and ``.`` are narrower. The model leaves out kerning, which narrows the card strings (checked below).
CARD_GLYPHS = {
    **dict.fromkeys("0123456789$+<>", 13.650),
    "-": 9.124,
    ",": 6.598,
    ".": 6.598,
    "%": 21.545,
    "e": 12.650,
    "M": 18.519,
    "B": 14.993,
    "C": 14.424,
    "T": 13.128,
    "–": 13.378,
    " ": 5.370,
    "N": 16.244,
    "U": 15.608,
    "a": 12.490,
    "d": 13.468,
    "f": 7.736,
    "g": 12.218,
    "i": 6.029,
    "l": 6.485,
    "m": 20.066,
    "n": 13.196,
    "o": 12.832,
    "p": 13.468,
    "r": 8.714,
    "s": 11.240,
    "t": 8.304,
}
#: Width (px) of a card's value in a row of four at 1,280 px with the sidebar open, leaving room for the main
#: area's 10 px scrollbar where the browser shows one (181 px without it); a wider value ends in "…".
CARD_WIDTH = 178.5


def _card_width(text: str) -> float:
    return sum(CARD_GLYPHS[c] for c in text)


def test_card_width_model_counts_the_sign():
    # Round-3 verifier: '+$2,737.18' was cut to '+$2,737.…' in its card while '-$2,737.18' fitted: '+' is as wide as
    # a digit and '-' much narrower, so counting characters without the sign missed it. Widths measured in Chromium
    # (Plex Sans Medium, 22.75 px, with kerning): the model is never narrower (Chromium rounds to 1/64 px), and at
    # most half a pixel wider.
    assert _card_width("+$2,737.18") - _card_width("-$2,737.18") == pytest.approx(13.650 - 9.124)
    measured = {
        "+$2,737.18": 122.375,
        "-$2,737.18": 117.859,
        "+$1.0e+15": 114.781,
        "Cannot lose": 122.953,
        "No profit": 91.188,
        "Unlimited": 101.750,
        "$441.0T": 87.953,
        ">99.9%": 82.750,
    }
    for text, width in measured.items():
        assert width - 1 / 64 <= _card_width(text) <= width + 0.5, text


@pytest.mark.parametrize(
    ("value", "signed", "text"),
    [
        (999.99, True, "+$999.99"),
        (-999.994, True, "-$999.99"),
        (999.996, False, "$1,000"),
        (2_737.18, True, "+$2,737"),
        (9_999.99, True, "+$10,000"),
        (19_847.23, False, "$19,847"),
        (-97_858.24, True, "-$97,858"),
        (999_999.4, True, "+$999,999"),
        (999_999.6, False, "$1.00M"),
        (-2_345_678.9, True, "-$2.35M"),
        (999_940_000.0, True, "+$999.9M"),
        (999_994_000.0, True, "+$1.00B"),
        (337_720_000.0, True, "+$337.7M"),  # SPX x 100,000 long straddle vega: cut at 158.5 px as '+$337.72M'
        (99_996_000.0, True, "+$100.0M"),
        (999_996_000.0, False, "$1.00B"),
        (2_157_150_000.0, True, "+$2.16B"),
        (441_049_535_530_000.0, False, "$441.0T"),
        (-999_996_000_000_000.0, False, "-$1.0e+15"),
        (1.03e15, True, "+$1.0e+15"),
        (1.5e16, False, "$1.5e+16"),
    ],
)
def test_money_short_fits_a_metric_card(value, signed, text):
    # Round-2 verifier: amounts were cut at "M", so the cards showed '$441,049,535.53M' at the input limits, and at
    # 1,280 px a vega of -$97,858.24 was already wider than its card. Round-3 verifier: a theta of '+$2,737.18' and
    # '+$1.00e+15' were still cut, as '+' is as wide as a digit. Cents stop at $1,000 and scientific notation has one
    # decimal, so every amount is at most nine characters with its sign and fits the card.
    assert tab_strategy.money_short(value, signed) == text
    assert len(text) <= 9
    assert _card_width(text) <= CARD_WIDTH


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("sign", [1, -1])
def test_every_card_amount_fits_at_laptop_width(sign, signed):
    # The widest amount of every tier (all digits are equally wide), and the boundaries between tiers.
    mantissas = (1.0, 5.0, 9.99, 9.994999, 9.995, 9.999999)
    for exponent in range(-3, 18):
        for mantissa in mantissas:
            value = sign * mantissa * 10.0**exponent
            text = tab_strategy.money_short(value, signed)
            assert _card_width(text) <= CARD_WIDTH, (value, text)
            for decimals in (2, 4):
                text = tab_strategy.number_short(value, decimals)
                assert _card_width(text) <= CARD_WIDTH, (value, decimals, text)


@pytest.mark.parametrize(
    ("value", "shown"),
    [
        (54.29505, 54.30),
        (-54.87316, -54.87),
        (0.004, 0.0),
        (999.996, 1000.0),
        (3861.945, 3862.0),
        (-2_345_678.9, -2_350_000.0),
        (337_720_000.0, 337_700_000.0),
        (2_157_150_000.0, 2_160_000_000.0),
        (1.03e15, 1.0e15),
        (float("inf"), float("inf")),
    ],
)
def test_var_delta_amounts_are_read_back_from_the_cards(value, shown):
    # The VaR delta line subtracts the amounts as the cards show them, read back from money_short's text in every
    # tier (cents, whole dollars, M, B, T, scientific). Equal to a few ulps (rel 1e-15: "2.16" x 1e9 is not exact in
    # binary), and the amount read back shows as the same text.
    assert tab_risk._as_shown(value) == pytest.approx(shown, rel=1e-15)
    assert tab_strategy.money_short(tab_risk._as_shown(value)) == tab_strategy.money_short(value)


def test_worded_card_values_fit_at_laptop_width():
    for text in ("No loss", "No profit", "Unlimited", "Too large", "–", ">99.9%", "<0.1%", "100.0%"):
        assert _card_width(text) <= CARD_WIDTH, text


@pytest.mark.parametrize(
    ("value", "decimals", "text"),
    [
        (-532.554, 2, "-532.55"),
        (-1e-12, 2, "0.00"),
        (-2.41634, 4, "-2.4163"),
        (999.99994, 4, "999.9999"),
        (1234.56789, 4, "1,234.57"),
        (-9_999.99, 2, "-9,999.99"),
        (20_000.0, 4, "20,000"),
        (187_182_738.53, 2, "187.2M"),
        (-3.2e15, 2, "-3.2e+15"),
    ],
)
def test_number_short_fits_a_metric_card(value, decimals, text):
    assert tab_strategy.number_short(value, decimals) == text
    assert len(text) <= 9
    assert _card_width(text) <= CARD_WIDTH


def test_index_sized_short_premium_cards_fit_at_laptop_width():
    # Round-3 verifier: a 100-lot SPX iron condor showed Theta / day '+$2,737.…' at 1,280 px.
    at = _app(spot=4500, strike=4500, contracts=100, strategy_category="Range-bound")
    at.selectbox(key="strategy_Range-bound").set_value("Iron Condor").run()
    _ok(at)
    theta = _metric(at, "Theta / day")
    assert theta.value == "+$2,737"
    assert "Exact: +\\$2,737.18" in theta.proto.help
    for label in ("Net premium", "Max profit", "Max loss", "Delta (shares)", "Gamma", "Theta / day", "Vega / vol pt"):
        assert _card_width(_metric(at, label).value) <= CARD_WIDTH, (label, _metric(at, label).value)


@pytest.mark.parametrize(
    ("category", "name"),
    [("Vertical spread", "Bull Call Spread"), ("Volatility", "Long Straddle"), ("Volatility", "Short Straddle")],
)
def test_cards_stay_readable_at_the_input_limits(category, name):
    # Round-2 verifier, at spot and strike 9,999,999, 100,000 contracts and 10,000 shares per contract (all within
    # the input limits): every dollar card and the share delta were cut off with an ellipsis. Round-3 verifier: the
    # straddles' theta, '-$1.03e+15' and '+$1.03e+15', were still cut.
    at = _app(
        spot=9999999,
        strike=9999999,
        contracts=100000,
        multiplier=10000,
        vol=500,
        days=1,
        strategy_category=category,
        **{f"strategy_{category}": name},
    )
    shortened = ["Net premium", "Max profit", "Max loss", "Delta (shares)", "Theta / day", "Vega / vol pt"]
    for label in [*shortened, "Gamma", "Probability of profit"]:
        value = _metric(at, label).value
        assert len(value) <= 9 and _card_width(value) <= CARD_WIDTH, (label, value)
    for label in shortened:
        value = _metric(at, label).value
        assert value in ("Unlimited", "No loss", "No profit") or "Exact:" in _metric(at, label).proto.help, label
    if name == "Short Straddle":
        assert _metric(at, "Theta / day").value == "+$1.0e+15"
        assert "Exact: +\\$1,03" in _metric(at, "Theta / day").proto.help


def test_shortened_amounts_at_the_input_limits_keep_the_exact_value_in_the_help():
    at = _app(spot=9999999, strike=9999999, contracts=100000, multiplier=10000)
    assert _metric(at, "Net premium").value == "$441.0T"
    assert "Exact: \\$441,049,535," in _metric(at, "Net premium").proto.help
    delta = _metric(at, "Delta (shares)")
    assert delta.value == "187.2M"
    assert "Exact: 187,182,738.53" in delta.proto.help


def test_no_loss_and_no_profit_wording_and_no_arrows():
    # Round-3 verifier: 'Cannot lose' (178 px) was cut to 'Cannot l…' at 1,280 px; the card now says 'No loss', like
    # 'No profit', and its help keeps the full sentence.
    box = [
        {"Type": "call", "Quantity": -1, "Strike": 110},
        {"Type": "put", "Quantity": 1, "Strike": 110},
        {"Type": "put", "Quantity": -1, "Strike": 90},
    ]
    at = _app(strategy_category="Custom")
    at = LegsEditor(at).edit({"0": {"Strike": 90}}, added_rows=box)
    loss = _metric(at, "Max loss")
    assert loss.value == "No loss"
    assert "The position cannot lose: it makes at least \\$" in loss.proto.help
    assert _metric(at, "Delta (shares)").value == "0.00"
    premium = _metric(at, "Net premium")
    assert premium.proto.delta == "debit (paid)"
    assert premium.proto.direction == premium.proto.MetricDirection.NONE  # no up arrow on a label
    short_box = [{**row, "Quantity": -row["Quantity"]} for row in box]
    at = LegsEditor(_app(strategy_category="Custom")).edit({"0": {"Strike": 90, "Quantity": -1}}, added_rows=short_box)
    assert _metric(at, "Max profit").value == "No profit"
    assert "No profit is possible" in _metric(at, "Max profit").proto.help


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
        if col not in table:  # the entry price is left out when every leg is entered at model value
            continue
        values = table[col].to_numpy(dtype=float)
        shown = [f"{v:.{decimals}f}" for v in values]
        assert not [s for s in shown if re.fullmatch(r"-0\.0+", s)], (col, shown)
        rounded += int(np.sum(np.abs(values) < 0.5 * 10**-decimals))
    assert rounded >= 5  # the case has values that round to zero
    assert shown_decimals == tab_strategy.LEGS_TABLE_DECIMALS


def test_whole_dollar_columns_have_no_negative_zeros():
    market = strat.Market(4500, 0.05, 0.2, 7 / 365)
    # The short put's vega is about -$0.20 for the position, which rounds to a whole-dollar -0.
    strategy = strat.Strategy("x", (strat.Leg("call", 1, 4500), strat.Leg("put", -1, 3940, premium=0.0)))
    table, decimals = tab_strategy.legs_table(strategy, market, 10_000)
    assert decimals["Cost"] == decimals["Vega/pt"] == 0 and decimals["Gamma"] == 4
    assert table["Vega/pt"].iloc[1] == 0.0 and not np.signbit(table["Vega/pt"].iloc[1])
    assert table["Entry price"].tolist()[1] == 0.0  # a fill price was entered, so the column is shown
    # Share counts shorten the same way as the Delta and Gamma cards: two decimals from 1,000, none from 10,000.
    assert (decimals["Delta"], decimals["Gamma"]) == (2, 4)
    table, decimals = tab_strategy.legs_table(strategy, market, 1_000_000)
    assert (decimals["Delta"], decimals["Gamma"], decimals["Theta/day"]) == (0, 2, 0)


def test_number_has_no_negative_zero():
    assert tab_strategy.number(-1e-12, 2) == "0.00"
    assert tab_strategy.number(-0.004, 2) == "0.00"
    assert tab_strategy.number(-0.006, 2) == "-0.01"


# --- Descriptions and valuation notes --------------------------------------------------------------------------


def test_calendar_is_described_with_its_expiries():
    at = _app(strategy_category="Time spread")
    assert "Short 1 × 100 Call (365d), Long 1 × 100 Call (730d)" in _position(at)


def _american(at: AppTest, method: str = "lr", steps: int = 501) -> AppTest:
    at.sidebar.radio(key="model").set_value("binomial").run()
    at.sidebar.radio(key="tree").set_value(method)
    at.sidebar.slider(key="steps").set_value(steps)
    at.sidebar.segmented_control(key="style").set_value("American").run()
    _ok(at)
    return at


def _american_note(at: AppTest) -> str:
    return next(i.value for i in at.info if "Early exercise is not modelled" in i.value)


def _dollars(text: str, after: str) -> float:
    """The (Markdown-escaped) dollar amount that directly precedes ``after`` in ``text``."""
    return float(re.search(r"\\\$([\d,.]+)" + re.escape(after), text).group(1).replace(",", ""))


@pytest.mark.parametrize(
    ("method", "steps", "days"), [("crr", 10, 365), ("lr", 10, 365), ("crr", 10, 30), ("lr", 501, 365)]
)
def test_american_note_measures_early_exercise_on_one_tree(method, steps, days):
    # Round-2 verifier: the note set the American tree against the closed-form European value, so the tree's
    # discretisation error read as early exercise. A no-dividend call, which is never exercised early, was "worth
    # $10.2534 with early exercise against $10.4506 European" on a 10-step tree: less than the European option.
    at = _american(_app(days=days), method, steps)
    note = _american_note(at)
    assert "one 100 call (the sidebar contract)" in note
    assert "early exercise adds \\$0.0000 per share on this tree" in note
    american, european = _dollars(note, " American"), _dollars(note, " European")
    assert american == european == float(_metric(at, "Call value").value.lstrip("$"))
    assert any("Early-exercise premium: call \\$0.0000," in c.value for c in at.caption)  # the Pricing caption
    bsm = bs.price(100, 100, days / 365, 0.05, 0.2, 0.0, "call")
    assert f"The strategy values it at \\${bsm:.4f}" in note  # the leg's Black-Scholes-Merton value


def test_american_note_reports_the_same_premium_as_the_pricing_tab():
    at = _american(_app(div=3, strategy_category="Single option"))
    at.selectbox(key="strategy_Single option").set_value("Long Put").run()
    _ok(at)
    note = _american_note(at)
    assert "one 100 put (the sidebar contract)" in note
    premium = _dollars(note, " per share on this tree")
    american, european = _dollars(note, " American"), _dollars(note, " European")
    assert american == float(_metric(at, "Put value").value.lstrip("$"))
    assert premium > 0 and premium == pytest.approx(american - european, abs=1.5e-4)
    assert any(f"put +\\${premium:.4f} (against the same tree" in c.value for c in at.caption)
    assert "The strategy values it at \\$6.7309" in note
    assert _metric(at, "Net premium").value == "$673.09"
    assert any("whatever the sidebar pricing model" in c.value for c in _open(at, "Risk").caption)


# --- Risk tab ------------------------------------------------------------------------------------------------------


def test_risk_heatmap_labels_show_the_exact_spot_shocks():
    at = _app("Risk")
    at.slider(key="grid_span").set_value(25).run()
    spec = _chart(at, "risk_heatmap")["data"][0]
    assert spec["y"][:4] == ["-25%", "-20.8%", "-16.7%", "-12.5%"]
    assert spec["y"][6] == "0%"
    assert spec["x"][2] == "IV 0%"


def test_spot_iv_heatmap_cells_stay_wide_enough_for_a_phone():
    # Final review: with the 13 spot shocks as columns next to a colour bar, the cell values were 4 px tall on a
    # 390 px phone. Like the spot x date grid, the 6 implied-vol shocks are now the columns and the spot shocks the
    # rows (highest at the top), and the colour bar is left out: the cell values already say it.
    heat = _chart(_app("Risk"), "risk_heatmap")["data"][0]
    assert heat["showscale"] is False
    assert heat["x"] == ["IV -50%", "IV -25%", "IV 0%", "IV +25%", "IV +50%", "IV +100%"]
    assert heat["y"][0] == "-30%" and heat["y"][6] == "0%" and heat["y"][-1] == "+30%"
    z = _array(heat["z"])
    assert z.shape == (13, 6)

    def bull_call_spread(spot, vol):  # the default position: 100 x (long 100 call, short 110 call), one year
        return 100 * (bs.price(spot, 100, 1, 0.05, vol) - bs.price(spot, 110, 1, 0.05, vol))

    today = bull_call_spread(100, 0.2)
    assert z[0, 0] == pytest.approx(bull_call_spread(70, 0.1) - today, abs=1e-6)  # spot -30%, IV -50%
    assert z[-1, -1] == pytest.approx(bull_call_spread(130, 0.4) - today, abs=1e-6)  # spot +30%, IV +100%
    assert z[0, -1] == pytest.approx(bull_call_spread(70, 0.4) - today, abs=1e-6)  # spot -30%, IV +100%


def test_spot_date_grid_steps_in_standard_deviations():
    at = _app("Risk")
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
    heat = _chart(_app("Risk", **query), "date_heatmap")["data"][0]
    assert len(set(heat["y"])) == len(heat["y"]) == 13


def test_time_decay_moves_scale_with_volatility():
    names = [trace["name"] for trace in _chart(_app("Risk", vol=40), "time_decay")["data"]]
    assert names == [
        "Spot -1σ (-33.0%)",
        "Spot -0.5σ (-18.1%)",
        "Spot unchanged",
        "Spot +0.5σ (+22.1%)",
        "Spot +1σ (+49.2%)",
    ]


def test_position_greek_profiles():
    at = _app("Risk")
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
    at = _app("Risk")
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
    at = _app("Risk")
    # The Fed scenario's vol shock is the Fed's own jump-off-to-peak VIX ratio, and its source line says so
    # (validator finding 3: the table showed 4.00x).
    table = next(df for df in at.dataframe if "Scenario" in df.value.columns).value.set_index("Scenario")
    assert table.loc["Fed severely adverse (2026)", "IV multiplier"] == pytest.approx(72.0 / 26.4, rel=1e-15)
    assert any("rises 26.4 → 72.0 to its Q2 2026 peak" in m.value for m in at.markdown)
    # The caption names the 1987 index and the rate the analogues shift, and says the ratio is not a level.
    sources = next(c.value for c in at.caption if c.value.startswith("Historical analogues"))
    for text in ("(the VXO for 1987)", "change in the Fed funds target", "does not set it to the index's level"):
        assert text in sources
    labels = [m.label for m in at.metric]
    assert "VaR 95% CI (simulation)" not in labels
    assert any("95% confidence intervals of the VaR estimates" in c.value for c in at.caption)
    full = next(c.value for c in at.caption if c.value.startswith("Full revaluation"))
    assert "ν ≤ 4" not in full  # Normal returns have no degrees of freedom
    assert "fatter tails" not in full
    at.selectbox(key="var_dist").set_value("student_t").run()
    full = next(c.value for c in at.caption if c.value.startswith("Full revaluation"))
    assert "ν ≤ 4" in full
    # Delta-normal VaR assumes Normal returns, so its gap to the VaR also shows the fat tails of Student-t returns,
    # and of GARCH ones beyond a day (one-day GARCH returns with Normal shocks are Normal).
    assert "vega, and the fatter tails of the return model." in full
    at.selectbox(key="var_dist").set_value("garch").run()
    full = next(c.value for c in at.caption if c.value.startswith("Full revaluation"))
    assert "fatter tails" not in full
    at.number_input(key="var_horizon").set_value(2).run()
    full = next(c.value for c in at.caption if c.value.startswith("Full revaluation"))
    assert "fatter tails of the return model" in full


def _stress_frame(at: AppTest):
    return next(df for df in at.dataframe if "Scenario" in df.value.columns)


def test_the_stress_table_marks_damped_iv_shocks_in_text():
    # Review panel (designer, finding 8): "IV damped" was a CheckboxColumn, which drew live-looking toggles in a
    # read-only table. It is text now: "yes" where a one-month (historical or Fed) shock is damped for longer-dated
    # legs, "—" where a hypothetical shock applies as it is.
    at = _app("Risk")
    frame = _stress_frame(at)
    assert json.loads(frame.proto.columns)["IV damped"]["type_config"] == {"type": "text"}
    every = (*scenarios.HISTORICAL_SCENARIOS, *scenarios.HYPOTHETICAL_SCENARIOS)
    marks = dict(zip(frame.value["Scenario"], frame.value["IV damped"], strict=True))
    assert marks == {sc.name: "yes" if sc.one_month_vol_shock else "—" for sc in every}
    assert set(marks.values()) == {"yes", "—"}
    at.toggle(key="stress_damp").set_value(False).run()
    _ok(at)
    assert set(_stress_frame(at).value["IV damped"]) == {"—"}


#: The review panel's VaR case (market maker, finding 2): a 45-day at-the-money short straddle on an index, one
#: contract of 100 shares.
SPX_STRADDLE = {
    "spot": 5800,
    "strike": 5800,
    "days": 45,
    "vol": 16,
    "rate": 4.3,
    "div": 1.3,
    "strategy": "Short Straddle",
}


def _var_configs(monkeypatch) -> list[risk.VaRConfig]:
    """The configuration of every VaR simulation from now on, in order (the cache is cleared, so none hides there)."""
    configs = []
    real = risk.value_at_risk
    monkeypatch.setattr(
        risk, "value_at_risk", lambda s, m, config, n=1.0: configs.append(config) or real(s, m, config, n)
    )
    cache.value_at_risk.clear()
    return configs


def _var_runs(monkeypatch) -> list[tuple[risk.VaRConfig, risk.VaRResult]]:
    """Like :func:`_var_configs`, with the result of each simulation."""
    runs = []
    real = risk.value_at_risk

    def spy(strategy, market, config, multiplier=1.0):
        runs.append((config, real(strategy, market, config, multiplier)))
        return runs[-1][1]

    monkeypatch.setattr(risk, "value_at_risk", spy)
    cache.value_at_risk.clear()
    return runs


def _var_cards(at: AppTest) -> list[tuple[str, str, str]]:
    """Label, value and delta line of every VaR and ES card."""
    return [(m.label, m.value, m.delta) for m in at.metric if re.search(r"VaR|ES |Shortfall", m.label)]


def test_var_simulates_index_implied_vol_dynamics_by_default(monkeypatch):
    # Review panel (market maker, finding 2): implied vol was frozen unless a toggle was switched on, so the VaR of a
    # short straddle left out its vega risk. Implied-vol moves are on by default now, with index (VIX-like)
    # dynamics, and the spot-only run beside them is the same simulation with implied vol frozen: same seed, paths,
    # confidence, horizon and return model.
    configs = _var_configs(monkeypatch)
    at = _app("Risk")
    toggle = at.toggle(key="var_volmoves")
    assert toggle.value is True and "index (VIX-like) dynamics" in toggle.help
    assert at.number_input(key="var_vov").value == 100.0 and at.slider(key="var_corr").value == -0.7
    moving = risk.VaRConfig(confidence=0.99, horizon_days=1, n_paths=100_000, seed=7, **INDEX_VOL)
    assert configs == [dataclasses.replace(moving, vol_of_vol=0.0), moving]
    full = next(c.value for c in at.caption if c.value.startswith("Full revaluation"))
    assert "implied vol moving (vol of vol 100% a year, correlation −0.70 with spot)" in full


@pytest.mark.parametrize(
    ("model", "steps"),
    [
        pytest.param(
            {"distribution": "student_t", "dof": 4.0},
            [[("selectbox", "var_dist", "student_t")], [("slider", "var_dof", 4)]],
            id="student_t",
        ),
        pytest.param(
            {"distribution": "garch", "garch_alpha": 0.05, "garch_beta": 0.93, "garch_t": True, "dof": 8.0},
            [
                [("selectbox", "var_dist", "garch")],
                [
                    ("number_input", "garch_alpha", 0.05),
                    ("number_input", "garch_beta", 0.93),
                    ("toggle", "garch_t", True),
                ],
                [("slider", "var_dof_garch", 8)],
            ],
            id="garch_t",
        ),
    ],
)
def test_the_spot_only_run_keeps_every_setting_but_the_implied_vol_moves(monkeypatch, model, steps):
    # The gap between the two VaRs is the vega risk only if the spot-only run differs from the other in the implied-vol
    # moves alone. A spot-only run at the default seed or return model would mix a seed or model difference into
    # it, which the tests at the defaults cannot see; here every setting is off its default. Each step's widgets
    # appear once the step before has run (the GARCH inputs once GARCH is chosen); the last step changes the return
    # model, so both runs are simulated again and are the last two.
    configs = _var_configs(monkeypatch)
    at = _app("Risk")
    at.selectbox(key="var_conf").set_value(0.975)
    at.number_input(key="var_horizon").set_value(5)
    at.select_slider(key="var_paths").set_value(10_000)
    at.number_input(key="var_vov").set_value(150.0)
    at.slider(key="var_corr").set_value(-0.5)
    at.number_input(key="var_seed").set_value(11)
    for step in steps:
        for kind, key, value in step:
            getattr(at, kind)(key=key).set_value(value)
        at.run()
        _ok(at)
    config = risk.VaRConfig(0.975, 5, 10_000, seed=11, vol_of_vol=1.5, spot_vol_correlation=-0.5, **model)
    assert configs[-2:] == [dataclasses.replace(config, vol_of_vol=0.0, spot_vol_correlation=-0.7), config]
    assert [label for label, *_ in _var_cards(at)] == [
        "VaR 97.5%",
        "Expected Shortfall 97.5%",
        "Spot-only VaR 97.5%",
        "Spot-only ES 97.5%",
    ]


def _shown(text: str) -> Decimal:
    """A card's amount as it reads, in exact decimal: ``$1,029`` is 1029 and ``$54.30`` is 54.30."""
    return Decimal(text.removeprefix("$").replace(",", ""))


def test_the_var_delta_line_adds_up_with_the_amounts_shown(monkeypatch):
    # Review: the delta line was the difference of the exact amounts, so on the default page it read −$0.58 under
    # cards of $54.30 and $54.87 (exact 54.295 and 54.873). It is the difference of the amounts as shown now,
    # computed here in exact decimal from the cards' text.
    runs = _var_runs(monkeypatch)
    at = _app("Risk")
    (_, spot_only), (_, moving) = runs
    cards = _var_cards(at)
    assert [value for _, value, _ in cards] == ["$54.30", "$61.77", "$54.87", "$62.91"]
    assert money(moving.var - spot_only.var) == "-$0.58"  # the exact amounts would not add up on screen
    for (_, value, delta), (_, frozen, _) in zip(cards[:2], cards[2:], strict=True):
        gap = _shown(value) - _shown(frozen)
        assert delta == md(f"{'+' if gap > 0 else MINUS}${abs(gap):,} vs spot only")
    assert [delta for *_, delta in cards[:2]] == [md("−$0.57 vs spot only"), md("−$1.14 vs spot only")]


def test_var_cards_show_the_spot_only_var_beside_the_var_with_implied_vol_moves():
    # The panel's numbers: with implied vol frozen (the old default) VaR $2,833 and ES $3,683; with index dynamics
    # $3,862 and $4,737, 27 % more. Both pairs are shown, and the headline cards say what the vol moves add.
    at = _app("Risk", **SPX_STRADDLE)
    assert _var_cards(at) == [
        ("VaR 99.0%", "$3,862", md("+$1,029 vs spot only")),
        ("Expected Shortfall 99.0%", "$4,737", md("+$1,054 vs spot only")),
        ("Spot-only VaR 99.0%", "$2,833", "implied vol frozen"),
        ("Spot-only ES 99.0%", "$3,683", "implied vol frozen"),
    ]
    # The library's numbers for the link's market, as the sidebar builds it, at the tab's defaults.
    market = strat.Market(5800, 4.3 / 100, 16 / 100, 45 / 365, 1.3 / 100)
    straddle = strat.build_template("Short Straddle", 5800, 10)
    frozen = risk.VaRConfig(0.99, 1, 100_000, seed=7)
    spot_only = risk.value_at_risk(straddle, market, frozen, 100)
    moving = risk.value_at_risk(straddle, market, dataclasses.replace(frozen, **INDEX_VOL), 100)
    shown = [moving.var, moving.expected_shortfall, spot_only.var, spot_only.expected_shortfall]
    assert [card[1] for card in _var_cards(at)] == [tab_strategy.money_short(x) for x in shown]
    # A short straddle is short vega: the vol moves add to both measures.
    assert moving.var >= spot_only.var and moving.expected_shortfall >= spot_only.expected_shortfall
    helps = {m.label: m.proto.help for m in at.metric}
    assert md(f"Exact: {money(moving.var)}") in helps["VaR 99.0%"]
    assert md(f"Exact: {money(spot_only.var)}") in helps["Spot-only VaR 99.0%"]
    assert "implied vol frozen" in helps["Spot-only VaR 99.0%"]
    # The delta-normal VaR, a card before, is in the table of statistics.
    stats = next(df.value for df in at.dataframe if "Statistic" in df.value.columns).set_index("Statistic")
    assert stats.loc["Delta-normal VaR 99.0%", "Value"] == money(moving.delta_normal_var)
    # With two VaRs on the cards, the simulation interval says which is which; delta-normal VaR leaves out vega too.
    (lo, hi), (lo_spot, hi_spot) = moving.var_ci95, spot_only.var_ci95
    uncertainty = next(c.value for c in at.caption if c.value.startswith("Simulation uncertainty"))
    assert uncertainty == md(
        f"Simulation uncertainty: the 95% confidence intervals of the VaR estimates are {money(lo)} to {money(hi)} "
        f"with implied-vol moves and {money(lo_spot)} to {money(hi_spot)} spot only (order statistics of the "
        "100,000 simulated outcomes of each run)."
    )
    full = next(c.value for c in at.caption if c.value.startswith("Full revaluation"))
    assert full.endswith(
        "with Normal returns: its gap to the full-revaluation VaR shows the position's gamma, time decay and vega."
    )


def test_implied_vol_moves_switched_off_show_the_spot_only_run_from_the_cache(monkeypatch):
    # Off, the tab shows one pair of cards, with implied vol frozen; that run is the spot-only one already simulated.
    # The spot-only run does not depend on the vol dynamics, so changing them re-runs only the other one: one
    # simulation per change, as before the spot-only run was added.
    configs = _var_configs(monkeypatch)
    at = _app("Risk", **SPX_STRADDLE)
    assert [c.vol_of_vol for c in configs] == [0.0, 1.0]
    at.toggle(key="var_volmoves").set_value(False).run()
    _ok(at)
    assert _var_cards(at) == [
        ("VaR 99.0%", "$2,833", "implied vol frozen"),
        ("Expected Shortfall 99.0%", "$3,683", "implied vol frozen"),
    ]
    assert "implied vol frozen (no vega risk)" in _metric(at, "VaR 99.0%").proto.help
    full = next(c.value for c in at.caption if c.value.startswith("Full"))
    assert "time decay and implied vol frozen" in full
    assert full.endswith("shows the position's gamma and time decay.")  # no vega in the full revaluation
    uncertainty = next(c.value for c in at.caption if c.value.startswith("Simulation uncertainty"))
    assert uncertainty.startswith("Simulation uncertainty: the 95% confidence interval of the VaR estimate is ")
    assert "spot only" not in uncertainty and uncertainty.endswith("of the 100,000 simulated outcomes).")
    at.toggle(key="var_volmoves").set_value(True).run()
    _ok(at)
    assert len(configs) == 2  # both runs came from the cache
    at.number_input(key="var_vov").set_value(150.0).run()
    at.slider(key="var_corr").set_value(-0.5).run()
    _ok(at)
    assert [(c.vol_of_vol, c.spot_vol_correlation) for c in configs[2:]] == [(1.5, -0.7), (1.5, -0.5)]
    assert [label for label, *_ in _var_cards(at)][2:] == ["Spot-only VaR 99.0%", "Spot-only ES 99.0%"]
    assert _var_cards(at)[2][1] == "$2,833"  # the spot-only VaR does not depend on the vol dynamics


# --- Chart and label helpers ---------------------------------------------------------------------------------


def _note(fig):
    return next((a for a in fig.layout.annotations if a.name == "outside"), None)


def test_var_caption_counts_legs_that_expire_on_the_horizon_day():
    # Final review: a 1-day bull call spread expires 0.69 trading days in, before a 1-day horizon, but its legs
    # settle on day 1, the horizon itself, so the caption left them out.
    at = _app("Risk", days=1)
    assert any("2 option leg(s) expire before the horizon" in c.value for c in at.caption)
    assert not any("expire before the horizon" in c.value for c in _app("Risk", days=2).caption)  # 1.38 trading days


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
    assert np.asarray(fig.data[0].text).tolist() == [["0", "0"], ["−150", "+200"]]
    small = charts.pnl_heatmap(["a"], ["x"], [[-0.004]], "spot", "vol")
    assert np.asarray(small.data[0].text).tolist() == [["0.00"]]


@pytest.mark.parametrize(
    ("value", "bound", "text"),
    [
        (-4.414, 5.0, "−4.41"),
        (-44.14, 50.0, "−44.1"),
        (-4410.4, 5000.0, "−4,410"),
        (-21571.5, 25000.0, "−21.6k"),
        (-215715.2, 250000.0, "−216k"),
        (0.2, 250000.0, "0"),
        (-1234567.0, 2e6, "−1.23M"),
        (21571.5, 25000.0, "+21.6k"),
        (4.414, 5.0, "+4.41"),
    ],
)
def test_pnl_cell_text_stays_short(value, bound, text):
    # Long cell values made Plotly shrink the heatmap text to about 6 px on a phone.
    assert charts.pnl_cell_text(value, bound) == text


def test_risk_heatmaps_give_the_colour_scale_width_to_their_cells():
    at = _app("Risk")
    assert _chart(at, "date_heatmap")["data"][0]["showscale"] is False  # the cell values already say it
    assert _chart(at, "risk_heatmap")["data"][0]["showscale"] is False


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


def test_time_decay_scenarios_use_the_ordered_ramp_not_categorical_colours():
    # The ordered spot scenarios (-1σ ... +1σ) were drawn in five unrelated categorical colours, all solid.
    traces = _chart(_app("Risk"), "time_decay")["data"]
    styles = [(trace["line"].get("color"), trace["line"].get("dash")) for trace in traces]
    assert styles == charts.scenario_styles(len(traces))
