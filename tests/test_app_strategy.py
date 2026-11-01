"""Strategy tab: app behaviour (streamlit.testing) and its display helpers; the Risk tab as the place that analyses
the Strategy tab's position (its own tests are in tests/test_app_risk.py).

The Strategy tab in the desk layout (UI spec 3.2; roadmap BK-2, BK-3, RK-9a, UI-4, UI-8, UI-9, UI-13): its acceptance
tests run on the synthetic index at the pinned valuation time (``_apptest.ASOF``), the others on the textbook market
the tests pin by default (S = K = 100, one year, σ 20%, r 5%, q 0). The legs without Streamlit are tested in
``tests/test_strategy_legs.py``.
"""

import base64
import dataclasses
import json
import re
import time
import urllib.parse
from datetime import date, datetime

import numpy as np
import pytest
from _apptest import ASOF, TEXTBOOK, make_app
from scipy.stats import lognorm, norm
from streamlit.proto.WidgetStates_pb2 import WidgetState
from streamlit.testing.v1 import AppTest

from options_analyzer import black_scholes as bs
from options_analyzer import clock, risk
from options_analyzer import strategies as strat
from options_analyzer.ui import charts, surface_state, tab_risk, tab_strategy
from options_analyzer.ui import format as fmt
from options_analyzer.ui import legs as lg
from options_analyzer.ui import market as mk
from options_analyzer.ui.app import TABS
from options_analyzer.ui.format import MINUS, md, money
from options_analyzer.vol_surface.published import ONE_SD, TWO_SD, Expiry

NO_EDITS = {"edited_rows": {}, "added_rows": [], "deleted_rows": []}
#: The Risk tab's default implied-vol dynamics in the VaR simulation, an index's (VIX-like).
INDEX_VOL = {"vol_of_vol": 1.0, "spot_vol_correlation": -0.7}
#: The synthetic index at the pinned valuation time: the app's default market.
INDEX = {"preset": None, "asof": ASOF}
DEC = Expiry(date(2026, 12, 18), "AM")
#: The default condor on the index (spec 3.2, default state).
CONDOR_LEGS = [("put", 1, 5300.0), ("put", -1, 5400.0), ("call", -1, 6150.0), ("call", 1, 6250.0)]


def _ok(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def _app(tab: str = "Strategy", **query) -> AppTest:
    """The app opened at ``?tab=<tab>`` and the other query parameters (only the open tab runs)."""
    at = make_app(tab=tab, **query)
    at.run()
    _ok(at)
    return at


def _index(tab: str = "Strategy", **query) -> AppTest:
    """The app on the synthetic index at the pinned valuation time."""
    return _app(tab, **INDEX, **query)


def _edit(at: AppTest) -> AppTest:
    """Switch the blotter to Edit, where the legs editor is."""
    at.segmented_control(key="st_blotter").set_value("Edit").run()
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
    """The element id of the legs editor on the page, or None when it is not there (another tab is open, or the
    blotter is in View)."""
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
    page with the same identity. When a run leaves it out (another tab is open, the blotter is in View) or gives it a
    new identity, the edits are gone: every run of the app, also one that opens another tab, is watched for that.
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
    """The Risk tab's description of the position it analyses: its position line, which goes on with the legs' expiry
    and the maximum loss (tests/test_app_risk.py)."""
    return next(m.value for m in at.markdown if "· max loss" in m.value)


def _position(at: AppTest) -> str:
    """The position the Risk tab analyses. It opens the Risk tab, which builds the position from the Strategy tab's
    session state (tab_strategy.build_position), and goes back to the Strategy tab, as a user who checks the risk of
    the legs they just edited does."""
    text = _analysed(_open(at, "Risk"))
    _open(at, "Strategy")
    return text


def _metric(at: AppTest, label: str):
    return next(m for m in at.metric if m.label == label)


def _tiles(at: AppTest) -> dict[str, tuple[str, str]]:
    """The Strategy tiles: label to (value, delta slot)."""
    labels = ("Net premium", "Package", "Max profit", "Max loss", "Breakevens", "P(profit)")
    return {m.label: (m.value, m.proto.delta) for m in at.metric if m.label in labels}


def _chart(at: AppTest, key: str) -> dict:
    return next(json.loads(c.proto.spec) for c in at.get("plotly_chart") if c.proto.id.endswith(key))


def _trace(spec: dict, name: str) -> dict:
    return next(t for t in spec["data"] if (t.get("name") or "").startswith(name))


def _array(value) -> np.ndarray:
    """A trace's data array; Plotly sends numeric NumPy arrays as base64 ``bdata``."""
    if isinstance(value, dict) and "bdata" in value:
        array = np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"])
        shape = value.get("shape")
        if shape is not None:
            array = array.reshape([int(n) for n in str(shape).split(",")] if isinstance(shape, str) else shape)
        return array
    return np.asarray(value)


def _blotter(at: AppTest):
    """The blotter (View): its data frame element."""
    return next(d for d in at.dataframe if d.proto.id.endswith("tbl_blotter"))


def _legs(at: AppTest) -> list[tuple]:
    """The blotter's legs as (type, quantity, strike, expiry), the subtotal and Total rows left out."""
    frame = _blotter(at).value
    rows = frame[frame["Type"].notna()]
    return [
        (t.lower(), int(q), float(k), e)
        for t, q, k, e in rows[["Type", "Qty", "Strike", "Expiry"]].itertuples(index=False)
    ]


def _caption(at: AppTest, text: str) -> str:
    return next(c.value for c in at.caption if text in c.value)


@pytest.fixture(scope="module")
def published():
    """The published surface the index preset shows at the pinned valuation time."""
    return surface_state.publish(mk.index_market(datetime.fromisoformat(ASOF)))


def _quote_value(published, legs, units: float = 100.0) -> float:
    """Σ q·mid × units of ``legs`` (kind, quantity, strike) on 18-Dec-26 AM: their value at the chain's mids."""
    return units * sum(q * published.quote(DEC, k, kind).mid for kind, q, k in legs)


# --- The index: the spec's default state and its acceptance tests (UI spec 3.2) ------------------------------------


def test_the_default_position_is_the_16_delta_condor_with_100_point_wings():
    at = _index()
    assert at.selectbox(key="strategy").value == "Range-bound · Iron Condor"
    assert at.selectbox(key="strategy_expiry").value == DEC
    assert (at.selectbox(key="strategy_short_delta").value, at.selectbox(key="strategy_wing_pts").value) == (16, 100)
    assert _legs(at) == [(kind, q, k, "18‑Dec‑26 AM") for kind, q, k in CONDOR_LEGS]
    # The defaults stay out of the link (a bound control at its default writes nothing).
    assert at.query_params == {"tab": "Strategy", "asof": ASOF}


def test_the_default_tiles(published):
    # Spec 3.2's goldens on the generator before SF-4 were $2,240 credit, 20.80 / 24.00, −$7,760, 5,378 / 6,172 and
    # 73.5%. On today's fit the code gives the numbers below, each checked here from the chain's quotes (mids, bids
    # and asks); P(profit) against an independent CDF in tests/test_strategy_legs.py.
    at = _index()
    credit = -_quote_value(published, CONDOR_LEGS)
    assert credit == pytest.approx(2235.0, abs=1e-6)
    tiles = _tiles(at)
    assert tiles["Net premium"] == ("$2,235", "credit")
    assert tiles["Package"] == ("20.80 / 23.90", "cr · theo 22.48")
    assert tiles["Max profit"] == ("$2,235", "")
    assert tiles["Max loss"] == (f"{MINUS}$7,765", "")  # 100 points less the credit, × 100: −$7,765
    low, high = 5400 - credit / 100, 6150 + credit / 100  # 5,377.65 and 6,172.35
    assert tiles["Breakevens"] == ("5,378 / 6,172", f"{MINUS}7.3% / +6.4%")
    assert f"{low:,.2f} ({MINUS}7.3%), {high:,.2f} (+6.4%)" in _metric(at, "Breakevens").proto.help
    value, basis = tiles["P(profit)"]
    assert basis == "RN smile" and abs(float(value.rstrip("%")) - 73.5) <= 1.0 and value == "74.0%"
    # The tiles' help states the exact amounts and the theo: entry at mid is −$12.77 against theo (spec: −$24).
    help_text = _metric(at, "Net premium").proto.help
    assert "22.35 per share at mid" in help_text and "\\$2,235.00 for the position" in help_text
    assert f"entry at mid is {MINUS}\\$12.77 against theo" in help_text
    assert "mid vs fit −0.04 vp in the package vega" in _metric(at, "Package").proto.help
    assert "\\$155.00 per side" in _metric(at, "Package").proto.help  # half the 3.10 spread, × 100


def test_the_blotter_has_a_row_per_leg_and_a_total(published):
    at = _index()
    blotter = _blotter(at)
    frame = blotter.value
    assert len(frame) == len(CONDOR_LEGS) + 1 and frame["On"].tolist()[-1] == "Total"
    quantities = frame["Qty"].dropna().tolist()
    assert all(float(q).is_integer() for q in quantities) and quantities == [1, -1, -1, 1]
    config = json.loads(blotter.proto.columns)
    assert config["Qty"]["type_config"]["format"] == "%+d"  # signed whole numbers, never 1.00
    assert list(frame.columns) == [
        "On", "Qty", "Type", "Expiry", "Strike", "Bid", "Ask", "Mid", "IV", "IV src", "Mid vs fit", "$Δ", "Γ P&L 1%",
        "Vega", "Θ", "Decay",
    ]  # fmt: skip
    assert frame["IV src"].tolist()[:4] == ["surface"] * 4
    # Bid, ask and mid are the chain's; IV the surface's at each strike.
    for row, (kind, _, k) in enumerate(CONDOR_LEGS):
        quote = published.quote(DEC, k, kind)
        assert (frame["Bid"][row], frame["Ask"][row], frame["Mid"][row]) == (quote.bid, quote.ask, round(quote.mid, 2))
        assert frame["IV"][row] == round(published.vol(k, DEC).vol * 100, 2)
    # The Total's $Δ is the position's that the Risk tab analyses (position_risk of build_position), to the dollar:
    # −$18,871 (the spec's −$19,525 was the generator before SF-4); Γ P&L 1% −$69, vega −$306 per vp, Θ +$41 a day.
    position = _built(**INDEX)
    total = strat.position_risk(position.strategy, position.market, multiplier=position.units)
    assert frame["$Δ"].tolist()[-1] == round(total.dollar_delta) == -18871
    assert (frame["Γ P&L 1%"].tolist()[-1], frame["Vega"].tolist()[-1], frame["Θ"].tolist()[-1]) == (-69, -306, 41)
    assert config["Decay"]["label"] == "1d decay ($)"  # Milestone 1's one-day decay, beside theta
    # The footer names the engine, the dynamics, the entry, the probability's basis and expired legs.
    assert _caption(at, "BSM European legs") == (
        "BSM European legs at surface IVs · sticky strike · entry at mid · P(profit) RN from the smile at 18‑Dec · "
        "expired legs out of every number"
    )


def test_a_leg_switched_off_leaves_every_number_and_the_chart():
    at = _edit(_index())
    lines = len(_strike_lines(at))
    editor = LegsEditor(at)
    at = editor.edit({"0": {"On": False}})  # the long 5,300 put, bought at its mid of 25.80
    assert _tiles(at)["Net premium"] == ("$4,815", "credit")  # 2,235 + 2,580: exactly that leg's premium
    assert len(_strike_lines(at)) == lines - 1 == 3
    at.segmented_control(key="st_blotter").set_value("View").run()
    assert _blotter(at).value["On"].tolist()[:4] == ["off", "on", "on", "on"]


def _strike_lines(at: AppTest) -> list[float]:
    """The chart's dotted strike lines."""
    shapes = _chart(at, "payoff_chart")["layout"]["shapes"]
    return [s["x0"] for s in shapes if s.get("type") == "line" and s.get("line", {}).get("dash") == "dot"]


def test_the_structure_steppers():
    at = _index()
    at.button(key="step_shift_up").click().run()
    assert [leg[2] for leg in _legs(at)] == [5350.0, 5450.0, 6200.0, 6300.0]
    at.button(key="step_shift_down").click().run()
    at.button(key="step_width_out").click().run()
    assert [leg[2] for leg in _legs(at)] == [5250.0, 5400.0, 6150.0, 6300.0]  # the body stays
    for _ in range(10):
        at.button(key="step_width_in").click().run()
    assert [leg[2] for leg in _legs(at)] == [5350.0, 5400.0, 6150.0, 6200.0]  # never crossed
    at.button(key="step_roll_next").click().run()
    assert {leg[3] for leg in _legs(at)} == {"15‑Jan‑27 AM"}
    at.button(key="step_flip").click().run()
    assert [leg[1] for leg in _legs(at)] == [-1, 1, 1, -1]
    _ok(at)
    # Moved legs are kept as they are, with a note, until reset.
    assert _kept(at)
    _edit(at).button(key="reset_legs").click().run()
    at.segmented_control(key="st_blotter").set_value("View").run()
    assert not _kept(at)
    assert _legs(at) == [(kind, q, k, "18‑Dec‑26 AM") for kind, q, k in CONDOR_LEGS]


def test_the_row_actions_apply_to_the_selected_leg():
    at = _index()
    at.selectbox(key="blotter_row").set_value(2).run()  # −1 6,150C
    at.button(key="row_duplicate").click().run()
    assert [leg[2] for leg in _legs(at)] == [5300.0, 5400.0, 6150.0, 6150.0, 6250.0]
    at.selectbox(key="blotter_row").set_value(3).run()
    at.button(key="row_flip").click().run()
    assert [leg[1] for leg in _legs(at)] == [1, -1, -1, 1, 1]
    at.button(key="row_remove").click().run()
    assert [leg[1] for leg in _legs(at)] == [1, -1, -1, 1]
    _ok(at)


def test_the_what_if_vol_shift_labels_the_t0_curve():
    at = _index(wv=3)
    assert any("SCENARIO" in m.value for m in at.markdown)
    assert _trace(_chart(at, "payoff_chart"), "T+0")["name"] == "T+0 · vol +3.00 vp"
    # The tiles are today's market, which no what-if shift moves.
    assert _tiles(at) == _tiles(_index())
    at.button(key="whatif_reset").click().run()
    _ok(at)
    assert [at.number_input(key=k).value for k in ("ws", "wv", "wr")] == [0.0, 0.0, 0] and at.slider(
        key="wd"
    ).value == 0
    assert not any("SCENARIO" in m.value for m in at.markdown)
    assert _trace(_chart(at, "payoff_chart"), "T+0")["name"] == "T+0"
    assert not {"ws", "wv", "wd", "wr"} & set(at.query_params)


def test_a_scenario_link_reproduces(published):
    query = {"ws": -2, "wv": 3, "wd": 5}
    at = _index(**query)
    assert at.slider(key="wd").label.startswith("T+5 · Fri 30‑Oct‑26")
    spec = _chart(at, "payoff_chart")
    t5 = _trace(spec, "T+5")
    assert t5["name"] == "T+5 · 30‑Oct · vol +3.00 vp"
    # The scenario point: spot 5,684 on the T+5 curve, every leg at its surface vol + 3 vp, 7 calendar days later.
    scenario = _trace(spec, "Scenario")
    as_of = datetime.fromisoformat(ASOF)
    tau = clock.time_to_expiry(as_of, DEC.day, style="AM") - clock.year_fraction(as_of, datetime(2026, 10, 30, 16))
    market = lg.price_index(lg.index_template("Iron Condor", published, DEC), published, as_of).market
    value = sum(
        q * (bs.price(5684.0, k, tau, market.rate, published.vol(k, DEC).vol + 0.03, market.dividend_yield, kind)
             - published.quote(DEC, k, kind).mid)
        for kind, q, k in CONDOR_LEGS
    ) * 100  # fmt: skip
    assert scenario["y"] == [pytest.approx(value, abs=1e-6)] and scenario["x"] == [5684.0]
    again = _index(**query)
    assert _chart(again, "payoff_chart")["data"] == spec["data"]


def test_the_view_and_the_scenario_leave_the_link_while_another_tab_is_open():
    # The share note says the link holds the Strategy view and the what-if shifts while the Strategy tab is open: their
    # widgets are drawn only there, and Streamlit drops the URL parameter of a widget it does not draw. The session
    # keeps them, and opening the tab again puts them back in the link.
    from options_analyzer.ui import header

    held = {"st_view", "st_units", "wv"}
    at = _index(st_view="Table", st_units="% of max loss", wv=3)
    assert held <= set(at.query_params)
    _open(at, "Pricing")
    assert not held & set(at.query_params)
    assert "while the Strategy tab is open, its view" in header.SHARE_NOTE
    _open(at, "Strategy")
    assert held <= set(at.query_params)
    assert at.segmented_control(key="st_view").value == "Table" and at.number_input(key="wv").value == 3.0


def test_the_table_view(published):
    # UI-9: the date x spot table. The cell at today's spot and date is the T+0 P&L there (−$12.77, theo less mid;
    # the spec's −$24 was the generator before SF-4), and the expiry column is the expiry curve.
    at = _index()
    at.segmented_control(key="st_view").set_value("Table").run()
    _ok(at)
    heat = _chart(at, "pnl_table")["data"][0]
    spots = np.array([float(y.split(" ")[0].replace(",", "")) for y in heat["y"]])
    z = _array(heat["z"])
    # Weekly on the last session of each week, then the last five sessions daily: every fifth session drifted to
    # Mondays after Thanksgiving (Mon 30-Nov, Mon 07-Dec, then Fri 11-Dec four sessions later).
    assert [x.replace("<br>", " ") for x in heat["x"]] == [
        "Fri 23‑Oct", "Fri 30‑Oct", "Fri 06‑Nov", "Fri 13‑Nov", "Fri 20‑Nov", "Fri 27‑Nov", "Fri 04‑Dec",
        "Fri 11‑Dec", "Mon 14‑Dec", "Tue 15‑Dec", "Wed 16‑Dec", "Thu 17‑Dec", "Exp 18‑Dec",
    ]  # fmt: skip
    row = int(np.flatnonzero(spots == 5800.0)[0])
    as_of = datetime.fromisoformat(ASOF)
    market = lg.price_index(lg.index_template("Iron Condor", published, DEC), published, as_of).market
    T = market.expiry
    theo = sum(
        q * bs.price(5800.0, k, T, market.rate, published.vol(k, DEC).vol, market.dividend_yield, kind)
        for kind, q, k in CONDOR_LEGS
    )
    assert z[row, 0] == pytest.approx(100 * theo - _quote_value(published, CONDOR_LEGS), abs=0.01)
    assert z[row, 0] == pytest.approx(-12.77, abs=0.01)
    payoff = sum(q * np.maximum(spots - k if kind == "call" else k - spots, 0.0) for kind, q, k in CONDOR_LEGS)
    assert z[:, -1] == pytest.approx(100 * payoff - _quote_value(published, CONDOR_LEGS), abs=1e-6)
    at.segmented_control(key="st_units").set_value("% of max loss").run()
    z_pct = _array(_chart(at, "pnl_table")["data"][0]["z"])
    assert z_pct[:, -1] == pytest.approx(z[:, -1] / 7765.0 * 100, rel=1e-9)
    assert z_pct.min() == pytest.approx(-100.0, abs=1e-9)  # the worst cell at expiry is the max loss


def test_the_chart_draws_the_rn_ranges_the_straddle_move_and_the_reference_lines(published):
    at = _index()
    spec = _chart(at, "payoff_chart")
    names = [t.get("name") for t in spec["data"]]
    lo68, hi68 = published.rn_range(DEC, ONE_SD)
    lo95, hi95 = published.rn_range(DEC, TWO_SD)
    # UI-13: the legend names the basis; the bounds are the smile's quantiles at 15.87% and 84.13% (5,491 and 6,162;
    # the spec's 5,492 and 6,167 were the fit before SF-4).
    assert names[:2] == [
        f"RN 68% to 18‑Dec: {lo68:,.0f}–{hi68:,.0f} (smile)",
        f"RN 95% to 18‑Dec: {lo95:,.0f}–{hi95:,.0f} (smile)",
    ]
    assert names[0] == "RN 68% to 18‑Dec: 5,491–6,162 (smile)"
    assert published.cdf(lo68, DEC) == pytest.approx((1 - ONE_SD) / 2, abs=1e-12)
    rects = [s for s in spec["layout"]["shapes"] if s["type"] == "rect"]
    assert [(r["x0"], r["x1"]) for r in rects] == [(lo68, hi68), (lo95, hi95)]
    # The straddle-implied move: the 5,830 straddle at mid over spot, two ticks on the x-axis, named in the legend
    # (a label beside the ticks overprinted the curves' end labels on a phone).
    move = 279.65 / 5800
    texts = [a["text"] for a in spec["layout"]["annotations"]]
    assert "straddle ±4.82%" in names and "straddle ±4.82%" not in texts
    ticks = [s["x0"] for s in spec["layout"]["shapes"] if s.get("line", {}).get("width") == 2]
    assert ticks == [pytest.approx(5800 * (1 - move)), pytest.approx(5800 * (1 + move))]
    # The expiry, T+0 and two dated curves (⅓ and ⅔ of the 38 sessions to expiry), each named at its right end.
    assert names[4:8] == ["Expiry 18‑Dec", "T+0", "T+13 · 11‑Nov", "T+26 · 01‑Dec"]
    assert {"exp 18‑Dec", "T+0", "T+13 · 11‑Nov", "T+26 · 01‑Dec"} <= set(texts)
    # Strikes 100 points apart in a range of about 2,200 share a label, which a phone's 230 px plot needs.
    assert {"K 5,300/5,400", "K 6,150/6,250", "S 5,800", " F"} <= set(texts)
    assert sorted(_strike_lines(at)) == [5300.0, 5400.0, 6150.0, 6250.0]
    assert _trace(spec, "Breakeven")["text"] == [f"BE 5,378 ({MINUS}7.3%)", "BE 6,172 (+6.4%)"]
    # The breakeven labels: two lines, outward, on the surface colour (strike lines ran through them).
    labels = {a["text"]: a for a in spec["layout"]["annotations"] if a["text"].startswith("BE ")}
    assert set(labels) == {f"BE 5,378<br>{MINUS}7.3%", "BE 6,172<br>+6.4%"}
    assert [labels[t]["xanchor"] for t in sorted(labels)] == ["right", "left"]
    assert all("bgcolor" in a for a in labels.values())
    # The note: the same days as the builder's expiry menu, on the third line under S and F.
    [note] = [a for a in spec["layout"]["annotations"] if a["text"] == "18‑Dec‑26 AM · 55d · RN smile"]
    assert note["yshift"] == -28
    menu = at.selectbox(key="strategy_expiry")
    assert menu.format_func(menu.value) == "18‑Dec‑26 AM · 55d"  # 55.73 days, floored as the what-if strip does


def test_the_range_control_and_the_greek_strip():
    at = _index()
    at.segmented_control(key="st_range").set_value("±5%").run()
    assert _chart(at, "payoff_chart")["layout"]["xaxis"]["range"] == [pytest.approx(5510.0), pytest.approx(6090.0)]
    at.segmented_control(key="st_range").set_value("RN 95%").run()
    rect = [s for s in _chart(at, "payoff_chart")["layout"]["shapes"] if s["type"] == "rect"][1]
    assert _chart(at, "payoff_chart")["layout"]["xaxis"]["range"] == [rect["x0"], rect["x1"]]
    at.segmented_control(key="st_greek").set_value("Θ").run()
    _ok(at)
    spec = _chart(at, "payoff_chart")
    strip = _trace(spec, "Θ ($/day)")
    spots, theta = _array(strip["x"]), _array(strip["y"])
    position = _built(**INDEX)
    i = int(np.argmin(np.abs(spots - 5800.0)))
    at_spot = strat.position_risk(position.strategy, replace_spot(position.market, spots[i]), multiplier=100).theta
    assert theta[i] == pytest.approx(at_spot, rel=1e-9)
    assert spec["layout"]["yaxis2"]["title"]["text"] == "Θ ($/day)"


def replace_spot(market: strat.Market, spot: float) -> strat.Market:
    return dataclasses.replace(market, spot=float(spot))


def test_a_calendar_reads_its_maximum_at_the_front_expiry_with_subtotals_per_expiry():
    at = _index(strategy="Long Call Calendar")
    loss, profit = _metric(at, "Max loss"), _metric(at, "Max profit")
    assert profit.proto.delta == "front expiry"
    assert "front expiry" in loss.proto.help and "their losses can exceed it" in loss.proto.help
    frame = _blotter(at).value
    assert frame["On"].tolist() == ["on", "on", "Σ 18‑Dec‑26", "Σ 19‑Feb‑27", "Total"]
    assert frame["$Δ"].tolist()[-1] == frame["$Δ"].tolist()[2] + frame["$Δ"].tolist()[3]
    # A dividend yield makes the long back call gain less than $1 per $1 rise: the loss far above is unbounded, but
    # only past about 3.4 x spot. The tile reads the worst P&L within the RN 99.9% range to the front expiry, the debit
    # (the back call is worth nothing at its low end), with that basis in its delta slot; the caption keeps the rest.
    as_of = datetime.fromisoformat(ASOF)
    published = surface_state.publish(mk.index_market(as_of))
    lo, hi = published.rn_range(DEC, 0.999)
    assert loss.value == f"{MINUS}$8,500" and loss.proto.delta == "RN 99.9% range"
    assert f"{lo:,.0f}–{hi:,.0f}" in loss.proto.help and "grows without bound" in loss.proto.help
    assert "the loss is unbounded" in _caption(at, "long back call")
    # The table's % of max loss divides by that loss: the worst cell at expiry is −100%, not a column of blanks.
    at.segmented_control(key="st_view").set_value("Table").run()
    at.segmented_control(key="st_units").set_value("% of max loss").run()
    _ok(at)
    z = _array(_chart(at, "pnl_table")["data"][0]["z"])
    assert np.isfinite(z).all() and -100.0 <= z[:, -1].min() < -99.0  # the 95% range's low end: -99.74%


def test_a_diagonals_breakevens_are_read_over_the_max_loss_range():
    # The index diagonal (+1 5,400C 19-Feb-27 / −1 6,150C 18-Dec-26) turns to a loss again only where the long back
    # call's shortfall of 1 − e^(−qτ) per $1 rise has eaten the spread's width: 124,510, +2,046.7%, which the tile
    # listed beside 5,851. The tile reads its breakevens over the Max loss tile's range and names the other in its
    # help. The far root is checked here as a root of the expiry P&L, from the closed form at the back leg's own vol.
    at = _index(strategy="Diagonal (Poor Man's Covered Call)")
    loss, be = _metric(at, "Max loss"), _metric(at, "Breakevens")
    lo, hi = (float(x.replace(",", "")) for x in re.search(r"to it, ([\d,]+)–([\d,]+)", loss.proto.help).groups())
    shown = [float(x.replace(",", "")) for x in be.value.split(" / ")]
    assert shown and all(lo <= b <= hi for b in shown)
    assert len(shown) == 1 and be.proto.delta == "+0.9%"
    far = float(re.search(r"beyond it: ([\d,.]+) \(", be.proto.help).group(1).replace(",", ""))
    assert far > hi and "Read within the RN 99.9% range" in be.proto.help
    context = mk.index_market(datetime.fromisoformat(ASOF))
    published = surface_state.publish(context)
    back = Expiry(date(2027, 2, 19), "AM")
    # Every leg at the front expiry's parity carry (legs.price_index); far in the money the vol does not matter.
    r, q = mk.carry(published.forward(DEC), published.discount_factor(DEC), context.spot, published.maturity(DEC))
    t_back = published.maturity(back) - published.maturity(DEC)
    entry = 510.65  # the debit per share at the chain's mids (the Net premium tile)
    assert _metric(at, "Net premium").value == "$51,065"

    def expiry_pnl(s):
        long_call = bs.price(s, 5400.0, t_back, r, published.vol(5400.0, back).vol, q)
        return long_call - max(s - 6150.0, 0.0) - entry

    assert expiry_pnl(far * 0.999) > 0 > expiry_pnl(far * 1.001)


def test_leg_edits_survive_a_visit_to_the_risk_tab_on_the_index():
    at = _edit(_index())
    editor = LegsEditor(at)
    at = editor.edit({"1": {"IV (%)": 25.0}, "3": {"Strike": 6300.0}})
    _open(at, "Risk")
    # The legs share one expiry, which the line names once with its time left (tab_strategy.Position.describe).
    assert "+1\u00a06,300C · 1 × 100 · 18‑Dec‑26 AM (55d) ·" in _analysed(at)
    _open(at, "Strategy")
    at.segmented_control(key="st_blotter").set_value("View").run()
    frame = _blotter(at).value
    assert frame["Strike"].tolist()[3] == 6300.0 and frame["IV src"].tolist()[1] == "override"
    assert frame["IV"].tolist()[1] == 25.0


def test_index_legs_keep_their_dates_when_the_valuation_time_moves(published):
    # Appendix A item 14: legs hold a listed date and settlement and are converted to years with
    # clock.time_to_expiry on each run. Edited legs are kept, so a later valuation time shortens them.
    at = _edit(_index())
    LegsEditor(at).edit({"0": {"Qty": 2}})
    later = datetime(2026, 10, 30, 16, 0)
    at.sidebar.datetime_input(key="asof").set_value(later).run()
    _ok(at)
    _open(at, "Risk")
    assert "+2\u00a05,300P /" in _analysed(at) and "· 18‑Dec‑26 AM (48d) ·" in _analysed(at)  # 55.7 days less a week
    expected = clock.time_to_expiry(later, DEC.day, style="AM")
    assert expected == pytest.approx(published.maturity(DEC) - 7 / 365, abs=1e-12)  # a week of calendar time
    built = _built(preset=None, asof="2026-10-30T16:00")
    assert {leg.expiry for leg in built.strategy.legs} == {clock.time_to_expiry(later, DEC.day, style="AM")}


def test_the_empty_and_invalid_states():
    at = _edit(_index())
    editor = LegsEditor(at)
    at = editor.edit({str(i): {"On": False} for i in range(4)})
    assert all(value == "–" for value, _ in _tiles(at).values())
    assert any("No legs are on" in i.value for i in at.info)
    # A zero quantity: one line under the editor naming the row; the other legs are analysed.
    editor = LegsEditor(_edit(_index()))
    editor.id = _editor_id(editor.at)
    editor.state = {**NO_EDITS, "edited_rows": {"0": {"Qty": 0}}}
    states = editor.at._tree.get_widget_states()
    states.widgets.append(WidgetState(id=editor.id, string_value=json.dumps(editor.state)))
    at = editor.at._run(states)
    assert [e.value for e in at.error] == [
        "Row 1: the quantity is zero; type a signed whole number, or switch the leg off."
    ]
    assert _tiles(at)["Net premium"] == ("$4,815", "credit")


def test_a_stock_template_under_the_index_is_replaced_with_a_note():
    at = _index(strategy="Covered Call")
    assert at.selectbox(key="strategy").value == "Range-bound · Iron Condor"
    assert [w.value for w in at.warning] == [
        "This link asks for Covered Call, which holds stock: SPX itself cannot be traded, so the Iron Condor is shown."
    ]
    assert "Range-bound · Covered Call" not in at.selectbox(key="strategy").options


def test_the_builder_sets_the_shorts_wings_and_expiry_and_the_link_keeps_them():
    at = _index()
    at.selectbox(key="strategy_short_delta").set_value(25)
    at.selectbox(key="strategy_wing_pts").set_value(50)
    at.selectbox(key="strategy_expiry").set_value(Expiry(date(2026, 11, 20), "AM")).run()
    _ok(at)
    assert [leg[3] for leg in _legs(at)] == ["20‑Nov‑26 AM"] * 4
    # The expiry menu writes its date to the link by hand; the browser writes the bound menus' options as shown,
    # plain numbers, so a link reopens the same legs.
    assert at.query_params["strategy_expiry"] == "2026-11-20 AM"
    assert at.selectbox(key="strategy_short_delta").proto.query_param_key == "strategy_short_delta"
    again = _index(strategy_expiry="2026-11-20 AM", strategy_short_delta=25, strategy_wing_pts=50)
    assert _legs(again) == _legs(at)
    # Strikes by width: ATMF ± n × the width, snapped to listed strikes.
    at.selectbox(key="strategy_strikes_by").set_value("Width").run()
    at.number_input(key="strategy_width_pct").set_value(2.0).run()
    _ok(at)
    assert len({leg[2] for leg in _legs(at)}) == 4


def test_risk_neutral_off_reads_probability_under_the_view():
    at = _index()
    at.toggle(key="risk_neutral").set_value(False).run()
    _ok(at)
    assert _tiles(at)["P(profit)"][1] == "view"
    assert "P(profit) view (your drift, flat ATMF vol) at 18‑Dec" in _caption(at, "BSM European legs")


def test_a_what_if_step_reruns_quickly():
    # Spec 2.12 budgets a what-if step on Strategy at 0.5 s; on a laptop the server run takes about 0.22 s. A loose
    # guard against a tenfold slowdown, not a benchmark (shared runners are 2 to 3 times slower).
    at = _index()
    start = time.perf_counter()
    at.number_input(key="wv").set_value(1.0).run()
    assert time.perf_counter() - start < 5.0
    _ok(at)


def test_tab_captions_are_short():
    # Principle 10: every caption in the tab body is at most 140 characters.
    at = _index(strategy="Long Call Calendar")
    body = [c.value for c in at.main.caption if not c.value.startswith(("The link holds", "Add this"))]
    assert body and max(len(c) for c in body) <= 140, [c for c in body if len(c) > 140]


# --- build_position: the Risk tab's view of the position without the controls ------------------------------------

#: A script that builds the position without drawing the Strategy tab's controls, as another page or a report would.
WITHOUT_CONTROLS = """
import streamlit as st
from options_analyzer.ui import inputs, tab_strategy
st.session_state["position"] = tab_strategy.build_position(inputs.sidebar())
"""


def _built(**query) -> tab_strategy.Position:
    """The position build_position makes from a link, without the controls."""
    page = AppTest.from_string(WITHOUT_CONTROLS)
    page.query_params.update({k: str(v) for k, v in query.items() if v is not None})
    page.run()
    assert not page.exception, [e.value for e in page.exception]
    assert not page.main.children  # nothing drawn in the main area
    return page.session_state["position"]


CONDOR_LINK = {"strategy_category": "Range-bound", "strategy_Range-bound": "Iron Condor"}


@pytest.mark.parametrize(
    "query",
    [
        {},
        {**CONDOR_LINK, "contracts": "2", "multiplier": "50", "commission": "0.65", "strategy_width_pct": "5"},
        {"strategy": "Custom", "contracts": "2.5"},
        {"strategy_category": "Time spread", "strike": "120", "days": "200"},
        # Values the controls would not take: each is replaced by the control's default.
        {"strategy": "Range-bound · Bull Call Spread", "contracts": "0"},
        {"strategy_category": "range-bound", "multiplier": "x", "commission": "-1", "strategy_width_pct": "50"},
        {"strategy_width_pct": "nan", "contracts": "1e9", "commission": "inf"},
        # The index: the template's expiry, strikes by delta or width, and an expiry no longer listed.
        {**INDEX},
        {**INDEX, "strategy": "Short Strangle", "strategy_short_delta": "25", "strategy_expiry": "2026-11-20 AM"},
        {**INDEX, "strategy": "iron butterfly", "strategy_wing_pts": "150", "contracts": "3"},
        {**INDEX, "strategy_strikes_by": "Width", "strategy_width_pct": "2", "strategy_expiry": "2026-11-19 AM"},
        {**INDEX, "strategy_category": "Time spread", "strategy_short_delta": "17", "strategy_wing_pts": "7"},
        # 25-point wings, no longer offered (they landed on the short where SPX lists every 50): the default wing.
        {**INDEX, "strategy_wing_pts": "25"},
    ],
    ids=[
        "defaults", "condor", "custom", "calendar", "out of range", "invalid", "not finite", "index", "index strangle",
        "index butterfly", "index width", "index legacy", "index old wing",
    ],
)  # fmt: skip
def test_build_position_reads_a_link_as_the_controls_do(monkeypatch, query):
    # Without controls() in the run, build_position reads the controls' URL parameters itself; the position must be
    # the one the app builds (with the controls drawn) for the same link.
    analysed = []
    real = tab_risk.render
    monkeypatch.setattr(
        tab_risk, "render", lambda position, inputs: analysed.append(position) or real(position, inputs)
    )
    query = TEXTBOOK | query
    _app("Risk", **query)
    built = _built(**query)
    assert built == analysed[-1]
    assert analysed[-1] is not None


# --- Leg edits are never silently lost (textbook market) -----------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "value"), [("vol", 25.0), ("vol", 30.0), ("days", 200), ("spot", 105.0), ("rate", 3.0)]
)
def test_leg_edits_survive_sidebar_changes(key, value):
    editor = LegsEditor(_edit(_app()))
    editor.edit({"0": {"Qty": 3}})
    assert "Long 3 × 100 Call" in _position(editor.at)
    editor.at.sidebar.number_input(key=key).set_value(value)
    at = editor.run()
    assert "**Bull Call Spread (modified)** — Long 3 × 100 Call, Short 1 × 110 Call" in _position(at)


def _kept(at: AppTest) -> bool:
    return any(c.value.startswith("Edited legs: kept") for c in at.caption)


def test_edited_template_is_kept_on_a_strike_change_until_reset():
    editor = LegsEditor(_edit(_app()))
    editor.edit({"0": {"Qty": 3}})
    editor.at.sidebar.number_input(key="strike").set_value(95.0)
    at = editor.run()
    assert "Long 3 × 100 Call, Short 1 × 110 Call" in _position(at)
    assert _kept(at)
    at.button(key="reset_legs").click()
    at = editor.run()
    assert "**Bull Call Spread** — Long 1 × 95 Call, Short 1 × 105 Call" in _position(at)
    assert not _kept(at)


def test_unedited_template_follows_the_sidebar_strike():
    at = _app()
    at.sidebar.number_input(key="strike").set_value(95.0).run()
    _ok(at)
    assert "**Bull Call Spread** — Long 1 × 95 Call, Short 1 × 105 Call" in _position(at)


def test_custom_position_survives_volatility_and_strike_changes():
    at = _app()
    at.selectbox(key="strategy").set_value("Custom").run()
    editor = LegsEditor(_edit(at))
    editor.edit(added_rows=[{"Type": "put", "Qty": -1, "Strike": 90}, {"Type": "stock", "Qty": 1}])
    built = "**Custom** — Long 1 × 100 Call, Short 1 × 90 Put, Long 1 share"
    assert built in _position(editor.at)
    for key, value in (("vol", 30.0), ("strike", 101.0)):
        editor.at.sidebar.number_input(key=key).set_value(value)
        assert built in _position(editor.run())


def test_edits_survive_switching_to_another_template_and_back():
    editor = LegsEditor(_edit(_app()))
    editor.edit({"0": {"Qty": 3}})
    editor.at.selectbox(key="strategy").set_value("Vertical spread · Bear Call Spread")
    assert "**Bear Call Spread** —" in _position(editor.run())
    editor.at.selectbox(key="strategy").set_value("Vertical spread · Bull Call Spread")
    assert "Long 3 × 100 Call, Short 1 × 110 Call" in _position(editor.run())


def test_blank_strike_means_the_sidebar_strike():
    editor = LegsEditor(_edit(_app()))
    at = editor.edit(added_rows=[{"Type": "put", "Qty": 1}])
    assert "Long 1 × 100 Put" in _position(at)
    # Like a blank expiry or IV, a blank strike follows the sidebar.
    editor.at.sidebar.number_input(key="strike").set_value(90.0)
    assert "Long 1 × 100 Call, Short 1 × 110 Call, Long 1 × 90 Put" in _position(editor.run())


def test_reverted_edit_on_kept_legs_is_not_moved_by_a_later_rerun():
    # Round-1 verifier: after a strike change kept the edited legs, editing them back to the old template values
    # left them on the old strikes, and the next rerun from any widget silently moved them to the new template.
    editor = LegsEditor(_edit(_app()))
    editor.edit({"0": {"Qty": 3}})
    editor.at.sidebar.number_input(key="strike").set_value(95.0)
    editor.run()
    editor.state = {**NO_EDITS, "edited_rows": {"0": {"Qty": 1}}}
    at = editor.run()
    kept = "**Bull Call Spread (modified)** — Long 1 × 100 Call, Short 1 × 110 Call"
    assert kept in _position(at)
    assert _kept(at)
    at.number_input(key="contracts").set_value(2)
    at = editor.run()
    assert f"{kept} · 2 × 100 multiplier" in _position(at)
    # Once the legs match the template on screen again, they follow it.
    at.sidebar.number_input(key="strike").set_value(100.0)
    at = editor.run()
    assert "**Bull Call Spread** — Long 1 × 100 Call, Short 1 × 110 Call" in _position(at)
    assert not _kept(at)
    at.sidebar.number_input(key="strike").set_value(90.0)
    assert "**Bull Call Spread** — Long 1 × 90 Call, Short 1 × 100 Call" in _position(editor.run())


def test_an_edit_that_arrives_with_a_strike_change_is_kept():
    # The browser sends both in one rerun when the strike changes while the edit's own rerun is still running.
    editor = LegsEditor(_edit(_app()))
    editor.at.sidebar.number_input(key="strike").set_value(95.0)
    at = editor.edit({"0": {"Qty": 3}})
    assert "**Bull Call Spread (modified)** — Long 3 × 100 Call, Short 1 × 110 Call" in _position(at)
    assert _kept(at)


def test_editor_identity_does_not_depend_on_sidebar_values():
    # Root cause of the lost edits: the column help quoted the sidebar values and so changed the editor's identity.
    at = _edit(_app(strategy_width_pct=10))  # a fixed width keeps the template itself unchanged
    before = _editor_id(at)
    for key, value in (("vol", 30.0), ("days", 90), ("spot", 120.0)):
        at.sidebar.number_input(key=key).set_value(value).run()
        assert _editor_id(at) == before


# --- The Risk tab's position while the Strategy tab is closed (tab_strategy.build_position) ----------------------


def _legs_value(units: float, legs, spot: float, vol: float) -> float:
    """Black-Scholes-Merton value ($) of one-year option ``legs`` ``(kind, quantity, strike)`` at the default rate."""
    return units * sum(q * bs.price(spot, k, 1.0, 0.05, vol, 0.0, kind) for kind, q, k in legs)


def _assert_heatmap_of(at: AppTest, units: float, legs) -> None:
    """The Risk tab's ladder (its default report) is the P&L of ``legs``, priced here leg by leg, to the rounding of
    its P&L column (cents below $1,000, else whole dollars)."""
    ladder = next(d.value for d in at.dataframe if "Move" in d.value.columns)
    today = _legs_value(units, legs, 100, 0.2)
    places = 2 if ladder["P&L"].abs().max() < 1000 else 0
    for spot, pnl in zip(ladder["Spot"], ladder["P&L"], strict=True):
        assert pnl == pytest.approx(_legs_value(units, legs, spot, 0.2) - today, abs=0.51 * 10**-places), spot


def test_a_link_to_the_risk_tab_analyses_the_linked_template():
    # A cold deep link: the Strategy tab never opens in this session, so its legs editor never runs and the Risk tab
    # builds the position from the link's parameters (here the two menus of a version 3.0 link).
    at = _app("Risk", contracts=2, commission=0.65, **CONDOR_LINK)
    assert _analysed(at).startswith(
        "**Iron Condor** — Long 1 × 80 Put, Short 1 × 90 Put, Short 1 × 110 Call, Long 1 × 120 Call · 2 × 100 "
        "multiplier · "
    )
    assert _editor_id(at) is None and _chart(at, "ladder_chart")
    condor = [("put", 1, 80), ("put", -1, 90), ("call", -1, 110), ("call", 1, 120)]
    _assert_heatmap_of(at, 200, condor)
    strategy = strat.Strategy("Iron Condor", strat.build_template("Iron Condor", 100, 10, 1.0).legs, fees=4 * 0.0065)
    # The VaR with the default index implied-vol dynamics, and beside it the spot-only VaR (implied vol frozen).
    at.pills(key="report").set_value("VaR").run()
    frozen = risk.VaRConfig(confidence=0.99, horizon_days=1, n_paths=100_000, seed=7)
    for config, var_label in (
        (dataclasses.replace(frozen, **INDEX_VOL), "VaR 99% 1D · MC"),
        (frozen, "Spot-only VaR"),
    ):
        var = risk.value_at_risk(strategy, strat.Market(100, 0.05, 0.2, 1.0), config, 200)
        # The VaR cards are to the cent while every amount on them is below $1,000 (about $300 here).
        assert var.var < 900 and _metric(at, var_label).value == tab_risk.exact(-var.var)
    # The link now holds the menu's one parameter.
    assert at.query_params["strategy"] == "Range-bound · Iron Condor" and "strategy_category" not in at.query_params


CONDOR = "**Iron Condor** — Long 1 × 80 Put, Short 1 × 90 Put, Short 1 × 110 Call, Long 1 × 120 Call"


def test_the_short_link_to_a_template_analyses_it_on_the_risk_tab():
    # The link of the acceptance criterion, exactly: AppTest sends its query parameters URL-encoded, as a browser does.
    query = {"tab": "Risk", "strategy": "Iron Condor"}
    assert urllib.parse.urlencode(query) == "tab=Risk&strategy=Iron+Condor"
    at = _app(**query)
    assert _analysed(at).startswith(f"{CONDOR} · 1 × 100 multiplier · ")
    assert _editor_id(at) is None and _chart(at, "ladder_chart")
    _assert_heatmap_of(at, 100, [("put", 1, 80), ("put", -1, 90), ("call", -1, 110), ("call", 1, 120)])
    # The link now holds the parameter the Strategy tab's menu is bound to, so a copy of it reopens the same view.
    assert at.query_params == {"tab": "Risk", "strategy": "Range-bound · Iron Condor"} | TEXTBOOK
    assert not at.warning
    _open(at, "Strategy")
    assert at.selectbox(key="strategy").value == "Range-bound · Iron Condor"


@pytest.mark.parametrize(
    ("name", "query", "analysed"),
    [
        ("iron-condor", {}, f"{CONDOR} · 1 × 100 multiplier"),
        ("JADE LIZARD", {"contracts": 3}, "**Jade Lizard** — Short 1 × 90 Put, Short 1 × 110 Call, Long 1 × 120 Call"),
        ("custom", {}, "**Custom** — Long 1 × 100 Call · 1 × 100 multiplier"),
        ("Range-bound · Iron Condor", {}, CONDOR),  # the menu's own label
        # The short form wins over the two-parameter form of earlier links.
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
    assert "strategy_category" not in at.query_params and "strategy_Vertical spread" not in at.query_params
    assert not at.warning


def test_the_two_menus_of_an_earlier_link_still_open_their_template():
    at = _app("Risk", strategy_category="Volatility", strategy_Volatility="Short Strangle")
    assert "**Short Strangle** —" in _analysed(at)
    assert at.query_params == {"tab": "Risk", "strategy": "Volatility · Short Strangle"} | TEXTBOOK
    # A category alone opens its first template.
    assert "**Long Straddle** —" in _analysed(_app("Risk", strategy_category="Volatility"))


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
    assert _analysed(at).startswith("**Bull Call Spread** — Long 1 × 100 Call, Short 1 × 110 Call · 1 × 100 multiplier")
    assert at.query_params == {"tab": "Risk"} | TEXTBOOK
    assert [w.value for w in at.warning] == notes
    at.number_input(key="contracts").set_value(2).run()  # the note is shown once
    assert not at.warning


def test_leg_edits_survive_a_visit_to_the_risk_tab():
    editor = LegsEditor(_edit(_app()))
    at = editor.edit({"1": {"Strike": 115.0}}, added_rows=[{"Type": "put", "Qty": -1, "Strike": 90}])
    cards = {m.label: m.value for m in at.metric}
    _open(at, "Risk")
    edited = "**Bull Call Spread (modified)** — Long 1 × 100 Call, Short 1 × 115 Call, Short 1 × 90 Put"
    assert _analysed(at).startswith(f"{edited} · 1 × 100 multiplier · ")
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
    at = _edit(_app())
    edit = {**NO_EDITS, "edited_rows": {"0": {"Qty": 3}}}
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
    assert shown["Qty"].tolist() == [3.0, -1.0] and shown["Strike"].tolist() == [100.0, 110.0]
    # Net premium: 3 x 100 call - 1 x 110 call at their Black-Scholes-Merton values (one year, 5 %, 20 %), for 100
    # shares; the tile shows whole dollars from $1,000 and the exact amount in its help.
    call = {strike: bs.price(100.0, strike, 1.0, 0.05, 0.2, 0.0, "call") for strike in (100.0, 110.0)}
    premium = _metric(at, "Net premium")
    assert premium.value == f"${100 * (3 * call[100.0] - call[110.0]):,.0f}" == "$2,531"
    assert f"${100 * (3 * call[100.0] - call[110.0]):,.2f}" in premium.help
    assert modified in _position(at)


@pytest.mark.parametrize(
    "edits",
    [
        {"edited_rows": {"0": {"Qty": 2, "Strike": None}, "3": {"IV (%)": 25.0, "Entry": 1.5}}},
        {"deleted_rows": [1, 2]},
        {"added_rows": [{"Type": "stock", "Qty": 1}, {"Type": "put", "Strike": 95.5, "Days": 30}]},
        {
            "edited_rows": {"2": {"Type": "put"}},
            "deleted_rows": [0, 3],
            "added_rows": [{"Type": "call", "Qty": -2}],
        },
    ],
    ids=["cells", "deletions", "additions", "all"],
)
# Streamlit's data editor deletes rows with DataFrame.drop(inplace=True), which pandas 3.1 deprecates.
@pytest.mark.filterwarnings("ignore:The inplace keyword in DataFrame.drop is deprecated")
def test_edits_are_applied_as_the_legs_editor_applies_them(edits):
    # tab_strategy._apply_edits stands in for the editor when an edit arrives with a switch to another tab: it must
    # give the legs that st.data_editor itself returns for the same edits.
    editor = LegsEditor(_edit(_app(strategy="Iron Condor")))
    editor.edit(edits.get("edited_rows"), edits.get("added_rows", ()), edits.get("deleted_rows", ()))
    legs = editor.at.session_state["strategy_legs"]["Textbook::Range-bound · Iron Condor"]
    assert not tab_strategy._same(legs["edited"], legs["seed"])
    assert tab_strategy._same(tab_strategy._apply_edits(legs["seed"], {**NO_EDITS, **edits}), legs["edited"])


def _condor_frame():
    return lg.textbook_template("Iron Condor", 100.0, 10.0, 1.0)


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
        (lambda f: f.astype({"Qty": int, "Days": "Int64", "Type": "string"}), True),
        (lambda f: f.set_axis([5, 6, 7, 8]), True),  # the editor's index does not matter
        (lambda f: f[["On", "Type", "Qty", "Strike"]], True),  # missing columns are blank, as Days, IV and Entry
        (lambda f: _changed(Type=(1, None)), False),
        (lambda f: _changed(Strike=(0, np.nan)), False),
        (lambda f: _changed(Entry=(2, 1.5)), False),
        (lambda f: _changed(On=(2, False)), False),
        (lambda f: f.iloc[:3], False),
    ],
    ids=[
        "object",
        "other dtypes",
        "index",
        "missing blank columns",
        "blank type",
        "blank strike",
        "entry",
        "off",
        "row",
    ],
)
def test_legs_compare_equal_whatever_dtypes_the_editor_returns(other, same):
    # tab_strategy._same decides whether legs follow the template; it runs on every run of the app.
    frame = _condor_frame()
    assert tab_strategy._same(frame, other(frame)) is same


def test_blank_cells_and_signed_zeros_compare_equal():
    assert tab_strategy._same(_changed(Type=(1, None)), _changed(Type=(1, np.nan)))
    assert tab_strategy._same(_changed(Qty=(0, -0.0)), _changed(Qty=(0, 0.0)))


# --- Strike width and defaults (textbook market) -----------------------------------------------------------------


def test_strike_width_follows_the_sidebar_strike():
    at = _app(spot=8, strike=8, strategy="Bull Put Spread")
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


def test_percent_width_is_rounded_to_the_cent():
    # Round-1 verifier: 10% of a 101.37 strike gave a 111.507 call.
    at = _app(spot=101.37, strike=101.37, strategy_width_pct=10)
    assert "Long 1 × 101.37 Call, Short 1 × 111.51 Call" in _position(at)


def test_sub_cent_strikes_are_shown_in_full():
    # Round-1 verifier: a penny condor's 0.975/1.025 strikes showed as 0.97/1.02.
    at = _app(spot=1, strike=1, days=30, strategy="Iron Condor")
    assert "Long 1 × 0.95 Put, Short 1 × 0.975 Put, Short 1 × 1.025 Call, Long 1 × 1.05 Call" in _position(at)
    blotter = _blotter(at)
    assert json.loads(blotter.proto.columns)["Strike"]["type_config"]["format"] == "%,.3f"
    assert blotter.value["Strike"].tolist()[:4] == [0.95, 0.975, 1.025, 1.05]
    editor = next(n.proto for n in _nodes(_edit(at)._tree) if "legs::" in getattr(getattr(n, "proto", None), "id", ""))
    assert json.loads(editor.columns)["Strike"]["type_config"]["format"] == tab_strategy.STRIKE_FORMAT
    # The format is printf-style and the browser applies it the way Python's % operator does.
    shown = [tab_strategy.STRIKE_FORMAT % k for k in (0.95, 0.975, 1.025, 1.05, 100.0, 111.51, 1 - 2 * 0.025)]
    assert shown == ["0.95", "0.975", "1.025", "1.05", "100", "111.51", "0.95"]


def test_large_strikes_are_shown_in_full():
    # Round-2 verifier: labels used 6 significant digits, so the Risk header read 'Long 1 × 12345.7 Put' and
    # 'Long 1 × 1.23457e+06 Call', and the legs table cut 1234567.89 to 1234567.9.
    at = _app(spot=12345.67, strike=12345.67, strategy="Long Put")
    assert "Long Put** — Long 1 × 12,345.67 Put ·" in _position(at)
    at = _app(spot=1234567.89, strike=1234567.89)
    assert "Long 1 × 1,234,567.89 Call, Short 1 × 1,334,567.89 Call" in _position(at)
    blotter = _blotter(at)
    assert json.loads(blotter.proto.columns)["Strike"]["type_config"]["format"] == "%,.2f"
    assert blotter.value["Strike"].tolist()[:2] == [1234567.89, 1334567.89]


def test_the_textbook_blotter_keeps_days_and_the_flat_vol_and_shows_cents():
    # The textbook market: legs keep days to expiry and the sidebar vol, with no quote columns; its one-lot Greeks
    # are below $10 a column, so they show cents (whole dollars read a theta of $2 and a Total of 0).
    at = _app()
    blotter = _blotter(at)
    frame = blotter.value
    assert list(frame.columns) == [
        "On", "Qty", "Type", "Expiry", "Strike", "IV", "IV src", "$Δ", "Γ P&L 1%", "Vega", "Θ", "Decay",
    ]  # fmt: skip
    assert frame["Expiry"].tolist()[:2] == ["365d", "365d"] and frame["IV src"].tolist()[:2] == ["flat", "flat"]
    assert frame["IV"].tolist()[:2] == [20.0, 20.0]
    config = json.loads(blotter.proto.columns)
    assert config["Θ"]["type_config"]["format"] == "%,.2f" and config["Vega"]["type_config"]["format"] == "%,.0f"
    market = strat.Market(100, 0.05, 0.2, 1.0)
    legs = strat.build_template("Bull Call Spread", 100, 10).legs
    thetas = [strat.position_risk(strat.Strategy("l", (leg,)), market, multiplier=100).theta for leg in legs]
    assert frame["Θ"].tolist() == [round(thetas[0], 2), round(thetas[1], 2), round(sum(thetas), 2)]
    assert _tiles(at)["Package"] == ("–", "theo 4.41")
    assert _tiles(at)["P(profit)"][1] == "flat vol"


def test_short_dated_condor_is_not_degenerate_by_default():
    # Review finding: a 10%-of-strike width put a 7-day condor's short strikes 3.6 standard deviations away.
    at = _app(days=7, strategy="Iron Condor")
    pop = float(_tiles(at)["P(profit)"][0].rstrip("%")) / 100
    assert 0.2 < pop < 0.8
    at.number_input(key="strategy_width_pct").set_value(10.0).run()
    assert _tiles(at)["P(profit)"][0] == ">99.9%"


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
    at = _app(days=7, strike=60, strategy="Short Put")
    assert _tiles(at)["Max loss"][0] == f"{MINUS}$6,000"  # the strike less a premium below half a cent
    assert _tiles(at)["P(profit)"][0] == ">99.9%"
    at = _app(days=7, strike=150)
    assert _tiles(at)["Max profit"][0].startswith("$")
    assert _tiles(at)["P(profit)"][0] == "<0.1%"


def test_zero_net_premium_is_neither_a_debit_nor_a_credit():
    # Final review: an at-the-money synthetic at a zero rate costs exactly nothing (C - P = S - K e^{-rT} = 0) but its
    # card read "$0.00 credit (received)".
    at = _app(rate=0, strategy="Synthetic Long Stock")
    assert _tiles(at)["Net premium"] == ("$0.00", "no premium")


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
    assert _trace(spec, "68% at expiry")["name"].endswith("(flat vol)")
    # The chart's probability range starts 2.5 standard deviations below the median (it ends capped at 3 x 110).
    spots = _array(_trace(spec, "Expiry 365d")["x"])
    assert spots[0] == pytest.approx(price.ppf(norm.cdf(-2.5)), rel=1e-9)


@pytest.mark.parametrize("spot", [1.0, 1.5, 5.0, 100.0])
def test_breakeven_levels_agree_with_their_percentage(spot):
    # Final review: at a $1 spot the condor's breakevens read "$0.96 (-3.8%)", although $0.96 is 4.0% below the spot.
    at = _app(spot=spot, strike=spot, days=30, strategy="Iron Condor")
    value, delta = _tiles(at)["Breakevens"]
    for level, percent in zip(value.split(" / "), delta.split(" / "), strict=True):
        assert f"{float(level.replace(',', '')) / spot - 1:+.1%}".replace("-", MINUS) == percent, (value, delta)
    pairs = re.findall(r"([\d,.]+) \(([+−][\d.]+%)\)", _metric(at, "Breakevens").proto.help)
    assert len(pairs) == 2
    for level, percent in pairs:
        assert f"{float(level.replace(',', '')) / spot - 1:+.1%}".replace("-", MINUS) == percent


@pytest.mark.parametrize("spot", [1e-4, 0.01, 0.5, 0.999, 1.0, 5.0, 9.99, 10.0, 100.0, 4500.0, 1e6])
def test_breakeven_decimals_resolve_a_tenth_of_a_percent(spot):
    decimals = tab_strategy.breakeven_decimals(spot)
    assert decimals >= 2  # at least cents
    assert 10.0**-decimals <= spot / 1000 * (1 + 1e-12)  # one unit of the last decimal is at most 0.1% of the spot
    assert decimals == 2 or 10.0 ** -(decimals - 1) > spot / 1000  # and no decimal more than that needs


def test_breakevens_of_a_one_dollar_underlying_show_a_tenth_of_a_cent():
    at = _app(spot=1, strike=1, days=30, strategy="Iron Condor")
    assert _tiles(at)["Breakevens"] == ("0.962 / 1.038", f"{MINUS}3.8% / +3.8%")
    marker = _trace(_chart(at, "payoff_chart"), "Breakeven")
    assert marker["text"] == [f"BE 0.962 ({MINUS}3.8%)", "BE 1.038 (+3.8%)"]


def test_both_breakevens_are_shown_in_full():
    # Review finding: the second breakeven was cut off in a narrow metric card.
    at = _app(strategy="Iron Condor")
    # In cents on a $100 underlying, as the help and the chart write them (the tile read 85.6).
    assert _tiles(at)["Breakevens"] == ("85.58 / 114.42", f"{MINUS}14.4% / +14.4%")
    assert f"85.58 ({MINUS}14.4%), 114.42 (+14.4%)" in _metric(at, "Breakevens").proto.help


def test_large_amounts_are_shortened_with_the_exact_value_in_the_help():
    at = _app(spot=4500, strike=4500, contracts=10)
    premium = _metric(at, "Net premium")
    assert premium.value == "$215,715"
    assert "\\$215,715.16 for the position" in premium.proto.help


#: Advance widths (px) of the metric-card value font (the theme's IBM Plex Sans Medium at 22.75 px), measured in
#: Chromium and rounded up. Digits, ``$``, ``+``, ``<`` and ``>`` are equally wide (``$`` is 0.02 px narrower); ``-``,
#: ``,`` and ``.`` are narrower. The model leaves out kerning, which narrows the card strings (checked below).
CARD_GLYPHS = {
    **dict.fromkeys("0123456789$+<>", 13.650),
    "-": 9.124,
    MINUS: 13.650,
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
    assert fmt.money_short(value, signed) == text
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
            text = fmt.money_short(value, signed)
            assert _card_width(text) <= CARD_WIDTH, (value, text)
            for decimals in (2, 4):
                text = fmt.number_short(value, decimals)
                assert _card_width(text) <= CARD_WIDTH, (value, decimals, text)


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
    assert fmt.number_short(value, decimals) == text
    assert len(text) <= 9
    assert _card_width(text) <= CARD_WIDTH


@pytest.mark.parametrize("name", ["Bull Call Spread", "Long Straddle", "Short Straddle"])
def test_tiles_stay_readable_at_the_input_limits(name):
    # Round-2 verifier, at spot and strike 9,999,999, 100,000 contracts and 10,000 shares per contract (all within
    # the input limits): every dollar card was cut off with an ellipsis. The dollar tiles are shortened to at most
    # nine characters with the exact amount in their help; card_row sizes the row by its widest tile (the
    # breakevens), and wraps it (tests/test_app_components.py).
    at = _app(
        spot=9999999, strike=9999999, contracts=100000, multiplier=10000, vol=500, days=1, strategy=name
    )  # fmt: skip
    for label in ("Net premium", "Max profit", "Max loss", "P(profit)"):
        value = _metric(at, label).value
        assert len(value) <= 9 and _card_width(value) <= CARD_WIDTH, (label, value)
    for label in ("Net premium", "Max profit", "Max loss"):
        metric = _metric(at, label)
        assert metric.value in ("Unlimited", "No loss", "No profit") or re.search(
            r"\\\$[\d,]+\.\d\d", metric.proto.help
        )


def test_no_loss_and_no_profit_wording_and_no_arrows():
    # Round-3 verifier: 'Cannot lose' (178 px) was cut to 'Cannot l…' at 1,280 px; the tile says 'No loss', like
    # 'No profit', and its help keeps the full sentence.
    box = [
        {"Type": "call", "Qty": -1, "Strike": 110},
        {"Type": "put", "Qty": 1, "Strike": 110},
        {"Type": "put", "Qty": -1, "Strike": 90},
    ]
    at = _edit(_app(strategy="Custom"))
    at = LegsEditor(at).edit({"0": {"Strike": 90}}, added_rows=box)
    loss = _metric(at, "Max loss")
    assert loss.value == "No loss"
    assert "The position cannot lose at the front expiry." in loss.proto.help
    premium = _metric(at, "Net premium")
    assert premium.proto.delta == "debit"
    assert premium.proto.direction == premium.proto.MetricDirection.NONE  # no up arrow on a label
    short_box = [{**row, "Qty": -row["Qty"]} for row in box]
    at = LegsEditor(_edit(_app(strategy="Custom"))).edit({"0": {"Strike": 90, "Qty": -1}}, added_rows=short_box)
    assert _metric(at, "Max profit").value == "No profit"
    assert "The position cannot profit" in _metric(at, "Max profit").proto.help


def test_the_blotter_has_no_negative_zeros():
    # Round-1 verifier: far out-of-the-money legs showed -0.00 in the legs table. The blotter's values are rounded to
    # their column's precision before they are sent (components.table), so no cell shows a negative zero.
    at = _app(strike=150, days=7)
    frame = _blotter(at).value
    rounded = 0
    for column in ("$Δ", "Γ P&L 1%", "Vega", "Θ", "Decay"):
        values = frame[column].to_numpy(dtype=float)
        assert not np.any((values == 0) & np.signbit(values)), column
        rounded += int(np.sum(values == 0))
    assert rounded >= 5  # the case has values that round to zero


# --- One-day decay beside theta / 365 (docs/reviews/panel-v3.0.md, market maker, finding 3) --------------------

#: The panel's SPX-like market (rates and volatility in percent, as the sidebar takes them).
SPX = {"spot": 5800, "strike": 5800, "rate": 4.3, "div": 1.3}


def test_a_one_day_put_shows_its_decay_to_expiry_beside_theta():
    # The panel: a 1-day at-the-money put worth $21.56 a share shows theta -$10.66 a day and loses all of it. The
    # blotter's decay column, right of theta, is the decay to expiry then.
    at = _app(strategy="Long Put", vol=18, days=1, **SPX)
    blotter = _blotter(at)
    market = strat.Market(5800, 0.043, 0.18, 1 / 365, 0.013)
    premium = 100 * bs.price(5800, 5800, 1 / 365, 0.043, 0.18, 0.013, "put")
    assert premium == pytest.approx(2156.11, abs=0.005)
    theta = strat.position_risk(strat.Strategy("p", (strat.Leg("put", 1, 5800),)), market, multiplier=100).theta
    assert theta == pytest.approx(-1066.05, abs=0.005)
    assert blotter.value["Decay"].tolist() == [round(-premium), round(-premium)]  # the leg and the Total
    assert blotter.value["Θ"].tolist() == [round(theta), round(theta)]
    config = json.loads(blotter.proto.columns)
    assert config["Decay"]["label"] == "To expiry ($)"
    names = list(blotter.value.columns)
    assert names.index("Decay") == names.index("Θ") + 1


def test_the_panels_seven_day_condor_shows_theta_and_its_one_day_decay():
    # The panel's 7-day condor: theta / 365 +$137.50 against +$151.97 by repricing, side by side in the Total row.
    at = _app(strategy="Iron Condor", vol=16, days=7, **SPX)
    frame = _blotter(at).value
    assert (frame["Θ"].tolist()[-1], frame["Decay"].tolist()[-1]) == (138, 152)
    assert json.loads(_blotter(at).proto.columns)["Decay"]["label"] == "1d decay ($)"


def test_a_position_of_shares_alone_has_no_decay_to_expiry():
    # No option expires, whatever the sidebar's days.
    market = strat.Market(100, 0.05, 0.2, 1 / 365, 0.02)
    shares = strat.Strategy("s", (strat.Leg("stock", 100),))
    assert not tab_strategy.expires_within_a_day(shares, market)
    assert tab_strategy.expires_within_a_day(strat.Strategy("c", (strat.Leg("call", 1, 100),)), market)


# --- Descriptions and valuation notes --------------------------------------------------------------------------


def test_calendar_is_described_with_its_expiries():
    at = _app(strategy="Long Call Calendar")
    assert "Short 1 × 100 Call (365d), Long 1 × 100 Call (730d)" in _position(at)


def test_under_the_american_engine_the_note_gives_the_early_exercise_the_legs_ignore():
    # The textbook market with the sidebar's American engine (ALO): the legs stay European BSM values, and the note
    # says what early exercise adds to each leg on the sidebar contract, by the engine against the closed form
    # (independently: options_analyzer.american and black_scholes). The sidebar offers no tree since UI spec
    # Appendix B, decision 8, so this note once tested a tree that the sidebar no longer offers and never showed.
    from options_analyzer import american

    at = _app(model="alo", strategy="Long Straddle")
    assert at.sidebar.segmented_control(key="model").value == "alo"
    (note,) = [i.value for i in at.info if "Early exercise" in i.value]
    for kind in ("call", "put"):
        us = american.price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, option_type=kind, scheme="accurate")
        eu = bs.price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, kind)
        sentence = (
            f"For one 100 {kind} (the sidebar contract), early exercise adds {money(us - eu, 4)} per share with "
            f"the American engine (ALO): {money(us, 4)} American against {money(eu, 4)} European."
        )
        assert md(sentence) in note
    assert us - eu > 0.3  # the put's premium at r 5%, q 0: a real amount the legs leave out
    # Under the closed form there is nothing to say.
    assert not [i.value for i in _app(strategy="Long Straddle").info if "Early exercise" in i.value]


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


@pytest.mark.parametrize(
    ("x", "label"), [(-0.125, "−12.5%"), (0.0, "0%"), (0.3, "+30%"), (-1 / 12, "−8.3%"), (12.5, "+1,250%")]
)
def test_shock_label(x, label):
    assert tab_risk.shock_label(x) == label


@pytest.mark.parametrize(
    ("text", "shown"),
    [("$70.55", "$70.55"), ("-$4.20", "−$4.20"), ("+$1.00", "+$1.00"), ("-0.512", "−0.512"), ("–", "–")],
)
def test_the_statistics_table_writes_the_minus_sign(text, shown):
    # The table of statistics shows money() and number(), which write a hyphen-minus, with the minus sign U+2212 (the
    # delta-gamma-theta VaR is negative when even the tail is a gain; a missing one reads as a dash).
    assert tab_risk._minus(text) == shown


def test_payoff_range_keeps_the_strikes_readable():
    # Review finding: at 80% volatility the axis ran to 1,800 with the strikes squashed into its first 6%.
    lo, hi = tab_strategy.payoff_range(100, [100, 110], 105.13, 0.8)
    assert 0 < lo < 100 and 110 < hi <= 330
    assert (100 - lo) / (hi - lo) > 0.2
    lo, hi = tab_strategy.payoff_range(100, [99.5, 100.5], 100.0, 0.005)
    assert lo < 99.5 and hi > 100.5 and hi - lo < 5
