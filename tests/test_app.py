"""End-to-end smoke tests of the Streamlit app with streamlit.testing (headless)."""

import json

import pytest
from streamlit.proto.WidgetStates_pb2 import WidgetState
from streamlit.testing.v1 import AppTest

from options_analyzer import strategies, vol_surface
from options_analyzer.ui.app import TABS


def _ok(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_default_render(app):
    _ok(app)
    labels = {m.label: m.value for m in app.metric}
    assert labels["Call value"] == "$10.4506"
    assert labels["Put value"] == "$5.5735"
    assert len(app.tabs) == 6


@pytest.mark.parametrize("model", ["binomial", "monte_carlo"])
def test_pricing_models(app, model):
    app.sidebar.radio(key="model").set_value(model).run()
    _ok(app)
    call = float(next(m.value for m in app.metric if m.label == "Call value").strip("$"))
    assert call == pytest.approx(10.4506, abs=0.02)


def test_american_binomial_with_dividends(app):
    app.sidebar.number_input(key="div").set_value(6.0).run()
    app.sidebar.radio(key="model").set_value("binomial").run()
    app.sidebar.segmented_control(key="style").set_value("American").run()
    _ok(app)


def test_invalid_crr_tree_falls_back_with_warning(app):
    app.sidebar.radio(key="model").set_value("binomial").run()
    app.sidebar.radio(key="tree").set_value("crr").run()
    app.sidebar.number_input(key="vol").set_value(0.5).run()
    app.sidebar.number_input(key="rate").set_value(30.0).run()
    app.sidebar.slider(key="steps").set_value(10).run()
    assert not app.exception
    assert any("not arbitrage-free" in w.value for w in app.warning)


@pytest.mark.parametrize("days", [1, 3650])
def test_extreme_expiries(app, days):
    app.sidebar.number_input(key="days").set_value(days).run()
    _ok(app)


def test_every_strategy_template(new_app):
    app = new_app(tab="Strategy").run()
    for category in strategies.TEMPLATE_CATEGORIES:
        app.selectbox(key="strategy_category").set_value(category).run()
        names = [t.name for t in strategies.TEMPLATES.values() if t.category == category]
        for name in names:
            app.selectbox(key=f"strategy_{category}").set_value(name).run()
            _ok(app)


def test_var_models(new_app):
    app = new_app(tab="Risk").run()
    for dist in ("student_t", "garch", "normal"):
        app.selectbox(key="var_dist").set_value(dist).run()
        _ok(app)
    assert any(m.label.startswith("VaR") for m in app.metric)


def test_volatility_surface_upload(new_app):
    app = new_app(tab="Vol surface").run()
    app.segmented_control(key="surface_source").set_value("Market quotes (CSV)").run()
    csv = vol_surface.sample_quotes().to_csv(index=False).encode()
    app.file_uploader(key="quotes_file").upload("quotes.csv", csv, "text/csv").run()
    _ok(app)
    assert any(m.label == "Calendar arbitrage" for m in app.metric)
    app.file_uploader(key="quotes_file").upload("bad.csv", b"a,b\n1,2\n", "text/csv").run()
    assert not app.exception
    assert any("Strike column" in e.value for e in app.error)


def test_implied_vol_round_trip(new_app):
    app = new_app(tab="IV").run()
    iv = next(m.value for m in app.metric if m.label == "Implied volatility")
    assert iv == "20.000%"


def _trace(at: AppTest, prefix: str) -> dict:
    for chart in at.get("plotly_chart"):
        for trace in json.loads(chart.proto.spec)["data"]:
            if (trace.get("name") or "").startswith(prefix):
                return trace
    raise AssertionError(f"no trace named {prefix!r}")


def test_iv_shift_leaves_the_expiry_curve_and_metrics_alone(new_app):
    # A calendar's back leg is alive at the front expiry; the slider must not move the at-expiry curve away
    # from the breakevens and maxima computed without the shift.
    app = new_app(tab="Strategy").run()
    app.sidebar.number_input(key="days").set_value(30).run()
    app.selectbox(key="strategy_category").set_value("Time spread").run()
    before = _trace(app, "At expiry")["y"]
    app.slider(key="pnl_iv_shift").set_value(10).run()
    _ok(app)
    assert _trace(app, "At expiry")["y"] == before
    assert _trace(app, "Today")["name"] == "Today, IV +10 pts"


#: Every chart of the app, by key; together the three pricing models draw each of them.
CHART_KEYS = {
    "sensitivity_chart", "heatmap_call", "heatmap_put", "tree_convergence_call", "tree_convergence_put",
    "exercise_boundary", "mc_paths", "mc_terminal", "payoff_chart", "risk_heatmap", "date_heatmap", "time_decay",
    "greek_profiles", "stress_chart", "var_hist", "surf_iv", "surf_lv", "smiles", "term_structure", "iv_curve",
}  # fmt: skip


def test_every_chart_has_the_desk_config_and_a_description(new_app):
    # Every chart goes through charts.show: the desk toolbar and SVG download, and an accessible name for screen
    # readers (Streamlit leaves a chart without one unnamed).
    seen = set()
    for query in ({}, {"model": "Binomial tree", "style": "American"}, {"model": "Monte Carlo"}):
        at = new_app(**query).run()
        _ok(at)
        charts = []
        for tab in TABS:  # only the open tab runs
            charts += _open(at, tab).get("plotly_chart")
        for chart in charts:
            key = chart.proto.id.rsplit("-", 1)[-1]
            config = json.loads(chart.proto.config)
            assert config["displaylogo"] is False and config["displayModeBar"] == "hover", key
            assert config["toImageButtonOptions"] == {"format": "svg", "filename": f"options-analyzer-{key}"}, key
            assert chart.proto.theme == "streamlit", key
            assert len(chart.proto.alt) >= 20 and not chart.proto.alt[0].islower(), (key, chart.proto.alt)
            seen.add(key)
    assert seen == CHART_KEYS


# --- Lazy tabs: only the open tab runs --------------------------------------------------------------------------

#: The charts each tab draws at the default URL.
TAB_CHARTS = {
    "Pricing": {"sensitivity_chart", "heatmap_call", "heatmap_put"},
    "Strategy": {"payoff_chart"},
    "Risk": {"risk_heatmap", "date_heatmap", "time_decay", "greek_profiles", "stress_chart", "var_hist"},
    "Vol surface": {"surf_iv", "surf_lv", "smiles", "term_structure"},
    "IV": {"iv_curve"},
    "Methodology": set(),
}  # fmt: skip
#: The Strategy tab's controls (at the default URL), which are drawn on every run: they are bound to the URL.
STRATEGY_CONTROLS = {
    "strategy_category", "strategy_Vertical spread", "strategy_width_pct", "contracts", "multiplier", "commission",
    "risk_neutral", "carry",
}  # fmt: skip


def _nodes(node):
    for child in getattr(node, "children", {}).values():
        yield child
        yield from _nodes(child)


def _chart_keys(at: AppTest) -> set[str]:
    return {chart.proto.id.rsplit("-", 1)[-1] for chart in at.get("plotly_chart")}


def _open(at: AppTest, tab: str) -> AppTest:
    """Open ``tab`` as a click on it does: the browser sends the tabs' new value with the rerun and sets the URL's
    ``tab`` parameter (none for the first tab)."""
    container = next(node for node in _nodes(at._tree) if getattr(node, "type", None) == "tab_container")
    states = at._tree.get_widget_states()
    states.widgets.append(WidgetState(id=container.proto.tab_container.id, string_value=tab))
    if tab == TABS[0]:
        at.query_params.pop("tab", None)
    else:
        at.query_params["tab"] = tab
    at._run(states)
    _ok(at)
    return at


def _assert_only_open(at: AppTest, tab: str) -> None:
    """``tab`` is open and drawn, and the closed tabs are empty but for the Strategy tab's controls."""
    assert [t.label for t in at.tabs] == list(TABS)
    assert _chart_keys(at) == TAB_CHARTS[tab]
    for block in at.tabs:
        nodes = list(_nodes(block))
        if block.label == tab:
            assert nodes, tab
        elif block.label == "Strategy":
            assert {n.key for n in nodes if getattr(n, "key", None)} == STRATEGY_CONTROLS
            assert {n.type for n in nodes} <= {"caption", "column", "flex_container", "number_input", "popover",
                                               "selectbox", "toggle"}  # fmt: skip
        else:
            assert not nodes, block.label


@pytest.mark.parametrize("tab", TABS)
def test_a_link_to_a_tab_runs_that_tab_only(new_app, tab):
    at = new_app(tab=tab).run()
    _ok(at)
    _assert_only_open(at, tab)
    assert at.query_params.get("tab") == (None if tab == TABS[0] else tab)  # the first tab needs no parameter


def test_switching_tabs_runs_each_tab_in_turn(app):
    # The order runs every tab once from every other: Pricing to Methodology and back.
    for tab in (*TABS[1:], *reversed(TABS[:-1])):
        _assert_only_open(_open(app, tab), tab)
    assert {m.label for m in app.metric} >= {"Call value", "Put value"}  # the price cards sit above the tabs


def _counted(calls: dict[str, int], name: str, function):
    """``function``, counting its calls in ``calls[name]``."""

    def counted(*args, **kwargs):
        calls[name] += 1
        return function(*args, **kwargs)

    return counted


def test_a_sidebar_change_recomputes_only_the_open_tab(new_app, monkeypatch):
    # Count the work of each tab in the run that follows a sidebar change: the charts it builds (every chart goes
    # through charts.show), the strategy analysis and the Monte Carlo VaR (its cache is cleared first, so that a
    # call cannot hide in it).
    from options_analyzer import risk
    from options_analyzer import strategies as strat
    from options_analyzer.ui import cache, charts

    shown, calls = [], {"analyze": 0, "value_at_risk": 0}
    for module, name in ((strat, "analyze"), (risk, "value_at_risk")):
        monkeypatch.setattr(module, name, _counted(calls, name, getattr(module, name)))
    real_show = charts.show
    monkeypatch.setattr(charts, "show", lambda fig, *, key, **kw: shown.append(key) or real_show(fig, key=key, **kw))
    for tab in TABS:
        cache.value_at_risk.clear()
        at = new_app(tab=tab).run()
        _ok(at)
        shown.clear()
        calls.update(analyze=0, value_at_risk=0)
        at.sidebar.number_input(key="spot").set_value(105.0).run()
        _ok(at)
        assert set(shown) == TAB_CHARTS[tab], tab
        assert calls == {"analyze": int(tab == "Strategy"), "value_at_risk": int(tab == "Risk")}, tab


def test_value_at_risk_is_not_simulated_while_the_pricing_tab_is_open(app, monkeypatch):
    from options_analyzer import risk
    from options_analyzer.ui import cache

    calls = []
    real = risk.value_at_risk
    monkeypatch.setattr(risk, "value_at_risk", lambda *args, **kwargs: calls.append(args) or real(*args, **kwargs))
    cache.value_at_risk.clear()
    for key, value in (("spot", 105.0), ("vol", 30.0), ("days", 90)):
        app.sidebar.number_input(key=key).set_value(value).run()
        _ok(app)
    app.selectbox(key="strategy_category").set_value("Range-bound").run()  # the Strategy controls are on every run
    _ok(app)
    assert calls == []
    _open(app, "Risk")  # the counter does see the simulation once the Risk tab opens
    assert len(calls) == 1
    assert any(chart.proto.id.endswith("-var_hist") for chart in app.get("plotly_chart"))


# --- Widget values outlive their closed tab ------------------------------------------------------------------------
# A widget that is not drawn in a run loses its value, so the widgets of the lazily run tabs keep theirs with
# persist_state; the Strategy tab's URL-bound controls are drawn on every run; the legs and the uploaded quotes are
# kept in the session by the app.


def _away_and_back(at: AppTest, tab: str, other: str = "Pricing") -> AppTest:
    _open(at, other)
    return _open(at, tab)


def test_strategy_values_survive_a_visit_to_the_risk_tab(new_app):
    at = new_app(tab="Strategy").run()
    at.number_input(key="contracts").set_value(3)
    at.number_input(key="multiplier").set_value(50)
    at.number_input(key="commission").set_value(1.25)
    at.number_input(key="strategy_width_pct").set_value(7.5)
    at.toggle(key="risk_neutral").set_value(False).run()
    at.number_input(key="drift").set_value(12.0)
    at.toggle(key="carry").set_value(True)
    at.slider(key="pnl_iv_shift").set_value(10)
    at.slider(key="pnl_date_364").set_value(90).run()
    _ok(at)
    before = {m.label: m.value for m in at.metric}
    _away_and_back(at, "Strategy", "Risk")
    assert {w.key: w.value for w in at.number_input}.items() >= {
        "contracts": 3, "multiplier": 50, "commission": 1.25, "strategy_width_pct": 7.5, "drift": 12.0,
    }.items()  # fmt: skip
    assert at.toggle(key="risk_neutral").value is False and at.toggle(key="carry").value is True
    assert at.slider(key="pnl_iv_shift").value == 10 and at.slider(key="pnl_date_364").value == 90
    assert {m.label: m.value for m in at.metric} == before
    assert before["Drift used"] == "12.00%"


def test_the_date_slider_starts_again_when_the_expiry_changes(new_app):
    # Its range follows the front expiry: a value is kept for one range only, as before the tabs ran lazily.
    at = new_app(tab="Strategy").run()
    at.slider(key="pnl_date_364").set_value(90).run()
    at.sidebar.number_input(key="days").set_value(200).run()
    assert at.slider(key="pnl_date_199").value == 100  # the default, the middle of the range
    at.sidebar.number_input(key="days").set_value(365).run()
    assert at.slider(key="pnl_date_364").value == 183
    _ok(at)


def test_risk_settings_survive_a_visit_to_another_tab(new_app):
    at = new_app(tab="Risk").run()
    at.slider(key="grid_span").set_value(40)
    at.slider(key="grid_days_365").set_value(30)
    at.select_slider(key="date_grid_reach").set_value(3.0)
    at.toggle(key="stress_hist").set_value(False)
    at.selectbox(key="var_conf").set_value(0.975)
    at.select_slider(key="var_paths").set_value(25_000)
    at.selectbox(key="var_dist").set_value("student_t").run()
    at.slider(key="var_dof").set_value(4)
    at.number_input(key="var_seed").set_value(11).run()
    _ok(at)
    before = {m.label: m.value for m in at.metric}
    _away_and_back(at, "Risk")
    assert at.slider(key="grid_span").value == 40 and at.slider(key="grid_days_365").value == 30
    assert at.select_slider(key="date_grid_reach").value == 3.0 and at.toggle(key="stress_hist").value is False
    assert at.selectbox(key="var_conf").value == 0.975 and at.select_slider(key="var_paths").value == 25_000
    assert at.selectbox(key="var_dist").value == "student_t" and at.slider(key="var_dof").value == 4
    assert at.number_input(key="var_seed").value == 11
    assert {m.label: m.value for m in at.metric} == before  # the same simulation: same seed, paths and model


def test_uploaded_quotes_survive_a_visit_to_another_tab(new_app):
    at = new_app(tab="Vol surface").run()
    at.segmented_control(key="surface_source").set_value("Market quotes (CSV)").run()
    csv = vol_surface.sample_quotes().to_csv(index=False).encode()
    at.file_uploader(key="quotes_file").upload("quotes.csv", csv, "text/csv").run()
    _ok(at)
    before = {m.label: m.value for m in at.metric}
    assert before["Calendar arbitrage"] == "0"
    _away_and_back(at, "Vol surface", "IV")
    # The uploader on screen is new and empty; the surface is still the one of the uploaded file.
    assert {m.label: m.value for m in at.metric} == before
    assert any("Showing quotes.csv, uploaded earlier in this session" in c.value for c in at.caption)
    # Removing the file (the uploader's x) is kept too.
    at.file_uploader(key="quotes_file").upload("other.csv", csv, "text/csv").run()
    at.file_uploader(key="quotes_file").clear().run()
    assert not any(m.label == "Calendar arbitrage" for m in at.metric)
    assert any("Upload a quotes file" in i.value for i in at.info)


def test_surface_and_iv_settings_survive_a_visit_to_another_tab(new_app):
    at = new_app(tab="Vol surface").run()
    at.number_input(key="ssvi_long").set_value(30.0)
    at.slider(key="ssvi_rho").set_value(-0.3)
    at.selectbox(key="surface_colors").set_value("Cividis").run()
    _ok(at)
    _away_and_back(at, "Vol surface")
    assert at.number_input(key="ssvi_long").value == 30.0 and at.slider(key="ssvi_rho").value == -0.3
    assert at.selectbox(key="surface_colors").value == "Cividis"

    _open(at, "IV")
    quote = next(n for n in at.number_input if n.label == "Market price")
    quote.set_value(12.0).run()
    at.segmented_control(key="iv_style_european").set_value("American").run()
    _ok(at)
    _away_and_back(at, "IV")
    assert next(n for n in at.number_input if n.label == "Market price").value == 12.0
    assert at.segmented_control(key="iv_style_european").value == "American"
    assert next(m.value for m in at.metric if m.label == "Implied volatility") == "24.112%"


def test_pricing_settings_survive_a_visit_to_another_tab(app):
    app.toggle(key="higher_greeks").set_value(True)
    app.selectbox(key="sens_quantity").set_value("Vega / vol pt")
    app.slider(key="hm_spot").set_value(35).run()
    _ok(app)
    _away_and_back(app, "Pricing", "Methodology")
    assert app.toggle(key="higher_greeks").value is True
    assert app.selectbox(key="sens_quantity").value == "Vega / vol pt" and app.slider(key="hm_spot").value == 35
