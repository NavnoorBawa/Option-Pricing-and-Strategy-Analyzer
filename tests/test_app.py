"""End-to-end smoke tests of the Streamlit app with streamlit.testing (headless)."""

import json

import pytest
from streamlit.testing.v1 import AppTest

from options_analyzer import strategies, vol_surface


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


def test_every_strategy_template(app):
    for category in strategies.TEMPLATE_CATEGORIES:
        app.selectbox(key="strategy_category").set_value(category).run()
        names = [t.name for t in strategies.TEMPLATES.values() if t.category == category]
        for name in names:
            app.selectbox(key=f"strategy_{category}").set_value(name).run()
            _ok(app)


def test_var_models(app):
    for dist in ("student_t", "garch", "normal"):
        app.selectbox(key="var_dist").set_value(dist).run()
        _ok(app)
    assert any(m.label.startswith("VaR") for m in app.metric)


def test_volatility_surface_upload(app):
    app.segmented_control(key="surface_source").set_value("Market quotes (CSV)").run()
    csv = vol_surface.sample_quotes().to_csv(index=False).encode()
    app.file_uploader(key="quotes_file").upload("quotes.csv", csv, "text/csv").run()
    _ok(app)
    assert any(m.label == "Calendar arbitrage" for m in app.metric)
    app.file_uploader(key="quotes_file").upload("bad.csv", b"a,b\n1,2\n", "text/csv").run()
    assert not app.exception
    assert any("Strike column" in e.value for e in app.error)


def test_implied_vol_round_trip(app):
    iv = next(m.value for m in app.metric if m.label == "Implied volatility")
    assert iv == "20.000%"


def _trace(at: AppTest, prefix: str) -> dict:
    for chart in at.get("plotly_chart"):
        for trace in json.loads(chart.proto.spec)["data"]:
            if (trace.get("name") or "").startswith(prefix):
                return trace
    raise AssertionError(f"no trace named {prefix!r}")


def test_iv_shift_leaves_the_expiry_curve_and_metrics_alone(app):
    # A calendar's back leg is alive at the front expiry; the slider must not move the at-expiry curve away
    # from the breakevens and maxima computed without the shift.
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
        for chart in at.get("plotly_chart"):
            key = chart.proto.id.rsplit("-", 1)[-1]
            config = json.loads(chart.proto.config)
            assert config["displaylogo"] is False and config["displayModeBar"] == "hover", key
            assert config["toImageButtonOptions"] == {"format": "svg", "filename": f"options-analyzer-{key}"}, key
            assert chart.proto.theme == "streamlit", key
            assert len(chart.proto.alt) >= 20 and not chart.proto.alt[0].islower(), (key, chart.proto.alt)
            seen.add(key)
    assert seen == CHART_KEYS
