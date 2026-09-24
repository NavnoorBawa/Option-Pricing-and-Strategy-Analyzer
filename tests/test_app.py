"""End-to-end smoke tests of the Streamlit app with streamlit.testing (headless)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from options_analyzer import strategies, vol_surface

APP = str(Path(__file__).resolve().parents[1] / "main.py")


@pytest.fixture
def app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


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
    assert any(m.label == "Calendar-arbitrage points" for m in app.metric)
    app.file_uploader(key="quotes_file").upload("bad.csv", b"a,b\n1,2\n", "text/csv").run()
    assert not app.exception
    assert any("Strike column" in e.value for e in app.error)


def test_implied_vol_round_trip(app):
    iv = next(m.value for m in app.metric if m.label == "Implied volatility")
    assert iv == "20.000%"
