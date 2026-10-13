"""Streamlit tests of the price cards, the Pricing & Greeks tab and the Implied Volatility tab."""

import base64
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr
from streamlit.testing.v1 import AppTest

from options_analyzer import binomial, monte_carlo
from options_analyzer import black_scholes as bs
from options_analyzer.ui import charts

APP = str(Path(__file__).resolve().parents[1] / "main.py")
EUROPEAN_NOTE = "Closed-form Black-Scholes-Merton values for European exercise"


@pytest.fixture
def app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def _ok(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def _metrics(at: AppTest) -> dict:
    return {m.label: m.value for m in at.metric}


def _chart(at: AppTest, key: str) -> dict:
    for chart in at.get("plotly_chart"):
        if chart.proto.id.endswith(f"-{key}"):
            return json.loads(chart.proto.spec)
    raise AssertionError(f"no chart with key {key!r}")


def _values(data) -> np.ndarray:
    """Plotly JSON array: a plain list or a typed array ``{"dtype", "bdata"}``."""
    if isinstance(data, dict):
        return np.frombuffer(base64.b64decode(data["bdata"]), dtype=data["dtype"])
    return np.asarray(data)


def _binomial(at: AppTest, american: bool = False, **settings) -> AppTest:
    at.sidebar.radio(key="model").set_value("binomial").run()
    for key, value in settings.items():
        widget = at.sidebar.slider(key=key) if key == "steps" else at.sidebar.radio(key=key)
        widget.set_value(value).run()
    if american:
        at.sidebar.segmented_control(key="style").set_value("American").run()
    _ok(at)
    return at


def _iv_style(at: AppTest):
    return next(s for s in at.segmented_control if s.key.startswith("iv_style"))


def _quote(at: AppTest):
    return next(n for n in at.number_input if n.label == "Market price")


def _dollars(text: str) -> float:
    return float(text.replace("\\", "").replace("$", "").replace(",", ""))


# --- Sidebar ------------------------------------------------------------------------------------------------------


def test_binomial_model_defaults_to_the_leisen_reimer_lattice(app):
    _binomial(app)
    assert app.sidebar.radio(key="tree").value == "lr"
    assert app.sidebar.slider(key="steps").value == 501


def test_every_sidebar_model_setting_loads_from_the_url():
    at = AppTest.from_file(APP, default_timeout=120)
    for key, value in {
        "model": "Binomial tree",
        "tree": "Cox-Ross-Rubinstein",
        "steps": "1000",
        "style": "American",
    }.items():
        at.query_params[key] = value
    at.run()
    _ok(at)
    assert at.sidebar.radio(key="tree").value == "crr"
    assert at.sidebar.slider(key="steps").value == 1000
    assert at.sidebar.segmented_control(key="style").value == "American"

    at = AppTest.from_file(APP, default_timeout=120)
    for key, value in {
        "model": "Monte Carlo",
        "paths": "250,000",
        "seed": "7",
        "antithetic": "false",
        "control_variate": "false",
    }.items():
        at.query_params[key] = value
    at.run()
    _ok(at)
    assert at.sidebar.select_slider(key="paths").value == 250_000
    assert at.sidebar.number_input(key="seed").value == 7
    assert at.sidebar.toggle(key="antithetic").value is False
    assert at.sidebar.toggle(key="control_variate").value is False


# --- Headline cards ---------------------------------------------------------------------------------------------


def test_fourth_card_reports_what_each_model_adds_instead_of_a_zero_parity_gap(app):
    # The parity card read $0.000000 for every European model by construction.
    cards = _metrics(app)
    assert "Put-call parity gap" not in cards
    d2 = (np.log(100 / 100) + (0.05 - 0.5 * 0.2**2)) / 0.2
    assert cards["Call ITM probability"] == f"{ndtr(d2) * 100:.2f}%"

    _binomial(app, tree="crr")
    cards = _metrics(app)
    errors = [
        abs(binomial.price(100, 100, 1, 0.05, 0.2, 0, k, 501) - bs.price(100, 100, 1, 0.05, 0.2, 0, k))
        for k in ("call", "put")
    ]
    assert _dollars(cards["Tree error vs BSM"]) == pytest.approx(max(errors), abs=1e-6)
    assert _dollars(cards["Tree error vs BSM"]) > 1e-3  # CRR's error at 501 steps is visible

    app.sidebar.segmented_control(key="style").set_value("American").run()
    cards = _metrics(app)
    american = binomial.price(100, 100, 1, 0.05, 0.2, 0, "put", 501, True, "crr")
    european = binomial.price(100, 100, 1, 0.05, 0.2, 0, "put", 501, False, "crr")
    assert _dollars(cards["Exercise premium"]) == pytest.approx(american - european, abs=1e-4)

    app.sidebar.radio(key="model").set_value("monte_carlo").run()
    _ok(app)
    se = monte_carlo.price_european(100, 100, 1, 0.05, 0.2, 0, n_paths=100_000, seed=42)
    assert _dollars(_metrics(app)["Std. error"]) == pytest.approx(max(r.std_error for r in se.values()), abs=1e-4)


# --- Pricing & Greeks tab --------------------------------------------------------------------------------------


def test_european_only_views_say_so_when_another_model_is_selected(app):
    assert not [c for c in app.caption if EUROPEAN_NOTE in c.value]
    _binomial(app, american=True)
    notes = [c.value for c in app.caption if EUROPEAN_NOTE in c.value]
    # Sensitivity explorer, price heatmaps and the option chain.
    assert len(notes) == 3
    assert all("Early exercise is ignored here" in n for n in notes)
    app.sidebar.radio(key="model").set_value("monte_carlo").run()
    assert len([c for c in app.caption if EUROPEAN_NOTE in c.value]) == 3


def test_tree_greeks_table_has_vega_rho_and_dividend_rho_from_the_tree(app):
    app.sidebar.number_input(key="strike").set_value(120.0).run()
    _binomial(app, american=True)
    table = next(d.value for d in app.dataframe if "Div. rho / 1%" in d.value.columns).set_index("Option")
    tree = binomial.price_with_greeks(100, 120, 1, 0.05, 0.2, 0, "put", 501, True, "lr")
    assert table.loc["Put", "Vega / vol pt"] == pytest.approx(tree.vega / 100)
    assert table.loc["Put", "Rho / 1%"] == pytest.approx(tree.rho / 100)
    assert table.loc["Put", "Div. rho / 1%"] == pytest.approx(tree.epsilon / 100)
    # The European (Black-Scholes-Merton) vega is almost three times larger for this in-the-money put.
    assert table.loc["Put", "Vega / vol pt"] < 0.5 * bs.greeks(100, 120, 1, 0.05, 0.2, 0, "put")["vega"] / 100


def test_convergence_charts_show_call_and_put_up_to_the_selected_steps(app):
    _binomial(app, american=True, steps=1500)
    for kind in ("call", "put"):
        spec = _chart(app, f"tree_convergence_{kind}")
        traces = {t["name"]: t for t in spec["data"]}
        assert set(traces) == {"Cox-Ross-Rubinstein", "Leisen-Reimer"}
        assert _values(traces["Cox-Ross-Rubinstein"]["x"]).max() == 1500
        assert _values(traces["Leisen-Reimer"]["x"]).max() == 1501
        # American exercise gets a converged reference line too.
        reference = binomial.extrapolated_price(100, 100, 1, 0.05, 0.2, 0, kind)
        assert any(a["text"] == f"Converged {reference:,.4f}" for a in spec["layout"]["annotations"])


def test_monte_carlo_sample_paths_are_scenarios_behind_the_price(app):
    app.sidebar.radio(key="model").set_value("monte_carlo").run()
    _ok(app)
    spec = _chart(app, "mc_paths")
    paths = [t for t in spec["data"] if not t.get("name")]
    assert len(paths) == 60
    draws = monte_carlo.terminal_prices(100, 1, 0.05, 0.2, 0, 100_000, 42, True)
    np.testing.assert_allclose([_values(p["y"])[-1] for p in paths], draws[:60], rtol=1e-12)
    mean = next(t for t in spec["data"] if t.get("name") == "Mean of all scenarios")
    assert _values(mean["y"])[-1] == pytest.approx(100 * np.exp(0.05))
    assert any("60 of the 100,000 scenarios behind the price" in m.value for m in app.markdown)


# --- Implied Volatility tab --------------------------------------------------------------------------------------


def test_iv_exercise_style_follows_the_sidebar_unless_chosen_in_the_tab(app):
    assert _iv_style(app).value == "European"
    _binomial(app, american=True)
    assert _iv_style(app).value == "American"
    assert any("No-arbitrage bounds for this American call" in c.value for c in app.caption)
    # The user's own choice is kept while the sidebar style stays the same...
    _iv_style(app).set_value("European").run()
    app.sidebar.number_input(key="vol").set_value(25.0).run()
    assert _iv_style(app).value == "European"
    # ...and the sidebar takes over again when its style changes.
    app.sidebar.segmented_control(key="style").set_value("European").run()
    app.sidebar.segmented_control(key="style").set_value("American").run()
    assert _iv_style(app).value == "American"


def test_iv_quote_survives_volatility_and_style_changes_and_resets_on_request(app):
    _quote(app).set_value(12.0).run()
    assert _metrics(app)["Implied volatility"] == "24.112%"
    app.sidebar.number_input(key="vol").set_value(25.0).run()
    assert _quote(app).value == 12.0
    assert _metrics(app)["Implied volatility"] == "24.112%"
    _iv_style(app).set_value("American").run()
    assert _quote(app).value == 12.0
    next(b for b in app.button if b.label == "Reset to model value").click().run()
    assert _quote(app).value == pytest.approx(bs.price(100, 100, 1, 0.05, 0.25), abs=1e-3)  # American call = European
    assert _metrics(app)["Implied volatility"] == "25.000%"
    # A different contract starts from its own model value.
    _quote(app).set_value(12.0).run()
    app.sidebar.number_input(key="strike").set_value(105.0).run()
    assert _quote(app).value == pytest.approx(bs.price(100, 105, 1, 0.05, 0.25), abs=1e-3)


def test_iv_tab_inverts_american_quotes_on_the_sidebar_tree(app):
    _binomial(app, american=True, tree="crr")
    headline_put = _metrics(app)["Put value"]
    next(s for s in app.segmented_control if s.key == "iv_type").set_value("put").run()
    _quote(app).set_value(_dollars(headline_put)).run()
    _ok(app)
    # The 501-step CRR value inverted on a Leisen-Reimer tree used to give 20.009%.
    assert _metrics(app)["Implied volatility"] == "20.000%"
    assert any("501-step Cox-Ross-Rubinstein tree" in c.value for c in app.caption)


def test_iv_time_value_is_quote_minus_intrinsic_value(app):
    assert _metrics(app)["Time value"] == "$10.4506"  # at the money: all of the premium
    app.sidebar.number_input(key="strike").set_value(120.0).run()
    next(s for s in app.segmented_control if s.key == "iv_type").set_value("put").run()
    put = bs.price(100, 120, 1, 0.05, 0.2, 0, "put")
    assert _dollars(_metrics(app)["Time value"]) == pytest.approx(put - 20.0, abs=1e-4)


def test_iv_gap_to_the_sidebar_never_shows_a_negative_zero(app):
    app.sidebar.number_input(key="strike").set_value(40.0).run()
    _ok(app)
    assert next(m for m in app.metric if m.label == "Implied volatility").delta == "+0.00 vol pts vs sidebar"


# --- Charts ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spot", "vol", "spot_span", "vol_span"),
    [(0.15, 0.20, 0.20, 0.50), (0.65, 0.08, 0.05, 0.50), (100.0, 0.01, 0.20, 0.10), (4500.0, 0.2, 0.2, 0.5)],
)
def test_price_heatmap_keeps_every_row_and_column_for_low_prices_and_volatilities(spot, vol, spot_span, vol_span):
    # Fixed two-decimal labels repeated for penny stocks and FX rates (Plotly then merges the columns), and one-
    # decimal vol labels repeated at 1% volatility.
    spots = spot * np.linspace(1 - spot_span, 1 + spot_span, 9)
    vols = vol * np.linspace(1 - vol_span, 1 + vol_span, 9)
    grid_s, grid_v = np.meshgrid(spots, vols)
    values = bs.price(grid_s, spot, 1.0, 0.05, grid_v, 0.0, "call")
    trace = charts.price_heatmap(spots, vols, values, "Call").data[0]
    assert len(set(trace.x)) == 9 and len(set(trace.y)) == 9
    # Cell text keeps about three significant digits of the largest value instead of reading 0.00 everywhere.
    largest = max(float(t.replace(",", "")) for row in trace.text for t in row)
    assert largest == pytest.approx(values.max(), rel=5e-3)


def test_sensitivity_chart_labels_cannot_overlap_the_axis_or_each_other():
    x = np.linspace(50, 150, 11)
    same = charts.sensitivity_chart(x, x, x, "Spot", "Value", 100.0, 100.0)
    assert [a.text for a in same.layout.annotations] == ["current = strike"]
    apart = charts.sensitivity_chart(x, x, x, "Spot", "Value", 100.0, 103.0)
    labels = {a.text: a for a in apart.layout.annotations}
    # Both at the top of the plot (not below it, over the tick labels), pointing away from each other.
    assert all(a.y == 1 and a.yanchor == "top" for a in labels.values())
    assert (labels["current"].xanchor, labels["strike"].xanchor) == ("right", "left")
