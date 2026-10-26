"""Streamlit tests of the price cards, the Pricing tab and the IV tab."""

import base64
import json
import re

import numpy as np
import pytest
from scipy.special import ndtr
from streamlit.testing.v1 import AppTest

from options_analyzer import binomial, monte_carlo
from options_analyzer import black_scholes as bs
from options_analyzer.ui import charts, components, tab_pricing

EUROPEAN_NOTE = "Closed-form Black-Scholes-Merton values for European exercise"


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


@pytest.fixture
def iv_app(new_app) -> AppTest:
    """The app with the IV tab open (only the open tab runs)."""
    at = new_app(tab="IV").run()
    _ok(at)
    return at


def _iv_style(at: AppTest):
    return next(s for s in at.segmented_control if s.key.startswith("iv_style"))


def _quote(at: AppTest):
    return next(n for n in at.number_input if n.label == "Market price")


def _dollars(text: str) -> float:
    return float(text.replace("\\", "").replace("$", "").replace(",", ""))


def _block(at: AppTest, key: str):
    """The container created with ``key``."""

    def find(node):
        for child in getattr(node, "children", {}).values():
            if str(getattr(getattr(child, "proto", None), "id", "")).endswith(f"-{key}"):
                return child
            if (found := find(child)) is not None:
                return found
        return None

    block = find(at.main)
    assert block is not None, f"no container with key {key!r}"
    return block


# --- Sidebar ------------------------------------------------------------------------------------------------------


def test_binomial_model_defaults_to_the_leisen_reimer_lattice(app):
    _binomial(app)
    assert app.sidebar.radio(key="tree").value == "lr"
    assert app.sidebar.slider(key="steps").value == 501


def test_every_sidebar_model_setting_loads_from_the_url(new_app):
    at = new_app()
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

    at = new_app()
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


def _price_card_style(at: AppTest) -> str:
    """The price cards' stylesheet (the app's own stylesheet comes first)."""
    return next(h.proto.body for h in at.get("html") if ".st-key-price_cards " in h.proto.body)


def test_price_cards_wrap_in_pairs_of_equal_width(app):
    # The four cards sat in one wrapping row sized by their content, so it broke 3 + 1 (the fourth card stretched
    # across the whole row) at 1,024 px with the sidebar open, and at 1,400 px with seven-digit prices.
    assert [m.label for m in _block(app, "price_cards_prices").metric] == ["Call value", "Put value"]
    assert [m.label for m in _block(app, "price_cards_model").metric] == ["Forward price", "Call ITM probability"]
    card = int(np.ceil(max(components._card_width(m.label, m.value) for m in app.metric[:4])))
    style = _price_card_style(app)
    cards_rule = f".st-key-price_cards_prices > *, .st-key-price_cards_model > * {{ flex: 1 1 {card}px !important; }}"
    assert cards_rule in style
    assert f"{{ flex: 1 1 {2 * card + 14}px !important; }}" in style  # each pair: two cards and the gap
    assert card == 174  # the ITM card's label is the widest


@pytest.mark.parametrize(
    ("values", "phone"),
    # Default inputs, and an index-level underlying (the forward is the widest value), on a 340 and a 360 px phone.
    [
        (("$10.4506", "$5.5735", "$105.1271", "55.96%"), 340),
        (("$10,450.5836", "$5,573.5260", "$105,127.1096", "55.96%"), 360),
    ],
)
def test_price_cards_stay_two_by_two_on_a_phone(app, values, phone):
    # Every card was based on its full one-line width (186 px), so on phones narrower than about 420 px both pairs
    # stacked: four full-width cards that pushed the tabs below the first screen. Where the row is narrower than one
    # full-size pair, the labels now wrap and the values shrink with the row (down to 16 px). In IBM Plex Sans the
    # values also never grow beyond their desktop size of 22.75 px: at 36 px on a phone they were cut to '$10.45…'.
    style = _price_card_style(app)
    assert ".st-key-price_cards { container: price-cards / inline-size; }" in style
    assert "@container price-cards (width < 362px) {" in style
    assert "white-space: normal;" in style
    # 5.19 em: "$105.1271" and a tenth of an em to spare
    assert "font-size: clamp(16px, calc((50cqw - 35px) / 5.19), 22.75px);" in style
    labels = ["Call value", "Put value", "Forward price", "Call ITM probability"]
    compact = components._cards_style(list(zip(labels, values, strict=True)), *tab_pricing.PRICE_CARDS)
    # The cards of a pair go one per line only on a row narrower than two compact cards and the gap.
    stack = int(re.search(r"@container price-cards \(width < (\d+)px\) \{ [^@]*repeat\(1, ", compact).group(1))
    row = phone - 2 * 14  # the page's side margins on a phone
    assert stack <= row  # two cards side by side
    em = max(components._value_em(v) for v in values)
    font = min(max(16.0, ((row - 14) / 2 - 28) / em), 22.75)  # the clamp() at this row width
    assert font * em <= (row - 14) / 2 - 28  # the widest value fits its card


@pytest.mark.parametrize(
    ("label", "value", "measured"),
    # max-content widths of the cards in Chromium (IBM Plex Sans, the theme's 14 px base), rounded up, in px
    [
        ("Call value", "$10.4506", 130.2),
        ("Put value", "$5.5735", 116.5),
        ("Forward price", "$1,051,271.0964", 211.6),
        ("Exercise premium", "$5,162.6001", 164.1),
        ("Tree error vs BSM", "$0.003499", 144.4),
        ("Call ITM probability", "55.96%", 154.6),
    ],
)
def test_card_width_estimate_covers_the_rendered_card(label, value, measured):
    # Never narrower than the card (its value would be cut), and not so wide that the row wraps much too early.
    assert measured <= components._card_width(label, value) <= measured + 20


@pytest.mark.parametrize(
    ("delta", "measured"),
    # max-content widths in Chromium of the implied-volatility card (value 20.000%) with its delta line, in px
    [("+0.00 vol pts vs sidebar", 169.3), ("-12.34 vol pts vs sidebar", 174.2), ("+123.45 vol pts vs sidebar", 184.0)],
)
def test_card_width_estimate_covers_the_delta_line(delta, measured):
    assert measured <= components._card_width("Implied volatility", "20.000%", delta=delta) <= measured + 20


# --- Pricing tab -------------------------------------------------------------------------------------------------


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


def test_converged_reference_stays_below_the_strike_for_extreme_volatility_and_expiry(new_app):
    # At 400% volatility and ten years the 4,003-step reference tree rounded its bottom spots to zero: the put's
    # "Converged" line read 103.7119 (above the strike of 100) next to a put value of $96.2280.
    at = new_app()
    at.query_params.update({"vol": "400", "days": "3650", "model": "Binomial tree", "style": "American"})
    at.run()
    _ok(at)
    put = _dollars(_metrics(at)["Put value"])
    texts = [a["text"] for a in _chart(at, "tree_convergence_put")["layout"]["annotations"]]
    reference = float(next(t for t in texts if t.startswith("Converged ")).split()[1].replace(",", ""))
    assert put < reference < 100.0
    assert reference == pytest.approx(put, abs=0.1)


def test_converged_reference_of_a_put_with_a_negative_rate_is_not_capped_at_the_strike(new_app):
    # With a negative rate the put is worth up to K e^(-rT), more than its strike: the "Converged" line read
    # 100.0000, 15.5 below both tree curves and the put card ($115.5045).
    at = new_app()
    at.query_params.update(
        {"spot": "50", "strike": "100", "days": "3650", "rate": "-5", "model": "Binomial tree", "style": "American"}
    )
    at.run()
    _ok(at)
    assert _metrics(at)["Put value"] == "$115.5045"
    texts = [a["text"] for a in _chart(at, "tree_convergence_put")["layout"]["annotations"]]
    assert "Converged 115.5045" in texts


def test_no_exercise_boundary_is_drawn_when_early_exercise_is_never_optimal(new_app):
    # At a zero rate without dividends neither option is ever exercised early (the premium card reads $0.0000), but
    # rounding noise drew a call boundary from about 148 down to 100 and a put boundary from about 67 up to 100.
    at = new_app()
    at.query_params.update({"rate": "0", "vol": "10", "model": "Binomial tree", "style": "American"})
    at.run()
    _ok(at)
    assert _metrics(at)["Exercise premium"] == "$0.0000"
    assert not [c for c in at.get("plotly_chart") if c.proto.id.endswith("-exercise_boundary")]
    assert any("Early exercise is never optimal" in i.value for i in at.info)


def test_exercise_boundaries_are_drawn_for_a_small_rate_equal_to_the_dividend_yield(new_app):
    # A 0.01% rate and dividend yield over one day at 0.1% volatility: both options have an exercise region ending at
    # the strike, but gains below 1e-12 of the strike were not counted, so the chart vanished and the app said early
    # exercise was never optimal.
    at = new_app()
    at.query_params.update(
        {"days": "1", "vol": "0.1", "rate": "0.01", "div": "0.01", "model": "Binomial tree", "style": "American"}
    )
    at.run()
    _ok(at)
    assert not any("Early exercise is never optimal" in i.value for i in at.info)
    curves = {t["name"]: _values(t["y"]) for t in _chart(at, "exercise_boundary")["data"]}
    assert set(curves) == {"Call boundary", "Put boundary"}
    assert curves["Put boundary"].max() == pytest.approx(100, abs=1e-3) and curves["Put boundary"].max() < 100
    assert curves["Call boundary"].min() == pytest.approx(100, abs=1e-3) and curves["Call boundary"].min() > 100


def test_both_exercise_boundaries_are_drawn_in_their_option_colours(app):
    # With dividends both boundaries are drawn, the call's first; it used to get the put's orange and vice versa.
    app.sidebar.number_input(key="div").set_value(3.0).run()
    _binomial(app, american=True)
    spec = _chart(app, "exercise_boundary")
    assert {t["name"]: t["line"]["color"] for t in spec["data"]} == {
        "Call boundary": charts.CALL,
        "Put boundary": charts.PUT,
    }


def test_monte_carlo_sample_paths_are_scenarios_behind_the_price(app):
    app.sidebar.radio(key="model").set_value("monte_carlo").run()
    _ok(app)
    spec = _chart(app, "mc_paths")
    paths = [t for t in spec["data"] if not t.get("name")]
    assert len(paths) == 60
    draws = monte_carlo.terminal_prices(100, 1, 0.05, 0.2, 0, 100_000, 42, True)
    np.testing.assert_allclose([_values(p["y"])[-1] for p in paths], draws[:60], rtol=1e-12)
    # The line is the forward, the scenarios' expected value, not their sample mean ("Mean of all scenarios").
    mean = next(t for t in spec["data"] if t.get("name") == "Risk-neutral mean (forward)")
    assert _values(mean["y"])[-1] == pytest.approx(100 * np.exp(0.05))
    assert _values(mean["y"])[-1] != pytest.approx(draws.mean(), abs=1e-3)
    assert any("differs from it by sampling noise" in c.value for c in app.caption)
    assert any("60 of the 100,000 scenarios behind the price" in m.value for m in app.markdown)


@pytest.mark.parametrize(("vol", "call", "put"), [("400", "$100.0000", "$60.6531"), ("300", "$99.9993", "$60.6524")])
def test_monte_carlo_cards_stay_right_at_hundreds_of_percent_of_volatility_over_years(vol, call, put, new_app):
    # Ten years at 400% volatility showed a call of $0.0000 and a put of -$39.3469; at 300% a call of $95.0341 with a
    # standard error of $0.0000 (the values are 100.0000 / 60.6531 and 99.9998 / 60.6529).
    at = new_app()
    at.query_params.update({"days": "3650", "vol": vol, "model": "Monte Carlo"})
    at.run()
    _ok(at)
    cards = _metrics(at)
    assert (cards["Call value"], cards["Put value"]) == (call, put)
    assert all(
        abs(_dollars(cards[f"{k.capitalize()} value"]) - bs.price(100, 100, 10, 0.05, float(vol) / 100, 0, k)) < 1e-3
        for k in ("call", "put")
    )
    assert any(tab_pricing.HEAVY_TAIL_NOTE in c.value for c in at.caption)


def test_monte_carlo_call_without_control_variate_follows_from_parity_at_large_sigma_sqrt_t(new_app):
    # Four years at 250% volatility with the control variate off: the cards showed a call of $31.0160 with a standard
    # error of $16.1587 (value 98.8770) next to a correct put of $80.7879, breaking put-call parity by $68.
    at = new_app()
    at.query_params.update(
        {"days": "1460", "vol": "250", "model": "Monte Carlo", "seed": "103", "control_variate": "false"}
    )
    at.run()
    _ok(at)
    assert at.sidebar.toggle(key="control_variate").value is False
    cards = _metrics(at)
    assert (cards["Call value"], cards["Put value"], cards["Std. error"]) == ("$98.9148", "$80.7879", "$0.0242")
    assert _dollars(cards["Call value"]) - _dollars(cards["Put value"]) == pytest.approx(100 - 100 * np.exp(-0.2), 1e-4)
    assert any(tab_pricing.HEAVY_TAIL_NOTE in c.value for c in at.caption)
    assert any("from σ√T = 2 on the call follows from the put by parity" in c.value for c in at.caption)


def test_monte_carlo_diagnostics_leave_undefined_cells_blank(new_app):
    # Streamlit writes "None" in missing cells unless told otherwise, right above a caption saying they are blank.
    at = new_app()
    at.query_params.update({"model": "Monte Carlo", "strike": "300", "days": "30", "vol": "10", "paths": "10,000"})
    at.run()
    _ok(at)
    table = next(d for d in at.dataframe if "Error / SE" in d.value.columns)
    assert table.value["Std. error"].isna().all() and table.value["Error / SE"].isna().all()
    assert table.proto.HasField("placeholder") and table.proto.placeholder == ""
    assert any("The standard error is left blank" in c.value for c in at.caption)


def test_greek_names_get_a_column_wide_enough_for_the_longest(app):
    # The grid sized the column from the first rows it drew and ignored "Epsilon (dividend rho) ε" as an outlier,
    # so the name was cut at 1,280 px and on phones ("Epsilon (di", and "Vomma (vo" with the higher-order Greeks).
    table = next(d for d in app.dataframe if "Greek" in d.value.columns)
    assert json.loads(table.proto.columns)["Greek"]["width"] == "medium"


# --- IV tab ------------------------------------------------------------------------------------------------------


def test_iv_exercise_style_follows_the_sidebar_unless_chosen_in_the_tab(iv_app):
    assert _iv_style(iv_app).value == "European"
    _binomial(iv_app, american=True)
    assert _iv_style(iv_app).value == "American"
    assert any("No-arbitrage bounds for this American call" in c.value for c in iv_app.caption)
    # The user's own choice is kept while the sidebar style stays the same...
    _iv_style(iv_app).set_value("European").run()
    iv_app.sidebar.number_input(key="vol").set_value(25.0).run()
    assert _iv_style(iv_app).value == "European"
    # ...and the sidebar takes over again when its style changes.
    iv_app.sidebar.segmented_control(key="style").set_value("European").run()
    iv_app.sidebar.segmented_control(key="style").set_value("American").run()
    assert _iv_style(iv_app).value == "American"


def test_iv_quote_survives_volatility_and_style_changes_and_resets_on_request(iv_app):
    _quote(iv_app).set_value(12.0).run()
    assert _metrics(iv_app)["Implied volatility"] == "24.112%"
    iv_app.sidebar.number_input(key="vol").set_value(25.0).run()
    assert _quote(iv_app).value == 12.0
    assert _metrics(iv_app)["Implied volatility"] == "24.112%"
    _iv_style(iv_app).set_value("American").run()
    assert _quote(iv_app).value == 12.0
    next(b for b in iv_app.button if b.label == "Reset to model value").click().run()
    assert _quote(iv_app).value == pytest.approx(
        bs.price(100, 100, 1, 0.05, 0.25), abs=1e-3
    )  # American call = European
    assert _metrics(iv_app)["Implied volatility"] == "25.000%"
    # A different contract starts from its own model value.
    _quote(iv_app).set_value(12.0).run()
    iv_app.sidebar.number_input(key="strike").set_value(105.0).run()
    assert _quote(iv_app).value == pytest.approx(bs.price(100, 105, 1, 0.05, 0.25), abs=1e-3)


def test_iv_tab_inverts_american_quotes_on_the_sidebar_tree(iv_app):
    _binomial(iv_app, american=True, tree="crr")
    headline_put = _metrics(iv_app)["Put value"]
    next(s for s in iv_app.segmented_control if s.key == "iv_type").set_value("put").run()
    _quote(iv_app).set_value(_dollars(headline_put)).run()
    _ok(iv_app)
    # The 501-step CRR value inverted on a Leisen-Reimer tree used to give 20.009%.
    assert _metrics(iv_app)["Implied volatility"] == "20.000%"
    assert any("501-step Cox-Ross-Rubinstein tree" in c.value for c in iv_app.caption)


def test_iv_tab_accepts_an_american_put_worth_more_than_its_strike_under_a_negative_rate(new_app):
    # With a negative rate the put is worth up to K e^(-rT): the tab used to cap it at the strike, showing the
    # bounds "$114.8721 <= price < $100.0000" and rejecting its own model value $115.5045 as a quote.
    at = new_app(tab="IV")
    at.query_params.update(
        {"spot": "50", "strike": "100", "days": "3650", "rate": "-5", "model": "Binomial tree", "style": "American"}
    )
    at.run()
    next(s for s in at.segmented_control if s.key == "iv_type").set_value("put").run()
    _ok(at)
    assert _quote(at).value == pytest.approx(115.5045, abs=1e-4)
    bounds = next(c.value for c in at.caption if c.value.startswith("No-arbitrage bounds for this American put"))
    assert bounds.replace("\\", "").endswith("$114.8721 ≤ price < $164.8721")
    assert _metrics(at)["Implied volatility"] == "20.000%"


def test_iv_time_value_is_quote_minus_intrinsic_value(iv_app):
    assert _metrics(iv_app)["Time value"] == "$10.4506"  # at the money: all of the premium
    iv_app.sidebar.number_input(key="strike").set_value(120.0).run()
    next(s for s in iv_app.segmented_control if s.key == "iv_type").set_value("put").run()
    put = bs.price(100, 120, 1, 0.05, 0.2, 0, "put")
    assert _dollars(_metrics(iv_app)["Time value"]) == pytest.approx(put - 20.0, abs=1e-4)


def test_iv_cards_wrap_in_pairs_like_the_price_cards(iv_app):
    # The four cards sat in st.columns(4) with no room to grow: at 1,280 px "Model check" read "$10.450…" for the
    # default contract, and with a $10,000,000 underlying every dollar card lost digits even at 1,400 px.
    assert [m.label for m in _block(iv_app, "iv_cards_solution").metric] == ["Implied volatility", "Model check"]
    assert [m.label for m in _block(iv_app, "iv_cards_option").metric] == ["Vega at IV", "Time value"]
    style = next(h.proto.body for h in iv_app.get("html") if ".st-key-iv_cards " in h.proto.body)
    assert ".st-key-iv_cards { container: iv-cards / inline-size; }" in style
    cards = [m for m in iv_app.metric if m.label in ("Implied volatility", "Model check", "Vega at IV", "Time value")]
    card = int(np.ceil(max(components._card_width(m.label, m.value, delta=m.proto.delta) for m in cards)))
    assert f".st-key-iv_cards_solution > *, .st-key-iv_cards_option > * {{ flex: 1 1 {card}px !important; }}" in style
    # On a phone the delta line wraps with the labels instead of being cut off.
    assert (
        "[data-testid='stMetricDelta']) :is([data-testid='stMarkdownContainer'], p) { white-space: normal; }" in style
    )


def test_iv_model_check_shows_the_quote_precision_and_the_full_value_in_its_help(iv_app):
    # Six decimals did not fit the card; the full value and its distance from the quote are in the help.
    check = next(m for m in iv_app.metric if m.label == "Model check")
    assert check.value == "$10.4506"
    value = bs.price(100, 100, 1, 0.05, 0.2)
    assert f"\\${value:,.6f}, " in check.proto.help and "from the quote." in check.proto.help


def test_iv_gap_to_the_sidebar_never_shows_a_negative_zero(iv_app):
    iv_app.sidebar.number_input(key="strike").set_value(40.0).run()
    _ok(iv_app)
    assert next(m for m in iv_app.metric if m.label == "Implied volatility").delta == "+0.00 vol pts vs sidebar"


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


def test_price_heatmaps_give_their_cells_the_full_width(app):
    # Side by side, each with a colour scale, the 9-column grids left the cell text at 5 px on a phone, 7 px at
    # 1,280 px and 9 px at 1,400 px (Plotly shrinks it to fit the cell). They are now stacked, without the scale.
    def parent(node, key):
        for child in getattr(node, "children", {}).values():
            if str(getattr(getattr(child, "proto", None), "id", "")).endswith(f"-{key}"):
                return node
            if (found := parent(child, key)) is not None:
                return found
        return None

    for kind in ("call", "put"):
        assert _chart(app, f"heatmap_{kind}")["data"][0]["showscale"] is False
        assert parent(app.main, f"heatmap_{kind}").type == "expander"  # not one of two columns
    assert charts.price_heatmap([1.0, 2.0], [0.1, 0.2], np.ones((2, 2)), "Call").data[0].showscale is False


def test_exercise_boundary_colours_follow_the_option_type():
    # Colours went by position: the first boundary was orange (the put colour) even when it was the call's.
    days = np.array([0.0, 100.0])
    call, put = (days, np.array([130.0, 120.0])), (days, np.array([80.0, 90.0]))
    both = charts.exercise_boundary_chart({"Call boundary": call, "Put boundary": put}, 100.0)
    assert {t.name: t.line.color for t in both.data} == {"Call boundary": charts.CALL, "Put boundary": charts.PUT}
    assert charts.exercise_boundary_chart({"Put boundary": put}, 100.0).data[0].line.color == charts.PUT


def test_sensitivity_chart_labels_cannot_overlap_the_axis_or_each_other():
    x = np.linspace(50, 150, 11)
    same = charts.sensitivity_chart(x, x, x, "Spot", "Value", 100.0, 100.0)
    assert [a.text for a in same.layout.annotations] == ["current = strike"]
    apart = charts.sensitivity_chart(x, x, x, "Spot", "Value", 100.0, 103.0)
    labels = {a.text: a for a in apart.layout.annotations}
    # Both at the top of the plot (not below it, over the tick labels), pointing away from each other.
    assert all(a.y == 1 and a.yanchor == "top" for a in labels.values())
    assert (labels["current"].xanchor, labels["strike"].xanchor) == ("right", "left")


def test_iv_tab_explains_a_tree_too_short_for_extreme_volatility(iv_app):
    # Round-3 verifier: a 15-step tree at 500% volatility over 10 years with a negative rate raised a raw
    # ValueError from the American value curve ("no arbitrage-free lattice exists"), shown as a crash.
    iv_app.sidebar.number_input(key="days").set_value(3650).run()
    iv_app.sidebar.number_input(key="rate").set_value(-5.0).run()
    iv_app.sidebar.number_input(key="vol").set_value(500.0).run()
    _binomial(iv_app, american=True, tree="lr", steps=15)
    for kind in ("call", "put"):
        iv_app.segmented_control(key="iv_type").set_value(kind).run()
        assert not iv_app.exception, [e.value for e in iv_app.exception]
        assert all("too few steps" in e.value for e in iv_app.error), [e.value for e in iv_app.error]
