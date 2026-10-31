"""Streamlit tests of the price cards, the Pricing tab and the IV tab."""

import base64
import json
import math
import re

import numpy as np
import pytest
from scipy.special import ndtr
from streamlit.testing.v1 import AppTest

from options_analyzer import american, binomial, monte_carlo
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
    # The ITM card's label, in small capitals, is the widest: 170 px (a golden value, which a change of the width
    # model must update on purpose).
    assert card == int(np.ceil(components._card_width("Call ITM probability", "55.96%"))) == 170


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
    # Compact below one full-size pair: two of the widest card (the ITM card's label) and the gap, 354 px (golden).
    pair = 2 * int(np.ceil(components._card_width("Call ITM probability", "55.96%"))) + 14
    assert pair == 354 and "@container price-cards (width < 354px) {" in style
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
    ("label", "value", "delta", "measured"),
    # max-content widths in Chromium of cards with a delta line (in a pill), rounded up, in px: the implied-volatility
    # card, and the Monte Carlo price cards with their interval (the model made that pill 13 px too narrow at first)
    # or the reason it is not shown. (The long reason, "Interval unreliable: 85 of 100,000 paths finish out of the
    # money", made its card 387.3 px wide, and at 1440 px the row of four, 249 px a card, went 2 + 2.)
    [
        ("Implied volatility", "20.000%", "+0.00 vol pts vs sidebar", 169.3),
        ("Implied volatility", "20.000%", "-12.34 vol pts vs sidebar", 174.2),
        ("Implied volatility", "20.000%", "+123.45 vol pts vs sidebar", 184.0),
        ("Call value", "$10.4409", r"\$10.4237 – \$10.4582, 95%", 193.3),
        ("Put value", "$5.5639", r"\$5.5466 – \$5.5811, 95%", 178.6),
        ("Call value", "$0.0018", "Interval unreliable: 85 ITM paths", 219.5),
        ("Put value", "$19.5097", "Interval unreliable: 85 OTM paths", 223.1),
    ],
)
def test_card_width_estimate_covers_the_delta_line(label, value, delta, measured):
    # Never narrower than the card (its delta would be cut with an ellipsis), and within 2 px of it.
    assert measured <= components._card_width(label, value, delta=delta) <= measured + 2


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
    # The interval column says why there is no interval: no path finished in the money.
    assert list(table.value["95% interval"]) == [
        "Interval unreliable: 0 of 10,000 paths finish in the money",
        "Interval unreliable: 0 of 10,000 paths finish out of the money",
    ]


def _shown(text: str) -> str:
    """A card's delta line as the browser shows it (Markdown escapes removed)."""
    return text.replace("\\$", "$")


def test_monte_carlo_price_cards_show_the_95_percent_interval(new_app):
    # The cards showed the standard error only ("STD. ERROR $0.0088" and "95% CI = price ± 1.96 SE"), so the user had
    # to work the interval out; the diagnostics table far below had it.
    at = new_app(model="Monte Carlo").run()
    _ok(at)
    res = monte_carlo.price_european(100, 100, 1, 0.05, 0.2, 0, n_paths=100_000, seed=42)
    cards = {m.label: m for m in at.metric}
    table = next(d for d in at.dataframe if "Error / SE" in d.value.columns).value.set_index("Option")
    for kind, r in res.items():
        assert r.reliable
        low, high = r.price - 1.959963984540054 * r.std_error, r.price + 1.959963984540054 * r.std_error
        card = cards[f"{kind.capitalize()} value"]
        assert _shown(card.delta) == f"${low:.4f} – ${high:.4f}, 95%"
        # Grey and without an arrow: a caption, not a change.
        assert card.proto.color == card.proto.MetricColor.GRAY
        assert card.proto.direction == card.proto.MetricDirection.NONE
        assert table.loc[kind.capitalize(), "95% interval"] == f"{low:.4f} – {high:.4f}"
        assert f"Standard error ${r.std_error:.4f}" in _shown(card.proto.help)
    assert _shown(cards["Call value"].delta) == "$10.4237 – $10.4582, 95%"  # the panel's example
    assert not cards["Forward price"].delta and not cards["Std. error"].delta


def test_monte_carlo_price_cards_flag_an_unreliable_interval(new_app):
    # With about 100 of 100,000 paths in the money (a 30-day call struck at 120), price ± 1.96 SE covers the value in
    # about 94% of runs, and in 88% with 10 such paths: the cards said "95% CI" whenever the standard error was finite.
    # The put on the same draws, deep in the money, is just as uncertain: with the control variate the two estimates
    # differ by a constant.
    at = new_app(model="Monte Carlo", strike=120, days=30).run()
    _ok(at)
    res = monte_carlo.price_european(100, 120, 30 / 365, 0.05, 0.2, 0, n_paths=100_000, seed=42)
    n_itm = res["call"].n_itm
    assert 0 < n_itm < monte_carlo.RELIABLE_PATHS and not res["call"].reliable and not res["put"].reliable
    cards = {m.label: m for m in at.metric}
    table = next(d for d in at.dataframe if "Error / SE" in d.value.columns).value.set_index("Option")
    # The card says it in short, so that the four cards stay on one line at 1440 px; its tooltip and the diagnostics
    # table say it in full.
    expected = {"Call": ("ITM", "in"), "Put": ("OTM", "out of")}
    for option, (scarce, side) in expected.items():
        card, sentence = cards[f"{option} value"], f"{n_itm} of 100,000 paths finish {side} the money"
        assert _shown(card.delta) == f"Interval unreliable: {n_itm} {scarce} paths"
        assert table.loc[option, "95% interval"] == f"Interval unreliable: {sentence}"
        assert _shown(card.proto.help) == (
            f"Interval unreliable: {sentence}. Standard error ${res['call'].std_error:.4f}. The 95% interval, price ± "
            f"1.96 SE, is not shown: with fewer than 300 paths finishing {side} the money it covers the value in less "
            "than 95% of runs (88% with 10 such paths, 92% with 30, 94% with 100)."
        )
    # The prices and the standard error are still shown, but not the interval's formula, which invited the user to
    # work out by hand the interval the cards withhold.
    assert _dollars(cards["Call value"].value) == pytest.approx(res["call"].price, abs=5e-5)
    assert _dollars(cards["Std. error"].value) == pytest.approx(res["call"].std_error, abs=5e-5)
    assert not any("1.96" in c.value for c in at.caption if c.value.startswith("Standard error"))
    assert any(f"fewer than {monte_carlo.RELIABLE_PATHS} paths" in c.value for c in at.caption)


def test_monte_carlo_interval_withheld_for_too_few_independent_samples():
    # The sidebar draws 10,000 paths or more (5,000 antithetic pairs, the fewest for which an interval is shown), but
    # the text says why for any result: here 500 pairs, with thousands of paths on each side of the strike.
    res = monte_carlo.price_european(100, 100, 1, 0.05, 0.2, 0, n_paths=1_000, seed=1)["call"]
    assert not res.reliable and min(res.n_itm, res.n_paths - res.n_itm) >= monte_carlo.RELIABLE_PATHS
    assert tab_pricing.mc_interval(res, short=True) == "Interval unreliable: too few paths"
    assert tab_pricing.mc_interval(res, dollars=False) == "Interval unreliable: 1,000 paths, fewer than 10,000"
    assert tab_pricing._mc_note(res).endswith(
        "is not shown: with fewer than 5,000 independent samples (10,000 paths in antithetic pairs) the standard error "
        "is itself too uncertain, and the interval covers the value in less than 95% of runs (93 to 94% with 500 "
        "antithetic pairs and the control variate)."
    )
    plain = monte_carlo.price_european(100, 100, 1, 0.05, 0.2, 0, n_paths=1_000, seed=1, antithetic=False)["call"]
    assert tab_pricing.mc_interval(plain) == "Interval unreliable: 1,000 paths, fewer than 5,000"
    assert "(5,000 paths) the standard error" in tab_pricing._mc_note(plain)


def test_greek_names_get_a_column_wide_enough_for_the_longest(app):
    # The grid sized the column from the first rows it drew and ignored "Epsilon (dividend rho) ε" as an outlier,
    # so the name was cut at 1,280 px and on phones ("Epsilon (di", and "Vomma (vo" with the higher-order Greeks).
    table = next(d for d in app.dataframe if "Greek" in d.value.columns)
    assert json.loads(table.proto.columns)["Greek"]["width"] == "medium"


# --- One-day decay beside theta / 365 -----------------------------------------------------------------------------

#: The panel's 1-day SPX-like contract (docs/reviews/panel-v3.0.md, market maker, finding 3).
SPX_1D = {"spot": 5800, "strike": 5800, "rate": 4.3, "div": 1.3, "vol": 18, "days": 1}
DAY = 1 / 365


def _greeks(at: AppTest):
    """The Greeks table, indexed by the Greek's name."""
    return next(d.value for d in at.dataframe if "Greek" in d.value.columns).set_index("Greek")


def _first(names: list[str], a: str, b: str) -> str:
    """Which of the adjacent rows ``a`` and ``b`` comes first."""
    assert abs(names.index(a) - names.index(b)) == 1, names
    return a if names.index(a) < names.index(b) else b


def test_a_one_day_at_the_money_put_decays_by_its_whole_premium(new_app):
    # The panel: worth 21.56, theta -10.66 a day, and the whole premium lost by expiry tomorrow.
    at = new_app(**SPX_1D).run()
    _ok(at)
    table = _greeks(at)
    assert _first(list(table.index), "Decay to expiry", "Theta Θ") == "Decay to expiry"  # the headline
    put = bs.price(5800, 5800, DAY, 0.043, 0.18, 0.013, "put")
    assert _dollars(_metrics(at)["Put value"]) == pytest.approx(21.56, abs=0.005)
    assert table.loc["Decay to expiry", "Put"] == pytest.approx(-put, rel=1e-12)  # intrinsic value 0 at the money
    assert table.loc["Theta Θ", "Put"] == pytest.approx(-10.66, abs=0.005)
    assert table.loc["Decay to expiry", "Unit"] == "until expiry, by repricing"
    # The chain says the same at its at-the-money row.
    chain = next(d.value for d in at.dataframe if "Strike" in d.value.columns).set_index("Strike")
    assert chain.loc[5800.0, "Put decay to expiry"] == pytest.approx(-put, rel=1e-12)
    assert "Call 1-day decay" not in chain.columns


def test_a_one_day_out_of_the_money_put_decays_by_less_than_theta_says(new_app):
    # The panel: the 5,750 put shows theta -6.97 a day, against a true decay of -5.19 (its whole value).
    at = new_app(**(SPX_1D | {"strike": 5750})).run()
    table = _greeks(at)
    assert table.loc["Decay to expiry", "Put"] == pytest.approx(-5.19, abs=0.005)
    assert table.loc["Theta Θ", "Put"] == pytest.approx(-6.97, abs=0.005)
    # The in-the-money call ends at its intrinsic value, 50.
    call = bs.price(5800, 5750, DAY, 0.043, 0.18, 0.013, "call")
    assert table.loc["Decay to expiry", "Call"] == pytest.approx(50 - call, rel=1e-12)


@pytest.mark.parametrize(("days", "first"), [(45, "Theta Θ"), (3, "Theta Θ"), (2, "1-day decay")])
def test_the_decay_leads_the_table_only_beyond_ten_percent(new_app, days, first):
    # At the money (spot = strike = 100, 20 % volatility, 5 % rate) theta / 365 and the 1-day decay differ by about
    # under 1 % at 45 days, 8.7 % (call) and 9.7 % (put) at 3 days, 14 % and 15 % at 2 days.
    at = new_app(days=days).run()
    _ok(at)
    table = _greeks(at)
    assert _first(list(table.index), "Theta Θ", "1-day decay") == first
    T = days * DAY
    for kind in ("call", "put"):
        decay = bs.price(100, 100, T - DAY, 0.05, 0.2, 0, kind) - bs.price(100, 100, T, 0.05, 0.2, 0, kind)
        assert table.loc["1-day decay", kind.capitalize()] == pytest.approx(decay, rel=1e-12)
    assert any("it is listed first when it differs from the theta row" in c.value for c in at.caption)


def test_the_decay_follows_the_american_engine(new_app):
    at = new_app(days=7, model="American (ALO)", div=3).run()
    _ok(at)
    table = _greeks(at)
    for kind in ("call", "put"):
        later = american.price(100, 100, 6 * DAY, 0.05, 0.2, 0.03, option_type=kind)
        now = american.price(100, 100, 7 * DAY, 0.05, 0.2, 0.03, option_type=kind)
        assert table.loc["1-day decay", kind.capitalize()] == pytest.approx(later - now, rel=1e-12)
        assert _dollars(_metrics(at)[f"{kind.capitalize()} value"]) == pytest.approx(now, abs=5e-5)
    # Within a day of expiry: the intrinsic value (zero at the money) minus the engine's value.
    at = new_app(days=1, model="American (ALO)").run()
    put = american.price(100, 100, DAY, 0.05, 0.2, 0.0, option_type="put")
    assert _greeks(at).loc["Decay to expiry", "Put"] == pytest.approx(-put, rel=1e-12)


def _tree_table(at: AppTest):
    """The tree's Greeks under Model diagnostics, indexed by option."""
    return next(d.value for d in at.dataframe if "Div. rho / 1%" in d.value.columns).set_index("Option")


#: The trees one day shorter than a 7-day 501-step tree: Leisen-Reimer keeps its steps; Cox-Ross-Rubinstein keeps
#: the time step, 501 x 6/7 = 429 3/7 steps, interpolated between the odd counts 429 and 431 (3/14 of the way).
LATER_TREES = {"Leisen-Reimer": ((501, 1.0),), "Cox-Ross-Rubinstein": ((429, 11 / 14), (431, 3 / 14))}


@pytest.mark.parametrize("lattice", list(LATER_TREES))
def test_the_tree_greeks_add_the_trees_own_decay(new_app, lattice):
    at = new_app(days=7, model="Binomial tree", tree=lattice, style="American").run()
    _ok(at)
    method = "lr" if lattice == "Leisen-Reimer" else "crr"
    tree = _tree_table(at)
    for kind in ("call", "put"):
        later = math.fsum(
            weight * binomial.price(100, 100, 6 * DAY, 0.05, 0.2, 0, kind, steps, True, method)
            for steps, weight in LATER_TREES[lattice]
        )
        now = binomial.price(100, 100, 7 * DAY, 0.05, 0.2, 0, kind, 501, True, method)
        assert tree.loc[kind.capitalize(), "1-day decay"] == pytest.approx(later - now, rel=1e-12)
        # The Greeks table stays Black-Scholes-Merton's for European exercise, its decay too.
        closed = bs.price(100, 100, 6 * DAY, 0.05, 0.2, 0, kind) - bs.price(100, 100, 7 * DAY, 0.05, 0.2, 0, kind)
        assert _greeks(at).loc["1-day decay", kind.capitalize()] == pytest.approx(closed, rel=1e-12)
    # At the money with a week left the tree's decay and theta / 365 are within 10 %: theta stays first.
    names = list(tree.columns)
    assert names.index("1-day decay") == names.index("Theta / day") + 1
    captions = [c.value for c in at.caption]
    interpolated = "interpolated between trees with 429 and 431 steps to this tree's time step (429.43 steps)"
    assert any(interpolated in c for c in captions) is (method == "crr")
    assert any("one day shorter with the same steps" in c for c in captions) is (method == "lr")
    assert not any("comes first" in c for c in captions)
    note = "It is not the decay of the American value in the price cards, which the tree's Greeks give."
    assert any(note in c for c in captions)


def test_an_american_put_exercised_at_once_does_not_decay_on_the_tree(new_app):
    # The review's case: a 7-day put struck at 110 on 100 is exercised at once on the American tree, worth exactly
    # 10 today and tomorrow, while the European put of the Greeks table gains 1.5 cents a day. The caption says the
    # table's decay is not the American value's.
    at = new_app(days=7, strike=110, model="Binomial tree", style="American").run()
    _ok(at)
    assert _dollars(_metrics(at)["Put value"]) == 10.0
    tree = _tree_table(at)
    assert tree.loc["Put", "1-day decay"] == 0.0 and tree.loc["Put", "Theta / day"] == 0.0
    european = bs.price(100, 110, 6 * DAY, 0.05, 0.2, 0, "put") - bs.price(100, 110, 7 * DAY, 0.05, 0.2, 0, "put")
    assert european == pytest.approx(0.0149, abs=5e-5)
    assert _greeks(at).loc["1-day decay", "Put"] == pytest.approx(european, rel=1e-12)
    assert any("It is not the decay of the American value in the price cards" in c.value for c in at.caption)


@pytest.mark.parametrize("lattice", list(LATER_TREES))
@pytest.mark.parametrize(("days", "leads"), [(2, True), (3, False), (45, False)])
def test_the_tree_table_puts_its_decay_first_beyond_ten_percent(new_app, lattice, days, leads):
    # The Greeks table's rule on the tree's own figures: at the money (100, 20 %, 5 %) theta / 365 and the 1-day
    # decay are 14 % and 15 % apart at 2 days (call, put), 8.7 % and 9.7 % at 3 days, under 1 % at 45 days; the
    # 501-step trees' thetas are within 0.07 % and their decays within 0.1 % of the closed form here, well inside
    # the 0.3 % left at 3 days.
    at = new_app(days=days, model="Binomial tree", tree=lattice).run()
    _ok(at)
    tree = _tree_table(at)
    for kind in ("Call", "Put"):
        theta, decay = tree.loc[kind, "Theta / day"], tree.loc[kind, "1-day decay"]
        assert bool(abs(decay - theta) > 0.1 * max(abs(decay), abs(theta))) is leads, kind
    names = list(tree.columns)
    assert names.index("1-day decay") - names.index("Theta / day") == (-1 if leads else 1)
    first = "The 1-day decay comes first: it differs from theta / 365 by more than 10% of the larger"
    assert any(first in c.value for c in at.caption) is leads


def test_the_tree_keeps_its_time_step_with_a_whole_number_of_steps(new_app):
    # 500 steps over 10 days: 450 steps over 9 days have the same time step exactly, so one tree is enough.
    at = new_app(days=10, steps=500, model="Binomial tree", tree="Cox-Ross-Rubinstein").run()
    _ok(at)
    later = binomial.price(100, 100, 9 * DAY, 0.05, 0.2, 0, "put", 450, False, "crr")
    now = binomial.price(100, 100, 10 * DAY, 0.05, 0.2, 0, "put", 500, False, "crr")
    assert _tree_table(at).loc["Put", "1-day decay"] == pytest.approx(later - now, rel=1e-12)
    assert any("one day shorter with 450 steps, this tree's time step," in c.value for c in at.caption)


def test_the_crr_decay_uses_the_shorter_time_step_where_the_longer_has_no_lattice(new_app):
    # 5 % volatility, rate 50 %, 1,828 days, 501 steps: the tree is arbitrage-free only just (|r| dt = 0.0049982
    # against sigma sqrt(dt) = 0.0049991). Its time step over 1,827 days is 500.73 steps; the 499-step tree's longer
    # step has no arbitrage-free lattice, so the 501-step tree alone values the option a day later.
    at = new_app(days=1828, vol=5, rate=50, model="Binomial tree", tree="Cox-Ross-Rubinstein").run()
    _ok(at)
    with pytest.raises(ValueError, match="not arbitrage-free"):
        binomial.price(100, 100, 1827 * DAY, 0.5, 0.05, 0, "call", 499, False, "crr")
    later = binomial.price(100, 100, 1827 * DAY, 0.5, 0.05, 0, "call", 501, False, "crr")
    now = binomial.price(100, 100, 1828 * DAY, 0.5, 0.05, 0, "call", 501, False, "crr")
    assert _tree_table(at).loc["Call", "1-day decay"] == pytest.approx(later - now, rel=1e-12)
    shorter = 1 - 501 * (1827 / 1828) / 501
    assert f"{shorter:.2%}" == "0.05%"
    note = (
        "one day shorter with 501 steps, a time step 0.05% shorter than this tree's (the next count down has no "
        "arbitrage-free lattice here)"
    )
    assert any(note in c.value for c in at.caption)


def test_the_tree_decays_to_its_intrinsic_value_within_a_day(new_app):
    at = new_app(days=1, strike=95, model="Binomial tree", style="American").run()
    _ok(at)
    tree = next(d.value for d in at.dataframe if "Div. rho / 1%" in d.value.columns).set_index("Option")
    for kind, intrinsic in (("call", 5.0), ("put", 0.0)):
        now = binomial.price(100, 95, DAY, 0.05, 0.2, 0, kind, 501, True, "lr")
        assert tree.loc[kind.capitalize(), "Decay to expiry"] == pytest.approx(intrinsic - now, rel=1e-12, abs=1e-15)
    note = "the decay to expiry is the intrinsic value at expiry minus the tree value"
    assert any(note in c.value for c in at.caption)
    # Within a day of expiry the decay to expiry leads the tree's table too.
    names = list(tree.columns)
    assert names.index("Decay to expiry") == names.index("Theta / day") - 1
    assert any("The decay to expiry comes first: the option expires within the day." in c.value for c in at.caption)


def test_monte_carlo_uses_the_closed_form_decay(new_app):
    at = new_app(days=1, model="Monte Carlo").run()
    _ok(at)
    call = bs.price(100, 100, DAY, 0.05, 0.2, 0, "call")
    assert _greeks(at).loc["Decay to expiry", "Call"] == pytest.approx(-call, rel=1e-12)
    assert any("by Black-Scholes-Merton, whose value the simulation estimates" in c.value for c in at.caption)


def test_the_sensitivity_explorer_plots_the_decay_against_the_days_left(app):
    app.selectbox(key="sens_quantity").set_value("1-day decay").run()
    app.selectbox(key="sens_axis").set_value("Days to expiry").run()
    _ok(app)
    chart = _chart(app, "sensitivity_chart")
    days = _values(chart["data"][0]["x"])
    call = _values(chart["data"][0]["y"])
    assert days[0] == 1 and call[0] == pytest.approx(-bs.price(100, 100, DAY, 0.05, 0.2, 0, "call"), rel=1e-12)
    T = days[-1] * DAY
    expected = bs.price(100, 100, T - DAY, 0.05, 0.2, 0, "call") - bs.price(100, 100, T, 0.05, 0.2, 0, "call")
    assert call[-1] == pytest.approx(expected, rel=1e-12)
    assert chart["layout"]["yaxis"]["title"]["text"] == "1-day decay ($)"


def test_the_theta_view_draws_the_decay_beside_theta(new_app):
    # Theta / 365 solid and the repriced day dashed, each in its option's colour, against the spot.
    at = new_app(days=3).run()
    at.selectbox(key="sens_quantity").set_value("Theta / day").run()
    _ok(at)
    chart = _chart(at, "sensitivity_chart")
    names = [trace["name"] for trace in chart["data"]]
    assert names == ["Theta / 365 · Call", "Theta / 365 · Put", "1-day decay · Call", "1-day decay · Put"]
    assert [trace["line"].get("dash") for trace in chart["data"]] == ["solid", "solid", "dash", "dash"]
    assert chart["data"][0]["line"]["color"] == chart["data"][2]["line"]["color"] != chart["data"][1]["line"]["color"]
    assert chart["layout"]["yaxis"]["title"]["text"] == "Theta / 365 and 1-day decay ($)"
    spots = _values(chart["data"][2]["x"])
    T = 3 * DAY
    for trace, kind in zip(chart["data"], ("call", "put") * 2, strict=True):
        y = _values(trace["y"])
        if trace["line"]["dash"] == "solid":
            expected = bs.greeks(spots, 100, T, 0.05, 0.2, 0, kind)["theta"] / 365
        else:
            expected = bs.price(spots, 100, T - DAY, 0.05, 0.2, 0, kind) - bs.price(spots, 100, T, 0.05, 0.2, 0, kind)
        np.testing.assert_allclose(y, expected, rtol=1e-12, atol=1e-15)


def test_within_a_day_the_explorer_names_the_decay_to_expiry(new_app):
    # Against spot every point has the sidebar's one day left; against the days left the 1-day decay keeps its name.
    at = new_app(days=1).run()
    at.selectbox(key="sens_quantity").set_value("1-day decay").run()
    _ok(at)
    assert _chart(at, "sensitivity_chart")["layout"]["yaxis"]["title"]["text"] == "Decay to expiry ($)"
    at.selectbox(key="sens_quantity").set_value("Theta / day").run()
    chart = _chart(at, "sensitivity_chart")
    assert [trace["name"] for trace in chart["data"]][2:] == ["Decay to expiry · Call", "Decay to expiry · Put"]
    assert chart["layout"]["yaxis"]["title"]["text"] == "Theta / 365 and decay to expiry ($)"
    at.selectbox(key="sens_axis").set_value("Days to expiry").run()
    assert _chart(at, "sensitivity_chart")["layout"]["yaxis"]["title"]["text"] == "Theta / 365 and 1-day decay ($)"


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


def test_iv_tab_says_when_a_quote_has_no_time_value(new_app):
    # The review's 2-day call struck at 60 with the spot at 100 and a rate of 4 %: at the sidebar's 20 % the model
    # value is the discounted intrinsic value 100 − 60·e^(−0.04·2/365) = 40.0131 to the last digit. The tab used to
    # show 0 % for it (and 94.5 % for 1e-13 more); now it says the quote has no time value and shows no volatility.
    # The value it names is the discounted intrinsic value, not the max(S − K, 0) = $40 of the Time value card.
    from options_analyzer.implied_vol import implied_volatility, no_arbitrage_bounds
    from options_analyzer.ui.format import pct

    at = new_app(tab="IV", strike="60", days="2", rate="4").run()
    _ok(at)
    lower, _ = no_arbitrage_bounds(100, 60, 2 / 365, 0.04, 0.0, "call")
    # Black-Scholes-Merton gives one ulp (7.1e-15) above the bound here: far within the margin 1e-12·100.
    assert _quote(at).value - lower == pytest.approx(0.0, abs=1e-13) and lower == pytest.approx(40.0131, abs=5e-5)
    [warning] = [w.value for w in at.warning]
    assert warning == (
        "No implied volatility: this quote has no time value. It is within \\$0.00000000010 (one part in 10¹² of the "
        "larger of spot and strike) of the no-arbitrage lower bound, the discounted intrinsic value "
        "S·e^(−qT) − K·e^(−rT) = \\$40.0131, which is the option's value at zero volatility, and every volatility from "
        "zero up to some level prices the option that close to it, so none can be singled out. The model value at the "
        "sidebar volatility is such a quote: the option is that deep in the money. For a volatility at this strike, "
        "invert the out-of-the-money option's quote: it is all time value."
    )
    assert "Implied volatility" not in _metrics(at) and not at.get("plotly_chart")
    # Half a cent of time value is a volatility (230 %).
    _quote(at).set_value(lower + 0.005).run()
    assert not at.exception and not at.error
    assert _metrics(at)["Implied volatility"] == pct(implied_volatility(lower + 0.005, 100, 60, 2 / 365, 0.04), 3)
    assert _metrics(at)["Implied volatility"] == "230.024%"
    # Out of the money, a quote of zero has no time value either.
    next(s for s in at.segmented_control if s.key == "iv_type").set_value("put").run()
    _quote(at).set_value(0.0).run()
    _ok(at)
    assert [w.value for w in at.warning] == [
        "No implied volatility: an out-of-the-money option is worth only its time value, and a quote of \\$0.0000 "
        "has none."
    ]
    # A month out the model value of the put struck at 120 has time value ($0.0013), and a typed quote at its
    # discounted intrinsic value has none.
    at = new_app(tab="IV", strike="120", days="30", rate="0").run()
    next(s for s in at.segmented_control if s.key == "iv_type").set_value("put").run()
    _quote(at).set_value(20.0).run()
    _ok(at)
    [warning] = [w.value for w in at.warning]
    assert "the discounted intrinsic value K·e^(−rT) − S·e^(−qT) = \\$20.0000," in warning
    assert "This is typical deep in the money close to expiry." in warning


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


def _model_check(at: AppTest) -> tuple[str, str, float]:
    """The Model check card's value, the full value in its help and the distance from the quote the help states."""
    check = next(m for m in at.metric if m.label == "Model check")
    match = re.fullmatch(
        r"Model value at the implied volatility \(should equal the quote\): \\\$([0-9,.]+), "
        r"(?:equal to the quote|\\\$([0-9.e+-]+) from the quote)\.",
        check.proto.help,
    )
    assert match, check.proto.help
    return check.value, match[1], float(match[2] or 0.0)


def test_iv_model_check_shows_the_quote_precision_and_the_full_value_in_its_help(iv_app):
    # Six decimals did not fit the card; the full value and its distance from the quote are in the help. The
    # default quote is the model value at the sidebar's 20 %, which Let's Be Rational inverts to the last bit, so
    # the help usually reads "equal to the quote."; another quote is repriced to within rounding.
    value, full, distance = _model_check(iv_app)
    assert value == "$10.4506" and full == f"{bs.price(100, 100, 1, 0.05, 0.2):,.6f}"
    assert distance <= 1e-12
    _quote(iv_app).set_value(12.3456).run()
    _ok(iv_app)
    value, full, distance = _model_check(iv_app)
    assert value == "$12.3456" and full == "12.345600"
    assert distance <= 1e-12  # measured 1.8e-15: bs.price at the implied volatility, not the quote's own digits


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
    def ancestors(node, key):
        for child in getattr(node, "children", {}).values():
            if str(getattr(getattr(child, "proto", None), "id", "")).endswith(f"-{key}"):
                return [node]
            if (found := ancestors(child, key)) is not None:
                return [*found, node]
        return None

    for kind in ("call", "put"):
        assert _chart(app, f"heatmap_{kind}")["data"][0]["showscale"] is False
        # charts.show draws each chart in its own chart_<key> container; the block around that is the expander.
        box, around = ancestors(app.main, f"heatmap_{kind}")[:2]
        assert len(box.children) == 1 and around.type == "expander"  # not one of two columns
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
