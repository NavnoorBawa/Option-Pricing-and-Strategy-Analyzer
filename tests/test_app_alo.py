"""The American engine in the app: the sidebar model and its scheme, the price cards, the Greeks, the engine's
diagnostics and exercise boundary, the header badge, the link, the fallback to the tree and the IV tab
(ui/model_config.py, ui/tab_pricing.py, ui/tab_iv.py, ui/header.py, ui/app.py), and the two library functions
behind them: ``american.greeks`` and ``implied_vol.implied_volatility_alo``."""

from __future__ import annotations

import base64
import dataclasses
import itertools
import json
import pickle
import re
import urllib.parse

import numpy as np
import pytest
from streamlit.proto.WidgetStates_pb2 import WidgetState
from streamlit.testing.v1 import AppTest
from test_app_header import _EMOJI, _SHORTCODE, _strings

from options_analyzer import american, binomial, implied_vol
from options_analyzer import black_scholes as bs
from options_analyzer.ui import header, tab_pricing
from options_analyzer.ui.app import TABS
from options_analyzer.ui.model_config import MODELS, ModelConfig

ALO = MODELS["alo"]
#: The default contract of the sidebar.
DEFAULT = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "q": 0.0}
GREEKS = ("delta", "gamma", "theta", "vega", "rho", "epsilon")


def _ok(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def _run(new_app, **query) -> AppTest:
    at = new_app(model=ALO, **query).run()
    _ok(at)
    return at


def _nodes(node):
    for child in getattr(node, "children", {}).values():
        yield child
        yield from _nodes(child)


def _metrics(at: AppTest) -> dict:
    return {m.label: m for m in at.metric}


def _table(at: AppTest, column: str):
    return next(d.value for d in at.dataframe if column in d.value.columns)


def _chart(at: AppTest, key: str) -> dict:
    return json.loads(next(c for c in at.get("plotly_chart") if c.proto.id.endswith(f"-{key}")).proto.spec)


def _values(data) -> np.ndarray:
    if isinstance(data, dict):
        return np.frombuffer(base64.b64decode(data["bdata"]), dtype=data["dtype"])
    return np.asarray(data)


def _badge(at: AppTest):
    return next(m for m in at.markdown if m.value.startswith(":gray-badge["))


def _price(kind: str, scheme: str = "accurate", **inputs) -> float:
    c = DEFAULT | inputs
    return american.price(c["S"], c["K"], c["T"], c["r"], c["sigma"], c["q"], option_type=kind, scheme=scheme)


def _open(at: AppTest, tab: str) -> AppTest:
    """Open ``tab`` as a click on it does (see tests/test_app.py)."""
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


# --- The sidebar model ---------------------------------------------------------------------------------------------


def test_the_engine_is_a_model_beside_the_others_with_a_scheme(new_app):
    at = _run(new_app)
    assert at.sidebar.radio(key="model").options == list(MODELS.values())  # the existing three first, unchanged
    assert at.sidebar.radio(key="model").value == "alo"
    scheme = at.sidebar.segmented_control(key="scheme")
    assert scheme.options == ["Fast", "Accurate", "High"] and scheme.value == "accurate"
    # It says that it prices American exercise; the scheme's help gives each scheme's error, in dollars.
    assert any(c.value.startswith("American exercise: the engine solves") for c in at.sidebar.caption)
    assert "accurate \\$0.000005" in scheme.proto.help
    # The tree's and the simulation's settings are not drawn.
    assert not at.sidebar.slider and not at.sidebar.toggle


@pytest.mark.parametrize("scheme", ["fast", "accurate", "high"])
def test_the_price_cards_are_the_engines_values(new_app, scheme):
    at = _run(new_app, scheme=scheme.capitalize(), div="2.5", strike="105")
    inputs = {"K": 105.0, "q": 0.025}
    cards = _metrics(at)
    engine = _table(at, "Premium").set_index("Option")
    for kind in ("call", "put"):
        value = _price(kind, scheme, **inputs)
        assert engine.loc[kind.capitalize(), "American"] == pytest.approx(value, abs=1e-12)
        assert cards[f"{kind.capitalize()} value"].value == f"${value:,.4f}"
        european = bs.price(100, 105, 1, 0.05, 0.2, 0.025, kind)
        assert engine.loc[kind.capitalize(), "European"] == pytest.approx(european, abs=1e-12)
        assert engine.loc[kind.capitalize(), "Premium"] == pytest.approx(value - european, abs=1e-12)
        assert engine.loc[kind.capitalize(), "Method"] == "Spectral collocation"
    premium = max(_price(k, scheme, **inputs) - bs.price(100, 105, 1, 0.05, 0.2, 0.025, k) for k in ("call", "put"))
    assert cards["Exercise premium"].value == f"${premium:,.4f}"
    assert "over the European value" in cards["Put value"].proto.help


def test_a_call_without_dividends_equals_black_scholes_merton_and_the_app_says_why(new_app):
    at = _run(new_app)
    engine = _table(at, "Premium").set_index("Option")
    call = bs.price(100, 100, 1, 0.05, 0.2, 0.0, "call")
    assert engine.loc["Call", "American"] == pytest.approx(call, abs=1e-12)
    assert engine.loc["Call", "Method"] == "European value (never exercised early)"
    why = "The call equals its Black-Scholes-Merton value: with no dividend yield, exercising a call early never pays"
    assert any(why in c.value for c in at.caption)
    assert "with no dividend yield" in _metrics(at)["Call value"].proto.help
    # With a dividend yield the call has an exercise premium, and the caption no longer says it equals BSM.
    at.sidebar.number_input(key="div").set_value(3.0).run()
    _ok(at)
    assert not any(why in c.value for c in at.caption)
    assert _table(at, "Premium").set_index("Option").loc["Call", "Premium"] > 1e-4


def test_the_header_badge_names_the_engine_and_its_scheme(new_app):
    at = _run(new_app)
    assert _badge(at).value == ":gray-badge[ALO · accurate · American]"
    assert _badge(at).proto.help == (
        "Pricing model of the price cards: Andersen-Lake-Offengenden engine · accurate scheme · American."
    )
    at = _run(new_app, scheme="High")
    assert _badge(at).value == ":gray-badge[ALO · high · American]"
    assert header.model_badge(ModelConfig(kind="alo", scheme="fast")) == "ALO · fast · American"


# --- The link ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("asked", "kind"),
    [
        ("ALO", "alo"), ("alo", "alo"), ("American (ALO)", "alo"), ("american-alo", "alo"), ("BSM", "bsm"),
        ("MC", "monte_carlo"), ("tree", "binomial"), ("Binomial tree", "binomial"), ("Monte Carlo", "monte_carlo"),
        ("Black-Scholes-Merton", "bsm"), ("binomial_tree", "binomial"),
    ],
)  # fmt: skip
def test_a_link_names_the_model_by_its_label_or_short_name(new_app, asked, kind):
    # Every label a link held before still works; ALO and the badge's other short names are aliases. The link is
    # rewritten to the label, so that a copy of it is the app's own.
    at = new_app(model=asked).run()
    _ok(at)
    assert at.sidebar.radio(key="model").value == kind
    assert at.query_params.get("model") == (None if kind == "bsm" else MODELS[kind])  # the default needs none
    assert not at.warning


def _block(at: AppTest, key: str):
    return next(n for n in _nodes(at.main) if str(getattr(getattr(n, "proto", None), "id", "")).endswith(f"-{key}"))


def test_the_share_link_reopens_the_same_engine_and_scheme(new_app):
    at = new_app(model="alo", scheme="high").run()
    _ok(at)
    assert at.sidebar.segmented_control(key="scheme").value == "high"
    [code] = _block(at, "desk_share").code
    assert code.value == "?model=American+%28ALO%29&scheme=High"
    # The link it shares opens the same model, scheme and values.
    again = new_app(**dict(urllib.parse.parse_qsl(code.value.removeprefix("?")))).run()
    _ok(again)
    assert again.sidebar.segmented_control(key="scheme").value == "high"
    assert _badge(again).value == _badge(at).value
    assert {m.label: m.value for m in again.metric} == {m.label: m.value for m in at.metric}


@pytest.mark.parametrize(
    ("query", "note"),
    [
        ({"model": "Heston"}, "This link asks for a pricing model named `Heston`, which does not exist, so "
                              "Black-Scholes-Merton is used."),
        ({"model": "ALO", "scheme": "ultra"}, "This link asks for a scheme named `ultra`, which does not exist, so "
                                              "the accurate scheme is used."),
    ],
)  # fmt: skip
def test_a_link_to_a_model_or_scheme_that_does_not_exist_says_so(new_app, query, note):
    at = new_app(**query).run()
    _ok(at)
    assert [w.value for w in at.warning] == [note]
    assert "scheme" not in at.query_params


# --- Greeks --------------------------------------------------------------------------------------------------------


def _differences(kind: str, **inputs) -> dict[str, float]:
    """Central differences of american.price with the bumps that american.greeks documents (GREEK_BUMPS)."""
    c = DEFAULT | inputs
    S, T, r, sigma, q = c["S"], c["T"], c["r"], c["sigma"], c["q"]

    def v(**bumped):
        return _price(kind, **(c | bumped))

    hs, ht, hv, hr = 1e-4 * S, 1e-4 * T, 1e-4, 1e-4
    return {
        "delta": (v(S=S + hs) - v(S=S - hs)) / (2 * hs),
        "gamma": (v(S=S + hs) - 2 * v() + v(S=S - hs)) / hs**2,
        "theta": -(v(T=T + ht) - v(T=T - ht)) / (2 * ht),
        "vega": (v(sigma=sigma + hv) - v(sigma=sigma - hv)) / (2 * hv),
        "rho": (v(r=r + hr) - v(r=r - hr)) / (2 * hr),
        "epsilon": (v(q=q + hr) - v(q=q - hr)) / (2 * hr),
    }


def test_the_greeks_are_finite_differences_of_the_engine_in_the_tabs_units(new_app):
    at = _run(new_app, div="2")
    table = _table(at, "Greek").set_index("Greek")
    assert list(table.index) == [f"{bs.GREEKS[n].label} {bs.GREEKS[n].symbol}" for n in GREEKS]
    assert list(table["Unit"]) == [bs.GREEKS[n].unit for n in GREEKS]  # the units of the European table
    for kind in ("call", "put"):
        expected = _differences(kind, q=0.02)
        shown = dict(zip(GREEKS, table[kind.capitalize()], strict=True))
        for name in GREEKS:
            assert shown[name] == pytest.approx(expected[name] * bs.GREEKS[name].scale, rel=1e-9, abs=1e-12), name
    assert any(c.value.startswith("American Greeks of the sidebar engine, in the units shown") for c in at.caption)
    assert not [t for t in at.toggle if t.key == "higher_greeks"]  # higher orders are Black-Scholes-Merton's only


def test_the_european_call_gets_the_closed_form_greeks():
    # A call without dividends is European: its Greeks are the analytical ones, and its finite differences agree.
    g = american.greeks(100, 100, 1.0, 0.05, 0.2, 0.0, option_type="call")
    exact = bs.greeks(100, 100, 1.0, 0.05, 0.2, 0.0, "call")
    differences = _differences("call")
    assert g.basis == "closed form" and g.one_sided == ()
    for name in GREEKS:
        assert g[name] == pytest.approx(exact[name], rel=1e-12, abs=1e-15), name
        assert differences[name] == pytest.approx(exact[name], rel=1e-7, abs=1e-9), name


def test_the_greeks_read_like_a_dict_and_survive_the_cache():
    g = american.greeks(100, 100, 1.0, 0.05, 0.2, 0.02, option_type="put")
    assert list(g) == list(GREEKS) == list(american.GREEK_NAMES) and len(g) == 6
    assert dict(g) == {name: getattr(g, name) for name in GREEKS}
    assert g.basis == "differences" and g.one_sided == ()
    with pytest.raises(KeyError):
        g["vanna"]
    assert pickle.loads(pickle.dumps(g)) == g  # st.cache_data pickles what it returns


@pytest.mark.parametrize("spot", [100.0, 120.0])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_at_zero_rate_and_yield_rho_and_dividend_rho_are_black_scholes_mertons(kind, spot):
    # Both options are European at r = q = 0, but a 1 bp move one way makes early exercise pay (a lower rate or a
    # higher yield for the call, a higher rate or a lower yield for the put). A central difference averaged the two
    # slopes of that kink: the call's rho at S = 120 came out 5.9 % low and its dividend rho 7 %.
    g = american.greeks(spot, 100, 1.0, 0.0, 0.2, 0.0, option_type=kind)
    exact = bs.greeks(spot, 100, 1.0, 0.0, 0.2, 0.0, kind)
    assert g.basis == "closed form"
    for name in GREEKS:
        assert g[name] == pytest.approx(exact[name], rel=1e-12, abs=1e-15), name
    central = _differences(kind, S=spot, r=0.0)
    assert abs(central["rho"] / exact["rho"] - 1) > 0.01 and abs(central["epsilon"] / exact["epsilon"] - 1) > 0.01

    # The closed form is the slope on the European side: a one-sided difference there agrees.
    def one_sided(name: str, side: float) -> float:
        h = 1e-5

        def v(x: float) -> float:
            return _price(kind, S=spot, **{"r": 0.0, "q": 0.0, name: x})

        return side * (4 * v(side * h / 2) - 3 * v(0.0) - v(side * h)) / h

    european_side = {"call": {"r": 1.0, "q": -1.0}, "put": {"r": -1.0, "q": 1.0}}[kind]
    assert one_sided("r", european_side["r"]) == pytest.approx(exact["rho"], rel=1e-7)
    assert one_sided("q", european_side["q"]) == pytest.approx(exact["epsilon"], rel=1e-7)
    assert abs(one_sided("r", -european_side["r"]) / exact["rho"] - 1) > 0.02  # the other side's slope


def test_at_zero_rate_and_yield_the_table_is_black_scholes_mertons_and_says_one_sided(new_app):
    at = _run(new_app, rate="0")
    table = _table(at, "Greek").set_index("Greek")
    for kind in ("call", "put"):
        exact = bs.greeks(100, 100, 1.0, 0.0, 0.2, 0.0, kind)
        for name, shown in zip(GREEKS, table[kind.capitalize()], strict=True):
            assert shown == pytest.approx(exact[name] * bs.GREEKS[name].scale, rel=1e-12, abs=1e-15), (kind, name)
    [note] = [c.value for c in at.caption if c.value.startswith("American Greeks of the sidebar engine")]
    assert "The call and the put equal their Black-Scholes-Merton values, so their Greeks are the closed-form" in note
    assert note.endswith(tab_pricing.ALO_ZERO_CARRY_NOTE)
    assert "so rho and dividend rho are one-sided here: the slopes on the side where it does not." in note
    # The cards' caption says why both equal Black-Scholes-Merton, in words that hold at a rate of zero.
    assert any(
        "exercising a call early never pays (it would pay the strike early and gain no dividends)" in c.value
        for c in at.caption
    )
    # A rate of zero with a dividend yield: the call has a premium, the put alone is European, and nothing is
    # one-sided.
    at = _run(new_app, rate="0", div="2")
    [note] = [c.value for c in at.caption if c.value.startswith("American Greeks of the sidebar engine")]
    assert "The put equals its Black-Scholes-Merton value, so its Greeks are the closed-form ones." in note
    assert "one-sided" not in note


@pytest.mark.parametrize("rate", [-0.0005, -0.03])
def test_a_bump_into_another_regime_is_replaced_by_a_one_sided_difference(rate):
    # A call at a negative rate without dividends has one exercise boundary, but 1 bp of dividend yield lower it
    # has two, and the engine hands it to the tree: the central difference read the tree's value as part of the
    # slope. The dividend rho is taken on the side of the inputs, as fine one-sided differences of the engine find.
    S, K, T, sigma = 100.0, 100.0, 1.0, 0.2
    assert american.price_details(S, K, T, rate, sigma, -1e-4, option_type="call").fallback_reason == "double boundary"
    assert american.price_details(S, K, T, rate, sigma, 1e-4, option_type="call").method == "alo"
    g = american.greeks(S, K, T, rate, sigma, 0.0, option_type="call")
    assert g.basis == "differences" and g.one_sided == ("epsilon",)
    h = 2e-6
    form = american.price_details(S, K, T, rate, sigma, 0.0, option_type="call").fixed_point  # held, as in greeks

    def v(q: float) -> float:
        return american.price(S, K, T, rate, sigma, q, option_type="call", fixed_point=form)

    reference = (4 * v(h / 2) - 3 * v(0.0) - v(h)) / h
    assert g["epsilon"] == pytest.approx(reference, rel=2e-6)
    central = _differences("call", r=rate)["epsilon"]
    assert abs(central / reference - 1) > 1e-4  # what the one-sided difference corrects
    # Every other Greek is still the central difference.
    for name in ("delta", "gamma", "theta", "vega", "rho"):
        assert g[name] == pytest.approx(_differences("call", r=rate)[name], rel=1e-9), name


def test_the_app_names_a_one_sided_greek(new_app):
    at = _run(new_app, rate="-1")
    [note] = [c.value for c in at.caption if c.value.startswith("American Greeks of the sidebar engine")]
    assert (
        "The call's dividend rho is one-sided: a bump one way would change how the option is valued (a different "
        "exercise regime, the tree instead of the engine or the reverse, or exercise at once), so the difference is "
        "taken on the other side alone."
    ) in note
    assert "The put equals its Black-Scholes-Merton value, so its Greeks are the closed-form ones." in note


def test_an_option_exercised_at_once_has_the_greeks_of_its_exercise_value(new_app):
    # Below its critical spot the put is worth K - S: delta -1 and the rest zero. Differences of a value close to a
    # strike of 1e7 gave a delta of -1.00024 at a spot of 0.01.
    for S, K in ((60.0, 100.0), (0.01, 1e7)):
        details = american.price_details(S, K, 30 / 365, 0.05, 0.1, 0.0, option_type="put")
        assert details.method == "alo" and details.critical_spot >= S
        g = american.greeks(S, K, 30 / 365, 0.05, 0.1, 0.0, option_type="put")
        assert g.basis == "exercise value"
        assert dict(g) == {"delta": -1.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0, "epsilon": 0.0}
    call = american.greeks(150, 100, 1.0, 0.0, 0.2, 0.3, option_type="call")
    assert american.price_details(150, 100, 1.0, 0.0, 0.2, 0.3, option_type="call").critical_spot <= 150
    assert call.basis == "exercise value" and call["delta"] == 1.0 and call["gamma"] == 0.0
    at = _run(new_app, spot="60")
    table = _table(at, "Greek").set_index("Greek")
    assert list(table["Put"]) == [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    [note] = [c.value for c in at.caption if c.value.startswith("American Greeks of the sidebar engine")]
    assert (
        "The put is exercised at once at this spot (at or below its critical spot), so its Greeks are those of its "
        "exercise value K − S: delta −1, the others zero."
    ) in note


def test_greeks_at_extreme_moneyness_are_not_rounding_noise(new_app):
    # A put the engine values as European (r < 0 <= q) at a spot of 0.01 against a strike of 1e7: the second
    # difference of a value near 1e7 was rounding (a gamma of -1862.6 per $1). It now has the closed form.
    query = {"spot": "0.01", "strike": "10000000", "days": "30", "vol": "0.1", "rate": "-5", "div": "5"}
    engine = _table(_run(new_app, **query), "Greek").set_index("Greek")
    closed = _table(new_app(**query).run(), "Greek").set_index("Greek")  # Black-Scholes-Merton
    np.testing.assert_allclose(engine["Put"], closed["Put"], rtol=1e-12, atol=1e-15)
    assert engine.loc["Gamma Γ", "Put"] >= 0.0


def test_greeks_stay_in_range_across_extreme_inputs():
    # 432 options from a spot over strike of 1e-9 to 1e9, at volatilities of 0.1 % and 20 %: every delta lies in
    # [0, 1] (call) or [-1, 0] (put) and every gamma is non-negative. Central differences throughout put 36 of them
    # out of range, all at a spot over strike beyond 1e3 or 1e-3.
    grid = itertools.product(
        [0.01, 100.0, 1e7], [0.01, 100.0, 1e7], [1 / 365, 1.0], [-0.05, 0.0, 0.05], [0.001, 0.2], [0.0, 0.05]
    )
    for S, K, T, r, sigma, q in grid:
        for kind, (lo, hi) in (("call", (0.0, 1.0)), ("put", (-1.0, 0.0))):
            g = american.greeks(S, K, T, r, sigma, q, option_type=kind)
            assert lo <= g["delta"] <= hi and g["gamma"] >= 0.0, (S, K, T, r, sigma, q, kind, dict(g))


def test_the_american_put_greeks_agree_with_a_fine_tree():
    fd = american.greeks(100, 100, 1.0, 0.05, 0.2, 0.0, option_type="put", scheme="high")
    tree = binomial.price_with_greeks(100, 100, 1.0, 0.05, 0.2, 0.0, "put", 4001, True, "lr")
    assert fd["delta"] == pytest.approx(tree.delta, abs=1e-4)
    assert fd["gamma"] == pytest.approx(tree.gamma, abs=1e-4)
    assert fd["theta"] == pytest.approx(tree.theta, rel=1e-3)
    assert fd["vega"] == pytest.approx(tree.vega, rel=1e-3)
    assert fd["rho"] == pytest.approx(tree.rho, rel=1e-3)
    assert fd["epsilon"] == pytest.approx(tree.epsilon, rel=1e-3)


def test_greeks_hold_the_scheme_across_a_hand_over():
    # At a stiffness just below 5 the fast scheme solves the option, but the lower volatility bump is stiffer and
    # would be handed to the accurate scheme: the value jumps there, and a plain difference reads the jump as vega.
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.10, 0.04475
    assert american.price_details(S, K, T, r, sigma, option_type="put", scheme="fast").scheme == "fast"
    assert american.price_details(S, K, T, r, sigma - 1e-4, option_type="put", scheme="fast").scheme == "accurate"
    held = dataclasses.replace(american.SCHEMES["fast"])
    vega = american.greeks(S, K, T, r, sigma, option_type="put", scheme="fast")["vega"]

    def central(scheme):
        up = american.price(S, K, T, r, sigma + 1e-4, option_type="put", scheme=scheme)
        down = american.price(S, K, T, r, sigma - 1e-4, option_type="put", scheme=scheme)
        return (up - down) / 2e-4

    assert vega == pytest.approx(central(held), rel=1e-12)
    assert abs(central("fast") - vega) > 0.005 * vega  # the jump that holding the scheme avoids
    assert vega == pytest.approx(american.greeks(S, K, T, r, sigma, option_type="put", scheme="high")["vega"], 1e-3)


def test_greeks_need_a_positive_volatility():
    with pytest.raises(ValueError, match="positive spot, strike, time to expiry and volatility"):
        american.greeks(100, 100, 1.0, 0.05, 0.0, option_type="put")


# --- Diagnostics and the exercise boundary ---------------------------------------------------------------------------


def test_the_exercise_boundary_chart_is_the_engines(new_app):
    at = _run(new_app, div="3")
    spec = _chart(at, "exercise_boundary")
    traces = {t["name"]: t for t in spec["data"]}
    assert set(traces) == {"Call boundary", "Put boundary"}
    for kind in ("call", "put"):
        tau, spots = american.exercise_boundary(100, 1.0, 0.05, 0.2, 0.03, option_type=kind)
        trace = traces[f"{kind.capitalize()} boundary"]
        np.testing.assert_allclose(_values(trace["x"]), (1.0 - tau[::-1]) * 365, rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(_values(trace["y"]), spots[::-1], rtol=1e-12)
    engine = _table(at, "Critical spot").set_index("Option")
    # Today's end of each curve is the critical spot of the engine table.
    assert _values(traces["Put boundary"]["y"])[0] == pytest.approx(engine.loc["Put", "Critical spot"], rel=1e-9)
    assert _values(traces["Call boundary"]["y"])[0] == pytest.approx(engine.loc["Call", "Critical spot"], rel=1e-9)
    # Drawn with the desk config, like every chart (charts.show).
    chart = next(c for c in at.get("plotly_chart") if c.proto.id.endswith("-exercise_boundary"))
    config = json.loads(chart.proto.config)
    assert config["displaylogo"] is False and chart.proto.alt.startswith("Early-exercise boundary of the American")


def test_without_dividends_only_the_put_has_a_boundary_and_the_caption_says_why(new_app):
    at = _run(new_app)
    assert [t["name"] for t in _chart(at, "exercise_boundary")["data"]] == ["Put boundary"]
    assert any("No call boundary: with no dividend yield" in c.value for c in at.caption)


def test_a_fallback_to_the_tree_is_stated_plainly(new_app):
    # A call at a negative rate without dividends is exercised early; at 500 % volatility over five years the
    # engine's boundary iteration does not converge, and the call is valued on the tree.
    at = _run(new_app, rate="-1", vol="500", days="1825")
    details = american.price_details(100, 100, 5.0, -0.01, 5.0, 0.0, option_type="call")
    assert details.fallback and details.fallback_reason == "not converged"
    [warning] = at.warning
    assert warning.value.startswith(
        "The call is valued on Leisen-Reimer trees (2,001 and 4,003 steps, extrapolated) instead of the American "
        "engine: the engine's iteration for the exercise boundary did not converge at these inputs (its last step"
    )
    engine = _table(at, "Premium").set_index("Option")
    assert engine.loc["Call", "Method"] == "Leisen-Reimer trees (fallback)"
    assert engine.loc["Call", "American"] == pytest.approx(details.value, abs=1e-12)
    assert "Valued on Leisen-Reimer trees, not by the engine" in _metrics(at)["Call value"].proto.help
    assert any("The call's are differences of tree values" in c.value for c in at.caption)
    # The put is European at a negative rate, and neither option has a boundary to draw.
    assert not [c for c in at.get("plotly_chart") if c.proto.id.endswith("-exercise_boundary")]
    [info] = [i.value for i in at.info]
    assert "No call boundary: the engine's iteration" in info and "No put boundary: at a rate of zero" in info


def test_a_double_boundary_fallback_names_its_reason():
    # Not reachable from the sidebar (its dividend yield is never negative), but the statement covers it.
    details = {
        kind: american.price_details(100, 100, 1.0, -0.01, 0.2, -0.02, option_type=kind) for kind in ("call", "put")
    }
    prices = tab_pricing.Prices(details["call"].value, details["put"].value, alo=details)
    assert tab_pricing.alo_fallbacks(prices) == [
        "The put is valued on Leisen-Reimer trees (2,001 and 4,003 steps, extrapolated) instead of the American "
        "engine: its exercise region lies between two boundaries, which the engine does not solve."
    ]


def test_european_only_views_say_so_under_the_engine(new_app):
    at = _run(new_app)
    notes = [c.value for c in at.caption if c.value.startswith("Closed-form Black-Scholes-Merton values")]
    assert len(notes) == 2 and all("(used for the price cards, the Greeks" in n for n in notes)
    assert all(n.endswith("Early exercise is ignored here.") for n in notes)


def test_every_tab_runs_under_the_engine_without_emoji_and_with_the_desk_charts(new_app):
    at = _run(new_app)
    for tab in TABS:
        _open(at, tab)
        for chart in at.get("plotly_chart"):
            assert json.loads(chart.proto.config)["displayModeBar"] == "hover"
            assert len(chart.proto.alt) >= 20
        for node in _nodes(at._tree):
            for text in _strings(node.proto) if getattr(node, "proto", None) is not None else ():
                assert not _EMOJI.search(text) and not _SHORTCODE.search(text), (tab, text[:120])


# --- IV tab --------------------------------------------------------------------------------------------------------


def _iv_type(at: AppTest):
    return next(s for s in at.segmented_control if s.key == "iv_type")


def _quote(at: AppTest):
    return next(n for n in at.number_input if n.label == "Market price")


@pytest.mark.parametrize("kind", ["call", "put"])
def test_iv_tab_inverts_the_engine(new_app, kind):
    at = _run(new_app, tab="IV", div="2")
    _iv_type(at).set_value(kind).run()
    _ok(at)
    style = next(s for s in at.segmented_control if s.key.startswith("iv_style"))
    assert style.value == "American"  # follows the sidebar: the engine prices American exercise
    # The quote starts at the engine's value, which the solver turns back into the sidebar's volatility.
    assert _quote(at).value == pytest.approx(_price(kind, q=0.02), abs=1e-12)
    cards = _metrics(at)
    assert cards["Implied volatility"].value == "20.000%"
    assert cards["Model check"].value == f"${_price(kind, q=0.02):,.4f}"
    vega = american.greeks(100, 100, 1.0, 0.05, 0.2, 0.02, option_type=kind)["vega"] / 100
    assert cards["Vega at IV"].value == f"${vega:,.4f}"
    assert any("solved with Brent's method on the sidebar's American engine (accurate scheme)" in c.value
               for c in at.caption)  # fmt: skip
    # Another quote: the volatility found reprices it.
    _quote(at).set_value(9.0).run()
    _ok(at)
    iv = float(_metrics(at)["Implied volatility"].value.rstrip("%")) / 100
    solved = implied_vol.implied_volatility_alo(9.0, 100, 100, 1.0, 0.05, 0.02, kind)
    assert iv == pytest.approx(solved, abs=5e-6)
    assert american.price(100, 100, 1.0, 0.05, solved, 0.02, option_type=kind) == pytest.approx(9.0, abs=1e-9)


def test_iv_curve_is_the_engines_value_at_41_volatilities(new_app):
    at = _run(new_app, tab="IV", div="2")
    _iv_type(at).set_value("call").run()
    _ok(at)
    model = next(t for t in _chart(at, "iv_curve")["data"] if t["name"] == "Model value")
    vols = _values(model["x"]) / 100
    # 41 points, as on the tree's curve, from 0.5 % to the largest of 2.5 times the implied volatility (20 %),
    # twice the sidebar's and 60 %.
    np.testing.assert_allclose(vols, np.linspace(0.005, 0.6, 41), rtol=1e-12)
    engine = american.price(100, 100, 1.0, 0.05, vols, 0.02, option_type="call")
    np.testing.assert_allclose(_values(model["y"]), engine, rtol=1e-12)


@pytest.mark.slow  # about 9 s: each tree value of the fallback takes about 0.1 s
def test_iv_tab_inverts_the_tree_fallback_at_the_engines_corner(new_app):
    # Where the engine hands the call to the tree (500 % volatility over five years at a rate of -1 %), the IV tab
    # inverts the tree values: the quote at the sidebar volatility gives it back, and the curve has 41 points.
    at = _run(new_app, tab="IV", rate="-1", vol="500", days="1825")
    _iv_type(at).set_value("call").run()
    _ok(at)
    assert american.price_details(100, 100, 5.0, -0.01, 5.0, 0.0, option_type="call").fallback
    assert _metrics(at)["Implied volatility"].value == "500.000%"
    model = next(t for t in _chart(at, "iv_curve")["data"] if t["name"] == "Model value")
    assert _values(model["x"]).size == 41


def test_iv_tab_checks_the_bounds_first_with_the_trees_messages(new_app):
    at = _run(new_app, tab="IV", strike="120")
    _iv_type(at).set_value("put").run()
    _quote(at).set_value(15.0).run()  # below the exercise value of 20
    [error] = at.error
    with pytest.raises(implied_vol.ImpliedVolError) as tree:
        implied_vol.implied_volatility_american(15.0, 100, 120, 1.0, 0.05, 0.0, "put")
    assert error.value == f"No implied volatility exists: {tree.value}."
    assert "below the American lower bound" in error.value


def test_iv_tab_keeps_the_tree_for_the_tree_model(new_app):
    at = new_app(tab="IV", model="Binomial tree", style="American").run()
    _ok(at)
    assert any("solved with Brent's method on the 501-step Leisen-Reimer tree" in c.value for c in at.caption)


def test_alo_implied_volatility_round_trips_and_shares_the_tree_solvers_messages():
    for kind in ("call", "put"):
        for sigma in (0.05, 0.2, 0.8):
            for q in (0.0, 0.04):
                quote = american.price(100, 100, 0.5, 0.03, sigma, q, option_type=kind)
                iv = implied_vol.implied_volatility_alo(quote, 100, 100, 0.5, 0.03, q, kind)
                assert iv == pytest.approx(sigma, abs=1e-8), (kind, sigma, q)
    for quote in (1.0, 125.0):  # below the exercise value; at the upper bound (the strike)
        with pytest.raises(implied_vol.ImpliedVolError) as alo:
            implied_vol.implied_volatility_alo(quote, 80, 100, 1.0, 0.05, 0.0, "put")
        with pytest.raises(implied_vol.ImpliedVolError) as tree:
            implied_vol.implied_volatility_american(quote, 80, 100, 1.0, 0.05, 0.0, "put")
        assert (str(alo.value), alo.value.reason) == (str(tree.value), tree.value.reason)
    # The value of exercising now is the zero-volatility value: zero, as the tree solver reports it.
    assert implied_vol.implied_volatility_alo(40.0, 60, 100, 1.0, 0.05, 0.0, "put") == 0.0
    # A put with q > r > 0 is worth most at zero volatility when exercised at a date before expiry (here 25 at
    # 13.9 years), above its bounds (23.25): a quote between the two has no volatility.
    assert american.price(100, 100, 20.0, 0.05, 0.0, 0.10, option_type="put") == pytest.approx(25.0, abs=1e-9)
    with pytest.raises(implied_vol.ImpliedVolError, match="below the zero-volatility American value 25") as alo:
        implied_vol.implied_volatility_alo(24.0, 100, 100, 20.0, 0.05, 0.10, "put")
    assert alo.value.reason == "below_lower_bound"


def test_the_methodology_describes_the_engine_with_its_references():
    from options_analyzer.ui import tab_methodology

    pricing = next(s for s in tab_methodology.sections() if s.name == "pricing").body
    assert "### American engine (Andersen–Lake–Offengenden)" in pricing
    for reference in ("doi:10.21314/JCF.2016.312", "doi:10.21314/JCF.2021.004", "arXiv:2109.15157",
                      "QdFpAmericanEngine"):  # fmt: skip
        assert reference in pricing
    assert re.search(r"\*accurate\* is within \\\$0\.000005", pricing)
    # The Greeks: their bumps with numbers and units kept together, and the one-sided slopes at r = q = 0.
    engine = pricing[pricing.index("### American engine") :]
    assert "±0.01%" in engine and "±0.01 %" not in engine
    assert "±0.01\xa0vol point" in engine and "±1\xa0bp" in engine
    assert "At a rate and dividend yield of zero both options are" in engine
    assert "so their rho and dividend rho are one-sided" in engine
