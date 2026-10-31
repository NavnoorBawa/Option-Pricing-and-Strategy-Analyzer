"""The Pricing and IV tabs in the desk layout (UI spec 3.1 and 3.5; roadmap PR-7, UI-1, UI-7).

AppTests at the pinned valuation time (``asof=2026-10-23T16:00``) check the tabs' numbers against references written
out here: Black-76 on the expiry's parity forward and discount factor, implied vols by Brent's method on that formula
(scipy), the smile's P(ITM) by a central difference of call values on the smile (Breeden-Litzenberger), and the
time to the AM settlement by minutes. The spec's example numbers came from an earlier fit; where the code differs,
these tests assert the code's numbers after checking them against the references (UI spec 3.1, acceptance).
"""

from __future__ import annotations

import base64
import json
import math
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest
from _apptest import ASOF
from scipy.optimize import brentq
from scipy.special import ndtr
from streamlit.testing.v1 import AppTest
from test_format_standard import HAND_FORMAT

from options_analyzer import american, monte_carlo
from options_analyzer import black_scholes as bs
from options_analyzer.ui import cache, charts, tab_iv, tab_pricing
from options_analyzer.ui import market as mk
from options_analyzer.ui.app import TABS

AT = datetime.fromisoformat(ASOF)
INDEX = {"preset": None, "asof": ASOF}
MINUS = "−"


def _ok(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def _tiles(at: AppTest) -> dict:
    return {m.label: m for m in at.metric}


#: The first column of each table that is not a widget (AppTest gives a dataframe without a selection no key).
FIRST_COLUMN = {
    "tbl_pricing_contract": "Type",
    "tbl_pricing_greeks": "Greek",
    "tbl_pricing_engines": "Engine",
    "tbl_pricing_tree_greeks": "Option",
}


def _frame(at: AppTest, key: str):
    """The dataframe drawn with ``key`` (the chain ladder's key starts with it)."""
    if key in FIRST_COLUMN:
        return next(d for d in at.dataframe if list(d.value.columns)[:1] == [FIRST_COLUMN[key]] and (
            key != "tbl_pricing_tree_greeks" or "Value" in d.value.columns))  # fmt: skip
    return next(d for d in at.dataframe if re.search(rf"-{re.escape(key)}[^-]*$", d.proto.id))


def _headers(at: AppTest, key: str) -> dict[str, str]:
    """Column name -> header shown, of a dataframe."""
    columns = json.loads(_frame(at, key).proto.columns)
    return {name: config.get("label", name) for name, config in columns.items() if not name.startswith("_")}


def _chart(at: AppTest, key: str) -> dict:
    return json.loads(next(c for c in at.get("plotly_chart") if c.proto.id.endswith(f"-{key}")).proto.spec)


def _values(data) -> np.ndarray:
    if isinstance(data, dict):
        return np.frombuffer(base64.b64decode(data["bdata"]), dtype=data["dtype"])
    return np.asarray(data, dtype=float)


def _number(text: str) -> float:
    return float(text.replace(MINUS, "-").replace(",", "").rstrip("%"))


def _black(F: float, K: float, T: float, DF: float, sigma: float, kind: str) -> float:
    """Black-76, written out: DF·(F·N(d1) − K·N(d2)) for a call, DF·(K·N(−d2) − F·N(−d1)) for a put."""
    root = sigma * math.sqrt(T)
    d1 = math.log(F / K) / root + 0.5 * root
    d2 = d1 - root
    if kind == "call":
        return DF * (F * ndtr(d1) - K * ndtr(d2))
    return DF * (K * ndtr(-d2) - F * ndtr(-d1))


def _implied(price: float, F: float, K: float, T: float, DF: float, kind: str) -> float:
    """Black-76 implied vol by Brent's method, to 1e-12 (an independent route from the app's Let's Be Rational)."""
    return brentq(lambda s: _black(F, K, T, DF, s, kind) - price, 1e-4, 3.0, xtol=1e-14)


@pytest.fixture(scope="module")
def default_index():
    """The default contract of the index at the pinned valuation time: 20-Nov-26 AM, strike 5,815."""
    context = mk.index_market(AT)
    row = context.expiry_row(date(2026, 11, 20))
    K = 5815.0
    T, F, DF = float(row["maturity"]), float(row["forward"]), float(row["discount_factor"])
    sigma = float(context.surface.implied_vol_at(K / context.spot, T, extrapolate=True))
    quotes = context.chain.quotes
    rows = quotes[(quotes["strike"] == K) & (quotes["settlement"] == "AM")]
    rows = rows[rows["expiry"].astype(str).str.startswith("2026-11-20")]
    book = {r.type: (float(r.bid), float(r.ask)) for r in rows.itertuples()}
    return {"context": context, "K": K, "T": T, "F": F, "DF": DF, "sigma": sigma, "book": book}


# --- UI-1: no price cards in the shell ------------------------------------------------------------------------------


@pytest.mark.parametrize("tab", TABS[1:])
def test_the_shell_draws_no_price_cards(new_app, tab):
    # The cards pushed every tab's content down (about 530 px on a phone) and described the wrong thing on Strategy,
    # Risk and Surface (UI spec, findings 2): only the Pricing tab shows its contract's prices.
    at = new_app(tab=tab).run()
    _ok(at)
    assert not {"Call", "Put", "Call value", "Put value"} & {m.label for m in at.metric}


# --- The tiles (UI spec 3.1 item 2) ---------------------------------------------------------------------------------


def test_the_tiles_at_the_default_index(index_app, default_index):
    # The spec's example (an earlier fit) reads CALL 98.57, PUT 98.35, STRADDLE 196.92 ±3.40%, σ 15.46%, FORWARD
    # 5,815.22 pair +0.05 bp. The code's fit gives the numbers below, each checked here against Black-76 on the
    # expiry's parity forward and discount factor at the surface's vol, and the chain's own quotes.
    d = default_index
    call, put = (_black(d["F"], d["K"], d["T"], d["DF"], d["sigma"], kind) for kind in ("call", "put"))
    tiles = _tiles(index_app)
    assert [m.label for m in index_app.metric] == ["Call", "Put", "Straddle", "Vol", "Forward", "Time"]
    assert (tiles["Call"].value, tiles["Put"].value) == (f"{call:.2f}", f"{put:.2f}") == ("97.81", "98.90")
    assert tiles["Call"].delta == f"{call / 5800:.2%} of S" == "1.69% of S"
    assert tiles["Straddle"].value == f"{call + put:.2f}" == "196.71"
    assert tiles["Straddle"].delta == f"±{(call + put) / 5800:.2%}" == "±3.39%"
    assert (tiles["Vol"].value, tiles["Vol"].delta) == (f"{d['sigma']:.2%}", "surface") == ("15.44%", "surface")
    # The forward this strike's pair implies, K + (C mid − P mid)/DF, against the expiry's parity forward.
    (cb, ca), (pb, pa) = d["book"]["call"], d["book"]["put"]
    pair = d["K"] + ((cb + ca) / 2 - (pb + pa) / 2) / d["DF"]
    gap_bp = (pair - d["F"]) / d["F"] * 1e4
    assert tiles["Forward"].value == f"{d['F']:,.2f}" == "5,813.91"
    assert tiles["Forward"].delta == f"pair {gap_bp:+.2f} bp".replace("-", MINUS) == f"pair {MINUS}0.29 bp"
    # Fri 23-Oct 16:00 to Fri 20-Nov 09:30 (AM settlement): 27 days 17 hours 30 minutes, floored to the hour.
    assert d["T"] * 525_600 == pytest.approx(27 * 1440 + 17 * 60 + 30, abs=1e-6)
    assert (tiles["Time"].value, tiles["Time"].delta) == ("27d 17h", "AM settle")
    assert "T = 0.075970 years" in tiles["Time"].proto.help
    # The exact value to 6 dp is in the tile's help.
    assert f"{call:.6f}" in tiles["Call"].proto.help


def test_the_textbook_tiles_read_a_dash_where_the_market_would_be(app):
    tiles = _tiles(app)
    assert (tiles["Call"].value, tiles["Put"].value) == ("10.45", "5.57")  # version 2's $10.4506 and $5.5735
    assert tiles["Forward"].value == f"{100 * math.exp(0.05):.2f}" and tiles["Forward"].delta == "pair –"
    assert "the textbook market has no quotes" in tiles["Forward"].proto.help
    assert tiles["Vol"].delta == "flat" and tiles["Time"].value == "365d"


def test_the_context_line_names_the_contract_and_the_scenario(new_app):
    at = new_app(**INDEX).run()
    _ok(at)
    line = "SPX 20‑11‑26 AM 5,815 · last trade Thu 19‑Nov · settles at the SOQ"
    line = line.replace("20‑11‑26", "20‑Nov‑26")
    assert at.caption[2].value == line or any(c.value == line for c in at.caption)
    # A what-if shift set on Strategy or Risk does not move the Pricing tab's market; its context line says so.
    at.session_state["ws"] = -2.0
    at.run()
    assert any(c.value == f"{line} · Scenario on Strategy, Risk: spot {MINUS}2.0%" for c in at.caption)


# --- The contract table (item 3) -------------------------------------------------------------------------------------


def test_the_contract_table_puts_the_market_beside_the_theo(index_app, default_index):
    d = default_index
    frame = _frame(index_app, "tbl_pricing_contract").value.set_index("Type")
    headers = _headers(index_app, "tbl_pricing_contract")
    assert list(headers.values())[:9] == [
        "Type", "Bid", "Ask", "Mid", "Theo", "IV bid (%)", "IV ask (%)", "Mid vs fit (vp)", "Band",
    ]  # fmt: skip
    assert not any("Edge" in h for h in headers.values())  # against the fit it is a residual, not an edge
    for kind in ("call", "put"):
        bid, ask = d["book"][kind]
        row = frame.loc[kind.capitalize()]
        assert (row["Bid"], row["Ask"], row["Mid"]) == (bid, ask, round((bid + ask) / 2, 2))
        iv_bid, iv_ask, iv_mid = (
            _implied(p, d["F"], d["K"], d["T"], d["DF"], kind) for p in (bid, ask, (bid + ask) / 2)
        )
        assert row["IV bid"] == round(iv_bid * 100, 2) and row["IV ask"] == round(iv_ask * 100, 2)
        assert row["Mid vs fit"] == round((iv_mid - d["sigma"]) * 100, 2)
        assert row["Band"] == "in band" and iv_bid <= d["sigma"] <= iv_ask
    assert (frame.loc["Call", "Mid vs fit"], frame.loc["Put", "Mid vs fit"]) == (-0.01, 0.02)


def test_an_override_shows_buy_and_sell_edge(new_app, default_index):
    # UI spec 3.1's example: vol 16 against IV ask 15.52 and bid 15.40 reads +0.48 and −0.60. On this fit's chain the
    # call's ask and bid invert to 15.51% and 15.34%, so +0.49 and −0.66.
    at = new_app(**INDEX, vol_src="Override", vol="16").run()
    _ok(at)
    d = default_index
    frame = _frame(at, "tbl_pricing_contract").value.set_index("Type")
    headers = set(_headers(at, "tbl_pricing_contract").values())
    assert {"Buy edge (vp)", "Sell edge (vp)"} <= headers and "Mid vs fit (vp)" not in headers
    bid, ask = d["book"]["call"]
    iv_bid, iv_ask = (_implied(p, d["F"], d["K"], d["T"], d["DF"], "call") for p in (bid, ask))
    assert frame.loc["Call", "Buy edge"] == round((0.16 - iv_ask) * 100, 2) == 0.49
    assert frame.loc["Call", "Sell edge"] == round((iv_bid - 0.16) * 100, 2) == -0.66
    assert frame.loc["Call", "Band"] == "outside"
    assert any(m.value == ":yellow-badge[OVERRIDE]" for m in at.markdown)
    assert _tiles(at)["Vol"].delta == "override +0.56 vp"


def test_the_contract_and_greeks_tables_give_one_put_delta(new_app):
    # The audit's worst trust failure was two put deltas on one screen (P1). Under the American engine both tables
    # show the engine's.
    at = new_app(model="American (ALO)", div="2").run()
    _ok(at)
    contract = _frame(at, "tbl_pricing_contract").value.set_index("Type")
    greeks = _frame(at, "tbl_pricing_greeks").value.set_index("Greek")
    engine = american.greeks(100, 100, 1.0, 0.05, 0.2, 0.02, option_type="put")["delta"]
    assert contract.loc["Put", "Δ"] == round(engine, 3)
    assert _number(greeks.loc["Δ", "Put (ALO)"]) == round(engine, 3)
    # No column says BSM unless the reference is switched on.
    for key in ("tbl_pricing_contract", "tbl_pricing_greeks"):
        assert not any("BSM" in h for h in _headers(at, key).values()), key
    at.toggle(key="pr_bsm_ref").set_value(True).run()
    assert {"Call BSM ref", "Put BSM ref"} <= set(_frame(at, "tbl_pricing_greeks").value.columns)


# --- The Greeks table (item 6) ---------------------------------------------------------------------------------------


def test_the_greeks_table_is_in_desk_units(app):
    table = _frame(app, "tbl_pricing_greeks").value.set_index("Greek")
    call = bs.greeks(100, 100, 1.0, 0.05, 0.2, 0.0, "call")
    expected = {
        "Δ": f"{call['delta']:+.3f}",
        "Δ fwd": f"{call['delta'] * math.exp(-0.05):+.3f}",
        "Γ 1%": f"{call['gamma'] * 100 / 100:+.3g}",
        "Vega": f"{call['vega'] / 100:+.2f}",
        "Θ": f"{call['theta'] / 365:+.2f}",
        "ρ": f"{call['rho'] / 100:+.2f}",
        "ε": f"{call['epsilon'] / 100:+.2f}",
    }
    for row, text in expected.items():
        assert table.loc[row, "Call (BSM)"] == text.replace("-", MINUS), row
    # The textbook market has no smile: P(ITM) is N(d2) at the flat vol, and says so.
    d2 = (math.log(100 / 100) + (0.05 - 0.02)) / 0.2
    assert table.loc["P(ITM)", "Unit"] == "at expiry, flat vol"
    assert table.loc["P(ITM)", "Call (BSM)"] == f"{ndtr(d2):.1%}"
    assert list(table.index)[-1] == "P(ITM)"


@pytest.mark.parametrize(("order", "rows"), [("+2nd", 4), ("+3rd", 8)])
def test_the_order_control_adds_second_and_third_order_greeks(app, order, rows):
    first = len(_frame(app, "tbl_pricing_greeks").value)
    app.segmented_control(key="pr_greeks").set_value(order).run()
    _ok(app)
    table = _frame(app, "tbl_pricing_greeks").value.set_index("Greek")
    assert len(table) == first + rows
    vanna = bs.greeks(100, 100, 1.0, 0.05, 0.2, 0.0, "call")["vanna"] * bs.GREEKS["vanna"].scale
    assert _number(table.loc["Vanna", "Call (BSM)"]) == pytest.approx(vanna, rel=5e-3)  # 3 significant digits


def test_p_itm_comes_from_the_smile(index_app, default_index):
    # Breeden-Litzenberger by a central difference of call values on the smile (each strike at its own surface
    # vol, 0.5 points either side): P(S_T > K) = −∂C/∂K ÷ DF. The difference's error is O(h²) of the density's
    # curvature, below 1e−5 here; the shown value has one decimal, so the bound is half a unit of it plus that.
    # (The spec's 57.3% and 42.7% are of the noise-free chain, which the app does not draw.)
    d = default_index
    context, h = d["context"], 0.5

    def call(k):
        vol = float(context.surface.implied_vol_at(k / context.spot, d["T"], extrapolate=True))
        return _black(d["F"], k, d["T"], d["DF"], vol, "call")

    above = -(call(d["K"] + h) - call(d["K"] - h)) / (2 * h) / d["DF"]
    table = _frame(index_app, "tbl_pricing_greeks").value.set_index("Greek")
    assert table.loc["P(ITM)", "Unit"] == "at expiry, RN smile"
    assert _number(table.loc["P(ITM)", "Call (BSM)"]) == pytest.approx(above * 100, abs=0.05 + 1e-3)
    assert _number(table.loc["P(ITM)", "Put (BSM)"]) == pytest.approx((1 - above) * 100, abs=0.05 + 1e-3)
    assert (table.loc["P(ITM)", "Call (BSM)"], table.loc["P(ITM)", "Put (BSM)"]) == ("57.1%", "42.9%")
    # N(d2) at the flat vol, for comparison: the caption gives it.
    root = d["sigma"] * math.sqrt(d["T"])
    d2 = math.log(d["F"] / d["K"]) / root - root / 2
    assert any(f"N(d₂) at flat vol: {ndtr(d2):.1%} call, {ndtr(-d2):.1%} put" in c.value for c in index_app.caption)
    assert f"{ndtr(d2):.1%}" == "49.0%"


# --- One-day decay beside theta (Milestone 1, now in the Greeks table) ------------------------------------------------


def _greek_rows(at: AppTest) -> list[str]:
    return list(_frame(at, "tbl_pricing_greeks").value["Greek"])


def test_a_one_day_at_the_money_put_decays_by_its_whole_premium(new_app):
    # The panel's case: a 1-day at-the-money put. Theta / 365 understates the day's cost; repriced at expiry the put
    # loses its whole premium, so the decay to expiry leads the table.
    at = new_app(days="1").run()
    _ok(at)
    rows = _greek_rows(at)
    assert rows.index("Decay to expiry") == rows.index("Θ") - 1
    table = _frame(at, "tbl_pricing_greeks").value.set_index("Greek")
    put = bs.price(100, 100, 1 / 365, 0.05, 0.2, 0.0, "put")
    assert table.loc["Decay to expiry", "Put (BSM)"] == f"{-put:.2f}".replace("-", MINUS)


@pytest.mark.parametrize(("days", "leads"), [(45, False), (3, False), (2, True)])
def test_the_decay_leads_the_table_only_beyond_ten_percent(new_app, days, leads):
    # At the money theta / 365 and the 1-day decay differ by 8.7 % (call) and 9.7 % (put) at 3 days, 14.1 % and
    # 15.2 % at 2 days (format.DECAY_SWITCH).
    at = new_app(days=str(days)).run()
    _ok(at)
    rows = _greek_rows(at)
    assert rows.index("1-day decay") == rows.index("Θ") + (-1 if leads else 1)


def test_the_decay_follows_the_american_engine(new_app):
    at = new_app(model="American (ALO)", days="30").run()
    _ok(at)
    table = _frame(at, "tbl_pricing_greeks").value.set_index("Greek")
    T = 30 / 365
    for kind in ("call", "put"):
        later = american.price(100, 100, T - 1 / 365, 0.05, 0.2, 0.0, option_type=kind)
        now = american.price(100, 100, T, 0.05, 0.2, 0.0, option_type=kind)
        assert table.loc["1-day decay", f"{kind.capitalize()} (ALO)"] == f"{later - now:+.2f}".replace("-", MINUS)


# --- The profile (item 4) --------------------------------------------------------------------------------------------


def test_the_profile_defaults_to_value_against_spot_over_ten_percent(index_app):
    spec = _chart(index_app, "pricing_profile")
    names = [t["name"] for t in spec["data"] if t.get("showlegend", True) is not False]
    assert names == ["Call", "Put"]
    x = _values(spec["data"][0]["x"])
    assert (x[0], x[-1], x.size) == (pytest.approx(5220.0), pytest.approx(6380.0), 121)
    assert spec["layout"]["xaxis"]["range"] == [pytest.approx(5220.0), pytest.approx(6380.0)]
    labels = [a["text"] for a in spec["layout"]["annotations"]]
    assert "S 5,800.00" in labels and "K 5,815" in labels and "F" in labels
    # The 95% risk-neutral range from the smile.
    index_app.segmented_control(key="pr_range").set_value("RN 95%").run()
    x = _values(_chart(index_app, "pricing_profile")["data"][0]["x"])
    lo, hi = tab_pricing.contract(_inputs_of(index_app)).surface.rn_range(
        tab_pricing.contract(_inputs_of(index_app)).expiry, 0.95
    )
    assert (x[0], x[-1]) == (pytest.approx(lo), pytest.approx(hi))


def _inputs_of(at: AppTest):
    """The index inputs of the pinned default contract, rebuilt as the sidebar builds them."""
    from options_analyzer.ui.inputs import Inputs
    from options_analyzer.ui.model_config import ModelConfig

    context = mk.index_market(AT)
    row = context.expiry_row(date(2026, 11, 20))
    T, F, DF = float(row["maturity"]), float(row["forward"]), float(row["discount_factor"])
    r, q = mk.carry(F, DF, context.spot, T)
    vol = float(context.surface.implied_vol_at(5815.0 / context.spot, T, extrapolate=True))
    return Inputs(
        context.spot, 5815.0, T * 365, vol, r, q, ModelConfig(), mk.INDEX, context, row["expiry"], "AM", T, vol
    )


def test_the_theta_profile_draws_the_decay_beside_theta(app):
    app.segmented_control(key="pr_metric").set_value("Θ").run()
    _ok(app)
    spec = _chart(app, "pricing_profile")
    traces = {t["name"]: t for t in spec["data"]}
    assert {"Call", "Put", "Call 1-day decay", "Put 1-day decay"} <= set(traces)
    assert traces["Call 1-day decay"]["line"]["dash"] == "dash"
    x = _values(traces["Put 1-day decay"]["x"])
    decay = tab_pricing.closed_form_decay(x, 100, 1.0, 0.05, 0.2, 0.0, "put")
    np.testing.assert_allclose(_values(traces["Put 1-day decay"]["y"]), decay, rtol=1e-12)


def test_under_the_american_engine_the_profile_draws_the_closed_form_as_a_reference(new_app):
    at = new_app(model="American (ALO)", div="2").run()
    _ok(at)
    traces = {t["name"]: t for t in _chart(at, "pricing_profile")["data"]}
    assert traces["Put BSM ref"]["line"]["dash"] == "dot"
    x = _values(traces["Put"]["x"])
    engine = american.price(x, 100, 1.0, 0.05, 0.2, 0.02, option_type="put")
    np.testing.assert_allclose(_values(traces["Put"]["y"]), engine, rtol=1e-12)
    # Delta by central differences of the engine on one boundary: within 1e-6 of american.greeks at the spot.
    at.segmented_control(key="pr_metric").set_value("Δ").run()
    delta = {t["name"]: t for t in _chart(at, "pricing_profile")["data"]}["Put"]
    spot = float(np.interp(100.0, _values(delta["x"]), _values(delta["y"])))
    assert spot == pytest.approx(american.greeks(100, 100, 1.0, 0.05, 0.2, 0.02, option_type="put")["delta"], abs=2e-3)


# --- The chain ladder (item 5, UI-7) ---------------------------------------------------------------------------------


def test_the_chain_ladder_spans_ten_delta_to_ten_delta(index_app):
    # 10Δ put to 10Δ call is 5,425 to 6,075 on the default expiry; Auto picks 25 points (27 rows) and the Pricing
    # strike, 5,815, is always a row: 28.
    table = _frame(index_app, "tbl_pricing_chain_").value
    strikes = list(table["Strike"])
    assert len(strikes) == 28 and (strikes[0], strikes[-1]) == (5425.0, 6075.0) and 5815.0 in strikes
    assert list(table.loc[table["Strike"] == 5815.0, "Mark"]) == ["←"]
    assert list(_headers(index_app, "tbl_pricing_chain_").values()) == [
        "Bid", "Ask", "Theo", "IV (%)", "Δ", "Strike", " ", "Δ", "IV (%)", "Theo", "Bid", "Ask",
    ]  # fmt: skip
    # Each theo is Black-76 at the strike's own surface vol.
    context = mk.index_market(AT)
    row = context.expiry_row(date(2026, 11, 20))
    T, F, DF = float(row["maturity"]), float(row["forward"]), float(row["discount_factor"])
    vol = float(context.surface.implied_vol_at(5500.0 / context.spot, T, extrapolate=True))
    theo = table.loc[table["Strike"] == 5500.0]
    assert float(theo["C Theo"].iloc[0]) == round(_black(F, 5500.0, T, DF, vol, "call"), 2)
    assert float(theo["P Theo"].iloc[0]) == round(_black(F, 5500.0, T, DF, vol, "put"), 2)


def test_the_chain_window_and_stride(index_app):
    index_app.segmented_control(key="pr_chain_win").set_value("All").run()
    strikes = list(_frame(index_app, "tbl_pricing_chain_").value["Strike"])
    assert len(strikes) <= tab_pricing.CHAIN_ROWS + 1 and all(k % 100 == 0 or k == 5815.0 for k in strikes)
    index_app.selectbox(key="pr_chain_stride").set_value(50).run()
    _ok(index_app)
    assert all(k % 50 == 0 or k == 5815.0 for k in _frame(index_app, "tbl_pricing_chain_").value["Strike"])


def test_chain_strikes_auto_stride():
    listed = np.arange(5400.0, 6101.0, 5.0)
    assert list(tab_pricing.chain_strikes(listed, 5800.0, 5850.0, "Auto", 5812.0)) == [
        *np.arange(5800.0, 5812.0, 5.0), 5812.0, *np.arange(5815.0, 5851.0, 5.0)
    ]  # fmt: skip
    assert tab_pricing.chain_strikes(listed, 5425.0, 6075.0, 25, 5815.0).size == 28


def test_a_click_on_a_chain_row_sets_the_strike(monkeypatch):
    # AppTest cannot fire on_select (UI spec 2.8): the callback is tested directly. A click on the 5,850 row.
    state = {"strike": 5815.0, "tbl_pricing_chain_5815.0": {"selection": {"rows": [2], "columns": []}}}
    monkeypatch.setattr(tab_pricing.st, "session_state", state)
    tab_pricing.on_chain_select("tbl_pricing_chain_5815.0", (5800.0, 5815.0, 5850.0))
    assert state["strike"] == 5850.0 and state[tab_pricing._CHAIN_MOVED] is True
    # No selection (a click that clears it) leaves the strike.
    state = {"strike": 5815.0, "k": {"selection": {"rows": [], "columns": []}}}
    monkeypatch.setattr(tab_pricing.st, "session_state", state)
    tab_pricing.on_chain_select("k", (5800.0,))
    assert state == {"strike": 5815.0, "k": {"selection": {"rows": [], "columns": []}}}


# --- Engine diagnostics (item 7, Appendix B decision 8) ---------------------------------------------------------------


def _count(monkeypatch, calls: dict, names: tuple[str, ...]) -> None:
    for name in names:
        real = getattr(cache, name)

        def counted(*args, _real=real, _name=name, **kwargs):
            calls[_name] = calls.get(_name, 0) + 1
            return _real(*args, **kwargs)

        monkeypatch.setattr(cache, name, counted)


def test_the_diagnostic_engines_run_only_while_their_expander_is_open(new_app, monkeypatch):
    calls: dict[str, int] = {}
    _count(monkeypatch, calls, ("tree_timed", "mc_timed", "tree_convergence", "tree_results", "alo_timed"))
    at = new_app(**INDEX).run()
    _ok(at)
    assert calls == {}
    at.session_state[tab_pricing.DIAGNOSTICS] = True
    at.run()
    _ok(at)
    assert calls["tree_timed"] == 3 and calls["mc_timed"] == 1 and calls["tree_convergence"] == 1
    assert "alo_timed" not in calls  # the index is European: no American engine rows


def test_the_engine_comparison_against_the_closed_form(new_app):
    at = new_app()
    at.session_state[tab_pricing.DIAGNOSTICS] = True
    at.run()
    _ok(at)
    table = _frame(at, "tbl_pricing_engines").value.set_index("Engine")
    assert list(table.index) == [
        "BSM · European", "LR 501 · European", "LR 2,001 · European", "CRR 2,001 · European", "MC 100,000 · European",
    ]  # fmt: skip
    call = bs.price(100, 100, 1.0, 0.05, 0.2, 0.0, "call")
    assert table.loc["BSM · European", "Call"] == round(call, 6)
    # Leisen-Reimer converges at second order: 2,001 steps within 1e-7 of the closed form here (measured 8.9e-8).
    assert abs(table.loc["LR 2,001 · European", "Call − ref"]) < 1e-7
    assert "call 10.4237 – 10.4582" in table.loc["MC 100,000 · European", "MC 95% interval"]


def test_a_simulated_price_with_too_few_paths_in_the_money_is_flagged(new_app):
    # Milestone 1: below 300 paths on the scarce side of the strike the interval under-covers (0.85 to 0.91), so it
    # is withheld with the count (a 120 call, 30 days out: about 85 of 100,000 paths finish in the money).
    at = new_app(strike="120", days="30")
    at.session_state[tab_pricing.DIAGNOSTICS] = True
    at.run()
    _ok(at)
    cell = _frame(at, "tbl_pricing_engines").value.set_index("Engine").loc["MC 100,000 · European", "MC 95% interval"]
    res = monte_carlo.price_european(100, 120, 30 / 365, 0.05, 0.2, 0.0, n_paths=100_000, seed=42)["call"]
    assert not res.reliable
    assert cell.startswith(f"call Interval unreliable: {res.n_itm:,} of 100,000 paths finish in the money")


def test_the_american_engine_rows_and_its_exercise_boundary(new_app):
    at = new_app(model="American (ALO)", div="3")
    at.session_state[tab_pricing.DIAGNOSTICS] = True
    at.run()
    _ok(at)
    table = _frame(at, "tbl_pricing_engines").value.set_index("Engine")
    assert list(table.index)[:3] == ["ALO fast", "ALO accurate", "ALO high"]
    assert table.loc["ALO high", "Put − ref"] == 0
    # ALO accurate within 5e-6 of high (its validated error, docs/methods/american_alo.md).
    assert abs(table.loc["ALO accurate", "Put − ref"]) < 5e-6
    spec = _chart(at, "pricing_boundary")
    assert {t["name"] for t in spec["data"]} == {"Call boundary", "Put boundary"}
    assert spec["layout"]["xaxis"]["range"][0] == 0.0  # the x-axis from today


def test_the_trees_own_greeks_carry_their_one_day_decay(new_app):
    # Milestone 1: the tree's decay reprices on trees a day shorter (tab_pricing.later_trees) and leads its theta
    # within two days of expiry.
    at = new_app(days="2")
    at.session_state[tab_pricing.DIAGNOSTICS] = True
    at.run()
    _ok(at)
    table = _frame(at, "tbl_pricing_tree_greeks").value
    columns = list(table.columns)
    assert columns.index("1-day decay") == columns.index("Θ") - 1
    exact = tab_pricing.closed_form_decay(100, 100, 2 / 365, 0.05, 0.2, 0.0, "call")
    assert float(table.set_index("Option").loc["Call", "1-day decay"]) == pytest.approx(exact, abs=0.005)


def test_monte_carlo_interval_withheld_for_too_few_independent_samples():
    res = monte_carlo.price_european(100, 100, 1, 0.05, 0.2, 0, n_paths=1_000, seed=1)["call"]
    assert not res.reliable and min(res.n_itm, res.n_paths - res.n_itm) >= monte_carlo.RELIABLE_PATHS
    assert tab_pricing.mc_interval(res, short=True) == "Interval unreliable: too few paths"
    assert tab_pricing.mc_interval(res, dollars=False) == "Interval unreliable: 1,000 paths, fewer than 10,000"


# --- Captions, footer, the number standard ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [INDEX, {}, {"model": "American (ALO)", "div": "2"}, {**INDEX, "vol_src": "Override", "vol": "16"}],
)
def test_every_caption_of_the_tab_is_at_most_140_characters(new_app, query):
    at = new_app(**query)
    at.session_state[tab_pricing.DIAGNOSTICS] = True
    at.run()
    _ok(at)
    pricing = at.tabs[0]
    captions = [c.value for c in pricing.caption]
    assert captions and all(len(c) <= 140 for c in captions), [c for c in captions if len(c) > 140]


def test_the_footer_names_the_engine_and_the_time_basis(index_app):
    assert any(
        c.value == "BSM · European · T to Fri 20‑Nov 09:30 ET = minutes/525,600 · σ from the surface at K/F "
        "100.0% · per share"
        for c in index_app.caption
    )
    assert index_app.button(key="pricing_methodology").label == "Methodology"


@pytest.mark.parametrize("module", [tab_pricing, tab_iv])
def test_the_tabs_format_numbers_only_through_the_standard(module):
    # The lint the tabs pass once they adopt the number standard (tests/test_format_standard.py).
    lines = Path(module.__file__).read_text(encoding="utf-8").splitlines()
    hits = [n for n, line in enumerate(lines, 1) if HAND_FORMAT.search(line)]
    assert not hits, hits


# --- The IV tab (UI spec 3.5) -----------------------------------------------------------------------------------------


def test_the_iv_tiles_at_the_default_index(new_app, default_index):
    # The spec's 15.40%, 15.46% and 15.52% are of its earlier chain (bid 98.20, ask 99.00); this chain quotes the
    # call 97.20 / 98.30, which Brent's method on Black-76 inverts to the vols below.
    at = new_app(**INDEX, tab="IV").run()
    _ok(at)
    d = default_index
    bid, ask = d["book"]["call"]
    ivs = [_implied(p, d["F"], d["K"], d["T"], d["DF"], "call") for p in (bid, (bid + ask) / 2, ask)]
    tiles = _tiles(at)
    assert [tiles[k].value for k in ("IV bid", "IV mid", "IV ask")] == [f"{v:.2%}" for v in ivs]
    assert [tiles[k].value for k in ("IV bid", "IV mid", "IV ask")] == ["15.34%", "15.43%", "15.51%"]
    assert ivs[0] <= ivs[1] <= ivs[2] and ivs[0] <= d["sigma"] <= ivs[2]
    assert tiles["Surface"].value == "15.44%" and tiles["Surface"].delta == f"mid vs fit {MINUS}0.01 vp"
    assert tiles["Bounds"].value == "inside"
    # The inputs start at the chain's quote; the index's exercise is European, and says why.
    assert [n.value for n in at.number_input if n.label in ("Bid", "Ask")] == [bid, ask]
    style = next(s for s in at.segmented_control if s.key.startswith("iv_style"))
    assert style.value == "European" and style.disabled
    assert not at.dataframe  # no expiry chain: chain-scale inversion is the Surface tab's


def test_a_price_below_the_floor_reads_dashes_and_names_the_bound(new_app):
    at = new_app(tab="IV").run()
    next(n for n in at.number_input if n.label == "Price").set_value(3.0).run()
    assert not at.exception
    tiles = _tiles(at)
    assert [tiles[k].value for k in ("IV bid", "IV", "IV ask")] == ["–", "–", "–"]
    [error] = [e.value for e in at.error]
    assert "\\$4.88" in error and not re.search(r"\d+\.\d{3,}", error)
    assert tiles["Bounds"].value == "below floor"
    assert _chart(at, "iv_price")["data"]  # the chart stays


def test_a_quote_with_no_time_value_is_told_apart(new_app):
    # Milestone 1 (no_time_value): an out-of-the-money option is worth only its time value.
    at = new_app(tab="IV", strike="150", days="5").run()
    next(n for n in at.number_input if n.label == "Price").set_value(0.0).run()
    assert not at.exception
    assert [w.value for w in at.warning] == [
        "No implied vol: an out-of-the-money option is worth only its time value, and a quote of \\$0.00 has none."
    ]
    assert _tiles(at)["IV"].value == "–"


def test_a_deep_in_the_money_quote_starts_inside_the_bounds(new_app):
    # A call at 60 on 100 two days out is worth its floor, 40.0131: the price starts at 40.02, not 40.01.
    at = new_app(tab="IV", strike="60", days="2").run()
    _ok(at)
    assert next(n for n in at.number_input if n.label == "Price").value == 40.02


def test_the_iv_chart_marks_the_quotes_and_their_vols(new_app):
    at = new_app(**INDEX, tab="IV").run()
    spec = _chart(at, "iv_price")
    names = [t["name"] for t in spec["data"]]
    assert names[0] == "Model value (BSM)" and {"IV bid", "IV mid", "IV ask"} <= set(names)
    assert [a["text"] for a in spec["layout"]["annotations"]][-3:] == ["Bid", "Mid", "Ask"]


def test_the_iv_tick_follows_the_index_rule():
    assert (tab_iv.tick(2.95, True), tab_iv.tick(3.0, True), tab_iv.tick(98.2, False)) == (0.05, 0.1, 0.01)


# --- The chart builders -----------------------------------------------------------------------------------------------


def test_the_error_chart_draws_a_slope_guide_on_log_axes():
    from options_analyzer.ui import charts_pricing

    steps = np.array([10, 100, 1000])
    fig = charts_pricing.error_chart({"LR call": (steps, 1.0 / steps**2, "call", "solid")}, reference="BSM")
    assert fig.layout.xaxis.type == "log" and fig.layout.yaxis.type == "log"
    guide = fig.data[-1]
    assert guide.name == "slope −1 (1/n)" and list(guide.y) == pytest.approx([1e-2, 1e-4])
    assert fig.layout.yaxis.title.text == "|Tree − BSM| ($)"


def test_exercise_boundary_colours_follow_the_option_type():
    days = np.array([0.0, 100.0])
    call, put = (days, np.array([130.0, 120.0])), (days, np.array([80.0, 90.0]))
    both = charts.exercise_boundary_chart({"Call boundary": call, "Put boundary": put}, 100.0)
    assert {t.name: t.line.color for t in both.data} == {"Call boundary": charts.CALL, "Put boundary": charts.PUT}
