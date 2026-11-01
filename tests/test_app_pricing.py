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
from test_app_alo import _row_places
from test_format_standard import HAND_FORMAT

from options_analyzer import american, monte_carlo
from options_analyzer import black_scholes as bs
from options_analyzer.ui import cache, charts, charts_pricing, components, tab_iv, tab_pricing
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


def _sci(text: str) -> float:
    """A difference written by format.signed_sci (``−8.9e−8``, ``0``) as a float."""
    return float(text.replace(MINUS, "-"))


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
    assert tiles["Straddle"].delta == f"±{(call + put) / 5800:.2%} of S" == "±3.39% of S"
    # The tile's move is the straddle-implied one (premium / spot), not the move to a breakeven: the breakevens are
    # K ± premium, asymmetric about spot, and the help names both.
    low, high = d["K"] - (call + put), d["K"] + (call + put)
    assert f"{low:,.2f} ({(low / 5800 - 1):.2%} of S)".replace("-", MINUS) in tiles["Straddle"].proto.help
    assert f"{high:,.2f} (+{(high / 5800 - 1):.2%} of S)" in tiles["Straddle"].proto.help
    assert "5,618.29 (−3.13% of S) and 6,011.71 (+3.65% of S)" in tiles["Straddle"].proto.help
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
    assert tiles["Forward"].value == f"{100 * math.exp(0.05):.2f}" and tiles["Forward"].delta == "from r and q"
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
    put = bs.greeks(100, 100, 1.0, 0.05, 0.2, 0.0, "put")

    def money(name: str, scale: float) -> str:
        # One precision a row: cents, or two significant digits of the row's largest value below 10 cents (the
        # put's Θ of -0.0049 a day read 0.00; the call's -0.0176 sets three decimals for the row).
        return f"{call[name] / scale:+.{_row_places(call[name] / scale, put[name] / scale)}f}"

    expected = {
        "Δ": f"{call['delta']:+.3f}",
        "Δ fwd": f"{call['delta'] * math.exp(-0.05):+.3f}",
        "Γ 1%": f"{call['gamma'] * 100 / 100:+.3g}",
        "Vega": money("vega", 100),
        "Θ": money("theta", 365),
        "ρ": money("rho", 100),
        "ε": money("epsilon", 100),
    }
    assert expected["Θ"] == "-0.018"
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


def test_third_order_greeks_are_in_the_gamma_the_table_shows(index_app, default_index):
    # At the index (S = 5,800, not 100) the table shows Γ 1% = Γ·S/100. Speed, zomma and color were in raw Γ per
    # point of spot, 58 times smaller than a change in that Γ 1%. Reference: the closed-form gamma written out here,
    # Γ = e^{−qT}·φ(d1)/(Sσ√T), turned into Γ 1% and differenced: by a 1% move in S (central, ±0.01%, then scaled
    # to 1%), by ±0.01 vol point (scaled to 1 point) and by ±1 minute (per calendar day, t = −T). Tolerance 5e-3
    # relative: the table shows 3 significant digits, rounded by at most 5e-3 relative; the differences are good to
    # about 1e-6.
    d = default_index
    index_app.segmented_control(key="pr_greeks").set_value("+3rd").run()
    _ok(index_app)
    table = _frame(index_app, "tbl_pricing_greeks").value.set_index("Greek")
    inputs = _inputs_of(index_app)
    S, K, T, r, q, sigma = inputs.spot, inputs.strike, inputs.expiry, inputs.rate, inputs.dividend_yield, d["sigma"]

    def gamma_1pct(s=S, v=sigma, t=T):
        d1 = (math.log(s / K) + (r - q + v * v / 2) * t) / (v * math.sqrt(t))
        return math.exp(-q * t) * math.exp(-d1 * d1 / 2) / math.sqrt(2 * math.pi) / (s * v * math.sqrt(t)) * s / 100

    h, minute = 1e-4, 1 / 525_600
    expected = {
        "Speed": (gamma_1pct(s=S * (1 + h)) - gamma_1pct(s=S * (1 - h))) / (2 * h) / 100,
        "Zomma": (gamma_1pct(v=sigma + h) - gamma_1pct(v=sigma - h)) / (2 * h) / 100,
        "Color": (gamma_1pct(t=T - minute) - gamma_1pct(t=T + minute)) / (2 * minute) / 365,
    }
    assert table.loc["Speed", "Unit"] == "Γ 1% per 1% move in S"
    assert table.loc["Zomma", "Unit"] == "Γ 1% per vol point"
    assert table.loc["Color", "Unit"] == "Γ 1% per calendar day"
    for row, value in expected.items():
        for column in ("Call (BSM)", "Put (BSM)"):  # one gamma, so one speed, zomma and color for both
            shown = table.loc[row, column]
            assert "e" not in shown, (row, shown)  # plain decimals, as the rest of the column
            assert _number(shown) == pytest.approx(value, rel=5e-3), (row, column)
    assert table.loc["Ultima", "Unit"] == "volga per vol point"  # the table's row is Volga, not vomma


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
    decay = {
        kind: american.price(100, 100, T - 1 / 365, 0.05, 0.2, 0.0, option_type=kind)
        - american.price(100, 100, T, 0.05, 0.2, 0.0, option_type=kind)
        for kind in ("call", "put")
    }
    places = _row_places(*decay.values())
    for kind, value in decay.items():
        assert table.loc["1-day decay", f"{kind.capitalize()} (ALO)"] == f"{value:+.{places}f}".replace("-", MINUS)


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
    # "RN 95%" holds 95%, not the 95.45% of ±2σ (the Strategy chart's range): P(S_T < K) = 1 + ∂C/∂K / DF, the
    # Breeden-Litzenberger digital, from Black-76 at the smile's own vol, differenced over ±0.5 point here, reads
    # 2.5% at the low end and 97.5% at the high one. Tolerance 2e-4: the central difference of a smooth smile over a
    # 1-point step is good to about 1e-6, and the surface's quantile inversion to its own 1e-4 grid.
    market = tab_pricing.contract(_inputs_of(index_app))
    T, F, DF = market.surface.maturity(market.expiry), market.forward, market.discount

    def below(k: float, h: float = 0.5) -> float:
        call = [_black(F, s, T, DF, market.surface.vol(s, market.expiry).vol, "call") for s in (k - h, k + h)]
        return 1.0 + (call[1] - call[0]) / (2 * h) / DF

    assert (below(x[0]), below(x[-1])) == (pytest.approx(0.025, abs=2e-4), pytest.approx(0.975, abs=2e-4))


def test_the_textbook_rn_range_is_the_lognormal_95_percent_range(app):
    # Under the textbook market the range is lognormal at the flat vol: F·exp(−σ²T/2 ± z σ√T) with z = Φ⁻¹(0.975).
    app.segmented_control(key="pr_range").set_value("RN 95%").run()
    x = _values(_chart(app, "pricing_profile")["data"][0]["x"])
    F, sigma, T = 100.0 * math.exp(0.05), 0.20, 1.0
    z = 1.959963984540054  # Φ⁻¹(0.975)
    centre = F * math.exp(-0.5 * sigma**2 * T)
    assert (x[0], x[-1]) == (pytest.approx(centre * math.exp(-z * sigma)), pytest.approx(centre * math.exp(z * sigma)))


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
    # 10Δ put to 10Δ call is 5,425 to 6,075 on the default expiry. The spec's Auto (at most 30 rows) took 25 points:
    # 28 rows, the Pricing strike's the 17th, below the fold of the 15-row frame (UI spec 2.8), and the page 2,492 px
    # high against the spec's 1,800. Auto keeps to 14 rows: 50 points, 5,450 to 6,050, and 5,815: 14 rows.
    table = _frame(index_app, "tbl_pricing_chain_").value
    strikes = list(table["Strike"])
    assert strikes == [*np.arange(5450.0, 5801.0, 50.0), 5815.0, *np.arange(5850.0, 6051.0, 50.0)]
    assert len(strikes) <= components.TABLE_ROWS  # the frame shows every row, the strike's among them
    assert list(table.loc[table["Strike"] == 5815.0, "Mark"]) == ["←"]
    # Each side's headers name it; nothing is pinned (a pinned column is drawn first, and the ladder is mirrored).
    headers = _headers(index_app, "tbl_pricing_chain_")
    assert list(headers.values()) == [
        "Call bid", "Call ask", "Call theo", "Call IV (%)", "Call Δ", "Strike", " ",
        "Put Δ", "Put IV (%)", "Put theo", "Put bid", "Put ask",
    ]  # fmt: skip
    columns = json.loads(_frame(index_app, "tbl_pricing_chain_").proto.columns)
    assert not any(config.get("pinned") for config in columns.values())
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
    _ok(index_app)
    table = _frame(index_app, "tbl_pricing_chain_").value
    strikes = list(table["Strike"])
    # Even 100 points gives 27 rows over every listed strike (Auto's coarsest), and 5,815: the frame scrolls.
    assert len(strikes) == 28 and all(k % 100 == 0 or k == 5815.0 for k in strikes)
    # Every listed strike, 4,150 to 6,850: below 4,600 the surface has no vol (beyond its reach), so those rows keep
    # their quotes and read "–" for theo and delta (they raised an error before).
    context = mk.index_market(AT)
    T = float(context.expiry_row(date(2026, 11, 20))["maturity"])
    reach = np.array(
        [np.isfinite(context.surface.implied_vol_at(k / context.spot, T, extrapolate=True)) for k in strikes]
    )
    assert (strikes[0], strikes[-1]) == (4200.0, 6800.0) and not reach.all()
    np.testing.assert_array_equal(np.isfinite(table["C Theo"].to_numpy(float)), reach)
    np.testing.assert_array_equal(np.isfinite(table["P Δ"].to_numpy(float)), reach)
    assert table["C Ask"].notna().all()
    index_app.selectbox(key="pr_chain_stride").set_value(50).run()
    _ok(index_app)
    assert all(k % 50 == 0 or k == 5815.0 for k in _frame(index_app, "tbl_pricing_chain_").value["Strike"])


def test_the_phone_ladder_puts_the_strike_first(index_app, default_index):
    # On a 390 px phone the mirrored ladder showed five call columns and cut the strike at the edge. Its phone set
    # (shown by the stylesheet below 641 px, tests/test_pricing_e2e.py) leads with the strike, pinned, then the call's
    # mid, IV and Δ and the put's Δ, IV and mid, one row per strike of the ladder.
    desk, phone = _frame(index_app, "tbl_pricing_chain_"), _frame(index_app, "tbl_pricing_ladder_phone_")
    assert list(phone.value.columns) == ["Strike", "Mark", "C Mid", "C IV", "C Δ", "P Δ", "P IV", "P Mid"]
    assert json.loads(phone.proto.columns)["Strike"]["pinned"] is True
    np.testing.assert_array_equal(phone.value["Strike"], desk.value["Strike"])
    for name in ("C IV", "C Δ", "P Δ", "P IV"):
        np.testing.assert_array_equal(phone.value[name], desk.value[name])
    row = phone.value.set_index("Strike").loc[5815.0]
    (cb, ca), (pb, pa) = default_index["book"]["call"], default_index["book"]["put"]
    assert row["Mark"] == "←"
    # An offer only (no bid) has no mid, not half its ask: the "All" window lists the far wings' zero bids.
    index_app.segmented_control(key="pr_chain_win").set_value("All").run()
    desk, phone = _frame(index_app, "tbl_pricing_chain_").value, _frame(index_app, "tbl_pricing_ladder_phone_").value
    for side in ("C", "P"):
        np.testing.assert_array_equal(phone[f"{side} Mid"].isna(), desk[f"{side} Bid"].isna())
    assert phone["C Mid"].isna().any() or phone["P Mid"].isna().any()
    assert (row["C Mid"], row["P Mid"]) == (pytest.approx((cb + ca) / 2), pytest.approx((pb + pa) / 2))


def test_chain_strikes_auto_stride():
    listed = np.arange(5400.0, 6101.0, 5.0)
    assert list(tab_pricing.chain_strikes(listed, 5800.0, 5850.0, "Auto", 5812.0)) == [
        *np.arange(5800.0, 5812.0, 5.0), 5812.0, *np.arange(5815.0, 5851.0, 5.0)
    ]  # fmt: skip
    assert tab_pricing.chain_strikes(listed, 5425.0, 6075.0, 25, 5815.0).size == 28
    assert tab_pricing.chain_strikes(listed, 5425.0, 6075.0, "Auto", 5815.0).size == 14


def test_a_click_on_a_chain_row_sets_the_strike(monkeypatch):
    # AppTest cannot fire on_select (UI spec 2.8): the callback is tested directly. A click on the 5,850 row.
    state = {"strike": 5815.0, "tbl_pricing_chain_0": {"selection": {"rows": [2], "columns": []}}}
    monkeypatch.setattr(tab_pricing.st, "session_state", state)
    tab_pricing.on_chain_select("tbl_pricing_chain_0", (5800.0, 5815.0, 5850.0))
    assert state["strike"] == 5850.0 and state[tab_pricing._CHAIN_MOVED] is True
    assert state[tab_pricing._CHAIN_CLICKS] == 1  # the next run's grid starts with no selection
    # No selection (a click that clears it) leaves the strike.
    state = {"strike": 5815.0, "k": {"selection": {"rows": [], "columns": []}}}
    monkeypatch.setattr(tab_pricing.st, "session_state", state)
    tab_pricing.on_chain_select("k", (5800.0,))
    assert state == {"strike": 5815.0, "k": {"selection": {"rows": [], "columns": []}}}


def test_a_strike_step_keeps_the_ladders_grids(new_app):
    # The grids' keys held the strike, so every strike step (sidebar, or a link) remounted both of them; only a handled
    # click starts new ones now.
    at = new_app(**INDEX).run()

    def keys():
        return {d.proto.id.rsplit("-", 1)[-1] for d in at.dataframe if "tbl_pricing_" in d.proto.id}

    before = keys()
    at.sidebar.number_input(key="strike").increment().run()
    _ok(at)
    assert at.sidebar.number_input(key="strike").value == 5820.0
    assert keys() == before and {"tbl_pricing_chain_0", "tbl_pricing_ladder_phone_0"} <= before


def test_a_click_on_a_row_beyond_the_surface_leaves_the_strike(monkeypatch):
    # The "All" window lists 4,200 to 4,500 with their quotes, beyond the smile's reach: a click on one crashed the
    # app (no vol to price it). The strike stays and the ladder says why.
    state = {"strike": 5815.0, "k": {"selection": {"rows": [0], "columns": []}}}
    monkeypatch.setattr(tab_pricing.st, "session_state", state)
    tab_pricing.on_chain_select("k", (4200.0, 5815.0), priced=(False, True), step=5.0)
    assert state["strike"] == 5815.0 and tab_pricing._CHAIN_MOVED not in state
    assert state[tab_pricing._CHAIN_NOTE].startswith("4,200 lies beyond the surface's reach at this expiry")


def test_a_strike_beyond_the_surface_says_so_instead_of_failing(new_app):
    # A link or the sidebar can ask for 4,300 on 20-Nov: the smile reaches 4,600 to 6,890 there (the strikes on the
    # 5-point grid where the surface's vol is finite, read here independently), so the page says so and stops.
    at = new_app(**INDEX, strike="4300").run()
    assert not at.exception
    context = mk.index_market(AT)
    T = float(context.expiry_row(date(2026, 11, 20))["maturity"])
    grid = np.arange(5.0, 4 * context.spot + 2.5, 5.0)
    reach = grid[[np.isfinite(context.surface.implied_vol_at(k / context.spot, T, extrapolate=True)) for k in grid]]
    assert (reach[0], reach[-1]) == (4600.0, 6890.0)
    assert [e.value for e in at.error] == [
        "The surface gives no vol at the strike 4,300: at this expiry its smile reaches the strikes 4,600 to 6,890, "
        "so nothing can be priced. Choose a strike in that range in the sidebar, or ATMF."
    ]
    assert not at.tabs and not at.metric
    # The sidebar stays, so the strike can be moved back.
    assert next(n for n in at.number_input if n.key == "strike").value == 4300.0


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
    _count(
        monkeypatch,
        calls,
        ("tree_timed", "mc_timed", "tree_convergence", "mc_convergence", "tree_results", "alo_timed"),
    )
    at = new_app(**INDEX).run()
    _ok(at)
    assert calls == {}
    at.session_state[tab_pricing.DIAGNOSTICS] = True
    at.run()
    _ok(at)
    assert calls["tree_timed"] == 3 and calls["mc_timed"] == 1 and calls["tree_convergence"] == 1
    assert calls["mc_convergence"] == 1
    assert "alo_timed" not in calls  # the index is European: no American engine rows


def test_the_convergence_chart_computes_no_reference_it_does_not_draw(monkeypatch):
    # The chart measures the trees against the engine table's reference; cache.tree_convergence also extrapolated
    # Leisen-Reimer trees for a reference of its own (0.19 s a contract), which the tab threw away.
    def refused(*args, **kwargs):
        raise AssertionError("an extrapolated reference was computed")

    monkeypatch.setattr(cache.binomial, "extrapolated_price", refused)
    series = cache.tree_convergence(101.0, 99.0, 0.75, 0.04, 0.25, 0.01, 61, True)
    assert set(series) == {"call", "put"} and len(series["call"]) == 2  # both lattices


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
    # The differences are in scientific notation (spec 2.5), one form a column: "−8.9e−8".
    assert abs(_sci(table.loc["LR 2,001 · European", "Call − ref"])) < 1e-7
    assert all(re.fullmatch(r"0|[+−]\d\.\de−?\d+", text) for text in table["Call − ref"]), list(table["Call − ref"])
    assert "call 10.4237 – 10.4582" in table.loc["MC 100,000 · European", "MC 95% interval"]
    # European trees keep put-call parity, so the call's and the put's errors are one: one line a lattice, each with
    # the guide of its order.
    names = [t["name"] for t in _chart(at, "pricing_convergence")["data"]]
    assert names == ["CRR call and put", "LR call and put", "CRR: slope −1 (1/n)", "LR: slope −2 (1/n²)"]


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
    assert table.loc["ALO high", "Put − ref"] == "0"
    # ALO accurate within 5e-6 of high (its validated error, docs/methods/american_alo.md).
    assert abs(_sci(table.loc["ALO accurate", "Put − ref"])) < 5e-6
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


def _default_k_over_f() -> str:
    """The default contract's K/F, from the chain's parity forward: 5,815 over 20-Nov-26's forward, in percent."""
    forward = float(mk.index_market(AT).expiry_row(date(2026, 11, 20))["forward"])
    return f"{5815.0 / forward * 100:.2f}%"


def test_the_footer_names_the_engine_and_the_time_basis(index_app):
    assert _default_k_over_f() == "100.02%"  # 1 decimal read it 100.0%, the sidebar 1.0002
    assert any(
        c.value == "BSM · European · T to Fri 20‑Nov 09:30 ET = minutes/525,600 · σ from the surface at K/F "
        "100.02% · per share"
        for c in index_app.caption
    )
    assert index_app.button(key="pricing_methodology").label == "Methodology"


def test_one_k_over_f_in_the_sidebar_the_pricing_tab_and_the_surface_readout(new_app):
    # The sidebar wrote K/F as 1.0002 while the Pricing tab and the Surface readout wrote 100.0%.
    at = new_app(**INDEX, tab="Surface").run()
    assert not at.exception
    k_over_f = f"K/F {_default_k_over_f()}"
    assert any(k_over_f in c.value for c in at.sidebar.caption)
    assert any(f"({k_over_f}," in c.value for c in at.main.caption)  # the Surface readout
    vol_tile = next(m for m in new_app(**INDEX).run().metric if m.label == "Vol")
    assert f"at {k_over_f} " in vol_tile.help
    texts = [c.value for c in at.caption] + [vol_tile.help]
    assert not [t for t in texts if re.search(r"K/F (1\.\d{4}|\d+\.\d%)", t)]


@pytest.mark.parametrize("module", [tab_pricing, tab_iv])
def test_the_tabs_format_numbers_only_through_the_standard(module):
    # The lint the tabs pass once they adopt the number standard (tests/test_format_standard.py).
    lines = Path(module.__file__).read_text(encoding="utf-8").splitlines()
    hits = [n for n, line in enumerate(lines, 1) if HAND_FORMAT.search(line)]
    assert not hits, hits


# --- The noise-free chain (UI spec 3.1 and 3.5 acceptance) -----------------------------------------------------------


@pytest.fixture
def noise_free(monkeypatch):
    """The synthetic index priced without its quote noise (``sample_chain(noise=False)``: each mid is the
    generator's model value, each spread its base width), the fixture the spec's probability and IV checks name.
    The market and its published surface are cached per valuation time, so both caches are cleared on the way in
    and out."""
    import functools

    from options_analyzer.market_data import sample_chain
    from options_analyzer.ui import surface_state

    def clear() -> None:
        mk.index_market.clear()
        surface_state._published.clear()

    monkeypatch.setattr(mk, "sample_chain", functools.partial(sample_chain, noise=False))
    clear()
    yield
    clear()


def test_on_the_noise_free_chain_p_itm_is_the_smiles_and_the_mid_is_the_fit(new_app, noise_free):
    # UI spec 3.1: P(ITM) (RN smile) within 0.5 pp of 57.3% (call) and 42.7% (put); this fit gives 57.1% and 42.9%.
    # N(d2) at the flat vol is 49.0% and 51.0% here (the spec's 49.2% and 50.8% came from an earlier fit).
    at = new_app(**INDEX).run()
    _ok(at)
    table = _frame(at, "tbl_pricing_greeks").value.set_index("Greek")
    call, put = (_number(table.loc["P(ITM)", f"{kind} (BSM)"]) for kind in ("Call", "Put"))
    assert abs(call - 57.3) <= 0.5 and abs(put - 42.7) <= 0.5 and (call, put) == (57.1, 42.9)
    assert any("N(d₂) at flat vol: 49.0% call, 51.0% put." in c.value for c in at.caption)
    # Without noise each mid is the model value, so the fit goes through it: Mid vs fit reads 0.00, in band, for both.
    contract = _frame(at, "tbl_pricing_contract").value.set_index("Type")
    assert list(contract["Mid vs fit"]) == [0.0, 0.0] and list(contract["Band"]) == ["in band", "in band"]


def test_on_the_noise_free_chain_the_iv_tiles_bracket_the_surface(new_app, noise_free):
    at = new_app(**INDEX, tab="IV").run()
    _ok(at)
    tiles = _tiles(at)
    bid, mid, ask, surface = (_number(tiles[k].value) for k in ("IV bid", "IV mid", "IV ask", "Surface"))
    assert bid <= mid <= ask and bid <= surface <= ask
    assert (bid, mid, ask, surface) == (15.36, 15.44, 15.51, 15.44)
    assert tiles["Surface"].delta == "mid vs fit 0.00 vp"


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


def test_the_iv_chart_marks_the_quotes_and_their_vols(new_app, default_index):
    # Each quote reads across to the curve and down to its vol; the legend names its price and vol. The vol axis
    # spans 1 vol point either side of the mid's vol (the bid and ask lines 0.17 vol points apart stand apart).
    at = new_app(**INDEX, tab="IV").run()
    spec = _chart(at, "iv_price")
    names = [t["name"] for t in spec["data"]]
    assert names == [
        "Model value (BSM)", "Bid 97.20 · 15.34%", "IV bid", "Mid 97.75 · 15.43%", "IV mid", "Ask 98.30 · 15.51%",
        "IV ask",
    ]  # fmt: skip
    d = default_index
    bid, ask = d["book"]["call"]
    mid = _implied((bid + ask) / 2, d["F"], d["K"], d["T"], d["DF"], "call") * 100
    assert spec["layout"]["xaxis"]["range"] == [pytest.approx(mid - 1.0, abs=1e-6), pytest.approx(mid + 1.0, abs=1e-6)]
    marker = next(t for t in spec["data"] if t["name"] == "IV ask")
    assert marker["x"][0] == pytest.approx(_implied(ask, d["F"], d["K"], d["T"], d["DF"], "call") * 100, abs=1e-6)
    # No floor band: the call's floor (zero, out of the money forward) is far below the window.
    assert not spec["layout"].get("shapes")


def test_an_offer_only_quote_inverts_its_ask_alone(new_app, default_index):
    # The 6,500 call on 20-Nov is offered at 0.10 with no bid. The chain, the contract table and the published surface
    # read a zero bid as none; the tab inverted it as a price ("no time value") and made a mid of 0.05, 1.78 vp over
    # the fit. Now the bid starts empty, only the ask is inverted, and mid vs fit reads "–".
    at = new_app(**INDEX, tab="IV", strike="6500").run()
    _ok(at)
    assert not at.warning
    assert [(n.label, n.value) for n in at.number_input if n.label in ("Bid", "Ask")] == [("Bid", None), ("Ask", 0.1)]
    d = default_index
    tiles = _tiles(at)
    assert tiles["IV bid"].value == "–" and tiles["IV bid"].proto.help == "No bid: the quote is an offer only."
    assert tiles["IV mid"].value == "–" and tiles["IV mid"].proto.help == "No mid: the quote is one-sided."
    assert tiles["IV ask"].value == f"{_implied(0.10, d['F'], 6500.0, d['T'], d['DF'], 'call'):.2%}"
    assert tiles["Surface"].delta == "mid vs fit –"
    assert "an offer only has no mid" in tiles["Surface"].proto.help
    assert [t["name"] for t in _chart(at, "iv_price")["data"]][1].startswith("Ask 0.10 · ")
    # A typed bid makes it two-sided again.
    next(n for n in at.number_input if n.label == "Bid").set_value(0.05).run()
    _ok(at)
    assert _tiles(at)["IV mid"].value != "–"


def test_the_pricing_view_reads_from_the_link(new_app):
    # UI spec 2.4 and 2.7 item 11: the profile, chain and Greeks controls and the diagnostics are bound to the URL.
    at = new_app(
        **INDEX, pr_metric="Vega", pr_x="Vol", pr_chain_win="All", pr_chain_stride="50 pts", pr_greeks="+2nd",
        pricing_diag="true",
    ).run()  # fmt: skip
    _ok(at)
    assert at.segmented_control(key="pr_metric").value == "Vega"
    assert at.segmented_control(key="pr_x").value == "Vol"
    assert at.segmented_control(key="pr_chain_win").value == "All"
    assert at.selectbox(key="pr_chain_stride").value == 50
    assert at.segmented_control(key="pr_greeks").value == "+2nd"
    assert any(d.proto.id.endswith("-tbl_pricing_engines") or "Engine" in d.value.columns for d in at.dataframe)


@pytest.mark.parametrize(("query", "where"), [({}, " (open below)"), ({"tab": "Risk"}, ", open there")])
def test_a_version_2_engine_link_opens_the_diagnostics_in_the_link(new_app, query, where):
    # Spec 2.4: an old link to a tree opens the closed form with the diagnostics open; on the Pricing tab the new link
    # holds pricing_diag (a link shared from that view opens them again), on another tab the session does (Streamlit
    # drops the parameter of a widget it does not draw).
    at = new_app(**INDEX, model="Binomial tree", **query).run()
    _ok(at)
    assert ("pricing_diag" in at.query_params) is not query
    assert at.session_state[tab_pricing.DIAGNOSTICS] is True
    assert any(f"Engine diagnostics{where}; the sidebar prices with" in w.value for w in at.warning)


def test_the_iv_tick_follows_the_index_rule():
    assert (tab_iv.tick(2.95, True), tab_iv.tick(3.0, True), tab_iv.tick(98.2, False)) == (0.05, 0.1, 0.01)


# --- The chart builders -----------------------------------------------------------------------------------------------


def test_the_error_chart_draws_each_lattices_order_beside_it():
    # CRR converges at first order, Leisen-Reimer at second: one slope −1 guide through the largest first error lay
    # on the CRR curves and said nothing of LR's. Each guide starts GUIDE_OFFSET above its own lattice's first error.
    steps = np.array([10, 100, 1000])
    fig = charts_pricing.error_chart(
        {"LR call": (steps, 1.0 / steps**2, "call", "solid"), "CRR call": (steps, 1.0 / steps, "call", "dot")},
        reference="BSM",
        guides=(("CRR", 1), ("LR", 2)),
    )
    assert fig.layout.xaxis.type == "log" and fig.layout.yaxis.type == "log"
    guides = {t.name: list(t.y) for t in fig.data[2:]}
    offset = charts_pricing.GUIDE_OFFSET
    assert guides == {
        "CRR: slope −1 (1/n)": pytest.approx([offset * 1e-1, offset * 1e-3]),
        "LR: slope −2 (1/n²)": pytest.approx([offset * 1e-2, offset * 1e-6]),
    }
    assert fig.layout.yaxis.title.text == "|Tree − BSM| ($)"


def test_the_monte_carlo_interval_narrows_as_one_over_root_n(new_app):
    # UI spec 3.1 item 7: the 95% interval's width against the paths, with a 1/√N guide, on the settings' seed. Each
    # width is 2 x 1.96 standard errors of monte_carlo.price_european at that sample, run here.
    at = new_app()
    at.session_state[tab_pricing.DIAGNOSTICS] = True
    at.run()
    _ok(at)
    spec = _chart(at, "pricing_mc_convergence")
    traces = {t["name"]: t for t in spec["data"]}
    paths = [1562, 6250, 25000, 100000]
    runs = [monte_carlo.price_european(100, 100, 1.0, 0.05, 0.2, 0.0, n_paths=n, seed=42) for n in paths]
    expected = {kind: [2 * 1.96 * run[kind].std_error for run in runs] for kind in ("call", "put")}
    # With the control variate the call and the put differ by a constant on the same draws: one error, one line (equal
    # to rounding: on Python 3.10 with numpy 1.26 the two standard errors differ in the last bit, 2e-17 of 0.069).
    np.testing.assert_allclose(expected["call"], expected["put"], rtol=1e-12)
    assert "Call" not in traces
    assert list(_values(traces["Call and put"]["x"])) == paths
    np.testing.assert_allclose(_values(traces["Call and put"]["y"]), expected["call"], rtol=1e-12)
    guide = _values(traces["1/√N"]["y"])
    assert guide[1] / guide[0] == pytest.approx(math.sqrt(paths[0] / paths[-1]))
    assert spec["layout"]["xaxis"]["type"] == "log" and spec["layout"]["yaxis"]["type"] == "log"


def test_the_profile_labels_do_not_print_over_one_another():
    # The default contract: spot 5,800, the forward 5,813.91 and the strike 5,815, a point apart on a ±10% axis,
    # under a legend of two curves. Drawn at the top of their lines the labels printed "FK 5,815", and the spot's
    # label printed over "Put" on a phone. On each row, the labels' boxes (and the legend's) do not overlap.
    marks = [(5800.0, "S 5,800.00", True), (5815.0, "K 5,815", False), (5813.91, "F", False)]
    legend = charts_pricing.legend_px(["Call", "Put"])
    rows = charts_pricing.label_rows(marks, 5220.0, 6380.0, legend)
    boxes: dict[int, list[tuple[float, float]]] = {0: [(0.0, legend)]}
    for (x, text, _), (row, left) in zip(marks, rows, strict=True):
        at = (x - 5220.0) / 1160.0 * charts_pricing.LABEL_PLOT_PX
        width = charts_pricing.LABEL_CHAR_PX * (len(text) + 1)
        box = (at - width, at) if left else (at, at + width)
        assert all(box[1] <= a or box[0] >= b for a, b in boxes.get(row, [])), (text, row)
        boxes.setdefault(row, []).append(box)
    assert rows[0][1] is True  # the spot's label reads to the left of its line
    # In the figure: one annotation a line, each on its row's offset.
    fig = charts_pricing.profile_chart(
        np.linspace(5220.0, 6380.0, 5),
        [charts_pricing.Line("Call", np.zeros(5), "call"), charts_pricing.Line("Put", np.zeros(5), "put")],
        x_title="Spot",
        y_title="Value ($ per share)",
        current=(5800.0, "S 5,800.00"),
        references=[(5815.0, "K 5,815", "strike"), (5813.91, "F", "forward")],
    )
    shown = {a.text: (a.yanchor, a.yshift or 0) for a in fig.layout.annotations}
    assert shown == {"S 5,800.00": ("top", 0), "K 5,815": ("top", 0), "F": ("bottom", 0)}


def test_exercise_boundary_colours_follow_the_option_type():
    days = np.array([0.0, 100.0])
    call, put = (days, np.array([130.0, 120.0])), (days, np.array([80.0, 90.0]))
    both = charts.exercise_boundary_chart({"Call boundary": call, "Put boundary": put}, 100.0)
    assert {t.name: t.line.color for t in both.data} == {"Call boundary": charts.CALL, "Put boundary": charts.PUT}


@pytest.mark.parametrize(("spot", "strike"), [("0.01", "0.01"), ("10000000", "0.01")])
def test_the_iv_tab_opens_inside_the_bounds_of_an_extreme_contract(new_app, spot, strike):
    # On a $0.01 spot and strike the bounds are 0.00049 and 0.01: in cents the price started at 0.00 and the tab
    # opened on "Price 0.00 is below the no-arbitrage floor $0.00". On a spot of 1e7 the cap rounded to itself and
    # the tab opened on "Price 10,000,000.00 is at or above the cap $10,000,000.00".
    at = new_app(tab="IV", spot=spot, strike=strike).run()
    _ok(at)
    S, K = float(spot), float(strike)
    lower, upper = max(S - K * math.exp(-0.05), 0.0), S  # q = 0: the call's no-arbitrage bounds, one year at 5%
    price = next(n for n in at.number_input if n.label == "Price")
    assert lower <= price.value < upper, (lower, price.value, upper)
    if S < 0.1:
        # Quoted to two significant digits of the cap (three decimals), in steps of a tenth of a cent.
        assert (price.proto.format, price.proto.step) == ("%.3f", pytest.approx(0.001))
        assert _tiles(at)["Bounds"].delta == f"floor {lower:.3f}"
