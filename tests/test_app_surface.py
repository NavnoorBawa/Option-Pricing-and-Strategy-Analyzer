"""Surface tab, end to end with streamlit.testing (headless): UI spec 3.4's acceptance tests that AppTest can run, the
callbacks of the expiry table (which AppTest cannot fire, so they are unit-tested), and uploads.

The index tests pin the valuation time (``_apptest.ASOF``, Friday 23 October 2026 at the cash close); the textbook
market is the default of ``new_app``.
"""

from __future__ import annotations

import base64
import io
import json
import math
import re
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _apptest import ASOF, make_app
from streamlit.testing.v1 import AppTest
from test_format_standard import HAND_FORMAT

from options_analyzer import vol_surface
from options_analyzer.market_data import sample_chain
from options_analyzer.ui import exports, surface_state, tab_surface, theme
from options_analyzer.ui import market as mk
from options_analyzer.ui.format import date as short_date
from options_analyzer.vol_surface import diagnostics as dg

AS_OF = datetime(2026, 10, 23, 16, 0)


def _ok(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@pytest.fixture(scope="module")
def index() -> AppTest:
    """The Surface tab of the synthetic index at the pinned valuation time, after its first run (read only)."""
    at = make_app(preset=None, asof=ASOF, tab="Surface").run()
    _ok(at)
    return at


@pytest.fixture
def app(new_app) -> AppTest:
    """The Surface tab of the textbook market (the parametric SSVI surface)."""
    at = new_app(tab="Surface").run()
    _ok(at)
    return at


def _metric(at: AppTest, label: str):
    return next(m for m in at.metric if m.label == label)


def _panel(at: AppTest) -> str:
    """The data panel's line: a chain's ``quoted → used``, a table of quotes' ``rows → used``."""
    lines = (m.value for m in at.markdown if m.value.startswith(":small["))
    return next(line for line in lines if re.search(r" (quoted|rows) → ", line))


def _expiries(at: AppTest) -> pd.DataFrame:
    return next(d.value for d in at.dataframe if {"Expiry", "Flags", "Used"} <= set(d.value.columns))


def _charts(at: AppTest) -> list[dict]:
    return [json.loads(c.proto.spec) for c in at.get("plotly_chart")]


def _array(values) -> np.ndarray:
    """A trace's array as sent: a list, or plotly's typed array (base64 bytes and a dtype)."""
    if isinstance(values, dict):
        return np.frombuffer(base64.b64decode(values["bdata"]), dtype=np.dtype(values["dtype"]))
    return np.asarray(values, dtype=float)


def _contract(at: AppTest) -> str:
    """The header's contract line (its spaces are no-break spaces, its hyphens no-break hyphens)."""
    lines = (m.value.replace("\xa0", " ").replace("\u2011", "-") for m in at.markdown)
    return next(line for line in lines if line.startswith("SPX ") and "· F " in line)


def _smile(at: AppTest) -> dict:
    return next(c for c in _charts(at) if any("bid–ask" in (t.get("name") or "") for t in c["data"]))


def _widget_keys(node) -> list[str]:
    """The keys of every element of an AppTest's tree that has one."""
    keys = [str(node.key)] if getattr(node, "key", None) else []
    for child in (getattr(node, "children", None) or {}).values():
        keys += _widget_keys(child)
    return keys


def _upload(at: AppTest, data: bytes, name: str = "quotes.csv") -> AppTest:
    at.segmented_control(key="surface_source").set_value(tab_surface.QUOTES).run()
    at.file_uploader(key="quotes_file").upload(name, data, "text/csv").run()
    return at


def _csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode()


# --- The default: the synthetic chain's published surface ------------------------------------------------------------


def test_the_data_panel_names_the_synthetic_source_and_its_counts_reconcile(index):
    panel = _panel(index)
    assert "Synthetic — not market data" in panel
    counts = {name: int(n.replace(",", "")) for n, name in re.findall(r"([\d,]+) (quoted|used)\b", panel)}
    inside = re.search(r"used \(([^)]*)\)", panel).group(1)  # the exclusions, by reason
    reasons = {name.strip(): int(n.replace(",", "")) for n, name in re.findall(r"([\d,]+) ([a-zA-Z ]+)", inside)}
    # The spec's example reads 5,020 → 2,363 (2,510 ITM side · 147 zero bid · 0 wide) at its commit; the code's
    # numbers, checked against parse_quotes in tests/test_surface_diagnostics.py, are these.
    assert counts == {"quoted": 5012, "used": 2357}
    assert reasons == {"zero bid": 149, "ITM side": 2506, "wide": 0}
    assert counts["quoted"] == counts["used"] + sum(reasons.values())
    assert "forward: parity · SVI arb‑free · mid" in panel


def test_the_tiles_are_the_desk_measures_of_the_published_surface(index):
    context = mk.index_market(AS_OF)
    surface = context.surface
    atmf = dg.atmf_vol(surface, 30 / 365)
    rr, bf = dg.risk_reversal(surface, 30 / 365, rate=mk.INDEX_CARRY[0])
    vix, _ = dg.vix_style_vol(context.fit, 30)
    tile = _metric(index, "ATMF 30D")
    assert tile.value == "15.38%" and tile.value == f"{atmf * 100:.2f}%"
    assert tile.delta == f"VIX‑style {vix * 100:.2f}%" == "VIX‑style 16.42%"
    assert _metric(index, "25Δ RR 30D").value == "−4.20 vp" == f"−{abs(rr) * 100:.2f} vp"
    assert _metric(index, "25Δ BF 30D").value == "+0.23 vp" == f"+{bf * 100:.2f} vp"
    assert _metric(index, "TERM 30→365D").value == "+1.62 vp"
    fit = _metric(index, tab_surface.FIT_LABEL)
    assert fit.value == "0.04 / 0.10" and fit.delta == "100% in band"
    assert _metric(index, "ARB").value == "0 / 0" and "0 / 0: passes" in _metric(index, "ARB").help
    assert len([m for m in index.metric if m.label in ("ATMF 30D", "ARB")]) == 2  # six tiles, the row of the spec


def test_the_expiry_table_has_the_twelve_columns_dates_with_settlement_and_no_years(index):
    table = _expiries(index)
    assert list(table.columns) == list(tab_surface.EXPIRY_HELP)
    assert len(table) == 16
    context = mk.index_market(AS_OF)
    names = [f"{short_date(d)} {s}" for d, s in context.expiries[["expiry", "settlement"]].itertuples(index=False)]
    assert list(table["Expiry"]) == names
    assert all(re.fullmatch(r"\d\d-[A-Z][a-z]{2}-\d\d (AM|PM)", e) for e in table["Expiry"])
    # The ATMF column is the surface at each expiry's parity forward, which the sidebar prices with.
    # (sent at the column's two decimals)
    np.testing.assert_allclose(table["ATMF"].to_numpy(), context.expiries["atm_vol"].to_numpy() * 100, atol=0.005)
    # Used: the screen's used quotes per expiry, which sum to the data panel's.
    assert table["Used"].sum() == 2357


def _texts_of_every_view(at: AppTest) -> list[str]:
    texts = [m.label for m in at.metric]
    for view in tab_surface.VIEWS:
        at.segmented_control(key="sf_view").set_value(view).run()
        _ok(at)
        for frame in at.dataframe:
            texts += [str(c) for c in frame.value.columns]
            texts += [str(c.get("label", "")) for c in json.loads(frame.proto.columns or "{}").values()]
            text = [c for c in frame.value.columns if not pd.api.types.is_numeric_dtype(frame.value[c])]
            texts += [str(v) for v in frame.value[text].to_numpy().ravel()]  # the text cells (SF-1: never "None")
        for chart in _charts(at):
            layout = chart["layout"]
            for key, axis in layout.items():
                if key.startswith(("xaxis", "yaxis")):
                    texts.append(str((axis.get("title") or {}).get("text", "")))
            for scene_axis in ("xaxis", "yaxis", "zaxis"):
                texts.append(str(((layout.get("scene") or {}).get(scene_axis) or {}).get("title", {}).get("text", "")))
    at.segmented_control(key="sf_view").set_value("Smile").run()
    return texts


def test_no_column_axis_title_or_tile_label_says_years(new_app):
    for at in (new_app(preset=None, asof=ASOF, tab="Surface").run(), new_app(tab="Surface").run()):
        _ok(at)
        texts = _texts_of_every_view(at)
        assert texts and not [t for t in texts if "year" in t.lower()], [t for t in texts if "year" in t.lower()]
        # SF-1: no literal None or nan in a cell, a label or a title (a missing value is the standard's –).
        assert not [t for t in texts if t.strip() in ("None", "nan", "NaN")]
        # Removed (UI spec 3.4): the colour-scale menu, two side-by-side 3-D charts and 3-D local vol, and "ATM means
        # K = S" (the desk's ATM is the forward).
        assert not [w for w in at.selectbox if "colour" in (w.label or "").lower()]
        assert not any("K = S" in c.value for c in at.caption)


def test_the_3d_view_is_one_opt_in_chart_of_implied_vol(new_app):
    at = new_app(preset=None, asof=ASOF, tab="Surface").run()
    assert not [c for c in _charts(at) if c["data"][0]["type"] == "surface"]  # not on the default view
    at.segmented_control(key="sf_view").set_value("3D").run()
    _ok(at)
    surfaces = [c for c in _charts(at) if c["data"][0]["type"] == "surface"]
    assert len(surfaces) == 1 and surfaces[0]["layout"]["scene"]["zaxis"]["title"]["text"] == "IV (%)"


def test_lowering_the_spread_limit_lowers_used_and_shows_the_fit_chip_everywhere(new_app):
    at = new_app(preset=None, asof=ASOF, tab="Surface").run()
    at.selectbox(key="sf_spread").set_value(1.0).run()
    _ok(at)
    panel = _panel(at)
    used = int(re.search(r"→ ([\d,]+) used", panel).group(1).replace(",", ""))
    wide = int(re.search(r"([\d,]+) wide", panel).group(1).replace(",", ""))
    assert used < 2357 and used + 2506 + 149 + wide == 5012 and wide == 1146
    assert any(m.value == ":yellow-badge[FIT]" for m in at.markdown)
    # The published surface follows the settings: the sidebar's contract reads the refitted surface.
    published = mk.index_market(AS_OF, _settings(spread=1.0))
    assert published.fit.counts()["used"] == used


def _settings(**changes) -> dg.FitSettings:
    filters = dg.QuoteFilters(max_spread=changes.get("spread", 4.0))
    return dg.FitSettings(filters, calibration=changes.get("calibration", "arbitrage_free"))


@pytest.mark.parametrize("tab", ["Pricing", "Strategy", "Risk", "Surface", "IV", "Methodology"])
def test_an_unconstrained_calibration_shows_the_fit_chip_on_every_tab_and_moves_no_price(new_app, tab):
    default = new_app(preset=None, asof=ASOF, tab=tab).run()
    at = new_app(preset=None, asof=ASOF, tab=tab)
    at.session_state["sf_cal"] = "Unconstrained"
    at.run()
    _ok(at)
    chip = next(b for b in at.markdown if b.value == ":yellow-badge[FIT]")
    assert chip is not None
    # The default chain's arbitrage-free fit never binds, so the unconstrained fit is the same surface: no number on
    # the tab's tiles moves (UI spec 3.4: "changes no number on Pricing"), on Pricing or any other tab.
    assert [(m.label, m.value) for m in at.metric] == [(m.label, m.value) for m in default.metric]
    assert not any(b.value == ":yellow-badge[FIT]" for b in default.markdown)


def test_the_pricing_vol_is_the_published_surface_at_the_contract(index):
    context = mk.index_market(AS_OF)
    published = surface_state.publish(context)
    row = context.expiry_row(context.default_expiry())
    K, expiry = context.atm_strike(row["expiry"]), row["expiry"]
    sidebar = context.surface.implied_vol_at(K / context.spot, float(row["maturity"]), extrapolate=True)
    assert published.vol(K, expiry).vol == pytest.approx(sidebar, abs=1e-10)
    # The contract line under the header shows it, and the readout on this tab reads the same surface.
    readout = next(c.value for c in index.caption if c.value.startswith("Pricing contract K"))
    assert f"σ {sidebar * 100:.2f}%" in readout and "inside quotes" in readout
    assert not [k for k in _widget_keys(index._tree) if "publish" in k]


def test_the_call_delta_axis_runs_from_90_to_10_delta(new_app):
    at = new_app(preset=None, asof=ASOF, tab="Surface").run()
    at.segmented_control(key="sf_x").set_value("Call Δ").run()
    _ok(at)
    layout = _smile(at)["layout"]
    axes = [v for k, v in layout.items() if k.startswith("xaxis")]
    assert any((a.get("title") or {}).get("text") == "Call delta" for a in axes)
    assert all(a.get("range") == [0.9, 0.1] for a in axes)
    # The vol axis spans what that window shows: every point drawn between 0.90 and 0.10 delta, with the headroom
    # for the note (24% of the span above, 5% below). It ran to over 30% for the far wings' offers, out of view.
    smile = _smile(at)
    points = [
        (_array(t["x"]), _array(t["y"]) + sign * _array(t.get("error_y", {}).get(arm, [0.0])))
        for t in smile["data"]
        if t.get("yaxis", "y") == "y"
        for sign, arm in ((1, "array"), (-1, "arrayminus"))
    ]
    y = np.concatenate([v[(x >= 0.1) & (x <= 0.9)] for x, v in points])
    y = y[np.isfinite(y)]
    lo, hi = y.min(), y.max()
    span = hi - lo
    np.testing.assert_allclose(layout["yaxis"]["range"], [lo - 0.05 * span, hi + 0.24 * span], rtol=1e-9)
    assert hi < 25  # the 20-Nov-26 smile between 0.90 and 0.10 delta (about 12% to 21% at the mids)


def test_the_smile_draws_one_bid_ask_bar_per_used_quote_and_dashes_the_extrapolated_fit(index):
    smile = _smile(index)
    bars = [t for t in smile["data"] if "bid–ask" in (t.get("name") or "")]
    count = sum(len(_array(t["x"])) for t in bars)
    table = _expiries(index)
    assert count == table.loc[3, "Used"] == 138
    extrapolated = next(t for t in smile["data"] if t.get("name") == "Fit, extrapolated")
    assert extrapolated["line"]["dash"] != "solid"
    note = next(a["text"] for a in smile["layout"]["annotations"] if "used" in a["text"])
    assert note.startswith("20-Nov-26 AM · 27d 17h<br>RMSE ") and note.endswith("100% in band · 138 used")


def test_every_status_of_the_quotes_view_is_used_or_a_reason(new_app):
    at = new_app(preset=None, asof=ASOF, tab="Surface").run()
    at.segmented_control(key="sf_view").set_value("Quotes").run()
    at.toggle(key="sf_all_quotes").set_value(True).run()
    _ok(at)
    quotes = next(d.value for d in at.dataframe if "Status" in d.value.columns and "IV mid" in d.value.columns)
    assert set(quotes["Status"]) <= {dg.USED, *dg.REASONS}
    assert (quotes["Status"] == dg.USED).sum() == _expiries(at).loc[3, "Used"]
    assert "Edge" not in " ".join(quotes.columns)  # against the fit it is Mid vs fit (spec 2.5)


def test_a_focus_expiry_from_the_table_moves_the_smile_and_the_contract(new_app):
    """The row click's callback writes the sidebar's expiry (tested below); here the run after it: the smile, the
    table's marker and the header's contract line follow."""
    at = new_app(preset=None, asof=ASOF, tab="Surface")
    at.session_state["expiry"] = "2026-12-18"
    at.run()
    _ok(at)
    note = next(a["text"] for a in _smile(at)["layout"]["annotations"] if "used" in a["text"])
    assert note.startswith("18-Dec-26 AM · 55d")
    assert _contract(at).startswith("SPX 18-Dec-26 AM ")


def test_the_row_click_and_the_step_buttons_set_the_focus_expiry(monkeypatch):
    state = {"tbl": {"selection": {"rows": [2], "columns": [], "cells": []}}, "expiry": "2026-11-20"}
    monkeypatch.setattr(tab_surface.st, "session_state", state)
    targets = ["2026-10-30", "2026-11-06", "2026-11-13", "2026-11-20"]
    tab_surface.on_expiry_select("tbl", targets, "expiry")
    assert state["expiry"] == "2026-11-13"
    state["tbl"] = {"selection": {"rows": [], "columns": [], "cells": []}}  # a click that clears the selection
    tab_surface.on_expiry_select("tbl", targets, "expiry")
    assert state["expiry"] == "2026-11-13"
    tab_surface.on_expiry_step(targets, "expiry", 3, +1)  # the last one: stays
    assert state["expiry"] == "2026-11-20"
    tab_surface.on_expiry_step(targets, "sf_focus", 0, -1)  # the first: stays
    assert state["sf_focus"] == "2026-10-30"
    tab_surface.on_expiry_step(list(range(8)), "sf_focus", 2, +1)
    assert state["sf_focus"] == 3


def test_the_term_view_has_forward_vols_and_the_grid_is_by_delta(new_app):
    at = new_app(preset=None, asof=ASOF, tab="Surface").run()
    at.segmented_control(key="sf_view").set_value("Term").run()
    _ok(at)
    strip = next(d.value for d in at.dataframe if "30→60d" in d.value.columns)
    windows = ["7→14d", "14→30d", "30→60d", "60→91d", "91→182d", "182→365d", "365→730d"]
    assert list(strip.columns) == ["Between", *windows]
    assert math.isnan(strip["365→730d"].iloc[0])  # beyond the last expiry (692 days): no extrapolation in time
    # The 25-delta lines are neither calls nor puts: not in the call or put colour (they were, put and call).
    term = next(c for c in _charts(at) if any(t.get("name") == "25Δ RR" for t in c["data"]))
    sides = {theme.SEMANTIC[mode][side] for mode in ("light", "dark") for side in ("call", "put")}
    assert not [t["name"] for t in term["data"] if t.get("name", "").startswith("25Δ") and t["line"]["color"] in sides]
    at.segmented_control(key="sf_view").set_value("Grid").run()
    _ok(at)
    grid = next(d.value for d in at.dataframe if "50Δ" in d.value.columns)
    columns = list(grid.columns)
    assert columns[0] == "Maturity" and columns[1] == "10Δp" and columns[9] == "50Δ" and columns[-1] == "10Δc"
    assert list(grid["Maturity"]) == ["7d", "14d", "30d", "60d", "91d", "182d", "365d"]  # 730 d: none


def _diagnostics(**state) -> AppTest:
    """The index's Surface tab with the Arbitrage and forwards expander open (and any other session state)."""
    at = make_app(preset=None, asof=ASOF, tab="Surface")
    at.session_state["surface_diag"] = True
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    _ok(at)
    return at


def test_the_forwards_read_by_the_expiry_tables_names_and_times_and_the_gap_by_tenths_of_a_bp():
    at = _diagnostics()
    expiries, forwards = _expiries(at), next(d.value for d in at.dataframe if "SE of F" in d.value.columns)
    assert list(forwards.columns[:2]) == ["Expiry", "DTE"]
    # One row per expiry, named and timed as the expiry table (they read 28 against 27d 17h for 20-Nov-26 AM).
    assert list(forwards["Expiry"]) == list(expiries["Expiry"]) and list(forwards["DTE"]) == list(expiries["DTE"])
    assert expiries["DTE"].iloc[3] == "27d 17h" and expiries["Expiry"].iloc[3] == "20-Nov-26 AM"
    # Carry gaps to 0.1 bp: the forwards' standard errors are 0.02 to 0.17 bp of the forward.
    for gaps in (expiries["Carry gap"], forwards["Gap"]):
        values = gaps.to_numpy(float)
        np.testing.assert_array_equal(values, np.round(values, 1))
    np.testing.assert_array_equal(expiries["Carry gap"].to_numpy(float), forwards["Gap"].to_numpy(float))


def test_forwards_from_the_rate_and_yield_have_no_carry_gap():
    """With Forwards = Rate and yield each forward is S·e^((r − q)T) at the reference carry the gap is measured
    against (the Treasury rate plus the box spread, 4.45%, and q 1.30%): every gap is 0 and none is flagged. They took
    the Treasury rate before, and read −1 to −7 bp on the first expiries against a help that called the reference
    their own."""
    at = _diagnostics(sf_fwd="Rate and yield")
    expiries = _expiries(at)
    assert (expiries["Carry gap"] == 0).all() and not expiries["Flags"].str.contains("carry").any()
    context = mk.index_market(AS_OF, dg.FitSettings(forward="rates"))
    r, q = mk.INDEX_CARRY
    T = context.expiries["maturity"].to_numpy(float)
    np.testing.assert_allclose(context.expiries["forward"], mk.INDEX_SPOT * np.exp((r - q) * T), rtol=1e-12)
    np.testing.assert_allclose(context.expiries["discount_factor"], np.exp(-r * T), rtol=1e-12)
    assert "4.45%" in tab_surface.EXPIRY_HELP["Carry gap"] and "1.30%" in tab_surface.EXPIRY_HELP["Carry gap"]


def test_durrlemans_g_covers_every_quoted_strike_of_the_focus_expiry():
    """The g(k) chart spans the focus expiry's quoted range as the smile does: it stopped at 4 ATM deviations, and
    left out K/F 79.1% to 84.3% of 20-Nov-26's quoted put wing."""
    at = _diagnostics()
    chart = next(c for c in _charts(at) if c["layout"]["yaxis"]["title"]["text"] == "g(k)")
    x = np.concatenate([_array(t["x"]) for t in chart["data"]])
    surface = mk.index_market(AS_OF).surface
    j = int(np.argmin(np.abs(surface.maturities * 365 - 27.7)))  # 20-Nov-26 AM, the focus
    lo, hi = surface.quoted_y[j]
    assert lo < -surface.reach_at(surface.maturities[j])  # the case: quotes beyond the reach
    assert x.min() <= lo and x.max() >= hi
    inside = next(t for t in chart["data"] if t["name"] == "g(k)")
    g = _array(inside["y"])
    assert np.nanmin(g) > 0  # the arbitrage-free fit: no butterfly arbitrage across the quotes


def test_the_tab_formats_numbers_only_through_the_standard_and_keeps_captions_short(index):
    lines = Path(tab_surface.__file__).read_text("utf-8").splitlines()
    assert not [n for n, line in enumerate(lines, 1) if HAND_FORMAT.search(line)]
    own = [c.value for c in index.main.caption if not c.value.startswith(("**Outlook", "Add this", "The link holds"))]
    assert own and all(len(c) <= 140 for c in own), [c for c in own if len(c) > 140]


def test_the_exports_carry_their_provenance(index, monkeypatch):
    context = mk.index_market(AS_OF)
    rows = tab_surface._chain_rows(context.fit, *mk.INDEX_CARRY, iso=True)
    prov = exports.Provenance(exports.SYNTHETIC, exports.SYNTHETIC_SOURCE, AS_OF, "Black-76", seed=0)
    shown = tab_surface.Shown(
        tab_surface.CHAIN, context.surface, context.fit, rows, *mk.INDEX_CARRY, AS_OF, "", (), prov, True
    )
    used = tab_surface._used_quotes(shown)
    assert len(used) == 2357 and "days" in used and "maturity" not in used
    payload = json.loads(tab_surface._svi_json(shown, prov))
    assert payload["provenance"]["source"].startswith("SYNTHETIC") and len(payload["expiries"]) == 16
    first = payload["expiries"][0]
    svi = context.surface.svi[0]
    assert first["expiry"] == "2026-10-30" and first["svi"]["a"] == svi.a and first["svi"]["sigma"] == svi.s
    grids = tab_surface._grid_frames(context.surface)
    assert grids["ImpliedVol"].index.name == "Days to expiry \\ Strike"
    labels = {b.label for b in index.get("download_button")}
    assert {"Implied vol grid (CSV)", "Used quotes (CSV)", "SVI parameters (JSON)", "The chain (CSV)"} <= labels


# --- The textbook market: the parametric SSVI surface ----------------------------------------------------------------


def test_the_textbook_surface_is_parametric_ssvi_on_constant_maturities(app):
    assert app.segmented_control(key="surface_source").value == tab_surface.SSVI
    assert app.segmented_control(key="surface_source").options == [tab_surface.SSVI, tab_surface.QUOTES]
    table = _expiries(app)
    assert list(table["Expiry"]) == [f"{d}d" for d in dg.CONSTANT_MATURITIES]
    assert "DTE" not in table.columns  # the name is the time to expiry ("7d" next to "7d 0h" before)
    # Forwards in index points, S·e^((r − q)T) at the sidebar's S 100, r 5% and q 0 (they read 1.00 to 1.11, in K/S,
    # before the surface carried the spot); sent at the column's two decimals.
    days = np.array(dg.CONSTANT_MATURITIES, dtype=float)
    np.testing.assert_allclose(table["F"].to_numpy(), 100 * np.exp(0.05 * days / 365), atol=0.005)
    params = vol_surface.SSVIParams(0.24, 0.20, 1.5, -0.6, 1.0, 0.5)  # the defaults at the sidebar's 20% vol
    assert _metric(app, "ATMF 30D").value == f"{math.sqrt(params.theta(30 / 365) / (30 / 365)) * 100:.2f}%"
    assert _metric(app, tab_surface.FIT_LABEL).value == "–"  # no quotes to fit
    # The footer and the readout describe a parametric surface: no SVI fit, no quotes (both said so before).
    captions = [c.value for c in app.caption]
    assert tab_surface.FOOTER_SSVI in captions and tab_surface.FOOTER not in captions
    readout = next(c for c in captions if c.startswith("Pricing contract K"))
    assert "inside its strike grid" in readout and "quotes" not in readout


def test_ssvi_atm_vols_follow_the_sidebar_volatility(app):
    assert app.number_input(key="ssvi_short").value == 24.0 and app.number_input(key="ssvi_long").value == 20.0
    app.sidebar.number_input(key="vol").set_value(40.0).run()
    _ok(app)
    assert app.number_input(key="ssvi_short").value == 48.0 and app.number_input(key="ssvi_long").value == 40.0
    app.number_input(key="ssvi_long").set_value(30.0).run()
    app.slider(key="ssvi_rho").set_value(-0.5).run()
    assert app.number_input(key="ssvi_long").value == 30.0  # an edit sticks until the sidebar volatility moves


# --- Uploads ----------------------------------------------------------------------------------------------------------


def test_the_empty_upload_state_lists_the_layouts_and_offers_the_synthetic_chain(new_app):
    """UI spec 3.4's empty state: the uploader, the accepted layouts one line each, a Use synthetic button and tiles
    that read –."""
    at = new_app(preset=None, asof=ASOF, tab="Surface").run()
    at.segmented_control(key="surface_source").set_value(tab_surface.QUOTES).run()
    _ok(at)
    text = next(m.value for m in at.markdown if m.value.startswith(":small[Upload a chain:"))
    assert text.count("\n- ") == len(tab_surface.LAYOUTS) + 1 == 4
    assert at.file_uploader(key="quotes_file") is not None
    assert {m.label: m.value for m in at.metric if m.label in ("ATMF 30D", "ARB")} == {"ATMF 30D": "–", "ARB": "–"}
    at.button(key="surface_use_synthetic").click().run()
    _ok(at)
    assert at.segmented_control(key="surface_source").value == tab_surface.CHAIN
    assert _metric(at, "ATMF 30D").value == "15.38%"


def test_an_uploaded_chain_is_screened_quote_by_quote_and_not_published(new_app):
    """A chain in the app's long layout (read_chain), at another level: parity gives its spot and forwards, the screen
    reconciles, and the other tabs keep the synthetic chain."""
    chain = sample_chain(6000.0, 0.04, 0.013, as_of=AS_OF, seed=5)
    at = new_app(preset=None, asof=ASOF, tab="Surface").run()
    _upload(at, _csv(chain.quotes), "my_chain.csv")
    _ok(at)
    panel = _panel(at)
    assert "Upload · `my_chain.csv`" in panel and "not published" in panel and "from parity" in panel
    quoted = int(re.search(r"([\d,]+) quoted", panel).group(1).replace(",", ""))
    assert quoted == len(chain.quotes)
    table = _expiries(at)
    np.testing.assert_allclose(table["F"].to_numpy(), chain.expiries["forward"].to_numpy(), rtol=2e-4)
    assert not any(m.value == ":yellow-badge[FIT]" for m in at.markdown)
    # The header's contract is still the synthetic chain's.
    assert _contract(at).endswith("F 5,813.91")


def test_a_table_of_quotes_still_builds_a_surface(app):
    _upload(app, _csv(vol_surface.sample_quotes(100, 0.05, 0.0, prices=True)))
    _ok(app)
    panel = _panel(app)
    assert "the user's table of quotes" in panel and "104 used" in panel
    assert _metric(app, "ARB").value == "0 / 0"
    assert list(_expiries(app)["Expiry"]) == ["30d", "60d", "91d", "182d", "273d", "365d", "547d", "730d"]


def test_parse_messages_stay_visible_when_the_fit_fails(app):
    frame = pd.DataFrame(
        {
            "Strike": [90, 95, 100, 105, 110, 100],
            "Days": [30, 30, 30, 30, 30, 60],
            "ImpliedVol": ["25%", "23%", "22%", "22%", "23%", "n/a"],
        }
    )
    _upload(app, _csv(frame))
    assert not app.exception
    assert any("two maturities" in e.value for e in app.error)
    captions = " ".join(c.value for c in app.caption)
    assert "Dropped 1 row(s)" in captions and "% signs" in captions
    assert all(m.value == "–" for m in app.metric if m.label in ("ATMF 30D", "ARB"))


def test_a_malformed_file_names_the_columns_found_and_needed(app):
    _upload(app, b"Price,When\n1,2\n")
    assert not app.exception
    assert any("found columns: Price, When" in e.value for e in app.error)


def test_quotes_far_from_the_sidebar_spot_are_flagged(app):
    quotes = vol_surface.sample_quotes(5000, 0.05, 0.0).rename(columns={"Maturity": "Maturity (years)"})
    _upload(app, _csv(quotes))
    _ok(app)
    assert any("far from the spot 100.00" in w.value for w in app.warning)


def test_a_guessed_maturity_unit_is_a_visible_warning(app):
    params = vol_surface.SSVIParams()
    rows = []
    for value in (6.65, 13.65, 20.65, 27.65):
        t = value / 365
        for m in np.linspace(0.8, 1.2, 9):
            iv = np.sqrt(params.total_variance(np.log(m) - 0.05 * t, t)[0] / t)
            rows.append({"Strike": round(100 * m, 4), "Maturity": value, "ImpliedVol": round(float(iv), 6)})
    _upload(app, _csv(pd.DataFrame(rows)))
    _ok(app)
    assert any("go up to 27.65, read as years; name the column 'Days'" in w.value for w in app.warning)


def test_an_arbitrage_free_calibration_holds_arbitrage_out_and_unconstrained_shows_it(app):
    quotes = vol_surface.sample_quotes(100, 0.05, 0.0)
    maturities = np.sort(quotes["Maturity"].unique())
    quotes.loc[quotes["Maturity"] == maturities[4], "ImpliedVol"] *= 0.8
    _upload(app, _csv(quotes))
    _ok(app)
    assert _metric(app, "ARB").value.startswith("0 / ")
    assert any("held free of static arbitrage" in w.value for w in app.warning)
    app.segmented_control(key="sf_cal").set_value("Unconstrained").run()
    _ok(app)
    assert int(_metric(app, "ARB").value.split(" / ")[0]) > 0
    assert any(m.value == ":yellow-badge[FIT]" for m in app.markdown)


def test_the_excel_workbook_is_only_built_when_downloaded(new_app, monkeypatch):
    # Writing every quote into the workbook on every rerun made each click on a large upload take seconds.
    built = []
    writer = pd.ExcelWriter

    def counting_writer(*args, **kwargs):
        built.append(1)
        return writer(*args, **kwargs)

    monkeypatch.setattr(pd, "ExcelWriter", counting_writer)
    at = new_app(preset=None, asof=ASOF, tab="Surface").run()
    at.segmented_control(key="sf_view").set_value("Term").run()
    _ok(at)
    assert not built
    context = mk.index_market(AS_OF)
    rows = tab_surface._chain_rows(context.fit, *mk.INDEX_CARRY, iso=True)
    prov = exports.Provenance(exports.SYNTHETIC, exports.SYNTHETIC_SOURCE, AS_OF, "Black-76", seed=0)
    shown = tab_surface.Shown(
        tab_surface.CHAIN, context.surface, context.fit, rows, *mk.INDEX_CARRY, AS_OF, "", (), prov, True
    )
    workbook = tab_surface._workbook(shown, prov)
    assert built
    with zipfile.ZipFile(io.BytesIO(workbook)) as archive:
        sheets = archive.read("xl/workbook.xml").decode()
    names = re.findall(r'<sheet name="([^"]+)"', sheets)
    assert names == ["Provenance", "ImpliedVol", "LocalVol", "Expiries", "Quotes"]


@pytest.mark.parametrize(
    ("n_bytes", "shown"),
    [(0, "0.0B"), (500, "500.0B"), (501, "0.5KB"), (1971, "2.0KB"), (500_000, "500.0KB"), (1_400_000, "1.4MB")],
)
def test_the_kept_file_size_reads_as_the_uploader_shows_it(n_bytes, shown):
    assert tab_surface._size(n_bytes) == shown


@pytest.mark.parametrize(
    ("valid", "quoted", "shown"),
    [(649, 652, "99.5%"), (1229, 1230, "99.9%"), (1, 1000, "0.1%"), (652, 652, "100%"), (0, 652, "0%"), (0, 0, "–")],
)
def test_local_vol_share_reads_100_or_0_only_when_exact(valid, quoted, shown):
    assert tab_surface._share(valid, quoted) == shown


def test_day_labels_tell_close_expiries_apart():
    assert tab_surface.day_labels(np.array([1, 2, 3, 30.0])) == ["1d", "2d", "3d", "30d"]
    assert tab_surface.day_labels(np.array([1.0, 1.4, 1.42])) == ["1.00d", "1.40d", "1.42d"]
