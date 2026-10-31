"""The Pricing tab in Chromium (UI spec 3.1 acceptance, roadmap PR-7, UI-1 and UI-7): where its tiles and profile
sit on a laptop and a phone, and a click on a row of the chain ladder, which AppTest cannot fire (UI spec 2.8: the
callback itself is unit-tested in tests/test_app_pricing.py).

Every page pins the valuation time (``asof=2026-10-23T16:00``), so the numbers do not depend on the day.
"""

from __future__ import annotations

import pytest
from test_theme import _settle, app_url, chromium  # noqa: F401  (fixtures)

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

INDEX = "/?asof=2026-10-23T16:00"

#: Boxes (viewport px) of the tiles, the profile chart, the footer and the Pricing contract's context line; the
#: main column's scroll height (Streamlit scrolls it, not the document) and how far the page scrolls sideways.
_LAYOUT = """() => {
  const box = s => { const e = document.querySelector(s); if (!e) return null; const r = e.getBoundingClientRect();
    return {top: r.top, bottom: r.bottom, left: r.left, right: r.right}; };
  const main = document.querySelector('[data-testid=stMain]');
  return {tiles: box('.st-key-pricing_tiles'), chart: box('.st-key-chart_pricing_profile [data-testid=stPlotlyChart]'),
          greeks: box('.st-key-pricing_views_greeks'),
          footer: box('.st-key-pricing_footer'), height: main.scrollHeight,
          sideways: document.scrollingElement.scrollWidth - innerWidth};
}"""


def _open(chromium, url: str, *, width: int, height: int, scheme: str = "dark"):  # noqa: F811
    page = chromium.new_page(viewport={"width": width, "height": height}, color_scheme=scheme)
    page.goto(url)
    page.locator(".st-key-chart_pricing_profile [data-testid=stPlotlyChart]").wait_for(state="visible", timeout=120_000)
    _settle(page)
    return page


@pytest.mark.parametrize("scheme", ["dark", "light"])
def test_the_answer_is_above_the_fold_on_a_laptop(chromium, app_url, scheme):  # noqa: F811
    # UI spec 3.1: at 1440 x 900 the tiles' top at most 180 px, the profile chart's bottom at most 900 px and the
    # page at most 1,800 px high with the diagnostics closed. Measured: 179 px, 900 px and 1,733 px (the context line
    # sits half the usual gap above the tiles; the profile and the Greeks table side by side; the ladder at 14 rows).
    # Before: 186 px and 2,492 px, with the 28-row ladder and the two views stacked.
    page = _open(chromium, app_url + INDEX, width=1440, height=900, scheme=scheme)
    try:
        layout = page.evaluate(_LAYOUT)
        assert layout["tiles"]["top"] <= 180, layout
        assert layout["chart"]["bottom"] <= 900, layout
        assert layout["height"] <= 1800, layout
        assert layout["sideways"] <= 0, layout
        # The Greeks table beside the profile, its top level with the profile's controls.
        assert layout["greeks"]["left"] > layout["chart"]["right"], layout
    finally:
        page.close()


def test_the_tab_fits_a_phone_without_sideways_scrolling(chromium, app_url):  # noqa: F811
    # UI spec 2.11: at 390 x 844 nothing scrolls sideways (the tables scroll inside themselves), and the six tiles go
    # in three pairs.
    page = _open(chromium, app_url + INDEX, width=390, height=844)
    try:
        assert page.evaluate(_LAYOUT)["sideways"] <= 0
        tops = page.evaluate(
            "() => [...document.querySelectorAll('.st-key-pricing_tiles [data-testid=stMetric]')]"
            ".map(c => Math.round(c.getBoundingClientRect().top))"
        )
        assert len(tops) == 6 and len(set(tops)) == 3, tops
    finally:
        page.close()


def _context_line(page) -> str:
    return page.locator("[role=tabpanel] [data-testid=stCaptionContainer]").first.inner_text()


def _call(page) -> str:
    card = page.locator(".st-key-pricing_tiles [data-testid=stMetric]").first
    return card.locator("[data-testid=stMetricValue]").inner_text()


def test_a_click_on_a_chain_row_prices_its_strike(chromium, app_url):  # noqa: F811
    # UI spec 3.1 acceptance: clicking the 5,850 row changes the context line to 5,850 and the CALL tile. The grid
    # draws on a canvas: the row's selection box is in its first column, rows 28 px under a 28 px header and 3 px of
    # frame. The default ladder runs 5,450 to 6,050 by 50 with 5,815 added, so 5,850 is its 10th row.
    page = _open(chromium, app_url + INDEX, width=1440, height=900)
    try:
        assert "SPX 20‑Nov‑26 AM 5,815 ·" in _context_line(page) and _call(page) == "97.81"
        grid = page.locator("[class*='st-key-tbl_pricing_chain_'] [data-testid=stDataFrame]")
        assert grid.count() == 1
        grid.scroll_into_view_if_needed()
        box = grid.bounding_box()
        page.mouse.click(box["x"] + 14, box["y"] + 3 + 28 + 9 * 28 + 14)
        page.wait_for_function(
            "() => [...document.querySelectorAll('[role=tabpanel] [data-testid=stCaptionContainer]')]"
            ".some(c => c.innerText.includes('AM 5,850 ·'))",
            timeout=60_000,
        )
        _settle(page)
        assert "SPX 20‑Nov‑26 AM 5,850 ·" in _context_line(page)
        assert _call(page) != "97.81"
        assert page.locator("[data-testid=stSidebar] input[aria-label=Strike]").input_value() == "5850"
    finally:
        page.close()


def test_a_click_on_a_row_beyond_the_surface_keeps_the_app(chromium, app_url):  # noqa: F811
    # The "All" window lists 4,200 to 4,500 with their quotes, beyond the smile's reach on 20-Nov: a click on the first
    # row crashed the app (the review's probe). Now the strike stays and the ladder says why.
    page = _open(chromium, app_url + INDEX + "&pr_chain_win=All", width=1440, height=900)
    try:
        grid = page.locator("[class*='st-key-tbl_pricing_chain_'] [data-testid=stDataFrame]")
        grid.scroll_into_view_if_needed()
        box = grid.bounding_box()
        page.mouse.click(box["x"] + 14, box["y"] + 3 + 28 + 14)
        page.get_by_text("4,200 lies beyond the surface's reach at this expiry").wait_for(timeout=60_000)
        _settle(page)
        assert page.locator("[data-testid=stException]").count() == 0
        assert page.locator("[data-testid=stSidebar] input[aria-label=Strike]").input_value() == "5815"
        assert _call(page) == "97.81"
    finally:
        page.close()


def test_the_iv_tiles_pair_on_a_phone(chromium, app_url):  # noqa: F811
    # The IV tab's six tiles in three pairs on a phone: IV bid beside IV ask, IV mid beside Surface, Vega beside Bounds
    # (in groups of three they went 2 + 1 twice, IV ask alone on its line).
    page = chromium.new_page(viewport={"width": 390, "height": 844}, color_scheme="light")
    try:
        page.goto(app_url + INDEX + "&tab=IV")
        page.locator(".st-key-iv_tiles [data-testid=stMetric]").first.wait_for(state="visible", timeout=120_000)
        _settle(page)
        rows = page.evaluate(
            "() => [...document.querySelectorAll('.st-key-iv_tiles [data-testid=stMetric]')].map(c => "
            "[Math.round(c.getBoundingClientRect().top), c.querySelector('[data-testid=stMetricLabel]').innerText])"
        )
        lines: dict[int, list[str]] = {}
        for top, label in rows:
            lines.setdefault(top, []).append(label.upper())
        assert list(lines.values()) == [["IV BID", "IV ASK"], ["IV MID", "SURFACE"], ["VEGA", "BOUNDS"]], rows
        assert page.evaluate("() => document.scrollingElement.scrollWidth - innerWidth") <= 0
    finally:
        page.close()
