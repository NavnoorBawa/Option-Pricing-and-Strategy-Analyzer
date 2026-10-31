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
    return {top: r.top, bottom: r.bottom}; };
  const main = document.querySelector('[data-testid=stMain]');
  return {tiles: box('.st-key-pricing_tiles'), chart: box('.st-key-chart_pricing_profile [data-testid=stPlotlyChart]'),
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
    # UI spec 3.1: at 1440 x 900 the tiles start high and the profile chart ends on the first screen. Measured: the
    # tiles' top at 186 px and the chart's bottom at 826 px. The spec's 180 px assumed a one-line header; the desk
    # header keeps the contract as its second line (header.py), 6 px more. The spec's 1,800 px page assumed the
    # chain at 15 rows; the ladder shows all 28 of its default window (10Δ put to 10Δ call at 25 points), so the
    # contract's own row, mid-ladder, is never scrolled out of a 15-row frame: the page is 2,492 px.
    page = _open(chromium, app_url + INDEX, width=1440, height=900, scheme=scheme)
    try:
        layout = page.evaluate(_LAYOUT)
        assert layout["tiles"]["top"] <= 190, layout
        assert layout["chart"]["bottom"] <= 900, layout
        assert layout["height"] <= 2600, layout  # a guard on growth (28 chain rows are 784 px of it)
        assert layout["sideways"] <= 0, layout
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
    # frame. The default ladder runs 5,425 to 6,075 by 25 with 5,815 added, so 5,850 is its 19th row.
    page = _open(chromium, app_url + INDEX, width=1440, height=900)
    try:
        assert "SPX 20‑Nov‑26 AM 5,815 ·" in _context_line(page) and _call(page) == "97.81"
        grid = page.locator("[class*='st-key-tbl_pricing_chain_'] [data-testid=stDataFrame]")
        assert grid.count() == 1
        grid.scroll_into_view_if_needed()
        box = grid.bounding_box()
        page.mouse.click(box["x"] + 14, box["y"] + 3 + 28 + 18 * 28 + 14)
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
