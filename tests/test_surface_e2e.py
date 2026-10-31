"""The Surface tab in Chromium (UI spec 3.4's Playwright acceptance tests): the smile draws one bid/ask bar per used
quote and dashes its extrapolated wings, a click on the expiry table moves the focus expiry (the smile's note and the
header's contract line follow), and the layout budget at 1,440 × 900 (tiles, smile and page height).

The app runs at the pinned valuation time (``?asof=2026-10-23T16:00``, the cash close of Friday 23 October 2026).
"""

from __future__ import annotations

from datetime import datetime

import pytest
from test_theme import app_url, chromium  # noqa: F401  (fixtures)

from options_analyzer.ui import market as mk
from options_analyzer.vol_surface import diagnostics as dg

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

AS_OF = datetime(2026, 10, 23, 16, 0)
#: The script has run to its end (see tests/test_desk_foundation_e2e.py).
_IDLE = "document.querySelector('[data-test-script-state]')?.getAttribute('data-test-script-state') === 'notRunning'"
#: The smile figure as Plotly holds it in the page.
_SMILE = "document.querySelector('.st-key-chart_smile .js-plotly-plot')"
#: The smile's note (top left inside the plot): the expiry, its time, RMSE, in band and quotes used.
_NOTE = f"() => ({_SMILE}?.layout?.annotations || []).map(a => a.text).find(t => t.endsWith(' used')) || ''"


def _open(chromium, url: str, *, width: int = 1440, height: int = 900):  # noqa: F811
    page = chromium.new_context(viewport={"width": width, "height": height}, color_scheme="light").new_page()
    page.goto(f"{url}/?asof=2026-10-23T16:00&tab=Surface")
    page.wait_for_function(f"() => {_IDLE} && !!({_SMILE}?.data)", timeout=120_000)
    return page


def _used(expiry: str) -> int:
    """Quotes the fit used at an expiry (ISO date), from the library: the expiry table's ``Used``."""
    fit = mk.index_market(AS_OF).fit
    label = next(i for i, d in enumerate(fit.expiries["expiry"]) if d.isoformat() == expiry)
    return int(((fit.quotes["label"] == label) & (fit.quotes["status"] == dg.USED)).sum())


def test_the_smile_has_one_bar_per_used_quote_and_a_dashed_extrapolation(chromium, app_url):  # noqa: F811
    page = _open(chromium, app_url)
    try:
        traces = page.evaluate(
            f"() => {_SMILE}.data.map(t => ({{name: t.name, n: t.x ? t.x.length : 0, dash: t.line && t.line.dash,"
            " bars: !!(t.error_y && t.error_y.array)}))"
        )
        bars = [t for t in traces if "bid–ask" in (t["name"] or "")]
        assert bars and all(t["bars"] for t in bars)
        assert sum(t["n"] for t in bars) == _used("2026-11-20") == 138
        extrapolated = next(t for t in traces if t["name"] == "Fit, extrapolated")
        assert extrapolated["dash"] not in (None, "solid")
    finally:
        page.context.close()


def test_the_tiles_and_the_smile_are_above_the_fold(chromium, app_url):  # noqa: F811
    """At 1,440 × 900 with the sidebar open: the tiles' top at most 260 px and the smile's at most 560 px (measured
    110 and 491 px; 595 px with the tiles in two rows of three), the six tiles on one line, and the page in the Smile
    view at most 2,000 px tall (measured 1,758 px; 3,111 px before the desk layout)."""
    page = _open(chromium, app_url)
    try:
        layout = page.evaluate(
            """() => {
              const top = s => document.querySelector(s).getBoundingClientRect().top;
              const tiles = [...document.querySelectorAll('.st-key-surface_tiles [data-testid=stMetric]')];
              const main = document.querySelector('[data-testid=stMain]');
              return {tiles: top('.st-key-surface_tiles'), smile: top('.st-key-chart_smile'),
                      lines: new Set(tiles.map(t => Math.round(t.getBoundingClientRect().top))).size,
                      count: tiles.length, height: main.scrollHeight};
            }"""
        )
        assert layout["count"] == 6 and layout["lines"] == 1, layout
        assert layout["tiles"] <= 260 and layout["smile"] <= 560, layout
        assert layout["height"] <= 2000, layout
    finally:
        page.context.close()


def test_a_row_click_moves_the_focus_expiry(chromium, app_url):  # noqa: F811
    """A click on the 18-Dec-26 row of the expiry table (the sixth) makes it the focus: the smile's note and the
    header's contract line (shown on every tab) name it."""
    page = _open(chromium, app_url)
    try:
        assert page.evaluate(_NOTE).startswith("20-Nov-26 AM · 27d 17h")
        box = page.locator(".st-key-tbl_surface_expiries_2026-11-20").bounding_box()
        # The header row and the rows are 28 px tall (components.ROW_HEIGHT); the selection column is the first.
        page.mouse.click(box["x"] + 20, box["y"] + 28 * 6 + 14)
        page.wait_for_function(f"() => ({_NOTE})().startsWith('18-Dec-26 AM')", timeout=60_000)
        page.wait_for_function(f"() => {_IDLE}", timeout=60_000)
        contract = page.locator(".st-key-contract_line").inner_text().replace("\xa0", " ").replace("‑", "-")
        assert contract.startswith("SPX 18-Dec-26 AM ")
        assert page.evaluate(_NOTE).endswith(f" · {_used('2026-12-18')} used")
    finally:
        page.context.close()
