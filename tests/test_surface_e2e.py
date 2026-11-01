"""The Surface tab in Chromium (UI spec 3.4's Playwright acceptance tests): the smile draws one bid/ask bar per used
quote and dashes its extrapolated wings, a click on the expiry table moves the focus expiry (the smile's note and the
sidebar's expiry follow), and the layout budget at 1,440 × 900 (tiles, smile and page height).

The app runs at the pinned valuation time (``?asof=2026-10-23T16:00``, the cash close of Friday 23 October 2026).
"""

from __future__ import annotations

import time
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
    # Idle, the smile drawn and the six tiles in the page (read once with four: the script state can read idle
    # before the last elements of the run are in the page).
    tiles = "document.querySelectorAll('.st-key-surface_tiles [data-testid=stMetric]').length === 6"
    page.wait_for_function(f"() => {_IDLE} && !!({_SMILE}?.data) && {tiles}", timeout=120_000)
    page.wait_for_timeout(500)  # the card row's stylesheet, an element of its own
    return page


def _used(expiry: str) -> int:
    """Quotes the fit used at an expiry (ISO date), from the library: the expiry table's ``Used``."""
    fit = mk.index_market(AS_OF).fit
    label = next(i for i, d in enumerate(fit.expiries["expiry"]) if d.isoformat() == expiry)
    return int(((fit.quotes["label"] == label) & (fit.quotes["status"] == dg.USED)).sum())


def test_the_smile_has_one_bar_per_used_quote_and_a_dashed_extrapolation(chromium, app_url):  # noqa: F811
    page = _open(chromium, app_url)
    try:
        # The bars as drawn: one error bar (bid to ask in vol) per point; only the used quotes' traces have them.
        drawn = page.evaluate("() => document.querySelectorAll('.st-key-chart_smile g.errorbar').length")
        assert drawn == _used("2026-11-20") == 138
        traces = page.evaluate(f"() => {_SMILE}.data.map(t => ({{name: t.name, dash: t.line && t.line.dash}}))")
        extrapolated = next(t for t in traces if t["name"] == "Fit, extrapolated")
        assert extrapolated["dash"] not in (None, "solid")
    finally:
        page.context.close()


def test_the_tiles_and_the_smile_are_above_the_fold(chromium, app_url):  # noqa: F811
    """At 1,440 × 900 with the sidebar open: the six tiles on one line, their top within the spec's 260 px, the
    smile's within its 560 px, and the page in the Smile view short.

    Measured (three loads, all alike): the tab bar ends at 126 px (the header holds no price cards since they moved
    into the Pricing tab), the data panel starts at 133 px and takes 87 px (the source, its line in two lines of small
    text, and Filters on one row), the tiles start at 220 px, the smile at 372 px, and the page is 1,640 px tall
    (3,111 px before the desk layout). The bounds leave about one line of small text (20 px) for the tiles, 28 px for
    the smile and 110 px (7%) for the page: the guards they replace (320, 560 and 2,000 px) still assumed the price
    cards in the header and let 100 px or more of regression through."""
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
        assert layout["tiles"] <= 240 and layout["smile"] <= 400, layout
        assert layout["height"] <= 1750, layout
    finally:
        page.context.close()


def test_a_row_click_moves_the_focus_expiry(chromium, app_url):  # noqa: F811
    """A click on the 18-Dec-26 row of the expiry table (the sixth) makes it the focus: the smile's note and the
    sidebar's expiry (the contract every tab prices) name it."""
    page = _open(chromium, app_url)
    try:
        assert page.evaluate(_NOTE).startswith("20-Nov-26 AM · 27d 17h")
        table = page.locator(".st-key-tbl_surface_expiries_2026-11-20")
        table.scroll_into_view_if_needed()  # below the fold at 900 px: a click there would land outside the page
        box = table.bounding_box()
        # The header row and the rows are 28 px tall (components.ROW_HEIGHT); the selection column is the first.
        page.mouse.click(box["x"] + 20, box["y"] + 28 * 6 + 14)
        page.wait_for_function(f"() => ({_NOTE})().startsWith('18-Dec-26 AM')", timeout=60_000)
        page.wait_for_function(f"() => {_IDLE}", timeout=60_000)
        expiry = page.locator("[data-testid=stSidebar] .st-key-expiry input").input_value()
        assert "18-Dec-26 AM · 55d" in expiry.replace("\u2011", "-"), expiry
        assert page.evaluate(_NOTE).endswith(f" · {_used('2026-12-18')} used")
    finally:
        page.context.close()


def test_a_step_of_the_focus_expiry_redraws_the_tab_in_a_loose_time(chromium, app_url):  # noqa: F811
    """A loose guard on the spec's 0.4 s budget for a change of the focus expiry: click to the smile's new note with
    the script idle, at most 2 s each (CI machines vary; this is not the budget).

    Measured at 1,440 × 900 on a 4-core machine shared with other jobs: 0.64 to 0.85 s a step (1.13 s for the first),
    of which the script run is 0.15 to 0.19 s (AppTest, warm) and the rest the browser's redraw. The spec's fragment
    does not apply: the focus expiry is the sidebar's (``?expiry=``), which sets the contract that every tab
    prices, so a step reruns the whole app."""
    page = _open(chromium, app_url)
    try:
        times = []
        for key in ("sf_next", "sf_next", "sf_prev", "sf_prev"):
            before = page.evaluate(_NOTE)
            start = time.perf_counter()
            page.locator(f".st-key-{key} button:visible").first.click()  # the stale copy of a rerun is hidden
            page.wait_for_function(f"() => ({_NOTE})() !== {before!r} && {_IDLE}", timeout=60_000)
            times.append(time.perf_counter() - start)
            page.wait_for_timeout(500)
        assert page.evaluate(_NOTE).startswith("20-Nov-26 AM")  # two steps forward and two back
        assert max(times) <= 2.0, times
    finally:
        page.context.close()
