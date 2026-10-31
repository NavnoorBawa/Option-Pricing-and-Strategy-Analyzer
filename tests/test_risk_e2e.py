"""The Risk tab in Chromium (UI spec 3.3 acceptance): its height and the place of its tiles and report, a click on a
scenario cell that sets the what-if strip (UI-10), the contrast of the stress bars' labels, the stress table's height
and no sideways scroll on a phone. On the synthetic index at the pinned valuation time, the app served by
tests/test_theme.py's ``app_url`` fixture.
"""

from __future__ import annotations

import re

import pytest
from test_theme import _settle, app_url, chromium  # noqa: F401  (fixtures)

from options_analyzer.ui import charts_risk, components, theme

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

ASOF = "asof=2026-10-23T16:00"

#: The page's height (Streamlit scrolls its main section), and the tops of the tiles, the report selector and the
#: first element under the selector, in page coordinates.
_MEASURE = """() => {
  const main = document.querySelector('[data-testid=stMain]') || document.documentElement;
  const top = e => e ? Math.round(e.getBoundingClientRect().top + main.scrollTop) : null;
  const pills = document.querySelector('.st-key-report');
  const tiles = [...document.querySelectorAll('.st-key-risk_tiles [data-testid=stMetric]')];
  let body = pills?.closest('[data-testid=stElementContainer]')?.nextElementSibling;
  while (body && body.getBoundingClientRect().height === 0) body = body.nextElementSibling;
  return {height: Math.max(document.documentElement.scrollHeight, main.scrollHeight),
          tiles: tiles.map(t => top(t)), report: top(pills), body: top(body),
          sideways: document.documentElement.scrollWidth > innerWidth || main.scrollWidth > main.clientWidth};
}"""


def _risk(chromium, app_url, query: str = "", *, width: int = 1440, scheme: str = "dark", touch: bool = False):  # noqa: F811
    context = chromium.new_context(
        viewport={"width": width, "height": 900 if width > 600 else 844},
        color_scheme=scheme,
        has_touch=touch,
        is_mobile=touch,
    )
    page = context.new_page()
    page.goto(f"{app_url}/?tab=Risk&{ASOF}{'&' + query if query else ''}")
    # The tiles' values, not their container: on a cold server the first run draws the page's skeleton (the
    # container with grey placeholders) before the market is built, and the script state can read idle meanwhile.
    page.locator("[role=tabpanel] .st-key-risk_tiles [data-testid=stMetricValue]").first.wait_for(
        state="visible", timeout=180_000
    )
    _settle(page)
    return page


def test_the_tab_is_one_screen_of_tiles_and_report(chromium, app_url):  # noqa: F811
    # Measured on this branch at 1,440 x 900 with the sidebar open: tiles at 369 px, the selector at 473 px, the
    # report at about 520 px, the page 1,159 px tall (5,106 px before, UI audit). The spec's 260 px for the tiles
    # assumes UI-1, which moves the four price cards (about 110 px) into the Pricing tab; until it merges they sit
    # above every tab, so the tiles' bound here is 260 + 120.
    page = _risk(chromium, app_url)
    try:
        found = page.evaluate(_MEASURE)
        assert len(found["tiles"]) == 6 and len(set(found["tiles"])) == 1, found  # one line of six
        assert found["tiles"][0] <= 380, found
        assert found["report"] <= 480, found
        assert found["body"] <= 600, found
        assert found["height"] <= 1_800, found
        assert not found["sideways"], found
    finally:
        page.context.close()


@pytest.mark.parametrize("report", ["Ladder", "Scenarios", "VaR"])
def test_no_sideways_scroll_on_a_phone(chromium, app_url, report):  # noqa: F811
    page = _risk(chromium, app_url, f"report={report}", width=390, touch=True)
    try:
        found = page.evaluate(_MEASURE)
        assert not found["sideways"], found
        # Three pairs of tiles, 2 + 2 + 2.
        assert len(set(found["tiles"])) == 3, found
    finally:
        page.context.close()


def test_a_click_on_a_scenario_cell_sets_the_what_if_strip(chromium, app_url):  # noqa: F811
    # UI-10: the (+5%, +2 vp) cell of the desk grid: row 4 of 7 (from -10 vp) and column 6 of 9 (from -10%), the
    # overlay's point 4 x 9 + 6 = 42.
    page = _risk(chromium, app_url, "report=Scenarios", scheme="light")
    try:
        points = page.locator(".st-key-sc_heatmap_True .scatterlayer .trace .points path")
        assert points.count() == 63
        box = points.nth(42).bounding_box()
        page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.locator(".st-key-whatif_scenario").wait_for(state="visible", timeout=60_000)
        _settle(page)
        strip = page.locator(".st-key-whatif").inner_text()
        assert "Spot +5.0%" in strip and "Vol +2.00 vp" in strip, strip
        assert "SCENARIO" in strip
        assert "what-if P&L" in page.locator(".st-key-risk_position").inner_text()
    finally:
        page.context.close()


#: The stress chart's amounts (the annotations right of the plot): their text, fill and box, with the boxes of the
#: plot area and of the scenario names (the y tick labels).
_BAR_LABELS = """() => {
  const chart = document.querySelector('.st-key-chart_stress_chart');
  const box = e => { const r = e.getBoundingClientRect(); return {l: r.left, r: r.right, t: r.top, b: r.bottom}; };
  const svg = chart.querySelector('svg.main-svg');
  return {
    svg: box(svg),
    plot: box(chart.querySelector('.cartesianlayer .bg, rect.nsewdrag')),
    names: [...chart.querySelectorAll('.ytick text')].map(box),
    labels: [...chart.querySelectorAll('.annotation-text')].map(t => ({
      label: t.textContent, text: getComputedStyle(t).fill, box: box(t)})),
  };
}"""


def _hex(css: str) -> str:
    red, green, blue = (int(float(v)) for v in re.findall(r"[\d.]+", css)[:3])
    return f"#{red:02X}{green:02X}{blue:02X}"


def _overlap(a: dict, b: dict) -> bool:
    return a["l"] < b["r"] and b["l"] < a["r"] and a["t"] < b["b"] and b["t"] < a["b"]


@pytest.mark.parametrize(("width", "scheme"), [(390, "light"), (1440, "dark")])
def test_stress_amounts_are_whole_and_clear_of_the_bars_and_names(chromium, app_url, width, scheme):  # noqa: F811
    # On a 390 px phone the amounts sat on the bars of a plot about 140 px wide: "−$3,35" for Black Monday's −$3,353
    # (its last digit clipped) and Tariff shock's "−$3,261" drawn over its own name. Now each amount is complete,
    # inside the chart, right of the plot area and clear of every name; in the text colour on the background
    # (spec 2.6, 4.5:1 at least).
    page = _risk(chromium, app_url, "report=Stress", width=width, scheme=scheme, touch=width < 600)
    try:
        found = page.evaluate(_BAR_LABELS)
        assert len(found["labels"]) == 14, found
        assert all(re.fullmatch(r"[+−]\$[0-9,]+(\.[0-9]{2})?", label["label"]) for label in found["labels"]), found
        surface = theme.SEMANTIC[scheme]["surface"]
        for label in found["labels"]:
            box = label["box"]
            assert box["l"] >= found["plot"]["r"] - 0.5 and box["r"] <= found["svg"]["r"] + 0.5, (label, found["plot"])
            assert not any(_overlap(box, name) for name in found["names"]), label
            assert charts_risk.contrast(_hex(label["text"]), surface) >= 4.5, label
        # The stress table shows every scenario without a scrollbar: the header and 14 rows of 28 px.
        # (the report's one table: a dataframe without selection carries no key class)
        tables = page.locator("[role=tabpanel] [data-testid=stDataFrame]")
        assert tables.count() == 1
        height = tables.first.bounding_box()["height"]
        assert abs(height - components.table_height(14)) <= 2, height
    finally:
        page.context.close()
