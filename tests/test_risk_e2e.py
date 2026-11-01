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
  const selector = document.querySelector('.st-key-report');
  const tiles = [...document.querySelectorAll('.st-key-risk_tiles [data-testid=stMetric]')];
  let body = selector?.closest('[data-testid=stElementContainer]')?.nextElementSibling;
  while (body && body.getBoundingClientRect().height === 0) body = body.nextElementSibling;
  return {height: Math.max(document.documentElement.scrollHeight, main.scrollHeight),
          tiles: tiles.map(t => top(t)), report: top(selector), body: top(body),
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
    # Measured at 1,440 x 900 with the sidebar open, light and dark alike, on the default Ladder report: tiles at
    # 276 px, the selector at 380 px, the report at 422 px, the page 1,066 px tall (5,106 px before the UI audit).
    # The guards allow about 10 px over each top (a line's leading in another font build) and 8% over the height, so
    # that a regression of one line or more fails: the bounds were 380, 480, 600 and 1,800, set while the header's
    # four price cards (UI-1) sat above every tab, and no longer caught a 100 px regression.
    page = _risk(chromium, app_url)
    try:
        found = page.evaluate(_MEASURE)
        assert len(found["tiles"]) == 6 and len(set(found["tiles"])) == 1, found  # one line of six
        assert found["tiles"][0] <= 286, found
        assert found["report"] <= 390, found
        assert found["body"] <= 432, found
        assert found["height"] <= 1_150, found
        assert not found["sideways"], found
    finally:
        page.context.close()


def test_the_var_cards_are_one_line_under_the_tiles(chromium, app_url):  # noqa: F811
    # Six VaR cards (the simulation, its spot-only twin, the comparator and the gap) on one line at 1,440 px with the
    # sidebar open, as wide as the six tiles above: with "Spot-only ES 97.5% 1D" they went 4 + 2, the last two
    # stretched across the row.
    page = _risk(chromium, app_url, "report=VaR", scheme="light")
    try:
        boxes = page.evaluate(
            """() => [...document.querySelectorAll('.st-key-var_cards [data-testid=stMetric]')]
                 .map(e => { const r = e.getBoundingClientRect(); return [Math.round(r.top), Math.round(r.width)]; })"""
        )
        tiles = page.evaluate(
            """() => [...document.querySelectorAll('.st-key-risk_tiles [data-testid=stMetric]')]
                 .map(e => Math.round(e.getBoundingClientRect().width))"""
        )
        assert len(boxes) == 6 and len({top for top, _ in boxes}) == 1, boxes
        assert all(abs(width - tiles[0]) <= 1 for _, width in boxes), (boxes, tiles)
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
        # The report selector, a segmented control, on one line: its six segments wrapped 5 + 1 (Explain alone) at
        # the theme's padding; with desk.css's phone padding they take 323 px of the 362 px column, 28 px high.
        selector = page.locator(".st-key-report").bounding_box()
        assert selector["height"] <= 32 and selector["width"] <= 362, selector
    finally:
        page.context.close()


@pytest.mark.xfail(
    strict=True,
    reason="the tiles sit at 276 px, 16 px past the spec's 260: the caption under the what-if strip ('Sticky strike: "
    "leg vols fixed by strike.', 20 px and a 14 px gap), which the spec's layout puts on the strip's row",
)
def test_the_tiles_meet_the_specs_bound(chromium, app_url):  # noqa: F811
    # Spec 3.3 acceptance: the tiles' top at most 260 px at 1,440 x 900. UI-1 has moved the price cards into the
    # Pricing tab (they took the tiles to 369 px); what is left is the strip's caption line. Strict: once the caption
    # joins the strip's row this passes and fails the suite, so that the guard above is tightened and the marker
    # removed.
    page = _risk(chromium, app_url)
    try:
        assert page.evaluate(_MEASURE)["tiles"][0] <= 260
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


#: The stress chart's annotations (the amounts right of the plot and the scenario names above the bars): their
#: text, fill and box, with the boxes of the plot area, of the bars and of any y tick label.
_BAR_LABELS = """() => {
  const chart = document.querySelector('.st-key-chart_stress_chart');
  const box = e => { const r = e.getBoundingClientRect(); return {l: r.left, r: r.right, t: r.top, b: r.bottom}; };
  const svg = chart.querySelector('svg.main-svg');
  return {
    svg: box(svg),
    plot: box(chart.querySelector('.cartesianlayer .bg, rect.nsewdrag')),
    ticks: [...chart.querySelectorAll('.ytick text')].map(box),
    bars: [...chart.querySelectorAll('.barlayer .point path')].map(box),
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
    # (spec 2.6, 4.5:1 at least). The names, left of the plot as tick labels, then left the bars about 70 px of the
    # phone: they now sit above their bars, inside the plot and clear of every bar, and the plot takes the width.
    page = _risk(chromium, app_url, "report=Stress", width=width, scheme=scheme, touch=width < 600)
    try:
        found = page.evaluate(_BAR_LABELS)
        amounts = [label for label in found["labels"] if re.fullmatch(r"[+−]\$[0-9,]+(\.[0-9]{2})?", label["label"])]
        names = [label for label in found["labels"] if label not in amounts]
        assert len(amounts) == len(names) == len(found["bars"]) == 14, found
        assert not found["ticks"], found["ticks"]
        surface = theme.SEMANTIC[scheme]["surface"]
        plot = found["plot"]
        for label in amounts:
            box = label["box"]
            assert box["l"] >= plot["r"] - 0.5 and box["r"] <= found["svg"]["r"] + 0.5, (label, plot)
            assert not any(_overlap(box, name["box"]) for name in names), label
        for name in names:
            box = name["box"]
            assert box["l"] >= plot["l"] - 0.5 and box["r"] <= plot["r"] + 0.5, (name, plot)
            assert not any(_overlap(box, bar) for bar in found["bars"]), (name, found["bars"])
        for label in found["labels"]:
            assert charts_risk.contrast(_hex(label["text"]), surface) >= 4.5, label
        # The plot is most of the chart: 230 px of the phone's 358 px column, against about 70 before.
        assert plot["r"] - plot["l"] >= (200 if width < 600 else 800), plot
        # The stress table shows every scenario without a scrollbar: the header and 14 rows of 28 px.
        # (the report's one table: a dataframe without selection carries no key class)
        tables = page.locator("[role=tabpanel] [data-testid=stDataFrame]")
        assert tables.count() == 1
        height = tables.first.bounding_box()["height"]
        assert abs(height - components.table_height(14)) <= 2, height
    finally:
        page.context.close()
