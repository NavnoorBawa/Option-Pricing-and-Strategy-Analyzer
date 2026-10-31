"""The desk foundation's shared pieces in Chromium (UI spec 2.6, 2.7 item 10, 2.8, 3.2 item 3): the what-if strip's
layout and amber state, the table helper's heights, input borders and the chart toolbars on a touch screen.

They run on tests/_desk_foundation_page.py, served from a temporary folder next to a link to the repository's
static/ folder (so that IBM Plex loads, as in the app: the strip's one-line budget is measured in that font), with
the repository as the working directory (so that the package imports and .streamlit/config.toml applies).
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from test_theme import _hex, chromium  # noqa: F401  (chromium is a fixture)

from options_analyzer.ui import components, theme

ROOT = Path(__file__).resolve().parents[1]
PAGE = Path(__file__).with_name("_desk_foundation_page.py")
#: The script has run to its end: "notRunning", which the app root reads only after a run (it reads "initial" before
#: the first, and a page waited on with "not running" alone was measured with one of its six inputs drawn).
_IDLE = "document.querySelector('[data-test-script-state]')?.getAttribute('data-test-script-state') === 'notRunning'"

pytestmark = [pytest.mark.e2e, pytest.mark.slow]


@pytest.fixture(scope="module")
def page_url(chromium, tmp_path_factory):  # noqa: F811
    folder = tmp_path_factory.mktemp("foundation_page")
    shutil.copy(PAGE, folder / "page.py")
    (folder / "static").symlink_to(ROOT / "static", target_is_directory=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    options = {"server.port": port, "server.address": "127.0.0.1", "server.headless": "true",
               "server.fileWatcherType": "none", "browser.gatherUsageStats": "false"}  # fmt: skip
    command = [sys.executable, "-m", "streamlit", "run", str(folder / "page.py")]
    command += [arg for key, value in options.items() for arg in (f"--{key}", str(value))]
    server = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        deadline = time.monotonic() + 120
        while True:
            try:
                direct.open(f"{url}/_stcore/health", timeout=2).close()
                break
            except OSError:
                if server.poll() is not None or time.monotonic() > deadline:
                    pytest.fail("the page did not start")
                time.sleep(0.25)
        yield url
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


def _open(chromium, url: str, *, width: int = 1440, scheme: str = "dark", touch: bool = False):  # noqa: F811
    context = chromium.new_context(
        viewport={"width": width, "height": 900}, color_scheme=scheme, has_touch=touch, is_mobile=touch
    )
    page = context.new_page()
    page.goto(url)
    page.wait_for_selector(".st-key-whatif", timeout=120_000)
    page.wait_for_function(
        f"() => {_IDLE} && !!document.querySelector('.st-key-chart_foundation_chart')", timeout=120_000
    )
    # The desk stylesheet reaches the page as an st.html element of its own, measured 0.6 to 1.0 s after the script
    # state reads idle on a loaded machine: wait for its rules rather than for a fixed time, or the amber rule and
    # the fonts are read before they apply.
    page.wait_for_function(_STYLED, timeout=60_000)
    # Every element the tests measure, drawn: the five number inputs (two in the sidebar, three in the strip) and
    # both tables, which load their grid separately and were once measured before it came.
    page.wait_for_function(_DRAWN, timeout=60_000)
    page.evaluate("document.fonts.ready.then(() => true)")
    page.wait_for_timeout(300)  # a settle for layout only; nothing below depends on it for correctness
    return context, page


_DRAWN = """() => document.querySelectorAll('[data-testid=stNumberInputContainer]').length >= 5
  && [...document.querySelectorAll('[data-testid=stDataFrame]')].filter(e => e.offsetHeight > 100).length === 2"""
#: True once the desk stylesheet's rules are in the document (its what-if rule is the marker).
_STYLED = """() => [...document.styleSheets].some(sheet => {
  try { return [...sheet.cssRules].some(r => (r.selectorText || '').includes('st-key-whatif_scenario')); }
  catch (e) { return false; }
})"""


#: The strip's widgets (each a flex item of the strip), their tops and bottoms, and every label's cut-off text.
_STRIP = """() => {
  const strip = document.querySelector('.st-key-whatif');
  const items = [...strip.children].filter(e => e.getBoundingClientRect().width > 0);
  const labels = [...strip.querySelectorAll('[data-testid=stWidgetLabel] p')];
  return {
    tops: items.map(e => Math.round(e.getBoundingClientRect().top)),
    bottoms: items.map(e => Math.round(e.getBoundingClientRect().bottom)),
    width: strip.getBoundingClientRect().width,
    clipped: labels.filter(p => p.scrollWidth > p.clientWidth + 1).map(p => p.innerText),
    wrapped: labels.filter(p => p.getBoundingClientRect().height > 1.5 * parseFloat(getComputedStyle(p).lineHeight))
               .map(p => p.innerText),
    rule: getComputedStyle(strip).borderLeftWidth + ' ' + getComputedStyle(strip).borderLeftColor,
    badge: !!strip.querySelector('.st-key-whatif_scenario'),
    plex: document.fonts.check('400 14px "IBM Plex Sans"'),
  };
}"""


@pytest.mark.parametrize("scheme", ["dark", "light"])
@pytest.mark.parametrize(
    "query",
    ["", "ws=-2&wv=3&wd=10&wr=25", "ws=-2.5&wv=-10&wvmode=relative&wd=38&wr=-50", "wv=10&wvmode=1M-scaled&wd=5"],
    ids=["rest", "scenario", "relative", "one-month"],
)
def test_the_what_if_strip_keeps_one_line_on_a_laptop(chromium, page_url, scheme, query):  # noqa: F811
    # UI spec 3.2 item 3: one line at rest, in the app's 1,038 px main column (1,440 px window, sidebar open); here
    # also with a scenario set in each vol mode, with no label wrapped or cut. While a shift is set, the amber rule
    # (the theme's yellowColor of each mode) and the SCENARIO badge; at rest neither.
    context, page = _open(chromium, f"{page_url}/?{query}", scheme=scheme)
    try:
        strip = page.evaluate(_STRIP)
        assert strip["plex"]
        assert 1000 <= strip["width"] <= 1040, strip["width"]
        assert max(strip["bottoms"]) - min(strip["tops"]) < 90, strip  # one row of labelled inputs
        assert strip["clipped"] == [] and strip["wrapped"] == [], strip
        active = bool(query)
        assert strip["badge"] is active
        amber = theme.AMBER[scheme]
        if active:
            width, color = strip["rule"].split(" ", 1)
            assert width == "3px" and _hex(color) == amber, strip["rule"]
        else:
            assert strip["rule"].startswith("0px"), strip["rule"]
    finally:
        context.close()


@pytest.mark.parametrize(("width", "query"), [(1440, "ws=-12.5&wv=-10&wd=39&wr=-500"), (390, "ws=-2&wv=3&wr=25")])
def test_the_strip_never_cuts_a_label_or_scrolls_sideways(chromium, page_url, width, query):  # noqa: F811
    # The longest labels (−12.5%, −10.00 vp, −500 bp: a rate below zero) may take a second line at 1,440 px, and a
    # phone stacks the strip; no label is ever cut and the page never scrolls sideways (WCAG 1.4.10).
    context, page = _open(chromium, f"{page_url}/?{query}", width=width, touch=width < 640)
    try:
        assert page.evaluate("document.scrollingElement.scrollWidth <= innerWidth")
        strip = page.evaluate(_STRIP)
        assert strip["clipped"] == [], strip
        assert "Rate −500 bp: 4.75% → −0.25%" in page.inner_text(".st-key-whatif") or width < 640
    finally:
        context.close()


def test_tables_take_their_content_height_up_to_fifteen_rows(chromium, page_url):  # noqa: F811
    # UI spec 2.8: height="content" up to 15 rows (no blank rows), then a fixed 15 rows that scroll; rows of 28 px.
    # components.table_height(15) must equal what "content" gives 15 rows, so that the two tables are as tall.
    context, page = _open(chromium, page_url)
    try:
        heights = page.evaluate(
            # The page's two tables, in order: 15 rows, then 20.
            "() => [...document.querySelectorAll('[data-testid=stDataFrame]')].map(e => e.offsetHeight)"
        )
        assert heights == [pytest.approx(components.table_height(15), abs=1)] * 2, heights
    finally:
        context.close()


def test_inputs_have_three_to_one_borders(chromium, page_url):  # noqa: F811
    # UI spec 2.6: #5B6577 on every input the user types into or picks from; a disabled one keeps the hairline.
    context, page = _open(chromium, page_url)
    try:
        colours = page.evaluate(
            """() => [...document.querySelectorAll(
                 '[data-testid=stNumberInputContainer], [data-testid=stSelectbox] > div > [role=group]')]
                 .map(e => [getComputedStyle(e).borderTopColor, !!e.querySelector('input:disabled')])"""
        )
        assert len(colours) >= 6
        for color, disabled in colours:
            assert (_hex(color) == theme.INPUT_BORDER) is not disabled, (color, disabled)
    finally:
        context.close()


def test_chart_toolbars_stay_hidden_after_a_tap_on_a_touch_screen(chromium, page_url):  # noqa: F811
    # UI spec 2.7 item 10 and 2.11: on a phone Plotly shows its toolbar always and Streamlit's element toolbar appears
    # on a tap; inside a chart's chart_KEY container neither may show. On a desktop the toolbar stays.
    hidden = """() => [...document.querySelectorAll(
        '.st-key-chart_foundation_chart :is(.modebar-container, [data-testid=stElementToolbar])')]
        .every(e => getComputedStyle(e).display === 'none' || e.getBoundingClientRect().width === 0)"""
    context, page = _open(chromium, page_url, width=390, touch=True)
    try:
        chart = page.locator(".st-key-chart_foundation_chart [data-testid=stPlotlyChart]")
        chart.scroll_into_view_if_needed()
        page.wait_for_selector(".st-key-chart_foundation_chart .modebar-container", state="attached", timeout=60_000)
        chart.tap()
        page.wait_for_timeout(500)
        assert page.evaluate(hidden)
    finally:
        context.close()
    context, page = _open(chromium, page_url)
    try:
        chart = page.locator(".st-key-chart_foundation_chart [data-testid=stPlotlyChart]")
        chart.scroll_into_view_if_needed()
        chart.hover()
        page.wait_for_timeout(500)
        assert not page.evaluate(hidden)
    finally:
        context.close()
