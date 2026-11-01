"""Measure what the app sends to the browser and how long it takes to settle, for the scorecard's desk-build rows.

Usage (from the repository root, with requirements-e2e.txt and Playwright's Chromium installed)::

    python scripts/app_payload.py                         # 3 repeats, median and range of each measure
    python scripts/app_payload.py --repeats 5 --chromium /path/to/chrome

The script starts ``streamlit run main.py`` on a free port and drives it in Chromium at 1,440 x 900 (dark) on the
synthetic index at the pinned valuation time of the README. For each measure it prints the median and range over the
repeats of

* the websocket bytes the server sent (Streamlit's protobuf messages as the page receives them, after the
  per-message compression is undone, so the figure does not depend on the compression level), and
* the time from the action to a settled page: Streamlit idle and no DOM change for 600 ms, the quiet window itself
  not counted.

Measures: the first load of each tab in a fresh page (the first one also cold, right after the server starts), a
Pricing strike step, a tab switch from Pricing to Strategy, a Strategy what-if spot step and a Strategy template
change. Timings depend on the machine and its load: the script prints the load average beside them.
"""

from __future__ import annotations

import argparse
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
#: The README's valuation time (scripts/screenshots.py), so the numbers do not change with the day they are taken.
QUERY = "?asof=2026-10-23T16:00"
TABS = ("Pricing", "Strategy", "Risk", "Surface", "IV", "Methodology")
#: Resolves with the time of the last DOM change once Streamlit is idle and nothing has changed for 600 ms.
SETTLED_JS = """() => new Promise(resolve => {
  let last = performance.now();
  const observer = new MutationObserver(() => { last = performance.now(); });
  observer.observe(document.body, {subtree: true, childList: true, attributes: true, characterData: true});
  const tick = () => {
    const running = document.querySelector('[data-testid="stStatusWidget"]');
    if (!running && performance.now() - last > 600) { observer.disconnect(); resolve(last); }
    else setTimeout(tick, 50);
  };
  setTimeout(tick, 100);
})"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Page:
    """A browser page that counts the websocket bytes it receives."""

    def __init__(self, browser, base: str):
        self.base = base
        self.page = browser.new_page(viewport={"width": 1440, "height": 900}, color_scheme="dark")
        self.received = 0
        self.page.on("websocket", lambda ws: ws.on("framereceived", self._count))

    def _count(self, payload) -> None:
        self.received += len(payload) if payload else 0

    def _settle(self, start_ms: float) -> float:
        """Seconds from ``start_ms`` (page clock) to the last DOM change before the page went quiet."""
        return (self.page.evaluate(SETTLED_JS) - start_ms) / 1000.0

    def load(self, query: str) -> tuple[float, float]:
        """Open ``query``; return (KB received, seconds to settle from navigation start)."""
        self.received = 0
        self.page.goto(self.base + query, wait_until="commit")
        self.page.wait_for_selector('[data-testid="stTab"]', timeout=180_000)
        seconds = self._settle(0.0)  # before the count: frames keep arriving until the page settles
        return self.received / 1024, seconds

    def act(self, action) -> tuple[float, float]:
        """Run ``action`` (a click) after the page is quiet; return (KB received, seconds to settle)."""
        self.page.wait_for_timeout(400)
        self.received = 0
        start = self.page.evaluate("performance.now()")
        action(self.page)
        seconds = self._settle(start)
        return self.received / 1024, seconds


def _step(key: str):
    return lambda page: page.locator(f".st-key-{key} [data-testid='stNumberInputStepUp']").first.click()


def _template(index: int):
    def pick(page):
        page.locator(".st-key-strategy input").first.click()
        page.get_by_role("option").nth(index).click()

    return pick


def _tab(label: str):
    return lambda page: page.get_by_role("tab", name=label, exact=True).click()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=3, help="repeats of each measure (default 3)")
    parser.add_argument("--chromium", help="Chromium executable (default: the one Playwright installed)")
    args = parser.parse_args()

    port = _free_port()
    base = f"http://127.0.0.1:{port}/"
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "main.py", "--server.headless", "true", "--server.port", str(port)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    results: dict[str, list[tuple[float, float]]] = {}

    def record(name: str, measure: tuple[float, float]) -> None:
        results.setdefault(name, []).append(measure)

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=args.chromium)
            first = Page(browser, base)
            for _ in range(120):  # the server needs a few seconds to start
                try:
                    record("Cold load, Pricing (first visit after start)", first.load(QUERY))
                    break
                except Exception:
                    time.sleep(0.5)
            first.page.close()
            for _ in range(args.repeats):
                for tab in TABS:
                    page = Page(browser, base)
                    record(f"First load, {tab}", page.load(QUERY + ("" if tab == "Pricing" else f"&tab={tab}")))
                    page.page.close()
                page = Page(browser, base)
                page.load(QUERY)
                record("Pricing strike step (+5)", page.act(_step("strike")))
                record("Tab switch, Pricing to Strategy", page.act(_tab("Strategy")))
                record("Strategy what-if spot step", page.act(_step("ws")))
                record("Strategy template change", page.act(_template(2)))
                page.page.close()
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=30)

    load = ", ".join(f"{value:.1f}" for value in os.getloadavg())
    print(f"{args.repeats} repeats, 1,440 x 900 dark, {QUERY}; load average {load}")
    print(f"{'Measure':48s} {'KB median (range)':>24s} {'s median (range)':>22s}")
    for name, values in results.items():
        kb, seconds = zip(*values, strict=True)
        print(
            f"{name:48s} {statistics.median(kb):9.1f} ({min(kb):.1f}-{max(kb):.1f}) "
            f"{statistics.median(seconds):9.2f} ({min(seconds):.2f}-{max(seconds):.2f})"
        )


if __name__ == "__main__":
    main()
