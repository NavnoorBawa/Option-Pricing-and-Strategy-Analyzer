"""Regenerate the README screenshots in docs/images from the running app.

Usage (from the repository root, with the app dependencies installed)::

    pip install playwright pillow && playwright install chromium
    python scripts/screenshots.py                    # writes docs/images/*.png
    python scripts/screenshots.py --chromium /path/to/chrome --out /tmp/shots

The script starts ``streamlit run main.py`` on a free port with the default sidebar inputs, opens each tab in a
1400 x 1700 dark-theme window, saves what is on screen, and stops the server. Re-run it whenever the interface
changes so that the README shows the app as it is.
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
#: Screenshot name: the tab's label and a chart that the tab draws when it runs (only the open tab runs, so a
#: click on a tab is complete once that chart is on the page).
TABS = {
    "pricing": ("Pricing", "sensitivity_chart"),
    "strategy": ("Strategy", "payoff_chart"),
    "risk": ("Risk", "var_hist"),
    "surface": ("Vol surface", "surf_iv"),
    "iv": ("IV", "iv_curve"),
}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_until_idle(page, settle: float = 2.0) -> None:
    """Wait until Streamlit has finished running the script, then let the charts draw."""
    for _ in range(240):
        if not page.locator('[data-testid="stStatusWidget"]').count():
            break
        time.sleep(0.5)
    time.sleep(settle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "images", help="output folder")
    parser.add_argument("--chromium", help="Chromium executable (default: the one Playwright installed)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "main.py", "--server.headless", "true", "--server.port", str(port)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=args.chromium)
            page = browser.new_page(viewport={"width": 1400, "height": 1700}, color_scheme="dark")
            for _ in range(120):  # the server needs a few seconds to start
                try:
                    page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
                    break
                except Exception:
                    time.sleep(0.5)
            page.wait_for_selector('[data-testid="stMetric"]', timeout=120_000)
            _wait_until_idle(page, 3.0)
            for name, (label, chart) in TABS.items():
                page.get_by_role("tab", name=label, exact=True).click()
                page.locator(f"[role=tabpanel] .st-key-{chart}").wait_for(state="visible", timeout=120_000)
                _wait_until_idle(page)
                path = args.out / f"{name}.png"
                page.screenshot(path=str(path))
                _shrink(path)
                print(f"saved {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
            if page.locator('[data-testid="stException"]').count():
                sys.exit("the app showed an exception; screenshots may be wrong")
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=30)


def _shrink(path: Path) -> None:
    """Quantise to a 256-colour palette when Pillow is available (screenshots shrink about 3x)."""
    try:
        from PIL import Image
    except ImportError:
        return
    with Image.open(path) as image:
        image.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)


if __name__ == "__main__":
    main()
