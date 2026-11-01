"""The README against the repository: its links resolve, its project layout lists every module, and its conventions
name the units the app shows."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _section(title: str) -> str:
    """The text of the README section headed ``## title``, up to the next section."""
    match = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", README, flags=re.MULTILINE | re.DOTALL)
    assert match, f"README has no section {title!r}"
    return match.group(1)


@pytest.mark.parametrize("target", sorted(set(re.findall(r"\]\(((?!https?://|#)[^)\s]+)\)", README))))
def test_every_local_link_and_image_in_the_readme_exists(target):
    # The version-1 audit moved from the repository root to docs/reviews; a link left behind would 404 on GitHub.
    assert (ROOT / target.split("#")[0]).exists(), target


def test_the_project_layout_lists_every_module_and_folder():
    # The layout left out backtest.py, attribution.py, margin.py and scripts/ after they were added.
    layout = _section("Project structure")
    listed = set(re.findall(r"^ *([\w./-]+?)/?(?: {2,}|$)", layout, flags=re.MULTILINE))
    package = ROOT / "options_analyzer"
    modules = {p.name for p in package.glob("*.py") if p.name != "__init__.py"}
    packages = {p.name for p in package.iterdir() if (p / "__init__.py").is_file()}
    folders = {"tests", "benchmarks", "scripts", "docs"}
    assert modules | packages | folders <= listed, sorted((modules | packages | folders) - listed)


def test_the_conventions_name_gamma_as_the_app_shows_it():
    # The README said gamma was shown per $1 spot move; the app shows Γ 1% (Δ per 1% move) on Pricing and Γ P&L 1%
    # (½Γ(0.01·S)², in $) on Strategy and Risk, and gamma per $1 nowhere.
    conventions = _section("Conventions")
    assert not re.search(r"per \$1 spot move \([^)]*gamma", conventions)
    ui = ROOT / "options_analyzer" / "ui"
    shown = {
        "Γ 1%": (ui / "tab_pricing.py").read_text(encoding="utf-8"),
        "Γ P&L 1%": (ui / "tab_risk.py").read_text(encoding="utf-8") + (ui / "tab_strategy.py").read_text("utf-8"),
    }
    for label, source in shown.items():
        assert f'"{label}"' in source, f"the app no longer labels gamma {label}"
        assert label.replace("%", " %") in conventions, f"the README does not name {label}"


def _screenshot_script() -> str:
    return (ROOT / "scripts" / "screenshots.py").read_text(encoding="utf-8")


def test_the_screenshot_script_waits_for_charts_the_tabs_still_draw():
    # The script clicks each tab and waits for one of its charts by Streamlit key: a key the tab no longer uses would
    # time the script out after two minutes.
    tabs = re.search(r"^TABS = \{\n(.*?)^\}", _screenshot_script(), flags=re.MULTILINE | re.DOTALL).group(1)
    keys = re.findall(r'\("\w+", "(\w+)"\)', tabs)
    assert len(keys) == 5
    ui = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "options_analyzer" / "ui").glob("*.py"))
    for key in keys:
        assert f'key="{key}"' in ui, f"no tab draws a chart with key {key!r}"


@pytest.mark.parametrize("image", sorted(set(re.findall(r"\]\((docs/images/[^)\s]+\.png)\)", README))))
def test_every_readme_screenshot_has_the_size_the_script_takes(image):
    # The README showed the removed version-2 interface in 1,400 x 1,700 frames that the current script (a 1,440 x
    # 1,000 window at a pinned valuation time) cannot make: the images were never regenerated after the redesign.
    # Images from the script have its window's size exactly; a hand-made or stale one almost never does.
    viewport = re.search(r'^VIEWPORT = \{"width": (\d+), "height": (\d+)\}', _screenshot_script(), flags=re.MULTILINE)
    header = (ROOT / image).read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    size = int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")  # the IHDR chunk
    assert size == (int(viewport.group(1)), int(viewport.group(2))), "regenerate it: python scripts/screenshots.py"
