"""The Methodology tab and its Markdown files, one per area (ui/methodology/*.md, package data)."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

import pytest

from options_analyzer.ui import tab_methodology

ROOT = Path(__file__).resolve().parents[1]
FILES = [tab_methodology.FOLDER / f"{name}.md" for name in tab_methodology.SECTIONS]


def test_every_methodology_file_is_rendered():
    # A file left out of SECTIONS would ship without ever being shown.
    assert sorted(tab_methodology.FOLDER.glob("*.md")) == sorted(FILES)
    assert tab_methodology.SECTIONS == ("data", "pricing", "risk", "surface", "validation")


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_each_file_holds_whole_sections_with_balanced_maths(path):
    # The files are joined into one Markdown text, so each must start a section of its own and close every
    # formula it opens: a "$" left open in one file would turn the next file into maths.
    text = path.read_text(encoding="utf-8")
    assert text.startswith("### ") and text.endswith("\n") and not text.endswith("\n\n")
    plain = text.replace(r"\$", "")  # a literal dollar sign
    assert plain.count("$$") % 2 == 0, "unclosed display maths"
    assert plain.replace("$$", "").count("$") % 2 == 0, "unclosed inline maths"


def test_the_files_join_with_one_blank_line_between_sections():
    text = tab_methodology.methodology_markdown()
    assert text == "\n\n".join(path.read_text(encoding="utf-8").strip() for path in FILES)
    assert "\n\n\n" not in text
    headings = re.findall(r"^### (.+)$", text, flags=re.MULTILINE)
    assert headings[0] == "Conventions" and headings[-1] == "References"


def test_the_tab_renders_the_files_verbatim(app):
    methodology = next(tab for tab in app.tabs if "Methodology" in tab.label)
    assert [m.value for m in methodology.markdown] == [tab_methodology.methodology_markdown()]


def test_the_methodology_files_are_package_data():
    # The installed app reads them from the package: the wheel must ship them (setuptools package-data).
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = config["tool"]["setuptools"]["package-data"]["options_analyzer.ui"]
    package = ROOT / "options_analyzer" / "ui"
    for path in FILES:
        relative = path.relative_to(package).as_posix()
        assert any(fnmatch.fnmatch(relative, pattern) for pattern in patterns), relative
