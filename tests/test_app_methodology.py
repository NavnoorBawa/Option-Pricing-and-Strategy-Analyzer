"""The Methodology (ui/tab_methodology.py and its Markdown files, ui/methodology/*.md, package data): the sections of
the files, the tab and the drawer, the validation results and the references.

The browser checks (the maths renders, the tab's height with its sections collapsed, the drawer's position) are in
tests/test_theme.py, next to the other tests that drive the app in Chromium.
"""

from __future__ import annotations

import fnmatch
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from options_analyzer.ui import app as desk
from options_analyzer.ui import tab_methodology
from options_analyzer.ui.tab_methodology import parse_section, validation_results

ROOT = Path(__file__).resolve().parents[1]
FILES = [tab_methodology.FOLDER / f"{name}.md" for name in tab_methodology.SECTIONS]
SECTIONS = tab_methodology.sections()
TITLES = [section.title for section in SECTIONS]
BY_NAME = {section.name: section for section in SECTIONS}


def _ok(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def _nodes(node):
    for child in getattr(node, "children", {}).values():
        yield child
        yield from _nodes(child)


def _of_type(node, kind: str) -> list:
    return [n for n in _nodes(node) if getattr(n, "type", None) == kind]


def _tab(at: AppTest):
    return next(tab for tab in at.tabs if tab.label == "Methodology")


def _drawn_tabs(at: AppTest) -> set[str]:
    """The tabs the run drew something in: the open tab, and the Strategy tab (its controls are drawn on every run)."""
    return {tab.label for tab in at.tabs if list(_nodes(tab))}


# --- The files ------------------------------------------------------------------------------------------------------


def test_every_methodology_file_is_a_section_in_the_order_of_the_spec():
    # A file left out of SECTIONS would ship without ever being shown. The order is the UI spec's (section 3.6), with
    # the Strategy tab's own section before Risk.
    assert sorted(tab_methodology.FOLDER.glob("*.md")) == sorted(FILES)
    assert TITLES == [
        "Data and time", "Pricing engines", "Implied vol", "Surface", "Strategy", "Risk", "Conventions and units",
        "Validation", "References", "Known limitations",
    ]  # fmt: skip
    anchors = [section.anchor for section in SECTIONS]
    assert anchors == [re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") for title in TITLES]
    assert len(set(anchors)) == len(anchors)


def test_every_tab_but_the_methodology_has_its_own_section():
    assert set(tab_methodology.TAB_SECTIONS) == set(desk.TABS) - {"Methodology"}
    assert {BY_NAME[name].title for name in tab_methodology.TAB_SECTIONS.values()} == {
        "Pricing engines", "Strategy", "Risk", "Surface", "Implied vol",
    }  # fmt: skip


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_each_file_is_one_section_with_balanced_maths(path):
    # Each section is drawn on its own (in an expander), so it must close every formula it opens: a "$" left open
    # would turn the rest of the section into maths, or leave a raw "$" on the page.
    text = path.read_text(encoding="utf-8")
    assert text.startswith("## ") and text.endswith("\n") and not text.endswith("\n\n")
    section = parse_section(path.stem, text)
    plain = section.body.replace(r"\$", "")  # a literal dollar sign
    assert plain.count("$$") % 2 == 0, "unclosed display maths"
    assert plain.replace("$$", "").count("$") % 2 == 0, "unclosed inline maths"


def test_the_first_line_is_one_caption_of_at_most_140_characters():
    assert len(tab_methodology.INTRO) <= 140


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("### Section\nText.\n", "must start with an '## ' title"),
        ("## \nText.\n", "must start with an '## ' title"),
        ("## Title\n", "has no text under its title"),
        ("## Title\n\n  \n", "has no text under its title"),
        ("## Title\nText.\n## Another\nMore.\n", "has a second '# ' or '## ' heading"),
        ("## Title\nText.\n# Top\n", "has a second '# ' or '## ' heading"),
    ],
    ids=["no title", "empty title", "no text", "blank text", "second title", "top-level heading"],
)
def test_a_malformed_file_is_refused(text, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        parse_section("example", text)


def test_subsections_lists_and_maths_stay_in_the_body():
    section = parse_section("example", "## Title\n\n### One\n#### Detail\n* item\n\n### Two\nText $x$.\n")
    assert section.body == "### One\n#### Detail\n* item\n\n### Two\nText $x$."


def test_the_methodology_files_are_package_data():
    # The installed app reads them from the package: the wheel must ship them (setuptools package-data), and the
    # validation results that the report writes next to them.
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = config["tool"]["setuptools"]["package-data"]["options_analyzer.ui"]
    package = ROOT / "options_analyzer" / "ui"
    for path in [*FILES, tab_methodology.VALIDATION_FILE]:
        relative = path.relative_to(package).as_posix()
        assert any(fnmatch.fnmatch(relative, pattern) for pattern in patterns), relative


# --- The tab -------------------------------------------------------------------------------------------------------


def test_the_tab_renders_every_file_in_collapsed_sections(new_app):
    # DK-6 acceptance 1: AppTest renders every methodology file without an exception.
    at = new_app(tab="Methodology").run()
    _ok(at)
    assert _drawn_tabs(at) == {"Methodology", "Strategy"}
    tab = _tab(at)
    expanders = _of_type(tab, "expander")
    assert [e.label for e in expanders] == TITLES
    assert not any(e.proto.expanded for e in expanders), "every section starts collapsed"
    # Each section's Markdown is drawn verbatim, in its own expander, after an anchor that the contents link to.
    assert [e.markdown[0].value for e in expanders] == [section.body for section in SECTIONS]
    anchors = [m.value for m in tab.markdown if m.value.startswith("<div id=")]
    assert anchors == [f'<div id="{section.anchor}"></div>' for section in SECTIONS]


def test_the_contents_link_to_every_section(new_app):
    at = new_app(tab="Methodology").run()
    _ok(at)
    tab = _tab(at)
    assert tab.caption[0].value == tab_methodology.INTRO
    contents = next(m.value for m in tab.markdown if m.value.startswith("**Contents**"))
    assert re.findall(r"\[([^\]]+)\]\(#([\w-]+)\)", contents) == [(s.title, s.anchor) for s in SECTIONS]


def test_the_tab_is_drawn_only_when_open(app):
    # Only the open tab runs: the first visit opens the Pricing tab, and the Methodology's text is not drawn.
    assert not any(e.label in TITLES for e in _of_type(app._tree, "expander"))


# --- The drawer ----------------------------------------------------------------------------------------------------


def _dialog(at: AppTest):
    dialogs = _of_type(at._tree, "dialog")
    assert len(dialogs) <= 1
    return dialogs[0] if dialogs else None


def test_opened_from_risk_the_drawers_first_section_is_risk_and_open(new_app):
    # The UI spec's acceptance test (section 3.6).
    at = new_app(tab="Risk").run()
    at.button(key="desk_methodology").click().run()
    _ok(at)
    first = _of_type(_dialog(at), "expander")[0]
    assert first.label == "Risk" and first.proto.expanded


def test_the_header_button_has_a_keyboard_shortcut_named_in_its_tooltip(app):
    # The UI spec (sections 2.4 and 3.6): Alt+Shift+M opens the drawer. The browser test presses it
    # (tests/test_theme.py); the stylesheet hides Streamlit's hint inside the button, so the tooltip names it.
    button = app.button(key="desk_methodology")
    assert button.proto.shortcut == "alt+shift+m"  # as Streamlit stores it
    assert button.proto.help.endswith("Shortcut: Alt+Shift+M (Option+Shift+M on a Mac).")


@pytest.mark.parametrize("tab", desk.TABS)
def test_the_header_button_opens_the_drawer_over_the_open_tab(new_app, tab):
    at = new_app(tab=tab).run()
    _ok(at)
    assert _dialog(at) is None
    at.button(key="desk_methodology").click().run()
    _ok(at)
    dialog = _dialog(at)
    assert dialog is not None
    settings = dialog.proto.dialog
    assert settings.title == "Methodology" and settings.is_open and settings.dismissible
    assert settings.position == settings.DialogPosition.RIGHT  # a drawer at full height (Streamlit 1.65 and later)
    assert settings.width == settings.DialogWidth.MEDIUM  # up to 750 px, which the user can widen
    assert settings.icon == ":material/menu_book:"
    # The open tab's section first and open, then the others in their order, collapsed; the Methodology tab has no
    # section of its own. The same text as the tab, without the line of contents and its anchors.
    name = tab_methodology.TAB_SECTIONS.get(tab)
    order = sorted(SECTIONS, key=lambda s: s.name != name)
    expanders = _of_type(dialog, "expander")
    assert [e.label for e in expanders] == [s.title for s in order]
    assert [e.proto.expanded for e in expanders] == [s.name == name for s in order]
    assert [e.markdown[0].value for e in expanders] == [s.body for s in order]
    assert dialog.caption[0].value == tab_methodology.INTRO
    assert not any(m.value.startswith(("**Contents**", "<div id=")) for m in dialog.markdown)
    # The open tab stays open and stays in the link.
    assert _drawn_tabs(at) == {tab, "Strategy"}
    assert at.query_params.get("tab") in ((tab, [tab]) if tab != desk.TABS[0] else (None, tab, [tab]))
    # The next run (any input) does not open it again.
    at.run()
    _ok(at)
    assert _dialog(at) is None


# --- Validation results --------------------------------------------------------------------------------------------

REPORT = {
    "commit": "0123456789abcdef0123456789abcdef01234567",
    "date": "2026-11-02",
    "tables": [
        {"title": "Table one", "columns": ["Check", "Cases", "Result"], "rows": [["a", 3, "pass"], ["b", 2.5, None]]},
        {"title": "Table two", "columns": ["Model"], "rows": [["x"]], "note": "A note."},
    ],
}


@pytest.fixture
def results_file(tmp_path, monkeypatch):
    """Where the app reads the validation results: a file of a temporary folder, not written yet."""
    path = tmp_path / "validation_results.json"
    monkeypatch.setattr(tab_methodology, "VALIDATION_FILE", path)
    return path


def _validation(at: AppTest):
    [section] = [e for e in _of_type(_tab(at), "expander") if e.label == "Validation"]
    return section


def test_until_the_report_is_generated_the_section_says_so(new_app, results_file):
    at = new_app(tab="Methodology").run()
    _ok(at)
    section = _validation(at)
    assert [m.value for m in section.markdown] == [
        BY_NAME["validation"].body,
        f"**Results.** {tab_methodology.NOT_GENERATED}",
    ]
    assert not section.table and not section.warning and not section.caption


def test_the_section_shows_the_reports_tables_with_its_commit_and_date(new_app, results_file):
    results_file.write_text(json.dumps(REPORT), encoding="utf-8")
    at = new_app(tab="Methodology").run()
    _ok(at)
    section = _validation(at)
    assert section.caption[0].value == (
        "Computed by the validation report at commit `0123456789abcdef0123456789abcdef01234567` on 2026-11-02."
    )
    assert [m.value for m in section.markdown][1:] == ["**Table one**", "**Table two**"]
    first, second = (table.value for table in section.table)
    assert list(first.columns) == ["Check", "Cases", "Result"]
    assert first.values.tolist() == [["a", "3", "pass"], ["b", "2.5", ""]]
    assert second.values.tolist() == [["x"]]
    assert [c.value for c in section.caption[1:]] == ["A note."]


def test_a_backtest_table_keeps_its_zones_as_text_and_its_tags(new_app, results_file):
    # The shape the spec asks of the VaR backtest (section 3.6, VL-7a): the zone is a word, not a colour, and each row
    # says its history is synthetic. The renderer shows every cell as the report wrote it. (Made-up rows, to test the
    # renderer only: the app ships no results until the report has run.)
    backtest = {
        "title": "VaR backtest (SYN HIST)",
        "columns": ["Days", "Exceptions / expected", "Kupiec p", "Zone", "History"],
        "rows": [[250, "4 / 2.5", "0.37", "green", "SYN HIST"], [1000, "17 / 10.0", "0.04", "yellow", "SYN HIST"]],
    }
    results_file.write_text(json.dumps({**REPORT, "tables": [backtest]}), encoding="utf-8")
    at = new_app(tab="Methodology").run()
    _ok(at)
    [table] = _validation(at).table
    assert table.value["Zone"].tolist() == ["green", "yellow"]
    assert table.value["History"].tolist() == ["SYN HIST", "SYN HIST"]


def test_an_unreadable_results_file_is_reported_not_shown(new_app, results_file):
    results_file.write_text('{"commit": "xyz", "date": "2026-11-02", "tables": []}', encoding="utf-8")
    at = new_app(tab="Methodology").run()
    assert not at.exception
    section = _validation(at)
    [warning] = section.warning
    assert warning.value.startswith("The validation results file cannot be read, so no results are shown: ")
    assert "'commit' must be a git commit" in warning.value
    assert not section.table


def test_a_missing_results_file_reads_as_none(tmp_path):
    assert validation_results(tmp_path / "validation_results.json") is None


def test_a_results_file_reads_as_its_tables(tmp_path):
    path = tmp_path / "validation_results.json"
    path.write_text(json.dumps(REPORT), encoding="utf-8")
    results = validation_results(path)
    assert results.commit == REPORT["commit"] and results.date.isoformat() == "2026-11-02"
    assert [t.title for t in results.tables] == ["Table one", "Table two"]
    assert results.tables[0].rows == (("a", "3", "pass"), ("b", "2.5", ""))
    assert results.tables[1].note == "A note." and results.tables[0].note == ""


def _report(**changes) -> dict:
    report = json.loads(json.dumps(REPORT))
    for key, value in changes.items():
        if key.startswith("table_"):
            report["tables"][0][key[len("table_") :]] = value
        elif value is None:
            del report[key]
        else:
            report[key] = value
    return report


@pytest.mark.parametrize(
    ("report", "message"),
    [
        ("{not json", "it is not valid JSON"),
        ([], "it must hold a JSON object"),
        (_report(commit=None), "'commit' must be a git commit"),
        (_report(commit="0123ab"), "'commit' must be a git commit"),
        (_report(commit="0123ABC"), "'commit' must be a git commit"),
        (_report(date=None), "'date' must be a date written YYYY-MM-DD"),
        (_report(date="02/11/2026"), "'date' must be a date written YYYY-MM-DD"),
        (_report(date="2026-02-30"), "'date' 2026-02-30 is not a calendar date"),
        (_report(tables=[]), "'tables' must be a non-empty list"),
        (_report(tables=["x"]), "table 1 must be an object"),
        (_report(table_title=""), "table 1: 'title' must be a non-empty string"),
        (_report(table_columns=[]), "table 1: 'columns' must be a non-empty list"),
        (_report(table_rows=[]), "table 1: 'rows' must be a non-empty list"),
        (_report(table_rows=[["a", 1]]), "table 1: row 1 must be a list of 3 cells"),
        (_report(table_rows=[["a", 1, {"x": 1}]]), "table 1: row 1 must be text, a number or null, not dict"),
        (_report(table_note=""), "table 1: 'note' must be a non-empty string"),
    ],
    ids=[
        "not json", "not an object", "no commit", "short commit", "upper-case commit", "no date", "other date format",
        "no such date", "no tables", "table not an object", "empty title", "no columns", "no rows", "short row",
        "object cell", "empty note",
    ],
)  # fmt: skip
def test_a_malformed_results_file_is_refused(tmp_path, report, message):
    path = tmp_path / "validation_results.json"
    path.write_text(report if isinstance(report, str) else json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(message)):
        validation_results(path)


def test_the_shipped_results_file_if_any_is_readable():
    # The validation report writes the file (roadmap item VL-7); until then there is none. Once it exists, the app
    # must be able to show it.
    if tab_methodology.VALIDATION_FILE.exists():
        assert validation_results() is not None
    else:
        assert validation_results() is None


# --- References ----------------------------------------------------------------------------------------------------

REFERENCES = tab_methodology.FOLDER / "references.md"
#: An entry of the reference list: a Markdown list item, which may continue on indented lines.
_ENTRY = re.compile(r"^\* (.+?)(?=^\* |^### |\Z)", flags=re.MULTILINE | re.DOTALL)
_LINK = re.compile(r"\[([^\]]+)\]\((https://[^\s]+?)\)(?=[,.\s]|$)")
_DOI = re.compile(r"10\.\d{4,9}/\S+")


def _entries() -> list[str]:
    return [" ".join(entry.split()) for entry in _ENTRY.findall(REFERENCES.read_text(encoding="utf-8"))]


def _links() -> list[tuple[str, str]]:
    return [link for entry in _entries() for link in _LINK.findall(entry)]


def test_every_reference_has_authors_a_year_and_a_link():
    entries = _entries()
    assert len(entries) == 19
    for entry in entries:
        assert re.match(r"^[A-Z][^()]+ \((19|20)\d\d\)\. ", entry), entry
        assert _LINK.search(entry), f"no link: {entry}"


def test_doi_and_arxiv_links_say_where_they_go():
    # A DOI link reads doi:<DOI> and resolves through doi.org; an arXiv link reads arXiv:<id> and goes to its
    # abstract; any other link names its host.
    links = _links()
    dois = [(text, url) for text, url in links if text.startswith("doi:")]
    assert len(dois) == 13
    for text, url in dois:
        doi = text.removeprefix("doi:")
        assert _DOI.fullmatch(doi) and url == f"https://doi.org/{doi}", text
    arxiv = [(text, url) for text, url in links if text.startswith("arXiv:")]
    assert arxiv == [
        ("arXiv:cond-mat/0104295", "https://arxiv.org/abs/cond-mat/0104295"),
        ("arXiv:1204.0646", "https://arxiv.org/abs/1204.0646"),
    ]
    for text, url in links:
        if not text.startswith(("doi:", "arXiv:")):
            host = urllib.parse.urlsplit(url).netloc.removeprefix("www.")
            assert text.startswith(host), (text, url)


#: Names that the Methodology's other files cite, each with the author that the references list for it.
CITED = {
    "Black–Scholes": "Black, F.", "Merton": "Merton, R. C.", "Cox–Ross–Rubinstein": "Cox, J. C.",
    "Leisen–Reimer": "Leisen, D. P. J.", "Peizer–Pratt": "Peizer, D. B.", "McDonald–Schroder": "McDonald, R. L.",
    "Brent": "Brent, R. P.", "Acerbi–Tasche": "Acerbi, C.", "GARCH": "Bollerslev, T.", "Dupire": "Dupire, B.",
    "Gatheral & Jacquier": "Gatheral, J. and Jacquier, A.", "Gatheral's": "Gatheral, J. (2006)",
    "Lee's": "Lee, R. W.", "Zeliade": "De Marco, S. and Martini, C.", "Fritsch–Butland": "Fritsch, F. N.",
    "2026 severely adverse": "Federal Reserve System (2026)",
}  # fmt: skip


def test_every_cited_work_is_in_the_references():
    text = " ".join(path.read_text(encoding="utf-8") for path in FILES if path != REFERENCES)
    references = " ".join(_entries())
    for name, author in CITED.items():
        assert name in text, f"{name} is no longer cited: update CITED"
        assert author in references, f"{name} is cited but {author} is not in the references"


def _get(url: str, *, attempts: int = 3) -> tuple[int, bytes]:
    """Status and body of a GET request, retried on network errors and server errors (5xx), which pass, but not on
    a client error (4xx: the link is wrong)."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (link check)"})
    for attempt in range(attempts):
        last = attempt == attempts - 1
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            if exc.code < 500 or last:
                return exc.code, b""
        except OSError:
            if last:
                raise
        time.sleep(5.0 * (attempt + 1))
    raise AssertionError("unreachable")


@pytest.mark.network
@pytest.mark.slow
def test_every_reference_link_resolves():
    # Every DOI is registered (the DOI handle service, which does not depend on the publisher's site answering a
    # script), every arXiv identifier is a paper, and every other link answers. Meant for a nightly run.
    failures = []
    for text, url in _links():
        if text.startswith("doi:"):
            doi = text.removeprefix("doi:")
            status, body = _get(f"https://doi.org/api/handles/{urllib.parse.quote(doi, safe='/')}")
            if status != 200 or json.loads(body).get("responseCode") != 1:
                failures.append(f"{text}: not a registered DOI ({status})")
        elif text.startswith("arXiv:"):
            status, body = _get(f"https://export.arxiv.org/api/query?id_list={text.removeprefix('arXiv:')}")
            if status != 200 or b"<entry>" not in body or b"<title>Error</title>" in body:
                failures.append(f"{text}: no such arXiv paper ({status})")
        else:
            status, _ = _get(url)
            if status >= 400:
                failures.append(f"{url}: HTTP {status}")
    assert not failures, failures
