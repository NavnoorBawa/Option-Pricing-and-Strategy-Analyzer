"""Methodology: the data, models, conventions, validation and references behind every number in the app.

The text is Markdown with LaTeX maths in ``methodology/`` (package data), one file per section (:data:`SECTIONS`),
so that each area documents its own models in its own file. A file starts with its section's ``## `` title; longer
sections divide into ``### `` subsections (:func:`parse_section`).

One renderer draws the Methodology in two places:

* the Methodology tab (:func:`render`): a line of contents that links to every section, then the sections,
  collapsed, so that the whole reference fits in about a screen. A link to a section, from the contents or from
  another page (``?tab=Methodology#references``), opens it; printing the tab prints every section, open
  (``static/desk.css``);
* the drawer that the header's Methodology button opens on the right, over the open tab, which stays as it was
  (:func:`drawer`). Streamlit has no way to open a dialog scrolled to a section, so the drawer starts with the open
  tab's own section (:data:`TAB_SECTIONS`), already open, followed by the others, collapsed.

The Validation section ends with the results of the validation report, read from ``methodology/
validation_results.json`` (:func:`validation_results`) with the commit and the date of its run. Until the report
has been generated, it says so: the app never shows results that were not computed.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from .format import right_aligned

#: Folder of the Methodology files.
FOLDER = Path(__file__).with_name("methodology")
#: The files, one per section, in reading order: where the inputs come from, the models from the option price up to
#: the position's risk, the conventions, then the evidence (validation, references and limitations).
SECTIONS = (
    "data", "pricing", "iv", "surface", "strategy", "risk", "conventions", "validation", "references", "limitations",
)  # fmt: skip
#: The section of each tab but the Methodology itself, which the drawer opens first (app.TABS names the tabs).
TAB_SECTIONS = {"Pricing": "pricing", "Strategy": "strategy", "Risk": "risk", "Surface": "surface", "IV": "iv"}
#: The section that ends with the validation report's results.
VALIDATION = "validation"
#: The validation report's results, written by the report (scripts/validation_report.py, roadmap item VL-7).
VALIDATION_FILE = FOLDER / "validation_results.json"
#: The first line of the Methodology (at most 140 characters, as every caption).
INTRO = (
    "How every number in the app is computed: data, models, conventions, validation and references. Click a "
    "section to open it."
)
#: What the Validation section says until the report has been generated.
NOT_GENERATED = "The validation report has not been generated for this version of the app, so there are no results yet."
#: The commit of a report run: 7 to 40 lowercase hexadecimal digits, as git writes it.
_COMMIT = re.compile(r"[0-9a-f]{7,40}")
#: A cell of a results table that reads as a number, which the table right-aligns: a JSON number as Python writes
#: it, or text such as 0.37, −2.5e-4, 1,000, 12 %, 3 bp, <0.001, or 4 / 2.5 (a count over its expected value).
_NUMBER = r"[<>≤≥]?\s?[−+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][−+-]?\d+)?(?:\s?(?:%|bp))?"
_NUMERIC = re.compile(rf"{_NUMBER}(?:\s?/\s?{_NUMBER})?")
#: The tab's links to its sections, a script that st.html runs in the page. The browser alone would scroll to a
#: section that is still collapsed, and not at all for a link from another page, since Streamlit draws the anchors
#: after the browser has looked for them. So a link of the line of contents, or a link to the tab with a section's
#: anchor (?tab=Methodology#references), opens that section and brings it to the top, without a script run. A click
#: leaves the address as it was, and the anchor of a link is removed once used, so that it does not stay in the
#: address on the other tabs. (``static/desk.css`` keeps the anchors clear of Streamlit's toolbar band.)
_ANCHORS = """<script>
(() => {
  // Open the section whose anchor is `id` (its title in lower case and hyphens) and keep it at the top of the page
  // for a second, while Streamlit's animation opens it: until then the page may be too short to scroll that far.
  const show = (id) => {
    const anchor = /^[a-z0-9-]+$/.test(id) && document.querySelector(`.st-key-methodology [id="${id}"]`);
    const details = anchor && anchor.closest("[data-testid=stVerticalBlock]").querySelector("details");
    if (!details) return false;
    if (!details.open) details.querySelector("summary").click();
    const top = new ResizeObserver(() => anchor.scrollIntoView({block: "start"}));
    top.observe(details);
    setTimeout(() => top.disconnect(), 1000);
    return true;
  };
  // A link from another page. The script can run before Streamlit has drawn the sections: look again for 5 s.
  const fromAddress = (tries = 50) => {
    const id = location.hash.slice(1);
    if (show(id)) history.replaceState(history.state, "", location.pathname + location.search);
    else if (id && tries > 0) setTimeout(() => fromAddress(tries - 1), 100);
  };
  fromAddress();
  if (window.methodologyLinks) return;  // the listeners, once per page: the script runs each time the tab opens
  window.methodologyLinks = true;
  addEventListener("hashchange", () => fromAddress());
  document.addEventListener("click", (event) => {
    const link = event.target.closest && event.target.closest(".st-key-methodology a[href^='#']");
    const plain = event.button === 0 && !(event.metaKey || event.ctrlKey || event.shiftKey || event.altKey);
    if (link && plain && show(link.hash.slice(1))) event.preventDefault();
  });
})();
</script>"""


@dataclass(frozen=True)
class Section:
    """One Methodology file: its name, title, anchor and Markdown text.

    Attributes
    ----------
    name : str
        The file's name without ``.md`` (one of :data:`SECTIONS`).
    title : str
        The section's title, for example ``Pricing engines``: the label of its expander.
    anchor : str
        The section's anchor in the tab, its title in lower case with a hyphen for each run of other characters
        (``pricing-engines``): the tab's contents link to ``#pricing-engines``.
    body : str
        The section's Markdown, without its title.
    """

    name: str
    title: str
    anchor: str
    body: str


@dataclass(frozen=True)
class ResultsTable:
    """One table of the validation report: its title, column names, rows of text and an optional note."""

    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    note: str = ""


@dataclass(frozen=True)
class ValidationResults:
    """The validation report's results: the commit and the date of its run, and its tables."""

    commit: str
    date: dt.date
    tables: tuple[ResultsTable, ...]


def parse_section(name: str, text: str) -> Section:
    """Read one Methodology file: its ``## `` title, then its text.

    Parameters
    ----------
    name : str
        The file's name without ``.md``.
    text : str
        The file's Markdown: a ``## `` title on the first line, then the section's text, which may divide into
        ``### `` subsections (or deeper ones) but has no ``# `` or ``## `` heading of its own.

    Returns
    -------
    Section
        The section, with its text stripped of surrounding blank lines.

    Raises
    ------
    ValueError
        If the file does not start with a title, has no text, or has a second title.

    Examples
    --------
    >>> section = parse_section("pricing", "## Pricing engines\\n\\n### Monte Carlo\\nExact $S_T$ sampling.\\n")
    >>> section.title, section.anchor, section.body
    ('Pricing engines', 'pricing-engines', '### Monte Carlo\\nExact $S_T$ sampling.')
    """
    first, _, body = text.strip().partition("\n")
    if not first.startswith("## ") or not first[3:].strip():
        raise ValueError(f"{name}.md must start with an '## ' title")
    if not body.strip():
        raise ValueError(f"{name}.md has no text under its title")
    if re.search(r"^##? ", body, flags=re.MULTILINE):
        raise ValueError(f"{name}.md has a second '# ' or '## ' heading: subsections take '### '")
    title = first[3:].strip()
    return Section(name, title, re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-"), body.strip("\n"))


def sections() -> tuple[Section, ...]:
    """The Methodology, section by section: every file of :data:`SECTIONS`, in order (:func:`parse_section`).

    Returns
    -------
    tuple of Section

    Examples
    --------
    >>> [section.title for section in sections()][:3]
    ['Data and time', 'Pricing engines', 'Implied vol']
    """
    return tuple(parse_section(name, (FOLDER / f"{name}.md").read_text(encoding="utf-8")) for name in SECTIONS)


def _text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string")
    return value.strip()


def _cell(value: object, where: str) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float)):  # bool is an int: it reads True or False
        return str(value)
    raise ValueError(f"{where} must be text, a number or null, not {type(value).__name__}")


def _table(spec: object, where: str) -> ResultsTable:
    if not isinstance(spec, dict):
        raise ValueError(f"{where} must be an object")
    columns = spec.get("columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError(f"{where}: 'columns' must be a non-empty list")
    names = tuple(_text(name, f"{where}: a column name") for name in columns)
    if len(set(names)) < len(names):
        raise ValueError(f"{where}: the column names must differ")
    rows = spec.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{where}: 'rows' must be a non-empty list")
    table = []
    for i, row in enumerate(rows, 1):
        if not isinstance(row, list) or len(row) != len(names):
            raise ValueError(f"{where}: row {i} must be a list of {len(names)} cells")
        table.append(tuple(_cell(value, f"{where}: row {i}") for value in row))
    note = spec.get("note")
    title = _text(spec.get("title"), f"{where}: 'title'")
    return ResultsTable(title, names, tuple(table), "" if note is None else _text(note, f"{where}: 'note'"))


def validation_results(path: Path | str | None = None) -> ValidationResults | None:
    """Read the validation report's results, or None when the report has not been generated.

    The file is a JSON object::

        {"commit": "<git commit of the run, 7 to 40 hexadecimal digits>",
         "date": "<date of the run, YYYY-MM-DD>",
         "tables": [{"title": "...", "columns": ["...", ...], "rows": [["...", ...], ...], "note": "..."}]}

    with at least one table, and in each table at least one row with one cell per column. The ``note`` is
    optional. The report writes each cell as it should read (a number with its units and precision, ``pass``, a
    zone such as ``green``); a cell that is a JSON number is shown as Python writes it, and ``null`` as an empty
    cell. The Validation section shows each cell's text as it is, and right-aligns the columns of numbers
    (:func:`numeric_columns`).

    Parameters
    ----------
    path : pathlib.Path or str, optional
        The results file (default :data:`VALIDATION_FILE`, read when the function is called).

    Returns
    -------
    ValidationResults or None
        The results, or None if the file does not exist.

    Raises
    ------
    ValueError
        If the file is not valid JSON or does not have the form above.

    Examples
    --------
    >>> import json, tempfile
    >>> validation_results(Path(tempfile.mkdtemp()) / "validation_results.json") is None
    True
    >>> path = Path(tempfile.mkdtemp()) / "validation_results.json"
    >>> table = {"title": "Example", "columns": ["Check", "Result"], "rows": [["put-call parity", "pass"]]}
    >>> _ = path.write_text(json.dumps({"commit": "0123abc", "date": "2026-11-02", "tables": [table]}))
    >>> results = validation_results(path)
    >>> results.commit, results.date, results.tables[0].rows
    ('0123abc', datetime.date(2026, 11, 2), (('put-call parity', 'pass'),))
    """
    path = Path(VALIDATION_FILE if path is None else path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        report = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"it is not valid JSON ({exc.msg} at line {exc.lineno})") from exc
    if not isinstance(report, dict):
        raise ValueError("it must hold a JSON object")
    commit = report.get("commit")
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise ValueError("'commit' must be a git commit: 7 to 40 lowercase hexadecimal digits")
    day = report.get("date")
    if not isinstance(day, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        raise ValueError("'date' must be a date written YYYY-MM-DD")
    try:
        date = dt.date.fromisoformat(day)
    except ValueError:
        raise ValueError(f"'date' {day} is not a calendar date") from None
    tables = report.get("tables")
    if not isinstance(tables, list) or not tables:
        raise ValueError("'tables' must be a non-empty list")
    return ValidationResults(commit, date, tuple(_table(spec, f"table {i}") for i, spec in enumerate(tables, 1)))


def numeric_columns(table: ResultsTable) -> tuple[str, ...]:
    """The columns of a results table that hold numbers, which the Validation section right-aligns.

    A column holds numbers when each of its cells but the empty ones reads as a number: a JSON number, or text
    such as ``0.37``, ``−2.5e-4``, ``1,000``, ``12 %``, ``3 bp``, ``<0.001`` or ``4 / 2.5`` (a count over its
    expected value). Digits are tabular in the app's font, so right-aligned numbers of the same precision line up.

    Parameters
    ----------
    table : ResultsTable
        A table of the validation report (:func:`validation_results`).

    Returns
    -------
    tuple of str
        The names of the columns of numbers, in the table's order.

    Examples
    --------
    >>> table = ResultsTable("Backtest", ("Days", "Exceptions / expected", "Kupiec p", "Zone"),
    ...                      (("250", "4 / 2.5", "0.37", "green"), ("1,000", "17 / 10.0", "<0.001", "red")))
    >>> numeric_columns(table)
    ('Days', 'Exceptions / expected', 'Kupiec p')
    """
    return tuple(
        name
        for i, name in enumerate(table.columns)
        if any(row[i] for row in table.rows) and all(_NUMERIC.fullmatch(row[i]) for row in table.rows if row[i])
    )


def _results() -> None:
    """The validation report's results, at the end of the Validation section, or why there are none."""
    try:
        results = validation_results()
    except (OSError, ValueError) as exc:
        st.warning(f"The validation results file cannot be read, so no results are shown: {exc}.")
        return
    if results is None:
        st.markdown(f"**Results.** {NOT_GENERATED}")
        return
    st.caption(f"Computed by the validation report at commit `{results.commit}` on {results.date.isoformat()}.")
    for table in results.tables:
        st.markdown(f"**{table.title}**")
        frame = pd.DataFrame(list(table.rows), columns=list(table.columns))
        text = [name for name in table.columns if name not in numeric_columns(table)]
        st.table(right_aligned(frame, text_columns=text), hide_index=True, border="horizontal")
        if table.note:
            st.caption(table.note)


def _sections(*, first: str | None, contents: bool) -> None:
    """The one renderer of the tab and the drawer: the first line, the line of contents if asked, and every
    section in an expander, ``first`` (a section's name, or None) first and open, the others collapsed in their
    order. Only the tab has the line of contents and the anchors it links to: in the drawer, the collapsed sections
    are the contents, and the drawer may open over the Methodology tab, whose anchors it would repeat."""
    every = sections()
    st.caption(INTRO)
    if contents:
        st.markdown("**Contents** · " + " · ".join(f"[{s.title}](#{s.anchor})" for s in every))
    for section in sorted(every, key=lambda s: s.name != first):  # a stable sort: the others keep their order
        # The anchor (an empty block, 0 px high) and the expander as one block, with no gap between them.
        with st.container(gap=None):
            if contents:
                st.markdown(f'<div id="{section.anchor}"></div>', unsafe_allow_html=True)
            with st.expander(section.title, expanded=section.name == first):
                st.markdown(section.body, anchors=False)
                if section.name == VALIDATION:
                    _results()


def render() -> None:
    """The Methodology tab: a line of contents that links to each section, then every section, collapsed.

    A link to a section, from the line of contents or from another page (``?tab=Methodology#references``), opens
    the section and scrolls to it (a script in the page, :data:`_ANCHORS`).

    Examples
    --------
    >>> from streamlit.testing.v1 import AppTest
    >>> def page():
    ...     from options_analyzer.ui import tab_methodology
    ...     tab_methodology.render()
    >>> at = AppTest.from_function(page).run()
    >>> [expander.label for expander in at.expander][:3], any(e.proto.expanded for e in at.expander)
    (['Data and time', 'Pricing engines', 'Implied vol'], False)
    """
    with st.container(key="methodology", width=760):
        _sections(first=None, contents=True)
        st.html(_ANCHORS, unsafe_allow_javascript=True)


@st.dialog("Methodology", position="right", width="large", icon=":material/menu_book:")
def drawer(tab: str) -> None:
    """The Methodology in a drawer on the right, at full height, over the open tab, which stays as it was.

    The drawer starts with the open tab's section, open, and follows with the others, collapsed, as the tab draws
    them (:func:`render`) but without the line of contents. Closing it (its close button, the Escape key or a click
    beside it) does not rerun the app. It is Streamlit's large dialog (UI spec 3.6): 1,120 px wide from a 1,280 px
    screen up, which the user can narrow by dragging its edge, and all but 21 px of a phone's width. Its text keeps the
    tab's measure, 760 px at most, so that lines stay short enough to read.

    Parameters
    ----------
    tab : str
        The label of the open tab (app.TABS); the Methodology tab, which has no section of its own, opens the
        drawer with every section collapsed.

    Examples
    --------
    >>> from streamlit.testing.v1 import AppTest
    >>> def page():
    ...     from options_analyzer.ui import tab_methodology
    ...     tab_methodology.drawer("IV")
    >>> at = AppTest.from_function(page).run()
    >>> at.expander[0].label, at.expander[0].proto.expanded, at.expander[1].proto.expanded
    ('Implied vol', True, False)
    """
    with st.container(key="methodology_drawer", width=760):
        _sections(first=TAB_SECTIONS.get(tab), contents=False)
