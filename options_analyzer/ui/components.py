"""Shared UI components: rows of bordered metric cards that wrap cleanly at any width.

Every metric row of the app is a :func:`card_row`. On its own, Streamlit bases each card on its own content and each
container on a fixed 8rem, so a row of four cards wraps 3 + 1, and ``st.columns`` stack every card at full width on
a phone. :func:`card_row` instead sizes all the cards of a row from a model of their text in the theme's IBM Plex
Sans and groups them, so that the row wraps whole groups, the cards of a line have one width and height and their
values one baseline, and no value is ever cut off.

:func:`current_key` serves the widgets that keep their values while their tab is closed but whose keys change on
purpose, and :func:`link_match` reads the names in a hand-typed link.

Tables follow the tables standard (UI spec 2.8): :func:`col` is the column registry, one ``st.column_config`` per
quantity of the number standard (:data:`COLUMN_KINDS`), and :func:`table` draws an ``st.dataframe`` with it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from .format import MISSING, ROW_HEIGHT

#: Card text in the theme's IBM Plex Sans: values in Medium at 22.75 px (the theme's metricValueFontSize, 1.625rem
#: of its 14 px base) and deltas in Regular at 12.25 px. Digits (tabular), "$", "+" and "−" are 0.6 em wide in Plex
#: Sans Medium; the other characters of amounts and percentages are listed, with the "M", "B", "T" and "e" of the
#: shortened amounts ("$46.62M", "$1.2e+15"). The words a card can show instead ("Unlimited", "No profit") are
#: modelled at 0.6 em a character, wider than they are.
_VALUE_PX, _DELTA_PX = 22.75, 12.25
_DIGIT_EM = 0.6
_CHAR_EM = {",": 0.29, ".": 0.29, " ": 0.236, "-": 0.401, "%": 0.947, "M": 0.814, "B": 0.659, "T": 0.577, "e": 0.556}
#: Labels, as static/desk.css sets them: Regular capitals at 0.72rem (10.08 px), each character followed by 0.06 em
#: of letter-spacing. The advance widths (em) of the capitals, digits and signs of Plex Sans Regular; any other
#: character counts as the widest of them, "%".
_LABEL_PX, _LABEL_TRACKING_EM = 10.08, 0.06
_LABEL_CHAR_EM = {
    **dict.fromkeys("0123456789+×−", 0.6),
    "A": 0.641, "B": 0.653, "C": 0.621, "D": 0.671, "E": 0.583, "F": 0.559, "G": 0.695, "H": 0.707, "I": 0.4,
    "J": 0.51, "K": 0.634, "L": 0.501, "M": 0.812, "N": 0.707, "O": 0.708, "P": 0.606, "Q": 0.708, "R": 0.64,
    "S": 0.581, "T": 0.572, "U": 0.678, "V": 0.609, "W": 0.891, "X": 0.613, "Y": 0.593, "Z": 0.58,
    " ": 0.236, ".": 0.272, ",": 0.272, ":": 0.292, "(": 0.335, ")": 0.335, "/": 0.383, "-": 0.399, "&": 0.694,
    "%": 0.927,
}  # fmt: skip
#: Deltas, Regular at 12.25 px in a pill with 0.375rem (5.25 px) of padding on each side: the advance widths of the
#: labels' characters, the lowercase letters and the "$" and "–" of an interval; any other character counts as the
#: widest, "%". (Every character at 0.517 em and no padding, which covered the pills of deltas of words, made the pill
#: of a Monte Carlo interval, "$10.4237 – $10.4582, 95%", 13 px narrower than the 165.3 px it measures in Chromium.)
_DELTA_PADDING_PX = 10.5
_DELTA_CHAR_EM = {
    **_LABEL_CHAR_EM,
    "a": 0.534, "b": 0.58, "c": 0.503, "d": 0.58, "e": 0.549, "f": 0.324, "g": 0.528, "h": 0.568, "i": 0.25,
    "j": 0.25, "k": 0.527, "l": 0.272, "m": 0.873, "n": 0.568, "o": 0.56, "p": 0.58, "q": 0.58, "r": 0.367,
    "s": 0.487, "t": 0.351, "u": 0.568, "v": 0.492, "w": 0.768, "x": 0.507, "y": 0.499, "z": 0.464,
    "$": 0.598, "–": 0.588,
}  # fmt: skip
#: Room to spare on every value and delta (about 2 px and 1 px on a desktop), for browsers that round the glyph widths
#: differently.
_VALUE_SPARE_EM = 0.1
#: Card padding and border, the label's help icon and its margin, and the gap between cards (gap="small").
_CARD_CHROME_PX, _HELP_ICON_PX, _CARD_GAP_PX = 28.0, 19.25, 14.0
#: Smallest value size on a narrow screen, where the values shrink so that the cards of a group stay side by side
#: (an index-level forward, "$105,127.1096", still fits half of a 360 px phone).
_MIN_VALUE_PX = 16.0

#: One card: ``(label, value, options)``, with ``options`` the further keyword arguments of ``st.metric``
#: (``help``, ``delta``, ``delta_color``, ...).
Card = tuple[str, Any, dict[str, Any]]


def _value_em(value: str) -> float:
    """Width of a card value in ems of its font size, with a little room to spare."""
    return sum(_CHAR_EM.get(c, _DIGIT_EM) for c in value) + _VALUE_SPARE_EM


def _label_px(text: str) -> float:
    """Width in px of a card label on one line, in capitals with their letter-spacing."""
    widest = max(_LABEL_CHAR_EM.values())
    return _LABEL_PX * sum(_LABEL_CHAR_EM.get(c, widest) + _LABEL_TRACKING_EM for c in text.upper())


def _delta_px(text: str) -> float:
    """Width in px of a card delta (the small line under the value) on one line, with its pill's padding and a little
    room to spare. A dollar sign escaped for Markdown (``\\$``) shows without its backslash. No delta, no pill: 0."""
    if not text:
        return 0.0
    widest = max(_DELTA_CHAR_EM.values())
    em = sum(_DELTA_CHAR_EM.get(c, widest) for c in text.replace("\\$", "$")) + _VALUE_SPARE_EM
    return _DELTA_PX * em + _DELTA_PADDING_PX


def _card_width(
    label: str, value: str, value_px: float = _VALUE_PX, label_lines: int | None = 1, delta: str = ""
) -> float:
    """Width in px a card needs to show its value on one line, at ``value_px``, and its label and delta (the small
    line under the value, if any) on at most ``label_lines`` lines each: 1 or 2, or None for one word per line."""

    def longest(text: str, width) -> float:
        """Width of the longest line of ``text`` wrapped at spaces onto ``label_lines`` lines, at the best break."""
        words = text.split()
        if label_lines is None:
            return max(map(width, words), default=0.0)
        if label_lines == 2 and len(words) > 1:
            return min(max(width(" ".join(words[:i])), width(" ".join(words[i:]))) for i in range(1, len(words)))
        return width(text)

    label_px = longest(label, _label_px) + _HELP_ICON_PX
    return max(value_px * _value_em(value), label_px, longest(delta, _delta_px)) + _CARD_CHROME_PX


def _texts(card: Card) -> tuple[str, str, str]:
    """The texts of a card that its width depends on: label, value as shown and delta ("" if none)."""
    label, value, options = card
    return label, str(value), str(options.get("delta") or "")


def _cards_style(
    cards: Sequence[tuple[str, ...]],
    key: str,
    group_keys: Sequence[str],
    *,
    size_with: Sequence[tuple[str, ...]] = (),
) -> str:
    """Stylesheet that sizes a card row and wraps it by whole groups.

    ``cards`` holds ``(label, value)`` or ``(label, value, delta)`` texts, dealt in order into equal groups; ``key``
    is the row's container key and ``group_keys`` those of its groups (see :func:`card_row`). The sizes cover the
    texts of ``cards`` and of ``size_with`` (the cards of rows next to this one, which then wrap at the same widths).

    At full size every card is based on the widest card's width, and each group on its cards side by side, so the
    row moves a whole group to its own line before any group has to wrap its cards: four cards in two pairs go 4
    across, then 2 + 2 (never 3 + 1). The cards of a line are as tall as the tallest (one with a delta line).

    Where the row is narrower than one group at full size (a phone, or a narrow window with the sidebar open), it is
    compact: each group takes a line of its own as a grid of equal columns, the labels and deltas wrap, and the
    values shrink to fit their card, down to 16 px (they never grow beyond their desktop size). A line of three or
    more cards needs room for every card's value at full size and its label on two lines at most; with less room,
    the group puts one card fewer on each line, so that a phone never shows a line of shrunken values under three
    lines of label. A pair stays side by side while its cards have their compact width (labels one word per line,
    16 px values), then stacks. Three cards thus go 3 across, then 2 + 1 with the third card as wide as the others
    rather than stretched across the row, then one per line. The values are sized for the cards' actual width. The
    cards of a line share their label, value and delta rows (CSS subgrid), so their values sit on one baseline
    however many lines their labels take. The sizes follow the row's own width (a container query), so an open
    sidebar on a narrow window is covered too.
    """

    def width(card: tuple[str, ...], value_px: float = _VALUE_PX, label_lines: int | None = 1) -> float:
        label, value, *delta = card
        return _card_width(label, value, value_px, label_lines, *delta)

    sized = [*cards, *size_with]
    per_group, gap = len(cards) // len(group_keys), int(_CARD_GAP_PX)
    card = int(np.ceil(max(width(c) for c in sized)))
    # The narrowest card of a line of three or more, and of a pair.
    three_up = int(np.ceil(max(width(c, _VALUE_PX, 2) for c in sized)))
    compact = int(np.ceil(max(width(c, _MIN_VALUE_PX, None) for c in sized)))
    em = max(_value_em(c[1]) for c in sized)
    group = per_group * card + (per_group - 1) * gap
    row, name = f".st-key-{key}", key.replace("_", "-")
    groups = [f".st-key-{k}" for k in group_keys]
    group_items = f"{row} > :has({', '.join(f'> {g}' for g in groups)})"
    each_card = ", ".join(f"{g} > *" for g in groups)
    # A card is three nested boxes (Streamlit's element container, the bordered stMetric and the padded box inside
    # it), each a subgrid on its group's rows, so that the label, value and delta of every card of a line share a row.
    subgrids = ", ".join(f"{g} > *, {g} [data-testid='stMetric'], {g} [data-testid='stMetric'] > div" for g in groups)

    def per_line(k: int) -> str:
        """Rules for ``k`` cards on each line of a group: equal columns, the gap above the cards after the first
        line (the grid has no row gap, which would also open between a card's label and value), and values sized
        for a ``k``-th of the row, less its share of the gaps and the card's padding."""
        later = ", ".join(f"{g} > :nth-child(n + {k + 1})" for g in groups)
        space = f"{100 / k:g}cqw - {gap * (k - 1) / k + _CARD_CHROME_PX:g}px"
        return (
            f"{', '.join(groups)} {{ grid-template-columns: repeat({k}, minmax(0, 1fr)); }} "
            + (f"{later} {{ margin-top: {gap}px; }} " if k < per_group else "")
            + f"{row} [data-testid='stMetricValue'] "
            f"{{ font-size: clamp({_MIN_VALUE_PX:g}px, calc(({space}) / {em:g}), {_VALUE_PX:g}px); }} "
        )

    rules = [
        f"{row} {{ container: {name} / inline-size; }}",
        f"{group_items} {{ flex: 1 1 {group}px !important; }}",
        f"{each_card} {{ flex: 1 1 {card}px !important; }}",
        # Every card as tall as its line: the groups stretch to the line (a row of four on one line) and the cards to
        # their group; the bordered box fills its card by the desk stylesheet (static/desk.css), as Streamlit's own
        # style has it too.
        f"{row}, {', '.join(groups)} {{ align-items: stretch; }}",
        f"@container {name} (width < {group}px) {{ "
        f"{group_items} {{ flex-basis: 100% !important; }} "
        f"{', '.join(groups)} {{ display: grid !important; row-gap: 0 !important; }} "
        f"{subgrids} {{ display: grid; grid-row: span 3; grid-template-rows: subgrid; row-gap: 0; }} "
        f"{row} [data-testid='stMetricLabel'] {{ align-self: start; }} "
        f"{row} :is([data-testid='stMetricLabel'], [data-testid='stMetricDelta']) "
        ":is([data-testid='stMarkdownContainer'], p) { white-space: normal; } "
        f"{per_line(per_group)}}}",
    ]
    # Narrower still: one card fewer per line each time a line can no longer hold its cards (k + 1 of them, each at
    # least ``three_up`` wide when k + 1 >= 3 and ``compact`` wide in a pair).
    rules += [
        f"@container {name} (width < {(k + 1) * (three_up if k >= 2 else compact) + k * gap}px) {{ {per_line(k)}}}"
        for k in range(per_group - 1, 0, -1)
    ]
    return f"<style>{' '.join(rules)}</style>"


def card_row(key: str, group_keys: Sequence[str], cards: Sequence[Card], *, size_with: Sequence[Card] = ()) -> None:
    """Draw a row of bordered metric cards that wraps by whole groups, so that no value is cut off at any width.

    The cards are dealt in order into equal groups, one per key of ``group_keys``. Four cards in two pairs go 4
    across, then 2 + 2, and on a phone stay 2 + 2 with smaller values (16 to 22.75 px); a pair stacks only when its
    cards no longer fit side by side even so (seven-digit prices). Three cards in one group go 3 across while each
    card has room for its value at full size (22.75 px) and its label on two lines at most, then 2 + 1 with the
    third card as wide as the others, then one per line: the default expected-P&L cards of the Strategy tab go 3
    across on a row of 409 px and up (phones 437 px wide and up), and 2 + 1 on 360 to 430 px phones. Every card of
    a line has the same width (that of the widest card's text, from a model of the theme's IBM Plex Sans) and
    height, and their values sit on one baseline. The stylesheet that sets these sizes is scoped to the row's key.

    Parameters
    ----------
    key : str
        Key of the row's container; its stylesheet applies to ``.st-key-<key>`` only.
    group_keys : sequence of str
        Keys of the group containers, in order. The number of cards must be a multiple of their number.
    cards : sequence of (label, value, options)
        ``label`` and ``value`` as ``st.metric`` takes them (the value as shown, such as ``"$10.4506"``), and
        ``options``, the further keyword arguments of ``st.metric``, such as ``help`` or ``delta``.
    size_with : sequence of (label, value, options), optional
        The cards of a row of the same shape that sits next to this one, drawn with this row's cards in its own
        ``size_with``: both rows are then sized from the cards of both, so they wrap at the same widths and their
        cards line up. Not drawn here.

    Raises
    ------
    ValueError
        If there are no cards or groups, or the cards do not divide evenly into the groups.

    Examples
    --------
    Four cards in two pairs (outside a running app Streamlit draws nothing):

    >>> card_row(
    ...     "demo_cards",
    ...     ("demo_cards_prices", "demo_cards_model"),
    ...     [
    ...         ("Call value", "$10.4506", {"help": "Closed-form European value"}),
    ...         ("Put value", "$5.5735", {}),
    ...         ("Forward price", "$105.1271", {}),
    ...         ("Call ITM probability", "55.96%", {}),
    ...     ],
    ... )
    """
    if not cards or not group_keys or len(cards) % len(group_keys):
        raise ValueError(f"{len(cards)} cards do not divide into {len(group_keys)} equal groups")
    st.html(_cards_style([_texts(c) for c in cards], key, group_keys, size_with=[_texts(c) for c in size_with]))
    row = st.container(horizontal=True, gap="small", key=key)
    groups = [row.container(horizontal=True, gap="small", key=k) for k in group_keys]
    per_group = len(cards) // len(groups)
    for i, (label, value, options) in enumerate(cards):
        groups[i // per_group].metric(label, value, border=True, **options)


def current_key(family: str, key: str) -> str:
    """Return ``key`` after forgetting the values kept under the other keys of its ``family``.

    For a widget that keeps its value while its tab is closed (``persist_state="session"``) and whose key changes on
    purpose, because it holds what the value belongs to (a contract, a range, a grid): the value is then kept only
    while the key stays the same, and a key used before starts from the widget's default again, as it did before
    the tabs ran lazily (a widget that is not drawn in a run loses its value).

    Parameters
    ----------
    family : str
        The prefix that the widget's keys share, such as ``"smile_pick_"``.
    key : str
        The widget's key in this run, which starts with ``family``.

    Returns
    -------
    str
        ``key``, for the widget's ``key`` argument.

    Examples
    --------
    Outside a running app the session state is a plain store (Streamlit logs a warning):

    >>> st.session_state["smile_pick_1"] = [0, 2]
    >>> current_key("smile_pick_", "smile_pick_2")
    'smile_pick_2'
    >>> "smile_pick_1" in st.session_state
    False
    """
    for other in [k for k in st.session_state if k.startswith(family) and k != key]:
        del st.session_state[other]
    return key


def _plain(text: str) -> str:
    """``text`` in lower case, with each run of characters other than letters and digits read as one space."""
    return " ".join(re.findall(r"[^\W_]+", text.casefold()))


def link_match(value: str, names: Iterable[str], *, aliases: Mapping[str, str] | None = None) -> str | None:
    """Return the name in ``names`` that a hand-typed link value stands for, or None.

    A widget bound to the URL accepts only its option, verbatim: ``?tab=risk`` or ``?strategy=iron-condor`` would
    be dropped. Here ``value`` is matched without case and with each run of characters other than letters and
    digits read as one space, first against ``names`` and then against the keys of ``aliases``.

    Parameters
    ----------
    value : str
        The value in the link, as ``st.query_params`` gives it (``+`` and ``%20`` already read as spaces).
    names : iterable of str
        The options, such as a widget's labels.
    aliases : mapping of str to str, optional
        Other names a link may use, each mapped to the name of ``names`` it stands for.

    Returns
    -------
    str or None
        The matching name, as written in ``names`` (or the alias's target), or None when nothing matches.

    Examples
    --------
    >>> link_match("iron-condor", ["Iron Butterfly", "Iron Condor"])
    'Iron Condor'
    >>> link_match("Vol+Surface", ["Pricing", "Surface"], aliases={"vol surface": "Surface"})
    'Surface'
    >>> link_match("XYZ", ["Pricing", "Surface"]) is None
    True
    """
    wanted = _plain(value)
    if not wanted:
        return None
    candidates = [(name, name) for name in names] + list((aliases or {}).items())
    return next((name for spelling, name in candidates if _plain(spelling) == wanted), None)


# --- Tables: the column registry and the table helper (UI spec 2.5 and 2.8) ---------------------------------------


@dataclass(frozen=True)
class ColumnKind:
    """How one quantity of the number standard shows in an ``st.dataframe`` column.

    ``unit`` goes in the header (``IV (%)``); the data are in that unit already (15.46 for 15.46%, 0.35 for 0.35 vp,
    45 for 45 bp), so the numbers on screen, copied and exported are the same. ``digits`` is the column's one
    precision; ``None`` means :func:`table` chooses it from the column's largest value (``auto``). ``format`` is the
    printf format with ``{d}`` for the digits; ``None`` for a text or date column.
    """

    unit: str
    format: str | None
    digits: int | None
    auto: str = ""  # how table() chooses the digits when ``digits`` is None: "money", "strike" or "bp"
    significant: bool = False  # ``digits`` are significant digits (%g), not decimals


#: The column kinds, named after the quantities of the number standard (UI spec 2.5). Number columns keep the printf
#: hyphen-minus: a NumberColumn format cannot emit U+2212, and a text column would lose its right alignment and its
#: sorting. Signed columns have no "+": printf's "+" flag also signs zero ("+0.00"), which the standard forbids.
#: Thousands separators come from printf's "," flag (Streamlit's sprintf, 1.65 and later), not from the locale.
COLUMN_KINDS: dict[str, ColumnKind] = {
    "text": ColumnKind("", None, None),
    "date": ColumnKind("", None, None),
    "level": ColumnKind("", "%,.{d}f", 2),  # spot, forward, index level: 5,815.22
    "strike": ColumnKind("", "%,.{d}f", None, "strike"),  # 5,815, or 582.5 where a strike needs a decimal
    "premium": ColumnKind("", "%,.{d}f", 2),  # theo, bid, ask, mid per share: 98.57
    "premium_diag": ColumnKind("", "%,.{d}f", 6),  # diagnostics only: 98.570661
    "money": ColumnKind("$", "%,.{d}f", None, "money"),  # unsigned amounts: 2 dp below $1,000, else whole dollars
    "pnl": ColumnKind("$", "%,.{d}f", None, "money"),  # P&L and losses, negative
    "dollar_delta": ColumnKind("$", "%,.{d}f", 0),  # $Δ
    "futures": ColumnKind("ES", "%.{d}f", 2),  # position delta in E-mini contracts; label "Δ (MES)" for micros
    "gamma_pnl": ColumnKind("$", "%,.{d}f", None, "money"),  # Γ P&L 1%: ½Γ(0.01S)² × units
    "vega": ColumnKind("$/vp", "%,.{d}f", None, "money"),
    "theta": ColumnKind("$/day", "%,.{d}f", None, "money"),
    "rho": ColumnKind("$/1%", "%,.{d}f", None, "money"),
    "vol": ColumnKind("%", "%.{d}f", 2),  # implied vol: 15.46
    "vol_change": ColumnKind("vp", "%.{d}f", 2),  # vol changes, RR, BF, mid vs fit, edges
    "rate": ColumnKind("%", "%.{d}f", 2),  # rates and yields: 4.75
    "bp": ColumnKind("bp", "%,.{d}f", None, "bp"),  # 0 dp; 1 dp below 10 bp, 2 dp below 1 bp
    "delta": ColumnKind("", "%.{d}f", 3),  # per option, spot convention: 0.508
    "gamma": ColumnKind("", "%.{d}g", 3, significant=True),  # Γ 1% per option: 0.0936
    "greek": ColumnKind("", "%.{d}g", 3, significant=True),  # vanna, volga, charm, veta and third order
    "probability": ColumnKind("%", "%.{d}f", 1),
    "share": ColumnKind("%", "%.{d}f", 1),  # a share of a whole: in band, confidence
    "moneyness": ColumnKind("%", "%.{d}f", 1),  # K/F
    "log_moneyness": ColumnKind("", "%.{d}f", 3),  # k = ln(K/F)
    "quantity": ColumnKind("", "%+d", 0),  # a leg's signed size; a zero quantity is refused before it shows
    "count": ColumnKind("", "%,d", 0),
    "days": ColumnKind("", "%,.{d}f", 0),  # DTE, whole days
    "year_fraction": ColumnKind("", "%.{d}f", 6),  # diagnostics only
}
#: The precision of a data-dependent kind when :func:`col` is called without ``digits`` (no data to choose from):
#: whole strikes (the index's 5-point grid), cents, whole basis points.
_AUTO_DEFAULT = {"strike": 0, "money": 2, "bp": 0}
#: Rows a table shows before it scrolls (UI spec 2.8): shorter tables take their content's height.
TABLE_ROWS = 15


def _unit_in(label: str, unit: str) -> bool:
    """Whether ``label`` names ``unit`` already (``$Δ`` names dollars; ``Mid vs fit (vp)`` names vol points)."""
    return unit in label or (unit.startswith("$") and "$" in label)


def col(
    kind: str,
    label: str,
    *,
    help: str,
    digits: int | None = None,
    pinned: bool | None = None,
    width: Any = None,
) -> Any:
    """The ``st.column_config`` of one column, from the column registry (:data:`COLUMN_KINDS`).

    The header gets the kind's unit in brackets unless it names it already (``col("vol", "IV", ...)`` heads the
    column ``IV (%)``); the numbers take the kind's one precision. Keep headers to 14 characters or fewer (spec 2.8).

    Parameters
    ----------
    kind : str
        A key of :data:`COLUMN_KINDS`: ``"strike"``, ``"premium"``, ``"vol"``, ``"vol_change"``, ``"pnl"``, ...
    label : str
        The header, without its unit.
    help : str
        The header's tooltip: what the column is, its sign convention and its basis. Every column has one.
    digits : int, optional
        The column's precision where the kind leaves it to the data (strikes, money, bp); :func:`table` passes the
        one it chose. Ignored by kinds with a fixed precision.
    pinned : bool, optional
        Keep the column in view while the table scrolls sideways (:func:`table` pins the first column).
    width : optional
        ``st.column_config``'s width (``"small"``, ``"medium"``, ``"large"`` or px).

    Returns
    -------
    dict
        The column configuration for ``st.dataframe(column_config=...)``.

    Raises
    ------
    ValueError
        If ``kind`` is unknown or ``help`` is blank.

    Examples
    --------
    >>> col("vol", "IV", help="Implied vol at the mid.")["label"]
    'IV (%)'
    >>> col("vega", "Vega", help="$ per vol point.")["type_config"]["format"]
    '%,.2f'
    >>> col("pnl", "P&L ($)", help="At expiry.", digits=0)["type_config"]["format"]
    '%,.0f'
    """
    if kind not in COLUMN_KINDS:
        raise ValueError(f"unknown column kind {kind!r}; the kinds are {', '.join(COLUMN_KINDS)}")
    if not help or not help.strip():
        raise ValueError(f"column {label!r} needs a help text (UI spec 2.8: every column has one)")
    spec = COLUMN_KINDS[kind]
    header = f"{label} ({spec.unit})" if spec.unit and not _unit_in(label, spec.unit) else label
    common = {"help": help, "pinned": pinned, "width": width}
    if kind == "text":
        return st.column_config.TextColumn(header, **common)
    if kind == "date":
        return st.column_config.DateColumn(header, format="DD-MMM-YY", **common)
    assert spec.format is not None
    places = spec.digits if spec.digits is not None else (_AUTO_DEFAULT[spec.auto] if digits is None else digits)
    return st.column_config.NumberColumn(header, format=spec.format.format(d=places), **common)


def _auto_digits(kind: ColumnKind, values: np.ndarray) -> int:
    """The precision :func:`table` chooses for a column whose kind leaves it to the data (one per column)."""
    finite = values[np.isfinite(values)]
    if kind.digits is not None:
        return kind.digits
    if not finite.size:
        return 0 if kind.auto == "bp" else 2
    largest = float(np.max(np.abs(finite)))
    if kind.auto == "money":  # cents below $1,000, whole dollars from $1,000 (spec 2.5, money)
        return 2 if round(largest, 2) < 1000 else 0
    if kind.auto == "bp":
        return 2 if round(largest, 2) < 1 else 1 if round(largest, 1) < 10 else 0
    # Strikes: no decimals while every strike is whole, else the fewest (up to two) that write them all.
    for places in range(3):
        if np.allclose(finite, np.round(finite, places), rtol=0, atol=1e-9):
            return places
    return 2


def _rounded(kind: ColumnKind, values: np.ndarray, places: int) -> np.ndarray:
    """``values`` at the column's precision, without negative zeros: printf writes −0.0001 at two decimals as
    ``-0.00``, the standard's ``0.00``."""
    if kind.significant:
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = 10.0 ** (places - 1 - np.floor(np.log10(np.abs(values))))
            rounded = np.where(np.isfinite(scale) & (values != 0), np.round(values * scale) / scale, values)
    else:
        rounded = np.round(values, places)
    return rounded + 0.0  # -0.0 + 0.0 is +0.0


def table(
    data: pd.DataFrame,
    kinds: Mapping[str, str],
    help: Mapping[str, str],
    *,
    labels: Mapping[str, str] | None = None,
    digits: Mapping[str, int] | None = None,
    max_rows: int = TABLE_ROWS,
    **kwargs: Any,
) -> Any:
    """Draw ``data`` as a desk table (UI spec 2.8): the registry's format for every column, one precision per
    column, the first column pinned, help on every column, rows of 28 px, and the content's height up to
    ``max_rows`` rows (then a fixed height of that many rows, which scrolls).

    Numeric columns are rounded to their precision before they are sent, so that no cell shows ``-0.00``; data are
    in the units of the kind (see :class:`ColumnKind`). Missing values show the en dash.

    Parameters
    ----------
    data : pandas.DataFrame
        The table, its columns in display units (vols in %, vol changes in vp, rate changes in bp).
    kinds : mapping of str to str
        Every column's kind (a key of :data:`COLUMN_KINDS`).
    help : mapping of str to str
        Every column's help text.
    labels : mapping of str to str, optional
        Headers other than the column names (without their units, which :func:`col` adds).
    digits : mapping of str to int, optional
        A precision other than the one chosen from the data, for a column whose kind leaves it to the data (whole
        dollars in a position's Γ P&L column whose largest value is below $1,000, as the spec's ``−$68``).
    max_rows : int, default 15
        Rows shown before the table scrolls.
    **kwargs
        Further keyword arguments of ``st.dataframe``: ``key``, ``selection_mode``, ``on_select``, ``width``, ...

    Returns
    -------
    DeltaGenerator or DataframeState
        What ``st.dataframe`` returns.

    Raises
    ------
    ValueError
        If a column has no kind or no help, or a kind is unknown.

    Examples
    --------
    Outside a running app Streamlit draws nothing:

    >>> frame = pd.DataFrame({"Strike": [5800.0, 5805.0], "IV": [15.4612, 15.3201]})
    >>> _ = table(frame, {"Strike": "strike", "IV": "vol"}, {"Strike": "Listed strike.", "IV": "Mid implied vol."})
    """
    missing = [c for c in data.columns if c not in kinds or c not in help]
    if missing:
        raise ValueError(f"every column needs a kind and a help text (UI spec 2.8); missing for {missing}")
    labels = labels or {}
    shown = data.copy()
    config = {}
    for i, name in enumerate(data.columns):
        kind = COLUMN_KINDS.get(kinds[name])
        if kind is None:
            raise ValueError(f"unknown column kind {kinds[name]!r} for column {name!r}")
        places = None
        if kind.format is not None:
            values = pd.to_numeric(data[name], errors="coerce").to_numpy(float)
            places = _auto_digits(kind, values)
            if kind.digits is None and digits and name in digits:
                places = int(digits[name])
            if kind.format.endswith("d"):  # whole numbers: printf's %d drops a fraction instead of rounding it
                whole = np.round(values) + 0.0
                shown[name] = pd.array(whole, dtype="Int64") if np.isnan(values).any() else whole
            else:
                shown[name] = _rounded(kind, values, places)
        config[name] = col(
            kinds[name], labels.get(name, str(name)), help=help[name], digits=places, pinned=True if i == 0 else None
        )
    rows = len(data)
    height: int | str = "content" if rows <= max_rows else table_height(max_rows)
    options = {"hide_index": True, "row_height": ROW_HEIGHT, "height": height, "placeholder": MISSING}
    return st.dataframe(shown, column_config=config, **(options | kwargs))


def table_height(rows: int) -> int:
    """Height in px of a table that shows ``rows`` rows and its header: what ``height="content"`` gives a table of
    that many rows at :data:`ROW_HEIGHT`: the header and the rows at 28 px and 6 px of frame (454 px for 15 rows,
    measured in Chromium on Streamlit 1.65 and 1.66; ``tests/test_desk_foundation_e2e.py`` checks it)."""
    return (rows + 1) * ROW_HEIGHT + 6
