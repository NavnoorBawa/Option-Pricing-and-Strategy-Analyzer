"""Shared UI components: rows of bordered metric cards that wrap cleanly at any width.

Every metric row of the app is a :func:`card_row`. On its own, Streamlit bases each card on its own content and each
container on a fixed 8rem, so a row of four cards wraps 3 + 1, and ``st.columns`` stack every card at full width on
a phone. :func:`card_row` instead sizes all the cards of a row from a model of their text in the theme's IBM Plex
Sans and groups them, so that the row wraps whole groups, the cards of a line have one width and their values one
baseline, and no value is ever cut off.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import streamlit as st

#: Card text in the theme's IBM Plex Sans: values in Medium at 22.75 px (the theme's metricValueFontSize, 1.625rem
#: of its 14 px base), labels (and deltas) in Regular at 12.25 px. Digits (tabular), "$", "+" and "−" are 0.6 em
#: wide in Plex Sans Medium; the narrower and wider characters are listed. Labels average at most 0.517 em per
#: character.
_VALUE_PX, _LABEL_PX, _LABEL_EM = 22.75, 12.25, 0.517
_DIGIT_EM = 0.6
_CHAR_EM = {",": 0.29, ".": 0.29, " ": 0.236, "-": 0.401, "%": 0.947}
#: Room to spare on every value (about 2 px on a desktop), for browsers that round the glyph widths differently.
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


def _card_width(
    label: str, value: str, value_px: float = _VALUE_PX, wrap_label: bool = False, delta: str = ""
) -> float:
    """Width in px a card needs to show its value on one line and its label and delta (the small line under the
    value, if any) on one line each (or, with ``wrap_label``, one word per line)."""

    def characters(text: str) -> int:
        return max((len(word) for word in text.split()), default=0) if wrap_label else len(text)

    label_px = _LABEL_PX * _LABEL_EM * characters(label) + _HELP_ICON_PX
    delta_px = _LABEL_PX * _LABEL_EM * characters(delta)
    return max(value_px * _value_em(value), label_px, delta_px) + _CARD_CHROME_PX


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
    across, then 2 + 2 (never 3 + 1).

    Where the row is narrower than one group at full size (a phone, or a narrow window with the sidebar open), it is
    compact: each group takes a line of its own as a grid of equal columns, the labels and deltas wrap, and the
    values shrink to fit their card, down to 16 px (they never grow beyond their desktop size). A group keeps all
    its cards on one line while each has room for its compact width (labels one word per line, 16 px values), then
    puts one card fewer on each line, down to one: a pair stays side by side on any phone, and three cards go 2 + 1
    on a row narrower than three compact cards, with the third card as wide as the others rather than stretched
    across the row. The values are sized for the cards' actual width. The cards of a line share their label, value
    and delta rows (CSS subgrid), so their values sit on one baseline however many lines their labels take. The
    sizes follow the row's own width (a container query), so an open sidebar on a narrow window is covered too.
    """

    def width(card: tuple[str, ...], value_px: float = _VALUE_PX, wrap: bool = False) -> float:
        label, value, *delta = card
        return _card_width(label, value, value_px, wrap, *delta)

    sized = [*cards, *size_with]
    per_group, gap = len(cards) // len(group_keys), int(_CARD_GAP_PX)
    card = int(np.ceil(max(width(c) for c in sized)))
    compact = int(np.ceil(max(width(c, _MIN_VALUE_PX, True) for c in sized)))
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
        f"@container {name} (width < {group}px) {{ "
        f"{group_items} {{ flex-basis: 100% !important; }} "
        f"{', '.join(groups)} {{ display: grid !important; row-gap: 0 !important; }} "
        f"{subgrids} {{ display: grid; grid-row: span 3; grid-template-rows: subgrid; row-gap: 0; }} "
        f"{row} [data-testid='stMetricLabel'] {{ align-self: start; }} "
        f"{row} :is([data-testid='stMetricLabel'], [data-testid='stMetricDelta']) "
        ":is([data-testid='stMarkdownContainer'], p) { white-space: normal; } "
        f"{per_line(per_group)}}}",
    ]
    # Narrower still: one card fewer per line each time a line can no longer hold its cards at their compact width.
    rules += [
        f"@container {name} (width < {(k + 1) * compact + k * gap}px) {{ {per_line(k)}}}"
        for k in range(per_group - 1, 0, -1)
    ]
    return f"<style>{' '.join(rules)}</style>"


def card_row(key: str, group_keys: Sequence[str], cards: Sequence[Card], *, size_with: Sequence[Card] = ()) -> None:
    """Draw a row of bordered metric cards that wraps by whole groups, so that no value is cut off at any width.

    The cards are dealt in order into equal groups, one per key of ``group_keys``. Four cards in two pairs go 4
    across, then 2 + 2, and on a phone stay 2 + 2 with smaller values (16 to 22.75 px); a pair stacks only when its
    cards no longer fit side by side even so (seven-digit prices). Three cards in one group go 3 across while each
    card has room for its label wrapped one word per line and a 16 px value, then 2 + 1 with the third card as wide
    as the others, then one per line: the default expected-P&L cards of the Strategy tab stay 3 across on phones
    371 px wide and up, and go 2 + 1 on a 360 px phone. Every card of a line has the same width (that of the widest
    card's text, from a model of the theme's IBM Plex Sans), and their values sit on one baseline. The stylesheet
    that sets these sizes is scoped to the row's key.

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
