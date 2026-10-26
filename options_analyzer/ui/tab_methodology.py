"""Methodology tab: models, conventions and limitations.

The text is Markdown with LaTeX maths, one file per area in ``methodology/`` (package data), so that each area
documents its own models in its own file.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

#: Folder of the Methodology files.
FOLDER = Path(__file__).with_name("methodology")
#: The files in reading order: the conventions of the inputs (data), the pricing models, the strategy and risk
#: analytics, the volatility surface, and the limitations and references (validation).
SECTIONS = ("data", "pricing", "risk", "surface", "validation")


def methodology_markdown() -> str:
    """The Methodology text: every file of :data:`SECTIONS`, in order, separated by a blank line.

    Returns
    -------
    str
        Markdown with LaTeX maths (``$...$`` and ``$$...$$``), as ``st.markdown`` renders it.

    Examples
    --------
    >>> methodology_markdown().startswith("### Conventions")
    True
    """
    return "\n\n".join((FOLDER / f"{name}.md").read_text(encoding="utf-8").strip() for name in SECTIONS)


def render() -> None:
    st.markdown(methodology_markdown())
