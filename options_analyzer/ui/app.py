"""Streamlit application entry point (``streamlit run main.py``)."""

from __future__ import annotations

from dataclasses import replace

import streamlit as st

from .. import __version__
from . import tab_iv, tab_methodology, tab_pricing, tab_risk, tab_strategy, tab_surface
from .components import link_match
from .format import md_code
from .inputs import sidebar

REPO_URL = "https://github.com/NavnoorBawa/Option-Pricing-and-Strategy-Analyzer"
#: The tabs, in order. The open tab is bound to the URL by its label, verbatim (``?tab=Risk``; no parameter for the
#: first tab), so the labels are single plain words: an icon would put its Markdown code into every link, and a
#: space would read ``+`` in it.
TABS = ("Pricing", "Strategy", "Risk", "Surface", "IV", "Methodology")
#: Other names a link may give a tab, besides its label: the longer names of the tabs (their labels in version 2).
#: A link's tab is matched without case or punctuation, so ``?tab=risk`` and ``?tab=vol-surface`` work too.
TAB_ALIASES = {
    "Pricing & Greeks": "Pricing",
    "Strategy Builder": "Strategy",
    "Risk & Scenarios": "Risk",
    "Vol surface": "Surface",
    "Volatility surface": "Surface",
    "Implied vol": "IV",
    "Implied volatility": "IV",
}
#: Session-state keys: the link was read (on the session's first run), and its notes, shown once.
_LINK_READ, _LINK_NOTES = "link_read", "link_notes"


def _shown(value: str) -> str:
    """A link's value as a note quotes it: as typed, in a code span (no link, icon or formatting), at most 40
    characters."""
    value = " ".join(value.split())
    return md_code(value if len(value) <= 40 else value[:39] + "…")


def _read_link() -> None:
    """Rewrite a hand-typed link to the parameters the widgets read, on the session's first run.

    Streamlit gives a widget that is bound to the URL its parameter only when that is exactly one of its options,
    and drops it otherwise: ``?tab=risk`` would open the first tab. So before any widget reads the URL, the tab is
    matched without case or punctuation, also by its longer names (:data:`TAB_ALIASES`), and a repeated ``tab`` gives
    its last value, as ``st.query_params`` reads it. The short form ``strategy=<template name>`` becomes the two
    parameters of the Strategy tab's menus (:func:`.tab_strategy.link_selection`). A value that matches nothing is
    dropped, with a note. The new link replaces the old one in one step (one browser history entry, as Streamlit's
    own corrections of a link make) and takes effect in a rerun, since a widget reads the URL its run started with.
    Later runs carry the URL that the app itself writes, so they are left alone (nor could a parameter that is
    bound to a widget be set then).
    """
    state = st.session_state
    if state.get(_LINK_READ):
        return
    state[_LINK_READ] = True
    link = {key: st.query_params.get_all(key) for key in st.query_params}
    params = {key: list(values) for key, values in link.items()}
    notes = []
    asked = params.pop("tab", [])
    if asked:
        tab = link_match(asked[-1], TABS, aliases=TAB_ALIASES)
        if tab is None and asked[-1].strip():
            notes.append(
                f"This link asks for a tab named {_shown(asked[-1])}, which does not exist, so the {TABS[0]} tab "
                "is open."
            )
        if tab not in (None, TABS[0]):  # the first tab has no parameter
            params["tab"] = [tab]
    if "strategy" in params:
        name = params.pop("strategy")[-1]
        selection = tab_strategy.link_selection(name)
        if selection is not None:
            for key in tab_strategy.SELECTION_PARAMS:
                params.pop(key, None)  # what the short form replaces
            params.update({key: [value] for key, value in selection.items()})
        elif name.strip():
            notes.append(
                f"This link asks for a strategy named {_shown(name)}, which is not one of the templates, so it was "
                "ignored."
            )
    state[_LINK_NOTES] = notes
    if params != link:
        st.query_params.from_dict(params)
        st.rerun()


def run() -> None:
    st.set_page_config(
        page_title="Options Pricing & Strategy Analyzer",
        page_icon="📈",
        layout="wide",
        # Open on desktops; on phones the sidebar would cover the whole page, so it starts collapsed there.
        initial_sidebar_state="auto",
        menu_items={
            "Get help": REPO_URL,
            "Report a bug": f"{REPO_URL}/issues",
            "About": f"Options Pricing & Strategy Analyzer v{__version__} by Navnoor Bawa. "
            "Educational tool, not investment advice.",
        },
    )
    _read_link()
    inputs = sidebar()

    st.title("Options Pricing & Strategy Analyzer")
    st.caption(
        "Black-Scholes-Merton, binomial and Monte Carlo pricing · Greeks to third order · multi-leg strategies · "
        "stress tests and VaR/ES · implied and local volatility surfaces"
    )
    for note in st.session_state.pop(_LINK_NOTES, ()):
        st.warning(note, icon=":material/link_off:")

    try:
        prices = tab_pricing.compute_prices(inputs)
    except ValueError as exc:
        st.warning(
            f"{inputs.model.label}: {exc} Showing Black-Scholes-Merton values instead.", icon=":material/warning:"
        )
        inputs = replace(inputs, model=replace(inputs.model, kind="bsm"))
        prices = tab_pricing.compute_prices(inputs)
    tab_pricing.render_price_cards(inputs, prices)

    # Only the open tab runs: opening a tab reruns the app, and the closed tabs stay empty (their widgets keep their
    # values with persist_state). The exception is the Strategy tab's controls (template, strike width, position
    # size, commission), which are drawn on every run: they are bound to the URL, and Streamlit drops the URL
    # parameter of a widget it does not draw, so a link to any tab keeps the position. The Risk tab analyses that
    # position (tab_strategy.build_position). The binding to the URL alone would rerun on a click as well; on_change
    # says so explicitly.
    pricing, strategy, risk, surface, iv, methodology = st.tabs(TABS, key="tab", on_change="rerun", bind="query-params")
    if pricing.open:
        with pricing:
            tab_pricing.render(inputs, prices)
    with strategy:
        tab_strategy.controls(inputs)
        if strategy.open:
            tab_strategy.render(inputs)
    if risk.open:
        with risk:
            tab_risk.render(tab_strategy.build_position(inputs))
    if surface.open:
        with surface:
            tab_surface.render(inputs)
    if iv.open:
        with iv:
            tab_iv.render(inputs)
    if methodology.open:
        with methodology:
            tab_methodology.render()
