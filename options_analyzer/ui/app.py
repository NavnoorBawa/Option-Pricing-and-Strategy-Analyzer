"""Streamlit application entry point (``streamlit run main.py``)."""

from __future__ import annotations

from dataclasses import replace

import streamlit as st

from .. import __version__
from . import tab_iv, tab_methodology, tab_pricing, tab_risk, tab_strategy, tab_surface
from .inputs import sidebar

REPO_URL = "https://github.com/NavnoorBawa/Option-Pricing-and-Strategy-Analyzer"
#: The tabs, in order. The open tab is bound to the URL by its label, verbatim (``?tab=Risk``; no parameter for the
#: first tab), so the labels are short plain text: an icon would put its Markdown code into every link.
TABS = ("Pricing", "Strategy", "Risk", "Vol surface", "IV", "Methodology")


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
    inputs = sidebar()

    st.title("Options Pricing & Strategy Analyzer")
    st.caption(
        "Black-Scholes-Merton, binomial and Monte Carlo pricing · Greeks to third order · multi-leg strategies · "
        "stress tests and VaR/ES · implied and local volatility surfaces"
    )

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
    # position (tab_strategy.build_position).
    pricing, strategy, risk, surface, iv, methodology = st.tabs(TABS, key="tab", bind="query-params")
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
