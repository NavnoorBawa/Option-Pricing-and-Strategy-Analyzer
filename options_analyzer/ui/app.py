"""Streamlit application entry point (``streamlit run main.py``)."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime

import numpy as np
import streamlit as st

from .. import __version__, clock
from . import header, tab_iv, tab_methodology, tab_pricing, tab_risk, tab_strategy, tab_surface, theme, whatif
from . import market as mk
from .components import link_match, link_shown
from .format import MINUS
from .format import strike as strike_text
from .inputs import (
    ASOF_RANGE,
    TEXTBOOK_LIMITS,
    TEXTBOOK_NAMES,
    Inputs,
    index_strike_limits,
    market_notes,
    sidebar,
    snap_strike,
)
from .model_config import ALO_SCHEMES, MODEL_ALIASES, MODELS, ModelConfig

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
#: Other names a link may give a market preset (``?preset=index``).
PRESET_ALIASES = {"index": mk.INDEX, "synthetic": mk.INDEX, "spx": mk.INDEX}
#: Parameters that only the textbook market has: a link with any of them and no preset (a link from version 2)
#: opens the textbook market, so that it reproduces the numbers it was made with.
TEXTBOOK_PARAMS = ("spot", "days", "rate", "div")
#: The settings of version 2's tree and Monte Carlo models, which a link to them carried (``?model=Monte+Carlo&paths=
#: 250%2C000``): the Engine diagnostics set their own, so the link reader drops them with the model.
LEGACY_ENGINE_PARAMS = ("tree", "steps", "style", "paths", "seed", "antithetic", "control_variate")
#: The one parameter bound to the URL whose empty value means something: the Strategy tab's Dated curves, none chosen.
EMPTY_PARAMS = ("st_dates",)
#: Session-state keys: the link was read (on the session's first run), and its notes, shown once.
_LINK_READ, _LINK_NOTES = "link_read", "link_notes"


def _link_asof(value: str) -> datetime | None:
    """A valuation time typed in a link, as New York wall time on its 5-minute step: ``2026-10-23T16:00`` (the
    widget's own form), with seconds or a time zone, or a date alone (``2026-10-23``: that day's close, or the close
    of the trading day before it). None when it is not a date or is out of range."""
    text = value.strip().replace(" ", "T")
    if text[-1:] in ("Z", "z"):  # UTC, which Python 3.10's fromisoformat reads only as "+00:00"
        text = text[:-1] + "+00:00"
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if len(value.strip()) == 10:  # a date alone
        # Checked before the calendar is asked: it covers 2000 to 2100 and raises outside it, and the trading day
        # before 1 January 2000 falls in 1999.
        if not ASOF_RANGE[0].date() <= moment.date() <= ASOF_RANGE[1].date():
            return None
        moment = clock.market_close(clock.previous_trading_day(moment.date())).replace(tzinfo=None)
    elif moment.tzinfo is not None:
        moment = clock.to_new_york(moment)
    moment = mk.snap(moment)
    return moment if ASOF_RANGE[0] <= moment <= ASOF_RANGE[1] else None


def _read_link() -> None:
    """Rewrite a hand-typed link to the parameters the widgets read, on the session's first run.

    Streamlit gives a widget that is bound to the URL its parameter only when that is exactly one of its options,
    and drops it otherwise: ``?tab=risk`` would open the first tab. So before any widget reads the URL, the tab is
    matched without case or punctuation, also by its longer names (:data:`TAB_ALIASES`), and a repeated ``tab`` gives
    its last value, as ``st.query_params`` reads it. The pricing model and the American engine's scheme are matched
    the same way, the model also by the header badge's short names (:data:`.model_config.MODEL_ALIASES`, so
    ``?model=ALO`` works). The Strategy tab's template, as ``strategy=<template name>`` or the two menus of
    earlier links, becomes the parameter of its menu (:func:`.tab_strategy.read_link`). A value that matches nothing is
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
    # An empty value (``?pr_metric=``) means no parameter: a required control would read it as no choice, None,
    # which the tabs cannot use (it raised KeyError), and an empty ``?spot=`` is no textbook input.
    params = {key: list(values) for key, values in link.items() if key in EMPTY_PARAMS or "".join(values).strip()}
    notes = []
    asked = params.pop("tab", [])
    if asked:
        tab = link_match(asked[-1], TABS, aliases=TAB_ALIASES)
        if tab is None and asked[-1].strip():
            notes.append(
                f"This link asks for a tab named {link_shown(asked[-1])}, which does not exist, so the {TABS[0]} tab "
                "is open."
            )
        if tab not in (None, TABS[0]):  # the first tab has no parameter
            params["tab"] = [tab]
    # The pricing model by its label or the badge's short name (?model=ALO), and the American engine's scheme.
    for key, names, aliases, default in (
        ("model", MODELS.values(), MODEL_ALIASES, MODELS["bsm"]),
        ("scheme", ALO_SCHEMES.values(), {}, f"the {ModelConfig().scheme} scheme"),
    ):
        if key in params:
            value = params.pop(key)[-1]
            name = link_match(value, names, aliases=aliases)
            if name is not None:
                params[key] = [name]
            elif value.strip():
                what = "pricing model" if key == "model" else key
                notes.append(
                    f"This link asks for a {what} named {link_shown(value)}, which does not exist, so {default} is "
                    "used."
                )
    # A version-2 link to a tree or to Monte Carlo (UI spec 2.4): those engines are diagnostics now, so the link opens
    # the closed form with the Pricing tab's Engine diagnostics open. Their settings are dropped with the model.
    legacy = params.get("model", [None])[-1]
    if legacy in (MODELS["binomial"], MODELS["monte_carlo"]):
        for key in (*LEGACY_ENGINE_PARAMS, "model"):
            params.pop(key, None)
        engine = {MODELS["binomial"]: "a binomial tree", MODELS["monte_carlo"]: "Monte Carlo"}[legacy]
        if "tab" not in params:
            # The link opens the Pricing tab: the expander is bound to the URL, so the new link holds it open and a
            # link shared from this view opens it again.
            params[tab_pricing.DIAGNOSTICS] = ["true"]
            where = " (open below)"
        else:
            # Another tab: Streamlit drops the parameter of a widget it does not draw, so the session holds it open
            # until the Pricing tab draws it.
            state[tab_pricing.DIAGNOSTICS] = True
            where = ", open there"
        notes.append(
            f"This link asks for {engine}, which now runs in the Pricing tab's Engine diagnostics{where}; the "
            f"sidebar prices with {MODELS['bsm']}."
        )
    # The market: a preset by its label or a short name (?preset=textbook), a version-2 link's typed inputs, and
    # the valuation time in any ISO form.
    if "preset" in params:
        value = params.pop("preset")[-1]
        preset = link_match(value, mk.PRESETS, aliases=PRESET_ALIASES)
        if preset is not None:
            if preset != mk.PRESETS[0]:  # the default preset has no parameter
                params["preset"] = [preset]
        elif value.strip():
            notes.append(
                f"This link asks for a market named {link_shown(value)}, which does not exist, so the {mk.PRESETS[0]} "
                "market is used."
            )
    elif any(key in params for key in TEXTBOOK_PARAMS):
        params["preset"] = [mk.TEXTBOOK]
    # SPX options are European: the index offers no American engine (model_config.model_sidebar).
    if params.get("preset", [mk.INDEX])[-1] == mk.INDEX and params.get("model", [None])[-1] == MODELS["alo"]:
        params.pop("model")
        params.pop("scheme", None)
        notes.append(
            f"This link asks for the American engine, but options on the {mk.INDEX} are European-exercise (as SPX "
            f"options are), so {MODELS['bsm']} is used. The American engine prices the Textbook market."
        )
    if "asof" in params:
        value = params.pop("asof")[-1]
        moment = _link_asof(value)
        if moment is not None:
            params["asof"] = [f"{moment:%Y-%m-%dT%H:%M}"]
        elif value.strip():
            notes.append(
                f"This link asks for a valuation time {link_shown(value)}, which is not a date and time between 2000 "
                "and 2097 (2026-10-23T16:00), so the last close is used."
            )
    notes += _read_numbers(params)
    # The Strategy tab's template: ?strategy=<name> and the two menus of earlier links (tab_strategy.read_link).
    notes += tab_strategy.read_link(params)
    state[_LINK_NOTES] = notes
    if params != link:
        st.query_params.from_dict(params)
        st.rerun()


def _number_params(index: bool) -> dict[str, tuple[str, tuple[float, float] | None]]:
    """The parameters bound to a number input or a slider of numbers: what a note calls each, and the range its widget
    accepts (None where the range depends on another setting: the vol shift's on its mode, the date's on the
    expiry; the widget checks those itself)."""
    textbook = {key: (name, TEXTBOOK_LIMITS[key]) for key, name in TEXTBOOK_NAMES.items()}
    common = {
        "contracts": ("contracts", tab_strategy.CONTRACTS),
        "multiplier": ("shares per contract", tab_strategy.MULTIPLIER),
        "commission": ("commission per contract", tab_strategy.COMMISSION),
        "strategy_width_pct": ("strike width", (tab_strategy.MIN_WIDTH_PCT, tab_strategy.MAX_WIDTH_PCT)),
        "ws": ("what-if spot move", whatif.SPOT_LIMITS),
        "wv": ("what-if vol shift", None),
        "wd": ("what-if date", None),
        "wr": ("what-if rate shift", whatif.BP_LIMITS),
    }
    if not index:
        return {**textbook, **common}
    return {"strike": ("strike", index_strike_limits()), "vol": textbook["vol"], **common}


def _read_numbers(params: dict[str, list[str]]) -> list[str]:
    """Drop, in place, the numbers of a link that its widgets cannot take, and return the notes.

    A number input or slider bound to the URL reads its parameter itself and falls back to its default when the
    value is out of range, but an integer one turns ``inf`` (or ``1e400``) into an int while it reads the link, which
    raises OverflowError on every run of every tab (the Strategy tab's controls, ``contracts`` among them, are drawn
    on every run). So a value that is not a finite number in the widget's range is dropped here, with a note, before
    any widget reads it. Under the index a strike off the 5-point grid is moved to the nearest strike on it, as a
    listed contract's strike is.
    """
    notes = []
    index = params.get("preset", [mk.INDEX])[-1] == mk.INDEX
    for key, (name, limits) in _number_params(index).items():
        if key not in params:
            continue
        value = params[key][-1]
        try:
            number = float(value)
        except ValueError:
            number = math.nan
        if math.isfinite(number) and (limits is None or limits[0] <= number <= limits[1]):
            if index and key == "strike":
                listed = snap_strike(number)
                if listed != number:
                    params[key] = [f"{listed:g}"]
                    notes.append(
                        f"This link asks for the strike {link_shown(value)}, which is not listed (every "
                        f"{limits[0]:g} points near the money), so the nearest listed one, "
                        f"{strike_text(listed, limits[0])}, is used."
                    )
            continue
        params.pop(key)
        wanted = "a finite number" if limits is None else f"a number from {limits[0]:,g} to {limits[1]:,g}"
        wanted = wanted.replace("-", MINUS)
        notes.append(
            f"This link gives the {name} as {link_shown(value)}, which is not {wanted}, so the default is used."
        )
    return notes


def beyond_reach(inputs: Inputs) -> str:
    """Why nothing can be priced when the index's strike lies beyond the reach of the surface at its expiry: the
    smile gives no vol there (``ArbitrageFreeSurface.implied_vol_at`` reads NaN beyond its quotes and its
    extrapolation), so no engine can price it. The message names the strikes it does reach, found on the strike
    grid up to the strike input's limit (about 4,600 reads, 0.1 s, on this error path only)."""
    context = inputs.context
    step = context.strike_step
    grid = np.arange(step, 4 * context.spot + step / 2, step)
    reached = [
        k
        for k in grid
        if math.isfinite(context.surface.implied_vol_at(k / context.spot, inputs.maturity, extrapolate=True))
    ]
    where = (
        f"at this expiry its smile reaches the strikes {strike_text(reached[0], step)} to "
        f"{strike_text(reached[-1], step)}"
        if reached
        else "the surface reaches no strike at this expiry"
    )
    return (
        f"The surface gives no vol at the strike {strike_text(inputs.strike, step)}: {where}, so nothing can be "
        "priced. Choose a strike in that range in the sidebar, or ATMF."
    )


def run() -> None:
    """One script run of the app: the page settings, stylesheet and logo, the sidebar, the header strip and the open
    tab. The shell draws no price cards (UI-1): the Pricing tab owns its headline tiles."""
    st.set_page_config(
        page_title="Options Analytics",
        # Streamlit links a Material icon favicon from fonts.gstatic.com: the app's only request to another host.
        # Where that host is blocked, the browser tab shows its default icon; everything else is served by the app.
        page_icon=":material/candlestick_chart:",
        layout="wide",
        # Open on desktops; on phones the sidebar would cover the whole page, so it starts collapsed there.
        initial_sidebar_state="auto",
        menu_items={
            "Get help": REPO_URL,
            "Report a bug": f"{REPO_URL}/issues",
            "About": f"Options Analytics v{__version__} (Option-Pricing-and-Strategy-Analyzer) by Navnoor Bawa. "
            "Educational tool, not investment advice.",
        },
    )
    theme.apply_stylesheet()
    header.logo()
    _read_link()
    inputs = sidebar()
    if not math.isfinite(inputs.volatility):
        # A strike beyond the surface's reach at the focus expiry (typed in the sidebar, or from a link): the tabs
        # cannot price it, so the page says why and stops, rather than fail inside the formulas.
        header.render(inputs, on_methodology=tab_methodology.open_drawer)
        st.error(beyond_reach(inputs), icon=":material/error:")
        st.stop()

    # The engine's prices come first (cached; under the closed form two formulas), so that if the American engine
    # cannot value the contract the header's badge names the engine of the values on screen.
    fallback = None
    try:
        prices = tab_pricing.compute_prices(inputs)
    except ValueError as exc:
        fallback = f"{inputs.model.label}: {exc} Showing Black-Scholes-Merton values instead."
        inputs = replace(inputs, model=replace(inputs.model, kind="bsm"))
        prices = tab_pricing.compute_prices(inputs)
    share = header.render(inputs, on_methodology=tab_methodology.open_drawer)
    drawer = st.session_state.pop(tab_methodology.DRAWER_ASKED, False)
    for note in st.session_state.pop(_LINK_NOTES, ()):
        st.warning(note, icon=":material/link_off:")
    for note in market_notes():
        st.info(note, icon=":material/event:")
    if fallback:
        st.warning(fallback, icon=":material/warning:")

    # Only the open tab runs: opening a tab reruns the app, and the closed tabs stay empty (their widgets keep their
    # values with persist_state). The exception is the Strategy tab's controls (template, strike width, position
    # size, commission), which are drawn on every run: they are bound to the URL, and Streamlit drops the URL
    # parameter of a widget it does not draw, so a link to any tab keeps the position. The binding to the URL alone
    # would rerun on a click as well; on_change says so explicitly.
    tabs = st.tabs(TABS, key="tab", on_change="rerun", bind="query-params")
    pricing, strategy, risk, surface, iv, methodology = tabs
    with strategy:
        tab_strategy.controls(inputs)
    # With the Strategy tab closed, its position is built without drawing anything (no analysis: 5-10 ms), whichever
    # tab is open, and before that tab runs. The legs editor commits a cell when it loses the focus, so an edit can
    # arrive with the click on another tab, and Streamlit drops the state of the editor at the end of a run that does
    # not draw it: build_position keeps that edit in the session. The Risk tab analyses the position.
    position = None if strategy.open else tab_strategy.build_position(inputs)
    if pricing.open:
        with pricing:
            tab_pricing.render(inputs, prices)
    if strategy.open:
        with strategy:
            tab_strategy.render(inputs)
    if risk.open:
        with risk:
            tab_risk.render(position, inputs)
    if surface.open:
        with surface:
            tab_surface.render(inputs)
    if iv.open:
        with iv:
            tab_iv.render(inputs)
    if methodology.open:
        with methodology:
            tab_methodology.render()
    # Last, once every widget bound to the URL has written its value there.
    header.fill_share(
        share, pinned={"asof": f"{inputs.context.as_of:%Y-%m-%dT%H:%M}"} if inputs.context is not None else None
    )
    # The Methodology drawer, over the open tab (which stays as it was), starting with that tab's section.
    if drawer:
        tab_methodology.drawer(next(label for label, tab in zip(TABS, tabs, strict=True) if tab.open))
