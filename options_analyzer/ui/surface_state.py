"""The app's published surface: one :class:`~options_analyzer.vol_surface.PublishedSurface` per market context, and
the fit settings it is built with.

The tabs will read their forwards, vols, risk-neutral ranges, quotes and delta strikes from :func:`current` (UI spec
2.0: the published surface is owned here and read everywhere else) as each tab's redesign adopts it. Under the
synthetic index preset it is the surface fitted to the synthetic chain at the valuation time
(:func:`.market.index_market`) with the Surface tab's fit settings (:func:`settings`); an uploaded chain is fitted on
the Surface tab only and does not replace it. The textbook preset has no chain and so no published surface:
:func:`current` returns None, and a tab will fall back to its flat vol, labelled ``flat vol``.

The mathematics lives in :mod:`options_analyzer.vol_surface.published` and :mod:`~options_analyzer.vol_surface.
diagnostics`, which hold no Streamlit code, so the library and its tests use them without the app; this module only
builds and caches the surface and reads the settings.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING

import streamlit as st

from ..vol_surface import PublishedSurface
from ..vol_surface.diagnostics import DEFAULT_SETTINGS, FitSettings, QuoteFilters
from . import market as mk

if TYPE_CHECKING:  # inputs imports this module, for the settings
    from .inputs import Inputs

#: The Surface tab's Filters popover: session-state key, label and default of each fit setting. The widgets keep
#: their values for the session (``persist_state``) while the tab is closed, so every tab reads the same settings.
FORWARD_CHOICES = {"Put-call parity": "parity", "Rate and yield": "rates"}
CALIBRATION_CHOICES = {"Arbitrage-free": "arbitrage_free", "Unconstrained": "unconstrained"}
#: Spread limits offered, as multiples of the expiry's median spread (inf: no limit).
SPREAD_CHOICES = (1.0, 2.0, 4.0, 8.0, math.inf)
KEYS = {
    "otm_only": "sf_otm",
    "max_spread": "sf_spread",
    "min_strikes": "sf_strikes",
    "min_days": "sf_days",
    "weeklies": "sf_weeklies",
    "forward": "sf_fwd",
    "calibration": "sf_cal",
}
#: The widget defaults: the default settings, as the widgets hold them.
WIDGET_DEFAULTS = {
    "sf_otm": DEFAULT_SETTINGS.filters.otm_only,
    "sf_spread": DEFAULT_SETTINGS.filters.max_spread,
    "sf_strikes": DEFAULT_SETTINGS.filters.min_strikes,
    "sf_days": DEFAULT_SETTINGS.filters.min_days,
    "sf_weeklies": DEFAULT_SETTINGS.filters.weeklies,
    "sf_fwd": next(k for k, v in FORWARD_CHOICES.items() if v == DEFAULT_SETTINGS.forward),
    "sf_cal": next(k for k, v in CALIBRATION_CHOICES.items() if v == DEFAULT_SETTINGS.calibration),
}


def settings() -> FitSettings:
    """The fit settings in the session (the Surface tab's Filters popover), the defaults where a widget has not run.

    Returns
    -------
    FitSettings
    """
    state = st.session_state

    def get(key: str):
        value = state.get(key)
        return WIDGET_DEFAULTS[key] if value is None else value

    filters = QuoteFilters(
        otm_only=bool(get("sf_otm")),
        max_spread=float(get("sf_spread")),
        min_strikes=int(get("sf_strikes")),
        min_days=float(get("sf_days")),
        weeklies=bool(get("sf_weeklies")),
    )
    return FitSettings(
        filters, FORWARD_CHOICES.get(get("sf_fwd"), "parity"), CALIBRATION_CHOICES.get(get("sf_cal"), "arbitrage_free")
    )


@st.cache_resource(show_spinner=False, max_entries=8)
def _published(as_of: datetime, fit_settings: FitSettings, _context: mk.MarketContext) -> PublishedSurface:
    """Cached per valuation time and fit settings, as :func:`.market.index_market` caches the context itself."""
    return publish(_context)


def publish(context: mk.MarketContext) -> PublishedSurface:
    """The published surface of a market context: its fitted surface, its expiries' forwards and discount factors,
    and its chain with the implied vols of every quote (about 30 ms for the default chain's 5,012 quotes).

    Parameters
    ----------
    context : MarketContext
        The index market at one valuation time.

    Returns
    -------
    PublishedSurface
    """
    return PublishedSurface.build(
        context.surface,
        context.expiries,
        context.chain.quotes,
        spot=context.spot,
        as_of=context.as_of,
        source=context.chain.label,
    )


def current(inputs: Inputs) -> PublishedSurface | None:
    """The surface every tab reads in this run: the index preset's (cached per valuation time and fit settings), or
    None under the textbook preset, which has no chain.

    Parameters
    ----------
    inputs : Inputs
        The sidebar's market.

    Returns
    -------
    PublishedSurface or None
    """
    context = inputs.context
    return None if context is None else _published(context.as_of, context.settings, context)
