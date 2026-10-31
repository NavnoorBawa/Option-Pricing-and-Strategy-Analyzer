"""The app's published surface: one :class:`~options_analyzer.vol_surface.PublishedSurface` per market context.

The tabs will read their forwards, vols, risk-neutral ranges, quotes and delta strikes from :func:`current` (UI spec
2.0: the published surface is owned here and read everywhere else) as each tab's redesign adopts it; none reads it
yet. Under the synthetic index preset it is the surface fitted to the synthetic chain at the valuation time
(:func:`.market.index_market`); an uploaded chain will publish through :func:`publish` once the Surface tab reads
uploads in the canonical schema (roadmap SF-11). The textbook preset has no chain and so no published surface:
:func:`current` returns None, and a tab will fall back to its flat vol, labelled ``flat vol``.

The mathematics lives in :mod:`options_analyzer.vol_surface.published`, which holds no Streamlit code, so the
library and its tests use it without the app; this module only builds and caches it.
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from ..vol_surface import PublishedSurface
from . import market as mk
from .inputs import Inputs


@st.cache_resource(show_spinner=False, max_entries=8)
def _published(as_of: datetime, _context: mk.MarketContext) -> PublishedSurface:
    """Cached per valuation time, as :func:`.market.index_market` caches the context itself (one per time)."""
    return publish(_context)


def publish(context: mk.MarketContext) -> PublishedSurface:
    """The published surface of a market context: its fitted surface, its expiries' forwards and discount factors,
    and its chain with the implied vols of every quote (about 30 ms for the default chain's 5,020 quotes).

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
    """The surface every tab reads in this run: the index preset's (cached per valuation time), or None under the
    textbook preset, which has no chain.

    Parameters
    ----------
    inputs : Inputs
        The sidebar's market.

    Returns
    -------
    PublishedSurface or None
    """
    context = inputs.context
    return None if context is None else _published(context.as_of, context)
