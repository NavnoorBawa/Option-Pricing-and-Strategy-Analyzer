"""The market the app values against (the context of every tab).

Two presets, which the sidebar's Market menu offers (``?preset=`` in a link):

* **Synthetic index**, the default: an SPX-like chain priced by :func:`options_analyzer.market_data.sample_chain`
  at the valuation time (labelled synthetic everywhere), read like an uploaded chain, quote by quote
  (:func:`options_analyzer.vol_surface.diagnostics.fit_chain`): each expiry's forward and discount factor from
  put-call parity, the Surface tab's quote screen, and the arbitrage-free, spread-weighted SVI surface fitted to the
  quotes it keeps. That fitted surface is *published*: the Pricing contract's vol, the forward and the carry of its
  expiry all come from it. The fit settings (``vol_surface.diagnostics.FitSettings``: the screen, the forwards and
  the calibration) are the Surface tab's, and the header shows a FIT chip while one is not at its default. The
  sidebar picks a fitted expiry and a strike; the spot is the chain's.
* **Textbook**: S = K = 100, one year, σ 20%, r 5%, q 0, the inputs of version 2, typed in the sidebar (spot,
  strike, days, volatility, rate, dividend yield), with no chain.

:func:`index_market` builds and caches the index market for one valuation time (about a second, once per time and
server process). :func:`last_close` is the default valuation time: the last completed cash close, so the default
is the same for every visitor all day and the numbers on screen reproduce from a shared link.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from .. import clock
from .. import vol_surface as vs
from ..market_data import SyntheticChain, sample_chain, to_quotes
from ..market_data.synthetic import BOX_SPREAD, SYNTHETIC_LABEL, strike_interval
from ..vol_surface.diagnostics import DEFAULT_SETTINGS, ChainFit, FitSettings, fit_chain
from .format import time_left

#: The presets, by the label the sidebar shows and a link holds (``?preset=Textbook``).
INDEX, TEXTBOOK = "Synthetic index", "Textbook"
PRESETS = (INDEX, TEXTBOOK)
#: The synthetic index: SPX-like level, a flat Treasury zero rate (the options discount at the box rate,
#: ``synthetic.BOX_SPREAD`` above it, which parity reads back from the quotes) and a continuous dividend yield.
INDEX_ROOT = "SPX"
INDEX_SPOT = 5800.0
INDEX_RATE = 0.04
INDEX_YIELD = 0.013
#: The reference carry an expiry's forward is measured against (the Surface tab's carry gap): the Treasury rate plus
#: the box spread the options finance at, and the yield. It is the carry of the forwards when the Surface tab's
#: Forwards setting is "Rate and yield" (their carry gap is then 0). The parity fit keeps the Treasury rate as its
#: fallback, as it always has: no expiry of the default chain needs the fallback, but the rate seeds each parity
#: fit's first guess, and a different seed moves the 2028 forward by 0.25 bp.
INDEX_CARRY = (INDEX_RATE + BOX_SPREAD, INDEX_YIELD)
#: How every view of the synthetic chain is labelled (``synthetic.SYNTHETIC_LABEL``).
SYNTHETIC_NOTE = f"{SYNTHETIC_LABEL}."
#: The expiry the index preset opens on: the listed one nearest this many calendar days.
DEFAULT_DAYS = 30
#: A valuation time is kept on 5-minute steps.
ASOF_STEP = timedelta(minutes=5)


@dataclass(frozen=True, eq=False)  # holds tables: compared by identity
class MarketContext:
    """The index market at one valuation time: the synthetic chain, its fitted surface and one row per expiry.

    ``expiries`` holds, for each fitted expiry in date order (every listed one unless the fit settings leave some
    out): ``expiry`` (date), ``root``, ``settlement`` (AM or PM), ``settles_at``, ``maturity`` (years, minutes /
    525,600), ``forward`` and ``discount_factor`` (from put-call parity, as fitted), ``atm_vol`` (the surface's vol
    at the forward), and ``surface_index`` (the row of the surface's maturities). ``fit`` is the chain read quote by
    quote, with its ``settings``.
    """

    as_of: datetime
    chain: SyntheticChain
    quotes: pd.DataFrame  # the chain as vol_surface.parse_quotes reads it (to_quotes), for download
    surface: vs.VolSurface
    expiries: pd.DataFrame
    messages: tuple[str, ...]
    fit: ChainFit | None = None
    settings: FitSettings = DEFAULT_SETTINGS

    @property
    def spot(self) -> float:
        return float(self.chain.spot)

    @property
    def strike_step(self) -> float:
        """The listed strike interval near the money."""
        return float(strike_interval(self.spot, "index"))

    def expiry_row(self, expiry: date) -> pd.Series:
        """The row of ``expiries`` for a listed expiry date (the nearest one when it is not listed)."""
        days = np.abs((pd.to_datetime(self.expiries["expiry"]) - pd.Timestamp(expiry)).dt.days.to_numpy())
        return self.expiries.iloc[int(np.argmin(days))]

    def default_expiry(self) -> date:
        """The listed expiry nearest :data:`DEFAULT_DAYS` calendar days."""
        days = self.expiries["maturity"].to_numpy(float) * 365.0
        return self.expiries["expiry"].iloc[int(np.argmin(np.abs(days - DEFAULT_DAYS)))]

    def atm_strike(self, expiry: date) -> float:
        """The listed strike nearest the forward of ``expiry``."""
        step = self.strike_step
        return float(step * round(float(self.expiry_row(expiry)["forward"]) / step))


def last_close(now: datetime | None = None) -> datetime:
    """The last completed cash close (16:00 New York, 13:00 on early closes) at or before ``now``, as New York wall
    time without a time zone (as ``st.datetime_input`` holds it)."""
    now = clock.to_new_york(now or datetime.now(clock.NEW_YORK))
    close = clock.market_close(clock.previous_trading_day(now.date()))  # that day itself when it is a trading day
    if close > now:
        close = clock.market_close(clock.previous_trading_day(close.date() - timedelta(days=1)))
    return close.replace(tzinfo=None)


def snap(moment: datetime) -> datetime:
    """A valuation time on its 5-minute step, without seconds, as New York wall time (no time zone)."""
    moment = clock.to_new_york(moment).replace(tzinfo=None, second=0, microsecond=0)
    return moment - timedelta(minutes=moment.minute % 5)


@st.cache_resource(show_spinner=False, max_entries=8)
def index_market(as_of: datetime, settings: FitSettings = DEFAULT_SETTINGS) -> MarketContext:
    """The synthetic index chain at ``as_of`` (New York wall time), read and fitted like an uploaded chain with the
    fit ``settings`` (the Surface tab's: :func:`.surface_state.settings`).

    Cached per valuation time and settings for the server process, so every visitor at the default time shares one
    fit (about a second).
    """
    chain = sample_chain(INDEX_SPOT, INDEX_RATE, INDEX_YIELD, as_of=as_of)
    # Forwards from the rate and yield take the options' own carry (the box rate): from the Treasury rate they missed
    # the carry the gap is measured against by 45 bp a year (−1 to −7 bp on the first expiries, flagged as gaps).
    rate = INDEX_CARRY[0] if settings.forward == "rates" else INDEX_RATE
    fit = fit_chain(
        chain.quotes, as_of=as_of, spot=INDEX_SPOT, rate=rate, dividend_yield=INDEX_YIELD, settings=settings
    )
    surface = fit.surface
    fitted = fit.expiries[fit.expiries["slice"] >= 0]
    listed = chain.expiries.assign(expiry=pd.to_datetime(chain.expiries["expiry"]).dt.date)
    expiries = fitted[["expiry", "settlement"]].merge(listed, on=["expiry", "settlement"], how="left")
    expiries = expiries[["expiry", "root", "settlement", "settles_at", "maturity"]].copy()
    expiries["surface_index"] = fitted["slice"].to_numpy(int)
    expiries["forward"] = fitted["forward"].to_numpy(float)
    expiries["discount_factor"] = fitted["discount_factor"].to_numpy(float)
    expiries["atm_vol"] = [
        surface.implied_vol_at(F / INDEX_SPOT, T, extrapolate=True)
        for F, T in zip(expiries["forward"], expiries["maturity"], strict=True)
    ]
    quotes = to_quotes(chain.quotes, as_of=as_of)
    return MarketContext(as_of, chain, quotes, surface, expiries.reset_index(drop=True), fit.messages, fit, settings)


def expiry_label(row: pd.Series, *, as_of: datetime | None = None) -> str:
    """A listed expiry as the desk names it: ``20‑Nov‑26 AM · 27d 17h`` (non-breaking hyphens in the date), with the
    time left to its settlement as every tab writes it (:func:`span`)."""
    when = pd.Timestamp(row["expiry"]).strftime("%d-%b-%y").replace("-", "\u2011")
    return f"{when} {row['settlement']} · {span(float(row['maturity']) * 365.0)}"


def span(days: float) -> str:
    """Time to expiry by the number standard (:func:`.format.time_left`, a countdown: whole units, never rounded
    up): ``55d`` from 30 days, ``27d 17h`` from a day, ``5h 16m`` below.

    >>> span(55.73), span(27.73), span(3.26), span(0.22)
    ('55d', '27d 17h', '3d 6h', '5h 16m')
    """
    return time_left(days * 1440.0)


def carry(forward: float, discount_factor: float, spot: float, maturity: float) -> tuple[float, float]:
    """The rate and dividend yield that put an expiry's forward and discount factor into Black-Scholes-Merton:
    r_T = −ln(DF) / T and q_T = r_T − ln(F / S) / T, so that S·e^{(r_T − q_T)T} = F and e^{−r_T T} = DF.

    >>> r, q = carry(5815.22, 0.99640, 5800.0, 0.07597)
    >>> round(r, 5), round(q, 5)
    (0.04747, 0.01298)
    """
    rate = -math.log(discount_factor) / maturity
    return rate, rate - math.log(forward / spot) / maturity


def session_note(as_of: datetime) -> str:
    """The cash market's state at a valuation time (New York wall time), for the sidebar's caption: ``cash close``,
    ``early close``, ``regular hours``, ``before the open``, ``after the close``, ``weekend`` or ``holiday``.

    >>> session_note(datetime(2026, 10, 23, 16, 0)), session_note(datetime(2026, 11, 26, 12, 0))
    ('cash close', 'holiday')
    """
    day = as_of.date()
    if not clock.is_trading_day(day):
        return "weekend" if day.weekday() >= 5 else "holiday"
    close = clock.market_close(day).replace(tzinfo=None)
    if as_of == close:
        return "early close" if close.hour < 16 else "cash close"
    if as_of < datetime.combine(day, clock.MARKET_OPEN):
        return "before the open"
    return "regular hours" if as_of < close else "after the close"
