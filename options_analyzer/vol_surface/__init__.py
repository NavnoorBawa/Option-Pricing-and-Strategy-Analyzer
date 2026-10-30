"""Implied and local volatility surfaces.

Surfaces are handled in *total implied variance* ``w(y, T) = sigma_BS^2 T``
as a function of forward log-moneyness ``y = ln(K / F_T)``, and local
volatility is obtained from Gatheral's form of Dupire's formula
(Gatheral, *The Volatility Surface*, 2006, eq. 1.10)::

    sigma_loc^2 = (dw/dT) / g(y, T),
    g = 1 - (y/w) w' + 1/4 (-1/4 - 1/w + y^2/w^2) w'^2 + 1/2 w''

with ``'`` denoting derivatives in ``y`` at fixed ``T`` and ``dw/dT`` taken at
fixed ``y``. Working in total variance avoids differentiating option prices
numerically (the classic source of noise in Dupire's formula), and the two
factors are exactly the static no-arbitrage conditions: ``dw/dT >= 0``
(no calendar arbitrage) and ``g >= 0`` (Durrleman's condition, no butterfly
arbitrage). Points violating either have no valid local volatility; they are
reported and left as NaN instead of being silently patched.

Two sources are supported:

* a parametric **SSVI** surface (Gatheral & Jacquier, 2014) with a
  power-law curvature function and a mean-reverting ATM variance term
  structure - free of static arbitrage by construction when
  ``eta (1 + |rho|) <= 2``;
* **market quotes** (strike, maturity and an implied vol, or an option price
  that is inverted with Black-76 at its expiry's forward and discount factor,
  read from put-call parity where calls and puts are quoted at common strikes,
  else from the rate and dividend yield). Each maturity is fitted with
  Gatheral's raw SVI smile, which gives ``w'`` and ``w''`` analytically.
  ``dw/dT`` at fixed ``y`` comes from secants between neighbouring maturities
  evaluated at the same ``y``: their Fritsch-Butland weighted harmonic mean
  inside, the one-sided three-point formula at the first maturity and the
  one-sided secant at the last. A secant that falls by more than the quotes'
  precision is calendar arbitrage; one that changes by less resolves nothing
  and is not used, and where no secant is left local vol is undefined.

The package's modules, each importing only those before it:

* :mod:`.svi` - the raw SVI smile and its calibration (:func:`fit_svi`);
* :mod:`.ssvi` - the SSVI surface (:class:`SSVIParams`) and :func:`sample_quotes`;
* :mod:`.local_vol` - Durrleman's ``g``, Gatheral's local variance and ``dw/dT`` between SVI slices;
* :mod:`.forwards` - each expiry's forward and discount factor from put-call parity (:func:`implied_forwards`);
* :mod:`.quotes` - reading a quotes table (:func:`parse_quotes`) and model-free static-arbitrage checks of the quotes;
* :mod:`.surface` - :class:`VolSurface`, :func:`ssvi_surface`, :func:`fit_market_surface` and :func:`surface_summary`.

Every name the single module ``vol_surface.py`` had is importable from here, its imports included (roadmap item
SF-2). Code that patches a constant or a helper must patch it in the module that uses it.
"""

# Every name below is re-exported (__all__ lists the public API), so none is an unused import.
# ruff: noqa: F401

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil import parser as dateutil_parser
from scipy.optimize import least_squares
from scipy.special import ndtr

from .. import black_scholes as bs
from ..implied_vol import ImpliedVolError, implied_volatility
from .forwards import FORWARD_SOURCES, carry_gaps, implied_forwards
from .local_vol import EXTRAPOLATION, _local_vol, _market_values, _MarketValues, _time_derivative, durrleman_g
from .quotes import (
    _ALIAS_KEYS,
    _ALIASES,
    _BRACKETS,
    _DECIMAL_COMMA,
    _NAME_THE_UNIT,
    _PRICE_STEPS,
    _TENOR,
    _THOUSANDS,
    _TIME_OF_DAY,
    _TIME_UNITS,
    _UNIT_SUFFIXES,
    _UNITS,
    _WEAK_IV,
    EXERCISE_STYLES,
    FLY_CHECK_VOL,
    FORWARD_METHODS,
    MATURITY_TOLERANCE,
    MAX_DAYS,
    MAX_PRICE_QUOTES,
    MAX_ROWS,
    MAX_USUAL_YEARS,
    MIN_PRICE_ERROR,
    MIN_QUOTED_VOL,
    QUOTED_VOL_TICK,
    SPREAD_CHECK_VOL,
    UNKNOWN_PRICE_ERROR,
    QuoteParseError,
    QuoteWarning,
    _black_prices,
    _butterfly_arbitrage,
    _expiries,
    _GuessedYears,
    _header,
    _holds_times,
    _looks_like_dates,
    _match_columns,
    _maturities,
    _merge_close_maturities,
    _numeric,
    _option_kind,
    _option_prices,
    _out_of_the_money,
    _parse_date,
    _price_step,
    _prices_to_vols,
    _spread_arbitrage,
    _strike_text,
    _time_column,
    parse_quotes,
)
from .ssvi import SSVIParams, sample_quotes
from .surface import (
    FAR_FROM_SPOT,
    MAX_GRID_POINTS,
    MAX_POINTS_PER_SLICE,
    MIN_GRID_POINTS_PER_SLICE,
    MIN_STRIKES_PER_SLICE,
    POOR_FIT_RMSE,
    VolSurface,
    _binned,
    _moneyness_grid,
    fit_market_surface,
    ssvi_surface,
    surface_summary,
)
from .svi import _LOG_S_RANGE, _RHO_FACE, _RHO_MAX, QUOTE_PRECISION, SVISlice, _svi_start_points, fit_svi

__all__ = [
    "EXERCISE_STYLES",
    "FORWARD_METHODS",
    "FORWARD_SOURCES",
    "QuoteParseError",
    "QuoteWarning",
    "SSVIParams",
    "SVISlice",
    "VolSurface",
    "carry_gaps",
    "fit_market_surface",
    "fit_svi",
    "implied_forwards",
    "parse_quotes",
    "sample_quotes",
    "ssvi_surface",
    "surface_summary",
]
