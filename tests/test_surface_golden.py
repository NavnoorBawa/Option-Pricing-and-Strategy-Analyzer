"""Golden outputs of the volatility-surface code (roadmap item SF-2).

``tests/fixtures/surface_golden.jsonl`` was first recorded from the flat module ``options_analyzer/vol_surface.py``
just before it became the ``options_analyzer.vol_surface`` package, and recorded again when parity forwards (SF-3),
the spread-weighted arbitrage-free SVI calibration (SF-4) and Lee-bounded wings (SF-8) changed these outputs on
purpose. For every case below it holds what
:func:`~options_analyzer.vol_surface.parse_quotes`, :func:`~options_analyzer.vol_surface.fit_market_surface` (or
:func:`~options_analyzer.vol_surface.ssvi_surface`), :func:`~options_analyzer.vol_surface.surface_summary` and
:meth:`~options_analyzer.vol_surface.VolSurface.implied_vol_at` returned or raised:

* messages, notes and error messages verbatim, with the :class:`~options_analyzer.vol_surface.QuoteWarning` ones marked;
* SVI parameters, each SVI slice's total variance at nine points of its quoted range, maturities, quoted ranges, fit
  RMSEs, the summary table and implied vols at chosen points as exact floats (JSON writes the shortest decimal that
  reads back as the same double);
* every large array as its shape, NaN count, sum and the SHA-256 of its float64 bytes, and its values to 10
  significant digits: all of them for the K/S grid and the implied- and local-vol grids, 256 at evenly spaced
  positions for the cleaned and fitted quote tables.

The cases are the sample quote tables (``sample_quotes`` seeds 0-3, implied vols and prices), the two CSV templates the
Surface tab offers, the three bundled chain fixtures (``tests/fixtures/chain_*.csv``), synthetic chains
(``market_data.sample_chain``), SSVI grids, small tables that take every branch of the quote reader but one it cannot
reach (a price grid of no positive price), cleaned quotes that raise every note
:func:`~options_analyzer.vol_surface.fit_market_surface` writes, in each of its wordings, and quotes it is not meant
for (not cleaned, or a spot that is not positive), where the error and the RuntimeWarnings are recorded.

Two checks. The last bits of ``exp``, ``log`` and the BLAS kernels depend on the library versions and on the CPU, so
bit-identity is only defined on the platform the fixture was recorded on: there, every output must be identical
(the whole case, serialised, digests included). Everywhere, every text, count, shape and NaN must be identical and
every number but the raw SVI parameters must agree to a relative 1e-6 (absolute 1e-9 near zero), the array values one
by one. Measured on the recording machine with NumPy's AVX-512 code paths and OpenBLAS's own kernel switched off
(``NPY_DISABLE_CPU_FEATURES``, ``OPENBLAS_CORETYPE=Haswell``), and with NumPy 1.26, SciPy 1.11 and pandas 2.1, every
text, count and NaN was the same, and the largest difference was 4.2% of that tolerance (4.2e-8 relative, a butterfly
in vol points; 4.1e-8 in the local-vol grids). Raw SVI parameters are not identifiable on a flat smile (there ``s``
moved by a factor of three while every vol agreed), so they are compared only through the variances and vols they
give.

The fit is not bit-reproducible on every input, before the split as after it: on some quote tables (about one in
twelve random ones) repeating the same call can end :func:`~options_analyzer.vol_surface.fit_svi`'s
Levenberg-Marquardt iterations (SciPy's MINPACK) a few ulps apart, depending on the state of the heap. That moved
implied and local vols by at most 4e-11 relative and never changed a text, a count or a NaN; the root cause is not
isolated. No case here gave a second result over 25 heap layouts, so a bit-identity failure on the recording platform
is a change of behaviour.

To record the fixture again (only for a change that is meant to alter these outputs)::

    PYTHONPATH=. python tests/test_surface_golden.py --record
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
import os
import platform
import sys
import warnings
from collections.abc import Callable, Iterator
from datetime import date, datetime
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy

from options_analyzer import black_scholes as bs
from options_analyzer import vol_surface as vs
from options_analyzer.implied_vol import implied_volatility
from options_analyzer.market_data import read_chain, sample_chain, synthetic, to_quotes

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = FIXTURES / "surface_golden.jsonl"
HEADER = (
    "# Provenance: synthetic",
    "# Description: outputs of options_analyzer.vol_surface for the cases in tests/test_surface_golden.py, recorded",
    "# after SF-3, SF-4 and SF-8 (parity forwards, the arbitrage-free SVI fit). The first JSON line holds the",
    "# recording platform and the module's names; each further line holds one case. Record again with",
    "# PYTHONPATH=. python tests/test_surface_golden.py --record",
)
#: Off the recording platform numbers must agree to RTOL (ATOL near zero). Since SF-4 the default calibration ends in
#: a constrained search (SLSQP) that stops at a tolerance, not at a bit pattern: with OpenBLAS on one thread instead
#: of several, or with NumPy's AVX-512 paths and OpenBLAS's own kernel switched off (NPY_DISABLE_CPU_FEATURES,
#: OPENBLAS_CORETYPE=Haswell), the largest relative difference outside the cases below was 1.0e-5 (a fit RMSE of the
#: bundled chain fixtures) and 5.1e-6 elsewhere; it was 4.2e-8 before SF-4.
RTOL, ATOL = 2e-5, 1e-9
#: The median |error| / half-spread of a near-exact fit is about 0.01 to 0.03 half-spread, solver noise: it moved by
#: up to 1.4e-5 half-spread (5e-4 relative) under those settings, so it is compared to this many half-spreads instead.
ERROR_RATIO = "Median |error| / half-spread"
ERROR_RATIO_ATOL = 5e-5
#: Quotes that hold static arbitrage themselves (or a smile at zero variance): their arbitrage-free fit is the end of
#: a constrained search whose path depends on the last bits of the BLAS kernels, so off the recording platform it may
#: end at another arbitrage-free smile, or, on fit_zero_total_variance, at none (and the surface says so). Under the
#: settings above local vols moved by up to 56 % and implied vols by up to 54 % on these four. Off the recording
#: platform they are checked for what holds on every path (test_path_sensitive_cases_keep_their_invariants).
PATH_SENSITIVE = (
    "fit_vertical_spread_arbitrage",
    "fit_zero_total_variance",
    "fit_rounded_midpoint_vols",
    "fit_rounded_midpoint_vols_with_a_spike",
)
DIGITS = 10  # significant digits of the array values kept for the comparison off the recording platform
SAMPLE = 256  # values kept of each quote table (the grids are kept whole)

AS_OF = datetime(2026, 10, 26, 10, 0)  # the valuation time of the chain fixtures (tests/test_market_data.py)
CHAIN_SPOT, UST, YIELD = 7915.0, 0.04, 0.013
BOX = UST + synthetic.BOX_SPREAD
SPY_DIVIDENDS = [(date(2026, 12, 18), 1.85), (date(2027, 3, 19), 1.75), (date(2027, 6, 18), 1.80)]


# --- Serialising outputs ------------------------------------------------------------------------------------------


def _float(value: float) -> float | str | None:
    """A float for JSON: itself (exact), None for NaN, a string for an infinity."""
    value = float(value)
    if math.isnan(value):
        return None
    return value if math.isfinite(value) else str(value)


def _floats(values: Any) -> list:
    return [_floats(v) if np.ndim(v) else _float(v) for v in np.asarray(values, dtype=float)]


def _array(values: Any, *, sample: bool = False) -> dict[str, Any]:
    """Shape, NaN count, sum and SHA-256 of an array's float64 bytes (little endian, C order), and its values in C
    order to :data:`DIGITS` significant digits: all of them, or with ``sample`` at most :data:`SAMPLE` of them at
    evenly spaced positions (a fixed choice for a given shape)."""
    array = np.ascontiguousarray(values, dtype="<f8")
    finite = array[np.isfinite(array)]
    flat = array.ravel()
    if sample and flat.size > SAMPLE:
        flat = flat[np.linspace(0, flat.size - 1, SAMPLE).round().astype(int)]
    return {
        "shape": list(array.shape),
        "nan": int(np.isnan(array).sum()),
        "sum": _float(finite.sum()) if finite.size else 0.0,
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "values": [_float(float(f"{v:.{DIGITS}g}")) for v in flat],
    }


def _frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Columns and dtypes, and the index and the (numeric) columns side by side as one array (values sampled)."""
    values = np.column_stack([frame.index.to_numpy(dtype=float), frame.to_numpy(dtype=float)])
    return {
        "columns": [str(c) for c in frame.columns],
        "dtypes": [str(t) for t in frame.dtypes],
        **_array(values, sample=True),
    }


def _error(exc: Exception) -> dict[str, str]:
    return {"error": type(exc).__name__, "message": str(exc)}


def _surface(surface: vs.VolSurface) -> dict[str, Any]:
    return {
        "source": surface.source,
        "carry": _float(surface.carry),
        "calendar_violations": surface.calendar_violations,
        "butterfly_violations": surface.butterfly_violations,
        "quoted_points": surface.quoted_points,
        "notes": list(surface.notes),
        "maturities": _floats(surface.maturities),
        "moneyness": _array(surface.moneyness),
        "implied_vol": _array(surface.implied_vol),
        "local_vol": _array(surface.local_vol),
        "quotes": None if surface.quotes is None else _frame(surface.quotes),
        "fit_rmse": None if surface.fit_rmse is None else _floats(surface.fit_rmse),
        "svi": [_floats(dataclasses.astuple(fit)) for fit in surface.svi],
        "svi_variance": _svi_variance(surface),
        "quoted_y": None if surface.quoted_y is None else _floats(surface.quoted_y),
        "ssvi": None if surface.ssvi is None else _floats(dataclasses.astuple(surface.ssvi)),
    }


def _svi_variance(surface: vs.VolSurface) -> list:
    """Each SVI slice's total variance at nine evenly spaced points of its quoted forward log-moneyness range: what
    the raw parameters give where they are identified by the quotes."""
    if surface.quoted_y is None:
        return []
    return [
        _floats(fit.total_variance(np.linspace(lo, hi, 9))[0])
        for fit, (lo, hi) in zip(surface.svi, surface.quoted_y, strict=True)
    ]


def _points(surface: vs.VolSurface) -> list:
    """Implied vols at chosen K/S between maturities, and within half a day of the first and the last."""
    Ts = surface.maturities
    times = [Ts[0] - 0.4 / 365, Ts[0] + 0.1 / 365, *(0.5 * (Ts[1:] + Ts[:-1])), Ts[-1] + 0.4 / 365]
    levels = (0.9, 1.0, 1.07)
    return [[_float(surface.implied_vol_at(m, T)) for m in levels] for T in times]


def _summary(surface: vs.VolSurface) -> dict[str, list]:
    summary = vs.surface_summary(surface)
    return {str(column): _floats(summary[column]) for column in summary.columns}


def _surface_outputs(make: Callable[[], vs.VolSurface]) -> dict[str, Any]:
    try:
        surface = make()
    except ValueError as exc:  # QuoteParseError, or a numerical failure of the fit
        return {"fit": _error(exc)}
    return {"fit": _surface(surface), "summary": _summary(surface), "points": _points(surface)}


def _quote_outputs(frame: pd.DataFrame, parse: dict[str, float], *, fit: bool = True) -> dict[str, Any]:
    """Parse ``frame`` with the keyword arguments ``parse``, then (if ``fit``) fit the cleaned quotes at the parse
    market, or at that of ``sample_quotes`` when the parse has no spot."""
    try:
        clean, messages = vs.parse_quotes(frame, **parse)
    except ValueError as exc:
        return {"parse": _error(exc)}
    outputs: dict[str, Any] = {
        "parse": {
            "messages": [str(message) for message in messages],
            "warnings": [i for i, message in enumerate(messages) if isinstance(message, vs.QuoteWarning)],
            "clean": _frame(clean),
        }
    }
    if not fit:
        return outputs
    market = (parse.get("spot", 100.0), parse.get("rate", 0.04), parse.get("dividend_yield", 0.0))
    return outputs | _surface_outputs(lambda: vs.fit_market_surface(clean, *market))


# --- The cases ----------------------------------------------------------------------------------------------------


def _ssvi_vols(params: vs.SSVIParams, carry: float, days, moneyness, spot: float = 100.0) -> pd.DataFrame:
    """Exact SSVI implied vols as a cleaned quotes table (no price errors)."""
    rows = []
    for d in days:
        t = d / 365.0
        for m in moneyness:
            iv = float(np.sqrt(params.total_variance(np.log(m) - carry * t, t)[0] / t))
            rows.append({"strike": spot * m, "maturity": round(t, 10), "iv": iv})
    return pd.DataFrame(rows)


def _sample_cases() -> Iterator[tuple[str, Callable[[], dict]]]:
    for seed in range(4):
        yield f"sample_quotes_seed{seed}", lambda seed=seed: _quote_outputs(vs.sample_quotes(seed=seed), {})
        yield (
            f"sample_prices_seed{seed}",
            lambda seed=seed: _quote_outputs(
                vs.sample_quotes(seed=seed, prices=True), {"spot": 100.0, "rate": 0.04, "dividend_yield": 0.0}
            ),
        )
    for prices in (False, True):  # the Surface tab's downloads at the default sidebar (spot 100, rate 5%, no yield)
        yield (
            f"app_template_{'prices' if prices else 'quotes'}",
            lambda prices=prices: _quote_outputs(
                pd.read_csv(io.StringIO(vs.sample_quotes(100.0, 0.05, 0.0, prices=prices).to_csv(index=False))),
                {"spot": 100.0, "rate": 0.05, "dividend_yield": 0.0},
            ),
        )


def _chain_cases() -> Iterator[tuple[str, Callable[[], dict]]]:
    for layout in ("long", "yfinance", "cboe_wide"):
        yield (
            f"fixture_chain_{layout}",
            lambda layout=layout: _quote_outputs(
                to_quotes(read_chain(FIXTURES / f"chain_{layout}.csv", as_of=AS_OF)[0], as_of=AS_OF),
                {"spot": CHAIN_SPOT, "rate": BOX, "dividend_yield": YIELD},
            ),
        )
    synthetic_chains = {
        "synthetic_index": (6000.0, YIELD, {}, YIELD),
        "synthetic_index_model": (6000.0, YIELD, {"noise": False}, YIELD),
        "synthetic_etf": (600.0, SPY_DIVIDENDS, {"style": "etf"}, 0.0),
    }
    for name, (spot, dividends, options, dividend_yield) in synthetic_chains.items():
        yield (
            name,
            lambda spot=spot, dividends=dividends, options=options, q=dividend_yield: _quote_outputs(
                to_quotes(sample_chain(spot, UST, dividends, as_of=AS_OF, **options).quotes, as_of=AS_OF),
                {"spot": spot, "rate": BOX, "dividend_yield": q},
            ),
        )


def _ssvi_cases() -> Iterator[tuple[str, Callable[[], dict]]]:
    grids = {
        "ssvi_default": (vs.SSVIParams(), 0.04, 0.01, np.linspace(0.7, 1.3, 25), np.linspace(0.1, 2.0, 12)),
        "ssvi_app_grid": (vs.SSVIParams(), 0.05, 0.0, np.linspace(0.6, 1.4, 41), np.linspace(0.05, 2.0, 30)),
        "ssvi_beyond_eta_bound": (
            vs.SSVIParams(0.3, 0.2, 0.8, rho=-0.7, eta=1.6, gamma=0.3),
            0.02,
            0.03,
            np.linspace(0.5, 1.6, 23),
            np.linspace(0.02, 3.0, 15),
        ),
    }
    for name, args in grids.items():
        yield name, lambda args=args: _surface_outputs(lambda: vs.ssvi_surface(*args))


def _quote_tables() -> dict[str, tuple[pd.DataFrame, dict[str, float]]]:
    """Small quote tables that take the branches of parse_quotes: name -> (table, parse arguments)."""
    iv = vs.sample_quotes(seed=0)  # Strike, Maturity (years, 4 decimals), ImpliedVol (decimal, 4 decimals)
    px = vs.sample_quotes(seed=0, prices=True)  # Strike, Days, Type, Price
    K, T, V = iv["Strike"], iv["Maturity"], iv["ImpliedVol"]
    D = (T * 365).round().astype(int)
    ranks = T.map({t: i + 1 for i, t in enumerate(sorted(T.unique()))})
    expiry = pd.Timestamp("2026-10-26") + pd.to_timedelta(D, unit="D")
    market = {"spot": 100.0, "rate": 0.04, "dividend_yield": 0.0}
    tables: dict[str, tuple[pd.DataFrame, dict[str, float]]] = {}

    def add(name, frame, parse=None):
        tables[name] = (pd.DataFrame(frame), parse or {})

    # Column names and units
    add("iv_days", {"Strike": K, "Days": D, "ImpliedVol": V})
    add("iv_percent_values", {"Strike": K, "Maturity": T, "ImpliedVol": V * 100})
    add("iv_percent_header", {"Strike": K, "Maturity": T, "IV (%)": V * 100})
    add("iv_percent_signs", {"Strike": K, "Maturity": T, "Implied Volatility": [f"{v * 100:.2f}%" for v in V]})
    add("iv_ambiguous_median", {"Strike": K, "Maturity": T, "IV": V * 10})
    add("vol_alias_and_days_header", {"Strike": K, "Maturity (days)": D, "Vol": V})
    add("bracketed_and_suffixed_headers", {"[STRIKE]": K, "Maturity_years": T, "Impl vol": V})
    add("maturity_in_days_unlabelled", {"Strike": K, "Maturity": D, "ImpliedVol": V})
    add("maturity_months", {"Strike": K, "Maturity (months)": (T * 12).round(4), "ImpliedVol": V})
    add("maturity_weeks", {"Strike": K, "Maturity (weeks)": (D / 7).round(4), "ImpliedVol": V})
    add("dte_hours", {"Strike": K, "DTE (hours)": D * 24, "ImpliedVol": V})
    add("maturity_m_refused", {"Strike": K, "Maturity (m)": T * 12, "ImpliedVol": V})
    add("years_whole_small", {"Strike": K, "Maturity": ranks.where(ranks <= 3), "ImpliedVol": V})
    add("years_whole_refused", {"Strike": K, "Maturity": ranks, "ImpliedVol": V})
    add("years_long_unlabelled", {"Strike": K, "Maturity": T * 2.5, "ImpliedVol": V})
    add("tenors_refused", {"Strike": K, "Tenor": [f"{max(round(t * 12), 1)}M" for t in T], "ImpliedVol": V})
    add("maturity_text_refused", {"Strike": K, "Maturity": "soon", "ImpliedVol": V})
    add("dates_refused", {"Strike": K, "Expiry": expiry.dt.strftime("%Y-%m-%d"), "ImpliedVol": V})
    add("yyyymmdd_refused", {"Strike": K, "Maturity": expiry.dt.strftime("%Y%m%d").astype(int), "ImpliedVol": V})
    add("expiration_dates_and_ttm", {"Strike": K, "Expiration": expiry.dt.strftime("%m/%d/%Y"), "TTM": T, "IV": V})
    add("days_and_maturity", {"Strike": K, "Maturity": T, "DTE": D, "IV": V})
    # Number formats
    add("thousands_separators", {"Strike": [f"{k * 50:,.2f}" for k in K], "Maturity": T, "IV": V})
    decimal = {"Strike": K, "Maturity": T, "IV": V}
    add("decimal_commas", {c: [f"{x:g}".replace(".", ",") for x in values] for c, values in decimal.items()})
    add("bad_commas", {"Strike": ["1,0,0.5", *map(str, K[1:])], "Maturity": T, "IV": V})
    add("currency_signs_and_spaces", {"Strike": [f"$ {k:.2f}" for k in K], "Maturity": T, "IV": [f" {v} " for v in V]})
    # Cleaning
    add("placeholder_vols", {"Strike": K, "Maturity": T, "IV": V.where(V.index % 9 != 0, 1e-05)})
    add("vols_above_500_percent", {"Strike": K, "Maturity": T, "IV": V.where(V.index % 31 != 5, 6.0)})
    bad = iv.astype(object)
    bad.loc[3, "Strike"], bad.loc[17, "Maturity"], bad.loc[40, "ImpliedVol"], bad.loc[41, "ImpliedVol"] = (
        np.nan,
        -0.5,
        "n/a",
        np.inf,
    )
    add("invalid_rows", bad)
    add("duplicates", pd.concat([iv, iv.head(10).assign(ImpliedVol=V.head(10) + 0.01)], ignore_index=True))
    add("close_maturities", {"Strike": K, "Days": D + np.where((D == 30) & (K.index % 2 == 0), 0.2, 0.0), "IV": V})
    stacked = pd.concat(
        [iv.assign(Type="call", ImpliedVol=V + 0.004), iv.assign(Type="put", ImpliedVol=V - 0.003)], ignore_index=True
    )
    add("calls_and_puts_with_vols", stacked, market)
    add("calls_and_puts_with_vols_without_spot", stacked)
    add("unknown_types_with_vols", iv.assign(Type=["call", "put", "x", None] * 26), market)
    # Refusals
    add("no_strike", {"Price level": K, "Maturity": T, "IV": V})
    add("no_vols_or_prices", {"Strike": K, "Maturity": T, "Volume": V})
    add("nothing_valid", {"Strike": K, "Maturity": T, "IV": -V})
    add("only_placeholders", {"Strike": K, "Maturity": T, "IV": V * 0 + 1e-05})
    add("placeholder_percent", {"Strike": K, "Maturity": T, "IV (%)": (V * 100).where(V.index % 9 != 0, 0.001)})
    add("only_vols_above_500_percent", {"Strike": K, "Maturity": T, "IV (%)": V * 0 + 600.0})
    add("strike_not_numeric", {"Strike": ["abc"] * len(K), "Maturity": T, "IV": V})
    add("weeks_beyond_100_years", {"Strike": K, "Maturity (weeks)": D * 10, "IV": V})
    add("too_many_rows", {"Strike": np.ones(vs.MAX_ROWS + 1), "Maturity": 1.0, "IV": 0.2})
    # Option prices
    bid = (px["Price"] - 0.02).clip(lower=0.0).round(2)
    ask = (px["Price"] + 0.02).round(2)
    blank = px["Price"].where(px.index % 3 != 0)
    add("prices_without_spot", px)
    add("prices_without_type", px.drop(columns="Type"), market)
    add("bid_ask_only", px.drop(columns="Price").assign(Bid=bid, Ask=ask), market)
    add("price_partly_blank", px.assign(Price=blank, Bid=bid, Ask=ask), market)
    add("crossed_quotes", px.drop(columns="Price").assign(Bid=bid, Ask=ask.where(px.index % 7 != 2, 0.0)), market)
    add("last_trade_only", px.rename(columns={"Price": "Last"}), market)
    add("last_trade_fallback", px.assign(Price=blank, **{"Last price": px["Price"]}), market)
    add("zero_bids_where_price_blank", px.assign(Price=blank, Bid=bid.where(blank.notna(), 0.0), Ask=ask), market)
    add("last_trade_blank_where_price_blank", px.assign(Price=blank, Last=px["Price"].where(blank.notna())), market)
    add("mid_next_to_bid_ask", px.drop(columns="Price").assign(Mid=(bid + ask) / 2, Bid=bid, Ask=ask), market)
    add("volume_next_to_prices", px.assign(Vol=1000), market)
    add("bad_option_types", px.assign(Type=px["Type"].where(px.index % 11 != 4, "straddle")), market)
    add("prices_outside_bounds", px.assign(Price=px["Price"].where(px.index % 13 != 6, 150.0)), market)
    add("no_numeric_price", px.assign(Price="n/a"), market)
    add("prices_all_outside_bounds", px.assign(Price=150.0), market)
    intrinsic = px.copy()
    t = intrinsic.loc[0, "Days"] / 365.0
    intrinsic.loc[0, ["Strike", "Type", "Price"]] = 82.5, "call", 100.0 - 82.5 * np.exp(-0.04 * t)  # no put at 82.5
    add("price_without_time_value", intrinsic, market)
    both = pd.concat([px, px.assign(Type=px["Type"].map({"call": "put", "put": "call"}), Price=px["Price"] + 1.0)])
    add("prices_for_calls_and_puts", both.reset_index(drop=True), market)
    many = {"Strike": np.linspace(50.0, 150.0, vs.MAX_PRICE_QUOTES + 1), "Days": 30, "Type": "call", "Price": 1.0}
    add("too_many_prices", many, market)
    return tables


def _table_cases() -> Iterator[tuple[str, Callable[[], dict]]]:
    """Parse only: the fit of quotes like these is covered by the other cases."""
    for name, (frame, parse) in _quote_tables().items():
        yield f"table_{name}", lambda frame=frame, parse=parse: _quote_outputs(frame, parse, fit=False)


def _fit_cases() -> Iterator[tuple[str, Callable[[], dict]]]:
    """Cleaned quotes that raise every note of fit_market_surface."""
    params = vs.SSVIParams()
    days = (30, 91, 182, 365, 730)
    grid = np.linspace(0.75, 1.3, 23)
    grid9 = np.linspace(80.0, 120.0, 9)
    exact = _ssvi_vols(params, 0.04, days, grid)
    thin = _ssvi_vols(params, 0.04, (60,), (0.9, 1.0, 1.1))
    dense = _ssvi_vols(params, 0.04, (91, 182), np.linspace(0.6, 1.5, 260))
    noisy = exact.assign(iv=exact["iv"] + 0.03 * np.sin(40 * np.log(exact["strike"] / 100)))
    spike = exact.assign(
        iv=exact["iv"] + np.where((exact["strike"].round(6) == 100.0) & (exact["maturity"] < 0.1), 0.04, 0)
    )
    steep = exact.assign(iv=np.clip(0.25 - 2.0 * np.log(exact["strike"] / 100) * (exact["maturity"] > 1.5), 0.01, None))
    priced = exact.assign(price_error=0.01)
    narrow = pd.concat([exact[exact["maturity"] < 0.1], _ssvi_vols(params, 0.04, (365,), np.linspace(0.4, 2.2, 23))])
    # The 366-day expiry has the total variance of the 365-day one, to quote precision.
    flat = _ssvi_vols(params, 0.0, (30, 91, 182, 365, 366), np.linspace(0.8, 1.2, 9)).round({"iv": 4})
    year, next_day = (flat["maturity"] == round(d / 365, 10) for d in (365, 366))
    flat.loc[next_day, "iv"] = np.round(flat.loc[year, "iv"].to_numpy() * np.sqrt(365 / 366), 4)
    # Rounded vols at $1 strikes with one 30-day call quoted 6 vol points too high: a close fit, a vertical spread.
    smiles = _ssvi_vols(params, 0.05, (14, 30, 60, 91), np.arange(90, 111) / 100).round({"iv": 4})
    smiles.loc[(smiles["maturity"] == round(30 / 365, 10)) & (smiles["strike"].round(6) == 110.0), "iv"] += 0.06
    mixed = spike.assign(price_error=np.where(spike.index % 2 == 0, 0.02, np.nan))
    # Vols of 0.01 vol point at the money: the fitted SVI minimum variance is slightly negative there.
    near_zero = pd.DataFrame(
        [{"strike": k, "maturity": t, "iv": 1e-4 + np.log(k / 100) ** 2} for t in (0.1, 0.5) for k in grid9]
    )
    cases = {
        "ssvi_exact": (exact, 100.0, 0.04, 0.0),
        "ssvi_exact_with_carry": (_ssvi_vols(params, 0.03, days, grid), 100.0, 0.05, 0.02),
        "ssvi_exact_with_price_errors": (priced, 100.0, 0.04, 0.0),
        "one_maturity": (exact[exact["maturity"] == exact["maturity"].min()], 100.0, 0.04, 0.0),
        "thin_maturity_skipped": (pd.concat([exact, thin]).sort_values(["maturity", "strike"]), 100.0, 0.04, 0.0),
        "dense_slices_binned": (dense, 100.0, 0.04, 0.0),
        "far_from_spot": (exact, 160.0, 0.04, 0.0),
        "smile_with_humps": (noisy, 100.0, 0.04, 0.0),
        "butterfly_in_quotes": (spike, 100.0, 0.04, 0.0),
        "vertical_spread_arbitrage": (steep, 100.0, 0.04, 0.0),
        "short_and_wide_maturities": (narrow.reset_index(drop=True), 100.0, 0.04, 0.0),
        "fine_grid": (exact, 100.0, 0.04, 0.0, 301),
        "coarse_grid": (exact, 100.0, 0.04, 0.0, 5),
        "unknown_price_errors": (exact.assign(price_error=np.nan), 100.0, 0.04, 0.0),
        "butterfly_with_some_price_errors": (mixed, 100.0, 0.04, 0.0),
        "butterfly_with_price_errors": (spike.assign(price_error=0.02), 100.0, 0.04, 0.0),
        "flat_within_quote_precision": (flat, 100.0, 0.0, 0.0),
        "vertical_spread_with_a_close_fit": (smiles, 100.0, 0.05, 0.0),
        "zero_total_variance": (near_zero, 100.0, 0.0, 0.0),
    }
    for name, args in cases.items():
        yield f"fit_{name}", lambda args=args: _surface_outputs(lambda: vs.fit_market_surface(*args))
    # Arbitrage in quoted vols that the rounding of the prices behind them could cause, then a spike beyond it.
    for name, spike_size in (("rounded_midpoint_vols", 0.0), ("rounded_midpoint_vols_with_a_spike", 0.05)):
        yield (
            f"fit_{name}",
            lambda spike_size=spike_size: _quote_outputs(
                _midpoint_vols(37.4, 0.10, spike_size), {"spot": 37.4, "rate": 0.05, "dividend_yield": 0.0}
            ),
        )


def _midpoint_vols(spot: float, tick: float, spike: float) -> pd.DataFrame:
    """Vols (4 dp) inverted from bid/ask midpoints in ``tick`` ticks around exact SSVI prices at half-dollar strikes,
    with ``spike`` added to the 30-day at-the-money vol."""
    params, r, rows = vs.SSVIParams(), 0.05, []
    for days in (7, 14, 30, 60, 91):
        t = days / 365
        for strike in np.arange(np.ceil(1.4 * spot), np.floor(2.6 * spot) + 1) / 2:
            iv = np.sqrt(params.total_variance(np.log(strike / spot) - r * t, t)[0] / t)
            kind = "call" if strike >= spot * np.exp(r * t) else "put"
            price = bs.price(spot, strike, t, r, iv, 0.0, kind)
            half = max(tick, 0.03 * price)
            bid, ask = np.floor((price - half) / tick) * tick, np.ceil((price + half) / tick) * tick
            if bid > 0:
                vol = implied_volatility((bid + ask) / 2, spot, strike, t, r, 0.0, kind)
                vol += spike if days == 30 and abs(strike - spot) <= 0.25 else 0.0
                rows.append({"Strike": strike, "Days": days, "IV": round(vol, 4)})
    return pd.DataFrame(rows)


def _unclean_outputs(make: Callable[[], vs.VolSurface]) -> dict[str, Any]:
    """What a fit of quotes that :func:`~options_analyzer.vol_surface.parse_quotes` did not clean gives, first with
    every RuntimeWarning shown (and recorded), then with RuntimeWarnings raised as errors (as in this suite). An error
    is recorded by its type, and by its message only when it comes from this package or NumPy (SciPy's messages
    change between versions)."""
    outputs: dict[str, Any] = {}
    for action in ("always", "error"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter(action, RuntimeWarning)
            try:
                outcome: dict[str, Any] = _surface(make())
            except (ValueError, RuntimeWarning) as exc:
                outcome = {"error": type(exc).__name__}
                if isinstance(exc, (vs.QuoteParseError, RuntimeWarning)):
                    outcome["message"] = str(exc)
        found = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
        outputs[action] = {"outcome": outcome, "runtime_warnings": found}
    return outputs


def _unclean_cases() -> Iterator[tuple[str, Callable[[], dict]]]:
    """Quotes the fit is not meant for: it reads and fits one maturity at a time, so the first maturity it fails on
    decides the error and the warnings."""
    strikes = np.arange(80.0, 131.0, 5.0)
    first = pd.DataFrame({"strike": strikes, "maturity": 0.1, "iv": 0.2 + 0.1 * np.log(100.0 / strikes) ** 2})
    first.loc[4, "iv"] = np.nan
    second = first.assign(maturity=0.5, iv=0.2)
    second.loc[0, "strike"] = 0.0
    nan_then_zero = pd.concat([first, second], ignore_index=True)
    clean = _ssvi_vols(vs.SSVIParams(), 0.04, (30, 91, 182), np.linspace(0.8, 1.2, 9))
    cases = {
        "nan_vol_then_zero_strike": (nan_then_zero, 100.0),
        "negative_spot": (clean, -100.0),
        "nan_spot": (clean, np.nan),
    }
    for name, (quotes, spot) in cases.items():
        yield (
            f"fit_unclean_{name}",
            lambda quotes=quotes, spot=spot: _unclean_outputs(lambda: vs.fit_market_surface(quotes, spot, 0.04, 0.0)),
        )


def _cases() -> dict[str, Callable[[], dict]]:
    groups = (_sample_cases, _chain_cases, _ssvi_cases, _table_cases, _fit_cases, _unclean_cases)
    return dict(item for group in groups for item in group())


CASES = _cases()


# --- Recording and comparing --------------------------------------------------------------------------------------


def _environment() -> dict[str, Any]:
    """What decides the last bits of the outputs: Python, the numerical libraries, the CPU and its SIMD features, and
    whether OpenBLAS runs threaded and on which kernel (one thread gave other last bits than two or four)."""
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as features
    except ImportError:  # NumPy 1.x
        from numpy.core._multiarray_umath import __cpu_features__ as features
    return {
        "python": ".".join(platform.python_version_tuple()[:2]),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "system": platform.system(),
        "machine": platform.machine(),
        "simd": sorted(name for name, enabled in features.items() if enabled),
        "blas_threaded": int(os.environ.get("OPENBLAS_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS") or 0) != 1
        and (os.cpu_count() or 1) > 1,
        "blas_kernel": os.environ.get("OPENBLAS_CORETYPE", "auto"),
    }


def _names() -> dict[str, Any]:
    return {"names": sorted(dir(vs)), "all": list(vs.__all__)}


def _record() -> None:
    lines = [*HEADER, json.dumps({"environment": _environment(), **_names()})]
    lines += [json.dumps({"case": name, **make()}) for name, make in CASES.items()]
    GOLDEN.write_text("\n".join(lines) + "\n", encoding="utf-8")


@cache
def _golden() -> tuple[dict[str, Any], dict[str, dict]]:
    rows = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    return rows[0], {row.pop("case"): row for row in rows[1:]}


@cache
def _outputs(name: str) -> dict:
    return json.loads(json.dumps(CASES[name]()))  # the same JSON round trip as the recorded outputs


def _differences(actual: Any, expected: Any, path: str = "", *, exact: bool) -> list[str]:
    """Where two serialised outputs differ: everything must be equal, except that when not ``exact`` numbers may
    differ by RTOL/ATOL, and digests and raw SVI parameters are not compared."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        if actual.keys() != expected.keys():
            return [f"{path}: keys {sorted(actual)} != {sorted(expected)}"]
        return [
            found
            for key in expected
            if exact or key not in ("sha256", "svi")
            for found in _differences(actual[key], expected[key], f"{path}.{key}", exact=exact)
        ]
    if isinstance(expected, list) and isinstance(actual, list):
        if len(actual) != len(expected):
            return [f"{path}: length {len(actual)} != {len(expected)}"]
        return [
            found
            for i, pair in enumerate(zip(actual, expected, strict=True))
            for found in _differences(*pair, f"{path}[{i}]", exact=exact)
        ]
    if isinstance(expected, float) and isinstance(actual, float) and not exact:
        atol = ERROR_RATIO_ATOL if ERROR_RATIO in path else ATOL
        close = math.isclose(actual, expected, rel_tol=RTOL, abs_tol=atol)
        return [] if close else [f"{path}: {actual!r} != {expected!r}"]
    same = type(actual) is type(expected) and (
        repr(actual) == repr(expected) if isinstance(actual, float) else actual == expected
    )
    return [] if same else [f"{path}: {actual!r} != {expected!r}"]


def test_the_fixture_covers_every_case():
    _, golden = _golden()
    assert list(golden) == list(CASES)


def test_every_name_of_the_flat_module_is_still_importable():
    """SF-2: dir(vol_surface) before the split is a subset of dir(vol_surface) after it, and __all__ is unchanged."""
    recorded, _ = _golden()
    assert set(recorded["names"]) - set(dir(vs)) == set()
    assert list(vs.__all__) == recorded["all"]
    star: dict[str, Any] = {}
    exec("from options_analyzer.vol_surface import *", star)
    assert set(star) - {"__builtins__"} == set(recorded["all"])


@pytest.mark.parametrize("name", list(CASES))
def test_outputs_agree_with_the_recording(name):
    """On any platform: every text, count and shape identical, every number within RTOL (ATOL near zero); the
    path-sensitive cases (PATH_SENSITIVE) on the recording platform only."""
    recorded, golden = _golden()
    if name in PATH_SENSITIVE and any(recorded["environment"].get(k) != v for k, v in _environment().items()):
        pytest.skip("its arbitrage-free search may end elsewhere off the recording platform; see PATH_SENSITIVE")
    assert _differences(_outputs(name), golden[name], exact=False) == []


@pytest.mark.parametrize("name", PATH_SENSITIVE)
def test_path_sensitive_cases_keep_their_invariants(name):
    """What holds on every path of the arbitrage-free search, on any platform: the same quotes, maturities and grid
    (not the points where the fit has a vol, which follow the fit); and every maturity either free of static
    arbitrage or named in the note that it kept its unconstrained fit."""
    _, golden = _golden()
    actual, expected = _outputs(name), golden[name]
    for key in ("maturities", "moneyness"):
        assert _differences(actual["fit"][key], expected["fit"][key], exact=False) == [], key
    assert actual.get("messages") == expected.get("messages")
    notes = " ".join(actual["fit"]["notes"])
    free = actual["fit"]["calendar_violations"] == 0 and actual["fit"]["butterfly_violations"] == 0
    assert free or "No SVI smile free of static arbitrage was found" in notes or "arbitrage" in notes


@pytest.mark.parametrize("name", list(CASES))
def test_outputs_are_bit_identical_on_the_recording_platform(name):
    recorded, golden = _golden()
    differs = [key for key, value in _environment().items() if recorded["environment"].get(key) != value]
    if differs:
        pytest.skip(f"bit-identity is defined on the recording platform only; this one differs in {', '.join(differs)}")
    assert _differences(_outputs(name), golden[name], exact=True) == []


if __name__ == "__main__":
    if sys.argv[1:] != ["--record"]:
        raise SystemExit("usage: PYTHONPATH=. python tests/test_surface_golden.py --record")
    _record()
