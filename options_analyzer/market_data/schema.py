"""The canonical option-chain table, and OCC option symbols.

The synthetic generator (:mod:`.synthetic`) and the upload adapters (:mod:`.adapters`) both produce this table, one
row per listed contract:

==========  ==============  ==========================================================================================
column      type            meaning
==========  ==============  ==========================================================================================
symbol      str             OCC option symbol without padding: root, expiry as YYMMDD, C or P, strike x 1000 in 8 digits
root        str             option root, e.g. SPX (AM-settled monthlies), SPXW (PM-settled weeklies) or SPY
expiry      datetime64[ns]  expiration date (a New York calendar date, at midnight)
settlement  str             ``"AM"`` (at the 09:30 ET open) or ``"PM"`` (at the close); see the clock module
type        str             ``"call"`` or ``"put"``
strike      float           strike, $ per share, on OCC's grid of 0.001
bid         float           best bid, $ per share; 0 when only an offer is quoted (a one-sided, zero-bid quote)
ask         float           best offer, $ per share
vendor_iv   float           the data vendor's implied vol (decimal), NaN when there is none; a diagnostic only, never
                            used to fit a surface
==========  ==============  ==========================================================================================
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd

__all__ = [
    "CHAIN_COLUMNS",
    "ChainSchemaError",
    "OccSymbol",
    "chain_frame",
    "fits_occ",
    "occ_symbol",
    "parse_occ_symbol",
    "validate_chain",
]

CHAIN_COLUMNS = ("symbol", "root", "expiry", "settlement", "type", "strike", "bid", "ask", "vendor_iv")
_TEXT_COLUMNS = ("symbol", "root", "settlement", "type")
_FLOAT_COLUMNS = ("strike", "bid", "ask", "vendor_iv")
#: ROOT (1-6 letters or digits, padded with spaces to 6 in OCC's 21-character form), YYMMDD, C or P, 8 digits.
_OCC = re.compile(r"([A-Z0-9]{1,6}) *(\d{6})([CP])(\d{8})")
_ROOT = re.compile(r"[A-Z0-9]{1,6}")
_STRIKE_PRECISION = 1000  # OCC writes the strike in thousandths of a dollar
_MAX_THOUSANDTHS = 10**8  # eight digits: strikes below 100,000
_FIRST_YEAR, _LAST_YEAR = 2000, 2099  # the two-digit years of an OCC symbol


class ChainSchemaError(ValueError):
    """An option-chain table does not follow the canonical schema (:data:`CHAIN_COLUMNS`)."""


class OccSymbol(NamedTuple):
    """The parts of an OCC option symbol."""

    root: str
    expiry: date
    option_type: Literal["call", "put"]
    strike: float


def occ_symbol(root: str, expiry: date, option_type: str, strike: float) -> str:
    """The OCC option symbol without padding: ``ROOT + YYMMDD + C/P + strike x 1000 in 8 digits``.

    Parameters
    ----------
    root : str
        Option root, 1 to 6 letters or digits.
    expiry : date
        Expiration date, 2000 to 2099.
    option_type : str
        ``"call"`` or ``"put"``.
    strike : float
        Strike in $ per share, a multiple of 0.001 below 100,000.

    Returns
    -------
    str

    Examples
    --------
    >>> occ_symbol("SPXW", date(2026, 11, 2), "call", 7930)
    'SPXW261102C07930000'
    >>> occ_symbol("SPY", date(2026, 12, 18), "put", 612.5)
    'SPY261218P00612500'
    """
    root = str(root).strip().upper()
    if not _ROOT.fullmatch(root):
        raise ValueError(f"an OCC root is 1 to 6 letters or digits, got {root!r}")
    if isinstance(expiry, datetime):
        expiry = expiry.date()
    if not _FIRST_YEAR <= expiry.year <= _LAST_YEAR:
        raise ValueError(f"OCC symbols write the year in two digits (2000-2099), got {expiry}")
    side = {"call": "C", "put": "P"}.get(str(option_type).strip().lower())
    if side is None:
        raise ValueError(f"option type must be 'call' or 'put', got {option_type!r}")
    thousandths = round(float(strike) * _STRIKE_PRECISION)
    if not (0 < thousandths < _MAX_THOUSANDTHS and abs(thousandths - float(strike) * _STRIKE_PRECISION) < 1e-6):
        raise ValueError(f"an OCC strike is a positive multiple of 0.001 below 100,000, got {strike!r}")
    return f"{root}{expiry:%y%m%d}{side}{thousandths:08d}"


def parse_occ_symbol(symbol: str) -> OccSymbol:
    """Split an OCC option symbol into root, expiry, type and strike.

    Accepts the compact form (``SPXW261102C07930000``) and OCC's 21-character form with the root padded by spaces
    (``SPXW  261102C07930000``), in either case.

    Raises
    ------
    ValueError
        If ``symbol`` is not an OCC option symbol.

    Examples
    --------
    >>> parse_occ_symbol("SPXW261102C07930000")
    OccSymbol(root='SPXW', expiry=datetime.date(2026, 11, 2), option_type='call', strike=7930.0)
    >>> parse_occ_symbol("SPY   261218P00612500").strike
    612.5
    """
    match = _OCC.fullmatch(str(symbol).strip().upper())
    if match is None:
        raise ValueError(f"not an OCC option symbol (ROOT + YYMMDD + C/P + 8 digits): {symbol!r}")
    root, yymmdd, side, strike = match.groups()
    try:
        expiry = date(2000 + int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:]))
    except ValueError:
        raise ValueError(f"the OCC symbol {symbol!r} has no valid date in {yymmdd!r}") from None
    return OccSymbol(root, expiry, "call" if side == "C" else "put", int(strike) / _STRIKE_PRECISION)


def fits_occ(root: pd.Series, expiry: pd.Series, strike: pd.Series) -> pd.Series:
    """Which contracts an OCC symbol can carry: a root of 1 to 6 capital letters or digits, an expiry in 2000-2099
    and a strike that is a positive multiple of 0.001 below 100,000 (False where a value is missing).

    Examples
    --------
    >>> fits_occ(pd.Series(["SPXW", "^SPX", "SPY"]), pd.to_datetime(pd.Series(["2026-11-02"] * 3)),
    ...          pd.Series([7930.0, 7930.0, 100.0004])).tolist()
    [True, False, False]
    """
    roots = root.map(lambda value: isinstance(value, str) and _ROOT.fullmatch(value) is not None).astype(bool)
    years = pd.to_datetime(expiry).dt.year
    thousandths = pd.to_numeric(strike, errors="coerce").astype(float) * _STRIKE_PRECISION
    on_grid = (thousandths - thousandths.round()).abs() < 1e-6
    strikes = on_grid & (thousandths.round() > 0) & (thousandths.round() < _MAX_THOUSANDTHS)
    return roots & years.between(_FIRST_YEAR, _LAST_YEAR).fillna(False).astype(bool) & strikes.fillna(False)


def chain_frame(columns: dict[str, Any]) -> pd.DataFrame:
    """A DataFrame with the canonical columns in order and types.

    Text as Python objects (also under pandas 3, whose default string type would otherwise differ between sources),
    expiries as ``datetime64[ns]`` and the numbers as floats. The values are not checked (see :func:`validate_chain`).

    Parameters
    ----------
    columns : dict of str to array-like
        A value or sequence for each column of :data:`CHAIN_COLUMNS`; ``vendor_iv`` may be left out (all NaN).
        Other keys are ignored.

    Returns
    -------
    pandas.DataFrame
        The columns of :data:`CHAIN_COLUMNS`, in that order.

    Examples
    --------
    >>> frame = chain_frame({"symbol": ["SPXW261102C07930000"], "root": ["SPXW"], "expiry": ["2026-11-02"],
    ...                      "settlement": ["PM"], "type": ["call"], "strike": [7930], "bid": [12.3], "ask": [12.9]})
    >>> frame.dtypes.astype(str).to_dict()  # doctest: +NORMALIZE_WHITESPACE
    {'symbol': 'object', 'root': 'object', 'expiry': 'datetime64[ns]', 'settlement': 'object', 'type': 'object',
     'strike': 'float64', 'bid': 'float64', 'ask': 'float64', 'vendor_iv': 'float64'}
    >>> validate_chain(frame)
    """
    data = {"vendor_iv": np.nan, **columns}
    frame = pd.DataFrame({column: data[column] for column in CHAIN_COLUMNS})
    return frame.astype(
        {
            **dict.fromkeys(_TEXT_COLUMNS, object),
            "expiry": "datetime64[ns]",
            **dict.fromkeys(_FLOAT_COLUMNS, float),
        }
    )


def _symbol_mismatches(chain: pd.DataFrame) -> int:
    """Rows whose symbol does not parse or disagrees with their root, expiry, type or strike."""
    bad = 0
    for symbol, root, expiry, kind, strike in chain[["symbol", "root", "expiry", "type", "strike"]].itertuples(
        index=False
    ):
        try:
            parsed = parse_occ_symbol(symbol)
        except ValueError:
            bad += 1
            continue
        same_strike = abs(parsed.strike - strike) < 0.5 / _STRIKE_PRECISION
        if (parsed.root, parsed.expiry, parsed.option_type) != (root, expiry.date(), kind) or not same_strike:
            bad += 1
    return bad


def validate_chain(chain: pd.DataFrame) -> None:
    """Check that ``chain`` follows the canonical schema; raise :class:`ChainSchemaError` listing every problem.

    Checks the columns (:data:`CHAIN_COLUMNS`, extra columns allowed) and their types, then the values: OCC symbols
    that agree with root, expiry, type and strike and appear once each; expiries that are dates; settlement AM or
    PM; type call or put; finite positive strikes on the 0.001 grid; finite bids of at least 0 and finite asks
    above 0 with the bid at most the ask; vendor implied vols NaN or finite and at least 0.

    Examples
    --------
    >>> chain = pd.DataFrame({"symbol": ["SPXW261102C07930000"], "root": ["SPXW"],
    ...                       "expiry": pd.to_datetime(["2026-11-02"]), "settlement": ["PM"], "type": ["call"],
    ...                       "strike": [7930.0], "bid": [12.3], "ask": [12.9], "vendor_iv": [np.nan]})
    >>> validate_chain(chain)
    >>> validate_chain(chain.assign(bid=13.0))
    Traceback (most recent call last):
    ...
    options_analyzer.market_data.schema.ChainSchemaError: bid: 1 row(s) with a bid above the ask
    """
    missing = [column for column in CHAIN_COLUMNS if column not in chain.columns]
    if missing:
        raise ChainSchemaError(f"missing column(s): {', '.join(missing)}")
    problems: list[str] = []
    for column in _TEXT_COLUMNS:
        values = chain[column]
        if not (pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values)):
            problems.append(f"{column}: expected text, got {values.dtype}")
        elif not values.map(lambda value: isinstance(value, str)).all():
            problems.append(f"{column}: every value must be text")
    for column in _FLOAT_COLUMNS:
        if not pd.api.types.is_float_dtype(chain[column]):
            problems.append(f"{column}: expected floats, got {chain[column].dtype}")
    if not pd.api.types.is_datetime64_dtype(chain["expiry"]) or getattr(chain["expiry"].dtype, "tz", None):
        problems.append(f"expiry: expected dates without a time zone (datetime64), got {chain['expiry'].dtype}")
    if problems:
        raise ChainSchemaError("; ".join(problems))

    def count(column: str, bad: pd.Series | np.ndarray, what: str) -> None:
        n_bad = int(np.sum(bad))
        if n_bad:
            problems.append(f"{column}: {n_bad} row(s) {what}")

    strike, bid, ask, vendor_iv = (chain[column].to_numpy(float) for column in _FLOAT_COLUMNS)
    expiry = chain["expiry"]
    count("expiry", expiry.isna() | (expiry != expiry.dt.normalize()), "that are not a date at midnight")
    count("settlement", ~chain["settlement"].isin(["AM", "PM"]), "other than 'AM' or 'PM'")
    count("type", ~chain["type"].isin(["call", "put"]), "other than 'call' or 'put'")
    count("root", ~chain["root"].str.fullmatch(r"[A-Z0-9]{1,6}").fillna(False).astype(bool), "not an OCC root")
    on_grid = np.abs(strike * _STRIKE_PRECISION - np.round(strike * _STRIKE_PRECISION)) < 1e-6
    count("strike", ~(np.isfinite(strike) & (strike > 0) & on_grid), "not a positive multiple of 0.001")
    count("bid", ~(np.isfinite(bid) & (bid >= 0)), "not a finite number of at least 0")
    count("ask", ~(np.isfinite(ask) & (ask > 0)), "not a finite number above 0")
    count("bid", bid > ask, "with a bid above the ask")
    count("vendor_iv", ~(np.isnan(vendor_iv) | (np.isfinite(vendor_iv) & (vendor_iv >= 0))), "negative or infinite")
    if not problems:  # the symbol check needs valid dates, types and strikes
        mismatched = _symbol_mismatches(chain)
        if mismatched:
            problems.append(f"symbol: {mismatched} row(s) that do not parse or disagree with the row")
        count("symbol", chain["symbol"].duplicated(), "repeated")
    if problems:
        raise ChainSchemaError("; ".join(problems))
