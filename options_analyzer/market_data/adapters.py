"""Adapters for option-chain files that users download themselves.

:func:`read_chain` reads three layouts into the canonical chain table (:mod:`.schema`), so that no more header guessing
is added to :func:`options_analyzer.vol_surface.parse_quotes`:

``"long"``
    The app's own layout, one row per contract: ``expiry``, ``type``, ``strike``, ``bid``, ``ask``, and ``root`` or
    ``symbol`` (or both), with ``settlement`` and ``vendor_iv`` optional (column names in any case).
``"yfinance"``
    What yfinance's ``Ticker.option_chain(date)`` returns for calls and puts, stacked over expiries and saved as CSV:
    ``contractSymbol``, ``bid``, ``ask`` and ``impliedVolatility`` are read, and ``strike`` and an ``expiry`` column,
    where there are, are checked against the symbol; the other columns are ignored.
``"cboe"``
    The CSV that Cboe's delayed-quotes page downloads by hand: lines about the underlying, then a header row
    ``Expiration Date, Calls, Last Sale, Net, Bid, Ask, ..., IV, ..., Strike, Puts, Last Sale, Net, Bid, Ask, ...``
    and one row per expiry and strike, the call left of the strike and the put right of it.

Lines that start with ``#`` before the table are skipped (the provenance header of the test fixtures).

Every layout identifies a contract by its OCC symbol (``ROOT + YYMMDD + C/P + strike x 1000``), or builds the symbol
from root, expiry, type and strike. The settlement style comes from the root
(:func:`options_analyzer.clock.settlement_style`) unless the long layout states it, and :func:`to_quotes` takes each
time to expiry from :mod:`options_analyzer.clock`. Contracts that cannot be read, whose symbol disagrees with their
other columns, that have settled, that have no offer, whose bid is above the ask, or whose bid is zero are dropped,
with one message per reason.

The vendor's implied vol is kept only as the ``vendor_iv`` diagnostic. Vendors compute it with their own rates,
dividends and models, so two vendors disagree on the same quote, and some write a placeholder (such as 1e-05) where
they have none; values below :data:`MIN_VENDOR_IV` are blanked. :func:`to_quotes` leaves it out, so it never reaches
a fit.
"""

from __future__ import annotations

import csv
import io
import os
import re
from collections.abc import Callable
from datetime import date, datetime
from typing import IO, Literal

import numpy as np
import pandas as pd
from dateutil import parser as dateutil_parser

from ..clock import settlement_style, time_to_expiry, to_new_york
from .schema import CHAIN_COLUMNS, chain_frame, occ_symbol, parse_occ_symbol, validate_chain

__all__ = ["LAYOUTS", "MIN_VENDOR_IV", "clean_chain", "read_chain", "to_quotes"]

LAYOUTS = ("long", "yfinance", "cboe")
#: Vendor implied vols below 1% are placeholders.
MIN_VENDOR_IV = 0.01

Layout = Literal["auto", "long", "yfinance", "cboe"]
FileLike = str | os.PathLike[str] | bytes | IO[str] | IO[bytes]
_NAMES = {"long": "the app's long layout", "yfinance": "a yfinance-style export", "cboe": "Cboe's download layout"}


# ---------------------------------------------------------------------------
# Reading the file
# ---------------------------------------------------------------------------


def _text(source: FileLike) -> str:
    """The file's text: a path, the bytes of a file, or an open file (an upload is one)."""
    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as handle:
            data: str | bytes = handle.read()
    elif isinstance(source, bytes):
        data = source
    elif hasattr(source, "read"):
        data = source.read()
    else:
        raise TypeError(f"expected a path, bytes, an open file or a DataFrame, got {type(source).__name__}")
    return data.decode("utf-8-sig") if isinstance(data, bytes) else data.lstrip("﻿")


def _key(name: object) -> str:
    """A column name reduced to lowercase letters and digits, without the ``.1`` pandas adds to a repeated name."""
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\.\d+$", "", str(name).strip()).lower())


def _table(text: str) -> tuple[str, pd.DataFrame]:
    """The layout and the table (all cells as text) of a chain file."""
    lines = text.splitlines()
    start = 0
    while start < len(lines) and (not lines[start].strip() or lines[start].lstrip().startswith("#")):
        start += 1
    for i in range(start, len(lines)):
        header = next(csv.reader([lines[i]]), [])
        if {"calls", "strike", "puts"} <= {_key(cell) for cell in header}:
            # Cboe: lines about the underlying come first, and the header names Bid, Ask and IV twice (calls, puts).
            frame = pd.read_csv(io.StringIO("\n".join(lines[i + 1 :])), header=None, dtype=str)
            frame.columns = [cell.strip() for cell in header][: frame.shape[1]]
            return "cboe", frame
    frame = pd.read_csv(io.StringIO("\n".join(lines[start:])), dtype=str)
    return _layout_of(frame), frame


def _layout_of(frame: pd.DataFrame) -> str:
    keys = {_key(column) for column in frame.columns}
    if "contractsymbol" in keys:
        return "yfinance"
    if {"calls", "strike", "puts"} <= keys:
        return "cboe"
    if {"expiry", "type", "strike", "bid", "ask"} <= keys and keys & {"root", "symbol"}:
        return "long"
    raise ValueError(
        "the file is not in a known chain layout: expected the app's long layout (expiry, type, strike, bid, ask and "
        "root or symbol), a yfinance export (contractSymbol, strike, bid, ask) or Cboe's download (Calls | Strike | "
        f"Puts); found the columns {', '.join(map(str, frame.columns))}"
    )


def _column(frame: pd.DataFrame, *names: str) -> pd.Series | None:
    keys = {_key(column): column for column in frame.columns}
    for name in names:
        if _key(name) in keys:
            return frame[keys[_key(name)]]
    return None


def _numbers(values: pd.Series | None, index: pd.Index) -> pd.Series:
    """Text cells as floats (spaces, $ signs and thousands separators removed); NaN where a cell is no number."""
    if values is None:
        return pd.Series(np.nan, index=index)
    text = values.astype("string").str.replace(r"[,\s$]", "", regex=True)
    return pd.to_numeric(text, errors="coerce").astype(float)


def _dates(values: pd.Series) -> pd.Series:
    """Text cells as dates (``2026-11-02``, ``11/2/2026``, ``Mon Nov 02 2026``); NaT where a cell is no date."""

    def parse(text: object) -> pd.Timestamp | None:
        try:
            return pd.Timestamp(dateutil_parser.parse(str(text)).date())
        except (ValueError, OverflowError):
            return None

    return values.map(parse).astype("datetime64[ns]")


def _flags(values: pd.Series) -> pd.Series:
    return values.fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# Layouts -> contracts
# ---------------------------------------------------------------------------


def _from_symbols(symbols: pd.Series) -> pd.DataFrame:
    """Root, expiry, type and strike parsed from OCC symbols; missing where a symbol does not parse."""
    parts: list[tuple[str | None, pd.Timestamp | None, str | None, float]] = []
    for symbol in symbols:
        try:
            parsed = parse_occ_symbol(symbol)
        except ValueError:
            parts.append((None, None, None, np.nan))
        else:
            parts.append((parsed.root, pd.Timestamp(parsed.expiry), parsed.option_type, parsed.strike))
    frame = pd.DataFrame(parts, columns=["root", "expiry", "type", "strike"], index=symbols.index)
    return frame.astype({"root": object, "expiry": "datetime64[ns]", "type": object, "strike": float})


def _contracts(
    index: pd.Index,
    *,
    symbol: pd.Series | None,
    root: pd.Series | None,
    expiry: pd.Series | None,
    kind: pd.Series | None,
    strike: pd.Series | None,
    bid: pd.Series | None,
    ask: pd.Series | None,
    vendor_iv: pd.Series | None,
    settlement: pd.Series | None = None,
) -> pd.DataFrame:
    """Contracts in the canonical columns (no symbol yet), plus ``_problem``: why a row cannot be used, or ''.

    Root, expiry, type and strike come from the symbol where there is one; where the row also gives one of them and
    it differs from the symbol's, the row is inconsistent (blank cells are not compared).
    """
    strikes = _numbers(strike, index)
    given_type = None
    if kind is not None:
        given_type = kind.astype("string").str.strip().str.lower().str[:1].map({"c": "call", "p": "put"})
    given_root = None if root is None else root.astype("string").str.strip().str.upper()
    given_expiry = None if expiry is None else _dates(expiry)
    disagree = pd.Series(False, index=index)
    if symbol is not None:
        frame = _from_symbols(symbol)
        unreadable = frame["root"].isna()
        if given_root is not None:
            disagree |= _flags(given_root.ne(frame["root"]) & given_root.notna())
        if given_expiry is not None:
            disagree |= _flags(given_expiry.ne(frame["expiry"]) & given_expiry.notna())
        if given_type is not None:
            disagree |= _flags(given_type.ne(frame["type"]) & given_type.notna())
        disagree |= _flags((strikes - frame["strike"]).abs() > 5e-4)
        disagree &= ~unreadable
    else:
        if given_root is None or given_expiry is None or given_type is None:
            raise ValueError("a chain without symbols needs root, expiry, type and strike columns")
        frame = pd.DataFrame(
            {
                "root": given_root.astype(object),
                "expiry": given_expiry,
                "type": given_type.astype(object),
                "strike": strikes,
            },
            index=index,
        )
        unreadable = frame.isna().any(axis=1)
    frame["bid"], frame["ask"], frame["vendor_iv"] = (
        _numbers(bid, index),
        _numbers(ask, index),
        _numbers(vendor_iv, index),
    )
    finite = np.isfinite(frame[["strike", "bid", "ask"]].to_numpy(dtype=float)).all(axis=1)
    unreadable |= pd.Series(~finite, index=index)
    styles = frame["root"].map(settlement_style, na_action="ignore")
    if settlement is not None:
        stated = settlement.astype("string").str.strip().str.upper()
        valid = _flags(stated.isin(["AM", "PM"]))
        unreadable |= _flags(stated.notna() & ~valid)
        styles = stated.where(valid, styles)
    frame["settlement"] = styles.astype(object)
    frame["_problem"] = np.where(unreadable, "unreadable", np.where(disagree, "inconsistent", ""))
    return frame


def _long(frame: pd.DataFrame) -> pd.DataFrame:
    return _contracts(
        frame.index,
        symbol=_column(frame, "symbol"),
        root=_column(frame, "root"),
        expiry=_column(frame, "expiry", "expiration"),
        kind=_column(frame, "type"),
        strike=_column(frame, "strike"),
        bid=_column(frame, "bid"),
        ask=_column(frame, "ask"),
        vendor_iv=_column(frame, "vendor_iv"),
        settlement=_column(frame, "settlement"),
    )


def _yfinance(frame: pd.DataFrame) -> pd.DataFrame:
    return _contracts(
        frame.index,
        symbol=_column(frame, "contractSymbol"),
        root=None,
        expiry=_column(frame, "expiry", "expiration", "expirationDate"),
        kind=None,
        strike=_column(frame, "strike"),
        bid=_column(frame, "bid"),
        ask=_column(frame, "ask"),
        vendor_iv=_column(frame, "impliedVolatility"),
    )


def _cboe(frame: pd.DataFrame) -> pd.DataFrame:
    """Each row holds a call (from Calls to Strike) and a put (from Puts on): one contract per side."""
    names = [_key(column) for column in frame.columns]
    calls, strike, puts = names.index("calls"), names.index("strike"), names.index("puts")
    expiry = frame.iloc[:, names.index("expirationdate")] if "expirationdate" in names else None
    sides = []
    for first, stop in ((calls, strike), (puts, len(names))):
        fields = {name: frame.iloc[:, first + i] for i, name in enumerate(names[first:stop]) if i}
        sides.append(
            _contracts(
                frame.index,
                symbol=frame.iloc[:, first],
                root=None,
                expiry=expiry,
                kind=None,
                strike=frame.iloc[:, strike],
                bid=fields.get("bid"),
                ask=fields.get("ask"),
                vendor_iv=fields.get("iv"),
            )
        )
    return pd.concat(sides, ignore_index=True)


_READERS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {"long": _long, "yfinance": _yfinance, "cboe": _cboe}


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def _years_to_settlement(expiry: pd.Series, settlement: pd.Series, as_of: datetime) -> np.ndarray:
    """Years from ``as_of`` to each contract's settlement (:func:`options_analyzer.clock.time_to_expiry`)."""
    years: dict[tuple[date, str], float] = {}
    out = np.empty(len(expiry))
    for i, (day, style) in enumerate(zip(expiry, settlement, strict=True)):
        key = (day.date(), style)
        if key not in years:
            years[key] = time_to_expiry(as_of, key[0], style=style)
        out[i] = years[key]
    return out


def _finish(frame: pd.DataFrame, as_of: datetime, messages: list[str]) -> pd.DataFrame:
    """Drop the rows that cannot be quoted (one message per reason), and return the canonical chain, sorted."""
    as_of = to_new_york(as_of)
    problem = frame["_problem"].to_numpy(dtype=object)
    usable = problem == ""
    settled = np.zeros(len(frame), dtype=bool)
    settled[usable] = _years_to_settlement(frame["expiry"][usable], frame["settlement"][usable], as_of) <= 0
    bid, ask = frame["bid"].to_numpy(float), frame["ask"].to_numpy(float)
    for reason, rows in (("expired", settled), ("no offer", ask <= 0), ("crossed", bid > ask), ("zero bid", bid <= 0)):
        problem = np.where((problem == "") & rows, reason, problem)
    texts = {
        "unreadable": "with a missing or unreadable symbol, expiry, type, strike, bid or ask",
        "inconsistent": "whose OCC symbol disagrees with their root, expiry, type or strike",
        "expired": f"that had settled by {as_of:%Y-%m-%d %H:%M} ET (expired)",
        "no offer": "with no offer (an ask of zero)",
        "crossed": "whose bid is above the ask (crossed)",
        "zero bid": "with a zero bid (offer only)",
    }
    for reason, text in texts.items():
        count = int(np.sum(problem == reason))
        if count:
            messages.append(f"Dropped {count} contract(s) {text}.")
    chain = frame[problem == ""].drop(columns="_problem")
    repeated = chain.duplicated(["root", "expiry", "type", "strike"])
    if repeated.any():
        messages.append(f"Dropped {int(repeated.sum())} repeated contract(s), keeping the first quote of each.")
        chain = chain[~repeated]
    placeholder = chain["vendor_iv"] < MIN_VENDOR_IV
    if placeholder.any():
        messages.append(
            f"Blanked {int(placeholder.sum())} vendor implied vol(s) below {MIN_VENDOR_IV:.0%}, the placeholders some "
            "vendors write where they have none; vendor implied vols are a diagnostic only and are never fitted."
        )
        chain = chain.assign(vendor_iv=chain["vendor_iv"].where(~placeholder))
    chain = chain.assign(_kind=chain["type"].map({"call": 0, "put": 1}))
    chain = chain.sort_values(["expiry", "root", "strike", "_kind"], kind="stable")
    symbols = [
        occ_symbol(root, expiry, kind, strike)
        for root, expiry, kind, strike in chain[["root", "expiry", "type", "strike"]].itertuples(index=False)
    ]
    chain = chain_frame({"symbol": symbols, **{column: chain[column].to_numpy() for column in CHAIN_COLUMNS[1:]}})
    validate_chain(chain)
    return chain


def clean_chain(chain: pd.DataFrame, *, as_of: datetime) -> tuple[pd.DataFrame, list[str]]:
    """Drop the contracts of a canonical chain that cannot be quoted at ``as_of``, with one message per reason.

    As :func:`read_chain` does for a file: contracts that have settled by ``as_of``
    (:func:`options_analyzer.clock.settlement_time`), that have no offer, whose bid is above the ask, or whose bid is
    zero are dropped, repeated contracts keep their first quote, and vendor implied vols below
    :data:`MIN_VENDOR_IV` are blanked. Symbols are rebuilt from root, expiry, type and strike.

    Parameters
    ----------
    chain : pandas.DataFrame
        A chain with the canonical columns (:data:`options_analyzer.market_data.schema.CHAIN_COLUMNS`).
    as_of : datetime
        Valuation time (a naive datetime is New York wall-clock time).

    Returns
    -------
    (pandas.DataFrame, list of str)
        The cleaned chain, sorted by expiry, root, strike and type, and the messages.

    Examples
    --------
    >>> from options_analyzer.market_data.synthetic import sample_chain
    >>> chain = sample_chain(6000.0, 0.04, 0.013, as_of=datetime(2026, 10, 26, 10, 0)).quotes
    >>> cleaned, messages = clean_chain(chain, as_of=datetime(2026, 11, 2, 10, 0))
    >>> for message in messages:
    ...     print(message)
    Dropped 184 contract(s) that had settled by 2026-11-02 10:00 ET (expired).
    Dropped 155 contract(s) with a zero bid (offer only).
    """
    missing = [column for column in CHAIN_COLUMNS if column not in chain.columns]
    if missing:
        raise ValueError(f"the chain lacks the column(s) {', '.join(missing)}")
    frame = chain[list(CHAIN_COLUMNS[1:])].assign(_problem="")
    messages: list[str] = []
    return _finish(frame, as_of, messages), messages


def read_chain(
    source: FileLike | pd.DataFrame, *, as_of: datetime, layout: Layout = "auto"
) -> tuple[pd.DataFrame, list[str]]:
    """Read a chain file in the long, yfinance or Cboe layout into a cleaned canonical chain.

    Parameters
    ----------
    source : path, bytes, open file or pandas.DataFrame
        The file (a path, its bytes, or an open file such as an upload), or a table already read (a DataFrame, such
        as yfinance returns).
    as_of : datetime
        Valuation time: contracts that have settled by then are dropped (a naive datetime is New York time).
    layout : {"auto", "long", "yfinance", "cboe"}, default "auto"
        The layout; ``"auto"`` recognises it from the header.

    Returns
    -------
    (pandas.DataFrame, list of str)
        The chain in the canonical schema (:mod:`options_analyzer.market_data.schema`), sorted by expiry, root, strike
        and type, and messages that name the layout and every reason a contract was dropped.

    Raises
    ------
    ValueError
        If the file is in no known layout, or not in the layout asked for.

    Examples
    --------
    >>> text = '''expiry,root,type,strike,bid,ask
    ... 2026-11-02,SPXW,call,7930,12.3,12.9
    ... 2026-11-02,SPXW,put,7930,11.8,12.4
    ... 2026-10-23,SPXW,call,7930,0.0,0.05
    ... '''
    >>> chain, messages = read_chain(text.encode(), as_of=datetime(2026, 10, 26, 10, 0))
    >>> chain["symbol"].tolist()
    ['SPXW261102C07930000', 'SPXW261102P07930000']
    >>> for message in messages:
    ...     print(message)
    Read 3 contract(s) in the app's long layout.
    Dropped 1 contract(s) that had settled by 2026-10-26 10:00 ET (expired).
    """
    if layout != "auto" and layout not in LAYOUTS:
        raise ValueError(f"layout must be 'auto' or one of {', '.join(LAYOUTS)}, got {layout!r}")
    if isinstance(source, pd.DataFrame):
        found, frame = _layout_of(source), source.astype("string").astype(object)
    else:
        found, frame = _table(_text(source))
    if layout != "auto" and found != layout:
        raise ValueError(f"the file looks like {_NAMES[found]}, not the {layout} layout")
    contracts = _READERS[found](frame.reset_index(drop=True))
    messages = [f"Read {len(contracts)} contract(s) in {_NAMES[found]}."]
    return _finish(contracts, as_of, messages), messages


def to_quotes(chain: pd.DataFrame, *, as_of: datetime) -> pd.DataFrame:
    """The table :func:`options_analyzer.vol_surface.parse_quotes` reads, from a canonical chain.

    Columns ``Strike``, ``Maturity (years)`` (from :func:`options_analyzer.clock.time_to_expiry` with each contract's
    settlement style), ``Type``, ``Bid`` and ``Ask``. ``vendor_iv`` is left out, so it never reaches a fit.

    Raises
    ------
    ValueError
        If a contract has settled by ``as_of`` (clean the chain first with :func:`read_chain` or
        :func:`clean_chain`).

    Examples
    --------
    >>> chain, _ = read_chain(b"symbol,expiry,type,strike,bid,ask\\nSPXW261102C07930000,2026-11-02,C,7930,12.3,12.9",
    ...                       as_of=datetime(2026, 11, 2, 10, 0))
    >>> quotes = to_quotes(chain, as_of=datetime(2026, 11, 2, 10, 0))
    >>> quotes.columns.tolist()
    ['Strike', 'Maturity (years)', 'Type', 'Bid', 'Ask']
    >>> float(quotes.loc[0, "Maturity (years)"]) * 525_600  # six hours to the 16:00 settlement
    360.0
    """
    maturity = _years_to_settlement(chain["expiry"], chain["settlement"], as_of)
    settled = int(np.sum(~(maturity > 0)))
    if settled:
        when = f"{to_new_york(as_of):%Y-%m-%d %H:%M} ET"
        raise ValueError(f"{settled} contract(s) had settled by {when}; clean the chain with clean_chain first")
    return pd.DataFrame(
        {
            "Strike": chain["strike"].to_numpy(float),
            "Maturity (years)": maturity,
            "Type": chain["type"].to_numpy(dtype=object),
            "Bid": chain["bid"].to_numpy(float),
            "Ask": chain["ask"].to_numpy(float),
        }
    )
