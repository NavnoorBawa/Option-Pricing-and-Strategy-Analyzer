"""Reading a table of option quotes into implied vols, and model-free checks of the quotes for static arbitrage.

:func:`parse_quotes` runs in stages: :func:`_map_columns` (which column holds what, from its header),
:func:`_normalise_units` (each column as numbers in its stated or guessed unit), :func:`_valid_rows` (rows with a
missing or non-positive value dropped), :func:`_invert` (implied vols as decimals, inverting option prices with
Black-Scholes-Merton) and :func:`_filter` (implausible vols, expiries within half a day merged, duplicates averaged).
The time to expiry (:func:`_maturities`) is read inside :func:`_normalise_units`, after the strike and before the
vols or prices, so that the messages and the first refusal come in the same order as before the stages were split.
:func:`_spread_arbitrage` and :func:`_butterfly_arbitrage` check the quotes themselves for static arbitrage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import cast

import numpy as np
import pandas as pd
from dateutil import parser as dateutil_parser
from scipy.special import ndtr

from ..implied_vol import ImpliedVolError, implied_volatility


class QuoteParseError(ValueError):
    """The uploaded quotes cannot be turned into a surface."""


class QuoteWarning(str):
    """A parse message about a unit that had to be guessed: the reading may be wrong, so show it prominently."""


# ---------------------------------------------------------------------------
# Columns and units
# ---------------------------------------------------------------------------


_UNITS = {
    "%": "percent",
    "pct": "percent",
    "percent": "percent",
    "d": "days",
    "day": "days",
    "days": "days",
    "y": "years",
    "yr": "years",
    "yrs": "years",
    "year": "years",
    "years": "years",
    **dict.fromkeys(("mo", "mos", "mon", "mth", "mths", "month", "months"), "months"),
    **dict.fromkeys(("w", "wk", "wks", "week", "weeks"), "weeks"),
    **dict.fromkeys(("h", "hr", "hrs", "hour", "hours"), "hours"),
    **dict.fromkeys(("min", "mins", "minute", "minutes"), "minutes"),
    "m": "m",  # months in tenor notation ("3M"), minutes elsewhere: refused as a unit of time to expiry
}
_UNIT_SUFFIXES = (
    ("percent", "percent"),
    ("pct", "percent"),
    ("years", "years"),
    ("yrs", "years"),
    ("days", "days"),
    ("months", "months"),
    ("weeks", "weeks"),
    ("hours", "hours"),
    ("minutes", "minutes"),
)
_BRACKETS = re.compile(r"[(\[{]([^)\]}]*)[)\]}]")
# Units of a time to expiry other than days and years: days per unit, and how the conversion is described.
_TIME_UNITS = {
    "months": (365.0 / 12.0, "months, divided by 12"),
    "weeks": (7.0, "weeks of 7 days, converted to years with ACT/365"),
    "hours": (1.0 / 24.0, "hours, converted to years with ACT/365"),
    "minutes": (1.0 / 1440.0, "minutes, converted to years with ACT/365"),
}


def _header(name) -> tuple[str, str | None]:
    """A column name reduced to lowercase letters and digits, plus the unit it states.

    The unit is percent, days, years, months, weeks, hours or minutes, or ``"m"``, which could be months or
    minutes. Bracketed text next to a name is removed but read for a unit (``"IV (% p.a.)"`` ->
    ``("iv", "percent")``, ``"Maturity (months)"`` -> ``("maturity", "months")``), as are a ``%`` sign and a
    trailing unit word (``"Maturity_years"`` -> ``("maturity", "years")``). A name wrapped entirely in brackets
    keeps its text (``"[STRIKE]"`` -> ``"strike"``).
    """
    text = str(name).strip().lower()
    unit = "percent" if "%" in text else None
    outside = _BRACKETS.sub(" ", text)
    if re.search(r"[a-z0-9]", outside):
        for inner in _BRACKETS.findall(text):
            for word in re.findall(r"[a-z]+", inner):
                unit = unit or _UNITS.get(word)
        text = outside
    base = re.sub(r"[^a-z0-9]", "", text)
    for suffix, suffix_unit in _UNIT_SUFFIXES:
        if base.endswith(suffix) and len(base) > len(suffix):
            base, unit = base[: -len(suffix)], unit or suffix_unit
            break
    if base.startswith("year"):  # "Years", "Years to expiry"
        unit = unit or "years"
    return base, unit


# Most specific first: "Implied Volatility" must win over a bare "Vol", which is often trading volume.
_ALIASES = {
    "strike": ("Strike", "Strike price", "K"),
    "iv": ("Implied volatility", "Implied vol", "Impl vol", "Imp vol", "Mid IV", "IV", "Volatility", "Vol", "Sigma"),
    "days": ("Days", "DTE", "Days to expiry", "Days to expiration", "Days to maturity", "Calendar days"),
    "maturity": (
        "Maturity",
        "Time to maturity",
        "Time to expiry",
        "Time to expiration",
        "Years to expiry",
        "Years to maturity",
        "Expiry",
        "Expiration",
        "Tenor",
        "TTM",
        "Years",
        "T",
    ),
    "type": ("Type", "Option type", "Call/Put", "Put/Call", "CP", "Right"),
    "price": ("Mid", "Mid price", "Price", "Option price", "Mark", "Premium"),
    "bid": ("Bid", "Bid price"),
    "ask": ("Ask", "Ask price", "Offer"),
    "last": ("Last", "Last price"),
}
_ALIAS_KEYS = {key: tuple(_header(alias)[0] for alias in names) for key, names in _ALIASES.items()}
_WEAK_IV = _header("Vol")[0]

MAX_ROWS = 200_000
MAX_PRICE_QUOTES = 10_000
MIN_QUOTED_VOL = 1e-4  # implied vols below 0.01% are placeholders (some data feeds write 1e-05 for none)


def _holds_times(frame: pd.DataFrame, column, unit: str | None) -> bool:
    """Whether a column has numbers that can be a time to expiry (not dates, not YYYYMMDD integers)."""
    days_per_unit = _TIME_UNITS[unit][0] if unit in _TIME_UNITS else 1.0
    try:
        return bool(_numeric(frame, column)[0].max() * days_per_unit <= MAX_DAYS)
    except QuoteParseError:
        return False


def _match_columns(frame: pd.DataFrame) -> dict[str, tuple[str, str | None]]:
    headers = {column: _header(column) for column in frame.columns}
    found: dict[str, tuple[str, str | None]] = {}
    claimed: set = set()
    for key, aliases in _ALIAS_KEYS.items():
        candidates = [c for alias in aliases for c, (base, _) in headers.items() if base == alias and c not in claimed]
        if not candidates:
            continue
        column = candidates[0]
        if key == "maturity":  # an expiry-date column often sits next to the time to expiry ("Expiration", "TTM")
            column = next((c for c in candidates if _holds_times(frame, c, headers[c][1])), column)
        found[key] = (column, headers[column][1])
        claimed.add(column)
    if "maturity" in found and found["maturity"][1] == "days":  # e.g. "Maturity (days)"
        days = found.pop("maturity")
        found.setdefault("days", days)
    return found


_THOUSANDS = r"[-+]?(?:[1-9]\d{0,2}(?:,\d{3})+|[1-9]\d?(?:,\d{2})+,\d{3})(?:\.\d*)?"  # 1,250,000.5 or 12,50,000.5
_DECIMAL_COMMA = r"[-+]?\d*,\d+"  # 25,3 or 0,0822


def _numeric(
    frame: pd.DataFrame, column, messages: list[str] | None = None, required: bool = True
) -> tuple[pd.Series, bool]:
    """A column as floats, with %, thousands separators, currency signs and spaces stripped.

    Commas are thousands separators when every number written with one is grouped that way (``1,250.50``), and
    decimal commas when every one has a single comma and no number in the column has a decimal point
    (``25,3``), which is then said in ``messages``; any other use of commas is refused. Also returns whether
    any value carried a ``%`` sign. A column with no numeric value at all raises :class:`QuoteParseError`
    naming it, or comes back as all NaN when it is not ``required``.
    """
    raw = frame[column]
    percent = False
    if pd.api.types.is_numeric_dtype(raw) and not pd.api.types.is_bool_dtype(raw):
        values = raw.astype(float)
    else:
        text = raw.astype("string").str.strip()
        percent = bool(text.str.endswith("%").fillna(False).any())
        text = text.str.replace(r"[\s$€£%]", "", regex=True)
        numbers = text.str.contains(r"\d", regex=True).fillna(False)
        commas = text[numbers & text.str.contains(",", regex=False).fillna(False)]
        points = bool((numbers & text.str.contains(".", regex=False).fillna(False)).any())
        if commas.str.fullmatch(_THOUSANDS).all():
            text = text.str.replace(",", "", regex=False)
        elif commas.str.fullmatch(_DECIMAL_COMMA).all() and not points:
            text = text.str.replace(",", ".", regex=False)
            if messages is not None:
                messages.append(f"Read the commas in '{column}' as decimal points (e.g. '{commas.iloc[0]}').")
        else:
            raise QuoteParseError(
                f"column '{column}' has commas that are neither thousands separators nor decimal commas "
                f"(e.g. '{commas.iloc[0]}'); write its numbers with a decimal point and no other separator"
            )
        values = pd.to_numeric(text, errors="coerce").astype(float)
    if required and not values.notna().any():
        sample = raw.dropna()
        example = f" (e.g. '{sample.iloc[0]}')" if len(sample) else ""
        raise QuoteParseError(f"column '{column}' has no numeric values{example}")
    return values, percent


# ---------------------------------------------------------------------------
# Time to expiry
# ---------------------------------------------------------------------------


MAX_DAYS = 36_500  # 100 years: larger "maturities" are neither years nor days (e.g. YYYYMMDD dates)
_TENOR = r"\d+(?:\.\d+)?\s*[dwmy]"  # "1M", "3m", "1W", "2Y", "30D"
_TIME_OF_DAY = r"\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:[ap]\.?m\.?)?"  # "16:00", "9:30 am": pandas dates them today


def _looks_like_dates(sample: pd.Series) -> bool:
    """Whether most of a column's text values are calendar dates, with or without a year.

    A date without a year ("Nov 20", "11/20") gets the year 1, and a time alone the year 1 and 1 January ("5h",
    "5pm"), so those times are not counted; nor are times of day ("16:00").
    """
    dates = [_parse_date(text) for text in sample]
    dated = np.array(
        [d is not None and (d.year >= 1900 or (d.year == 1 and (d.month, d.day) != (1, 1))) for d in dates], dtype=bool
    )
    dated &= ~sample.str.lower().str.fullmatch(_TIME_OF_DAY).to_numpy(dtype=bool)
    return bool(len(sample)) and bool(dated.mean() > 0.5)


def _parse_date(text: str) -> datetime | None:
    """``text`` as a date with the year 1 when it has none, or None. Parsed with dateutil (a pandas dependency)
    rather than pandas, which returns NaT for the year 1 before pandas 3."""
    try:
        return dateutil_parser.parse(text, default=datetime(1, 1, 1))
    except (ValueError, OverflowError):
        return None


def _time_column(
    frame: pd.DataFrame, column, messages: list[str] | None = None, days_per_unit: float = 1.0
) -> pd.Series:
    """A maturity or days column as floats; dates, tenors and impossible values raise :class:`QuoteParseError`.

    ``days_per_unit`` is the column's unit in days when its header states one (7 for weeks), so values worth
    more than 100 years are refused in any unit.
    """
    try:
        values = _numeric(frame, column, messages)[0]
    except QuoteParseError:
        sample = frame[column].dropna().astype(str).str.strip().head(50)
        if len(sample) and sample.str.lower().str.fullmatch(_TENOR).mean() > 0.5:
            raise QuoteParseError(
                f"column '{column}' holds tenors (e.g. '{sample.iloc[0]}'), but the surface needs the time to expiry "
                "as a number: add a Days column (calendar days to expiry) or a Maturity column in years"
            ) from None
        if _looks_like_dates(sample):
            raise QuoteParseError(
                f"column '{column}' holds dates (e.g. '{sample.iloc[0]}'), but the surface needs the time to expiry: "
                "add a Days column (calendar days to expiry) or a Maturity column in years"
            ) from None
        raise
    largest = values.max()
    if largest * days_per_unit > MAX_DAYS:
        hint = " (8-digit values look like YYYYMMDD dates)" if 19_000_101 <= largest <= 29_991_231 else ""
        what = "neither years nor days" if days_per_unit == 1.0 else "more than 100 years"
        raise QuoteParseError(
            f"column '{column}' goes up to {largest:,.0f}, which is {what}{hint}; "
            "add a Days column (calendar days to expiry) or a Maturity column in years"
        )
    return values


MAX_USUAL_YEARS = 3.0  # listed options rarely run longer: unlabelled maturities beyond this may well be days
_NAME_THE_UNIT = "name the column 'Days' if they are days to expiry, or 'Maturity (years)' if they are years"


@dataclass(frozen=True)
class _GuessedYears:
    """A maturity column read as years although its header states no unit."""

    column: str
    described: str  # what its values look like: "go up to 9.65", "are whole numbers up to 4"
    suspect: bool  # whether the values alone are reason to doubt years


def _maturities(
    frame: pd.DataFrame, found: dict, used: list[str], messages: list[str]
) -> tuple[pd.Series, _GuessedYears | None]:
    """Time to expiry in years, from a days column if there is one, otherwise from a maturity column.

    A unit of months, weeks, hours or minutes stated in the header is converted; ``"m"`` alone (months or
    minutes) is refused. A maturity column whose header does not say years is converted from days when its
    values go above 30 (no listed option runs that many years). Up to 30, whole numbers from 5 up are as likely
    to be years (a 1, 2, 3, 5, 7, 10-year tenor grid) as days (a weekly chain at 7, 14, 21, 28), so the file is
    refused with a request to name the unit; anything smaller is read as years. Returns ``(years, guess)``:
    ``guess`` is a :class:`_GuessedYears` when years was only assumed (suspect for small whole numbers and
    values above :data:`MAX_USUAL_YEARS`), else None; :func:`parse_quotes` warns about the suspect ones.
    """
    column, unit = found["days"] if "days" in found else found["maturity"]
    if unit == "m":
        raise QuoteParseError(
            f"the unit 'm' in '{column}' could mean months or minutes; name the column 'Maturity (months)', "
            "'Days' or 'Maturity (years)'"
        )
    if "days" in found and "maturity" in found:
        messages.append(f"Ignored '{found['maturity'][0]}': the time to expiry comes from '{column}'.")
    if unit in _TIME_UNITS:
        days_per_unit, described = _TIME_UNITS[unit]
        used.append(f"maturity from '{column}' ({described})")
        return _time_column(frame, column, messages, days_per_unit) * (days_per_unit / 365.0), None
    if "days" in found:
        used.append(f"maturity from '{column}' (days, converted to years with ACT/365)")
        return _time_column(frame, column, messages) / 365.0, None
    values = _time_column(frame, column, messages)
    positive = values[values > 0]
    largest = values.max()
    whole = bool(len(positive)) and bool(np.all(positive == positive.round()))
    if largest > 30:
        messages.append(
            f"Maturities in '{column}' go up to {largest:g}, which only makes sense in days; converted days to "
            "years (ACT/365). Name the column 'Days' to make this explicit."
        )
        used.append(f"maturity from '{column}' (days)")
        return values / 365.0, None
    used.append(f"maturity from '{column}' (years)")
    if unit == "years":
        return values, None
    if whole and largest >= 5:
        raise QuoteParseError(
            f"the maturities in '{column}' are whole numbers from {positive.min():g} to {largest:g}, which "
            f"could be years or days; {_NAME_THE_UNIT}"
        )
    if whole:
        return values, _GuessedYears(column, f"are whole numbers up to {largest:g}", True)
    return values, _GuessedYears(column, f"go up to {largest:g}", largest > MAX_USUAL_YEARS)


MATURITY_TOLERANCE = 0.5 / 365  # maturities this close (half a day, in years) are the same expiry


def _expiries(maturity) -> np.ndarray:
    """Expiry labels for times to expiry in years: each expiry spans at most :data:`MATURITY_TOLERANCE`.

    A label starts at the shortest time not yet labelled and takes every time up to half a day longer, so one
    expiry quoted at different times of a session (fractional days to expiry) gets one label, while listed
    expiries, at least a day apart, keep their own; times spread over more than half a day are never chained
    into one expiry.
    """
    maturity = np.asarray(maturity, dtype=float)
    unique = np.unique(maturity)
    labels = np.zeros(len(unique), dtype=int)
    start = unique[0] if len(unique) else 0.0
    for i in range(1, len(unique)):
        if unique[i] - start > MATURITY_TOLERANCE:
            start = unique[i]
            labels[i] = labels[i - 1] + 1
        else:
            labels[i] = labels[i - 1]
    return labels[np.searchsorted(unique, maturity)]


def _merge_close_maturities(data: pd.DataFrame, messages: list[str]) -> pd.DataFrame:
    """Set the times to expiry of each expiry (:func:`_expiries`) to their median, keeping total variance."""
    maturity = data["maturity"].to_numpy(float)
    labels = _expiries(maturity)
    n_times, n_expiries = len(np.unique(maturity)), int(labels.max()) + 1
    if n_expiries == n_times:
        return data
    snapped = pd.Series(maturity).groupby(labels).transform("median").to_numpy()
    messages.append(
        f"Read {n_times} times to expiry as {n_expiries} expiries: times within half a day of each other were "
        "set to their median, keeping each quote's total implied variance."
    )
    return data.assign(maturity=snapped, iv=data["iv"].to_numpy(float) * np.sqrt(maturity / snapped))


# ---------------------------------------------------------------------------
# Option prices and their inversion
# ---------------------------------------------------------------------------


def _option_kind(frame: pd.DataFrame, column) -> pd.Series:
    first = frame[column].astype("string").str.strip().str.lower().str[:1]
    return first.map({"c": "call", "p": "put"})


def _out_of_the_money(
    data: pd.DataFrame, spot: float, rate: float, dividend_yield: float, messages: list[str]
) -> pd.DataFrame:
    """Where an expiry and strike are quoted for both a call and a put, keep the out-of-the-money one.

    In-the-money quotes are the less reliable side (wide spreads, stale trades, early-exercise premium, and
    the placeholder vols some data feeds write). Rows without a call/put type are kept.
    """
    forward = spot * np.exp((rate - dividend_yield) * data["maturity"])
    otm = (data["type"] == "call") == (data["strike"] >= forward)
    expiry = _expiries(data["maturity"])
    both = data.assign(expiry=expiry).groupby(["expiry", "strike"])["type"].transform("nunique") > 1
    if (both & ~otm & data["type"].notna()).any():
        messages.append(
            f"Used the out-of-the-money option at {int((both & otm).sum())} strike(s) quoted for both calls and puts."
        )
        data = data[~both | otm | data["type"].isna()]
    return data


def _prices_to_vols(
    data: pd.DataFrame, spot: float, rate: float, dividend_yield: float, messages: list[str]
) -> pd.DataFrame:
    """Invert option prices into Black-Scholes-Merton implied vols, keeping the out-of-the-money side."""
    data = _out_of_the_money(data, spot, rate, dividend_yield, messages)
    if len(data) > MAX_PRICE_QUOTES:
        raise QuoteParseError(
            f"the file has {len(data):,} option prices to invert; at most {MAX_PRICE_QUOTES:,} are supported, "
            "so filter the chain (for example to fewer expiries) first"
        )
    vols, reasons, no_time_value = [], [], 0
    for price, strike, T, kind in data[["price", "strike", "maturity", "type"]].itertuples(index=False):
        try:
            vols.append(implied_volatility(price, spot, strike, T, rate, dividend_yield, kind))
        except ImpliedVolError as exc:
            vols.append(np.nan)
            if exc.reason == "no_time_value":
                no_time_value += 1
            else:
                reasons.append(str(exc))
    data = data.assign(iv=vols)
    if reasons:
        messages.append(
            f"Dropped {len(reasons)} price quote(s) outside the no-arbitrage bounds at spot {spot:,.2f}, "
            f"rate {rate:.2%} and dividend yield {dividend_yield:.2%} (e.g. {reasons[0]})."
        )
    if no_time_value:
        # In the money within 1e-12·max(S, K) of the intrinsic value, the last bits of a price decide its implied
        # vol (anything from 0 to 100 % and more for a short expiry), so such a quote would only add noise to the fit.
        messages.append(
            f"Dropped {no_time_value} price quote(s) with no time value: each is within one part in 10¹² (of the "
            "larger of spot and strike) of its intrinsic value, the no-arbitrage lower bound, so its implied vol "
            "cannot be determined."
        )
    zero_vol = int((data["iv"] == 0).sum())
    if zero_vol:  # time value so small that the volatility is below 1e-12 (an out-of-the-money quote near zero)
        messages.append(f"Dropped {zero_vol} price quote(s) whose implied vol is zero to machine precision.")
    data = data[data["iv"] > 0]
    messages.append(
        f"Inverted {len(data)} option price(s) into implied vols with Black-Scholes-Merton (European exercise)."
    )
    return data


_PRICE_STEPS = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)


def _price_step(prices: pd.Series) -> float:
    """The coarsest usual price increment that every positive price is a whole multiple of, or 0 if none is.

    A midpoint of a bid and an ask in 0.05 ticks lies on a 0.025 grid, and the price the quotes stand for can be
    up to 0.025 from it, so a price may be off by one step.
    """
    values = prices[prices > 0].to_numpy(float)
    if not len(values):
        return 0.0
    steps = [step for step in _PRICE_STEPS if np.all(np.abs(values / step - np.round(values / step)) < 1e-6)]
    return max(steps, default=0.0)


def _option_prices(
    frame: pd.DataFrame, found: dict, used: list[str], messages: list[str]
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Each row's option price and how far it may be off, and the rows dropped for a zero bid or an ask below the bid.

    The quoted price (Price, Mid, Mark) where it is positive, else the bid/ask midpoint, else the last trade: an
    empty or partly blank price column next to valid bids and asks no longer blocks them. A price may be off by
    the step of its column's grid (:func:`_price_step`), and one with a bid and an ask (``0 <= bid <= ask``) by
    at least as much as takes it to the farther of them: half the spread for their midpoint, so that a price
    can stand for buying at the ask or selling at the bid.
    """
    price = pd.Series(np.nan, index=frame.index)
    error = pd.Series(np.nan, index=frame.index)
    unusable = pd.Series(False, index=frame.index)
    if "price" in found:
        column = found["price"][0]
        price = _numeric(frame, column, messages, required=False)[0].where(lambda values: values > 0)
        if price.notna().any():
            used.append(f"option price from '{column}'")
            error = error.where(price.isna(), _price_step(price))
    bid_ask = None
    if "bid" in found and "ask" in found:  # read even when every row has a price: they bound its error
        bid_ask = (
            _numeric(frame, found["bid"][0], messages, required=False)[0],
            _numeric(frame, found["ask"][0], messages, required=False)[0],
        )
    if bid_ask is not None and price.isna().any():
        price, error, unusable = _bid_ask_midpoints(price, error, *bid_ask, found, used, messages)
    if "last" in found and (price.isna() & ~unusable).any():
        price, error = _last_trades(frame, found, price, error, unusable, used, messages)
    if bid_ask is not None:
        # A Mid or Mark next to the bid and ask was allowed only its grid's step (0.025 for midpoints of 0.05
        # ticks), so adding Mid = (Bid + Ask) / 2 to an arbitrage-free Bid/Ask file reported static arbitrage.
        bid, ask = bid_ask
        quoted = price.notna() & (bid >= 0) & (ask > 0) & (ask >= bid)  # a 0/0 row is no market, not a spread
        widest = np.maximum(error, np.maximum(ask - price, price - bid))  # a Series: NumPy ufuncs keep the index
        error = error.where(~quoted, cast(pd.Series, widest))
    if price.isna().all() and not unusable.all():
        columns = ", ".join(repr(found[key][0]) for key in ("price", "bid", "ask", "last") if key in found)
        raise QuoteParseError(f"no numeric option price in {columns}")
    return price, error, unusable


def _bid_ask_midpoints(
    price: pd.Series,
    error: pd.Series,
    bid: pd.Series,
    ask: pd.Series,
    found: dict,
    used: list[str],
    messages: list[str],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """The bid/ask midpoint, off by half the spread, where ``price`` is missing; and the rows without a usable bid and
    ask (a zero bid or an ask below the bid), which are dropped with a message."""
    need = price.isna() & bid.notna() & ask.notna()
    unusable = need & ((bid <= 0) | (ask < bid))
    if unusable.any():
        messages.append(f"Dropped {int(unusable.sum())} quote(s) with a zero bid or an ask below the bid.")
    filled = need & ~unusable
    if filled.any():
        price = price.where(~filled, (bid + ask) / 2)
        error = error.where(~filled, (ask - bid) / 2)
        where = f" where '{found['price'][0]}' is blank" if "price" in found else ""
        used.append(f"option price as the bid/ask midpoint of '{found['bid'][0]}' and '{found['ask'][0]}'{where}")
    return price, error, unusable


def _last_trades(
    frame: pd.DataFrame,
    found: dict,
    price: pd.Series,
    error: pd.Series,
    unusable: pd.Series,
    used: list[str],
    messages: list[str],
) -> tuple[pd.Series, pd.Series]:
    """The last trade, off by its grid's step, where there is still no price and the row is not ``unusable``."""
    last = _numeric(frame, found["last"][0], messages, required=False)[0]
    filled = price.isna() & ~unusable & (last > 0)
    if filled.any():
        price = price.where(~filled, last)
        error = error.where(~filled, _price_step(last))
        where = " where no other price is given" if "price" in found or "bid" in found else ""
        used.append(f"option price from '{found['last'][0]}' (last trade){where}")
    return price, error


# ---------------------------------------------------------------------------
# parse_quotes and its stages
# ---------------------------------------------------------------------------


def parse_quotes(
    frame: pd.DataFrame, spot: float | None = None, rate: float = 0.0, dividend_yield: float = 0.0
) -> tuple[pd.DataFrame, list[str]]:
    """Validate and clean a quotes table.

    Accepts (case, spacing and punctuation ignored; common aliases allowed) a
    ``Strike`` column, a ``Days`` column or a ``Maturity`` column in years (a days
    column wins when both are present), and either an ``ImpliedVol`` column
    (decimal or percent) or option prices (``Price``/``Mid``, with ``Bid`` and ``Ask``
    filling the rows where it is blank) with a call/put ``Type`` column. Prices are
    inverted with Black-Scholes-Merton at ``spot``, ``rate`` and ``dividend_yield``;
    where an expiry and strike are quoted for both a call and a put, the
    out-of-the-money one is kept, for implied vols too when a ``Type`` column and the
    spot are given. Units in brackets are read (``IV (%)``, ``Maturity (days)``,
    ``Maturity (months)``, ``Maturity (weeks)``, ``DTE (hours)``; a bare ``m`` is
    refused as months or minutes), values may carry ``%`` signs, thousands
    separators or decimal commas, a maturity column that can only hold days (see
    :func:`_maturities`) is converted, and one that could hold years or days is
    refused; values read as years without a stated unit get a warning when they
    could well be days. Dates, tenors (``3M``) and values above 100 years are
    refused. Times to expiry within half a day of each other are one expiry
    (:func:`_expiries`), and implied vols below :data:`MIN_QUOTED_VOL` are dropped as
    placeholders.

    Returns ``(clean, messages)`` where ``clean`` has columns ``strike``,
    ``maturity`` (years), ``iv`` (decimal) and ``price_error`` (how far the option
    price behind each quote may be off: the step of its price grid, or the distance to
    the farther of its bid and ask when the file gives them and that is more, which is
    half the spread for a Bid/Ask midpoint; NaN for an implied vol), and ``messages`` describes
    every interpretation and adjustment; a :class:`QuoteWarning` among them flags a
    unit that was guessed. Raises :class:`QuoteParseError` if the data is unusable.
    """
    messages: list[str] = []
    found = _map_columns(frame, messages)
    columns = _normalise_units(frame.reset_index(drop=True), found, spot, messages)
    data = _invert(_valid_rows(columns, messages), columns.iv_percent, spot, rate, dividend_yield, messages)
    return _filter(data, columns.guess, messages), messages


# The stages of parse_quotes, in the order it runs them. Each appends its messages to ``messages``, so they read in
# that order, and the first refusal (QuoteParseError) wins.


def _map_columns(frame: pd.DataFrame, messages: list[str]) -> dict[str, tuple[str, str | None]]:
    """Stage 1 of :func:`parse_quotes`: which column holds what (:func:`_match_columns`).

    A bare ``Vol`` next to option prices is read as trading volume. Refuses a table with more than :data:`MAX_ROWS`
    rows, without a strike or a time to expiry, or with neither implied vols nor option prices.
    """
    if len(frame) > MAX_ROWS:
        raise QuoteParseError(f"the file has {len(frame):,} rows; at most {MAX_ROWS:,} are supported")
    found = _match_columns(frame)
    has_prices = "price" in found or ("bid" in found and "ask" in found) or "last" in found
    if has_prices and "iv" in found and _header(found["iv"][0])[0] == _WEAK_IV:
        messages.append(
            f"Read '{found['iv'][0]}' as trading volume because the file has option prices; "
            "name the column 'IV' if it holds implied vols."
        )
        del found["iv"]
    if "strike" not in found or ("maturity" not in found and "days" not in found):
        raise QuoteParseError(
            "the file needs a Strike column and a Maturity column (in years) or a Days column; "
            f"found columns: {', '.join(map(str, frame.columns))}"
        )
    if "iv" not in found and not has_prices:
        raise QuoteParseError(
            "the file needs an ImpliedVol column, or option prices (Price, or Bid and Ask) with a call/put Type "
            f"column; found columns: {', '.join(map(str, frame.columns))}"
        )
    return found


@dataclass(frozen=True, eq=False)  # holds tables: compared by identity
class _Columns:
    """The quotes as numbers in their units (stage 2 of :func:`parse_quotes`), one row per row of the file."""

    data: pd.DataFrame  # strike, maturity (years) and iv as quoted (with type), or type, price and price_error
    reported: pd.Series  # rows already dropped with a specific message
    guess: _GuessedYears | None  # see _maturities
    iv_percent: str | None  # "header" or "values" when the header or % signs say the vols are percentages


def _normalise_units(
    frame: pd.DataFrame, found: dict[str, tuple[str, str | None]], spot: float | None, messages: list[str]
) -> _Columns:
    """Stage 2 of :func:`parse_quotes`: each column as numbers in its stated or guessed unit (:func:`_numeric`).

    The strike, then the time to expiry in years (:func:`_maturities`), then the implied vols
    (:func:`_implied_vols`) or the option prices (:func:`_prices`); then the message that says which column gave
    what. ``frame`` has a fresh index.
    """
    data = pd.DataFrame({"strike": _numeric(frame, found["strike"][0], messages)[0]})
    used = [f"strike from '{found['strike'][0]}'"]
    reported = pd.Series(False, index=data.index)  # rows already dropped with a specific message
    data["maturity"], guess = _maturities(frame, found, used, messages)
    iv_percent = None
    if "iv" in found:
        values, iv_percent = _implied_vols(frame, found, spot, used, messages)
    else:
        values, dropped = _prices(frame, found, spot, used, messages)
        reported |= dropped
    for name, column in values.items():
        data[name] = column
    messages.insert(0, "Read " + ", ".join(used) + ".")
    return _Columns(data, reported, guess, iv_percent)


def _implied_vols(
    frame: pd.DataFrame,
    found: dict[str, tuple[str, str | None]],
    spot: float | None,
    used: list[str],
    messages: list[str],
) -> tuple[dict[str, pd.Series], str | None]:
    """The implied vols as quoted, with the call/put type when there is a spot to tell the out-of-the-money side,
    and whether the header (``"header"``) or ``%`` signs (``"values"``) say they are percentages."""
    column, unit = found["iv"]
    iv, percent_signs = _numeric(frame, column, messages)
    iv_percent = "header" if unit == "percent" else "values" if percent_signs else None
    values = {"iv": iv}
    used.append(f"implied vol from '{column}'")
    if "type" in found and spot is not None:  # calls and puts stacked: keep the out-of-the-money side
        values["type"] = _option_kind(frame, found["type"][0])
        used.append(f"call/put from '{found['type'][0]}'")
    return values, iv_percent


def _prices(
    frame: pd.DataFrame,
    found: dict[str, tuple[str, str | None]],
    spot: float | None,
    used: list[str],
    messages: list[str],
) -> tuple[dict[str, pd.Series], pd.Series]:
    """Each row's call/put type, option price and price error (:func:`_option_prices`), and the rows dropped with a
    message: no usable bid and ask, or a type that is neither call nor put. Prices need a type and a spot."""
    if "type" not in found:
        raise QuoteParseError(
            "option prices need a Type column saying call or put for each row; "
            f"found columns: {', '.join(map(str, frame.columns))}"
        )
    if spot is None:
        raise QuoteParseError("option prices need the spot price, rate and dividend yield to be inverted")
    kind = _option_kind(frame, found["type"][0])
    price, price_error, unusable = _option_prices(frame, found, used, messages)
    used.append(f"call/put from '{found['type'][0]}'")
    bad_type = kind.isna() & frame[found["type"][0]].notna()
    if bad_type.any():
        messages.append(f"Dropped {int(bad_type.sum())} row(s) whose type is neither call nor put.")
    return {"type": kind, "price": price, "price_error": price_error}, unusable | bad_type


def _valid_rows(columns: _Columns, messages: list[str]) -> pd.DataFrame:
    """Stage 3 of :func:`parse_quotes`: the rows with every value present and a positive strike, maturity and vol or
    price; the rows that no earlier message explains are counted in one."""
    data = columns.data
    value = "iv" if "iv" in data else "price"
    # An implied vol needs no type: a row whose type is blank or unknown is kept as it is.
    checked = data.drop(columns="type") if value == "iv" and "type" in data else data
    valid = checked.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    valid &= (data["strike"] > 0) & (data["maturity"] > 0) & (data[value] > 0)
    unexplained = int((~valid & ~columns.reported).sum())
    if unexplained:
        messages.append(f"Dropped {unexplained} row(s) with missing, non-numeric or non-positive values.")
    data = data[valid].copy()
    if data.empty:
        raise QuoteParseError("no valid rows remain after cleaning")
    return data


def _invert(
    data: pd.DataFrame,
    iv_percent: str | None,
    spot: float | None,
    rate: float,
    dividend_yield: float,
    messages: list[str],
) -> pd.DataFrame:
    """Stage 4 of :func:`parse_quotes`: implied vols as decimals, inverted from option prices
    (:func:`_prices_to_vols`) or read from the quoted vols in their unit (:func:`_quoted_vols`)."""
    if "iv" in data:
        return _quoted_vols(data, iv_percent, spot, rate, dividend_yield, messages)
    assert spot is not None  # prices without a spot were refused in stage 2
    data = _prices_to_vols(data, spot, rate, dividend_yield, messages)
    if data.empty:
        raise QuoteParseError("no option price could be inverted into an implied vol; check the spot, rate and units")
    return data


def _quoted_vols(
    data: pd.DataFrame,
    iv_percent: str | None,
    spot: float | None,
    rate: float,
    dividend_yield: float,
    messages: list[str],
) -> pd.DataFrame:
    """Quoted implied vols as decimals: the out-of-the-money side where a strike has both, placeholders
    (:data:`MIN_QUOTED_VOL`) dropped, and percentages divided by 100, with a warning for an ambiguous median."""
    if "type" in data and spot is not None:  # a type is only read with a spot (stage 2)
        data = _out_of_the_money(data, spot, rate, dividend_yield, messages)
    # MIN_QUOTED_VOL is a decimal vol: 100 times that in a file of percentages (recognised as below).
    in_percent = iv_percent is not None or data.loc[data["iv"] >= MIN_QUOTED_VOL, "iv"].median() > 3.0
    placeholder = data["iv"] < MIN_QUOTED_VOL * (100.0 if in_percent else 1.0)
    if placeholder.any():
        messages.append(
            f"Dropped {int(placeholder.sum())} quote(s) with an implied vol below {MIN_QUOTED_VOL:.2%}, a "
            "placeholder some data feeds write where there is no vol."
        )
        data = data[~placeholder].copy()
        if data.empty:
            raise QuoteParseError("no valid rows remain after cleaning")
    median = float(data["iv"].median())
    if iv_percent is not None:
        data["iv"] = data["iv"] / 100.0
        source = "the column header" if iv_percent == "header" else "the % signs"
        messages.append(f"Implied vols read as percentages (from {source}); divided by 100.")
    elif median > 3.0:
        data["iv"] = data["iv"] / 100.0
        messages.append(f"Implied vols look like percentages (median {median:.4g}); divided by 100.")
    else:
        messages.append(f"Implied vols read as decimals (median {median:.4g} = {median:.2%}).")
    if iv_percent is None and 1.0 <= median <= 5.0:
        messages.append(
            QuoteWarning(
                f"A median implied vol of {median:.4g} is ambiguous (it could be {median:.4g}% or {median:.0%}); "
                "name the column 'IV (%)' if the values are percentages."
            )
        )
    return data


def _filter(data: pd.DataFrame, guess: _GuessedYears | None, messages: list[str]) -> pd.DataFrame:
    """Stage 5 of :func:`parse_quotes`: implied vols above 500% dropped, times within half a day made one expiry
    (:func:`_merge_close_maturities`), duplicate quotes averaged, and the clean table in its final layout; with a
    warning when years was assumed for maturities that could well be days (:func:`_maturities`)."""
    if (data["iv"] > 5.0).any():
        n_bad = int((data["iv"] > 5.0).sum())
        data = data[data["iv"] <= 5.0]
        messages.append(f"Dropped {n_bad} quote(s) with implied vol above 500%.")
    if data.empty:
        raise QuoteParseError("no valid rows remain after cleaning")

    data = _merge_close_maturities(data, messages)
    data = data.assign(maturity=data["maturity"].round(10))
    if "price_error" not in data:  # implied vols: the price behind each one, and so its rounding, is unknown
        data = data.assign(price_error=np.nan)
    duplicated = data.duplicated(["maturity", "strike"], keep=False)
    if duplicated.any():
        messages.append(f"Averaged {int(duplicated.sum())} duplicate (maturity, strike) quotes.")
        data = data.groupby(["maturity", "strike"], as_index=False)[["iv", "price_error"]].mean()
    columns = ["strike", "maturity", "iv", "price_error"]
    data = data.sort_values(["maturity", "strike"]).reset_index(drop=True)[columns]
    if guess is not None and guess.suspect:  # years was assumed for values that could well be days
        warning = f"Maturities in '{guess.column}' {guess.described}, read as years; {_NAME_THE_UNIT}."
        messages.append(QuoteWarning(warning))
    return data


# ---------------------------------------------------------------------------
# Static arbitrage in the quotes (model-free)
# ---------------------------------------------------------------------------


SPREAD_CHECK_VOL = 0.01  # vol error allowed on each quote before prices that run the wrong way with strike count
FLY_CHECK_VOL = 0.005  # vol error allowed on each quote before a butterfly with a negative cost counts
# Price error allowed on a quoted implied vol, whose price and tick are unknown: this fraction of the forward or
# MIN_PRICE_ERROR (half a cent, the rounding of a price in cents; at most 1% of the strike spacing, since sub-cent
# strikes are priced in sub-cent ticks), whichever is more.
UNKNOWN_PRICE_ERROR = 1e-4
MIN_PRICE_ERROR = 0.005
# Static arbitrage in quoted implied vols that goes away if each price behind them is off by half this tick is
# reported as possible, not certain: vols inverted from midpoints of bids and asks in 0.05 ticks carry that rounding
# (at most 5% of the strike spacing).
QUOTED_VOL_TICK = 0.05


def _black_prices(strike, iv, T: float, forward: float, vol_error: float, price_error=0.0):
    """Strikes in ascending order with their undiscounted Black call and put prices at ``forward``, and the slack.

    The slack is how far a price moves when its vol is ``vol_error`` off (vega times the error), plus each
    quote's undiscounted ``price_error`` (its rounding or half its bid/ask spread) and 1e-12 of the forward.
    """
    order = np.argsort(strike, kind="stable")
    strike, iv = np.asarray(strike, dtype=float)[order], np.asarray(iv, dtype=float)[order]
    price_error = np.broadcast_to(np.asarray(price_error, dtype=float), order.shape)[order]
    sd = iv * np.sqrt(T)
    d1 = np.log(forward / strike) / sd + 0.5 * sd
    # Both prices directly: a call from put-call parity is rounding noise far out of the money.
    put = strike * ndtr(sd - d1) - forward * ndtr(-d1)
    call = forward * ndtr(d1) - strike * ndtr(d1 - sd)
    vega = forward * np.sqrt(T) * np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)
    return strike, call, put, vol_error * vega + price_error + 1e-12 * forward


def _spread_arbitrage(strike, iv, T: float, forward: float, price_error=0.0) -> bool:
    """Whether quoted vols price some put below a put at a lower strike, or some call above a call at a lower strike.

    Undiscounted Black prices at ``forward``; each quote's vol may be off by :data:`SPREAD_CHECK_VOL`, and its
    price by its undiscounted ``price_error``, in whichever direction removes the arbitrage. No arbitrage-free
    market prices a put below a lower-strike put or a call above a lower-strike call, so this is model-free.
    """
    _, call, put, slack = _black_prices(strike, iv, T, forward, SPREAD_CHECK_VOL, price_error)
    put_floor = np.maximum.accumulate(put - slack)[:-1]  # a put at a higher strike is worth at least this
    call_cap = np.minimum.accumulate(call + slack)[:-1]  # a call at a higher strike is worth at most this
    return bool(np.any(put[1:] + slack[1:] < put_floor) or np.any(call[1:] - slack[1:] > call_cap))


def _butterfly_arbitrage(
    strike, iv, T: float, forward: float, price_error=0.0
) -> tuple[int, tuple[float, float, float] | None]:
    """Butterflies of three neighbouring quoted strikes that cost less than nothing: ``(count, worst strikes)``.

    A long butterfly ``lam C(K1) - C(K2) + (1 - lam) C(K3)`` with ``K2 = lam K1 + (1 - lam) K3`` never pays
    less than zero, so in an arbitrage-free market it costs at least zero (call prices are convex in strike;
    put butterflies cost the same by put-call parity). Undiscounted Black prices at ``forward``; each quote's
    vol may be off by :data:`FLY_CHECK_VOL`, and its price by its undiscounted ``price_error``, in whichever
    direction removes the arbitrage, so quote noise and price rounding are not reported (with half the bid/ask
    spread as the price error, a reported butterfly costs less than nothing bought at the ask and sold at the
    bid). Model-free: it checks the quotes, not a fitted smile. Repeated strikes are averaged first.
    """
    unique, which = np.unique(np.asarray(strike, dtype=float), return_inverse=True)
    if len(unique) < 3:
        return 0, None
    count = np.bincount(which)
    iv = np.bincount(which, np.asarray(iv, dtype=float)) / count
    price_error = np.broadcast_to(np.asarray(price_error, dtype=float), which.shape)
    K, call, _, slack = _black_prices(unique, iv, T, forward, FLY_CHECK_VOL, np.bincount(which, price_error) / count)
    lam = (K[2:] - K[1:-1]) / (K[2:] - K[:-2])
    cost = lam * (call[:-2] + slack[:-2]) + (1.0 - lam) * (call[2:] + slack[2:]) - (call[1:-1] - slack[1:-1])
    bad = cost < 0
    if not bad.any():
        return 0, None
    worst = int(np.argmin(np.where(bad, cost, np.inf)))
    return int(bad.sum()), (float(K[worst]), float(K[worst + 1]), float(K[worst + 2]))


def _strike_text(strike: float) -> str:
    """A strike for a message: thousands separators and no trailing zeros (``2,452.5``, ``100``)."""
    return f"{strike:,.2f}".rstrip("0").rstrip(".")
