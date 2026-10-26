"""Number formatting for the user interface.

The functions from :func:`signed` on are the desk number-format standard:

* Signed numbers (P&L, Greeks, changes) carry an explicit sign, with the true minus sign U+2212 (as wide as a
  digit in IBM Plex, so signed columns line up) and no sign at zero: never ``−0.00``.
* Greeks keep at least three significant digits (:func:`greek`), so small second-order Greeks never read
  ``0.000000``; dollar Greeks have one explicit definition each (:func:`dollar_delta` and the functions after it).
* Dates read ``20-Oct-27 (365d)`` (:func:`expiry`), money in accounting style ``($1,234.56)`` where a column of
  amounts needs it (:func:`accounting`), and position sizes are whole numbers (:func:`quantity`).

Tables: numbers are right-aligned (:func:`right_aligned`), small static tables use
``st.table(..., border="horizontal")`` and large or sortable ones ``st.dataframe(..., row_height=ROW_HEIGHT)`` with
``st.column_config.NumberColumn`` formats.
"""

from __future__ import annotations

import datetime as dt
import math
import re

from .. import black_scholes as bs


def money(value: float, decimals: int = 2, signed: bool = False) -> str:
    """``$1,234.56``; ``Unlimited`` for infinities; optional explicit sign."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if math.isinf(value):
        return "Unlimited"
    if abs(value) < 0.5 * 10**-decimals:
        value = 0.0
    sign = "-" if value < 0 else ("+" if signed and value > 0 else "")
    return f"{sign}${abs(value):,.{decimals}f}"


def number(value: float, decimals: int) -> str:
    """Fixed-point number without a negative zero (``-0.00`` reads as a short position)."""
    if abs(value) < 0.5 * 10**-decimals:
        value = 0.0
    return f"{value:,.{decimals}f}"


def shown_decimals(size: float, decimals: int) -> int:
    """Decimals that keep a number of magnitude ``size`` short: ``decimals`` below 1,000, at most two below
    10,000 and none from 10,000. The share and gamma cards and the whole-position columns of the legs table use it."""
    if round(size, decimals) < 1e3:
        return decimals
    return min(decimals, 2) if round(size, 2) < 1e4 else 0


#: Scales of the shortened amounts on metric cards, after whole numbers below a million.
SHORT_SCALES = ((1e6, "M"), (1e9, "B"), (1e12, "T"))


def _scaled(value: float, prefix: str, signed: bool) -> str:
    """``1.23M``, ``12.35B`` or ``123.5T`` (with ``prefix`` after the sign), and ``1.2e+15`` beyond."""
    size = abs(value)
    sign = "-" if value < 0 else ("+" if signed else "")
    for scale, suffix in SHORT_SCALES:
        mantissa = size / scale
        decimals = 2 if round(mantissa, 2) < 100 else 1  # three or four significant figures: '+$337.7M' fits
        if round(mantissa, decimals) < 1000:
            return f"{sign}{prefix}{mantissa:.{decimals}f}{suffix}"
    return f"{sign}{prefix}{size:.1e}"


def money_short(value: float, signed: bool = False) -> str:
    """Dollar amount short enough for a metric card at laptop width, sign included: at most nine characters.

    Cents below $1,000, whole dollars below a million, then ``$1.23M``, ``$1.23B`` and ``$1.23T``, and
    scientific notation (``$1.2e+15``) from a thousand trillion. Show the exact amount next to it (the help).
    The sign counts: ``+`` is as wide as a digit, so ``+$2,737.18`` would not fit where ``-$2,737.18`` does.
    """
    if value is None or not math.isfinite(value):
        return money(value, 2, signed)
    if round(abs(value)) >= 1e6:
        return _scaled(value, "$", signed)
    return money(value, 2 if round(abs(value), 2) < 1e3 else 0, signed)


def number_short(value: float, decimals: int) -> str:
    """Like :func:`money_short` for a plain number shown with ``decimals``: fewer decimals from 1,000."""
    if math.isfinite(value) and round(abs(value)) >= 1e6:
        return _scaled(value, "", False)
    return number(value, shown_decimals(abs(value), decimals) if math.isfinite(value) else decimals)


def pct(value: float, decimals: int = 2, signed: bool = False) -> str:
    """Format a decimal fraction as a percentage (0.1234 -> ``12.34%``)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return f"{value * 100:{'+' if signed else ''}.{decimals}f}%"


def md(text: str) -> str:
    """Escape dollar signs so Markdown does not render ``$a ... $b`` as LaTeX math."""
    return text.replace("$", r"\$")


def md_code(text: str) -> str:
    """``text`` as inline code, so that Streamlit's Markdown shows it as typed.

    For text the user supplies, such as a file name or a value in a link. Escaping its Markdown characters would not
    do: Streamlit still turns ``http://...`` and ``www....`` into links and ``:material/...:`` into icons. Inside a
    code span nothing is formatted. The span's fence is one backtick longer than any run of backticks in ``text``
    (CommonMark), and every run of whitespace is one space, since a blank line would end the span.

    Parameters
    ----------
    text : str
        Plain text.

    Returns
    -------
    str
        Markdown for ``text`` in a code span.

    Examples
    --------
    >>> md_code("quotes.csv")
    '`quotes.csv`'
    >>> md_code("[x](http://a.b) `y`")
    '`` [x](http://a.b) `y` ``'
    """
    text = " ".join(text.split())
    fence = "`" * (max(map(len, re.findall("`+", text)), default=0) + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


# --- Desk number-format standard ---------------------------------------------------------------------------------

#: The minus sign (U+2212) of signed numbers.
MINUS = "−"
#: Row height (px) of large tables: ``st.dataframe(..., row_height=ROW_HEIGHT)``.
ROW_HEIGHT = 28
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _missing(value) -> bool:
    return value is None or math.isnan(value)


def signed(value: float, decimals: int = 2, prefix: str = "", suffix: str = "") -> str:
    """``+1,234.50`` or ``−1,234.50``: an explicit sign (U+2212 for minus), then ``prefix`` (``+$1,234.50``), the
    number and ``suffix`` (``+2.50 vol pts``).

    A value that rounds to zero at ``decimals`` has no sign (``0.00``, never ``−0.00``), so ``signed(x)`` and
    ``signed(-x)`` always have the same length. Infinities read ``+∞`` and ``−∞``; NaN and ``None`` read ``–``.
    """
    if _missing(value):
        return "–"
    sign = "+" if value > 0 else MINUS
    if math.isinf(value):
        return f"{sign}∞"
    text = f"{abs(value):,.{decimals}f}"
    if text.strip("0.,") == "":
        return f"{prefix}{text}{suffix}"
    return f"{sign}{prefix}{text}{suffix}"


def _significant(size: float, digits: int) -> str:
    """A positive ``size`` with at least ``digits`` significant digits: fixed-point down to 1e-6, then
    ``1.230e−7``."""
    if size < 1e-6:
        mantissa, exponent = f"{size:.{digits - 1}e}".split("e")
        return f"{mantissa}e{MINUS if exponent.startswith('-') else ''}{int(exponent[1:])}"
    decimals = max(0, digits - 1 - math.floor(math.log10(size)))
    return f"{size:,.{decimals}f}"


def greek(value: float, kind: str, digits: int = 4, unit: bool = True) -> str:
    """A Greek in its display unit with at least ``digits`` significant digits (and never fewer than three).

    ``value`` is in the model's raw units, as :func:`options_analyzer.black_scholes.greeks` returns it; it is
    scaled by ``black_scholes.GREEKS[kind]`` (theta per calendar day, vega per vol point, ...), whose unit follows
    the number when ``unit`` is true: ``greek(0.0123, "vanna")`` reads ``+0.0001230 Δ per 1 vol point``.
    Zero reads ``0``.
    """
    info = bs.GREEKS[kind]
    if _missing(value):
        text = "–"
    else:
        scaled = value * info.scale
        if scaled == 0:
            text = "0"
        elif math.isinf(scaled):
            text = signed(scaled)
        else:
            text = ("+" if scaled > 0 else MINUS) + _significant(abs(scaled), max(3, digits))
    return f"{text} {info.unit}" if unit else text


def dollar_delta(delta: float, spot: float, units: float) -> float:
    """$Δ, the dollar value of a position's delta ("delta dollars"): Δ · S · units."""
    return delta * spot * units


def gamma_1pct(gamma: float, spot: float, units: float) -> float:
    """1% Γ, the change in a position's delta (in shares) for a 1% move in spot: Γ · S · 0.01 · units."""
    return gamma * spot * 0.01 * units


def dollar_gamma_1pct(gamma: float, spot: float, units: float) -> float:
    """$Γ per 1%, the gamma P&L in dollars of a 1% move in spot: ½ · Γ · (0.01 · S)² · units."""
    return 0.5 * gamma * (0.01 * spot) ** 2 * units


def dollar_vega(vega: float, units: float) -> float:
    """Vega in dollars per vol point (raw vega is per 1.00, i.e. 100 vol points): ν · 0.01 · units."""
    return vega * 0.01 * units


def dollar_theta(theta: float, units: float) -> float:
    """Theta in dollars per calendar day (raw theta is per year): θ / 365 · units."""
    return theta / 365.0 * units


def dollar_rho(rho: float, units: float) -> float:
    """Rho in dollars per 1% move in the rate (raw rho is per 1.00, i.e. 100%): ρ · 0.01 · units."""
    return rho * 0.01 * units


def level(value: float) -> str:
    """A price level (spot, strike, forward): ``5,812.37``, ``0.0420`` or ``0.00004087``.

    Two decimals with thousands separators from 1, four decimals from 0.01 and four significant digits below, so
    that a small level (a forward that a high dividend yield pulls down) never reads ``0.0000``. A negative level
    takes the minus sign U+2212; NaN and ``None`` read ``–``.

    Parameters
    ----------
    value : float
        The level, in currency units.

    Returns
    -------
    str
        The level as shown, without a currency sign.

    Examples
    --------
    >>> level(5812.374), level(0.042), level(4.0866e-5)
    ('5,812.37', '0.0420', '0.00004087')
    """
    if _missing(value):
        return "–"
    size = abs(value)
    if math.isinf(size):
        text = "∞"
    elif round(size, 2) >= 1:
        text = f"{size:,.2f}"
    elif round(size, 4) >= 0.01 or size == 0:
        text = f"{size:.4f}"
    else:
        text = _significant(size, 4)
    return MINUS + text if value < 0 else text


def date(day: dt.date) -> str:
    """``20-Oct-27``: day, English month abbreviation (whatever the server's locale) and two-digit year."""
    return f"{day.day:02d}-{_MONTHS[day.month - 1]}-{day.year % 100:02d}"


def expiry(days: float, today: dt.date | None = None) -> str:
    """The date ``days`` calendar days after ``today`` (default: the server's date) and the days left:
    ``20-Oct-27 (365d)``."""
    days = round(days)
    return f"{date((today or dt.date.today()) + dt.timedelta(days=days))} ({days}d)"


def accounting(value: float, decimals: int = 2, prefix: str = "$") -> str:
    """Accounting style: ``$1,234.56``, and ``($1,234.56)`` for a negative amount (never ``($0.00)``);
    ``Unlimited`` and ``(Unlimited)`` for infinities."""
    if _missing(value):
        return "–"
    text = "Unlimited" if math.isinf(value) else f"{prefix}{abs(value):,.{decimals}f}"
    rounds_to_zero = text[len(prefix) :].strip("0.,") == ""
    return f"({text})" if value < 0 and not rounds_to_zero else text


def quantity(value: float, sign: bool = True) -> str:
    """A position size as a whole number with thousands separators: ``+1,000``, ``−5`` and ``0`` (``1,000``
    without the plus sign when ``sign`` is false)."""
    if _missing(value) or math.isinf(value):
        return signed(value)
    whole = round(value)
    text = f"{abs(whole):,d}"
    if whole < 0:
        return MINUS + text
    return f"+{text}" if sign and whole > 0 else text


def right_aligned(frame, text_columns=()):
    """A pandas Styler of ``frame`` for ``st.table`` with every column except ``text_columns`` right-aligned.

    Use it for columns of formatted text (from :func:`signed` and the other formatters), which ``st.table``
    left-aligns with an inline style (hence ``!important``). Numeric columns are right-aligned already; to show
    them with a formatter, keep the numbers and pass it to ``frame.style.format`` instead.
    """
    text = [c for c in frame.columns if c not in set(text_columns)]
    return frame.style.set_properties(subset=text, **{"text-align": "right !important"})
