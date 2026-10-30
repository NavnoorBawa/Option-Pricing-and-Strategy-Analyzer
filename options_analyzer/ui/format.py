"""Number formatting for the user interface.

The functions from :func:`signed` on are the desk number-format standard:

* Signed numbers (P&L, Greeks, changes) carry an explicit sign, with the true minus sign U+2212 (as wide as a
  digit in IBM Plex, so signed columns line up) and no sign at zero: never ``−0.00``.
* Greeks keep at least three significant digits (:func:`greek`), so small second-order Greeks never read
  ``0.000000``; dollar Greeks have one explicit definition each (:func:`dollar_delta` and the functions after it).
  Time decay heads a card or table as theta / 365 or as the one-day decay by repricing, by one rule
  (:func:`decay_leads`), under one of two names (:func:`decay_label`).
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
from .. import clock


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


#: Theta / 365 stays the headline while it is within this fraction of the one-day decay by repricing, measured
#: against the larger of the two in size. At the money (100, 20 % volatility, 5 % rate) the two differ by 8.7 % (call)
#: and 9.7 % (put) at 3 days to expiry and by 14.1 % and 15.2 % at 2 days; on the panel's 7-day SPX-like iron condor
#: by 9.5 % ($137.50 against $151.97), so that one keeps theta.
DECAY_SWITCH = 0.10


def decay_label(to_expiry: bool) -> str:
    """The name of the one-day decay by repricing: ``Decay to expiry`` when the (front) expiry falls within the day,
    so that the figure is the whole remaining time value, and ``1-day decay`` otherwise."""
    return "Decay to expiry" if to_expiry else "1-day decay"


def decay_leads(theta_per_day: float, decay: float, to_expiry: bool = False) -> bool:
    """Whether the one-day decay by repricing is the headline rather than theta / 365.

    Theta / 365 is the instantaneous rate; the decay is the change in value over the next calendar day, spot and
    implied volatility unchanged. The decay leads when the (front) expiry falls within the day, where a rate per day
    means nothing, and when the two differ by more than :data:`DECAY_SWITCH` of the larger in size:
    ``|decay − θ/365| > 0.10 · max(|decay|, |θ/365|)``. Two zeros agree.

    Examples
    --------
    >>> decay_leads(-10.66, -21.56), decay_leads(137.50, 151.97), decay_leads(-0.01757, -0.01758)
    (True, False, False)
    >>> decay_leads(-0.01757, -0.01758, to_expiry=True)
    True
    """
    return bool(to_expiry or abs(decay - theta_per_day) > DECAY_SWITCH * max(abs(decay), abs(theta_per_day)))


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


# --- The number standard by quantity (UI spec 2.5) -------------------------------------------------------------------
#
# One function per quantity of the standard's table. All of them follow its rules: the minus sign U+2212, no negative
# zero and no sign on a value that rounds to zero, the en dash for a missing value (NaN or None; the reason goes in
# the help), and dates with non-breaking hyphens (U+2011) so that ``20‑Nov‑26`` never wraps at a hyphen. Inside
# ``st.dataframe`` number columns the same quantities come from the column registry, ``components.col``.

#: The text of a missing or unreliable value.
MISSING = "–"
#: The non-breaking hyphen (U+2011) of dates in running text.
NB_HYPHEN = "‑"
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
#: Index futures by name: the dollar value of one index point (CME: E-mini S&P 500 $50, Micro E-mini $5).
FUTURES = {"ES": 50.0, "MES": 5.0}
#: The bases a probability is tagged with: from the published surface's risk-neutral CDF (``RN smile``), N(d₂) at
#: one vol (``flat vol``, the textbook market), or the user's drift at flat ATMF vol (``view``).
PROBABILITY_BASES = ("RN smile", "flat vol", "view")


def _absent(value) -> bool:
    """None or NaN."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def _minus(value: float, text: str) -> str:
    """``text`` (the size of ``value`` as shown) with U+2212 in front for a negative value that does not show as
    zero."""
    return MINUS + text if value < 0 and text.strip("0.,") else text


def _decimals_of(step: float, most: int) -> int:
    """Fewest decimals (at most ``most``) that write ``step`` exactly: 0 for 5, 1 for 2.5, 2 for 0.25."""
    for decimals in range(most):
        scaled = step * 10**decimals
        if abs(scaled - round(scaled)) < 1e-9 * max(1.0, abs(scaled)):
            return decimals
    return most


def strike(k: float, increment: float) -> str:
    """A strike at the precision of the chain's strike increment: no decimals when the increment is whole.

    A strike off the increment's grid (a typed one) keeps the decimals it needs, at most two, so that it never reads
    as a listed neighbour.

    Parameters
    ----------
    k : float
        The strike, in index points or $.
    increment : float
        The chain's strike interval near this strike (``synthetic.strike_interval``): 5 for SPX, 2.5 or 0.5 for
        ETFs, 0.01 for a typed textbook strike.

    Returns
    -------
    str
        ``5,815`` or ``582.5``; ``–`` when missing.

    Examples
    --------
    >>> strike(5815.0, 5.0), strike(582.5, 2.5), strike(5812.37, 5.0), strike(100.0, 0.01)
    ('5,815', '582.5', '5,812.37', '100.00')
    """
    if _absent(k) or math.isinf(k):
        return MISSING
    decimals = max(_decimals_of(increment, 2), _decimals_of(k, 2))
    return _minus(k, f"{abs(k):,.{decimals}f}")


def premium(x: float, digits: int = 2) -> str:
    """An option value per share (theo, bid, ask, mid, entry, package): 2 decimals, or 6 in diagnostics.

    Parameters
    ----------
    x : float
        $ per share (index points).
    digits : int, default 2
        Decimals: 2 everywhere, 6 in the engine-diagnostics tables only.

    Returns
    -------
    str
        ``98.57`` or ``98.570661``; a negative value takes U+2212; ``–`` when missing.

    Examples
    --------
    >>> premium(98.570661), premium(98.570661, 6), premium(1234.5), premium(-0.004), premium(float("nan"))
    ('98.57', '98.570661', '1,234.50', '0.00', '–')
    """
    if _absent(x):
        return MISSING
    if math.isinf(x):
        return signed(x)
    return _minus(x, f"{abs(x):,.{digits}f}")


def signed_sci(x: float, digits: int = 2) -> str:
    """A small signed difference in scientific notation, for diagnostics: ``−3.1e−7``.

    ``digits`` significant digits, U+2212 for the sign and for a negative exponent, no ``+`` in the exponent, and
    ``0`` for an exact zero.

    Parameters
    ----------
    x : float
        The difference (an engine's error against a reference, a parity residual).
    digits : int, default 2
        Significant digits of the mantissa, at least 1.

    Returns
    -------
    str
        ``+3.1e−7``, ``−3.1e−7`` or ``0``; ``–`` when missing.

    Examples
    --------
    >>> signed_sci(-3.1e-7), signed_sci(3.14159e-7, 3), signed_sci(2.5e3), signed_sci(0.0), signed_sci(-0.0)
    ('−3.1e−7', '+3.14e−7', '+2.5e3', '0', '0')
    """
    if _absent(x):
        return MISSING
    if x == 0:
        return "0"
    if math.isinf(x):
        return signed(x)
    mantissa, exponent = f"{abs(x):.{max(1, digits) - 1}e}".split("e")
    power = int(exponent)
    return f"{'+' if x > 0 else MINUS}{mantissa}e{MINUS if power < 0 else ''}{abs(power)}"


def vol(sigma: float, digits: int = 2) -> str:
    """An implied vol in percent with 2 decimals: ``15.46%``.

    Parameters
    ----------
    sigma : float
        The vol as a decimal (0.1546).
    digits : int, default 2
        Decimals of the percentage.

    Returns
    -------
    str
        ``15.46%``; ``–`` when missing (beyond the surface's reach, a zero bid).

    Examples
    --------
    >>> vol(0.154649), vol(0.2), vol(None)
    ('15.46%', '20.00%', '–')
    """
    if _absent(sigma) or math.isinf(sigma):
        return MISSING
    return _minus(sigma, f"{abs(sigma) * 100:.{digits}f}") + "%"


def vol_change(dv: float, digits: int = 2) -> str:
    """A change or difference of implied vols in vol points (vp), signed: ``+0.35 vp``.

    For vol shifts, risk reversals, butterflies, skews and the edges (mid vs fit, buy edge, sell edge; UI spec 2.5).

    Parameters
    ----------
    dv : float
        The difference of two vols as a decimal: 0.0035 is 0.35 vp.
    digits : int, default 2
        Decimals of the vol points.

    Returns
    -------
    str
        ``+0.35 vp``, ``−4.21 vp`` or ``0.00 vp`` (no sign at zero); ``–`` when missing.

    Examples
    --------
    >>> vol_change(0.0035), vol_change(-0.0421), vol_change(-0.00004), vol_change(float("nan"))
    ('+0.35 vp', '−4.21 vp', '0.00 vp', '–')
    """
    if _absent(dv):
        return MISSING
    return signed(dv * 100, digits, suffix=" vp")


def bp(x: float) -> str:
    """A rate change, carry gap or spread in basis points, signed.

    Whole basis points from 10 bp, one decimal from 1 bp and two below (``+45 bp``, ``+4.5 bp``, ``+0.05 bp``),
    judged on the value as rounded, so that 9.96 bp reads ``+10 bp`` and not ``+10.0 bp``. An exact zero reads
    ``0 bp``; a value that rounds to zero has no sign.

    Parameters
    ----------
    x : float
        The change as a decimal: 0.0045 is 45 bp.

    Returns
    -------
    str
        ``+45 bp``; ``–`` when missing.

    Examples
    --------
    >>> bp(0.0045), bp(-0.00045), bp(5e-6), bp(0.000996), bp(0.0), bp(-1e-9)
    ('+45 bp', '−4.5 bp', '+0.05 bp', '+10 bp', '0 bp', '0.00 bp')
    """
    if _absent(x):
        return MISSING
    if x == 0:
        return "0 bp"
    size = abs(x) * 1e4
    decimals = 2 if round(size, 2) < 1 else 1 if round(size, 1) < 10 else 0
    return signed(x * 1e4, decimals, suffix=" bp")


def delta_label(delta: float, *, side: bool = True, atm: bool = True) -> str:
    """A delta in delta points, for pickers and buckets: ``16Δ``, ``25Δp``, ``ATM``, ``25Δc``.

    Parameters
    ----------
    delta : float
        The option's delta as a decimal, negative for a put (spot convention, e^{−qT}N(d₁)).
    side : bool, default True
        Append ``c`` or ``p`` from the sign (buckets); a picker that is per side already leaves it off.
    atm : bool, default True
        Name 50 delta points ``ATM`` (buckets); ``False`` keeps ``50Δ`` (the Grid view's columns).

    Returns
    -------
    str
        The label; ``–`` when missing.

    Examples
    --------
    >>> delta_label(0.16, side=False), delta_label(-0.2501), delta_label(0.5), delta_label(-0.5, atm=False)
    ('16Δ', '25Δp', 'ATM', '50Δp')
    """
    if _absent(delta):
        return MISSING
    points = round(abs(delta) * 100)
    if points == 50 and atm:
        return "ATM"
    return f"{points}Δ" + (("c" if delta > 0 else "p") if side and points else "")


def futures_contracts(delta: float, *, carry_factor: float = 1.0, contract: str = "ES") -> float:
    """A position delta, in units of the index, as a number of index futures: Δ ÷ (multiplier · e^{(r−q)T_fut}).

    A future moves e^{(r−q)T_fut} points per point of the index (its forward's sensitivity to spot), so one ES
    contract carries 50 · e^{(r−q)T_fut} units of index delta (UI spec 2.5, position delta).

    Parameters
    ----------
    delta : float
        Position delta in units of the index: per-option delta × contracts × multiplier.
    carry_factor : float, default 1.0
        e^{(r−q)T_fut} to the future's expiry, its forward over spot; 1 ignores the carry.
    contract : {"ES", "MES"}, default "ES"
        E-mini ($50 per point) or Micro E-mini ($5) S&P 500.

    Returns
    -------
    float
        Contracts, signed.

    Examples
    --------
    >>> round(futures_contracts(-336.6, carry_factor=1.00527), 4)
    -6.6967
    """
    return delta / (FUTURES[contract] * carry_factor)


def futures_equiv(delta: float, *, carry_factor: float = 1.0, contract: str = "ES", digits: int = 2) -> str:
    """A position delta as index futures, for the index presets: ``−0.07 ES`` (see :func:`futures_contracts`).

    Parameters
    ----------
    delta : float
        Position delta in units of the index.
    carry_factor : float, default 1.0
        e^{(r−q)T_fut} to the future's expiry.
    contract : {"ES", "MES"}, default "ES"
        The contract counted in.
    digits : int, default 2
        Decimals.

    Returns
    -------
    str
        ``−0.07 ES`` or ``0.00 MES`` (no sign at zero); ``–`` when missing.

    Examples
    --------
    >>> futures_equiv(-3.366, carry_factor=1.00527), futures_equiv(-3.366, carry_factor=1.00527, contract="MES")
    ('−0.07 ES', '−0.67 MES')
    """
    if _absent(delta):
        return MISSING
    return signed(futures_contracts(delta, carry_factor=carry_factor, contract=contract), digits, suffix=f" {contract}")


def moneyness(k: float, forward: float, *, log: bool = False) -> str:
    """Moneyness against the forward: K/F in percent with 1 decimal (``97.5%``), or k = ln(K/F) signed with 3
    decimals (``−0.025``).

    Parameters
    ----------
    k, forward : float
        The strike and its expiry's forward, positive.
    log : bool, default False
        Log-moneyness ln(K/F) instead of K/F.

    Returns
    -------
    str
        The moneyness; ``–`` when either is missing or not positive.

    Examples
    --------
    >>> moneyness(5670.0, 5815.22), moneyness(5670.0, 5815.22, log=True), moneyness(5815.22, 5815.22, log=True)
    ('97.5%', '−0.025', '0.000')
    """
    if _absent(k) or _absent(forward) or not (k > 0 and forward > 0):
        return MISSING
    if log:
        return signed(math.log(k / forward), 3)
    return f"{k / forward * 100:.1f}%"


def time_left(minutes: float, *, expiry_day: bool = False) -> str:
    """Time to expiry: whole days from 30 days (``55d``), days and hours from 1 to 30 days (``27d 17h``), hours and
    minutes below a day (``6h 30m``), and ``0DTE 2h 15m`` on the expiry's own day.

    Each part is floored, so the text never promises time that is not left; nothing left reads ``expired``.

    Parameters
    ----------
    minutes : float
        Minutes to the settlement (``clock.year_fraction`` × 525,600).
    expiry_day : bool, default False
        Whether the valuation time falls on the expiry's date (a 0DTE option).

    Returns
    -------
    str
        The time left; ``–`` when missing.

    Examples
    --------
    >>> time_left(55.7 * 1440), time_left(27 * 1440 + 17.9 * 60), time_left(390), time_left(135, expiry_day=True)
    ('55d', '27d 17h', '6h 30m', '0DTE 2h 15m')
    >>> time_left(30 * 1440), time_left(1440), time_left(0.5), time_left(0)
    ('30d', '1d 0h', '0h 0m', 'expired')
    """
    if _absent(minutes):
        return MISSING
    if minutes <= 0:
        return "expired"
    whole = math.floor(minutes + 1e-9)  # a float a hair under a whole minute (from a year fraction) is that minute
    days, rest = divmod(whole, 1440)
    hours, mins = divmod(rest, 60)
    if days >= 30:
        return f"{days}d"
    if days >= 1:
        return f"{days}d {hours}h"
    return f"{'0DTE ' if expiry_day else ''}{hours}h {mins}m"


def _nb_date(day: dt.date) -> str:
    """``20‑Nov‑26``, with non-breaking hyphens."""
    return date(day).replace("-", NB_HYPHEN)


def session_label(n: int, day: dt.date) -> str:
    """A scenario date: ``T+N`` trading sessions after the valuation date, then that date (``T+10 · Fri 06‑Nov‑26``).

    The caller counts the sessions with ``clock.is_trading_day`` (``whatif.session_dates``); this only writes them.

    Parameters
    ----------
    n : int
        Trading sessions after the valuation date; 0 is the valuation date itself.
    day : datetime.date
        The date of session ``n``.

    Returns
    -------
    str
        The label, with non-breaking hyphens in the date.

    Examples
    --------
    >>> session_label(10, dt.date(2026, 11, 6)), session_label(0, dt.date(2026, 10, 23))
    ('T+10 · Fri 06‑Nov‑26', 'T+0 · Fri 23‑Oct‑26')
    """
    return f"T+{int(n)} · {_WEEKDAYS[day.weekday()]} {_nb_date(day)}"


def timestamp(moment: dt.datetime | None) -> str:
    """A valuation or quote time in New York: ``23‑Oct‑26 16:00 ET``.

    Parameters
    ----------
    moment : datetime.datetime
        A time-zone-aware time, converted to New York, or a naive one, read as New York wall time (as the app's
        ``asof`` holds it).

    Returns
    -------
    str
        The time, with non-breaking hyphens in the date; ``–`` when missing.

    Examples
    --------
    >>> timestamp(dt.datetime(2026, 10, 23, 16, 0))
    '23‑Oct‑26 16:00 ET'
    >>> timestamp(dt.datetime(2026, 10, 23, 20, 0, tzinfo=dt.timezone.utc))
    '23‑Oct‑26 16:00 ET'
    """
    if moment is None:
        return MISSING
    if moment.tzinfo is not None and moment.utcoffset() is not None:
        moment = clock.to_new_york(moment)
    return f"{_nb_date(moment.date())} {moment:%H:%M} ET"


def probability_text(p: float, basis: str | None = None, *, sure: bool = False, never: bool = False) -> str:
    """A probability in percent with one decimal, never rounded to a certainty the payoff does not have, tagged with
    its basis.

    ``sure`` says the event cannot fail (a position that cannot lose) and ``never`` that it cannot happen. Otherwise a
    probability that rounds (or saturates in floating point) to 100% or 0% reads ``>99.9%`` or ``<0.1%``.

    Parameters
    ----------
    p : float
        The probability, 0 to 1.
    basis : {"RN smile", "flat vol", "view"}, optional
        What it is computed from (:data:`PROBABILITY_BASES`), written after the number. Leave it out only where the
        basis is stated next to the number.
    sure, never : bool, default False
        The payoff makes the event certain, or impossible.

    Returns
    -------
    str
        ``73.5% RN smile`` or ``>99.9%``; ``–`` when missing.

    Raises
    ------
    ValueError
        If ``basis`` is not one of :data:`PROBABILITY_BASES`.

    Examples
    --------
    >>> probability_text(0.7349, "RN smile"), probability_text(0.99962), probability_text(1.0, sure=True)
    ('73.5% RN smile', '>99.9%', '100.0%')
    """
    if basis is not None and basis not in PROBABILITY_BASES:
        raise ValueError(f"basis must be one of {PROBABILITY_BASES}, got {basis!r}")
    if _absent(p):
        return MISSING
    if p >= 0.9995 and not sure:
        text = ">99.9%"
    elif p <= 0.0005 and not never:
        text = "<0.1%"
    else:
        text = f"{p:.1%}"
    return f"{text} {basis}" if basis else text
