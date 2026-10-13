"""Number formatting for the user interface."""

from __future__ import annotations

import math


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
