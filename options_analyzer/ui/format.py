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


def pct(value: float, decimals: int = 2, signed: bool = False) -> str:
    """Format a decimal fraction as a percentage (0.1234 -> ``12.34%``)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return f"{value * 100:{'+' if signed else ''}.{decimals}f}%"


def md(text: str) -> str:
    """Escape dollar signs so Markdown does not render ``$a ... $b`` as LaTeX math."""
    return text.replace("$", r"\$")
