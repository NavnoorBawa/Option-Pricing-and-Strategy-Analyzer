"""The desk number-format standard in options_analyzer.ui.format."""

from __future__ import annotations

import datetime as dt
import math
import re

import numpy as np
import pandas as pd
import pytest

from options_analyzer import black_scholes as bs
from options_analyzer.ui import format as fmt
from options_analyzer.ui.format import MINUS


def _sample(n: int = 4000, seed: int = 7) -> list[float]:
    """Values across 24 orders of magnitude of both signs, the rounding boundaries and the awkward floats."""
    rng = np.random.default_rng(seed)
    magnitudes = 10.0 ** rng.uniform(-12, 12, n)
    values = list(magnitudes * rng.choice([-1.0, 1.0], n))
    for decimals in range(7):
        step = 10.0**-decimals
        values += [s * k * step for s in (-1.0, 1.0) for k in (0.4999999, 0.5, 0.5000001, 1.0, 9.5, 99.5, 999.5)]
    values += [0.0, -0.0, 5e-324, -5e-324, 1e300, -1e300, 0.005, -0.005, 0.015, -0.015, 2.675, -2.675]
    return values


# --- signed -----------------------------------------------------------------------------------------------------


def _check_signed(value: float, decimals: int) -> None:
    text = fmt.signed(value, decimals)
    plain = f"{abs(value):,.{decimals}f}"
    assert "-" not in text  # the minus sign is U+2212, never a hyphen
    assert not re.fullmatch(f"[+{MINUS}][0.,]+", text), text  # never −0.00 or +0.00
    if plain.strip("0.,"):  # visible digits: always signed, with the sign of the value
        assert text == ("+" if value > 0 else MINUS) + plain
    else:
        assert text == plain
    assert len(text) == len(fmt.signed(-value, decimals))  # signed columns line up


@pytest.mark.parametrize("decimals", [0, 1, 2, 4, 6])
def test_signed_properties_over_a_wide_sample(decimals):
    for value in _sample():
        _check_signed(value, decimals)


def test_signed_properties_with_hypothesis():
    hypothesis = pytest.importorskip("hypothesis")
    strategies = hypothesis.strategies

    @hypothesis.settings(max_examples=500, deadline=None, database=None)
    @hypothesis.given(strategies.floats(allow_nan=False, allow_infinity=False), strategies.integers(0, 8))
    def check(value, decimals):
        _check_signed(value, decimals)

    check()


def test_signed_examples():
    assert fmt.signed(1234.5) == "+1,234.50"
    assert fmt.signed(-1234.5, 2, "$") == f"{MINUS}$1,234.50"
    assert fmt.signed(2.5, 2, suffix=" vol pts") == "+2.50 vol pts"
    assert fmt.signed(-0.004) == "0.00" and fmt.signed(-0.004, 2, "$") == "$0.00"
    assert fmt.signed(0.0, 0) == "0"
    assert fmt.signed(math.inf) == "+∞" and fmt.signed(-math.inf, 2, "$") == f"{MINUS}∞"
    assert fmt.signed(math.nan) == "–" and fmt.signed(None) == "–"
    assert fmt.signed(np.float64(-3.14159), 3) == f"{MINUS}3.142"


# --- greek ------------------------------------------------------------------------------------------------------


def _significant_digits(text: str) -> int:
    number = text.split(" ")[0].lstrip(f"+{MINUS}").replace(",", "")
    mantissa = number.split("e")[0]
    return len(mantissa.replace(".", "").lstrip("0"))


@pytest.mark.parametrize("kind", list(bs.GREEKS))
def test_greek_keeps_three_significant_digits_at_any_size(kind):
    scale = bs.GREEKS[kind].scale
    for value in _sample(800, seed=hash(kind) % 1000):
        if value == 0 or not math.isfinite(value * scale) or value * scale == 0:
            continue
        text = fmt.greek(value, kind, unit=False)
        assert _significant_digits(text) >= 3, (value, text)
        assert text[0] == ("+" if value > 0 else MINUS), text
        assert "-" not in text, text
        # And the number it shows is the scaled Greek to its precision.
        shown = float(text.lstrip("+").replace(MINUS, "-").replace(",", "").replace(f"e{MINUS}", "e-"))
        assert shown == pytest.approx(value * scale, rel=5e-3), (value, text)


@pytest.mark.parametrize(("spot", "days"), [(100.0, 365), (100.0, 30), (60.0, 365), (100.0, 3650)])
def test_small_higher_order_greeks_keep_their_digits(spot, days):
    # Fixed six decimals print tiny higher-order Greeks with one or two digits, or as 0.000000 (ultima per vol
    # point is of order 1e-7); greek() keeps three significant digits and the unit.
    raw = bs.greeks(spot, 100.0, days / 365, 0.05, 0.2, 0.0, "call")
    for kind in ("vanna", "charm", "vomma", "veta", "speed", "zomma", "color", "ultima"):
        value = raw[kind]
        if value == 0:
            continue
        text = fmt.greek(value, kind)
        assert _significant_digits(text) >= 3, (kind, text)
        assert text.endswith(" " + bs.GREEKS[kind].unit)
    assert fmt.greek(raw["ultima"], "ultima", unit=False) != "+0.000000"


def test_greek_examples():
    assert fmt.greek(0.636831, "delta") == "+0.6368 per $1 move in spot"
    assert fmt.greek(-6.4155, "theta", unit=False) == f"{MINUS}0.01758"  # per calendar day
    assert fmt.greek(37.524, "vega", unit=False) == "+0.3752"  # per vol point
    assert fmt.greek(1.234e-9, "gamma", unit=False) == f"+1.234e{MINUS}9"
    assert fmt.greek(12345.678, "delta", unit=False) == "+12,346"
    assert fmt.greek(0.0, "rho") == "0 per 1% rate move"
    assert fmt.greek(math.nan, "rho", unit=False) == "–"
    assert fmt.greek(0.0123, "vanna", digits=2, unit=False) == "+0.000123"  # never fewer than three


# --- dollar Greeks ----------------------------------------------------------------------------------------------


def test_dollar_greeks_equal_their_definitions():
    rng = np.random.default_rng(11)
    for _ in range(500):
        greek, spot = rng.normal() * 10.0 ** rng.uniform(-6, 3), 10.0 ** rng.uniform(-2, 5)
        units = float(rng.integers(-10_000, 10_000))

        def close(a, b):
            return abs(a - b) <= 1e-12 * max(1.0, abs(b))

        assert close(fmt.dollar_delta(greek, spot, units), greek * spot * units)
        assert close(fmt.gamma_1pct(greek, spot, units), greek * spot * 0.01 * units)
        assert close(fmt.dollar_gamma_1pct(greek, spot, units), 0.5 * greek * (0.01 * spot) ** 2 * units)
        assert close(fmt.dollar_vega(greek, units), greek * bs.GREEKS["vega"].scale * units)
        assert close(fmt.dollar_theta(greek, units), greek * bs.GREEKS["theta"].scale * units)
        assert close(fmt.dollar_rho(greek, units), greek * bs.GREEKS["rho"].scale * units)


def test_dollar_greeks_of_a_long_call():
    # 10 contracts of 100 shares of the default at-the-money one-year call.
    g = bs.greeks(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "call")
    units = 1000.0
    assert fmt.dollar_delta(g["delta"], 100.0, units) == pytest.approx(63_683.1, abs=0.1)
    # A 1% move ($1) changes the delta by Γ·S·1% shares and earns ½Γ($1)² per share from gamma.
    assert fmt.gamma_1pct(g["gamma"], 100.0, units) == pytest.approx(18.762, abs=1e-3)
    assert fmt.dollar_gamma_1pct(g["gamma"], 100.0, units) == pytest.approx(9.381, abs=1e-3)
    assert fmt.dollar_vega(g["vega"], units) == pytest.approx(375.24, abs=0.01)
    assert fmt.dollar_theta(g["theta"], units) == pytest.approx(-17.573, abs=1e-3)
    assert fmt.dollar_rho(g["rho"], units) == pytest.approx(532.33, abs=0.01)


# --- levels -------------------------------------------------------------------------------------------------------


def _check_level(value: float) -> None:
    text = fmt.level(value)
    shown = float(text.replace(",", "").replace(MINUS, "-"))
    size = abs(value)
    if round(size, 2) >= 1:  # two decimals with separators
        assert re.fullmatch(r"\d{1,3}(,\d{3})*\.\d{2}", text), text
        assert abs(shown - size) <= 0.005 * (1 + 1e-12), (value, text)
    elif round(size, 4) >= 0.01:  # four decimals
        assert re.fullmatch(r"0\.\d{4}", text), text
        assert abs(shown - size) <= 0.00005 * (1 + 1e-9), (value, text)
    else:  # four significant digits: never 0.0000
        digits = re.sub(r"e.*|[^0-9]", "", text).lstrip("0")
        assert len(digits) >= 4, text
        assert shown == pytest.approx(size, rel=5e-4), (value, text)


def test_level_over_a_wide_sample():
    for value in _sample():
        if 0 < value < 1e15:
            _check_level(value)


def test_level_examples():
    assert [fmt.level(v) for v in (5812.374, 45123.5, 1.0, 0.996, 0.99, 0.042, 0.01, 4.0866e-5, 1.23e-7)] == [
        "5,812.37", "45,123.50", "1.00", "1.00", "0.9900", "0.0420", "0.0100", "0.00004087", f"1.230e{MINUS}7",
    ]  # fmt: skip
    assert fmt.level(0.0) == "0.0000" and fmt.level(-3.2) == f"{MINUS}3.20"
    assert fmt.level(float("nan")) == fmt.level(None) == "–" and fmt.level(math.inf) == "∞"


# --- dates, accounting, quantities --------------------------------------------------------------------------------


def test_dates_read_day_month_year_with_the_days_left():
    assert fmt.date(dt.date(2027, 10, 20)) == "20-Oct-27"
    assert fmt.date(dt.date(2030, 1, 5)) == "05-Jan-30"
    assert fmt.expiry(365, dt.date(2026, 10, 20)) == "20-Oct-27 (365d)"
    assert fmt.expiry(1, dt.date(2026, 12, 31)) == "01-Jan-27 (1d)"
    assert fmt.expiry(29.6, dt.date(2026, 2, 1)) == "03-Mar-26 (30d)"
    assert re.fullmatch(r"\d{2}-[A-Z][a-z]{2}-\d{2} \(7d\)", fmt.expiry(7))  # from the server's date


def test_accounting():
    assert fmt.accounting(1234.5) == "$1,234.50"
    assert fmt.accounting(-1234.5) == "($1,234.50)"
    assert fmt.accounting(-0.004) == "$0.00"  # never ($0.00)
    assert fmt.accounting(-1234.5, 0, "") == "(1,234)"
    assert fmt.accounting(-math.inf) == "(Unlimited)" and fmt.accounting(math.inf) == "Unlimited"
    assert fmt.accounting(math.nan) == "–"


def test_quantity():
    assert [fmt.quantity(v) for v in (10, -5, 0, 1234.4, -0.4, 2.5)] == ["+10", f"{MINUS}5", "0", "+1,234", "0", "+2"]
    assert [fmt.quantity(v, sign=False) for v in (1000, -5, 0)] == ["1,000", f"{MINUS}5", "0"]
    assert fmt.quantity(math.nan) == "–"


def test_right_aligned_overrides_the_text_alignment_of_formatted_columns():
    frame = pd.DataFrame({"Scenario": ["Crash", "Rally"], "P&L": [fmt.signed(-12.5, 2, "$"), fmt.signed(3.0, 2, "$")]})
    html = fmt.right_aligned(frame, text_columns=["Scenario"]).to_html()
    styled = re.findall(r"#(T_\w+_row\d+_col(\d+))\s*\{\s*text-align: right !important;", html)
    assert {col for _, col in styled} == {"1"}  # the P&L column only


def test_legacy_formats_are_unchanged():
    # The existing helpers keep their hyphen-minus until the tabs move to the desk standard.
    assert fmt.money(-1234.5) == "-$1,234.50" and fmt.money(12.0, signed=True) == "+$12.00"
    assert fmt.number(-0.004, 2) == "0.00" and fmt.pct(0.1234) == "12.34%"
    assert fmt.money_short(-2_345_678.9, True) == "-$2.35M"


@pytest.mark.parametrize(
    ("text", "markdown"),
    [
        ("quotes.csv", "`quotes.csv`"),
        ("http://example.com/x", "`http://example.com/x`"),
        ("a `b` c", "``a `b` c``"),
        ("`a`", "`` `a` ``"),
        ("a ``` b", "````a ``` b````"),
        ("two\n\nlines\tand  spaces ", "`two lines and spaces`"),
    ],
)
def test_md_code_quotes_text_as_one_code_span(text, markdown):
    assert fmt.md_code(text) == markdown


def test_md_code_spans_render_as_the_text_typed():
    # CommonMark: a code span's content is the text between its fences, less one leading and trailing space when both
    # are there; inside it nothing is a link, emphasis or math.
    for text in ("[a](http://b.c) *y* $x$", "www.example.com", "`a` b", "a ``` b", ":material/home:"):
        span = fmt.md_code(text)
        fence = re.match("`+", span).group()
        inner = span[len(fence) : -len(fence)]
        if inner.startswith(" ") and inner.endswith(" ") and inner.strip():
            inner = inner[1:-1]
        assert inner == text
        assert fence not in inner  # no run of backticks in the text closes the span early
