"""The illustrative FINRA 4210(g)-style portfolio margin (options_analyzer/margin.py).

The P&L at the valuation points is checked against a Black-Scholes-Merton closed form written here (scipy's normal
CDF), not the package's pricer, and the rule's numbers against the rule's text: +6% / -8% for a high-cap broad-based
index, +/-10% and +/-15% for the other classes ((g)(2)(F)), and $0.375 times the multiplier per contract
((g)(7)(B)), read from https://www.finra.org/rules-guidance/rulebooks/finra-rules/4210 on 2026-10-31.
"""

from __future__ import annotations

import doctest
import math

import numpy as np
import pytest
from scipy.stats import norm

from options_analyzer import margin
from options_analyzer.strategies import Leg, Market, Strategy, build_template

#: An SPX-like market: the default index's carry at the 20-Nov-26 expiry, rounded.
SPX = Market(5800.0, 0.0475, 0.155, 28 / 365, 0.013)


def _bsm(kind: str, s: float, k: float, t: float, r: float, q: float, sigma: float) -> float:
    """Black-Scholes-Merton with a continuous yield, from the textbook formula."""
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if kind == "call":
        return s * math.exp(-q * t) * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)
    return k * math.exp(-r * t) * norm.cdf(-d2) - s * math.exp(-q * t) * norm.cdf(-d1)


def _value(legs: list[tuple[str, float, float]], spot: float, market: Market) -> float:
    return sum(
        q * _bsm(kind, spot, k, market.expiry, market.rate, market.dividend_yield, market.volatility)
        for kind, q, k in legs
    )


@pytest.mark.parametrize(
    ("product_class", "low", "high"),
    [("high_cap_index", -8.0, 6.0), ("broad_index", -10.0, 10.0), ("equity", -15.0, 15.0)],
)
def test_ten_equidistant_points_over_the_rules_range(product_class, low, high):
    points = 100 * margin.valuation_points(product_class)
    assert points.size == 10 and points[0] == pytest.approx(low, abs=1e-12) and points[-1] == pytest.approx(high)
    # Equidistant: every step is the range over nine (the rounding of linspace, a few ulps).
    np.testing.assert_allclose(np.diff(points), (high - low) / 9, rtol=1e-12)


def test_an_unknown_product_class_is_refused():
    with pytest.raises(ValueError, match="product_class"):
        margin.valuation_points("spx")
    with pytest.raises(ValueError, match="product_class"):
        margin.portfolio_margin(Strategy("c", (Leg("call", 1, 5800),)), SPX, product_class="index")


def test_the_pnl_at_each_point_is_the_closed_form_revaluation():
    # A bull call spread, 3 contracts of 100: every point's P&L is the closed form at the moved spot minus today's
    # value, to 1e-9 of the position's notional scale (both sides are double-precision BSM; measured gap 1e-11).
    legs = [("call", 1.0, 5815.0), ("call", -1.0, 5915.0)]
    spread = Strategy("spread", tuple(Leg(kind, q, k) for kind, q, k in legs))
    result = margin.portfolio_margin(spread, SPX, contracts=3, multiplier=100)
    today = _value(legs, SPX.spot, SPX)
    expected = [300 * (_value(legs, SPX.spot * (1 + x), SPX) - today) for x in result.points]
    np.testing.assert_allclose(result.pnl, expected, rtol=0, atol=1e-7)
    assert result.worst == pytest.approx(min(expected), abs=1e-7)
    assert result.worst_point == pytest.approx(-0.08)  # a long call spread loses most at the largest fall
    # Six contracts at $37.50, the long ones each worth more than that.
    assert result.minimum == pytest.approx(6 * 37.5)
    assert result.requirement == pytest.approx(-min(expected), abs=1e-7)


def test_a_short_put_is_charged_its_loss_at_minus_eight_percent():
    put = Strategy("short put", (Leg("put", -1, 5800),))
    result = margin.portfolio_margin(put, SPX)
    loss = 100 * (_value([("put", -1.0, 5800.0)], 5800 * 0.92, SPX) - _value([("put", -1.0, 5800.0)], 5800, SPX))
    assert result.worst == pytest.approx(loss, abs=1e-7)
    assert result.requirement == pytest.approx(-loss, abs=1e-7) and result.requirement > result.minimum


def test_the_minimum_is_0_375_times_the_multiplier_per_contract_and_at_most_a_long_contracts_value():
    condor = build_template("Iron Condor", 5800, 100, 55 / 365)
    assert margin.minimum_requirement(condor, SPX) == pytest.approx(4 * 37.5)
    assert margin.minimum_requirement(condor, SPX, contracts=5, multiplier=10) == pytest.approx(20 * 0.375 * 10)
    # A long call far out of the money is worth less than $37.50 a contract: it is charged its value.
    far = Leg("call", 2, 6700)
    value = _bsm("call", 5800, 6700, SPX.expiry, SPX.rate, SPX.dividend_yield, SPX.volatility)
    assert 0 < 100 * value < 37.5
    assert margin.minimum_requirement(Strategy("far", (far,)), SPX) == pytest.approx(2 * 100 * value, rel=1e-9)
    # Shares carry no minimum.
    assert margin.minimum_requirement(Strategy("stock", (Leg("stock", 100),)), SPX) == 0.0


def test_a_riskless_position_is_charged_the_minimum():
    # A conversion (long one share per unit, long put, short call at one strike) has the same value at every spot
    # without dividends (with a yield q its delta is 1 - e^(-qT)): no point loses, so the requirement is the minimum.
    conversion = Strategy("conversion", (Leg("stock", 1), Leg("put", 1, 5800), Leg("call", -1, 5800)))
    result = margin.portfolio_margin(conversion, Market(5800.0, 0.0475, 0.155, 28 / 365))
    assert np.max(np.abs(result.pnl)) < 1e-6
    assert result.requirement == pytest.approx(result.minimum) == pytest.approx(2 * 37.5)


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_the_sizes_must_be_finite(bad):
    with pytest.raises(ValueError, match="contracts"):
        margin.portfolio_margin(Strategy("c", (Leg("call", 1, 5800),)), SPX, contracts=bad)
    with pytest.raises(ValueError, match="multiplier"):
        margin.minimum_requirement(Strategy("c", (Leg("call", 1, 5800),)), SPX, multiplier=bad)


def test_the_docstring_examples_hold():
    # The suite runs without --doctest-modules: every example line runs here.
    result = doctest.testmod(margin)
    assert result.failed == 0 and result.attempted == 12
