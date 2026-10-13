import itertools

import numpy as np
import pytest

from options_analyzer import black_scholes as bs
from options_analyzer.implied_vol import (
    ImpliedVolError,
    american_bounds,
    implied_volatility,
    implied_volatility_american,
    lowest_tree_volatility,
    no_arbitrage_bounds,
)

# 2,520 cases: expiries from 17 hours to 5 years, volatilities from 1% to 300%, negative to positive rates.
GRID = list(
    itertools.product(
        [50, 70, 90, 100, 110, 130, 200],  # strike (spot = 100)
        [0.002, 0.01, 0.05, 0.25, 1.0, 5.0],  # years
        [0.01, 0.05, 0.3, 1.5, 3.0],  # volatility
        [-0.01, 0.0, 0.05],  # rate
        [0.0, 0.03],  # dividend yield
        ["call", "put"],
    )
)


def test_grid_size():
    assert len(GRID) == 2_520


@pytest.mark.parametrize(("K", "T", "sigma", "r", "q", "kind"), GRID)
def test_round_trip(K, T, sigma, r, q, kind):
    price = bs.price(100, K, T, r, sigma, q, kind)
    iv = implied_volatility(price, 100, K, T, r, q, kind)
    # Where vega is negligible the volatility is not identifiable; the price must still be reproduced.
    assert abs(iv - sigma) < 1e-6 or abs(bs.price(100, K, T, r, iv, q, kind) - price) < 1e-10


@pytest.mark.parametrize(
    ("T", "sigma"),
    [(0.005, 0.25), (0.25, 0.03), (1.0, 1.5)],  # cases the old solver got wrong (0.30, floor 5%, cap 100%)
)
def test_previous_failure_cases(T, sigma):
    price = bs.price(100, 100, T, 0.05, sigma)
    assert implied_volatility(price, 100, 100, T, 0.05) == pytest.approx(sigma, abs=1e-8)


def test_arbitrage_violations_raise():
    lower, upper = no_arbitrage_bounds(100, 90, 1, 0.05, 0.0, "call")
    with pytest.raises(ImpliedVolError, match="below"):
        implied_volatility(lower - 0.01, 100, 90, 1, 0.05)
    with pytest.raises(ImpliedVolError, match="upper bound"):
        implied_volatility(upper, 100, 90, 1, 0.05)
    with pytest.raises(ImpliedVolError, match="expired"):
        implied_volatility(5.0, 100, 90, 0.0, 0.05)


def test_zero_time_value_gives_zero_vol():
    lower, _ = no_arbitrage_bounds(100, 80, 1, 0.05, 0.0, "call")
    assert implied_volatility(lower, 100, 80, 1, 0.05) == 0.0


def test_tiny_but_positive_price_is_solved():
    price = bs.price(100, 250, 0.1, 0.05, 0.4, option_type="call")
    assert 0 < price < 1e-10
    assert implied_volatility(price, 100, 250, 0.1, 0.05) == pytest.approx(0.4, abs=1e-6)


AMERICAN = list(itertools.product([80, 100, 125], [0.1, 1.0], [0.1, 0.3, 0.8], ["call", "put"], [0.0, 0.04]))


@pytest.mark.parametrize(("K", "T", "sigma", "kind", "q"), AMERICAN)
def test_american_round_trip(K, T, sigma, kind, q):
    from options_analyzer import binomial

    price = binomial.price(100, K, T, 0.05, sigma, q, kind, 501, american=True, method="lr")
    iv = implied_volatility_american(price, 100, K, T, 0.05, q, kind)
    reproduced = binomial.price(100, K, T, 0.05, max(iv, 1e-6), q, kind, 501, american=True, method="lr")
    # Quotes equal to the zero-volatility value (e.g. immediate exercise) are not identifiable: IV = 0.
    assert abs(iv - sigma) < 1e-5 or (iv == 0.0 and reproduced == pytest.approx(price, abs=1e-9))


def test_american_bounds_reject_impossible_quotes():
    with pytest.raises(ImpliedVolError, match="immediate exercise"):
        implied_volatility_american(5.0, 100, 90, 1, 0.05, 0, "call")
    with pytest.raises(ImpliedVolError, match="upper bound"):
        implied_volatility_american(150.0, 100, 90, 1, 0.05, 0, "call")


@pytest.mark.parametrize(
    ("S", "K", "r", "q", "kind"),
    [(50, 100, -0.05, 0.0, "put"), (100, 50, 0.0, -0.05, "call"), (30, 100, -0.03, 0.02, "put")],
)
def test_american_quote_above_the_strike_with_a_negative_rate_is_solved(S, K, r, q, kind):
    # With a negative rate a put is worth up to K e^(-rT), more than its strike (and a call with a negative
    # dividend yield up to S e^(-qT)). The bounds used to stop at K (or S): at S = 50, K = 100, ten years and
    # r = -5% the lower bound 114.8721 was above the upper bound 100, and the tree's own value 115.5045 was
    # rejected as "at or above the American upper bound 100".
    from options_analyzer import binomial

    price = binomial.price(S, K, 10, r, 0.2, q, kind, 501, american=True, method="lr")
    lower, upper = american_bounds(S, K, 10, r, q, kind)
    assert lower <= price < upper
    assert implied_volatility_american(price, S, K, 10, r, q, kind) == pytest.approx(0.2, abs=1e-7)


@pytest.mark.parametrize(("r", "q"), [(-0.05, 0.0), (-0.02, 0.03), (0.0, -0.05), (0.05, -0.02), (0.05, 0.0)])
@pytest.mark.parametrize("S", [1, 50, 100, 200])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_american_bounds_are_ordered_and_contain_the_tree_value(S, r, q, kind):
    from options_analyzer import binomial

    lower, upper = american_bounds(S, 100, 10, r, q, kind)
    assert lower <= upper
    # The upper bound is the strike (or spot) received when exercise is worth most: at once, or at expiry
    # when the rate (or dividend yield) is negative.
    if kind == "put":
        assert upper == pytest.approx(100 * max(1.0, np.exp(-r * 10)), rel=1e-15)
    else:
        assert upper == pytest.approx(S * max(1.0, np.exp(-q * 10)), rel=1e-15)
    for sigma in (0.2, 1.0, 3.0):
        value = binomial.price(S, 100, 10, r, sigma, q, kind, 501, american=True, method="lr")
        assert lower - 1e-9 * upper <= value <= upper * (1 + 1e-12)


@pytest.mark.parametrize("sigma", [16.5, 18.0, 19.9])
def test_volatilities_up_to_the_cap_are_solved(sigma):
    price = bs.price(100, 100, 0.01, 0, sigma, 0, "call")
    assert implied_volatility(price, 100, 100, 0.01, 0, 0, "call") == pytest.approx(sigma, rel=1e-9)


@pytest.mark.parametrize("quote", [1e-6, 1e-5, 3e-5])
def test_tiny_time_value_is_reproduced(quote):
    iv = implied_volatility(quote, 100, 100, 1.0, 0, 0, "call")
    assert bs.price(100, 100, 1.0, 0, iv, 0, "call") == pytest.approx(quote, rel=1e-6)


def test_american_volatility_up_to_the_cap_is_solved():
    from options_analyzer import binomial

    price = binomial.price(100, 100, 0.1, 0.05, 17.0, 0, "call", 501, american=True, method="lr")
    assert implied_volatility_american(price, 100, 100, 0.1, 0.05, 0, "call") == pytest.approx(17.0, rel=1e-6)


@pytest.mark.parametrize(("method", "steps"), [("crr", 101), ("crr", 1000), ("lr", 101), ("lr", 1001)])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_american_iv_on_a_given_tree_gives_back_its_volatility(method, steps, kind):
    # The Implied Volatility tab inverts on the sidebar's tree, so the headline American price maps back to the
    # sidebar volatility (a CRR price inverted on the 501-step Leisen-Reimer tree used to give 20.009%).
    from options_analyzer import binomial

    price = binomial.price(100, 100, 1, 0.05, 0.2, 0.02, kind, steps, american=True, method=method)
    iv = implied_volatility_american(price, 100, 100, 1, 0.05, 0.02, kind, steps=steps, method=method)
    assert iv == pytest.approx(0.2, abs=1e-7)


def test_crr_american_iv_searches_only_volatilities_the_tree_can_represent():
    from options_analyzer import binomial

    lowest = lowest_tree_volatility(1, 0.30, 0.0, 10, "crr")
    assert lowest == pytest.approx(0.30 * np.sqrt(0.1), rel=1e-5)
    assert lowest_tree_volatility(1, 0.30, 0.0, 10, "lr") == 1e-6
    price = binomial.price(100, 100, 1, 0.30, 0.15, 0, "put", 10, american=True, method="crr")
    assert implied_volatility_american(price, 100, 100, 1, 0.30, 0, "put", steps=10, method="crr") == pytest.approx(
        0.15, abs=1e-7
    )
    # Below the value at the lowest volatility the tree can represent (1.5e-6 here): reported, not a crash.
    with pytest.raises(ImpliedVolError, match="CRR tree can represent"):
        implied_volatility_american(1e-7, 100, 100, 1, 0.30, 0, "put", steps=10, method="crr")
