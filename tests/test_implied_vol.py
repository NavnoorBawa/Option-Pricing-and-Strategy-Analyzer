import itertools

import pytest

from options_analyzer import black_scholes as bs
from options_analyzer.implied_vol import ImpliedVolError, implied_volatility, no_arbitrage_bounds

GRID = list(
    itertools.product(
        [50, 80, 100, 120, 200],  # strike (spot = 100)
        [0.002, 0.01, 0.25, 1.0, 5.0],  # years
        [0.01, 0.05, 0.3, 1.5, 3.0],  # volatility
        [-0.01, 0.05],  # rate
        [0.0, 0.03],  # dividend yield
        ["call", "put"],
    )
)


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
