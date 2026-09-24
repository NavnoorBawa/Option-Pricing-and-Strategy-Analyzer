import numpy as np
import pytest

from options_analyzer import binomial as bn
from options_analyzer import black_scholes as bs


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_leisen_reimer_converges_to_black_scholes(option_type):
    ref = bs.price(100, 105, 1.0, 0.05, 0.2, 0.01, option_type)
    assert bn.price(100, 105, 1.0, 0.05, 0.2, 0.01, option_type, 101, method="lr") == pytest.approx(ref, abs=1e-4)
    assert bn.price(100, 105, 1.0, 0.05, 0.2, 0.01, option_type, 1001, method="lr") == pytest.approx(ref, abs=1e-6)


def test_crr_converges_with_known_rate():
    ref = bs.price(100, 100, 1.0, 0.05, 0.2)
    errors = [abs(bn.price(100, 100, 1.0, 0.05, 0.2, 0, "call", n) - ref) for n in (100, 1000)]
    assert errors[1] < errors[0] and errors[1] < 5e-3


def test_leisen_reimer_rounds_to_odd_steps():
    assert bn.price(100, 95, 0.5, 0.03, 0.3, steps=200, method="lr") == bn.price(
        100, 95, 0.5, 0.03, 0.3, steps=201, method="lr"
    )


# American reference values from QuantLib's FdBlackScholesVanillaEngine (800 x 800 grid). QuantLib derives
# the year fraction from calendar dates with ACT/365, so "6 months" is 182/365 and "3 months" 91/365.
QUANTLIB_AMERICAN = [
    # S, K, T, r, sigma, q, type, value
    (100, 100, 365 / 365, 0.05, 0.20, 0.00, "put", 6.08962),
    (100, 110, 182 / 365, 0.05, 0.30, 0.00, "put", 13.38007),
    (100, 90, 730 / 365, 0.08, 0.25, 0.00, "put", 4.90432),
    (100, 100, 365 / 365, 0.05, 0.20, 0.04, "call", 8.11824),
    (100, 100, 365 / 365, -0.01, 0.20, 0.00, "call", 7.56837),
    (40, 45, 91 / 365, 0.03, 0.40, 0.02, "put", 6.35827),
]


@pytest.mark.parametrize(("S", "K", "T", "r", "sigma", "q", "kind", "value"), QUANTLIB_AMERICAN)
def test_american_prices_match_quantlib(S, K, T, r, sigma, q, kind, value):
    assert bn.price(S, K, T, r, sigma, q, kind, 1001, american=True, method="lr") == pytest.approx(value, abs=2e-3)


def test_american_put_matches_hull_example():
    # Hull Example 21.1 (S=50, K=50, r=10%, sigma=40%, 5 months): 4.28 with many steps.
    assert bn.price(50, 50, 5 / 12, 0.10, 0.40, 0, "put", 2000, american=True) == pytest.approx(4.284, abs=2e-3)


def test_american_call_without_dividends_equals_european():
    euro = bs.price(100, 100, 1, 0.05, 0.25, 0.0, "call")
    assert bn.price(100, 100, 1, 0.05, 0.25, 0.0, "call", 1001, american=True, method="lr") == pytest.approx(
        euro, abs=1e-5
    )


def test_american_at_least_european():
    for kind in ("call", "put"):
        euro = bn.price(100, 100, 1, 0.05, 0.3, 0.06, kind, 501, american=False, method="lr")
        amer = bn.price(100, 100, 1, 0.05, 0.3, 0.06, kind, 501, american=True, method="lr")
        assert amer >= euro - 1e-12


def test_tree_greeks_match_black_scholes_for_european():
    res = bn.price_with_greeks(100, 100, 1, 0.05, 0.2, 0.03, "call", 1001, method="lr")
    g = bs.greeks(100, 100, 1, 0.05, 0.2, 0.03, "call")
    assert res.delta == pytest.approx(g["delta"], abs=1e-4)
    assert res.gamma == pytest.approx(g["gamma"], abs=1e-4)
    assert res.theta == pytest.approx(g["theta"], abs=0.02)
    crr = bn.price_with_greeks(100, 100, 1, 0.05, 0.2, 0.03, "call", 1000, method="crr")
    assert crr.theta == pytest.approx(g["theta"], abs=0.02)


def test_early_exercise_boundary_for_american_put():
    res = bn.price_with_greeks(100, 100, 1, 0.05, 0.25, 0.0, "put", 400, american=True, method="crr")
    times, spots = res.exercise_boundary
    assert len(times) > 100
    assert np.all(spots < 100)  # exercise only below the strike
    assert spots[-1] > spots[0]  # the boundary rises towards the strike as expiry approaches


def test_crr_rejects_arbitrage_inputs():
    with pytest.raises(ValueError, match="not arbitrage-free"):
        bn.price(100, 100, 1, 0.30, 0.005, 0, "call", 10, method="crr")
    # Leisen-Reimer is always valid.
    assert bn.price(100, 100, 1, 0.30, 0.005, 0, "call", 11, method="lr") == pytest.approx(
        bs.price(100, 100, 1, 0.30, 0.005), abs=1e-3
    )


def test_degenerate_inputs():
    assert bn.price(105, 100, 0.0, 0.05, 0.2) == pytest.approx(5.0)
    assert bn.price(100, 100, 1.0, 0.05, 0.0) == pytest.approx(100 - 100 * np.exp(-0.05))
    # With zero volatility an American put is exercised immediately when that beats waiting.
    assert bn.price(80, 100, 1.0, 0.05, 0.0, 0, "put", american=True) == pytest.approx(20.0)
    with pytest.raises(ValueError):
        bn.price(100, 100, 1, 0.05, 0.2, steps=0)
    with pytest.raises(ValueError):
        bn.price(100, 100, 1, 0.05, 0.2, method="trinomial")
