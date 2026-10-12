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


# Converged American reference values: Richardson extrapolation of Leisen-Reimer trees with 9,999 and 19,999
# steps (first-order convergence with early exercise), cross-checked against a 20,000-step CRR tree and the
# published 6.0904 for the first case. Year fractions follow ACT/365 dates: "6 months" is 182/365.
AMERICAN_REFERENCE = [
    # S, K, T, r, sigma, q, type, value
    (100, 100, 365 / 365, 0.05, 0.20, 0.00, "put", 6.090371),
    (100, 110, 182 / 365, 0.05, 0.30, 0.00, "put", 13.380688),
    (100, 90, 730 / 365, 0.08, 0.25, 0.00, "put", 4.905740),
    (100, 100, 365 / 365, 0.05, 0.20, 0.04, "call", 8.118240),
    (100, 100, 365 / 365, -0.01, 0.20, 0.00, "call", 7.568545),
    (40, 45, 91 / 365, 0.03, 0.40, 0.02, "put", 6.358318),
    (50, 50, 5 / 12, 0.10, 0.40, 0.00, "put", 4.284216),  # Hull, Example 21.1 (4.28)
]


@pytest.mark.parametrize(("S", "K", "T", "r", "sigma", "q", "kind", "value"), AMERICAN_REFERENCE)
def test_american_prices_match_converged_references(S, K, T, r, sigma, q, kind, value):
    assert bn.price(S, K, T, r, sigma, q, kind, 2001, american=True, method="lr") == pytest.approx(value, abs=3e-4)
    assert bn.price(S, K, T, r, sigma, q, kind, 2000, american=True, method="crr") == pytest.approx(value, abs=1e-3)


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


# --- Regression tests for findings of the independent review of the rebuilt engine ---------------


@pytest.mark.parametrize("method", ["crr", "lr"])
@pytest.mark.parametrize("american", [False, True])
def test_extremely_wide_lattice_stays_finite(method, american):
    # sigma sqrt(T n) = 5 * sqrt(10 * 2000) puts the top node beyond e^709: it used to overflow to inf/NaN.
    res = bn.price_with_greeks(100, 100, 10, 0.05, 5.0, 0, "call", 2000, american, method)
    assert res.price == pytest.approx(bs.price(100, 100, 10, 0.05, 5.0), abs=1e-6)
    assert all(np.isfinite([res.delta, res.gamma, res.theta]))


@pytest.mark.parametrize("days", [1, 3, 7, 30])
def test_leisen_reimer_theta_matches_analytic_for_short_expiries(days):
    T = days / 365
    res = bn.price_with_greeks(100, 100, T, 0.05, 0.2, 0, "call", 501, method="lr")
    assert res.theta == pytest.approx(bs.greeks(100, 100, T, 0.05, 0.2)["theta"], rel=1e-3)


def test_american_leisen_reimer_theta_matches_crr():
    lr = bn.price_with_greeks(100, 100, 7 / 365, 0.05, 0.3, 0, "put", 1001, True, "lr").theta
    crr = bn.price_with_greeks(100, 100, 7 / 365, 0.05, 0.3, 0, "put", 4000, True, "crr").theta
    assert lr == pytest.approx(crr, rel=2e-3)


def test_exercise_boundary_ignores_levels_entirely_inside_the_exercise_region():
    # Deep in-the-money put: the early levels are all exercised, and their top node is not the boundary.
    res = bn.price_with_greeks(60, 100, 1, 0.05, 0.2, 0, "put", 501, True, "crr")
    assert res.price == pytest.approx(40.0)
    _, spots = res.exercise_boundary
    assert spots.min() > 78  # the critical price a year from expiry is about 81; the old output started at 60


def test_leisen_reimer_accepts_the_maximum_step_count():
    res = bn.price_with_greeks(100, 100, 1, 0.05, 0.2, steps=20_000, method="lr")
    assert res.steps == 19_999
    assert res.price == pytest.approx(bs.price(100, 100, 1, 0.05, 0.2), abs=1e-7)


def test_zero_volatility_american_greeks_reflect_immediate_exercise():
    res = bn.price_with_greeks(80, 100, 1, 0.05, 0.0, 0, "put", 500, True, "crr")
    assert (res.price, res.delta, res.gamma, res.theta) == (20.0, -1.0, 0.0, 0.0)
