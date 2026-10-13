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


# --- Regression tests for the third review round ------------------------------------------------------------------


@pytest.mark.parametrize("method", ["crr", "lr"])
@pytest.mark.parametrize("sigma", [12.0, 16.0])
def test_calls_stay_exact_when_the_lattice_has_to_be_capped(method, sigma):
    # sigma^2 T of 1,440+ puts the share-measure mass of the call beyond the e^700 node cap: the capped call
    # lattice used to return 67.83 (CRR) / 46.36 (LR) at sigma = 12 and ~1e-23 at sigma = 16, against 100.
    value = bn.price(100, 100, 10, 0.05, sigma, 0, "call", 2000, method=method)
    assert value == pytest.approx(bs.price(100, 100, 10, 0.05, sigma), abs=1e-6)


@pytest.mark.parametrize("sigma", [6.0, 8.0])
def test_american_call_on_a_wide_lattice_respects_the_spot_bound(sigma):
    # Used to be 173.79 and 108.46, above the spot of 100 (an arbitrage). By put-call symmetry it equals the put
    # with spot and strike, and rate and dividend yield, swapped.
    call = bn.price(100, 100, 10, 0.05, sigma, 0.02, "call", 2000, american=True, method="crr")
    assert call <= 100.0
    assert call == pytest.approx(bn.price(100, 100, 10, 0.02, sigma, 0.05, "put", 2000, True, "crr"), rel=1e-12)


def _direct_crr_call(S, K, T, r, sigma, q, n, american):
    """Textbook CRR call induction on the call payoff (no symmetry), with the step-1 and step-2 node values."""
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    p = (np.exp((r - q) * dt) - 1 / u) / (u - 1 / u)
    j = np.arange(n + 1)
    values = np.maximum(S * u ** (2 * j - n) - K, 0.0)
    levels = {}
    for i in range(n - 1, -1, -1):
        values = np.exp(-r * dt) * (p * values[1:] + (1 - p) * values[:-1])
        if american:
            values = np.maximum(values, S * u ** (2 * np.arange(i + 1) - i) - K)
        levels[i] = values
    return levels


@pytest.mark.parametrize("american", [False, True])
@pytest.mark.parametrize(("S", "q"), [(90, 0.0), (100, 0.04), (115, 0.08)])
def test_call_by_symmetry_equals_the_direct_call_lattice(S, q, american):
    levels = _direct_crr_call(S, 100, 0.75, 0.05, 0.3, q, 300, american)
    res = bn.price_with_greeks(S, 100, 0.75, 0.05, 0.3, q, "call", 300, american, "crr")
    u = np.exp(0.3 * np.sqrt(0.75 / 300))
    assert res.price == pytest.approx(levels[0][0], rel=1e-11)
    assert res.delta == pytest.approx((levels[1][1] - levels[1][0]) / (S * (u - 1 / u)), rel=1e-9)


def test_tree_vega_rho_and_dividend_rho_match_black_scholes_for_european_exercise():
    for kind in ("call", "put"):
        res = bn.price_with_greeks(100, 100, 1, 0.05, 0.2, 0.03, kind, 1001, method="lr")
        g = bs.greeks(100, 100, 1, 0.05, 0.2, 0.03, kind)
        for name in ("vega", "rho", "epsilon"):
            assert getattr(res, name) == pytest.approx(g[name], rel=1e-4), (kind, name)


@pytest.mark.parametrize(("method", "steps"), [("lr", 1001), ("crr", 1000)])
def test_american_tree_vega_and_rho_include_early_exercise(method, steps):
    # References: central differences on an 8,001-step Leisen-Reimer tree. The European values (vega 0.3407 and
    # rho -0.8868 per point) are several times larger, which is why the tree table needs its own.
    res = bn.price_with_greeks(100, 120, 1, 0.05, 0.2, 0, "put", steps, True, method)
    assert res.vega / 100 == pytest.approx(0.12317, abs=1e-3)
    assert res.rho / 100 == pytest.approx(-0.14823, abs=1e-3)
    assert res.epsilon / 100 == pytest.approx(0.12661, abs=1e-3)
    # Deep enough in the money to exercise now: the value is the intrinsic 40 whatever the volatility or rates.
    deep = bn.price_with_greeks(100, 140, 1, 0.05, 0.2, 0, "put", steps, True, method)
    assert deep.price == pytest.approx(40.0) and (deep.vega, deep.rho, deep.epsilon) == (0.0, 0.0, 0.0)


def test_bumped_greeks_fall_back_to_a_one_sided_difference_where_the_bumped_tree_is_invalid():
    # CRR needs sigma > |r - q| sqrt(dt) = 0.0949 here: 0.097 is valid, 0.097 - 0.005 is not.
    args = (100, 100, 1, 0.30)
    res = bn.price_with_greeks(*args, 0.097, 0, "call", 10, False, "crr")
    up = bn.price(*args, 0.102, 0, "call", 10, False, "crr")
    assert res.vega == pytest.approx((up - res.price) / 0.005, rel=1e-12)


@pytest.mark.parametrize("method", ["crr", "lr"])
@pytest.mark.parametrize("S", [60, 100, 130])
def test_exercise_boundary_of_an_american_put_is_monotone_and_smooth(method, S):
    # The raw highest-exercised-node boundary alternated between the interleaved grids of even and odd steps
    # (454 direction reversals in 477 points for S = 100). The exact boundary rises monotonically to the strike.
    steps = 501
    times, spots = bn.price_with_greeks(S, 100, 1, 0.05, 0.2, 0, "put", steps, True, method).exercise_boundary
    assert len(times) > 400 and np.all(np.diff(times) > 0)
    assert np.all(np.diff(spots) >= 0.0)
    spacing = 100 * 2 * 0.2 * np.sqrt(1 / steps)  # node spacing near the strike
    assert np.median(np.diff(spots)) < 0.02 * spacing
    assert np.all(spots <= 100.0)
    # A 20,000-step Leisen-Reimer tree puts the critical price one year before expiry at 80.92.
    assert spots[0] == pytest.approx(80.92, abs=1.0)


def test_exercise_boundary_of_an_american_call_with_dividends_falls_towards_the_strike():
    times, spots = bn.price_with_greeks(100, 100, 1, 0.05, 0.3, 0.04, "call", 501, True, "lr").exercise_boundary
    assert len(times) > 400
    assert np.all(np.diff(spots) <= 0.0) and np.all(spots >= 100.0)
    assert spots[-1] == pytest.approx(125.0, abs=1.5)  # max(K, rK/q) = 125 at expiry


@pytest.mark.parametrize(("S", "K", "T", "r", "sigma", "q", "kind", "value"), AMERICAN_REFERENCE)
def test_extrapolated_price_matches_converged_references(S, K, T, r, sigma, q, kind, value):
    assert bn.extrapolated_price(S, K, T, r, sigma, q, kind) == pytest.approx(value, abs=2e-5)
    assert bn.extrapolated_price(S, K, T, r, sigma, q, kind, american=False) == bs.price(S, K, T, r, sigma, q, kind)


# --- Regression tests for the fourth review round -----------------------------------------------------------------


def _perpetual_american(S, K, r, q, sigma, option_type):
    """Perpetual American option (Merton, 1973): an upper bound for the same option with any finite expiry."""
    a = (r - q) / sigma**2 - 0.5
    root = np.sqrt(a**2 + 2 * r / sigma**2)
    power = -a + root if option_type == "call" else -a - root
    boundary = power * K / (power - 1)
    return bs.option_sign(option_type) * (boundary - K) * (S / boundary) ** power


@pytest.mark.parametrize("method", ["crr", "lr"])
@pytest.mark.parametrize(
    ("sigma", "q", "option_type"), [(5.5, 0.02, "call"), (6.0, 0.02, "call"), (8.0, 0.02, "call"), (6.0, 0.0, "put")]
)
def test_american_value_stays_right_when_the_bottom_nodes_of_the_lattice_are_capped(method, sigma, q, option_type):
    # With ln S - sigma sqrt(T n) below about -745 the bottom spots rounded to zero, and the American induction
    # carried the zeros back to the root: exercising was then worth the whole strike at every node, 99.99 for the
    # call (100 e^(-q dt)) and 99.975 for the put, both above the perpetual American value.
    bound = _perpetual_american(100, 100, 0.05, q, sigma, option_type)
    values = [bn.price(100, 100, 10, 0.05, sigma, q, option_type, n, True, method) for n in (1000, 1500, 2000)]
    assert max(values) < bound
    assert np.all(np.diff(values) > 0) and np.ptp(values) < 0.05  # converging towards the bound, not jumping


def test_american_call_on_a_capped_lattice_matches_its_converged_value():
    # Perpetual bound 99.1388; the 500-step tree (not capped) gives 99.09.
    assert bn.price(100, 100, 10, 0.05, 6.0, 0.02, "call", 2000, True, "crr") == pytest.approx(99.13, abs=0.01)


@pytest.mark.parametrize("sigma", [4.0, 4.5, 5.0])
@pytest.mark.parametrize(("q", "option_type"), [(0.0, "put"), (0.02, "call")])
def test_extrapolated_price_on_a_wide_lattice_stays_close_to_the_tree(sigma, q, option_type):
    # Inside the sidebar's range (vol up to 500%, 10 years) the 4,003-step tree hit the zero-spot bug and the
    # extrapolation amplified it: the put's "Converged" line read 103.71 at sigma = 4, above its strike of 100.
    value = bn.extrapolated_price(100, 100, 10, 0.05, sigma, q, option_type)
    assert value == pytest.approx(bn.price(100, 100, 10, 0.05, sigma, q, option_type, 2001, True, "lr"), abs=0.02)
    assert value < 100.0


@pytest.mark.parametrize("S", [30.0, 300.0])
def test_extrapolated_price_stays_within_the_no_arbitrage_bounds(S, monkeypatch):
    # Without early exercise (a call with no dividends) the American value is the European one; the extrapolation
    # of two trees used to land up to 4e-8 below it.
    assert bn.extrapolated_price(S, 100, 1, 0.0, 0.5, 0, "call") == bs.price(S, 100, 1, 0.0, 0.5, 0, "call")
    # A finer tree that overshot would not lift the value above the strike (put) or the spot (call).
    monkeypatch.setattr(bn, "price", lambda *args: 2.0 * args[0] if args[7] > 2001 else 0.9 * args[0])
    assert bn.extrapolated_price(S, 100, 1, 0.05, 0.3, 0.0, "call") == S
