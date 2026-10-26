"""options_analyzer.american: the Andersen-Lake-Offengenden engine against QuantLib reference prices, and the
properties an American option must have (roadmap item PR-2).

The reference grid (tests/fixtures/alo_grid_qdfp.npz, written by scripts/gen_alo_reference.py) holds the 4,495
American puts of Healy (2021, Table 5) priced with QuantLib's QdFp engine, so these tests need no QuantLib;
tests/oracle/test_ql_alo.py compares with QuantLib directly.
"""

from __future__ import annotations

import base64
import io
import math
import statistics
import time
from pathlib import Path

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from options_analyzer import american as am
from options_analyzer import binomial
from options_analyzer import black_scholes as bs

FIXTURE = Path(__file__).parent / "fixtures" / "alo_grid_qdfp.npz"

#: Early-exercise premia of the example of Andersen et al. (2016): S = K = 100, q = 5 %, sigma = 25 %, one year,
#: with r = 5 % (the density form) and 7.5 % (the cdf form). From QuantLib 1.44's QdFp engine at its tightest
#: settings (tanh-sinh to 1e-14, 40 iterations, 81 nodes), which agree with the "high" scheme to 1.0e-12 and 4.6e-11.
LITERATURE_PREMIA = {0.05: 0.1069527027479, 0.075: 0.3671111406253}


def _grid() -> dict[str, np.ndarray]:
    lines = FIXTURE.read_text(encoding="ascii").splitlines()
    payload = base64.b64decode("".join(line for line in lines if not line.startswith("#")))
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


@pytest.fixture(scope="module")
def grid() -> dict[str, np.ndarray]:
    return _grid()


def _grid_prices(grid: dict[str, np.ndarray], scheme: str) -> np.ndarray:
    return np.asarray(
        am.price(
            grid["spot"],
            grid["strike"],
            grid["maturity"],
            grid["rate"],
            grid["volatility"],
            grid["dividend_yield"],
            option_type="put",
            scheme=scheme,
        )
    )


# ------------------------------------------------------------------------------------------------ reference grid


def test_the_fixture_is_healys_table_5_grid(grid):
    """4,495 puts: Table 5's parameter values, each option worth at least 0.5, as Healy (2021) describes."""
    assert grid["spot"].size == 4495
    assert float(grid["strike"]) == 100.0
    assert set(grid["rate"]) == {0.02, 0.04, 0.06, 0.08, 0.10}
    assert set(grid["dividend_yield"]) == {0.0, 0.04, 0.08, 0.12}
    assert set(grid["spot"]) == {25.0, 50.0, 80.0, 90.0, 100.0, 110.0, 120.0, 150.0, 175.0, 200.0}
    assert set(grid["maturity"]) == {1 / 12, 0.25, 0.5, 0.75, 1.0}
    assert set(grid["volatility"]) == {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
    assert grid["qdfp_high"].min() >= 0.5
    # QdFp high precision is itself accurate to about 1e-6: the strict QuantLib column differs from it by at most
    # 1.04e-6 (RMSE 5.2e-8), always near the exercise boundary.
    difference = grid["qdfp_high"] - grid["qdfp_strict"]
    assert np.abs(difference).max() < 1.1e-6
    assert np.sqrt(np.mean(difference**2)) < 6e-8


# Targets of roadmap item PR-2 against QdFp high precision (RMSE: fast 1e-4, accurate 5e-6 with max 5e-5, high
# 3e-7) and bounds with headroom over the measured errors. Measured against qdfp_high / qdfp_strict: fast RMSE
# 3.7e-5 / 3.7e-5, max 7.1e-4 / 7.1e-4; accurate 1.6e-7 / 1.7e-7, max 3.7e-6 / 4.8e-6; high 5.2e-8 / 1.8e-9, max
# 1.0e-6 / 5.4e-8. High's error against qdfp_high is that column's own error.
ACCURACY = {
    # scheme: (RMSE vs qdfp_high, max vs qdfp_high, RMSE vs qdfp_strict, max vs qdfp_strict)
    "fast": (1e-4, 1e-3, 1e-4, 1e-3),
    "accurate": (5e-6, 5e-5, 5e-7, 1e-5),
    "high": (3e-7, 1.2e-6, 1e-8, 2e-7),
}


@pytest.mark.parametrize("scheme", sorted(ACCURACY))
def test_accuracy_on_the_4495_put_grid(grid, scheme):
    rmse_high, max_high, rmse_strict, max_strict = ACCURACY[scheme]
    ours = _grid_prices(grid, scheme)
    for column, rmse, largest in (("qdfp_high", rmse_high, max_high), ("qdfp_strict", rmse_strict, max_strict)):
        error = ours - grid[column]
        assert np.sqrt(np.mean(error**2)) <= rmse, (column, np.sqrt(np.mean(error**2)))
        assert np.abs(error).max() <= largest, (column, np.abs(error).max())


@pytest.mark.parametrize("r", sorted(LITERATURE_PREMIA))
@pytest.mark.parametrize(("scheme", "tolerance"), [("fast", 2e-5), ("accurate", 1e-8), ("high", 1e-10)])
def test_the_premium_of_the_andersen_lake_offengenden_example(r, scheme, tolerance):
    details = am.price_details(100, 100, 1.0, r, 0.25, 0.05, option_type="put", scheme=scheme)
    assert details.fixed_point == ("density" if r == 0.05 else "cdf")
    assert details.european == pytest.approx(bs.price(100, 100, 1.0, r, 0.25, 0.05, "put"), rel=1e-14)
    assert abs(details.premium - LITERATURE_PREMIA[r]) <= tolerance


def _median_seconds(function, cases, repeats: int = 3) -> float:
    """The best of ``repeats`` medians of single calls (the boundary cache is cleared before each)."""
    medians = []
    for _ in range(repeats):
        times = []
        for case in cases:
            am._put_boundary.cache_clear()
            start = time.perf_counter()
            function(*case)
            times.append(time.perf_counter() - start)
        medians.append(statistics.median(times))
    return min(medians)


@pytest.mark.parametrize(("scheme", "budget"), [("fast", 5e-3), ("accurate", 10e-3)])
def test_time_per_option(grid, scheme, budget):
    """One option at a time, each solving its own boundary: a guard against slowdowns, not a benchmark.

    Measured on a 4-core container: fast 0.4 ms, accurate 0.6-0.9 ms, high 1.5 ms (benchmarks/bench_american.py); on
    GitHub's shared runners fast takes 1.0-1.4 ms. The budgets leave room for slow runners and still catch a slowdown
    of an order of magnitude.
    """
    rows = np.random.default_rng(5).choice(grid["spot"].size, size=100, replace=False)
    cases = [
        (grid["spot"][i], 100.0, grid["maturity"][i], grid["rate"][i], grid["volatility"][i], grid["dividend_yield"][i])
        for i in rows
    ]

    def one(S, K, T, r, sigma, q):
        return am.price(float(S), K, float(T), float(r), float(sigma), float(q), option_type="put", scheme=scheme)

    assert _median_seconds(one, cases) <= budget


def test_a_41_strike_strip_solves_one_boundary_within_15_ms():
    strikes = np.linspace(80.0, 120.0, 41)
    for kind in ("put", "call"):
        am._put_boundary.cache_clear()
        values = am.price(100.0, strikes, 0.5, 0.05, 0.25, 0.02, option_type=kind)
        assert am._put_boundary.cache_info().misses == 1
        one_by_one = [am.price(100.0, k, 0.5, 0.05, 0.25, 0.02, option_type=kind) for k in strikes]
        np.testing.assert_allclose(values, one_by_one, rtol=1e-14, atol=0)
    seconds = _median_seconds(lambda: am.price(100.0, strikes, 0.5, 0.05, 0.25, 0.02, option_type="put"), [()] * 5)
    assert seconds <= 15e-3  # measured: about 0.7 ms


# -------------------------------------------------------------------------------------------------- properties

rates = st.floats(-0.05, 0.2)
spots = st.floats(20.0, 300.0)
maturities = st.floats(1 / 365, 5.0)
volatilities = st.floats(0.02, 1.5)


@given(S=spots, K=spots, T=maturities, r=rates, q=rates, sigma=volatilities)
def test_put_call_symmetry(S, K, T, r, q, sigma):
    """C(S, K; r, q) = P(K, S; q, r) (McDonald and Schroder, 1998), to 1e-14 relative."""
    assume(am.exercise_regime(r, q, option_type="call") != "double")  # the tree fallback is tested below
    call = am.price(S, K, T, r, sigma, q, option_type="call")
    put = am.price(K, S, T, q, sigma, r, option_type="put")
    assert abs(call - put) <= 1e-14 * max(call, put, 1e-300)


@given(S=spots, K=spots, T=maturities, r=st.floats(0.0, 0.2), sigma=volatilities)
def test_a_call_without_dividends_is_european(S, K, T, r, sigma):
    """Early exercise of a call is never optimal when q = 0 <= r: its value is the BSM one, to 1e-12."""
    value = am.price(S, K, T, r, sigma, 0.0, option_type="call")
    assert value == pytest.approx(bs.price(S, K, T, r, sigma, 0.0, "call"), rel=1e-12, abs=1e-12)
    assert am.price_details(S, K, T, r, sigma, 0.0, option_type="call").method == "european"


@given(S=spots, K=spots, T=maturities, r=st.floats(-0.05, 0.0), extra=st.floats(0.0, 0.2), sigma=volatilities)
def test_a_put_with_r_at_most_zero_at_most_q_minus_r_is_european(S, K, T, r, extra, sigma):
    """r <= 0 <= q - r (Healy 2021, Proposition 2): the BSM value."""
    q = r + extra
    value = am.price(S, K, T, r, sigma, q, option_type="put")
    assert value == pytest.approx(bs.price(S, K, T, r, sigma, q, "put"), rel=1e-12, abs=1e-12)


@given(S=spots, K=spots, T=maturities, r=rates, q=rates, sigma=volatilities, kind=st.sampled_from(["call", "put"]))
def test_the_value_lies_between_the_no_arbitrage_bounds(S, K, T, r, q, sigma, kind):
    """max(European, intrinsic) <= American <= binomial.american_upper_bound.

    A call's European value comes from its symmetric put, so it may differ from black_scholes.price in the last
    bits of terms of order S and K (1e-14 of them).
    """
    assume(am.exercise_regime(r, q, option_type=kind) != "double")
    details = am.price_details(S, K, T, r, sigma, q, option_type=kind)
    european = bs.price(S, K, T, r, sigma, q, kind)
    intrinsic = max(bs.option_sign(kind) * (S - K), 0.0)
    assert details.value >= max(details.european, intrinsic)
    assert details.european == pytest.approx(european, rel=1e-12, abs=1e-14 * max(S, K))
    assert details.value <= binomial.american_upper_bound(S, K, T, r, q, kind)


@pytest.mark.parametrize("scheme", sorted(am.SCHEMES))
def test_the_value_is_monotone_in_volatility_expiry_and_strike(scheme):
    """Non-decreasing in sigma and T, and in K for a put (non-increasing for a call), along lines of +-10 % in
    steps of 2 % through 80 random options. Measured over 400 options and 21-point lines: no decrease larger than
    1.4e-14 (rounding where the value is flat)."""
    rng = np.random.default_rng(11)
    line = np.linspace(0.9, 1.1, 11)
    for _ in range(80):
        kind = str(rng.choice(["call", "put"]))
        K, T = 100 * math.exp(rng.uniform(-0.6, 0.6)), 10 ** rng.uniform(-2, math.log10(5))
        sigma, r, q = rng.uniform(0.05, 0.8), rng.uniform(0.0, 0.12), rng.uniform(-0.03, 0.12)
        sign = 1.0 if kind == "put" else -1.0
        lines = {
            "sigma": am.price(100, K, T, r, sigma * line, q, option_type=kind, scheme=scheme),
            "T": am.price(100, K, T * line, r, sigma, q, option_type=kind, scheme=scheme),
            "K": sign * am.price(100, K * line, T, r, sigma, q, option_type=kind, scheme=scheme),
        }
        for name, values in lines.items():
            assert np.diff(values).min() >= -1e-12, (name, kind, K, T, r, q, sigma)


# Largest rise of a put's boundary (fall of a call's) with time to expiry, relative to the boundary: the ripple of
# the Chebyshev interpolant between nodes, largest just before expiry at high volatility. The plan asks for a
# boundary that never rises; no scheme meets that exactly. Measured over r in [0, 0.3], q in [-0.05, 0.12], T in
# [1/52, 20] and sigma in [0.05, 1.5]: fast 2.1e-2 (sigma = 1.5), accurate 3.0e-5 and high 3.9e-6; over r up to
# 10 %, q up to 10 %, T up to 10 years and sigma up to 60 %: fast 8.1e-3, accurate 9.1e-8 and high 6.8e-9.
BOUNDARY_RIPPLE = {"fast": 2.5e-2, "accurate": 5e-5, "high": 6e-6}
#: Boundaries that do not converge in that set (r = 0 with q < 0 converges slowest): the engine rejects them.
NOT_CONVERGED = {"fast": {(0.0, -0.05, 5.0, 1.5), (0.0, -0.05, 20.0, 1.5)}, "accurate": set(), "high": set()}


@pytest.mark.parametrize("scheme", sorted(am.SCHEMES))
@pytest.mark.parametrize("kind", ["put", "call"])
def test_the_boundary_moves_monotonically_and_starts_at_x(scheme, kind):
    worst, rejected = 0.0, set()
    for r in (0.0, 0.01, 0.05, 0.1, 0.3):
        for q in (-0.05, 0.0, 0.02, 0.05, 0.12):
            if am.exercise_regime(r, q, option_type=kind) != "single":
                continue
            for T in (1 / 52, 0.25, 1.0, 5.0, 20.0):
                for sigma in (0.05, 0.2, 0.5, 1.5):
                    try:
                        _, b = am.exercise_boundary(
                            100.0, T, r, sigma, q, option_type=kind, scheme=scheme, tau=np.linspace(0.0, T, 2001)
                        )
                    except ValueError as error:
                        assert "has not converged" in str(error)
                        rejected.add((r, q, T, sigma))
                        continue
                    # Expiry: X = K min(1, r / q) for a put and K max(1, r / q) for a call (q > 0), K otherwise.
                    put_r, put_q = (q, r) if kind == "call" else (r, q)
                    x = 100.0 * (min(1.0, put_r / put_q) if put_q > 0 else 1.0)
                    assert b[0] == pytest.approx(x if kind == "put" else 1e4 / x, rel=1e-14)
                    normalised = b / 100.0 if kind == "put" else 100.0 / b
                    worst = max(worst, float(np.max(np.diff(normalised) / normalised[:-1])))
    assert worst <= BOUNDARY_RIPPLE[scheme]
    assert rejected == (NOT_CONVERGED[scheme] if kind == "put" else set())


def test_the_boundary_separates_exercise_from_holding():
    """Just below a put's critical spot (above a call's) the value is the intrinsic value; just inside it is more."""
    for kind, r, q in (("put", 0.06, 0.02), ("put", 0.05, 0.05), ("call", 0.03, 0.08), ("put", 0.0, -0.04)):
        details = am.price_details(100.0, 100.0, 0.75, r, 0.3, q, option_type=kind)
        b = details.critical_spot
        sign = bs.option_sign(kind)
        exercised = b * (1 + sign * 1e-6)  # a put is exercised below its critical spot, a call above
        held = b * (1 - sign * 1e-3)
        assert am.price(exercised, 100.0, 0.75, r, 0.3, q, option_type=kind) == max(sign * (exercised - 100.0), 0)
        assert am.price(held, 100.0, 0.75, r, 0.3, q, option_type=kind) > max(sign * (held - 100.0), 0)
        tau, boundary = am.exercise_boundary(100.0, 0.75, r, 0.3, q, option_type=kind)
        assert boundary[-1] == pytest.approx(b, rel=1e-12)
        assert tau[0] == 0.0 and tau[-1] == 0.75 and np.all(np.diff(tau) > 0)


# ------------------------------------------------------------------------------------------------------ regimes


@pytest.mark.parametrize(
    ("r", "q", "put", "call"),
    [
        (0.05, 0.02, "single", "single"),
        (0.05, 0.0, "single", "european"),
        (0.0, 0.03, "european", "single"),
        (0.0, -0.03, "single", "european"),
        (-0.01, 0.02, "european", "single"),
        (-0.01, -0.02, "double", "european"),
        (-0.02, -0.01, "european", "double"),
        (-0.01, -0.01, "european", "european"),
    ],
)
def test_exercise_regimes(r, q, put, call):
    """Healy (2021): a put is European for r <= 0, r <= q and has two boundaries for q < r < 0; a call by symmetry."""
    assert am.exercise_regime(r, q, option_type="put") == put
    assert am.exercise_regime(r, q, option_type="call") == call


def test_the_double_boundary_case_falls_back_to_the_tree_and_says_so():
    """QuantLib refuses q < r < 0 for a put; its FD engine on a 4000 x 4000 grid gives 7.62526 for this put."""
    details = am.price_details(100.0, 100.0, 1.0, -0.01, 0.2, -0.02, option_type="put")
    assert details.method == "tree" and details.fallback and details.fallback_reason == "double boundary"
    assert math.isnan(details.critical_spot) and details.fixed_point is None and math.isnan(details.last_step)
    assert abs(details.value - 7.62526) <= 1e-4  # measured 7.6252878 (QuantLib FD 7.6252621)
    assert details.value == am.price(100.0, 100.0, 1.0, -0.01, 0.2, -0.02, option_type="put")
    assert details.value == binomial.extrapolated_price(100.0, 100.0, 1.0, -0.01, 0.2, -0.02, "put")
    call = am.price_details(100.0, 100.0, 1.0, -0.02, 0.2, -0.01, option_type="call")
    assert call.fallback and call.value == details.value
    with pytest.raises(ValueError, match="two exercise boundaries"):
        am.exercise_boundary(100.0, 1.0, -0.01, 0.2, -0.02, option_type="put")


def test_a_boundary_that_has_not_converged_falls_back_to_the_tree_and_says_so():
    """Fast's 3 iterations do not settle a 20-year put with r = 0, q = -5 % and sigma = 150 %: its last step moves
    ln B by 0.13 (accurate's by 4e-3, high's by 4e-5). The tree value is within 1e-6 of the high scheme's."""
    args = (100.0, 100.0, 20.0, 0.0, 1.5, -0.05)
    fast = am.price_details(*args, option_type="put", scheme="fast")
    assert (fast.method, fast.fallback, fast.fallback_reason) == ("tree", True, "not converged")
    assert fast.last_step > 1e-2 and math.isnan(fast.critical_spot)
    assert fast.value == am.price(*args, option_type="put", scheme="fast")
    assert fast.value == binomial.extrapolated_price(100.0, 100.0, 20.0, 0.0, 1.5, -0.05, "put")
    high = am.price_details(*args, option_type="put", scheme="high")
    assert (high.method, high.fallback, high.fallback_reason) == ("alo", False, None) and high.last_step < 1e-3
    assert abs(fast.value - high.value) <= 1e-6  # measured 3.4e-7
    # The symmetric call takes the same path.
    call = am.price_details(100.0, 100.0, 20.0, -0.05, 1.5, 0.0, option_type="call", scheme="fast")
    assert call.fallback_reason == "not converged" and call.value == fast.value


def test_european_regimes_report_no_early_exercise():
    put = am.price_details(100.0, 90.0, 1.0, -0.01, 0.3, 0.01, option_type="put")
    call = am.price_details(100.0, 90.0, 1.0, 0.03, 0.3, 0.0, option_type="call")
    assert (put.method, put.premium, put.critical_spot) == ("european", 0.0, 0.0)
    assert (call.method, call.premium, call.critical_spot) == ("european", 0.0, math.inf)
    tau, b = am.exercise_boundary(90.0, 1.0, 0.03, 0.3, 0.0, option_type="call")
    assert np.all(np.isinf(b)) and tau.size == 101


# --------------------------------------------------------------------------------------------- degenerate inputs


def test_expired_and_zero_volatility_options():
    assert am.price(90.0, 100.0, 0.0, 0.05, 0.2, option_type="put") == 10.0
    assert am.price(90.0, 100.0, -1.0, 0.05, 0.2, option_type="call") == 0.0
    assert am.price_details(90.0, 100.0, 0.0, 0.05, 0.2, option_type="put").method == "intrinsic"
    # sigma = 0: the best of exercising at any date t, K e^{-rt} - S e^{-qt}. With q > r the stock falls faster
    # than the strike's present value, so waiting until t* = ln(r K / (q S)) / (r - q) = 11.76 years is best.
    S, K, r, q, T = 90.0, 100.0, 0.05, 0.10, 20.0
    t = np.linspace(0.0, T, 2_000_001)
    best = float(np.max(K * np.exp(-r * t) - S * np.exp(-q * t)))
    details = am.price_details(S, K, T, r, 0.0, q, option_type="put")
    assert details.method == "exact"
    assert details.value == pytest.approx(best, abs=1e-9)
    assert details.value > K - S + 17
    # Exercise at once when q <= r: the intrinsic value; and the symmetric call.
    assert am.price(90.0, 100.0, 1.0, 0.05, 0.0, 0.0, option_type="put") == 10.0
    assert am.price(K, S, T, q, 0.0, r, option_type="call") == details.value


def test_zero_spot_and_zero_strike():
    # A put on a worthless stock pays its strike: at once with r >= 0, at expiry (worth more) with r < 0.
    assert am.price(0.0, 100.0, 1.0, 0.05, 0.2, option_type="put") == 100.0
    assert am.price(0.0, 100.0, 1.0, 0.0, 0.2, -0.02, option_type="put") == 100.0
    assert am.price(0.0, 100.0, 1.0, -0.01, 0.2, -0.02, option_type="put") == pytest.approx(100 * math.exp(0.01))
    assert am.price(50.0, 0.0, 1.0, 0.05, 0.2, option_type="put") == 0.0
    # A zero-strike call is the stock (q >= 0: now; q < 0: at expiry); a call on a worthless stock is worthless.
    assert am.price(50.0, 0.0, 1.0, 0.05, 0.2, 0.02, option_type="call") == 50.0
    assert am.price(0.0, 100.0, 1.0, 0.05, 0.2, 0.02, option_type="call") == 0.0


def test_arrays_broadcast_and_match_scalar_calls():
    S = np.array([[80.0], [100.0], [120.0]])
    T = np.array([0.25, 1.0])
    values = am.price(S, 100.0, T, [0.05, 0.03], 0.25, 0.02, option_type="put")
    assert values.shape == (3, 2)
    for i in range(3):
        for j in range(2):
            expected = am.price(float(S[i, 0]), 100.0, float(T[j]), [0.05, 0.03][j], 0.25, 0.02, option_type="put")
            assert values[i, j] == pytest.approx(expected, rel=1e-14)  # summation order differs by batch shape
    assert isinstance(am.price(np.float64(100.0), 100, 1, 0.05, 0.2, 0.02), float)
    assert isinstance(am.price(np.array(100.0), 100, 1, 0.05, 0.2, 0.02), float)
    assert am.price([], 100.0, 1.0, 0.05, 0.2).shape == (0,)


@pytest.mark.parametrize("field", ["S", "K", "T", "r", "sigma", "q"])
@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_inputs_name_the_field(field, bad):
    args = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, q=0.02)
    args[field] = bad
    with pytest.raises(ValueError, match=rf"^{field} must be finite"):
        am.price(**args, option_type="put")
    args[field] = np.array([1.0, bad])
    with pytest.raises(ValueError, match=rf"^{field} must be finite"):
        am.price(**args, option_type="put")


def test_invalid_arguments():
    with pytest.raises(ValueError, match="non-negative"):
        am.price(-1.0, 100.0, 1.0, 0.05, 0.2)
    with pytest.raises(ValueError, match="volatility"):
        am.price(100.0, 100.0, 1.0, 0.05, -0.2)
    with pytest.raises(ValueError, match="scheme must be one of"):
        am.price(100.0, 100.0, 1.0, 0.05, 0.2, scheme="quick")
    with pytest.raises(ValueError, match="option_type"):
        am.price(100.0, 100.0, 1.0, 0.05, 0.2, option_type="straddle")
    with pytest.raises(ValueError, match="at least 1"):
        am.Scheme("broken", l=0, m=1, n=4, p=8)
    with pytest.raises(ValueError, match="scalars"):
        am.price_details([100.0, 101.0], 100.0, 1.0, 0.05, 0.2)
    with pytest.raises(ValueError, match="positive"):
        am.exercise_boundary(100.0, 1.0, 0.05, 0.0)
    with pytest.raises(ValueError, match=r"\[0, T\]"):
        am.exercise_boundary(100.0, 1.0, 0.05, 0.2, tau=[0.5, 1.5])


def test_schemes_by_name_or_object():
    custom = am.Scheme("custom", l=24, m=8, n=14, p=48)
    assert am.price(90.0, 100.0, 1.0, 0.05, 0.3, 0.01, option_type="put", scheme=" Accurate ") == am.price(
        90.0, 100.0, 1.0, 0.05, 0.3, 0.01, option_type="put", scheme=custom
    )
    assert am.price_details(90.0, 100.0, 1.0, 0.05, 0.3, option_type="put", scheme=custom).scheme == "custom"
    assert sorted(am.SCHEMES) == ["accurate", "fast", "high"]


def _within_simple_bounds(value: float, S: float, K: float, T: float, r: float, q: float, kind: str) -> bool:
    """At least the intrinsic value; a put at most K max(1, e^{-rT}), a call at most S max(1, e^{-qT})."""
    intrinsic = max(bs.option_sign(kind) * (S - K), 0.0)
    cap = K * max(1.0, math.exp(-r * T)) if kind == "put" else S * max(1.0, math.exp(-q * T))
    return intrinsic <= value <= cap * (1 + 1e-15)


# Valid inputs at the edges of the double range: subnormal and tiny volatilities, expiries down to the smallest
# double, rate and dividend-yield horizons r T and |q| T beyond 709 (where e^{rT} overflows), huge volatilities.
EXTREMES = [
    (100.0, 100.0, 1e-6, 0.05, 0.2, 0.0),
    (100.0, 100.0, 1.0, 0.05, 1e-6, 0.0),
    (100.0, 100.0, 1.0, 0.05, 8.0, 0.0),
    (100.0, 100.0, 100.0, 0.05, 0.2, 0.02),
    (100.0, 100.0, 1.0, 1e-9, 0.2, 0.05),
    (100.0, 100.0, 1.0, 0.8, 0.2, 0.0),
    (1e-6, 100.0, 1.0, 0.05, 0.2, 0.0),
    (1e6, 100.0, 1.0, 0.05, 0.2, 0.0),
    *((90.0, 100.0, 1.0, 0.05, sigma, 0.02) for sigma in (1e-80, 1e-170, 1e-200, 1e-320, 5e-324)),
    *((90.0, 100.0, 20.0, 0.05, sigma, 0.10) for sigma in (1e-80, 1e-200, 5e-324)),
    *((100.0, 100.0, T, 0.05, 0.2, 0.0) for T in (1e-150, 1e-200, 1e-300, 5e-324)),
    (100.0, 100.0, 1e-20, 0.05, 200.0, 0.0),
    (100.0, 100.0, 1e5, 0.05, 0.2, 0.0),
    (100.0, 100.0, 1e5, 0.05, 1e-10, 0.0),
    (90.0, 100.0, 100.0, 8.0, 0.2, 0.0),
    (100.0, 100.0, 100.0, 0.05, 0.2, 8.0),
    (90.0, 100.0, 1e4, 0.0, 8.0, -0.03),
    (90.0, 100.0, 100.0, -0.01, 50.0, -0.02),
]


@pytest.mark.parametrize("args", EXTREMES)
def test_no_runtime_warning_at_extreme_inputs(args):
    """pytest turns RuntimeWarning into an error: every scheme stays quiet, finite and within the simple bounds."""
    S, K, T, r, _, q = args
    for kind in ("call", "put"):
        put_r = q if kind == "call" else r
        if put_r < 0 and -put_r * T > 700:
            continue  # the value itself, K e^{|r| T} (a call: S e^{|q| T}), overflows
        for scheme in am.SCHEMES:
            value = am.price(*args, option_type=kind, scheme=scheme)
            assert np.isfinite(value) and _within_simple_bounds(value, S, K, T, r, q, kind), (kind, scheme, value)


def test_a_dividend_yield_of_minus_800_percent_over_a_century():
    """The stock grows at 805 % a year: a put is worth its intrinsic value or nothing; its symmetric call is the
    K e^{qT} that overflows, so only the put is valued."""
    for S, expected in ((90.0, 10.0), (110.0, 0.0)):
        assert am.price(S, 100.0, 100.0, 0.05, 0.2, -8.0, option_type="put") == pytest.approx(expected, abs=1e-12)


def test_degenerate_inputs_take_their_limits():
    # sigma -> 0: the zero-volatility value. With r > q a put in the money is exercised at once (black_scholes
    # gives the European 6.905 at sigma = 1e-80); with q > r it waits for t* = 11.76 years.
    zero_vol_wait = am.price(90.0, 100.0, 20.0, 0.05, 0.0, 0.10, option_type="put")
    assert zero_vol_wait == pytest.approx(27.777777777777, abs=1e-9)
    for sigma in (1e-9, 1e-80, 1e-200, 5e-324):
        assert am.price(90.0, 100.0, 1.0, 0.05, sigma, 0.02, option_type="put") == 10.0
        assert am.price(90.0, 100.0, 20.0, 0.05, sigma, 0.10, option_type="put") == zero_vol_wait
        assert am.price_details(90.0, 100.0, 1.0, 0.05, sigma, 0.02, option_type="put").method == "exact"
    assert bs.price(90.0, 100.0, 1.0, 0.05, 1e-80, 0.02, "put") == pytest.approx(6.905061852463433, rel=1e-15)
    # T -> 0: max(European, intrinsic), the premium being below 1e-16 of the strike.
    for T in (1e-150, 1e-300, 5e-324):
        assert am.price(100.0, 100.0, T, 0.05, 0.2, option_type="put") == pytest.approx(0.0, abs=1e-12)
        assert am.price(90.0, 100.0, T, 0.05, 0.2, option_type="put") == 10.0
        assert am.price(110.0, 100.0, T, 0.05, 0.2, 0.03, option_type="call") == 10.0
    # The exact branch and the engine agree across the total-volatility threshold (1e-8), within 4e-9 S.
    for args in ((100.0, 100.0, 1.0, 0.05, 0.02), (100.0, 100.0, 1.0, 0.05, 0.0), (90.0, 100.0, 20.0, 0.05, 0.10)):
        S, K, T, r, q = args
        below = am.price_details(S, K, T, r, 0.999e-8 / math.sqrt(T), q, option_type="put")
        above = am.price_details(S, K, T, r, 1.001e-8 / math.sqrt(T), q, option_type="put")
        assert (below.method, above.method) == ("exact", "alo")
        assert abs(below.value - above.value) <= 4e-9 * S
    # Bounds that meet need no tree: a double-boundary put at 5,000 % volatility is worth K e^{-rT}.
    details = am.price_details(90.0, 100.0, 100.0, -0.01, 50.0, -0.02, option_type="put")
    assert details.method == "exact" and details.value == pytest.approx(100.0 * math.e, rel=1e-14)
    # At zero total volatility the boundary is X = K min(1, r / q) at every time to expiry.
    _, b = am.exercise_boundary(100.0, 20.0, 0.05, 1e-12, 0.10, option_type="put")
    assert np.all(b == 50.0)


# ----------------------------------------------------------------------------------- long expiries, hand-over

#: The perpetual put (Merton 1973) at r = 5 %, q = 0, sigma = 20 %: S* = 71.43, worth 12.320033 at S = K = 100.
PERPETUAL = binomial._perpetual_put(100.0, 100.0, 0.05, 0.0, 0.2)


def test_long_expiries_approach_the_perpetual_put():
    """Every scheme reaches the perpetual value from below; "fast" and "accurate" hand over to "high" (before the
    hand-over "fast" gave 13.5713 at 1,000 years, 6.5480 at 3,000 and 25.6631 at 10,000)."""
    assert pytest.approx(12.320032867762633, rel=1e-15) == PERPETUAL
    for scheme in am.SCHEMES:
        values = [am.price(100.0, 100.0, T, 0.05, 0.2, option_type="put", scheme=scheme) for T in (30, 100, 1e3, 1e4)]
        assert np.all(np.diff(values) >= 0) and values[-1] <= PERPETUAL * (1 + 1e-15)
        # QdFp strict at 30 years: 12.20212273175 (measured errors: fast 2.8e-3, accurate 3.0e-5, high 3.9e-7)
        assert abs(values[0] - 12.20212273175) <= {"fast": 4e-3, "accurate": 5e-5, "high": 1e-6}[scheme]
        for T in (1e3, 1e4, 1e5):
            details = am.price_details(100.0, 100.0, T, 0.05, 0.2, option_type="put", scheme=scheme)
            assert details.scheme == "high" and abs(details.value - PERPETUAL) <= 1e-6


def test_fast_and_accurate_hand_over_beyond_their_domain():
    """The stiffness max(T c^2 / sigma^2, T max(|r|, |q|)) picks the scheme; a custom scheme is used as given."""
    cases = {  # (T, r, q, sigma): the scheme that "fast" ends with
        (0.5, 0.05, 0.02, 0.25): "fast",  # stiffness 0.025
        (30.0, 0.05, 0.0, 0.2): "fast",  # 1.9
        (10.0, 0.05, 0.02, 0.05): "accurate",  # 10 from the drift (0.5 from the rate)
        (5.0, 0.25, 0.0, 0.02): "high",  # 781
        (100.0, 0.05, 0.0, 0.2): "accurate",  # 6.25
    }
    for (T, r, q, sigma), expected in cases.items():
        assert am.price_details(100.0, 100.0, T, r, sigma, q, option_type="put", scheme="fast").scheme == expected
        assert am.price_details(100.0, 100.0, T, r, sigma, q, option_type="put", scheme="high").scheme == "high"
    # The low-volatility put of the review: "fast" was 0.01800, 39 % below QdFp strict (0.029418589).
    value = am.price(100.0, 100.0, 5.0, 0.25, 0.02, option_type="put", scheme="fast")
    assert value == pytest.approx(0.029418589, abs=1e-8)
    mine = am.Scheme("mine", l=8, m=3, n=7, p=24)
    assert am.price_details(100.0, 100.0, 5.0, 0.25, 0.02, option_type="put", scheme=mine).scheme == "mine"


@settings(max_examples=200, deadline=None)
@given(
    S=spots,
    K=spots,
    T=st.floats(1.0, 30.0),
    r=st.one_of(st.just(0.0), st.floats(1e-8, 0.25)),  # binomial._perpetual_put overflows at subnormal rates
    q=st.one_of(st.just(0.0), st.floats(-0.05, -1e-8), st.floats(1e-8, 0.25)),
    sigma=st.floats(0.02, 1.5),
    kind=st.sampled_from(["call", "put"]),
    scheme=st.sampled_from(sorted(am.SCHEMES)),
)
def test_long_dated_values_lie_within_the_tree_modules_bounds(S, K, T, r, q, sigma, kind, scheme):
    """binomial._american_bounds (European and intrinsic below; the strike and the perpetual put above) is an
    independent implementation of the bounds: before the bounds were applied, "fast" crossed the upper one in
    94 of a random sweep by up to 7.3e-2, "accurate" in 89 by 5.1e-4 and "high" in 44 by 1.25e-5."""
    assume(am.exercise_regime(r, q, option_type=kind) != "double")
    value = am.price(S, K, T, r, sigma, q, option_type=kind, scheme=scheme)
    european = bs.price(S, K, T, r, sigma, q, kind)
    lower, upper = binomial._american_bounds(S, K, T, r, sigma, q, bs.option_sign(kind), european)
    assert lower - 1e-12 * max(1.0, upper) <= value <= upper + 1e-12 * max(1.0, upper)


@pytest.mark.parametrize("scheme", sorted(am.SCHEMES))
def test_long_dated_values_rise_with_expiry(scheme):
    """Along +-10 % lines in T through 150 random options with expiries of 1 to 30 years. Measured: no step down
    larger than 7.5e-9 (a 13 % put near its perpetual value, where V changes by 1e-9 a year and the error of
    "high" is 1.5e-7); before the hand-over "fast" fell by 1.1e-4 (a call, T = 14.3, q = 13.7 %)."""
    rng = np.random.default_rng(23)
    line = np.linspace(0.9, 1.1, 11)
    for _ in range(150):
        kind = str(rng.choice(["call", "put"]))
        K, T, sigma = 100 * math.exp(rng.uniform(-0.6, 0.6)), rng.uniform(1.0, 30.0), rng.uniform(0.05, 0.8)
        r, q = rng.uniform(0.0, 0.15), rng.uniform(-0.03, 0.15)
        if am.exercise_regime(r, q, option_type=kind) == "double":
            continue
        values = am.price(100.0, K, T * line, r, sigma, q, option_type=kind, scheme=scheme)
        assert np.diff(values).min() >= -2e-8 * max(1.0, float(values.max())), (kind, K, T, r, q, sigma)


# ------------------------------------------------------------------------------------------- fixed-point form


def test_the_fixed_point_form_can_be_fixed_across_the_switch():
    """At |r - q| = 1e-3 the automatic choice changes form and the value jumps by the change of discretisation
    error (fast 3.9e-4 here); with the form fixed the change is what the slope predicts (3e-10 over 2e-12)."""
    q_cdf, q_density = 0.05 - 1e-3 - 1e-12, 0.05 - 1e-3 + 1e-12
    jump = {"fast": (2e-4, 5e-4), "accurate": (5e-7, 2e-6), "high": (5e-10, 5e-9)}
    for scheme, (least, most) in jump.items():
        auto = [
            am.price(110.0, 100.0, 10.0, 0.05, 0.3, q, option_type="put", scheme=scheme) for q in (q_cdf, q_density)
        ]
        assert least <= abs(auto[1] - auto[0]) <= most
        for form in ("cdf", "density"):
            fixed = [
                am.price(110.0, 100.0, 10.0, 0.05, 0.3, q, option_type="put", scheme=scheme, fixed_point=form)
                for q in (q_cdf, q_density)
            ]
            assert abs(fixed[1] - fixed[0]) <= 1e-9
            details = am.price_details(110.0, 100.0, 10.0, 0.05, 0.3, q_cdf, option_type="put", fixed_point=form)
            assert details.fixed_point == form
    assert am.price_details(110.0, 100.0, 10.0, 0.05, 0.3, q_cdf, option_type="put").fixed_point == "cdf"
    assert am.price_details(110.0, 100.0, 10.0, 0.05, 0.3, q_density, option_type="put").fixed_point == "density"
    with pytest.raises(ValueError, match="fixed_point"):
        am.price(100.0, 100.0, 1.0, 0.05, 0.2, option_type="put", fixed_point="newton")


#: RMSE against qdfp_strict with each form forced on the whole grid (|r - q| up to 0.1). Measured: cdf as the
#: automatic choice (3.7e-5, 1.7e-7, 1.8e-9); density 5.7e-6, 2.8e-7 and 1.9e-7, max 2.6e-4, 1.7e-5 and 1.3e-5.
FORCED_FORM_RMSE = {"fast": (1e-4, 1e-5), "accurate": (5e-7, 5e-7), "high": (1e-8, 5e-7)}


@pytest.mark.parametrize("scheme", sorted(FORCED_FORM_RMSE))
def test_either_form_converges_on_the_grid(grid, scheme):
    for form, bound in zip(("cdf", "density"), FORCED_FORM_RMSE[scheme], strict=True):
        values = am.price(
            grid["spot"],
            100.0,
            grid["maturity"],
            grid["rate"],
            grid["volatility"],
            grid["dividend_yield"],
            option_type="put",
            scheme=scheme,
            fixed_point=form,
        )
        assert np.sqrt(np.mean((values - grid["qdfp_strict"]) ** 2)) <= bound, form


# ------------------------------------------------------------------------------------------------- batches


def test_batches_are_grouped_by_one_sort():
    """Options sharing (T, r, sigma, q) share a boundary, whatever their order in the batch; grouping 400,000 rows
    into 4,000 groups takes one sort (11 ms measured; the earlier one scan per group took about 1.6 s)."""
    rng = np.random.default_rng(3)
    T = rng.choice([0.25, 0.5, 1.0], size=60)
    sigma = rng.choice([0.2, 0.3], size=60)
    S = rng.uniform(70.0, 130.0, size=60)
    values = am.price(S, 100.0, T, 0.05, sigma, 0.02, option_type="put", scheme="fast")
    for i in range(60):
        expected = am.price(
            float(S[i]), 100.0, float(T[i]), 0.05, float(sigma[i]), 0.02, option_type="put", scheme="fast"
        )
        assert values[i] == pytest.approx(expected, rel=1e-14)
    keys = (np.repeat(np.arange(2000.0), 200), np.zeros(400_000), np.tile(np.arange(2.0), 200_000), np.ones(400_000))
    start = time.perf_counter()
    groups = am._groups(*keys)
    seconds = time.perf_counter() - start
    assert len(groups) == 4000 and sum(rows.size for _, rows in groups) == 400_000
    assert all(np.all(keys[0][rows] == key[0]) and np.all(keys[2][rows] == key[2]) for key, rows in groups)
    assert seconds <= 0.5
