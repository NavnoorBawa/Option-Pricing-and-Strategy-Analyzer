import itertools
import math

import numpy as np
import pytest

from options_analyzer import black_scholes as bs
from options_analyzer.implied_vol import (
    REJECTION_REASONS,
    ImpliedVolError,
    american_bounds,
    implied_volatility,
    implied_volatility_american,
    implied_volatility_batch,
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
    lower, _ = no_arbitrage_bounds(100, K, T, r, q, kind)
    try:
        iv = implied_volatility(price, 100, K, T, r, q, kind)
    except ImpliedVolError as error:
        # 682 of the 2,520 cases: 450 in the money within 1e-12·max(S, K) of the lower bound (at most 0.75 of that
        # margin above it), where the last digits of the price would decide the volatility, and 232 out of the money
        # that bs.price rounds to zero. Both used to give whatever volatility repriced them, 0 included. The 1e-3
        # allows for the rounding of the intrinsic value, about ε·max(S, K) = 2.2e-4 of the margin.
        assert error.reason == "no_time_value" and price - lower <= 1e-12 * max(100, K) * (1 + 1e-3), error
        return
    # Measured: at most 1.0e-6, a 5-year call struck at 50 at 5 % volatility with 3.4e-10 of time value.
    assert iv == pytest.approx(sigma, rel=1e-5)
    # And within what the rounding of the price moves the volatility, 8·ε·max(S, K)/vega: black_scholes.price and the
    # conversion to the out-of-the-money price each round by a few ε·max(S, K). Measured: at most 1.7·ε·max(S, K)/vega
    # (with NumPy 1.26 and SciPy 1.11 too). For a one-year option at the money at 30 % the bound is 4.5e-15.
    vega = bs.greeks(100, K, T, r, sigma, q, kind)["vega"]
    assert abs(iv - sigma) * vega <= 8 * EPS * max(100, K)


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


def test_a_quote_at_the_lower_bound_has_no_time_value():
    # It used to give 0.0, while 1e-13 more gave 94.5 % (test_the_two_day_call_of_the_review): no volatility, and a
    # reason.
    lower, _ = no_arbitrage_bounds(100, 80, 1, 0.05, 0.0, "call")
    with pytest.raises(ImpliedVolError) as error:
        implied_volatility(lower, 100, 80, 1, 0.05)
    assert error.value.reason == "no_time_value" and "no_time_value" in REJECTION_REASONS
    assert str(error.value) == (
        "price 23.9016 is within 1e-10 (1e-12·max(S, K)) of the no-arbitrage lower bound 23.9016 (discounted "
        "intrinsic value), so it has no time value to identify a volatility from: its last digits would decide it"
    )


def test_tiny_but_positive_price_is_solved():
    price = bs.price(100, 250, 0.1, 0.05, 0.4, option_type="call")
    assert 0 < price < 1e-10
    assert implied_volatility(price, 100, 250, 0.1, 0.05) == pytest.approx(0.4, abs=1e-6)


AMERICAN = list(itertools.product([80, 100, 125], [0.1, 1.0], [0.1, 0.3, 0.8], ["call", "put"], [0.0, 0.04]))


@pytest.mark.parametrize(("K", "T", "sigma", "kind", "q"), AMERICAN)
def test_american_round_trip(K, T, sigma, kind, q):
    from options_analyzer import binomial

    price = binomial.price(100, K, T, 0.05, sigma, q, kind, 501, american=True, method="lr")
    try:
        iv = implied_volatility_american(price, 100, K, T, 0.05, q, kind)
    except ImpliedVolError as error:
        # A quote equal to the zero-volatility value (here the value of exercising now) is not identifiable.
        floor = binomial.price(100, K, T, 0.05, 1e-6, q, kind, 501, american=True, method="lr")
        assert error.reason == "no_time_value" and abs(price - floor) <= 1e-12 * max(100, K), error
        return
    assert abs(iv - sigma) < 1e-5


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


# ------------------------------------------------------------------- Let's Be Rational and the batch solver

EPS = np.finfo(float).eps


def _random_quotes(n, seed, *, z_max=8.0):
    """Spot 100, expiries from a day to 5 years, vols 1% to 300%, strikes up to z_max standard deviations away."""
    rng = np.random.default_rng(seed)
    T = 10 ** rng.uniform(np.log10(1 / 365), np.log10(5), n)
    sigma = 10 ** rng.uniform(-2, np.log10(3), n)
    r = rng.uniform(-0.02, 0.1, n)
    q = rng.uniform(0, 0.05, n)
    K = 100 * np.exp((r - q) * T + rng.uniform(-z_max, z_max, n) * sigma * np.sqrt(T))
    kind = np.where(rng.random(n) < 0.5, "call", "put")
    return K, T, sigma, r, q, kind


def _exact_quote(S, K, T, r, q, sigma, kind):
    """Price, condition number and time value (in units of the price's last place), with 60-digit mpmath."""
    mp = pytest.importorskip("mpmath")
    with mp.workdps(60):
        s, k, t, rr, qq = (mp.mpf(float(v)) for v in (S, K, T, r, q))
        phi = 1 if kind == "call" else -1
        fwd, pv = s * mp.exp(-qq * t), k * mp.exp(-rr * t)
        intrinsic = max(phi * (fwd - pv), 0)
        if sigma == 0:
            return intrinsic, math.inf, 0.0
        v = mp.mpf(float(sigma))
        d1 = (mp.log(s / k) + (rr - qq + v * v / 2) * t) / (v * mp.sqrt(t))
        d2 = d1 - v * mp.sqrt(t)

        def ncdf(z):
            return mp.erfc(-z / mp.sqrt(2)) / 2

        price = phi * (fwd * ncdf(phi * d1) - pv * ncdf(phi * d2))
        if not price > 0:
            return price, math.inf, 0.0
        vega = fwd * mp.exp(-d1 * d1 / 2) / mp.sqrt(2 * mp.pi) * mp.sqrt(t)
        return price, min(float(1 + abs(price / (v * vega))), 1e300), float((price - intrinsic) / (EPS * price))


def _exact_time_value(price, S, K, T, r, q, kind):
    """``price`` (a double) less the discounted intrinsic value ``max(±(S e^(-qT) - K e^(-rT)), 0)``, at 60 digits."""
    mp = pytest.importorskip("mpmath")
    with mp.workdps(60):
        s, k, t, rr, qq = (mp.mpf(float(v)) for v in (S, K, T, r, q))
        phi = 1 if kind == "call" else -1
        return float(mp.mpf(float(price)) - max(phi * (s * mp.exp(-qq * t) - k * mp.exp(-rr * t)), 0))


def _price_level_study(n, seed):
    """implied_volatility on exactly priced quotes: the relative error must be at most max(1e-14, 8·cond·ε).

    8 rather than 4 (as for the normalised inversion in test_lets_be_rational.py) because converting spot,
    strike, rate, dividend yield and expiry to ln(F/K) and an out-of-the-money forward price rounds too. An
    in-the-money quote whose rounded price keeps no more than 1e-12·max(S, K) of time value must be rejected as
    having none (within 1e-3 of that margin either side, where the rounding of the intrinsic value can tip it,
    either outcome is right), and no out-of-the-money quote may be: its price is all time value. A quote that rounds
    onto the upper bound (a call worth the discounted spot to 16 digits) must be rejected as such. Returns the
    failures, the worst error over its tolerance, the worst error where cond < 10, and how many quotes priced at
    least 1e-300 were solved (out of the money among them) and rejected for each reason.
    """
    K, T, sigma, r, q, kind = _random_quotes(n, seed)
    failures, worst, well_conditioned = [], 0.0, 0.0
    counts = {"quotes": 0, "solved": 0, "solved out of the money": 0, "no_time_value": 0, "at_or_above_upper_bound": 0}
    for i in range(n):
        price, cond, _ = _exact_quote(100.0, K[i], T[i], r[i], q[i], sigma[i], kind[i])
        if price < 1e-300:
            continue
        counts["quotes"] += 1
        quote, margin = float(price), 1e-12 * max(100.0, K[i])
        forward = 100.0 * math.exp((r[i] - q[i]) * T[i])
        in_the_money = forward > K[i] if kind[i] == "call" else forward < K[i]
        time_value = _exact_time_value(quote, 100.0, K[i], T[i], r[i], q[i], kind[i])
        case = f"K={K[i]!r} T={T[i]!r} sigma={sigma[i]!r} {kind[i]}"
        try:
            vol = implied_volatility(quote, 100.0, K[i], T[i], r[i], q[i], kind[i])
        except ImpliedVolError as error:
            _, upper = no_arbitrage_bounds(100.0, K[i], T[i], r[i], q[i], kind[i])
            expected = error.reason == "at_or_above_upper_bound" and quote >= upper
            expected |= error.reason == "no_time_value" and in_the_money and time_value <= margin * (1 + 1e-3)
            if not expected:
                failures.append(f"{case}: {error}")
            counts[error.reason] = counts.get(error.reason, 0) + 1
            continue
        counts["solved"] += 1
        counts["solved out of the money"] += not in_the_money
        if in_the_money and time_value < margin * (1 - 1e-3):
            failures.append(f"{case}: {vol!r} without time value")
        error = abs(vol - sigma[i]) / sigma[i]
        tolerance = max(1e-14, 8 * cond * EPS)
        worst = max(worst, error / tolerance)
        well_conditioned = max(well_conditioned, error) if cond < 10 else well_conditioned
        if not error <= tolerance:
            failures.append(f"{case}: {error:.3g} > {tolerance:.3g}")
    return failures, worst, well_conditioned, counts


@pytest.mark.oracle
def test_price_level_accuracy_fast_subset():
    failures, worst, _, counts = _price_level_study(1_000, seed=2)
    assert not failures, failures[:5]
    assert worst < 1
    # Measured: all 1,000 priced at least 1e-300; 874 solved (487 out of the money), 126 in the money without time
    # value (they used to be given the volatility that repriced their last digits).
    assert counts == {
        "quotes": 1_000,
        "solved": 874,
        "solved out of the money": 487,
        "no_time_value": 126,
        "at_or_above_upper_bound": 0,
    }


@pytest.mark.oracle
@pytest.mark.slow
def test_price_level_accuracy_20k():
    # Measured on these 20,000 quotes (about 10 s of mpmath), all priced at least 1e-300: 17,791 solved (10,058 out of
    # the money), 2,203 in the money without time value and 6 rounded onto the upper bound. The worst error is 0.57
    # of the tolerance, and at most 5.3e-15 where cond < 10. Before ln(F/K) and the forward intrinsic value were
    # computed without cancellation (log1p and expm1) near-the-money errors reached 6.7e-13.
    failures, worst, well_conditioned, counts = _price_level_study(20_000, seed=2)
    assert not failures, failures[:5]
    assert worst < 1
    assert well_conditioned <= 1e-14
    assert counts == {
        "quotes": 20_000,
        "solved": 17_791,
        "solved out of the money": 10_058,
        "no_time_value": 2_203,
        "at_or_above_upper_bound": 6,
    }


def test_out_of_the_money_volatilities_are_now_exact():
    # test_round_trip accepts 1e-6 (or a repriced quote); out of the money the solver now returns the volatility.
    for K, T, sigma, r, q, kind in GRID[::7]:
        price = bs.price(100, K, T, r, sigma, q, kind)
        forward = 100 * np.exp((r - q) * T)
        out_of_the_money = (kind == "call") == (forward <= K)
        if out_of_the_money and price > 1e-250:
            assert implied_volatility(price, 100, K, T, r, q, kind) == pytest.approx(sigma, rel=2e-9)


def test_brent_takes_over_where_lets_be_rational_reports_no_convergence(monkeypatch):
    from options_analyzer import lets_be_rational

    price = bs.price(100, 110, 0.5, 0.03, 0.25, 0.01, "put")
    expected = implied_volatility(price, 100, 110, 0.5, 0.03, 0.01, "put")
    calls = []

    def not_converged(beta, x, iterations=2):
        calls.append((beta, x))
        return math.nan, False

    monkeypatch.setattr(lets_be_rational, "_lets_be_rational_s", not_converged)
    assert implied_volatility(price, 100, 110, 0.5, 0.03, 0.01, "put") == pytest.approx(expected, abs=1e-12)
    assert calls  # the fast path was tried first
    # Brent runs on the normalised price, which keeps its relative accuracy where bs.price cancels: one-day
    # options a few strikes out of the money at low volatility, where bs.price is off by up to 7.6e-11. Measured:
    # at most 2.5e-14 here, against 1.9e-13 for Brent on bs.price; both are within tol = 1e-12.
    for K, sigma, kind in ((104.0, 0.1, "call"), (98.0, 0.05, "put"), (101.0, 0.02, "call")):
        exact, _, _ = _exact_quote(100.0, K, 1 / 365, 0.03, 0.01, sigma, kind)
        assert implied_volatility(float(exact), 100, K, 1 / 365, 0.03, 0.01, kind) == pytest.approx(
            sigma, rel=1e-13, abs=0
        )
    with pytest.raises(ImpliedVolError, match="exceeds 2000%") as error:  # the fallback keeps the volatility cap
        implied_volatility(99.9999, 100, 100, 0.01, 0, 0, "call")
    assert error.value.reason == "above_max_vol"

    # The batch hands rows that do not converge to the scalar path (here falling back to Brent as well).
    def none_converged(beta, x, iterations=2):
        return np.full(len(beta), np.nan), np.zeros(len(beta), dtype=bool)

    monkeypatch.setattr(lets_be_rational, "_lets_be_rational_v", none_converged)
    batch = implied_volatility_batch(
        [price, 99.9999], 100, [110, 100], [0.5, 0.01], [0.03, 0], [0.01, 0], ["put", "call"], return_reasons=True
    )
    assert batch.vol[0] == pytest.approx(expected, abs=1e-12) and np.isnan(batch.vol[1])
    assert batch.reason.tolist() == ["", "above_max_vol"]


def test_rate_and_dividend_yield_must_be_finite():
    with pytest.raises(ImpliedVolError, match="rate must be a finite number"):
        implied_volatility(5.0, 100, 100, 1, math.nan)
    with pytest.raises(ImpliedVolError, match="dividend yield must be a finite number"):
        implied_volatility(5.0, 100, 100, 1, 0.05, math.inf)


def test_errors_carry_a_reason_code_and_survive_pickling():
    import pickle

    cases = [
        ((math.nan, 100, 100, 1, 0.05), "not_finite"),
        ((5.0, -100, 100, 1, 0.05), "not_positive"),
        ((5.0, 100, 100, 0, 0.05), "expired"),
        ((5.0, 100, 100, 1, -800.0), "not_finite"),  # K e^(-rT) overflows
        ((1.0, 100, 90, 1, 0.05), "below_lower_bound"),
        ((100.0, 100, 90, 1, 0.05), "at_or_above_upper_bound"),
        ((99.9999, 100, 100, 0.01, 0.0), "above_max_vol"),
        ((40.0, 100, 60, 2 / 365, 0.0), "no_time_value"),  # a call at its intrinsic value
        ((0.0, 100, 120, 1, 0.05), "no_time_value"),  # out of the money, quoted at 0
    ]
    for args, reason in cases:
        with pytest.raises(ImpliedVolError) as error:
            implied_volatility(*args)
        assert error.value.reason == reason and reason in REJECTION_REASONS
        copy = pickle.loads(pickle.dumps(error.value))
        assert copy.reason == reason and str(copy) == str(error.value)
    with pytest.raises(ImpliedVolError) as error:
        implied_volatility_american(5.0, 100, 90, 1, 0.05, 0, "call")
    assert error.value.reason == "below_lower_bound"
    default = ImpliedVolError("no specific reason")  # the default code is a documented one too
    assert default.reason == "invalid" and default.reason in REJECTION_REASONS
    assert pickle.loads(pickle.dumps(default)).reason == "invalid"


def _rejected_quotes():
    """Quotes that implied_volatility rejects: every reason, every check in its order, the bounds hit exactly."""
    lower, upper = no_arbitrage_bounds(100, 90, 1, 0.05, 0.0, "call")
    put_lower, put_upper = no_arbitrage_bounds(100, 120, 2, 0.03, 0.01, "put")
    _, short_upper = no_arbitrage_bounds(100, 90, 0.01, 0.05, 0.0, "call")
    return [
        (math.nan, 100, 90, 1, 0.05, 0.0, "call"),
        (5.0, math.inf, 90, 1, 0.05, 0.0, "call"),
        (5.0, 100, math.nan, 1, 0.05, 0.0, "call"),
        (5.0, 100, 90, math.nan, 0.05, 0.0, "put"),
        (5.0, 100, 90, 1, math.nan, 0.0, "put"),
        (5.0, 100, 90, 1, 0.05, -math.inf, "put"),
        (math.nan, -100, 90, 1, 0.05, 0.0, "call"),  # the first failing check wins
        (5.0, 0.0, 90, 1, 0.05, 0.0, "call"),
        (5.0, 100, -90, 1, 0.05, 0.0, "call"),
        (5.0, 100, 90, 0.0, 0.05, 0.0, "call"),
        (5.0, 100, 90, -1.0, 0.05, 0.0, "put"),
        (lower - 0.01, 100, 90, 1, 0.05, 0.0, "call"),
        (lower - 2e-9, 100, 90, 1, 0.05, 0.0, "call"),  # just beyond the 1e-12·max(S, K) tolerance
        (upper, 100, 90, 1, 0.05, 0.0, "call"),
        (float(np.nextafter(upper, np.inf)), 100, 90, 1, 0.05, 0.0, "call"),
        (put_upper, 100, 120, 2, 0.03, 0.01, "put"),
        (put_lower - 1e-6, 100, 120, 2, 0.03, 0.01, "put"),
        (99.9999, 100, 100, 0.01, 0.0, 0.0, "call"),  # above the 2,000% cap
        # No time value: in the money on the bound, inside the tolerance below it and inside the margin above it;
        # out of the money at 0, inside the tolerance below 0, and at 5e-324, which √(FK) = 107 divides to 0.
        (lower, 100, 90, 1, 0.05, 0.0, "call"),
        (lower - 1e-11, 100, 90, 1, 0.05, 0.0, "call"),
        (lower + 1e-11, 100, 90, 1, 0.05, 0.0, "call"),
        (put_lower, 100, 120, 2, 0.03, 0.01, "put"),
        (0.0, 100, 120, 1, 0.05, 0.0, "call"),
        (-1e-11, 100, 120, 1, 0.05, 0.0, "call"),
        (5e-324, 100, 120, 1, 0.05, 0.0, "call"),
        # An ulp below the upper bound the total volatility σ√T is 16.6 (σ = 166 at 0.01 years): the cap.
        (float(np.nextafter(short_upper, 0)), 100, 90, 0.01, 0.05, 0.0, "call"),
    ]


def test_batch_rejections_equal_the_scalar_errors():
    quotes = _rejected_quotes()
    columns = [np.array(column) for column in zip(*quotes, strict=True)]
    result = implied_volatility_batch(*columns, return_reasons=True)
    assert np.isnan(result.vol).all()
    for i, quote in enumerate(quotes):
        with pytest.raises(ImpliedVolError) as error:
            implied_volatility(*quote)
        assert result.reason[i] == error.value.reason, quote
        assert result.message[i] == str(error.value), quote


def test_batch_agrees_with_the_scalar_path():
    K, T, sigma, r, q, kind = _random_quotes(4_000, seed=9)
    prices = np.array([bs.price(100, *args) for args in zip(K, T, r, sigma, q, kind, strict=True)])
    prices[::50] = -1.0  # rejected
    on_the_bound = slice(1, None, 50)
    prices[on_the_bound] = [
        no_arbitrage_bounds(100, *args)[0]
        for args in zip(
            K[on_the_bound], T[on_the_bound], r[on_the_bound], q[on_the_bound], kind[on_the_bound], strict=True
        )
    ]
    result = implied_volatility_batch(prices, 100, K, T, r, q, kind, return_reasons=True)
    vega = np.array([bs.greeks(100, *args)["vega"] for args in zip(K, T, r, sigma, q, kind, strict=True)])
    with np.errstate(divide="ignore"):
        cond = 1 + prices / (sigma * vega)  # the two paths round ln(F/K) differently (NumPy's and libm's log1p)
    for i in range(len(prices)):
        try:
            expected, reason, message = implied_volatility(prices[i], 100, K[i], T[i], r[i], q[i], kind[i]), "", ""
        except ImpliedVolError as error:
            expected, reason, message = math.nan, error.reason, str(error)
        assert result.reason[i] == reason and result.message[i] == message, i
        if np.isnan(expected):
            assert np.isnan(result.vol[i]), i
        else:
            assert result.vol[i] == pytest.approx(expected, rel=max(1e-14, 8 * cond[i] * EPS), abs=1e-15), i


def test_batch_broadcasts_and_keeps_shapes():
    strikes = np.array([[80.0], [100.0], [125.0]])  # (3, 1)
    expiries = np.array([0.1, 0.5, 1.0, 2.0])  # (4,)
    prices = bs.price(100, strikes, expiries, 0.03, 0.3, 0.01, "put")
    vol = implied_volatility_batch(prices, 100, strikes, expiries, 0.03, 0.01, "put")
    assert vol.shape == (3, 4)
    np.testing.assert_allclose(vol, 0.3, rtol=1e-12)
    kinds = np.array([["call", "put"]] * 3)  # an option type per quote
    calls, puts = (bs.price(100, strikes, 0.5, 0.03, 0.3, 0.0, kind) for kind in ("call", "put"))
    prices = np.where(kinds == "call", calls, puts)
    np.testing.assert_allclose(implied_volatility_batch(prices, 100, strikes, 0.5, 0.03, 0.0, kinds), 0.3, rtol=1e-12)
    scalar = implied_volatility_batch(bs.price(100, 100, 1, 0.05, 0.2), 100, 100, 1, 0.05)
    assert scalar.shape == () and float(scalar) == pytest.approx(0.2, rel=1e-14)
    empty = implied_volatility_batch([], 100, 100, 1, 0.05, return_reasons=True)
    assert empty.vol.shape == empty.reason.shape == empty.message.shape == empty.vega.shape == (0,)
    one = implied_volatility_batch(5.0, 100, 100, 1, 0.05, return_reasons=True)
    assert one.vol.shape == one.vega.shape == () and float(one.vega) > 0
    with pytest.raises(ValueError, match="option_type") as scalar_error:
        implied_volatility(5.0, 100, 100, 1, 0.05, 0.0, "straddle")
    with pytest.raises(ValueError, match="option_type") as batch_error:
        implied_volatility_batch([5.0, 5.0], 100, 100, 1, 0.05, 0.0, ["call", "straddle"])
    assert str(batch_error.value) == str(scalar_error.value) == "option_type must be 'call' or 'put', got 'straddle'"


def test_batch_quotes_without_time_value():
    # On the bound, 1e-11 below it (inside the 1e-12·max(S, K) tolerance) and 1e-11 above it (inside the margin): no
    # time value. The first two used to give 0.0 and the third whatever volatility its last digits implied (4.13 %).
    # 1e-9 above the bound is ten margins of time value, and is solved (4.61 %).
    lower, _ = no_arbitrage_bounds(100, 80, 1, 0.05, 0.0, "call")
    quotes = [lower, lower - 1e-11, lower + 1e-11, lower + 1e-9]
    result = implied_volatility_batch(quotes, 100, 80, 1, 0.05, return_reasons=True)
    assert result.reason.tolist() == ["no_time_value"] * 3 + [""]
    assert np.isnan(result.vol[:3]).all() and np.isnan(result.vega[:3]).all()
    assert result.vol[3] == pytest.approx(implied_volatility(quotes[3], 100, 80, 1, 0.05), rel=1e-12)


def _assert_batch_equals_scalar(quotes):
    """The batch gives each quote (a row of implied_volatility's arguments) the scalar path's result and error."""
    columns = [np.array(column) for column in zip(*quotes, strict=True)]
    result = implied_volatility_batch(*columns, return_reasons=True)
    for i, quote in enumerate(quotes):
        try:
            expected, reason, message = implied_volatility(*quote), "", ""
        except ImpliedVolError as error:
            expected, reason, message = math.nan, error.reason, str(error)
        assert (result.reason[i], result.message[i]) == (reason, message), quote
        np.testing.assert_allclose(result.vol[i], expected, rtol=1e-12, err_msg=str(quote))  # NaN matches NaN
    return result


# S e^(-qT) or K e^(-rT) overflows (-qT or -rT beyond about 709): rejected as not finite on both paths. Before, the
# out-of-the-money price was inf - inf = NaN or the normalised price 0: the first two rows were "exceeds 2000%" on
# the scalar path and 0.0 in the batch, and the call at r = -800 (-80,000 %) was 0.0 on both, although its
# volatility, 38.4, is above the cap.
OVERFLOWING = [
    (5.0, 100, 100, 10, -75.0, -75.0, "call"),
    (188.66, 40309.17, 56.76, 2.04, -421.77, -619.89, "call"),
    (5.0, 100, 100, 1, -800.0, 0.0, "call"),
    (5.0, 100, 100, 1, -800.0, 0.0, "put"),
    (5.0, 100, 100, 1, 0.0, -800.0, "call"),
    (5.0, 1e308, 100, 1, 0.0, -1.0, "put"),  # the discount factor is finite, the product is not
]


def test_batch_raises_no_warnings_on_extreme_inputs():
    import warnings

    extreme = [
        (1e-300, 100, 100, 1, 0.05, 0.0, "call"),
        (5.0, 1e300, 100, 1, 0.05, 0.0, "call"),
        (5.0, 1e-300, 100, 1, 0.05, 0.0, "call"),
        (5.0, 100, 100, 1e-12, 0.05, 0.0, "call"),
        (1e300, 100, 100, 1, 0.05, 0.0, "call"),
        (5.0, 100, 100, 1e4, 2.0, 0.0, "call"),  # e^(-rT) underflows to 0
        (5.0, 100, 100, 1, 800.0, 0.0, "call"),
        (5.0, 100, 100, 1, 800.0, 0.0, "put"),
        (5.0, 100, 100, 1, 0.0, 800.0, "put"),  # e^(-qT) underflows to 0
    ]
    # S/K under- or overflows (finite, positive S and K): a row of its own, not a ValueError for the batch.
    log_ratio = [
        (1e-210, 1e-200, 1e200, 1.0, 0.0, 0.0, "call"),
        (0.0, 5e-324, 1e300, 1.0, 0.0, 0.0, "call"),
        (5e-324, 5e-324, 1e300, 1.0, 0.0, 0.0, "put"),
        (1e-210, 1e200, 1e-200, 1.0, 0.0, 0.0, "put"),
        (10.0, 100, 100, 1.0, 0.0, 0.0, "call"),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _assert_batch_equals_scalar(extreme)
        result = _assert_batch_equals_scalar(log_ratio)
        overflowing = _assert_batch_equals_scalar(OVERFLOWING)
    # The second row, an out-of-the-money call quoted at 0, used to give a volatility of 0.
    assert result.reason.tolist() == ["above_max_vol", "no_time_value", "below_lower_bound", "above_max_vol", ""]
    assert np.isnan(result.vol[:4]).all()
    assert np.isnan(overflowing.vol).all() and set(overflowing.reason) == {"not_finite"}
    assert overflowing.message[2] == (
        "the discounted spot S e^(-qT) (100) or strike K e^(-rT) (inf) overflows; "
        "check the rate, dividend yield and time to expiry"
    )


def test_strike_ratios_beyond_double_precision():
    # ln(S/K) = ln S - ln K where S/K under- or overflows. With √(FK) = 1, ln(F/K) = ∓921 and σ√T = 40 the quote is
    # b(∓921, 40) = 1.144e-203, which the Householder steps solve (σ = 10 at T = 16), as does the batch; at T = 1 the
    # volatility, 40, is above the cap. Before, math.log(0) raised a bare ValueError (on both paths).
    mp = pytest.importorskip("mpmath")
    with mp.workdps(60):
        x = mp.log(mp.mpf(1e-200)) - mp.log(mp.mpf(1e200))
        quote = float(
            mp.exp(x / 2) * mp.erfc((-x / 40 - 20) / mp.sqrt(2)) / 2
            - mp.exp(-x / 2) * mp.erfc((-x / 40 + 20) / mp.sqrt(2)) / 2
        )
    cases = [
        (1e-200, 1e200, 16.0, "call"),
        (1e200, 1e-200, 16.0, "put"),
    ]  # out of the money: the quote is all time value
    for S, K, T, kind in cases:
        assert implied_volatility(quote, S, K, T, 0.0, 0.0, kind) == pytest.approx(10.0, rel=1e-13, abs=0)
        with pytest.raises(ImpliedVolError, match="exceeds 2000%"):
            implied_volatility(quote, S, K, 1.0, 0.0, 0.0, kind)
    S, K, T, kind = (np.array(column) for column in zip(*cases, strict=True))
    np.testing.assert_allclose(implied_volatility_batch(quote, S, K, T, 0.0, 0.0, kind), 10.0, rtol=1e-13)


def test_batch_agrees_with_the_scalar_path_next_to_the_upper_bound():
    # Quotes one, two and three ulps below the upper bound, where Let's Be Rational's maximum b_max = e^(-|x|/2)
    # decides between a volatility and "exceeds 2000%" and NumPy's exp can differ from libm's in the last bit.
    # Before the batch handed these quotes to implied_volatility, 32 of 20,000 got a different answer.
    rng = np.random.default_rng(1)
    n = 3_000
    T = 10 ** rng.uniform(-3, 1, n)
    K = 100 * np.exp(rng.uniform(-3, 3, n))
    r, q = rng.uniform(-0.05, 0.1, n), rng.uniform(0, 0.08, n)
    kind = np.where(rng.random(n) < 0.5, "call", "put")
    prices = np.array([no_arbitrage_bounds(100.0, *args)[1] for args in zip(K, T, r, q, kind, strict=True)])
    ulps = np.repeat([1, 2, 3], n // 3)
    for step in range(3):
        prices = np.where(ulps > step, np.nextafter(prices, -np.inf), prices)
    # One that used to differ: solved by implied_volatility (9.67), "exceeds 2000%" in the batch.
    case = (77.29860108934649, 100.0, 91.9566381557159, 2.938453621776211, 0.05909274252020007, 0.03955456019107547)
    prices, S = np.append(prices, case[0]), np.append(np.full(n, 100.0), case[1])
    K, T, r, q, kind = (
        np.append(K, case[2]),
        np.append(T, case[3]),
        np.append(r, case[4]),
        np.append(q, case[5]),
        np.append(kind, "put"),
    )
    result = implied_volatility_batch(prices, S, K, T, r, q, kind, return_reasons=True)
    expected = np.full(n + 1, np.nan)
    for i in range(n + 1):
        try:
            expected[i], reason, message = implied_volatility(prices[i], S[i], K[i], T[i], r[i], q[i], kind[i]), "", ""
        except ImpliedVolError as error:
            reason, message = error.reason, str(error)
        assert result.reason[i] == reason and result.message[i] == message, i
    np.testing.assert_array_equal(result.vol, expected)  # implied_volatility's own numbers
    assert result.vol[-1] == pytest.approx(9.6749, abs=1e-4)
    assert 0.1 < np.mean(np.isnan(expected)) < 0.9  # both outcomes occur (measured: 72 % are above the cap)


def test_batch_is_vectorised():
    import time

    K, T, sigma, r, q, kind = _random_quotes(10_000, seed=4, z_max=3.0)
    prices = np.array([bs.price(100, *args) for args in zip(K, T, r, sigma, q, kind, strict=True)])
    start = time.perf_counter()
    vol = implied_volatility_batch(prices, 100, K, T, r, q, kind)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0  # about 15 ms here (benchmarks/bench_iv.py); the scalar Brent loop took 7.65 s
    np.testing.assert_allclose(vol, sigma, rtol=1e-8)


# ------------------------------------------------------------------------------- quotes without time value, vega


def _band_edges(lower: float, margin: float) -> tuple[float, float]:
    """The largest quote at most ``margin`` above ``lower`` and the smallest at least ``margin`` below it.

    For a lower bound the solver computes exactly (no rate or dividend yield and ``|ln(F/K)| >= 1``, so that the
    intrinsic value is ``S - K`` or ``K - S``), ``quote - lower`` is exact too (Sterbenz): these are the last quotes
    on each side without time value, whatever the arithmetic around them.
    """
    high = lower + margin
    while high - lower > margin:
        high = np.nextafter(high, -np.inf)
    while np.nextafter(high, np.inf) - lower <= margin:
        high = np.nextafter(high, np.inf)
    low = lower - margin  # the bound check rejects a quote below this double
    return float(high), float(low)


def _reprices(vol: float, quote: float, S, K, T, r, q, kind) -> bool:
    """The 60-digit Black-Scholes-Merton price at ``vol`` is ``quote`` to within the rounding ``vol`` carries.

    The volatility's own error, ``cond·ε``, moves the price by ``σ·vega/price`` times that in relative terms, so the
    bound is ``8·cond/(cond - 1)·ε`` relative, ``cond = 1 + price/(σ·vega)`` as in :func:`_exact_quote`.
    """
    exact, cond, _ = _exact_quote(S, K, T, r, q, vol, kind)
    return abs(float(exact) - quote) <= 8 * EPS * cond / (cond - 1) * quote


@pytest.mark.oracle
@pytest.mark.parametrize(("S", "K", "kind"), [(100.0, 30.0, "call"), (30.0, 100.0, "put")])
def test_quotes_on_both_sides_of_an_exact_intrinsic_value(S, K, kind):
    # Deep in the money with no rate or dividend yield: the lower bound is 70 exactly, the margin 1e-12·100. Every
    # quote from the last double at least the margin below 70 to the last at most the margin above it has no time
    # value; one ulp beyond either edge the quote is below the bound, or is solved and reprices.
    margin = 1e-12 * max(S, K)
    lower, _ = no_arbitrage_bounds(S, K, 0.5, 0.0, 0.0, kind)
    assert lower == 70.0
    high, low = _band_edges(lower, margin)
    up, down = (float(np.nextafter(lower, side)) for side in (np.inf, -np.inf))
    flat = [lower, up, down, lower + 1e-13 * max(S, K), lower - 1e-13 * max(S, K), high, low]
    beyond = [float(np.nextafter(high, np.inf)), float(np.nextafter(low, -np.inf)), lower + 0.01]
    quotes = [(quote, S, K, 0.5, 0.0, 0.0, kind) for quote in flat + beyond]
    result = _assert_batch_equals_scalar(quotes)  # the batch gives each the scalar path's outcome and message
    assert result.reason.tolist() == ["no_time_value"] * len(flat) + ["", "below_lower_bound", ""]
    assert result.message[0] == (
        "price 70 is within 1e-10 (1e-12·max(S, K)) of the no-arbitrage lower bound 70 (discounted intrinsic value), "
        "so it has no time value to identify a volatility from: its last digits would decide it"
    )
    for i in (len(flat), len(flat) + 2):
        assert _reprices(result.vol[i], quotes[i][0], S, K, 0.5, 0.0, 0.0, kind), quotes[i]
    vega = result.vega[len(flat) :]  # nonzero but small just above the margin (1.7e-8 for the call)
    assert np.isnan(result.vega[: len(flat)]).all() and vega[0] > 0 and np.isnan(vega[1]) and vega[2] > 0


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("S", "K", "T", "r", "q", "kind"),
    [
        (100.0, 90.0, 0.01, 0.03, 0.01, "call"),  # near the money, where the intrinsic value comes from expm1
        (95.0, 100.0, 0.05, -0.01, 0.02, "put"),
        (100.0, 20.0, 2.0, 0.05, 0.0, "call"),
        (10.0, 100.0, 1.0, 0.05, 0.03, "put"),
    ],
)
def test_quotes_on_both_sides_of_the_intrinsic_value(S, K, T, r, q, kind):
    # With rates the solver's intrinsic value and the bound can differ by an ulp, so the band's edges are tested 1e-3
    # of the margin inside and outside (each rounding is about ε·max(S, K) = 2.2e-4 of it).
    margin = 1e-12 * max(S, K)
    lower, _ = no_arbitrage_bounds(S, K, T, r, q, kind)
    assert lower > 1.0  # in the money
    up, down = (float(np.nextafter(lower, side)) for side in (np.inf, -np.inf))
    flat = [lower, up, down, lower + 1e-13 * max(S, K), lower - 1e-13 * max(S, K)]
    flat += [lower + margin * (1 - 1e-3), lower - margin * (1 - 1e-3)]
    beyond = [lower + margin * (1 + 1e-3), lower - margin * (1 + 1e-3), lower + 1e-6 * max(S, K)]
    quotes = [(quote, S, K, T, r, q, kind) for quote in flat + beyond]
    result = _assert_batch_equals_scalar(quotes)
    assert result.reason.tolist() == ["no_time_value"] * len(flat) + ["", "below_lower_bound", ""]
    for i in (len(flat), len(flat) + 2):
        assert _reprices(result.vol[i], quotes[i][0], S, K, T, r, q, kind), quotes[i]


@pytest.mark.oracle
@pytest.mark.parametrize(("S", "K", "kind"), [(100.0, 200.0, "call"), (200.0, 100.0, "put")])
def test_out_of_the_money_quotes_keep_their_relative_precision(S, K, kind):
    # Out of the money the price is all time value: zero (or less, within the bound check's tolerance) has none, and
    # neither has 5e-324, which √(FK) = 141 divides to zero; every positive quote that survives the division is
    # inverted, however small, and reprices to its own relative precision. Below -margin a quote is below the bound.
    margin = 1e-12 * max(S, K)
    flat = [0.0, -1e-13 * max(S, K), -margin, 5e-324]
    solved = [1e-300, 1e-13 * max(S, K), margin * (1 - 1e-3), margin, margin * (1 + 1e-3), 1e-3]
    beyond = [float(np.nextafter(-margin, -np.inf))]
    quotes = [(quote, S, K, 0.5, 0.0, 0.0, kind) for quote in flat + solved + beyond]
    result = _assert_batch_equals_scalar(quotes)
    assert result.reason.tolist() == ["no_time_value"] * len(flat) + [""] * len(solved) + ["below_lower_bound"]
    assert result.message[0] == (
        "price 0 of an out-of-the-money option, which is worth only its time value, has no time value: no volatility "
        "can be identified from it"
    )
    assert result.message[3] == (
        "the time value of price 4.94066e-324 underflows to zero once divided by √(S e^(-qT)·K e^(-rT)) = 141.421: "
        "no volatility can be identified from it"
    )
    for i in range(len(flat), len(flat) + len(solved)):
        assert _reprices(result.vol[i], quotes[i][0], S, K, 0.5, 0.0, 0.0, kind), quotes[i]


@pytest.mark.oracle
def test_the_two_day_call_of_the_review():
    # The quant-dev review's example: spot 100, strike 60, two days, rate 4 %. The solver gave 0 % at the intrinsic
    # value, 94.5 % at 1e-13 above it and 230 % at half a cent above it; the first two have no time value. Half a
    # cent of time value is a real volatility, which reprices the quote (60-digit mpmath).
    lower, _ = no_arbitrage_bounds(100, 60, 2 / 365, 0.04, 0.0, "call")
    for quote in (lower, lower + 1e-13):
        with pytest.raises(ImpliedVolError) as error:
            implied_volatility(quote, 100, 60, 2 / 365, 0.04)
        assert error.value.reason == "no_time_value"
    vol = implied_volatility(lower + 0.005, 100, 60, 2 / 365, 0.04)
    assert vol == pytest.approx(2.3002, abs=1e-4)
    assert _reprices(vol, lower + 0.005, 100.0, 60.0, 2 / 365, 0.04, 0.0, "call")


def _exact_time_values(prices, S, K, T, r, q, kind) -> np.ndarray:
    """Each double price less its discounted intrinsic value ``max(±(S e^(-qT) - K e^(-rT)), 0)``, at 40 digits."""
    mp = pytest.importorskip("mpmath")
    out = np.empty(len(prices))
    with mp.workdps(40):
        for i, price in enumerate(prices):
            phi = 1 if kind[i] == "call" else -1
            fwd = mp.mpf(float(S)) * mp.exp(-mp.mpf(float(q[i])) * mp.mpf(float(T[i])))
            pv = mp.mpf(float(K[i])) * mp.exp(-mp.mpf(float(r[i])) * mp.mpf(float(T[i])))
            out[i] = float(mp.mpf(float(price)) - max(phi * (fwd - pv), 0))
    return out


@pytest.mark.oracle
def test_accuracy_does_not_depend_on_the_conditioning_once_there_is_time_value():
    # The conditioning-independent check the review asked for, on 20,000 random quotes over the documented domain
    # (_random_quotes: a day to 5 years, 1 % to 300 %, strikes up to 8 standard deviations away) priced in double
    # precision with black_scholes.price, each one's time value at 40 digits from its double price. Before, 703
    # (3.5 %) came back more than 1 % off the volatility that priced them (some at 0), every one with a time value of
    # at most 5.7e-16·max(S, K) and the reason "".
    K, T, sigma, r, q, kind = _random_quotes(20_000, seed=2026)
    prices = np.array([bs.price(100, *args) for args in zip(K, T, r, sigma, q, kind, strict=True)])
    result = implied_volatility_batch(prices, 100.0, K, T, r, q, kind, return_reasons=True)
    for i in range(len(prices)):  # implied_volatility decides alike (the batch repeats its arithmetic)
        try:
            implied_volatility(prices[i], 100.0, K[i], T[i], r[i], q[i], kind[i])
        except ImpliedVolError as error:
            assert error.reason == result.reason[i], i
        else:
            assert result.reason[i] == "", i
    time_value = _exact_time_values(prices, 100.0, K, T, r, q, kind)
    margin = 1e-12 * np.maximum(100.0, K)
    forward = 100.0 * np.exp((r - q) * T)
    in_the_money = np.where(kind == "call", forward > K, forward < K)
    at_upper = result.reason == "at_or_above_upper_bound"  # worth its upper bound (S e^(-qT) or K e^(-rT)) to 16 digits
    # Within 4 ε of its upper bound a quote's outcome is decided by the last bit of exp, which differs between CPUs
    # (5 of these 7 were at the bound here and 6 on a CI runner): either outcome is right there, and only there.
    upper = np.where(kind == "call", 100.0 * np.exp(-q * T), K * np.exp(-r * T))
    near_upper = prices >= upper * (1 - 4 * np.finfo(float).eps)
    assert near_upper.sum() == 7 and (at_upper <= near_upper).all()
    assert (time_value[at_upper] <= margin[at_upper]).all()
    # The rounding of the price and of the intrinsic value, a few ε·max(S, K), is 2.2e-4 of the margin per ε: within
    # 1e-3 of the margin either outcome is right.
    clear = time_value > margin * (1 + 1e-3)
    flat = in_the_money & (time_value <= margin * (1 - 1e-3)) & ~near_upper
    assert (result.reason[clear] == "").all()
    assert (result.reason[flat] == "no_time_value").all()
    assert (result.reason[~in_the_money] == "").all()  # out of the money every positive quote is inverted
    error = np.abs(result.vol - sigma) / sigma
    solved = (result.reason == "") & (time_value > margin * (1 - 1e-3))
    # Measured: at most 7.7e-6, just above the margin (6.2e-6 with NumPy 1.26 and SciPy 1.11; seeds 2, 3 and 5 give
    # 4.7e-6 to 5.4e-6, 4.4e-6 to 6.9e-6 with the older libraries, and exactly rounded prices instead of
    # black_scholes.price 5.3e-6), falling in proportion to the time value (at most 4.8e-7 beyond ten margins, 6.4e-7
    # with NumPy 1.26 and SciPy 1.11; up to 1.5e-6 for other seeds, so this second bound holds for this set only).
    assert error[solved].max() < 1e-5
    assert error[solved & (time_value > 10 * margin)].max() < 1e-6
    # Out of the money with less time value than the margin (prices from 2.6e-21 up) the price keeps its relative
    # precision, and so does the volatility. Measured: at most 1.4e-12.
    tiny = ~in_the_money & ~clear
    assert error[tiny].max() < 1e-11
    # Measured: 15,427 quotes with more time value than the margin, 2,257 in the money with less (no time value; 2,259
    # with the two near their upper bound that this machine's exp leaves below it) and 2,309 out of the money with less.
    assert clear.sum() == 15_427 and flat.sum() == 2_257 and tiny.sum() == 2_309


@pytest.mark.oracle
def test_batch_vega_is_the_black_scholes_vega_at_the_volatility_found():
    # S e^(-qT)·φ(d₁)·√T at the solved volatility, per unit of volatility, against 40-digit mpmath; NaN where rejected.
    mp = pytest.importorskip("mpmath")
    K, T, sigma, r, q, kind = _random_quotes(500, seed=5, z_max=4.0)
    prices = np.array([bs.price(100, *args) for args in zip(K, T, r, sigma, q, kind, strict=True)])
    prices[::25] = -1.0  # below the bound
    result = implied_volatility_batch(prices, 100.0, K, T, r, q, kind, return_reasons=True)
    rejected = result.reason != ""
    assert rejected.sum() == 20 and np.isnan(result.vega[rejected]).all()
    with mp.workdps(40):
        for i in np.flatnonzero(~rejected):
            s, k, t, rr, qq, v = (mp.mpf(float(a)) for a in (100.0, K[i], T[i], r[i], q[i], result.vol[i]))
            d1 = (mp.log(s / k) + (rr - qq + v * v / 2) * t) / (v * mp.sqrt(t))
            exact = s * mp.exp(-qq * t) * mp.exp(-d1 * d1 / 2) / mp.sqrt(2 * mp.pi) * mp.sqrt(t)
            # Within 4 standard deviations d₁²/2 <= 8, so the rounding of ln(F/K) moves the vega by under 1e-14.
            assert result.vega[i] == pytest.approx(float(exact), rel=1e-12), i
    # The unit of black_scholes.greeks: per 1.00 of volatility (divide by 100 for a vol point).
    i = int(np.flatnonzero(~rejected)[0])
    greeks = bs.greeks(100.0, K[i], T[i], r[i], result.vol[i], q[i], kind[i])
    assert result.vega[i] == pytest.approx(greeks["vega"], rel=1e-12)


@pytest.mark.parametrize("solver", ["engine", "tree"])
def test_american_quotes_at_and_just_above_the_zero_volatility_value(solver):
    # A deep in-the-money put (spot 60, strike 100, one year, rate 5 %) is worth 40, exercised at once, at every
    # volatility up to 38.5 % (the engine; 39.0 % on the 501-step tree). Before, 40 gave 0 and 40 + 1e-13 gave
    # 38.5 %. Now every quote within the margin 1e-12·100 of 40 has no time value; below it the bound check rejects
    # the quote (the exercise value), and above it the volatility is the edge of that range.
    from options_analyzer.implied_vol import implied_volatility_alo

    solve = implied_volatility_alo if solver == "engine" else implied_volatility_american
    margin = 1e-12 * 100
    up, down = (float(np.nextafter(40.0, side)) for side in (np.inf, -np.inf))
    for quote in (40.0, up, down, 40 + 1e-13 * 100, 40 - 1e-13 * 100, 40 + margin * 0.999, 40 - margin * 0.999):
        with pytest.raises(ImpliedVolError) as error:
            solve(quote, 60, 100, 1.0, 0.05, 0.0, "put")
        assert error.value.reason == "no_time_value", quote
    with pytest.raises(ImpliedVolError) as error:
        solve(40.0, 60, 100, 1.0, 0.05, 0.0, "put")
    assert str(error.value) == (
        "price 40 is no more than 1e-10 (1e-12·max(S, K)) above the zero-volatility American value 40, which every "
        "volatility up to some level gives to within that margin, so it has no time value to identify a volatility "
        "from"
    )
    with pytest.raises(ImpliedVolError) as error:
        solve(40 - margin * 1.001, 60, 100, 1.0, 0.05, 0.0, "put")
    assert error.value.reason == "below_lower_bound"
    edge = solve(40 + margin * 1.001, 60, 100, 1.0, 0.05, 0.0, "put")
    assert edge == pytest.approx(0.3853 if solver == "engine" else 0.3901, abs=1e-4)
    # Out of the money (an at-the-money put, zero-volatility value 0) the margin applies too, unlike the European
    # solver's: these engines are accurate only in absolute terms near zero volatility. Just above it a volatility is
    # only as good as the engine there (implied_volatility_alo's docstring): about 0.0005 % and 0.06 % at 1e-9. The
    # engine's value moves with the library versions there: 4.62e-6 with NumPy 2.5 and SciPy 1.18, 4.54e-6 with
    # NumPy 1.26 and SciPy 1.11.
    for quote in (0.0, 1e-13 * 100, margin * 0.999):
        with pytest.raises(ImpliedVolError) as error:
            solve(quote, 100, 100, 1.0, 0.05, 0.0, "put")
        assert error.value.reason == "no_time_value", quote
    # Its value rises from 0 at any positive volatility (about K·σ²/(2r)·e^(-1) for small σ), so the message blames
    # the engines' accuracy there, not a range of volatilities that all give 0.
    assert str(error.value) == (
        "price 9.99e-11 is no more than 1e-10 (1e-12·max(S, K)) above 0, the option's value at zero volatility, and "
        "the American engines are not that accurate near zero volatility, so no volatility can be identified from it"
    )
    expected, rel = (4.6e-6, 0.05) if solver == "engine" else (5.8e-4, 0.01)
    assert solve(1e-9, 100, 100, 1.0, 0.05, 0.0, "put") == pytest.approx(expected, rel=rel)


def test_american_engines_near_zero_volatility_are_as_documented():
    # implied_volatility_alo's docstring: the 501-step tree values a one-year at-the-money put (rate 5 %) at 0.046 %
    # volatility at 1.3e-13, where the engine gives 7.8e-5.
    from options_analyzer import american, binomial

    tree = binomial.price(100, 100, 1.0, 0.05, 4.6e-4, 0.0, "put", 501, american=True, method="lr")
    engine = float(american.price(100, 100, 1.0, 0.05, 4.6e-4, 0.0, option_type="put"))
    assert tree == pytest.approx(1.3e-13, rel=0.05) and engine == pytest.approx(7.8e-5, rel=0.01)


def test_american_call_without_dividends_agrees_with_the_european_solver_at_its_intrinsic_value():
    # Never exercised early, the call's zero-volatility value is the European lower bound: the engine (which values
    # it as European) and implied_volatility agree on which quotes have time value, and on their volatility.
    from options_analyzer.implied_vol import implied_volatility_alo

    S, K, T, r = 100.0, 60.0, 2 / 365, 0.04
    lower, _ = no_arbitrage_bounds(S, K, T, r, 0.0, "call")
    margin = 1e-12 * S
    for quote in (lower, lower + 1e-13 * S, lower - 1e-13 * S, lower + margin * 0.999, lower - margin * 0.999):
        for solve in (implied_volatility, implied_volatility_alo):
            with pytest.raises(ImpliedVolError) as error:
                solve(quote, S, K, T, r, 0.0, "call")
            assert error.value.reason == "no_time_value", (solve, quote)
    for quote in (lower + margin * 1.001, lower + 1e-6, lower + 0.005):
        european = implied_volatility(quote, S, K, T, r, 0.0, "call")
        engine = implied_volatility_alo(quote, S, K, T, r, 0.0, "call")
        # The engine prices the call with its own formula: at one volatility the two values differ by their
        # rounding, a fraction of ε·S, which moves the volatility by that over vega; and the engine's Brent stops
        # within its tolerance, 1e-10. That bound is 5.1e-6, 9.2e-10 and 4.4e-11 relative here; measured: 7.8e-7,
        # 8.2e-11 and 7.4e-12 (5.9e-7, 4.7e-11 and 7.5e-12 with NumPy 1.26 and SciPy 1.11).
        vega = bs.greeks(S, K, T, r, european, 0.0, "call")["vega"]
        assert abs(engine - european) <= EPS * S / vega + 1e-10, quote


def test_a_quote_at_the_crr_trees_floor_is_not_given_a_volatility():
    # The 10-step CRR tree at r = 30 % cannot represent volatilities below 9.49 %; a quote at its value there used to
    # give 0 % (as if it were the zero-volatility value), though the volatility could be anything up to 9.49 %.
    from options_analyzer import binomial

    lowest = lowest_tree_volatility(1, 0.30, 0.0, 10, "crr")
    floor = binomial.price(100, 100, 1, 0.30, lowest, 0, "put", 10, american=True, method="crr")
    with pytest.raises(ImpliedVolError) as error:
        implied_volatility_american(floor, 100, 100, 1, 0.30, 0, "put", steps=10, method="crr")
    assert error.value.reason == "below_tree_floor"
    assert str(error.value) == (
        f"price {floor:.6g} is no more than 1e-10 above {floor:.6g}, the value at 9.4868% volatility, the lowest a "
        "10-step CRR tree can represent; use more steps or the Leisen-Reimer tree"
    )
    above = implied_volatility_american(floor + 1e-6, 100, 100, 1, 0.30, 0, "put", steps=10, method="crr")
    assert above > lowest
    # A quote at the put's zero-volatility value, 0, has no time value on that tree too (it used to get
    # "below_tree_floor", as if more steps could find its volatility).
    with pytest.raises(ImpliedVolError) as error:
        implied_volatility_american(0.0, 100, 100, 1, 0.30, 0, "put", steps=10, method="crr")
    assert error.value.reason == "no_time_value"


@pytest.mark.parametrize("steps", [10, 100])
def test_a_deep_in_the_money_quote_at_the_crr_trees_floor_has_no_time_value(steps):
    # The put struck at 100 with the spot at 60 (one year, rate 5 %) is worth 40, exercised at once, at every
    # volatility up to 38.5 %, and so at the lowest the CRR tree represents (0.50 % with 100 steps, 1.58 % with 10).
    # A quote there has no time value, as on the Leisen-Reimer tree and the engine; it used to be rejected as below
    # the tree's floor ("use more steps or the Leisen-Reimer tree", which cannot help). Above the margin the tree
    # finds the edge of that range.
    from options_analyzer import binomial
    from options_analyzer.implied_vol import implied_volatility_alo

    lowest = lowest_tree_volatility(1.0, 0.05, 0.0, steps, "crr")
    assert lowest == pytest.approx(0.05 / math.sqrt(steps), rel=1e-5)
    assert binomial.price(60, 100, 1.0, 0.05, lowest, 0.0, "put", steps, american=True, method="crr") == 40.0
    margin = 1e-12 * 100
    for quote in (40.0, 40 + 1e-13 * 100, 40 + margin * 0.999, 40 - margin * 0.999):
        messages = set()
        for solve in (
            lambda quote: implied_volatility_american(quote, 60, 100, 1.0, 0.05, 0.0, "put", steps=steps, method="crr"),
            lambda quote: implied_volatility_american(quote, 60, 100, 1.0, 0.05, 0.0, "put"),
            lambda quote: implied_volatility_alo(quote, 60, 100, 1.0, 0.05, 0.0, "put"),
        ):
            with pytest.raises(ImpliedVolError) as error:
                solve(quote)
            assert error.value.reason == "no_time_value", quote
            messages.add(str(error.value))
        assert len(messages) == 1, messages  # the same zero-volatility value, 40, and the same words
    vol = implied_volatility_american(40 + margin * 1.001, 60, 100, 1.0, 0.05, 0.0, "put", steps=steps, method="crr")
    assert vol == pytest.approx({10: 0.4252, 100: 0.3956}[steps], abs=1e-4)  # 0.3901 on the Leisen-Reimer tree
