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


def _price_level_study(n, seed):
    """implied_volatility on exactly priced quotes: the relative error must be at most max(1e-14, 8·cond·ε).

    8 rather than 4 (as for the normalised inversion in test_lets_be_rational.py) because converting spot,
    strike, rate, dividend yield and expiry to ln(F/K) and an out-of-the-money forward price rounds too. Where an
    in-the-money quote's time value is below two units in its last place the volatility is not recoverable and
    the result must reprice the quote instead (backward stability); a quote that rounds onto the upper bound
    (a call worth the discounted spot to 16 digits) must be rejected as such.
    """
    K, T, sigma, r, q, kind = _random_quotes(n, seed)
    failures, worst = [], 0.0
    for i in range(n):
        price, cond, time_value_ulps = _exact_quote(100.0, K[i], T[i], r[i], q[i], sigma[i], kind[i])
        if price < 1e-300:
            continue
        try:
            vol = implied_volatility(float(price), 100.0, K[i], T[i], r[i], q[i], kind[i])
        except ImpliedVolError as error:
            _, upper = no_arbitrage_bounds(100.0, K[i], T[i], r[i], q[i], kind[i])
            if not (error.reason == "at_or_above_upper_bound" and float(price) >= upper):
                failures.append(f"K={K[i]!r} T={T[i]!r} sigma={sigma[i]!r} {kind[i]}: {error}")
            continue
        error = abs(vol - sigma[i]) / sigma[i]
        if time_value_ulps >= 2:
            tolerance = max(1e-14, 8 * cond * EPS)
            worst = max(worst, error / tolerance)
            if not error <= tolerance:
                failures.append(f"K={K[i]!r} T={T[i]!r} sigma={sigma[i]!r} {kind[i]}: {error:.3g} > {tolerance:.3g}")
        else:
            repriced, _, _ = _exact_quote(100.0, K[i], T[i], r[i], q[i], vol, kind[i])
            if not abs(float(repriced) - float(price)) <= 4 * EPS * float(price):
                failures.append(f"K={K[i]!r} T={T[i]!r} sigma={sigma[i]!r} {kind[i]}: {vol!r} does not reprice")
    return failures, worst


@pytest.mark.oracle
def test_price_level_accuracy_fast_subset():
    failures, worst = _price_level_study(1_000, seed=2)
    assert not failures, failures[:5]
    assert worst < 1


@pytest.mark.oracle
@pytest.mark.slow
def test_price_level_accuracy_20k():
    # Measured on these 20,000 quotes: 19,334 keep their time value and the worst error is within the tolerance,
    # at most 5.3e-15 where cond < 10. Before ln(F/K) and the forward intrinsic value were computed without
    # cancellation (log1p and expm1) near-the-money errors reached 6.7e-13.
    failures, worst = _price_level_study(20_000, seed=2)
    assert not failures, failures[:5]
    assert worst < 1


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
        ((1.0, 100, 90, 1, 0.05), "below_lower_bound"),
        ((100.0, 100, 90, 1, 0.05), "at_or_above_upper_bound"),
        ((99.9999, 100, 100, 0.01, 0.0), "above_max_vol"),
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
    assert empty.vol.shape == empty.reason.shape == empty.message.shape == (0,)
    with pytest.raises(ValueError, match="option_type"):
        implied_volatility_batch([5.0], 100, 100, 1, 0.05, 0.0, ["straddle"])


def test_batch_zero_time_value():
    lower, _ = no_arbitrage_bounds(100, 80, 1, 0.05, 0.0, "call")
    vol = implied_volatility_batch([lower, lower - 1e-11], 100, 80, 1, 0.05)
    np.testing.assert_array_equal(vol, [0.0, 0.0])  # on the bound, and inside the 1e-12·max(S, K) tolerance


def test_batch_raises_no_warnings_on_extreme_inputs():
    import warnings

    prices = np.array([1e-300, 5.0, 5.0, 5.0, 1e300, 5.0])
    spots = np.array([100, 1e300, 1e-300, 100, 100, 100])
    expiries = np.array([1, 1, 1, 1e-12, 1, 1e4])
    rates = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 2.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        implied_volatility_batch(prices, spots, 100, expiries, rates, return_reasons=True)
        implied_volatility_batch(5.0, 100, 100, 1, 800.0)  # e^(-rT) underflows
        implied_volatility_batch(5.0, 100, 100, 1, -800.0)  # e^(-rT) overflows


def test_batch_is_vectorised():
    import time

    K, T, sigma, r, q, kind = _random_quotes(10_000, seed=4, z_max=3.0)
    prices = np.array([bs.price(100, *args) for args in zip(K, T, r, sigma, q, kind, strict=True)])
    start = time.perf_counter()
    vol = implied_volatility_batch(prices, 100, K, T, r, q, kind)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0  # about 15 ms here (benchmarks/bench_iv.py); the scalar Brent loop took 7.65 s
    np.testing.assert_allclose(vol, sigma, rtol=1e-8)
