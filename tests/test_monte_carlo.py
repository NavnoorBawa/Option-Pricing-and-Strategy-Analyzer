import numpy as np
import pytest

from options_analyzer import black_scholes as bs
from options_analyzer import monte_carlo as mc

ARGS = (100, 110, 0.75, 0.04, 0.3, 0.01)


@pytest.mark.parametrize("antithetic", [False, True])
@pytest.mark.parametrize("control_variate", [False, True])
def test_estimates_are_within_four_standard_errors(antithetic, control_variate):
    res = mc.price_european(*ARGS, n_paths=100_000, seed=3, antithetic=antithetic, control_variate=control_variate)
    for kind, estimate in res.items():
        ref = bs.price(*ARGS, kind)
        assert abs(estimate.price - ref) < 4 * estimate.std_error, kind


def test_variance_reduction_lowers_standard_error():
    plain = mc.price_european(*ARGS, n_paths=50_000, seed=1, antithetic=False, control_variate=False)["call"]
    both = mc.price_european(*ARGS, n_paths=50_000, seed=1, antithetic=True, control_variate=True)["call"]
    assert both.std_error < plain.std_error / 3


def test_reproducible_with_seed_and_parity_with_common_draws():
    a = mc.price_european(*ARGS, n_paths=20_000, seed=11)
    b = mc.price_european(*ARGS, n_paths=20_000, seed=11)
    assert a["call"].price == b["call"].price
    S, K, T, r, _, q = ARGS
    parity = a["call"].price - a["put"].price - (S * np.exp(-q * T) - K * np.exp(-r * T))
    assert abs(parity) < 1e-9  # exact with the control variate on shared draws


def test_confidence_interval_coverage():
    ref = bs.price(*ARGS, "call")
    hits = sum(
        lo <= ref <= hi for lo, hi in (mc.price_european(*ARGS, n_paths=5_000, seed=s)["call"].ci95 for s in range(200))
    )
    assert 0.90 <= hits / 200 <= 0.99


def test_degenerate_inputs_return_deterministic_values():
    res = mc.price_european(100, 100, 1, 0.05, 0.0, n_paths=1000)
    assert res["call"].price == pytest.approx(100 - 100 * np.exp(-0.05))
    assert res["call"].std_error == 0.0


def test_simulated_paths_shape_and_martingale():
    times, paths = mc.simulate_paths(100, 1.0, 0.05, 0.2, 0.0, n_paths=20_000, n_steps=12, seed=0)
    assert paths.shape == (20_000, 13) and times[-1] == pytest.approx(1.0)
    assert np.all(paths[:, 0] == 100)
    assert paths[:, -1].mean() * np.exp(-0.05) == pytest.approx(100, rel=0.01)


def test_input_validation():
    with pytest.raises(ValueError):
        mc.price_european(100, 100, 1, 0.05, 0.2, n_paths=10)
    with pytest.raises(ValueError):
        mc.price_european(-1, 100, 1, 0.05, 0.2)


def test_standard_error_is_not_reported_when_no_path_crosses_the_strike():
    # Far out of the money: every path finishes below the strike, so the control-variate residuals are
    # constant and the sample cannot estimate its error (it used to report round-off, Error/SE ~ -1e8).
    res = mc.price_european(100, 140, 30 / 365, 0.05, 0.2, n_paths=100_000, seed=42)
    assert res["call"].itm_fraction == 0.0 and res["put"].itm_fraction == 1.0
    assert np.isnan(res["call"].std_error) and np.isnan(res["put"].std_error)
    ordinary = mc.price_european(*ARGS, n_paths=10_000)["call"]
    assert 0 < ordinary.itm_fraction < 1 and ordinary.std_error > 0


def test_odd_antithetic_sample_count_is_rounded_up_to_pairs():
    assert mc.terminal_prices(100, 1, 0.05, 0.2, n_paths=101).shape == (102,)


def test_standard_error_is_kept_without_control_variate_when_every_path_is_in_the_money():
    res = mc.price_european(100, 50, 0.25, 0.05, 0.2, n_paths=100_000, seed=42, control_variate=False)["call"]
    assert res.itm_fraction == 1.0 and 0 < res.std_error < 0.01
    lo, hi = res.ci95
    assert lo <= bs.price(100, 50, 0.25, 0.05, 0.2) <= hi


def _bounds(S, K, T, r, q, kind):
    """European no-arbitrage bounds ``(lower, upper)``."""
    forward, strike = S * np.exp(-q * T), K * np.exp(-r * T)
    if kind == "call":
        return max(forward - strike, 0.0), forward
    return max(strike - forward, 0.0), strike


@pytest.mark.parametrize("control_variate", [True, False])
@pytest.mark.parametrize(
    ("T", "sigma", "seed"),
    [(10, 3.0, 42), (10, 4.0, 42), (5, 4.0, 42), (5, 5.0, 42), (10, 2.5, 42), (10, 2.0, 1), (5, 4.0, 7)],
)
def test_heavy_tailed_samples_price_the_put_directly_and_the_call_by_parity(T, sigma, seed, control_variate):
    # At sigma sqrt(T) of about 6 or more the mean of S_T rests on paths too rare to draw. The control variate then
    # showed call $95.0341 against 99.9998 with SE 0.00004 (10 years, 300%), and a put of -$39.3469 (10 years, 400%);
    # without it the call read $0.0116.
    res = mc.price_european(100, 100, T, 0.05, sigma, 0.0, n_paths=100_000, seed=seed, control_variate=control_variate)
    for kind, estimate in res.items():
        lower, upper = _bounds(100, 100, T, 0.05, 0.0, kind)
        assert lower <= estimate.price <= upper, kind
        assert estimate.price == pytest.approx(bs.price(100, 100, T, 0.05, sigma, 0.0, kind), abs=1e-3 * upper), kind
        assert estimate.heavy_tailed and not estimate.control_variate
    assert res["call"].std_error == res["put"].std_error
    assert res["call"].price - res["put"].price == pytest.approx(100 - 100 * np.exp(-0.05 * T), abs=1e-9)


@pytest.mark.parametrize("control_variate", [True, False])
def test_ordinary_samples_are_not_treated_as_heavy_tailed(control_variate):
    for seed in range(20):
        res = mc.price_european(*ARGS, n_paths=20_000, seed=seed, control_variate=control_variate)
        assert not any(r.heavy_tailed for r in res.values())
        assert all(r.control_variate == control_variate for r in res.values())
    # Several years at 100% volatility (sigma sqrt(T) = 3) still uses the control variate.
    res = mc.price_european(100, 100, 9, 0.05, 1.0, 0.0, n_paths=100_000, seed=42)
    assert not res["call"].heavy_tailed and res["call"].control_variate


@pytest.mark.parametrize(
    ("S", "K", "days", "sigma", "r", "q"),
    [(934.06, 1_056_799.31, 1787, 1.4172, -0.0357, 0.42), (739.49, 8_278_445.76, 795, 2.1936, 0.1573, 0.035)],
)
def test_estimates_stay_within_the_no_arbitrage_bounds(S, K, days, sigma, r, q):
    # Deep out of the money the control-variate estimate came out negative: call -$1.8765 (SE 11, value 6.0440) and
    # -$11.3591 (SE 77, value 52.1846).
    T = days / 365
    res = mc.price_european(S, K, T, r, sigma, q, n_paths=100_000, seed=42)
    for kind, estimate in res.items():
        lower, upper = _bounds(S, K, T, r, q, kind)
        assert lower <= estimate.price <= upper, kind
        # Keeping the estimate within the bounds only ever moves it towards the value.
        assert abs(estimate.price - bs.price(S, K, T, r, sigma, q, kind)) < 2 * estimate.std_error, kind


@pytest.mark.parametrize("antithetic", [False, True])
def test_scenario_paths_end_at_the_pricing_draws(antithetic):
    # The sample-paths chart used to come from a separate simulation, so none of its end points was a scenario
    # behind the price.
    times, paths = mc.scenario_paths(100, 1, 0.05, 0.2, 0.01, 100_000, 42, antithetic, n_show=60, n_steps=50)
    draws = mc.terminal_prices(100, 1, 0.05, 0.2, 0.01, 100_000, 42, antithetic)
    assert paths.shape == (60, 51) and times[0] == 0 and times[-1] == 1
    np.testing.assert_array_equal(paths[:, -1], draws[:60])
    np.testing.assert_allclose(paths[:, 0], 100.0, rtol=1e-12)


def test_scenario_paths_are_gbm_paths():
    # A Brownian bridge to a log-normal end point has the GBM marginal at every date.
    times, paths = mc.scenario_paths(100, 2, 0.05, 0.3, 0.0, 40_000, 7, False, n_show=40_000, n_steps=8)
    log_mid = np.log(paths[:, 4] / 100)
    t = times[4]
    assert log_mid.mean() == pytest.approx((0.05 - 0.5 * 0.3**2) * t, abs=4 * 0.3 * np.sqrt(t / 40_000))
    assert log_mid.std() == pytest.approx(0.3 * np.sqrt(t), rel=0.02)
    increments = np.diff(np.log(paths), axis=1)
    assert np.corrcoef(increments[:, 0], increments[:, -1])[0, 1] == pytest.approx(0.0, abs=0.03)


def test_scenario_paths_without_volatility_follow_the_forward():
    times, paths = mc.scenario_paths(100, 1, 0.05, 0.0, 0.02, 1_000, 1, True, n_show=3, n_steps=4)
    np.testing.assert_allclose(paths, np.tile(100 * np.exp(0.03 * times), (3, 1)))
