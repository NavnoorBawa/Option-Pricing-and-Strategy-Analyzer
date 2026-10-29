import numpy as np
import pytest
from scipy.special import ndtri
from scipy.stats import binom

from options_analyzer import black_scholes as bs
from options_analyzer import monte_carlo as mc

ARGS = (100, 110, 0.75, 0.04, 0.3, 0.01)
_Z_95 = 1.959963984540054


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
    # Near the money both sides of the strike have over 3,300 of the 10,000 paths (5,000 antithetic pairs, the fewest
    # for which an interval is shown: mc.RELIABLE_SAMPLES), so every interval is shown, and they cover the value at a
    # rate inside the two-sided 99% binomial band around 95% (181 to 197 of 200): 190 of 200.
    ref = bs.price(*ARGS, "call")
    results = [mc.price_european(*ARGS, n_paths=10_000, seed=s)["call"] for s in range(200)]
    assert all(r.reliable for r in results)
    hits = sum(lo <= ref <= hi for lo, hi in (r.ci95 for r in results))
    low, high = binom.interval(0.99, len(results), 0.95)
    assert low <= hits <= high


# --- Reliability of the 95% interval ------------------------------------------------------------------------------

#: The coverage study: a 30-day call at S = 100, r = 5%, sigma = 20%, q = 0, priced as the app prices it (antithetic
#: pairs and the control variate) from 20,000 paths, with strikes at which about 10, 30, 100, 300 and 1,000 paths
#: finish on the scarce side of the strike.
STUDY = (100.0, 30 / 365, 0.05, 0.2, 0.0)
STUDY_PATHS = 20_000
STUDY_LEVELS = (10, 30, 100, 300, 1000)


def _study_strike(level: int, deep: bool) -> float:
    """The call strike at which ``level`` of ``STUDY_PATHS`` terminal prices are expected to finish in the money, or,
    ``deep`` in the money, out of it: the closed-form probability P(S_T > K) = N(d2) is ``level / STUDY_PATHS``."""
    S, T, r, sigma, q = STUDY
    d2 = ndtri(level / STUDY_PATHS)
    return S * np.exp((r - q - 0.5 * sigma**2) * T + (d2 if deep else -d2) * sigma * np.sqrt(T))


def _coverage_study(level: int, seeds: range, deep: bool) -> bool:
    """Price the study's call once per seed and check what the reliability flag promises. Returns whether the
    coverage of all intervals, shown or not, falls short of the binomial band.

    * The flag counts the scarce side of the strike: the in-the-money paths of an out-of-the-money call, the
      out-of-the-money paths of a deep in-the-money one (the control variate takes out the payoff's linear part, so
      what is left is the put's payoff on the same draws).
    * The intervals flagged reliable cover the closed-form value at a rate of at least 93% and not below the
      two-sided 99% binomial band around 95% (a rate of 0.95 lies below it in one study in 200); inside that band
      where every run is flagged reliable. Where the flag keeps only some runs of a level (about 300 expected paths),
      their rate may lie above the band: the flag counts each run's own paths, and the runs it keeps drew at least the
      expected number into the scarce side, so their standard errors come out larger. Over 100,000 seeds at 300 the
      50,738 runs kept covered 0.957, above the band for that many (0.9475 to 0.9525), and the others 0.935.
    * Where the coverage of all intervals, price ± 1.96 SE whether shown or not, falls short of that band, the flag is
      raised (on every run, at levels so far below the threshold that no run reaches it).
    """
    S, T, r, sigma, q = STUDY
    K = _study_strike(level, deep)
    value = bs.price(S, K, T, r, sigma, q, "call")
    results = [mc.price_european(S, K, T, r, sigma, q, ("call",), STUDY_PATHS, seed)["call"] for seed in seeds]
    reliable = np.array([res.reliable for res in results])
    covered = np.array([abs(res.price - value) <= _Z_95 * res.std_error for res in results])  # NaN: not covered
    scarce = np.array([res.n_paths - res.n_itm if deep else res.n_itm for res in results])
    np.testing.assert_array_equal(reliable, scarce >= mc.RELIABLE_PATHS)
    # The interval shown is price ± 1.96 SE, and none is shown for an unreliable result.
    for res, shown in zip(results, covered, strict=True):
        low, high = res.ci95
        assert (low <= value <= high) == (shown and res.reliable)
    if reliable.any():
        hits, kept = int(covered[reliable].sum()), int(reliable.sum())
        low, high = binom.interval(0.99, kept, 0.95)
        assert hits >= low and hits >= 0.93 * kept, (level, hits, kept)
        if reliable.all():
            assert hits <= high, (level, hits, kept)
    short = bool(covered.sum() < binom.interval(0.99, len(results), 0.95)[0])
    if short:
        assert not reliable.all(), level
    if level <= 100:  # P(Poisson(100) >= 300) is below 1e-50: every run has fewer than 300 scarce paths.
        assert not reliable.any(), level
    if level >= 1000:
        assert reliable.all(), level
    return short


@pytest.mark.parametrize("level", STUDY_LEVELS)
def test_intervals_flagged_reliable_cover_the_value_95_percent_of_the_time(level):
    # Coverage over seeds 0-399 (out-of-the-money call): 0.848 at about 10 in-the-money paths, 0.900 at 30, 0.935 at
    # 100 (all flagged), 0.970 for the 199 intervals shown at 300 (0.953 for all 400) and 0.940 at 1,000 (all
    # shown). The band is 368 to 390 hits in 400: 10 paths (0.878 over 20,000 seeds) fall below it on any block of
    # seeds, 30 (0.922) on about a third of them (18 of the 50 in 0-19,999), so the shortfall is asserted at 10 only.
    # This is a regression check at fixed seeds: applied to the 49 other blocks of 400 seeds in 0-19,999 it fails on
    # none at 10, 30 and 100 paths and on 2 at 1,000, and at 300 on 9 of the 249 other blocks in 0-99,999, on the 0.93
    # floor, where about 200 intervals are shown and the floor is 1.9 standard errors below their rate of 0.957. The
    # slow study below, with 4,000 seeds, is the statistical check.
    short = _coverage_study(level, range(400), deep=False)
    if level <= 10:
        assert short


@pytest.mark.slow
@pytest.mark.parametrize("deep", [False, True], ids=["out of the money", "deep in the money"])
@pytest.mark.parametrize("level", STUDY_LEVELS)
def test_intervals_flagged_reliable_cover_the_value_95_percent_of_the_time_over_4000_seeds(level, deep):
    # Coverage over seeds 0-3999, out-of-the-money call / deep in-the-money call (2 to 3 s each): 0.872 / 0.875 at
    # about 10 scarce paths, 0.919 / 0.920 at 30, 0.935 / 0.935 at 100 (all flagged, and all below the band of 3,764
    # to 3,835 hits in 4,000), 0.951 / 0.950 for the 2,068 intervals shown at 300 (band 0.937 to 0.962; 0.943 for all
    # 4,000) and 0.946 / 0.946 at 1,000 (all shown). Over 20,000 seeds the out-of-the-money call covered 0.878, 0.922,
    # 0.940, 0.946 and 0.947 (monte_carlo.RELIABLE_PATHS), and over 100,000 seeds the runs kept at 300 covered 0.957.
    # What holds for any block of 4,000 seeds, not only this one: 10 and 30 paths fall short of the band (by 12 and
    # 4.5 standard errors of their rates over 4,000 seeds), and the runs kept at 300 stay above its lower edge (by
    # 4.5, for about 2,000 of them). At 100 paths the coverage, 0.940, sits on that edge, 0.941, so whether a block
    # falls short is a coin toss (this one does); the flag is raised on every run there regardless. At 1,000 paths,
    # 0.947 is 1.7 standard errors above the edge: about one block in 20 would fall below it (none of the 5 in
    # 0-19,999 does).
    short = _coverage_study(level, range(4000), deep=deep)
    if level <= 30:
        assert short


def test_reliability_flag_counts_the_scarce_side_of_the_strike():
    # Exactly at the threshold: strikes between consecutive terminal prices leave 299 and then 300 paths in the money.
    S, T, r, sigma, q = STUDY
    draws = np.sort(mc.terminal_prices(S, T, r, sigma, q, STUDY_PATHS, 7))[::-1]
    for n_itm in (mc.RELIABLE_PATHS - 1, mc.RELIABLE_PATHS):
        K = 0.5 * (draws[n_itm - 1] + draws[n_itm])
        res = mc.price_european(S, K, T, r, sigma, q, n_paths=STUDY_PATHS, seed=7)
        call, put = res["call"], res["put"]
        assert (call.n_itm, put.n_itm) == (n_itm, STUDY_PATHS - n_itm)
        assert call.itm_fraction == n_itm / STUDY_PATHS
        # With the control variate the deep in-the-money put rests on the call's paths too.
        assert call.reliable == put.reliable == (n_itm >= mc.RELIABLE_PATHS)
        assert np.isnan(call.ci95).all() != call.reliable and np.isnan(put.ci95).all() != put.reliable
        # The standard error is still reported, the same for both (their control-variate residuals differ by a
        # constant).
        assert call.std_error == pytest.approx(put.std_error, rel=1e-9) and call.std_error > 0
        # Averaged as it is, the put's payoff rests on its own in-the-money paths, which are almost all of them.
        plain = mc.price_european(S, K, T, r, sigma, q, n_paths=STUDY_PATHS, seed=7, control_variate=False)
        assert plain["call"].reliable == (n_itm >= mc.RELIABLE_PATHS) and plain["put"].reliable


def test_reliability_needs_enough_independent_samples():
    # Both sides of an at-the-money strike hold thousands of paths, but below 5,000 independent samples the standard
    # error is itself too uncertain: 4,999 antithetic pairs (9,998 paths) or 4,998 paths without them are flagged,
    # 5,000 are not.
    for antithetic, enough in ((True, 2 * mc.RELIABLE_SAMPLES), (False, mc.RELIABLE_SAMPLES)):
        for n_paths in (enough - 2, enough):
            results = mc.price_european(100, 100, 1, 0.05, 0.2, 0, n_paths=n_paths, seed=1, antithetic=antithetic)
            for res in results.values():
                assert res.n_samples == (n_paths // 2 if antithetic else n_paths)
                assert min(res.n_itm, n_paths - res.n_itm) > 3 * mc.RELIABLE_PATHS
                assert res.reliable == (n_paths >= enough)
                assert np.isnan(res.ci95).all() != res.reliable


@pytest.mark.slow
@pytest.mark.parametrize(
    "args",
    [ARGS, (100, 100, 2.0, 0.03, 0.6, 0.0)],
    ids=["9-month call at 30% struck 10% out of the money", "2-year call at 60% at the money"],
)
def test_too_few_independent_samples_under_cover_and_are_flagged(args):
    # With antithetic pairs and the control variate (the app's default), 500 pairs cover the value in 3,717 and 3,702
    # of 4,000 runs (seeds 0-3999), below the two-sided 99% binomial band around 95%, 3,764 to 3,835, and every run is
    # flagged; over 8,000 seeds they covered 0.931 and 0.928, 2.5 and 3.3 standard errors below the band's lower edge,
    # so almost any block of 4,000 seeds falls short. 5,000 pairs cover 3,781 and 3,788 and every run is shown; over
    # 8,000 seeds they covered 0.946 and 0.947, 1.4 and 1.7 standard errors above that edge, so about one block in 13
    # and one in 22 would fall below it (both blocks of 4,000 in 0-7,999 pass).
    value = bs.price(*args, "call")
    low, high = binom.interval(0.99, 4000, 0.95)
    for n_paths, shown in ((2 * 500, False), (2 * mc.RELIABLE_SAMPLES, True)):
        results = [mc.price_european(*args, ("call",), n_paths, seed)["call"] for seed in range(4000)]
        hits = sum(abs(res.price - value) <= _Z_95 * res.std_error for res in results)
        assert all(res.reliable == shown for res in results), n_paths
        if shown:
            assert low <= hits <= high and hits >= 0.93 * len(results), hits
        else:
            assert hits < low, hits


def test_results_state_their_reliability():
    # A result built by hand states whether it is reliable and how many paths finished in the money: before, it
    # silently defaulted to unreliable (an interval of NaN) with no path in the money.
    with pytest.raises(TypeError, match="n_itm"):
        mc.MonteCarloResult(1.0, 0.1, 1_000, True, True)
    res = mc.MonteCarloResult(1.0, 0.1, 1_000, True, True, 0.5, n_itm=500, reliable=True)
    assert res.ci95 == (1.0 - _Z_95 * 0.1, 1.0 + _Z_95 * 0.1) and res.n_samples == 500
    assert np.isnan(mc.MonteCarloResult(1.0, 0.1, 1_000, False, True, n_itm=5, reliable=False).ci95).all()


def test_results_without_an_error_estimate_or_without_sampling():
    # No path in the money: the standard error is NaN, so nothing is reliable.
    res = mc.price_european(100, 140, 30 / 365, 0.05, 0.2, 0.0, n_paths=100_000, seed=42)
    assert (res["call"].n_itm, res["put"].n_itm) == (0, 100_000)
    assert not res["call"].reliable and not res["put"].reliable
    assert np.isnan(res["call"].ci95).all() and np.isnan(res["put"].ci95).all()
    # Nothing to simulate: the value is exact, and so is its (empty) interval.
    for T, sigma in ((0.0, 0.2), (1.0, 0.0)):
        res = mc.price_european(100, 90, T, 0.05, sigma, 0.0, n_paths=1_000)
        call, put = res["call"], res["put"]
        assert (call.n_itm, put.n_itm) == (1_000, 0) and call.reliable and put.reliable
        assert call.ci95 == (call.price, call.price) and call.std_error == 0.0


def test_heavy_tailed_samples_count_the_scarce_side():
    # Ten years at 300%: the put is averaged directly and the call follows by parity. Almost no simulated price
    # finishes above the strike, so neither interval can be trusted (the prices are still right:
    # test_heavy_tailed_samples_price_the_put_directly_and_the_call_by_parity).
    res = mc.price_european(100, 100, 10, 0.05, 3.0, 0.0, n_paths=100_000, seed=42)
    call, put = res["call"], res["put"]
    assert call.heavy_tailed and call.n_itm < mc.RELIABLE_PATHS
    assert not call.reliable and not put.reliable


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


@pytest.mark.parametrize(("T", "sigma", "seed"), [(4, 2.5, 103), (5, 1.5, 185), (1, 2.0, 7)])
def test_without_control_variate_the_call_follows_from_parity_from_a_sigma_sqrt_t_of_2(T, sigma, seed):
    # The 5-standard-error check did not fire, and the plain call read $31.0160 with SE 16.16 against 98.8770 (4
    # years, 250%) and $63.6000 with SE 8.25 against 91.7617 (5 years, 150%), while the put was right.
    res = mc.price_european(100, 100, T, 0.05, sigma, 0.0, n_paths=100_000, seed=seed, control_variate=False)
    call, put = res["call"], res["put"]
    assert call.heavy_tailed and put.heavy_tailed and not call.control_variate
    assert call.price - put.price == pytest.approx(100 - 100 * np.exp(-0.05 * T), abs=1e-9)
    assert call.std_error == put.std_error
    lo, hi = call.ci95
    assert lo <= bs.price(100, 100, T, 0.05, sigma, 0.0, "call") <= hi


def test_without_control_variate_the_call_stays_close_over_many_seeds_at_large_sigma_sqrt_t():
    # Four years at 250% volatility (sigma sqrt(T) = 5): the plain call missed its interval in 11 of these 40 runs,
    # by up to tens of dollars.
    value = bs.price(100, 100, 4, 0.05, 2.5, 0.0, "call")
    errors = [
        mc.price_european(100, 100, 4, 0.05, 2.5, 0.0, n_paths=100_000, seed=seed, control_variate=False)["call"].price
        - value
        for seed in range(40)
    ]
    assert max(abs(e) for e in errors) < 0.1


def test_without_control_variate_the_call_is_averaged_below_a_sigma_sqrt_t_of_2():
    # Just below the threshold the toggle still shows plain Monte Carlo: the call is its own average (not the put
    # plus the forward), with its own, larger standard error.
    res = mc.price_european(100, 100, 4, 0.05, 0.99, 0.0, n_paths=100_000, seed=3, control_variate=False)
    call, put = res["call"], res["put"]
    assert not call.heavy_tailed and not call.control_variate
    assert call.price - put.price != pytest.approx(100 - 100 * np.exp(-0.2), abs=1e-6)
    assert call.std_error > 10 * put.std_error


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
