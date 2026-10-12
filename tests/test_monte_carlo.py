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
