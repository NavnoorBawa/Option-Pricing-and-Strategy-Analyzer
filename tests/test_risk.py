import numpy as np
import pytest
from scipy.stats import norm

from options_analyzer import risk
from options_analyzer import strategies as S

MARKET = S.Market(100, 0.05, 0.2, 0.5)
STOCK = S.Strategy("Stock", (S.Leg("stock", 1),))


@pytest.mark.parametrize("confidence", [0.95, 0.975, 0.99])
def test_stock_var_matches_lognormal_quantile(confidence):
    res = risk.value_at_risk(STOCK, MARKET, risk.VaRConfig(confidence, 1, 400_000, "normal", seed=5))
    h = 1 / 252
    exact = 100 * (1 - np.exp(-0.5 * 0.04 * h + 0.2 * np.sqrt(h) * norm.ppf(1 - confidence)))
    lo, hi = res.var_ci95
    assert lo - 0.01 <= exact <= hi + 0.01
    assert res.var == pytest.approx(exact, rel=0.01)
    assert res.expected_shortfall >= res.var


def test_acerbi_tasche_estimator_definition():
    pnl = -np.arange(1.0, 101.0)  # losses 1..100
    var, es, _ = risk._var_es(pnl, 0.95)
    assert var == 96.0  # w = 5 worst outcomes: 100, 99, 98, 97, 96
    assert es == pytest.approx(98.0)


def test_student_t_preserves_variance_and_adds_tails():
    normal = risk.value_at_risk(STOCK, MARKET, risk.VaRConfig(0.99, 1, 200_000, "normal", seed=1))
    fat = risk.value_at_risk(STOCK, MARKET, risk.VaRConfig(0.99, 1, 200_000, "student_t", dof=4, seed=1))
    assert fat.std == pytest.approx(normal.std, rel=0.03)
    assert fat.excess_kurtosis > 1.0


def test_garch_variance_targeting_reproduces_horizon_volatility():
    cfg = risk.VaRConfig(0.99, 20, 200_000, "garch", seed=2)
    spot, _ = risk.simulate_horizon(MARKET, cfg)
    assert np.log(spot / 100).std() == pytest.approx(0.2 * np.sqrt(20 / 252), rel=0.02)


def test_garch_shocks_feed_variance():
    # With high alpha, volatility clusters: long-horizon returns get fatter tails than Normal.
    cfg = risk.VaRConfig(0.99, 20, 200_000, "garch", garch_alpha=0.25, garch_beta=0.70, seed=3)
    spot, _ = risk.simulate_horizon(MARKET, cfg)
    r = np.log(spot / 100)
    kurt = np.mean((r - r.mean()) ** 4) / r.var() ** 2 - 3
    assert kurt > 0.2


def test_option_var_uses_full_revaluation():
    straddle = S.build_template("Short Straddle", 100, 10)
    res = risk.value_at_risk(straddle, MARKET, risk.VaRConfig(0.99, 5, 100_000, "normal", seed=4), multiplier=100)
    assert res.var > 0 and res.expected_shortfall > res.var
    # Short gamma: the linear approximation badly understates the loss.
    assert res.var > 1.5 * res.delta_normal_var


def test_vol_of_vol_adds_vega_risk_to_short_vol_positions():
    straddle = S.build_template("Short Straddle", 100, 10)
    base = risk.value_at_risk(straddle, MARKET, risk.VaRConfig(0.99, 5, 100_000, "normal", seed=4))
    with_vol = risk.value_at_risk(straddle, MARKET, risk.VaRConfig(0.99, 5, 100_000, "normal", vol_of_vol=1.0, seed=4))
    assert with_vol.var > base.var


def test_reproducible_with_seed():
    cfg = risk.VaRConfig(0.99, 3, 20_000, "garch", seed=9)
    a = risk.value_at_risk(STOCK, MARKET, cfg)
    b = risk.value_at_risk(STOCK, MARKET, cfg)
    assert a.var == b.var


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence": 1.0},
        {"horizon_days": 0},
        {"n_paths": 10},
        {"distribution": "cauchy"},
        {"distribution": "student_t", "dof": 2.0},
        {"distribution": "garch", "garch_alpha": 0.5, "garch_beta": 0.6},
        {"spot_vol_correlation": 2.0},
    ],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        risk.VaRConfig(**kwargs)


# --- Regression tests for findings of the independent review of the rebuilt engine ---------------


def test_legs_expiring_before_the_horizon_settle_on_their_expiry_day():
    from options_analyzer import black_scholes as bs

    market = S.Market(100, 0.05, 0.30, 7 / 365)
    calendar = S.build_template("Long Call Calendar", 100, 10, market.expiry)
    res = risk.value_at_risk(calendar, market, risk.VaRConfig(0.99, 10, 200_000, "normal", seed=1), 100)
    assert res.settled_legs == 1
    # Independent path simulation: the front call settles at day 5 (ceil of 4.8 trading days).
    rng = np.random.default_rng(99)
    dv = 0.09 / 252
    s5 = 100 * np.exp(-2.5 * dv + np.sqrt(5 * dv) * rng.standard_normal(400_000))
    s10 = s5 * np.exp(-2.5 * dv + np.sqrt(5 * dv) * rng.standard_normal(400_000))
    later = -np.maximum(s5 - 100, 0) * np.exp(0.05 * 5 / 252) + bs.price(s10, 100, 14 / 365 - 10 / 252, 0.05, 0.3)
    reference = -np.quantile(100 * (later - float(S.position_value(calendar, market))), 0.01)
    assert res.var == pytest.approx(reference, rel=0.03)
    assert res.var > 500  # the old horizon-spot settlement reported about 73


@pytest.mark.parametrize(
    "cfg",
    [
        {"distribution": "normal"},
        {"distribution": "student_t", "dof": 3},
        {"distribution": "garch", "garch_alpha": 0.3, "garch_beta": 0.69, "horizon_days": 60},
    ],
)
def test_implied_vol_multiplier_is_a_martingale(cfg):
    config = risk.VaRConfig(
        **(
            {
                "confidence": 0.99,
                "horizon_days": 20,
                "n_paths": 200_000,
                "vol_of_vol": 1.5,
                "spot_vol_correlation": -0.9,
                "seed": 2,
            }
            | cfg
        )
    )
    _, multiplier = risk.simulate_horizon(MARKET, config)
    assert multiplier.mean() == pytest.approx(1.0, abs=0.01)
    assert multiplier.max() < 100


@pytest.mark.parametrize(
    "kwargs",
    [
        {"distribution": "garch", "garch_alpha": float("nan")},
        {"distribution": "student_t", "dof": float("inf")},
        {"drift": float("nan")},
        {"horizon_days": 1.9},
        {"n_paths": 1_000, "confidence": 0.999},
    ],
)
def test_config_rejects_non_finite_and_thin_tails(kwargs):
    with pytest.raises(ValueError):
        risk.VaRConfig(**kwargs)


def test_riskless_position_has_no_higher_moments():
    box = S.Strategy("box", (S.Leg("call", 1, 90), S.Leg("call", -1, 110), S.Leg("put", 1, 110), S.Leg("put", -1, 90)))
    res = risk.value_at_risk(box, MARKET, risk.VaRConfig(0.99, 1, 10_000, "normal"))
    assert np.isnan(res.skewness) and np.isnan(res.excess_kurtosis) and res.std == 0.0
