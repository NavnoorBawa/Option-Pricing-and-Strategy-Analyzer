from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from options_analyzer import black_scholes as bs
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
    market = S.Market(100, 0.05, 0.30, 7 / 365)
    calendar = S.build_template("Long Call Calendar", 100, 10, market.expiry)
    res = risk.value_at_risk(calendar, market, risk.VaRConfig(0.99, 10, 200_000, "normal", seed=1), 100)
    assert res.settled_legs == 2  # the 14-day leg (9.67 trading days) settles on day 10, the horizon
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


# --- Final review -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("days", "horizon", "settled"),
    [(30, 21, 4), (30, 20, 0), (30, 22, 4), (1, 1, 4), (7, 5, 4), (7, 4, 0)],
)
def test_legs_expiring_during_the_last_day_count_as_settled(days, horizon, settled):
    # Final review: a 30-day condor at a 21-day horizon expires at 20.7 trading days, but the caption left out the
    # settlement sentence because the legs settle on day 21, the horizon itself.
    market = S.Market(100, 0.05, 0.30, days / 365)
    condor = S.build_template("Iron Condor", 100, 10)
    config = risk.VaRConfig(0.99, horizon, 20_000, "normal", seed=3)
    res = risk.value_at_risk(condor, market, config, 100)
    assert res.settled_legs == settled == 4 * (days / 365 * 252 < horizon)
    if settled and np.ceil(days / 365 * 252) == horizon:
        # Every leg expires during the horizon's last day: the P&L is the payoff on the horizon price (no interest
        # left to earn) minus today's value.
        spot, _ = risk.simulate_horizon(market, config)
        payoff = (
            np.maximum(80 - spot, 0) - np.maximum(90 - spot, 0) - np.maximum(spot - 110, 0) + np.maximum(spot - 120, 0)
        )
        expected = 100 * (payoff - float(S.position_value(condor, market)))
        assert res.pnl == pytest.approx(expected, abs=1e-9)


# --- Implied vol frozen and moving (the Risk tab shows both) -------------------------------------------------------

#: The case of the review panel's market maker (docs/reviews/panel-v3.0.md, finding 2): a 45-day at-the-money
#: short straddle on an index.
SPX = S.Market(5800, 0.043, 0.16, 45 / 365, 0.013)
STRADDLE = S.build_template("Short Straddle", 5800, 10)
#: The Risk tab's default implied-vol dynamics, an index's (VIX-like): vol of vol 100 % a year, correlation -0.7.
INDEX_VOL = {"vol_of_vol": 1.0, "spot_vol_correlation": -0.7}


def test_frozen_implied_vol_var_is_full_revaluation_on_the_simulated_spot():
    # vol_of_vol = 0 (the library default and the Risk tab's spot-only run): the P&L is the straddle re-priced in
    # closed form, one trading day later at the same implied vol, on the seed's Gaussian spot draws. Rebuilt here from
    # the model's definition, not from risk.py: the paths agree to 3e-10 dollars (measured; the straddle is worth
    # about $26,000 and the two sides order the floating-point operations differently), VaR and ES to rounding. A
    # change to the frozen-vol run, such as a vol shock drawn before the spot shocks, fails here.
    K, r, q, sigma, T, h = 5800.0, 0.043, 0.013, 0.16, 45 / 365, 1 / 252
    z = np.random.default_rng(7).standard_normal(100_000)
    spot = 5800.0 * np.exp((-q - 0.5 * sigma**2) * h + sigma * np.sqrt(h) * z)

    def straddle(s, t):
        return bs.price(s, K, t, r, sigma, q, "call") + bs.price(s, K, t, r, sigma, q, "put")

    pnl = -100 * (straddle(spot, T - h) - straddle(5800.0, T))
    worst = np.sort(pnl)[:1000]  # the floor(100,000 x 1 %) worst outcomes
    res = risk.value_at_risk(STRADDLE, SPX, risk.VaRConfig(0.99, 1, 100_000, "normal", seed=7), 100)
    np.testing.assert_allclose(res.pnl, pnl, rtol=0, atol=1e-8)
    assert res.var == pytest.approx(-worst[-1], rel=1e-12)
    assert res.expected_shortfall == pytest.approx(-worst.mean(), rel=1e-12)


def test_frozen_implied_vol_understates_the_var_of_a_short_straddle_by_27_percent():
    # The panel's numbers, one contract (100 shares), one day, 99 %, 100,000 paths, seed 7: VaR $2,833 and ES $3,683
    # with implied vol frozen (vol_of_vol = 0, unchanged by making index dynamics the Risk tab's default), against
    # $3,862 and $4,737 with index dynamics. Pinned to 1e-9 relative: far above rounding (the frozen run's VaR and ES
    # match the rebuild above to 3e-16) and far below any change of the model. They rest on NumPy's Generator giving
    # a seed the same normal draws in every supported NumPy, which the surface goldens (seeded sample_quotes noise)
    # rely on too; should that ever change, the rebuild above still checks the frozen run.
    frozen = risk.value_at_risk(STRADDLE, SPX, risk.VaRConfig(0.99, 1, 100_000, seed=7), 100)
    moving = risk.value_at_risk(STRADDLE, SPX, risk.VaRConfig(0.99, 1, 100_000, seed=7, **INDEX_VOL), 100)
    assert (frozen.var, frozen.expected_shortfall) == pytest.approx((2832.9076587059944, 3683.1833471947934), rel=1e-9)
    assert (moving.var, moving.expected_shortfall) == pytest.approx((3861.9450277254145, 4737.131879977531), rel=1e-9)
    assert 1 - frozen.var / moving.var == pytest.approx(0.2665, abs=1e-4)
    # Far outside the simulation noise: even the two 95 % intervals are apart (the runs share their spot paths, so
    # the noise of the gap itself is smaller still).
    assert frozen.var_ci95[1] < moving.var_ci95[0]
    # The Methodology quotes this case: its numbers must be these.
    text = " ".join((Path(risk.__file__).parent / "ui" / "methodology" / "risk.md").read_text(encoding="utf-8").split())
    assert rf"VaR is \${moving.var:,.0f} with implied-vol moves and \${frozen.var:,.0f} without them" in text
    assert f"understates it by {100 * (1 - frozen.var / moving.var):.0f} %" in text


@pytest.mark.parametrize(
    ("days", "vol", "cfg"),
    [
        (7, 0.16, {}),
        (45, 0.16, {}),
        (180, 0.30, {}),
        (365, 0.16, {"horizon_days": 10}),
        (45, 0.16, {"horizon_days": 5, "distribution": "student_t", "dof": 4}),
        (45, 0.16, {"horizon_days": 5, "distribution": "garch"}),
        (45, 0.16, {"spot_vol_correlation": 0.0}),
    ],
)
def test_index_vol_dynamics_add_to_the_var_and_es_of_a_short_straddle(days, vol, cfg):
    # Short vega and short gamma: implied-vol moves widen the loss tail, so the VaR and ES with them exceed the
    # spot-only ones on the same spot paths. The smallest gap here is the 7-day straddle's VaR, +1.2 % (a week from
    # expiry, vega is small next to gamma); the 45-day one gains 36 % at this seed, and 46 % without the correlation.
    market = S.Market(5800, 0.043, vol, days / 365, 0.013)
    base = {"confidence": 0.99, "horizon_days": 1, "n_paths": 50_000, "spot_vol_correlation": -0.7, "seed": 3} | cfg
    moving = risk.value_at_risk(STRADDLE, market, risk.VaRConfig(**(base | {"vol_of_vol": 1.0})), 100)
    frozen = risk.value_at_risk(STRADDLE, market, risk.VaRConfig(**(base | {"vol_of_vol": 0.0})), 100)
    assert moving.var > frozen.var
    assert moving.expected_shortfall > frozen.expected_shortfall


@pytest.mark.parametrize(
    "cfg",
    [
        {"distribution": "normal"},
        {"distribution": "student_t", "dof": 4},
        {"distribution": "garch", "horizon_days": 10},
    ],
)
def test_implied_vol_moves_leave_the_spot_paths_unchanged(cfg):
    # The Risk tab calls the gap between its two VaRs the vega risk. That holds because one seed gives both runs the
    # same spot paths: the simulation draws every spot shock first, and vol shocks only for a positive vol of vol.
    # So a position without vega (stock) has the same P&L, path by path, either way.
    base = {"confidence": 0.99, "horizon_days": 5, "n_paths": 20_000, "seed": 11} | cfg
    moving, frozen = risk.VaRConfig(**base, **INDEX_VOL), risk.VaRConfig(**base)
    spot_moving, multiplier = risk.simulate_horizon(MARKET, moving)
    spot_frozen, ones = risk.simulate_horizon(MARKET, frozen)
    assert np.array_equal(spot_moving, spot_frozen)
    assert np.all(ones == 1.0) and multiplier.std() > 0.1
    pnl_moving, pnl_frozen = (risk.value_at_risk(STOCK, MARKET, config, 100).pnl for config in (moving, frozen))
    assert np.array_equal(pnl_moving, pnl_frozen)
