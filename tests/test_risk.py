import doctest
import math
from pathlib import Path

import mpmath
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.special import erfcinv, erfinv
from scipy.stats import binom, norm

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
    # The Methodology quotes this case: its numbers must be these, and the spread of the understatement over seeds 0
    # to 9 (24.2 % at seed 9 to 27.0 % at seed 4, measured; ten pairs of runs, about 1.5 s).
    text = " ".join((Path(risk.__file__).parent / "ui" / "methodology" / "risk.md").read_text(encoding="utf-8").split())
    assert "100,000 paths, seed 7, the app's default" in text
    assert rf"VaR is \${moving.var:,.0f} with implied-vol moves and \${frozen.var:,.0f} without them" in text
    assert f"understates it by {100 * (1 - frozen.var / moving.var):.0f} %" in text
    gaps = [
        1
        - risk.value_at_risk(STRADDLE, SPX, risk.VaRConfig(0.99, 1, 100_000, seed=seed), 100).var
        / risk.value_at_risk(STRADDLE, SPX, risk.VaRConfig(0.99, 1, 100_000, seed=seed, **INDEX_VOL), 100).var
        for seed in range(10)
    ]
    assert f"({100 * min(gaps):.0f} % to {100 * max(gaps):.0f} % over seeds 0 to 9)" in text


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
    # spot-only ones on the same spot paths. The smallest gaps here are the 7-day straddle's, +1.2 % on VaR and +0.9 %
    # on ES (a week from expiry, vega is small next to gamma), then the Student-t ES's, +2.2 %; the 45-day VaR gains
    # 36 % at this seed, and 46 % without the correlation (measured).
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


# --- Delta-gamma-theta VaR, the analytic comparator (roadmap RK-6) -------------------------------------------------

#: The comparator's examples (docs and roadmap): spot 100, volatility 20 %, rate 4 %, 30 days, strikes 5 apart.
DG_MARKET = S.Market(100, 0.04, 0.2, 30 / 365)
#: Ten templates, at least one of every kind: single option, stock and option, vertical spread, volatility (long
#: and short gamma), range-bound, synthetic and time spread.
DG_TEMPLATES = (
    "Long Call",
    "Covered Call",
    "Bull Call Spread",
    "Long Straddle",
    "Short Straddle",
    "Short Strangle",
    "Iron Condor",
    "Long Call Butterfly",
    "Risk Reversal",
    "Long Call Calendar",
)


def _dollar_greeks(name: str) -> tuple[float, float, float]:
    """Delta, gamma and theta (per year) of one contract (100 shares) of a template in :data:`DG_MARKET`."""
    greeks = S.position_greeks(S.build_template(name, 100, 5, DG_MARKET.expiry), DG_MARKET)
    return tuple(100 * float(greeks[k]) for k in ("delta", "gamma", "theta"))


def _quadratic(delta, gamma, theta, spot, vol, horizon_days):
    """The delta-gamma-theta P&L as ``c + a z + b z²`` in a standard Normal ``z``, and its size |c| + |a| + |b|."""
    tau = horizon_days / 252
    s = spot * vol * math.sqrt(tau)
    c, a, b = theta * tau, delta * s, 0.5 * gamma * s * s
    return c, a, b, abs(c) + abs(a) + abs(b)


def _reference(delta, gamma, theta, spot, vol, horizon_days, confidence):
    """VaR and ES of the delta-gamma-theta P&L in 30-digit mpmath, without the chi-square or any quadrature.

    In a standard Normal z the P&L is c + a z + b z². The quantile q solves P(P&L <= q) = 1 - confidence by
    bisection on the Normal probability of the set {P&L <= q} (an interval around the vertex z = -a / (2b) when
    b > 0, the two tails beyond the vertex when b < 0, a half-line when b = 0), and ES is minus the closed form of
    the integral of (c + a z + b z²) φ(z) over that set (∫φ = Φ, ∫zφ = -φ, ∫z²φ = Φ - zφ), over 1 - confidence.
    """
    with mpmath.workdps(30):
        mpf, inf = mpmath.mpf, mpmath.inf
        tau = mpf(horizon_days) / 252
        s = mpf(spot) * mpf(vol) * mpmath.sqrt(tau)
        c, a, b = mpf(theta) * tau, mpf(delta) * s, mpf(gamma) * s * s / 2
        tail = 1 - mpf(confidence)
        if a == 0 and b == 0:  # a constant P&L
            return float(-c), float(-c)

        def loss_set(q):
            if b == 0:
                root = (q - c) / a
                return [(-inf, root)] if a > 0 else [(root, inf)]
            vertex = -a / (2 * b)
            r2 = (q - c) / b + vertex**2  # the set is (z - vertex)² <= r2 when b > 0, >= r2 when b < 0
            if b > 0:
                return [] if r2 <= 0 else [(vertex - mpmath.sqrt(r2), vertex + mpmath.sqrt(r2))]
            return [(-inf, inf)] if r2 <= 0 else [(-inf, vertex - mpmath.sqrt(r2)), (vertex + mpmath.sqrt(r2), inf)]

        size = abs(c) + abs(a) + abs(b)
        lo, hi = c - 200 * size, c + 200 * size  # the quantile is within 10 standard deviations: 10 |a| + 100 |b|
        for _ in range(110):  # 400 size / 2^110 = 3e-31 size
            mid = (lo + hi) / 2
            if sum(mpmath.ncdf(right) - mpmath.ncdf(left) for left, right in loss_set(mid)) < tail:
                lo = mid
            else:
                hi = mid
        q = (lo + hi) / 2

        def phi(z):
            return mpmath.npdf(z) if mpmath.isfinite(z) else mpf(0)

        def z_phi(z):
            return z * mpmath.npdf(z) if mpmath.isfinite(z) else mpf(0)

        def integral(left, right):
            mass = mpmath.ncdf(right) - mpmath.ncdf(left)
            return c * mass + a * (phi(left) - phi(right)) + b * (mass + z_phi(left) - z_phi(right))

        return float(-q), float(-sum(integral(left, right) for left, right in loss_set(q)) / tail)


def _positions(n: int, seed: int) -> list[tuple]:
    """Random positions over the domain the docstring's accuracy statement covers: delta 1e-3 to 1e3 shares and gamma
    1e-4 to 1e2 shares per $1 of either sign (one in twenty zero), theta to 1e4 a year, spot 1 to 1e4, volatility
    1 % to 200 %, 1 to 60 trading days, confidence 51 % to 99.99 %."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        delta = rng.normal() * 10 ** rng.uniform(-3, 3) * (rng.random() > 0.05)
        gamma = rng.normal() * 10 ** rng.uniform(-4, 2) * (rng.random() > 0.05)
        theta = rng.normal() * 10 ** rng.uniform(-1, 4)
        confidence = float(rng.choice([0.51, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9999]))
        spot, vol = 10 ** rng.uniform(0, 4), 10 ** rng.uniform(-2, 0.3)
        out.append((float(delta), float(gamma), float(theta), spot, vol, int(rng.integers(1, 61)), confidence))
    return out


#: Positions where the method changes: no gamma, no delta (λ = 0, the central chi-square) with either sign of gamma,
#: neither (a constant P&L), the far tail, a confidence just above one half, the straddle of the examples, and a delta
#: so small that λ is subnormal (5e-161 shares: λ = 2e-321, where ncx2.isf put the VaR $0.016 too high, 2e-4 of it,
#: before the function read the central quantile below λ = 1e-30; found by Hypothesis).
DG_EDGES = [
    (25.0, 0.0, -300.0, 100, 0.2, 1, 0.99),
    (0.0, 4.0, -900.0, 100, 0.2, 1, 0.99),
    (0.0, -4.0, 900.0, 100, 0.2, 5, 0.9999),
    (0.0, 0.0, 1000.0, 100, 0.2, 10, 0.95),
    (-6.854, -13.864, 2781.9, 100, 0.2, 1, 0.5001),
    (40.0, 1e-9, 0.0, 100, 0.2, 1, 0.99),
    (-40.0, -1e-9, 0.0, 100, 0.2, 1, 0.99),
    (*_dollar_greeks("Short Straddle"), 100, 0.2, 1, 0.99),
    (5.677225589706171e-161, -1.0, -21343.0, 100, 0.2, 1, 0.75),
    (-5.677225589706171e-161, 1.0, 21343.0, 100, 0.2, 1, 0.75),
]


@pytest.mark.parametrize("position", _positions(40, seed=6) + DG_EDGES)
def test_delta_gamma_var_matches_an_independent_mpmath_reference(position):
    # The chi-square quantile and the ES quadrature against 30-digit mpmath that uses neither. Over 3,000 random
    # positions like these and 400 extreme ones (delta down to 1e-300, gamma down to 1e-300, the switch to the Normal
    # quantile, confidence up to 99.9999 %) the worst errors were 1.9e-14 (VaR) and 1.3e-13 (ES) of the P&L's size
    # |θ| h/252 + |Δ| s + ½ |Γ| s², in SciPy 1.11 and 1.18; the tolerances are three times those. The integration's
    # own error estimate, es_error, was at most 1.6e-9 of the size over 20,000 random positions.
    var, es = _reference(*position)
    result = risk.delta_gamma_var(*position)
    *_, size = _quadratic(*position[:6])
    assert result.var == pytest.approx(var, rel=0, abs=6e-14 * size)
    assert result.expected_shortfall == pytest.approx(es, rel=0, abs=4e-13 * size)
    assert result.es_error <= 5e-9 * size


@pytest.mark.parametrize(("delta", "theta", "horizon", "confidence"), [(100, 0, 1, 0.99), (-37.5, 2_500, 10, 0.975)])
def test_without_gamma_it_is_the_delta_normal_var_less_the_time_decay(delta, theta, horizon, confidence):
    # Roadmap RK-6, acceptance (1). With no gamma the P&L is Normal: VaR is the delta-normal |Δ| S σ √(h/252) z_α
    # less the time decay θ h/252, and ES is |Δ| S σ √(h/252) φ(z_α) / (1 - α) less the same, both in closed form here.
    # VaR to rounding; ES within the integration's error estimate and rounding (1e-14 of the P&L's size).
    tau = horizon / 252
    s = 100 * 0.2 * math.sqrt(tau)
    z = norm.ppf(confidence)
    result = risk.delta_gamma_var(delta, 0.0, theta, 100, 0.2, horizon, confidence)
    assert result.var == pytest.approx(abs(delta) * s * z - theta * tau, rel=1e-14)
    exact_es = abs(delta) * s * norm.pdf(z) / (1 - confidence) - theta * tau
    size = abs(theta * tau) + abs(delta) * s
    assert abs(result.expected_shortfall - exact_es) <= result.es_error + 1e-14 * size


@pytest.mark.parametrize(
    "strategy", [STOCK, S.build_template("Synthetic Long Stock", 100, 5)], ids=["stock", "synthetic"]
)
def test_var_result_stores_the_comparator_without_gamma_as_delta_normal_less_the_time_decay(strategy):
    # Acceptance (1) through value_at_risk, on positions without gamma: shares paying a 3 % dividend yield (theta
    # S q a year, the dividend they accrue: $5.95 over 5 days for 100 shares) and a synthetic long stock (call minus
    # put: their gammas cancel exactly; its theta is the carry of the strike). The comparator is the delta-normal VaR
    # that the result shows beside it, less the horizon's theta.
    market = S.Market(100, 0.05, 0.2, 0.5, 0.03)
    greeks = S.position_greeks(strategy, market)
    assert greeks["gamma"] == 0.0 and greeks["theta"] != 0.0
    res = risk.value_at_risk(strategy, market, risk.VaRConfig(0.99, 5, 10_000, seed=1), 100)
    theta_h = 100 * greeks["theta"] * 5 / 252
    assert res.delta_gamma.var == pytest.approx(res.delta_normal_var - theta_h, rel=1e-14)


@pytest.mark.parametrize(("delta", "sign"), [(30.0, 1), (30.0, -1), (-30.0, 1), (-30.0, -1)])
def test_var_and_es_are_continuous_in_gamma_and_tend_to_the_gamma_free_values(delta, sign):
    # As gamma goes to zero the vertex of the parabola moves out without bound, where the chi-square fails, so the
    # function reads the quantile off the Normal distribution once the loss region's part beyond the vertex holds
    # less than 2^-56 of the tail. (a) No jump where it switches: on both sides of the switch the VaR and ES agree with
    # the mpmath reference to the tolerances of the test above. (b) Below the switch the VaR and ES are those of the
    # Normal tail, -c - a z - b z² at z = ∓z_α and -c + |a| φ(z_α)/(1-α) - b (1 + z_α φ(z_α)/(1-α)) in closed form,
    # which tend linearly in gamma to the gamma-free values, down to gamma = ±1e-16.
    spot, vol, horizon, confidence = 100.0, 0.2, 1, 0.99
    s = spot * vol * math.sqrt(horizon / 252)
    z_alpha = norm.ppf(confidence)
    z_far = -norm.ppf(risk._FAR_TAIL * (1 - confidence))
    switch = sign * 2 * abs(delta) / ((z_alpha + z_far) * s)  # |Δ s| = (z_α + z_f) |Γ| s² / 2
    # The function's own test of which side it is on: the Normal quantile below the switch, the chi-square above.
    below, above = switch * (1 - 1e-12), switch * (1 + 1e-12)
    assert abs(delta * s) >= (z_alpha + z_far) * 0.5 * abs(below) * s * s
    assert abs(delta * s) < (z_alpha + z_far) * 0.5 * abs(above) * s * s
    for gamma in (below, above, switch * 0.999, switch * 1.001):
        var, es = _reference(delta, gamma, 50.0, spot, vol, horizon, confidence)
        result = risk.delta_gamma_var(delta, gamma, 50.0, spot, vol, horizon, confidence)
        size = _quadratic(delta, gamma, 50.0, spot, vol, horizon)[-1]
        assert result.var == pytest.approx(var, rel=0, abs=6e-14 * size)
        assert result.expected_shortfall == pytest.approx(es, rel=0, abs=4e-13 * size)
    tail_mass = norm.pdf(z_alpha) / (1 - confidence)
    flat = risk.delta_gamma_var(delta, 0.0, 50.0, spot, vol, horizon, confidence)
    for k in range(1, 17):
        gamma = sign * 10.0**-k
        c, a, b, size = _quadratic(delta, gamma, 50.0, spot, vol, horizon)
        result = risk.delta_gamma_var(delta, gamma, 50.0, spot, vol, horizon, confidence)
        assert result.var == pytest.approx(-c + abs(a) * z_alpha - b * z_alpha**2, rel=0, abs=1e-14 * size)
        exact_es = -c + abs(a) * tail_mass - b * (1 + z_alpha * tail_mass)
        assert abs(result.expected_shortfall - exact_es) <= result.es_error + 1e-14 * size
        assert abs(result.var - flat.var) <= 0.5 * s * s * 10.0**-k * z_alpha**2 + 1e-14 * size
    assert risk.delta_gamma_var(delta, sign * 1e-16, 50.0, spot, vol, horizon, confidence).var == pytest.approx(
        flat.var, rel=1e-15
    )


@pytest.mark.parametrize("sign", [1, -1])
def test_without_delta_the_var_is_linear_in_gamma_down_to_zero(sign):
    # With no delta the P&L is c + b z²: a central chi-square, the vertex at the centre. The loss region is |z| <= r
    # for long gamma, with erf(r / √2) = 1 - α, and |z| >= r for short gamma, with erfc(r / √2) = 1 - α, so in
    # closed form VaR = -c - b r² and ES = -c - b (1 ∓ 2 r φ(r) / (1 - α)): linear in gamma, and -c at gamma = 0.
    confidence, theta = 0.99, -300.0 * sign
    tail = 1 - confidence
    r = math.sqrt(2) * (erfinv(tail) if sign > 0 else erfcinv(tail))
    for k in range(17):
        gamma = sign * 10.0**-k
        c, _, b, size = _quadratic(0.0, gamma, theta, 100, 0.2, 1)
        result = risk.delta_gamma_var(0.0, gamma, theta, 100, 0.2, 1, confidence)
        assert result.var == pytest.approx(-c - b * r * r, rel=0, abs=1e-14 * size)
        exact_es = -c - b * (1 - sign * 2 * r * norm.pdf(r) / tail)
        assert abs(result.expected_shortfall - exact_es) <= result.es_error + 1e-14 * size
    assert risk.delta_gamma_var(0.0, 0.0, theta, 100, 0.2, 1, confidence).var == pytest.approx(-theta / 252, rel=1e-15)


#: One Normal sample for every template, so that the ten checks below share their draws.
DG_PATHS = 400_000
DG_SEED = 2024


@pytest.mark.parametrize("name", DG_TEMPLATES)
def test_delta_gamma_var_is_the_quantile_of_the_simulated_quadratic_pnl(name):
    # Acceptance (2): 400,000 draws of the quadratic P&L itself. If the VaR is exact, the number of draws that lose
    # more than it is Binomial(400,000, 1 %); its two-sided 95 % interval is 3,877 to 4,124. Measured at this seed:
    # 4,028 for the templates that lose on a fall (Long Call, Covered Call, Bull Call Spread, Risk Reversal), 4,024
    # (Long Call Calendar), 3,932 (Long Straddle, whose losses are the small moves), 4,097, 4,084, 4,089 and 4,044
    # (Short Straddle, Short Strangle, Iron Condor, Long Call Butterfly). The ES is checked through
    # ES = VaR + E[(loss - VaR)+] / (1 - α), which holds at the exact VaR: the sample mean of (loss - VaR)+ has a
    # 95 % interval of ±1.96 standard errors, and every template's ES is inside it (Short Straddle: 92.79 in 92.36
    # to 94.31). The draws are shared, so the ten checks are not independent: at seed 1 the low draws are more
    # frequent than at most seeds and six of the ten templates fall outside together (4,173 to 4,194 exceedances). The
    # mpmath test above is the tight check; this one checks the distribution.
    delta, gamma, theta = _dollar_greeks(name)
    result = risk.delta_gamma_var(delta, gamma, theta, 100, 0.2, 1, 0.99)
    c, a, b, _ = _quadratic(delta, gamma, theta, 100, 0.2, 1)
    z = np.random.default_rng(DG_SEED).standard_normal(DG_PATHS)
    loss = -(c + a * z + b * z * z)
    exceed = int(np.count_nonzero(loss > result.var))
    assert binom.ppf(0.025, DG_PATHS, 0.01) <= exceed <= binom.ppf(0.975, DG_PATHS, 0.01)
    excess = np.maximum(loss - result.var, 0.0)
    half = 1.959963984540054 * excess.std(ddof=1) / math.sqrt(DG_PATHS)
    assert abs(result.var + excess.mean() / 0.01 - result.expected_shortfall) <= half / 0.01


def _methodology() -> str:
    return " ".join((Path(risk.__file__).parent / "ui" / "methodology" / "risk.md").read_text(encoding="utf-8").split())


def test_delta_gamma_var_is_within_5_percent_of_full_revaluation_at_one_day():
    # Acceptance (3): one day, 99 %, against full revaluation (400,000 paths, seed 7, implied vol frozen). Measured
    # gaps: Long Call +1.6 %, Covered Call +0.8 %, Bull Call Spread +3.2 % (the largest), Long Straddle -0.1 %, Short
    # Straddle +0.4 % ($70.55 against $70.29), Short Strangle -0.3 %, Iron Condor +1.1 %, Long Call Butterfly +1.4 %
    # ($20.10 against $19.82), Risk Reversal -1.4 %, Long Call Calendar +0.1 %. The simulation's own 95 % interval
    # spans at most 2.1 % of its VaR (the calendar; 1.7 % for the straddle), so 5 % bounds the comparator's gap. The
    # delta-normal VaR of the straddle is $20.09, 71 % short. ES gaps are larger, up to +4.9 % (butterfly; not
    # asserted). About 2 s.
    gaps = {}
    for name in DG_TEMPLATES:
        strategy = S.build_template(name, 100, 5, DG_MARKET.expiry)
        res = risk.value_at_risk(strategy, DG_MARKET, risk.VaRConfig(0.99, 1, 400_000, seed=7), 100)
        lo, hi = res.var_ci95
        assert hi - lo <= 0.025 * res.var, name
        gaps[name] = res.delta_gamma.var / res.var - 1
    worst = max(abs(gap) for gap in gaps.values())
    assert worst <= 0.05, gaps
    assert f"the one-day gap is at most {100 * worst:.1f} %" in _methodology()


def test_the_gap_to_full_revaluation_shows_what_the_quadratic_leaves_out():
    # Where gamma changes fast with spot, the quadratic misses the third-order term. A put ratio backspread (short
    # the 100 put, long two 95 puts) has gamma 1.67 shares per $1 at 100 and 4.80 at 98: as the spot falls its
    # convexity grows, so it loses less than the quadratic says. Measured: $35.10 against $31.81 by full
    # revaluation (95 % interval $31.76 to $31.85), +10.4 %, the only template of the 26 beyond 5 %.
    strategy = S.build_template("Put Ratio Backspread", 100, 5)
    res = risk.value_at_risk(strategy, DG_MARKET, risk.VaRConfig(0.99, 1, 400_000, seed=7), 100)
    assert res.delta_gamma.var > 1.05 * res.var_ci95[1]
    gap = res.delta_gamma.var / res.var - 1
    assert f"on a put ratio backspread it is {100 * gap:.0f} %, since its gamma grows" in _methodology()


CONFIDENCES = [0.5001, 0.6, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9999, 0.99999]


@pytest.mark.parametrize("name", DG_TEMPLATES)
def test_delta_gamma_var_and_es_rise_with_confidence(name):
    # Acceptance (4): VaR and ES never fall as the confidence rises, and ES is at least VaR (to rounding: 4e-13 of the
    # size, the ES tolerance above), from just above 50 % to 99.999 %.
    greeks = _dollar_greeks(name)
    results = [risk.delta_gamma_var(*greeks, 100, 0.2, 1, c) for c in CONFIDENCES]
    size = _quadratic(*greeks, 100, 0.2, 1)[-1]
    assert np.all(np.diff([r.var for r in results]) > 0)
    assert np.all(np.diff([r.expected_shortfall for r in results]) > 0)
    assert all(r.expected_shortfall >= r.var - 4e-13 * size for r in results)


#: Greeks for the property below; subnormal ones (below 2.2e-308 shares) are left out, since their P&L is rounded to
#: the nearest 5e-324 and VaR and ES then differ by that.
greek_values = {"allow_subnormal": False}


@given(
    delta=st.floats(-1e4, 1e4, **greek_values),
    gamma=st.floats(-1e3, 1e3, **greek_values),
    theta=st.floats(-1e5, 1e5, **greek_values),
    horizon=st.integers(1, 60),
    confidences=st.lists(st.floats(0.5, 0.99999, exclude_min=True), min_size=2, max_size=2),
)
def test_delta_gamma_var_is_monotone_in_confidence(delta, gamma, theta, horizon, confidences):
    # Acceptance (4) on any position: a higher confidence never gives a lower VaR or ES, and ES is at least VaR. Two
    # confidences can be closer than the errors of the results, so VaR may fall by 6e-14 of the P&L's size and ES by
    # 4e-13 of it plus the integration's error estimate, the tolerances of the mpmath test above.
    low, high = sorted(confidences)
    a, b = (risk.delta_gamma_var(delta, gamma, theta, 100, 0.2, horizon, c) for c in (low, high))
    size = _quadratic(delta, gamma, theta, 100, 0.2, horizon)[-1]
    assert b.var >= a.var - 6e-14 * size
    assert b.expected_shortfall >= a.expected_shortfall - a.es_error - b.es_error - 4e-13 * size
    assert a.expected_shortfall >= a.var - a.es_error - 4e-13 * size


def test_var_result_carries_the_comparator_of_the_position_greeks():
    # value_at_risk stores delta_gamma_var of the position's Greeks (times the multiplier), the market volatility and
    # the configured horizon and confidence: the same for any return model, vol of vol, seed or number of paths.
    straddle = S.build_template("Short Straddle", 100, 5)
    expected = risk.delta_gamma_var(*_dollar_greeks("Short Straddle"), 100, 0.2, 3, 0.975)
    for cfg in (
        {},
        {"distribution": "student_t", "dof": 4},
        {"distribution": "garch"},
        {"vol_of_vol": 1.0, "seed": 3, "n_paths": 20_000},
    ):
        config = risk.VaRConfig(**({"confidence": 0.975, "horizon_days": 3, "n_paths": 10_000} | cfg))
        assert risk.value_at_risk(straddle, DG_MARKET, config, 100).delta_gamma == expected


def test_var_result_has_no_comparator_when_a_greek_is_not_a_number(monkeypatch):
    real = risk.position_greeks
    monkeypatch.setattr(risk, "position_greeks", lambda *args: real(*args) | {"gamma": float("nan")})
    res = risk.value_at_risk(S.build_template("Short Straddle", 100, 5), DG_MARKET, risk.VaRConfig(n_paths=10_000))
    assert res.delta_gamma is None and res.var > 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"delta": float("nan")}, "delta must be a finite number"),
        ({"gamma": float("inf")}, "gamma must be a finite number"),
        ({"theta": "12"}, "theta must be a number"),
        ({"delta": np.array([1.0, 2.0])}, "delta must be a single number"),
        ({"spot": 0.0}, "spot must be positive"),
        ({"vol": -0.2}, "vol must be positive"),
        ({"horizon_days": 0}, "horizon_days must be positive"),
        ({"es_rtol": 0.0}, "es_rtol must be positive"),
        ({"confidence": 0.5}, "confidence must be between 0.5 and 1"),
        ({"confidence": 1.0}, "confidence must be between 0.5 and 1"),
        ({"delta": 1e308, "spot": 1e10}, "overflows"),
    ],
)
def test_delta_gamma_var_rejects_invalid_input(kwargs, message):
    args = {"delta": 1.0, "gamma": 1.0, "theta": 0.0, "spot": 100.0, "vol": 0.2, "horizon_days": 1, "confidence": 0.99}
    with pytest.raises(ValueError, match=message):
        risk.delta_gamma_var(**(args | kwargs))


def test_delta_gamma_var_options_are_keyword_only():
    with pytest.raises(TypeError):
        risk.delta_gamma_var(1.0, 1.0, 0.0, 100.0, 0.2, 1, 0.99, 1e-8)


def test_the_delta_gamma_var_examples_run():
    # The docstring's examples are the documentation's numbers ($70.55 and $92.79 for the straddle, $293.09 without
    # gamma): run them, since the suite does not collect doctests.
    runner = doctest.DocTestRunner()
    for test in doctest.DocTestFinder().find(risk.delta_gamma_var, globs={"delta_gamma_var": risk.delta_gamma_var}):
        runner.run(test)
    assert runner.tries >= 6 and runner.failures == 0


def test_the_methodology_quotes_the_comparator_on_the_index_straddle():
    # The Methodology's numbers for the 45-day index straddle with implied vol frozen (one day, 99 %, 100,000 paths,
    # seed 7, the app's default): full revaluation $2,833, delta-gamma-theta $2,882, delta-normal $1,016.
    res = risk.value_at_risk(STRADDLE, SPX, risk.VaRConfig(0.99, 1, 100_000, seed=7), 100)
    assert (
        rf"the one-day 99 % VaR is \${res.delta_gamma.var:,.0f} delta-gamma-theta and \${res.var:,.0f} by full "
        rf"revaluation (\${res.delta_normal_var:,.0f} delta-normal)" in _methodology()
    )
