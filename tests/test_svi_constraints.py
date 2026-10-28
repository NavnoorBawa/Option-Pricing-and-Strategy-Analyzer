"""Spread-weighted SVI calibration constrained against static arbitrage (roadmap item SF-4).

References: Gatheral and Jacquier's arbitrageable SVI slice (Example 3.1 of "Arbitrage-free SVI volatility surfaces",
2014), exact SSVI surfaces (arbitrage-free by construction), and the true smile behind noisy quotes.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

from options_analyzer import vol_surface as vs
from options_analyzer.vol_surface import svi

#: Gatheral and Jacquier (2014), Example 3.1 (after Axel Vogt): raw SVI (a, b, rho, m, s) at T = 1, with a negative
#: density (Durrleman's g < 0) near y = 0.9.
VOGT = svi.SVISlice(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153)


def _g(fit: svi.SVISlice, y: np.ndarray) -> np.ndarray:
    return svi.durrleman_g(y, *fit.total_variance(y))


def _rmse(fit: svi.SVISlice, y, iv, T: float) -> float:
    return float(np.sqrt(np.mean((np.sqrt(fit.total_variance(y)[0] / T) - iv) ** 2)))


def test_acceptance_1_the_vogt_slice_is_fitted_without_butterfly_arbitrage():
    """Quotes on Vogt's slice at T = 1, and the same plus 0.02 of total variance at T = 1.5: the arbitrage-free
    surface has no butterfly point and a local vol at every quoted grid point, 0.64 vol point (RMSE) from the
    arbitrageable quotes (the roadmap's prototype: 0.63). Unconstrained, the fit reproduces the arbitrage."""
    y = np.linspace(-1.5, 1.5, 61)
    assert _g(VOGT, y).min() < -0.03  # the slice itself is arbitrageable
    rows = [
        {"strike": 100 * math.exp(k), "maturity": T, "iv": math.sqrt((float(VOGT.total_variance(k)[0]) + shift) / T)}
        for T, shift in ((1.0, 0.0), (1.5, 0.02))
        for k in y
    ]
    quotes = pd.DataFrame(rows)
    fitted = vs.fit_market_surface(quotes, 100.0, 0.0, 0.0)
    assert fitted.butterfly_violations == 0 and fitted.calendar_violations == 0
    assert np.isfinite(fitted.local_vol[np.isfinite(fitted.implied_vol)]).all()
    assert 0.5 < fitted.fit_rmse[0] * 100 < 1.0 and fitted.fit_rmse[1] < 1e-9
    assert fitted.svi[0].total_variance(y)[0].min() > 0 and _g(fitted.svi[0], np.linspace(-1.8, 1.8, 721)).min() > 0
    assert any("negative density (butterfly arbitrage)" in note for note in fitted.notes)
    free = vs.fit_market_surface(quotes, 100.0, 0.0, 0.0, calibration="unconstrained")
    assert free.butterfly_violations > 0 and free.fit_rmse.max() < 1e-9


def test_acceptance_2_crossing_slices_are_held_apart_and_the_cost_reported():
    """Two smiles that cross: the later one's wings fall below the earlier one's. The arbitrage-free fit has no
    calendar point on the overlap of their quotes, and the summary reports what that cost."""
    T1, T2, y = 0.25, 0.5, np.linspace(-0.4, 0.4, 17)
    early = 0.04 + 0.30 * y**2  # steep smile, total variance
    late = 0.06 + 0.05 * y**2  # flat smile: below the early one beyond |y| = 0.28
    rows = [
        {"strike": 100 * math.exp(k), "maturity": T, "iv": math.sqrt(w / T)}
        for T, ws in ((T1, early), (T2, late))
        for k, w in zip(y, ws, strict=True)
    ]
    quotes = pd.DataFrame(rows)
    free = vs.fit_market_surface(quotes, 100.0, 0.0, 0.0, calibration="unconstrained")
    held = vs.fit_market_surface(quotes, 100.0, 0.0, 0.0)
    assert free.calendar_violations > 0 and held.calendar_violations == 0
    w1, w2 = held.svi[0].total_variance(y)[0], held.svi[1].total_variance(y)[0]
    assert (w2 - w1).min() >= -2e-4 * math.sqrt(w1.max() * T2)  # within the quotes' precision (see svi._Constraints)
    cost = vs.surface_summary(held)["Arbitrage cost (vol pts)"]
    # The early smile is a parabola, which SVI's straight wings can follow inside the quotes only by bending into a
    # negative density beyond them: keeping the density positive out to |ln K/F| = 1.5 costs it 0.06 vol point.
    assert cost.iloc[0] < 0.1 and cost.iloc[1] > 0.05
    assert any("total variance falls below that of the maturity before" in note for note in held.notes)


def _random_surface(rng: np.random.Generator) -> tuple[vs.SSVIParams, pd.DataFrame]:
    rho = rng.uniform(-0.9, 0.0)
    params = vs.SSVIParams(
        rng.uniform(0.12, 0.45),
        rng.uniform(0.12, 0.35),
        rng.uniform(0.5, 4.0),
        rho,
        rng.uniform(0.2, 1.0) * 2 / (1 + abs(rho)),  # arbitrage-free: eta (1 + |rho|) <= 2
        rng.uniform(0.25, 0.5),
    )
    rows = []
    for days in (14, 30, 60, 91, 182, 365):
        T = days / 365
        for k in np.linspace(-2.0, 1.2, 15) * math.sqrt(params.theta(T)):
            iv = math.sqrt(float(params.total_variance(k, T)[0]) / T) + rng.normal(0.0, 0.002)
            rows.append({"strike": 100 * math.exp(k), "maturity": T, "iv": iv})
    return params, pd.DataFrame(rows)


@pytest.mark.slow
def test_acceptance_3_random_arbitrage_free_surfaces_with_noise():
    """200 random arbitrage-free SSVI surfaces, six expiries from two weeks to a year, 15 strikes each from -2 to
    +1.2 standard deviations, with 0.2 vol point of noise: 28 of the unconstrained fits have static arbitrage, none
    of the arbitrage-free ones has a butterfly point or a calendar point where both expiries are quoted, and no
    expiry's RMSE grows by more than 10 % (measured: 6 % at most, 1.1 % for a whole surface).

    The surface's calendar count also compares each expiry with its neighbours' smiles up to 25 % of their quoted
    range beyond their quotes, where local vol extrapolates them: there 9 of the 200 surfaces keep 1 to 3 grid points
    (the constraint is held on the quoted overlap only, as holding it beyond the quotes left fits up to 300 times
    worse). This test counts calendar points on the quoted overlap of each pair of expiries.
    """
    rng = np.random.default_rng(2024)
    for _ in range(200):
        _, quotes = _random_surface(rng)
        held = vs.fit_market_surface(quotes, 100.0, 0.0, 0.0)
        free = vs.fit_market_surface(quotes, 100.0, 0.0, 0.0, calibration="unconstrained")
        assert held.butterfly_violations == 0
        assert (held.fit_rmse <= 1.1 * free.fit_rmse).all()
        for i in range(1, len(held.svi)):
            lo = max(held.quoted_y[i, 0], held.quoted_y[i - 1, 0])
            hi = min(held.quoted_y[i, 1], held.quoted_y[i - 1, 1])
            y = np.linspace(lo, hi, 50)
            fall = held.svi[i - 1].total_variance(y)[0] - held.svi[i].total_variance(y)[0]
            assert fall.max() <= 2 * math.sqrt(held.svi[i].total_variance(y)[0].max() * held.maturities[i]) * 1e-4


def test_acceptance_3_light():
    """The study above on 12 surfaces, in the default run."""
    rng = np.random.default_rng(2024)
    for _ in range(12):
        _, quotes = _random_surface(rng)
        held = vs.fit_market_surface(quotes, 100.0, 0.0, 0.0)
        free = vs.fit_market_surface(quotes, 100.0, 0.0, 0.0, calibration="unconstrained")
        assert held.butterfly_violations == 0 and (held.fit_rmse <= 1.1 * free.fit_rmse).all()


def _spread_study(seed: int, weighted: bool) -> float:
    """The largest error within one standard deviation of the money of an SVI fit to a 30-day SSVI smile at 41
    strikes from -3 to +2 standard deviations, quoted with half-spreads of the larger of $0.025 (a tick) and 0.05 % of
    vega plus 1 % of the option's value on a $100 forward, each mid placed uniformly within its spread."""
    params, T, forward = vs.SSVIParams(), 30 / 365, 100.0
    y = np.linspace(-3.0, 2.0, 41) * math.sqrt(params.theta(T))
    w = params.total_variance(y, T)[0]
    sd = np.sqrt(w)
    d1 = -y / sd + 0.5 * sd
    strike = forward * np.exp(y)
    value = np.where(y >= 0, forward * ndtr(d1) - strike * ndtr(d1 - sd), strike * ndtr(sd - d1) - forward * ndtr(-d1))
    vega = forward * math.sqrt(T) * np.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    half = np.maximum(0.025, 0.0005 * vega + 0.01 * value) / vega  # in vol: 0.2 vol point at the money, 3 in the wings
    true = np.sqrt(w / T)
    quoted = true + half * np.random.default_rng(seed).uniform(-1.0, 1.0, len(y))
    fit = svi.fit_svi(y, quoted, T, svi.spread_weights(half) if weighted else None)
    core = np.abs(y) <= math.sqrt(params.theta(T))
    return float(np.abs(np.sqrt(fit.total_variance(y[core])[0] / T) - true[core]).max())


def test_acceptance_4_spread_weights_follow_the_tight_quotes():
    """Over 50 noise draws, the spread-weighted fit's median largest error within one standard deviation is at most
    0.7 times the unweighted fit's (measured 0.58: 0.096 against 0.166 vol point). The gain grows with the range of
    spreads across strikes: 0.58-0.64 where wing spreads are 15 times the at-the-money ones (tick-bound wings, as
    here), 0.84 where they are 7 times (a 91-day smile quoted in pennies)."""
    weighted = np.median([_spread_study(seed, True) for seed in range(50)])
    unweighted = np.median([_spread_study(seed, False) for seed in range(50)])
    assert weighted <= 0.7 * unweighted


def test_acceptance_5_equal_spreads_are_the_unweighted_fit_bit_for_bit():
    params, T = vs.SSVIParams(), 0.25
    y = np.linspace(-0.5, 0.4, 25)
    iv = np.sqrt(params.total_variance(y, T)[0] / T) + np.random.default_rng(3).normal(0, 0.003, len(y))
    weights = svi.spread_weights(np.full(len(y), 0.004))
    assert (weights == 1.0).all()
    assert svi.fit_svi(y, iv, T, weights) == svi.fit_svi(y, iv, T)
    # Dense slices are binned: equal weights give the unweighted bins exactly.
    from options_analyzer.vol_surface.surface import _binned

    dense_y = np.linspace(-0.5, 0.4, 900)
    dense_iv = np.sqrt(params.total_variance(dense_y, T)[0] / T)
    plain, ones = _binned(dense_y, dense_iv), _binned(dense_y, dense_iv, np.ones(900))
    assert np.array_equal(plain[0], ones[0]) and np.array_equal(plain[1], ones[1]) and (ones[2] == 1.0).all()


def test_spread_weights():
    assert svi.spread_weights([0.001, 0.002, 0.004]).tolist() == [2.0, 1.0, 0.5]  # 1 / h over the median
    # Half-spreads below the floor count at the floor; the cap holds the tightest quote at 5 times the median.
    capped = svi.spread_weights([1e-6, 0.01, 0.01, 0.01])
    assert capped[0] == svi.MAX_WEIGHT and (capped[1:] == 1.0).all()
    # Unknown spreads get the median weight; with none known the fit is unweighted.
    assert svi.spread_weights([0.002, np.nan, 0.003, 0.004]).tolist() == pytest.approx([1.5, 1.0, 1.0, 0.75])
    assert svi.spread_weights([np.nan, np.nan]) is None


def test_arbitrage_free_quotes_are_fitted_exactly_as_without_the_constraints():
    """On noise-free SSVI quotes the constraints never bind: the arbitrage-free fit is the unconstrained one."""
    params, T = vs.SSVIParams(), 0.5
    y = np.linspace(-0.6, 0.5, 21)
    iv = np.sqrt(params.total_variance(y, T)[0] / T)
    fit, free = svi.fit_svi_pair(y, iv, T)
    assert fit is free and _rmse(fit, y, iv, T) < 1e-12


def test_quotes_with_prices_are_weighted_by_their_spreads_and_report_the_fit_against_them():
    """A chain with bid/ask quotes: every fitted vol is inside its quote's spread, and the summary says so."""
    rows = []
    params, r = vs.SSVIParams(), 0.03
    for days in (30, 91, 182):
        T = days / 365
        forward = 100 * math.exp(r * T)
        for strike in np.arange(75.0, 126.0, 2.5):
            iv = math.sqrt(float(params.total_variance(math.log(strike / forward), T)[0]) / T)
            for kind in ("call", "put"):
                from options_analyzer import black_scholes as bs

                value = float(bs.price(100.0, strike, T, r, iv, 0.0, kind))
                rows.append({"Strike": strike, "Days": days, "Type": kind, "Bid": value - 0.02, "Ask": value + 0.02})
    clean, _ = vs.parse_quotes(pd.DataFrame(rows), 100.0, r, 0.0)
    surface = vs.fit_market_surface(clean, 100.0, r, 0.0)
    assert surface.quotes["vol_half_spread"].notna().all()
    summary = vs.surface_summary(surface)
    assert (summary["Inside bid/ask (%)"] == 100.0).all() and (summary["Median |error| / half-spread"] < 0.05).all()
    assert (summary["Arbitrage cost (vol pts)"] == 0.0).all()


def test_the_calibration_is_checked():
    with pytest.raises(ValueError, match="calibration must be one of 'arbitrage_free', 'unconstrained'"):
        vs.fit_market_surface(_random_surface(np.random.default_rng(0))[1], 100.0, 0.0, 0.0, calibration="free")


# --- Wing extrapolation within Lee's bound (roadmap item SF-8) -------------------------------------------------------


def _fitted_surfaces() -> list[vs.VolSurface]:
    rng = np.random.default_rng(7)
    surfaces = [vs.fit_market_surface(_random_surface(rng)[1], 100.0, 0.0, 0.0) for _ in range(8)]
    vogt = pd.DataFrame(
        [
            {"strike": 100 * math.exp(k), "maturity": T, "iv": math.sqrt(float(VOGT.total_variance(k)[0] + s) / T)}
            for T, s in ((1.0, 0.0), (1.5, 0.02))
            for k in np.linspace(-1.5, 1.5, 61)
        ]
    )
    return [*surfaces, vs.fit_market_surface(vogt, 100.0, 0.0, 0.0)]


def test_extrapolated_wings_keep_lees_slope_and_a_positive_density():
    """Every fitted smile's total variance grows with a slope of at most 2 out to |ln K/F| = 3 (Lee's moment bound:
    SVI's wing slope is b (1 + |rho|), which the fit holds to 2), and, fitted arbitrage-free, keeps Durrleman's
    g > 0 out to its reach (four at-the-money standard deviations, at most 1.5), checked on 4,001 points. Without
    the wing points in the constraint grid, 96 of 411 fitted smiles had g <= 0 somewhere between their quotes and
    |ln K/F| = 3 (worst -9.1)."""
    y = np.linspace(-3.0, 3.0, 1201)
    for surface in _fitted_surfaces():
        for fit, reach, (lo, hi) in zip(surface.svi, surface.wing_reach, surface.quoted_y, strict=True):
            assert reach == svi.wing_reach(fit) or reach == pytest.approx(svi.wing_reach(fit), rel=0.2)
            assert np.abs(fit.total_variance(y)[1]).max() <= 2.0 + 1e-12
            wings = np.linspace(min(lo, -reach), max(hi, reach), 4001)
            assert fit.total_variance(wings)[0].min() > 0 and _g(fit, wings).min() > 0


def test_extrapolation_is_off_by_default_and_flagged_when_on():
    surface = vs.fit_market_surface(vs.parse_quotes(vs.sample_quotes())[0], 100.0, 0.04, 0.0)
    T = float(surface.maturities[2])
    far = 1.45  # K/S beyond the quoted 0.7-1.3
    assert np.isnan(surface.implied_vol_at(far, T)) and not surface.is_extrapolated(1.0, T)
    wing = surface.implied_vol_at(far, T, extrapolate=True)
    k = math.log(far) - surface.log_forward_ratio[2]
    assert wing == pytest.approx(math.sqrt(float(surface.svi[2].total_variance(k)[0]) / T), rel=1e-12)
    assert surface.is_extrapolated(far, T)
    reach = float(surface.wing_reach[2])
    beyond = math.exp(surface.log_forward_ratio[2] + reach + 0.01)
    assert np.isnan(surface.implied_vol_at(beyond, T, extrapolate=True))  # beyond the reach
    levels = np.array([0.5, 1.0, far])
    inside, local = surface.at_moneyness(levels)
    outside, local_out = surface.at_moneyness(levels, extrapolate=True)
    assert np.isnan(inside[:, [0, 2]]).all() and np.isfinite(outside[:, 1]).all()
    # Off the quotes, a K/S is read only where it lies within four standard deviations of the forward: 0.5 and 1.45
    # at the longer expiries, neither at the first (30 days).
    for column, level in ((0, 0.5), (2, far)):
        reachable = np.abs(np.log(level) - surface.log_forward_ratio) <= surface.wing_reach
        assert (np.isfinite(outside[:, column]) == reachable).all() and reachable[-1] and not reachable[0]
    assert np.array_equal(inside[:, 1], outside[:, 1]) and np.array_equal(local, local_out, equal_nan=True)
    assert (surface.quoted_mask(levels) == np.isfinite(inside)).all()
    ssvi = vs.ssvi_surface(vs.SSVIParams(), 0.04, 0.0, np.linspace(0.8, 1.2, 9), np.array([0.25, 0.5]))
    assert np.isnan(ssvi.implied_vol_at(1.4, 0.25)) and ssvi.is_extrapolated(1.4, 0.25)
