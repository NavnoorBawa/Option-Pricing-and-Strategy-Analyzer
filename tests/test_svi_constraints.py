"""Spread-weighted SVI calibration constrained against static arbitrage (roadmap item SF-4).

References: Gatheral and Jacquier's arbitrageable SVI slice (Example 3.1 of "Arbitrage-free SVI volatility surfaces",
2014), exact SSVI surfaces (arbitrage-free by construction), and the true smile behind noisy quotes.
"""

from __future__ import annotations

import dataclasses
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
    surface has no butterfly point and a local vol at every quoted grid point, 0.6305 vol point (RMSE) from the
    arbitrageable quotes (the roadmap's prototype: 0.63; its acceptance bound is 1). The test holds it to 5 % of
    the measured value either side, 0.60 to 0.66, as a change of the solver or its starts may move it a little and
    a regression moves it more. Unconstrained, the fit reproduces the arbitrage."""
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
    assert 0.60 < fitted.fit_rmse[0] * 100 < 0.66 and fitted.fit_rmse[1] < 1e-9
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
    # The early smile has no arbitrage of its own (nothing before it, a positive density): no cost. Holding the late
    # one above it costs 2.45 vol points (its RMSE, against 0.0018 unconstrained).
    assert cost.iloc[0] == 0.0 and cost.iloc[1] > 2.0
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
    expiry's RMSE grows by more than 15 % (measured: 8.6 % at most; the RMSE over all of a surface's quotes, 2.5 % at
    most; with one BLAS thread 7.6 % and 1.5 %, and 11.6 % on one expiry on a CI runner's CPU, as the solvers' last
    digits, and so the local optimum a search ends in, depend on the threads and the CPU). The slowest surface took
    0.6 s of processor time on a shared four-core machine.

    The surface's calendar count also compares each expiry with its neighbours' smiles up to 25 % of their quoted
    range beyond their quotes, where local vol extrapolates them; the constraint is held on the quoted overlap only,
    as holding it beyond the quotes left fits up to 300 times worse. None of the 200 surfaces has a calendar point
    there either, but this test counts calendar points on the quoted overlap of each pair of expiries, where the
    constraint is held.
    """
    rng = np.random.default_rng(2024)
    for _ in range(200):
        _, quotes = _random_surface(rng)
        held = vs.fit_market_surface(quotes, 100.0, 0.0, 0.0)
        free = vs.fit_market_surface(quotes, 100.0, 0.0, 0.0, calibration="unconstrained")
        assert held.butterfly_violations == 0
        assert (held.fit_rmse <= 1.15 * free.fit_rmse).all()
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
        assert held.butterfly_violations == 0 and (held.fit_rmse <= 1.15 * free.fit_rmse).all()


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
    fit = svi.fit_svi(y, quoted, T, weights=svi.spread_weights(half) if weighted else None)
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
    assert svi.fit_svi(y, iv, T, weights=weights) == svi.fit_svi(y, iv, T)
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
    |ln K/F| = 3 (worst -9.1). The surface reads each smile out to the reach of its unconstrained fit, within 2 % of
    the arbitrage-free fit's own (measured 0.74 % at most over these 50 smiles)."""
    y = np.linspace(-3.0, 3.0, 1201)
    for surface in _fitted_surfaces():
        for fit, reach, (lo, hi) in zip(surface.svi, surface.wing_reach, surface.quoted_y, strict=True):
            assert reach == pytest.approx(svi.wing_reach(fit), rel=0.02)
            assert np.abs(fit.total_variance(y)[1]).max() <= 2.0 + 1e-12
            wings = np.linspace(min(lo, -reach), max(hi, reach), 4001)
            assert fit.total_variance(wings)[0].min() > 0 and _g(fit, wings).min() > 0


@pytest.mark.parametrize(
    ("b", "rho", "T", "m"),
    [(1.3, -0.7, 1.0, 0.0), (2.5, -0.2, 2.0, 0.0), (1.2, -0.95, 0.5, 0.3), (4.0, -0.99, 0.25, 0.0)],
)
def test_lees_bound_holds_without_the_arbitrage_constraints_too(b, rho, T, m):
    """Raw SVI smiles whose wing slope b (1 + |rho|) is 2.21 to 7.96, beyond Lee's bound, fitted unconstrained on 25
    quotes over [-1, 0.4]: the fit's slope is at most 2. With the bound only a penalty, the second and third were
    fitted with slopes of 2.000019 and 2.000059 and the last with 2.0024, and the readout extrapolated from wings
    steeper than any arbitrage-free one; the fit's b is now projected onto the bound. 1e-12 allows the rounding of
    2 / (1 + |rho|) times (1 + |rho|)."""
    true = svi.SVISlice(0.04 * T, b, rho, m, 0.3)
    y = np.linspace(-1.0, 0.4, 25)
    fit = svi.fit_svi(y, np.sqrt(true.total_variance(y)[0] / T), T, arbitrage_free=False)
    assert fit.b * (1.0 + abs(fit.rho)) <= 2.0 + 1e-12
    assert np.abs(fit.total_variance(np.linspace(-3.0, 3.0, 601))[1]).max() <= 2.0 + 1e-12


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
    # The smile chart reads implied vol alone, without the local vol (and its dw/dT) of at_moneyness: bit for bit the
    # same, for market quotes, SSVI and a surface known only on its grid, inside the quotes and beyond them.
    grid_only = dataclasses.replace(surface, svi=(), quoted_y=None)
    dense = np.linspace(0.45, 1.6, 47)
    for each in (surface, ssvi, grid_only):
        for extrapolate in (False, True):
            alone = each.implied_at_moneyness(dense, extrapolate=extrapolate)
            assert np.array_equal(alone, each.at_moneyness(dense, extrapolate=extrapolate)[0], equal_nan=True)
            assert np.isfinite(alone).any() and np.isnan(alone).any()


# --- The arbitrage-free search: gradients, starts and what it does when it fails -------------------------------------


@pytest.mark.parametrize(
    "fit", [VOGT, svi.SVISlice(0.02, 0.4, -0.7, 0.05, 0.1), svi.SVISlice(0.1, 1.2, 0.3, -0.2, 0.02)]
)
def test_parameter_derivatives_match_central_differences(fit):
    """The derivatives of (w, dw/dy, d2w/dy2) in (a, b, rho, m, s), which the solver's constraint gradients use,
    against central differences with a step h of 1e-6 of each parameter's size: their error is of order h^2 times the
    third derivative, below 1e-9 here, and their rounding about 1e-16 / h relative, so 1e-7 of the largest derivative
    of each column bounds both (measured 4e-9 at most)."""
    y = np.linspace(-1.5, 1.5, 301)
    analytic = fit.parameter_derivatives(y)
    # The solver's own evaluation (svi._smile) is the same arithmetic, bit for bit.
    fast = svi._smile(np.array([fit.a, fit.b, fit.rho, fit.m, fit.s]), y, derivatives=True)
    assert all(np.array_equal(a, b) for a, b in zip(fast, (*fit.total_variance(y), *analytic), strict=True))
    params = np.array([fit.a, fit.b, fit.rho, fit.m, fit.s])
    for j in range(5):
        h = 1e-6 * max(abs(params[j]), 0.1)
        up, down = params.copy(), params.copy()
        up[j] += h
        down[j] -= h
        numeric = [
            (u - d) / (2 * h)
            for u, d in zip(svi.SVISlice(*up).total_variance(y), svi.SVISlice(*down).total_variance(y), strict=True)
        ]
        for exact, approx in zip(analytic, numeric, strict=True):
            scale = max(np.abs(exact[:, j]).max(), 1e-12)
            assert np.abs(exact[:, j] - approx).max() <= 1e-7 * scale


def _crossing_smiles() -> pd.DataFrame:
    """Two of the reviewer's 20 independent smiles (level, skew and curvature drawn at random, seed 5): at 121 and 159
    days, 41 strikes over ln K/F from -0.6 to 0.4, the later one's quotes below the earlier one's smile at every
    strike (calendar arbitrage across the whole overlap)."""
    rng = np.random.default_rng(5)
    rng.normal(size=3 * 20 * 41)  # the draws the reviewer's probe made first
    rows = []
    for days in np.linspace(7, 730, 20).round():
        level, skew, curvature = rng.uniform(0.1, 0.6), rng.uniform(-0.6, 0.2), rng.uniform(0, 2)
        for k in np.linspace(-0.6, 0.4, 41):
            iv = max(0.03, level + skew * k + curvature * k * k)
            rows.append({"strike": 100 * math.exp(k), "maturity": days / 365, "iv": iv})
    quotes = pd.DataFrame(rows)
    maturities = np.sort(quotes["maturity"].unique())
    return quotes[quotes["maturity"].isin(maturities[3:5])]


def test_quotes_below_the_previous_smile_everywhere_get_the_nearest_arbitrage_free_fit():
    """From the unconstrained optimum and from every quasi-explicit start the solver ends with "inequality
    constraints incompatible", and the slice used to keep its calendar arbitrage (82 grid points) with an Arbitrage
    cost of 0. From the previous slice raised to these quotes, which lie 0.01 to 0.44 below the earlier smile in total
    variance, it finds an arbitrage-free smile 34.5 vol points (RMSE) from them (the nearly flat start finds the same
    one); against the unconstrained fit's 0.8, an Arbitrage cost of 33.7 vol points."""
    quotes = _crossing_smiles()
    free = vs.fit_market_surface(quotes, 100.0, 0.0, 0.0, calibration="unconstrained")
    assert free.calendar_violations > 0
    previous = (free.svi[0], tuple(free.quoted_y[0]))
    T = float(free.maturities[1])
    rows = quotes[np.isclose(quotes["maturity"], T)]
    y, iv = np.log(rows["strike"].to_numpy() / 100.0), rows["iv"].to_numpy()
    constraints = svi._Constraints.build(y.min(), y.max(), T, previous, svi.wing_reach(free.svi[1]))
    others = [free.svi[1], *(svi.SVISlice(*start) for start in svi._svi_start_points(y, iv, T))]
    assert all(constraints.nearest(y, iv, T, None, start, svi._Budget(10**6)) is None for start in others)
    raised = constraints.nearest(y, iv, T, None, svi._raised(free.svi[0], y, iv, T), svi._Budget(10**6))
    assert raised is not None and not any(constraints.broken(raised))
    held = vs.fit_market_surface(quotes, 100.0, 0.0, 0.0)
    assert held.calendar_violations == 0 and held.butterfly_violations == 0
    previous = (held.svi[0], tuple(held.quoted_y[0]))
    reach = float(held.wing_reach[1])  # the unconstrained fit's, which the constraints are held out to
    assert svi.static_arbitrage(held.svi[1], (y.min(), y.max()), T, previous, reach=reach) == (False, False)
    assert not [note for note in held.notes if note.startswith("No SVI smile free of static arbitrage")]
    assert 34.0 < _rmse(held.svi[1], y, iv, T) * 100 < 35.0
    assert 33.0 < vs.surface_summary(held)["Arbitrage cost (vol pts)"].iloc[1] < 34.5


def test_a_noisy_first_slice_reaches_an_arbitrage_free_fit_from_a_flat_smile():
    """The 7-day slice of one of the reviewer's noisy surfaces (default SSVI, 41 strikes from -2.5 to +1.5 standard
    deviations, 8 vol points of noise, seed 6): its unconstrained fit has a negative density, and from it and from
    every quasi-explicit start the solver ends "inequality constraints incompatible". From a nearly flat smile, which
    meets the constraints, it reaches an arbitrage-free fit 6.5 % worse in RMSE (6.45 against 6.06 vol points); the
    slice used to keep its arbitrage without a word."""
    params, rng = vs.SSVIParams(), np.random.default_rng(6)
    rng.normal(size=2 * 20 * 41)  # the draws of the probe's 1 and 3 vol point surfaces, made first
    T = 7 / 365
    y = np.linspace(-2.5, 1.5, 41) * math.sqrt(params.theta(T))
    iv = np.maximum(0.02, np.sqrt(params.total_variance(y, T)[0] / T) + rng.normal(0, 0.08, len(y)))
    fit, free = svi.fit_svi_pair(y, iv, T)
    constraints = svi._Constraints.build(y.min(), y.max(), T, None, svi.wing_reach(free))
    assert constraints.broken(free) == (True, False)
    others = [free, *(svi.SVISlice(*start) for start in svi._svi_start_points(y, iv, T))]
    assert all(constraints.nearest(y, iv, T, None, start, svi._Budget(10**6)) is None for start in others)
    assert constraints.met(constraints.flat(y, iv, T))
    assert fit is not free
    assert svi.static_arbitrage(fit, (y.min(), y.max()), T, None, reach=svi.wing_reach(free)) == (False, False)
    assert _rmse(fit, y, iv, T) <= 1.1 * _rmse(free, y, iv, T)  # the 10 % the search accepts


def test_a_slice_no_start_can_free_of_arbitrage_keeps_its_fit_with_a_note_and_a_blank_cost(monkeypatch):
    """With no solver iterations to spend (the cap a slow search can hit), no start meets the constraints: each
    slice keeps its unconstrained fit, a note names the maturities and the arbitrage left, and their Arbitrage cost
    is NaN (blank) rather than 0."""
    monkeypatch.setattr(svi, "SLSQP_BUDGET", 0)
    held = vs.fit_market_surface(_crossing_smiles(), 100.0, 0.0, 0.0)
    free = vs.fit_market_surface(_crossing_smiles(), 100.0, 0.0, 0.0, calibration="unconstrained")
    assert held.calendar_violations == free.calendar_violations > 0
    assert np.array_equal(held.fit_rmse, free.fit_rmse)
    left = [note for note in held.notes if note.startswith("No SVI smile free of static arbitrage")]
    assert len(left) == 1 and "maturities 0.332y, 0.436y, so they keep their unconstrained fit" in left[0]
    assert "at 0.436y its total variance falls below that of the maturity before (calendar)" in left[0]
    assert "at 0.332y it has a negative density (butterfly)" in left[0]
    assert vs.surface_summary(held)["Arbitrage cost (vol pts)"].isna().all()
    assert not [note for note in held.notes if note.startswith("The SVI fit was held free")]
    assert "allowed for the whole surface" not in left[0]  # each slice's budget ran out, not the surface's
    # One such maturity, in the singular.
    from types import SimpleNamespace

    from options_analyzer.vol_surface.surface import _arbitrage_left

    slices = [
        SimpleNamespace(left=(False, False), budget_spent=False),
        SimpleNamespace(left=(False, True), budget_spent=False),
    ]
    assert _arbitrage_left(slices, np.array([0.25, 0.5])) == [  # type: ignore[arg-type]
        "No SVI smile free of static arbitrage was found at maturity 0.5y, so it keeps its unconstrained fit and its "
        "arbitrage: at 0.5y its total variance falls below that of the maturity before (calendar). Its Arbitrage cost "
        "is blank; check those quotes."
    ]


def test_a_surface_that_spends_its_solver_iterations_leaves_the_rest_with_a_note(monkeypatch):
    """The maturities share svi.SURFACE_BUDGET iterations in order of maturity. With 20 of them, the crossing smiles'
    first maturity (a negative density) still reaches an arbitrage-free smile, and the second, which needs more, keeps
    its unconstrained fit and its calendar arbitrage, with a note saying the surface's iterations ran out; its
    Arbitrage cost is blank and the first's is not."""
    monkeypatch.setattr(svi, "SURFACE_BUDGET", 20)
    held = vs.fit_market_surface(_crossing_smiles(), 100.0, 0.0, 0.0)
    assert held.butterfly_violations == 0 and held.calendar_violations > 0
    left = [note for note in held.notes if note.startswith("No SVI smile free of static arbitrage")]
    assert len(left) == 1 and "maturity 0.436y, so it keeps its unconstrained fit" in left[0]
    assert left[0].endswith(
        "At 0.436y the search had run out of the 20 solver iterations allowed for the whole surface, a cap that keeps "
        "a very noisy chain to a few seconds."
    )
    cost = vs.surface_summary(held)["Arbitrage cost (vol pts)"]
    assert np.isfinite(cost.iloc[0]) and np.isnan(cost.iloc[1])


def test_a_shared_budget_changes_no_fit_it_covers_and_is_spent_by_it():
    """fit_svi_pair given a surface's budget spends at most SLSQP_BUDGET of it and returns the fit it returns alone
    (the noisy 7-day slice above, 308 iterations); given none left, it keeps the unconstrained fit."""
    params, rng = vs.SSVIParams(), np.random.default_rng(6)
    rng.normal(size=2 * 20 * 41)
    T = 7 / 365
    y = np.linspace(-2.5, 1.5, 41) * math.sqrt(params.theta(T))
    iv = np.maximum(0.02, np.sqrt(params.total_variance(y, T)[0] / T) + rng.normal(0, 0.08, len(y)))
    alone, free = svi.fit_svi_pair(y, iv, T)
    budget = svi._Budget(10**6)
    assert svi.fit_svi_pair(y, iv, T, budget=budget) == (alone, free)
    assert 0 < 10**6 - budget.left <= svi.SLSQP_BUDGET
    empty = svi._Budget(0)
    fit, free = svi.fit_svi_pair(y, iv, T, budget=empty)
    assert fit is free and empty.left == 0


@pytest.mark.slow
def test_static_arbitrage_checks_a_constrained_fit_out_to_the_reach_it_was_held_to():
    """One of the reviewer's noisy surfaces (20 expiries of 41 strikes, 8 vol points of noise, seed 5): every
    arbitrage-free fit meets the constraints out to the unconstrained fit's wing reach, where fit_svi holds them and
    the surface reads its wings. Some reach further themselves, and their density is negative out there, which
    static_arbitrage with its default reach reports: with one BLAS thread the 0.644y and 1.79y fits reach 0.754 and
    1.211 against 0.732 and 0.806, with g <= 0 at 30 and 53 points beyond; with threads only the first."""
    rng, params = np.random.default_rng(5), vs.SSVIParams()
    frames = []
    for noise in (0.01, 0.03, 0.08):
        rows = []
        for days in np.linspace(7, 730, 20).round():
            T = days / 365
            for k in np.linspace(-2.5, 1.5, 41) * math.sqrt(params.theta(T)):
                iv = math.sqrt(params.total_variance(k, T)[0] / T) + rng.normal(0, noise)
                rows.append({"strike": 100 * math.exp(k), "maturity": T, "iv": max(0.02, iv)})
        frames.append(pd.DataFrame(rows))
    surface = vs.fit_market_surface(frames[-1], 100.0, 0.0, 0.0)
    assert not [note for note in surface.notes if note.startswith("No SVI smile free of static arbitrage")]
    own = []
    for i, fit in enumerate(surface.svi):
        previous = (surface.svi[i - 1], tuple(surface.quoted_y[i - 1])) if i else None
        args = (fit, tuple(surface.quoted_y[i]), float(surface.maturities[i]), previous)
        assert svi.static_arbitrage(*args, reach=float(surface.wing_reach[i])) == (False, False)
        own.append(svi.static_arbitrage(*args))
    assert any(butterfly for butterfly, _ in own)  # the fits differ in their last digits with the BLAS threads


def test_exact_prices_without_a_spread_leave_inside_bid_ask_blank():
    """Prices with no bid/ask and on no price grid (price error 0) have no spread to measure the fit against: their
    half-spread in vol is NaN, so Inside bid/ask and Median |error| / half-spread are blank for their expiries,
    while the fit still weights them by their vega. Before, one unit of price stood in for the spread: exact prices
    with a vol point of noise read 100 % inside. An expiry quoted with bid and ask in the same file keeps its
    figures."""
    from options_analyzer import black_scholes as bs

    rng = np.random.default_rng(0)
    rows = []
    for days in (30, 91, 182):
        T = days / 365
        for strike in np.linspace(75.0, 125.0, 21):
            iv = 0.2 + 0.1 * math.log(strike / 100.0) ** 2 + rng.normal(0, 0.01)
            kind = "call" if strike >= 100.0 else "put"
            price = float(bs.price(100.0, strike, T, 0.03, iv, 0.0, kind))
            spread = days == 182  # only the last expiry is quoted with a bid and an ask
            rows.append(
                {
                    "Strike": strike,
                    "Days": days,
                    "Type": kind,
                    "Price": price,
                    "Bid": price - 0.02 if spread else np.nan,
                    "Ask": price + 0.02 if spread else np.nan,
                }
            )
    clean, _ = vs.parse_quotes(pd.DataFrame(rows), 100.0, 0.03, 0.0)
    surface = vs.fit_market_surface(clean, 100.0, 0.03, 0.0)
    last = np.isclose(surface.quotes["maturity"], 182 / 365)
    assert surface.quotes.loc[~last, "vol_half_spread"].isna().all()
    assert surface.quotes.loc[last, "vol_half_spread"].notna().all()
    summary = vs.surface_summary(surface)
    assert summary["Inside bid/ask (%)"].iloc[:2].isna().all()
    assert summary["Median |error| / half-spread"].iloc[:2].isna().all()
    assert np.isfinite(summary["Inside bid/ask (%)"].iloc[2])
    # Fitted all the same, by vega where there is no spread: RMSE 2.6, 0.8 and 1.1 vol points against 1 vol point of
    # noise (the 30-day wings, with little vega, are followed least).
    assert (surface.fit_rmse < 0.03).all()


def test_the_arbitrage_cost_is_the_rise_in_spread_weighted_rmse_and_never_negative():
    """The synthetic SPX chain at a spot of 6,000 on 30 October at 10:00 (noise seed 2), read with no dividend yield:
    at the first expiry the constraints lower the unweighted RMSE by 0.0004 vol point, which the Arbitrage cost used
    to show as -0.000. It is now the rise in RMSE with each quote weighted as in the fit (one over its half-spread in
    vol, relative to the median), recomputed here from the two calibrations' fitted vols: 0.00001 vol point there."""
    from datetime import datetime

    from options_analyzer.market_data import sample_chain, to_quotes

    as_of = datetime(2026, 10, 30, 10)
    chain = sample_chain(6000.0, 0.04, 0.013, as_of=as_of, noise=True, seed=2)
    clean, _, table = vs.parse_quotes(to_quotes(chain.quotes, as_of=as_of), 6000.0, 0.04, 0.0, return_forwards=True)
    held = vs.fit_market_surface(clean, 6000.0, 0.04, 0.0, forwards=table)
    free = vs.fit_market_surface(clean, 6000.0, 0.04, 0.0, forwards=table, calibration="unconstrained")
    assert (held.fit_rmse - held.fit_rmse_unconstrained).min() < -1e-6  # the unweighted RMSE fell
    expected = []
    for T in held.maturities:
        ours, theirs = (s.quotes[np.isclose(s.quotes["maturity"], T)] for s in (held, free))
        weights = svi.spread_weights(ours["vol_half_spread"].to_numpy())  # every quote here has a bid and an ask
        rmse = [np.sqrt(np.mean((weights * (q["fitted_iv"] - q["iv"]).to_numpy()) ** 2)) for q in (ours, theirs)]
        expected.append(max(rmse[0] - rmse[1], 0.0))
    assert held.arbitrage_cost == pytest.approx(np.array(expected), abs=1e-15)
    assert (vs.surface_summary(held)["Arbitrage cost (vol pts)"] >= 0.0).all()
