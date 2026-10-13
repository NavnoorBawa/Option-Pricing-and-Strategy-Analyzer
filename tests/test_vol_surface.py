import io

import numpy as np
import pandas as pd
import pytest

from options_analyzer import black_scholes as bs
from options_analyzer import vol_surface as vs

M = np.linspace(0.7, 1.3, 25)
T = np.linspace(0.1, 2.0, 12)
SAMPLE_DAYS = (30, 60, 91, 182, 273, 365, 547, 730)


def ssvi_quotes(params, rate, dividend_yield, days, moneyness, spot=100.0, decimals=None) -> pd.DataFrame:
    """Exact SSVI implied vols (optionally rounded) as a cleaned quotes table."""
    rows = []
    for d in days:
        t = d / 365.0
        for m in moneyness:
            y = np.log(m) - (rate - dividend_yield) * t
            iv = float(np.sqrt(params.total_variance(y, t)[0] / t))
            rows.append({"strike": spot * m, "maturity": round(t, 10), "iv": round(iv, decimals) if decimals else iv})
    return pd.DataFrame(rows)


def relative_local_vol_error(fitted, params, rate, dividend_yield):
    true = vs.ssvi_surface(params, rate, dividend_yield, fitted.moneyness, fitted.maturities)
    return np.abs(fitted.local_vol / true.local_vol - 1.0), np.abs(fitted.implied_vol - true.implied_vol)


def test_flat_surface_gives_flat_local_vol():
    params = vs.SSVIParams(0.2, 0.2, 1.5, rho=0.0, eta=1e-9, gamma=0.5)
    surface = vs.ssvi_surface(params, 0.03, 0.01, M, T)
    assert np.allclose(surface.implied_vol, 0.2)
    assert np.allclose(surface.local_vol, 0.2)


def test_gatheral_local_vol_matches_price_based_dupire():
    params = vs.SSVIParams()
    r, q, spot = 0.03, 0.01, 100.0

    def iv(K, t):
        y = np.log(K / (spot * np.exp((r - q) * t)))
        return np.sqrt(params.total_variance(y, t)[0] / t)

    def call(K, t):
        return bs.price(spot, K, t, r, iv(K, t), q, "call")

    for K, t in [(80.0, 0.5), (95.0, 0.7), (115.0, 1.5)]:
        hk, ht = 0.01, 1e-4
        c_t = (call(K, t + ht) - call(K, t - ht)) / (2 * ht)
        c_k = (call(K + hk, t) - call(K - hk, t)) / (2 * hk)
        c_kk = (call(K + hk, t) - 2 * call(K, t) + call(K - hk, t)) / hk**2
        dupire = np.sqrt((c_t + (r - q) * K * c_k + q * call(K, t)) / (0.5 * K * K * c_kk))
        gatheral = vs.ssvi_surface(params, r, q, np.array([K / spot]), np.array([t])).local_vol[0, 0]
        assert gatheral == pytest.approx(dupire, rel=1e-6)


def test_ssvi_default_is_arbitrage_free():
    surface = vs.ssvi_surface(vs.SSVIParams(), 0.04, 0.0, np.linspace(0.5, 1.8, 60), np.linspace(0.02, 5, 40))
    assert surface.calendar_violations == 0 and surface.butterfly_violations == 0
    assert np.isfinite(surface.local_vol).all()


def test_ssvi_flags_parameters_outside_the_no_arbitrage_region():
    surface = vs.ssvi_surface(vs.SSVIParams(rho=-0.9, eta=3.9), 0.0, 0.0, M, T)
    assert surface.notes


# ---------------------------------------------------------------------------
# SVI fit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("t", [2 / 365, 30 / 365, 1.0, 3.0])
def test_svi_fit_recovers_an_ssvi_slice_exactly(t):
    # An SSVI slice is a raw-SVI smile with a = theta (1 - rho^2) / 2, b = theta phi / 2, m = -rho / phi,
    # s = sqrt(1 - rho^2) / phi, so noise-free quotes must be reproduced, derivatives included.
    params = vs.SSVIParams()
    y = np.log(np.arange(0.7, 1.3001, 0.05))
    w, w_y, w_yy, _ = params.total_variance(y, t)
    fitted = vs.fit_svi(y, np.sqrt(w / t), t)
    fw, fw_y, fw_yy = fitted.total_variance(y)
    assert np.allclose(fw, w, rtol=1e-9, atol=0)
    assert np.allclose(fw_y, w_y, rtol=1e-6, atol=1e-12)
    assert np.allclose(fw_yy, w_yy, rtol=1e-6)
    theta = params.theta(t)
    phi = params.phi(theta)
    assert fitted.rho == pytest.approx(params.rho, abs=1e-6)
    assert fitted.b == pytest.approx(theta * phi / 2, rel=1e-6)


def test_svi_fit_keeps_variance_positive_and_wings_within_lees_bound():
    rng = np.random.default_rng(3)
    y = np.linspace(-0.6, 0.4, 15)
    iv = 0.25 + 0.3 * y**2 + rng.normal(0, 0.02, y.size)  # noisy, so the fit can wander
    fitted = vs.fit_svi(y, iv, 0.5)
    assert fitted.min_variance >= -1e-10
    assert fitted.b * (1 + abs(fitted.rho)) <= 2.0 + 1e-6


# ---------------------------------------------------------------------------
# Market surface accuracy against the exact SSVI surface
# ---------------------------------------------------------------------------


def test_noise_free_quotes_recover_the_surface_including_end_maturities_and_grid_edges():
    params, r, q = vs.SSVIParams(), 0.03, 0.01
    fitted = vs.fit_market_surface(ssvi_quotes(params, r, q, range(182, 731, 14), M), 100, r, q)
    assert fitted.moneyness[0] == pytest.approx(0.7) and fitted.moneyness[-1] == pytest.approx(1.3)
    assert np.isfinite(fitted.implied_vol).all() and np.isfinite(fitted.local_vol).all()
    lv_err, iv_err = relative_local_vol_error(fitted, params, r, q)
    assert iv_err.max() < 1e-9
    assert lv_err[1:-1].max() < 1e-3  # interior maturities: Fritsch-Butland secants (measured 0.014%)
    assert lv_err[0].max() < 1e-3  # one-sided three-point formula (measured 0.047%; the secant gave 0.71%)
    assert lv_err[-1].max() < 0.005  # one-sided secant (measured 0.12%)
    assert lv_err[:, [0, -1]].max() < 0.01  # outermost strikes: analytic SVI derivatives, no boundary condition
    assert fitted.calendar_violations == 0 and fitted.butterfly_violations == 0
    assert np.all(fitted.fit_rmse < 1e-9)


def test_inverted_term_structure_keeps_a_sound_last_maturity():
    # PCHIP's end formula clipped dw/dT to 0 here, so the whole 2-year slice showed a local vol of 0.
    params = vs.SSVIParams(0.8, 0.2, 3.0, -0.7, 1.0, 0.5)
    strikes = np.arange(0.8, 1.2001, 0.05)
    coarse = vs.fit_market_surface(ssvi_quotes(params, 0.0, 0.0, (30, 91, 365, 730), strikes), 100, 0.0, 0.0)
    lv_err, _ = relative_local_vol_error(coarse, params, 0.0, 0.0)
    assert coarse.calendar_violations == 0 and np.isfinite(coarse.local_vol).all()
    assert coarse.local_vol[-1].min() > 0.15
    assert lv_err[-1].max() < 0.12  # first-order one-sided secant over a whole year (measured 9.6%)
    dense = vs.fit_market_surface(ssvi_quotes(params, 0.0, 0.0, range(30, 731, 15), strikes), 100, 0.0, 0.0)
    lv_err, _ = relative_local_vol_error(dense, params, 0.0, 0.0)
    assert lv_err[1:].max() < 0.005  # measured 0.3%


@pytest.mark.parametrize(
    "low, high, days",
    [
        (0.8, 1.2, (30, 91, 182, 365, 730)),
        (0.9, 1.1, (30, 91, 182, 365, 730)),
        (0.7, 1.3, (30, 91, 182, 365, 730, 1825)),
    ],
)
def test_identical_strikes_with_carry_keep_the_full_quoted_range(low, high, days):
    # The old grid intersected the slices in forward moneyness and again in K/S, losing the carry twice
    # (0.8-1.2 became 0.88-1.09 and 0.9-1.1 was rejected at a 5% rate).
    params, r = vs.SSVIParams(), 0.05
    fitted = vs.fit_market_surface(ssvi_quotes(params, r, 0.0, days, np.linspace(low, high, 9)), 100, r, 0.0)
    assert fitted.moneyness[0] == pytest.approx(low) and fitted.moneyness[-1] == pytest.approx(high)
    assert np.isfinite(fitted.implied_vol).all() and np.isfinite(fitted.local_vol).all()
    summary = vs.surface_summary(fitted)
    if low <= 0.9 and high >= 1.1:
        assert np.isfinite(summary["Skew 90-110 (vol pts)"]).all() and np.isfinite(summary["Butterfly (vol pts)"]).all()
    lv_err, _ = relative_local_vol_error(fitted, params, r, 0.0)
    assert lv_err[1:-1].max() < 0.02


def test_interior_calendar_arbitrage_is_flagged_and_left_undefined():
    # PCHIP set dw/dT to exactly 0 at an interior sign change: 0 violations and a local vol of 0%.
    quotes, _ = vs.parse_quotes(vs.sample_quotes())
    maturities = np.sort(quotes["maturity"].unique())
    quotes.loc[quotes["maturity"] == maturities[4], "iv"] *= 0.8
    fitted = vs.fit_market_surface(quotes, 100, 0.04, 0.0)
    assert fitted.calendar_violations > 0
    assert np.isnan(fitted.local_vol[3]).all() and np.isnan(fitted.local_vol[4]).all()  # both ends of the drop
    defined = fitted.local_vol[np.isfinite(fitted.local_vol)]
    assert defined.size and defined.min() > 0.05  # nothing silently set to zero


def test_calendar_arbitrage_at_the_last_maturity_flags_both_ends():
    quotes, _ = vs.parse_quotes(vs.sample_quotes())
    last = quotes["maturity"].max()
    quotes.loc[quotes["maturity"] == last, "iv"] *= 0.6
    fitted = vs.fit_market_surface(quotes, 100, 0.04, 0.0)
    assert fitted.calendar_violations > 0
    assert np.isnan(fitted.local_vol[-1]).all() and np.isnan(fitted.local_vol[-2]).all()


def test_rounded_dense_chains_do_not_raise_spurious_calendar_flags():
    params = vs.SSVIParams()
    quotes = ssvi_quotes(params, 0.04, 0.0, range(1, 21), np.linspace(0.7, 1.3, 1000), decimals=4)
    fitted = vs.fit_market_surface(quotes, 100, 0.04, 0.0)
    assert fitted.calendar_violations == 0 and fitted.butterfly_violations == 0
    assert any("at most 200 points" in note for note in fitted.notes)
    assert len(fitted.quotes) == len(quotes)  # every quote is kept for the smile chart, only the fit is binned


@pytest.mark.parametrize("seed", range(4))
def test_sample_quotes_local_vol_is_close_to_the_generating_surface(seed):
    # The sample adds 0.2 vol-point noise. The old smoothing-spline fit was off by 57-176 vol points of local
    # vol (seeds 0-3, max) with no arbitrage flag. The SVI fit is off by at most 1.9-4.4 points at the first
    # maturity (three-point end formula; the one-sided secant gave 4.7-6.0) and 1.3-2.3 points elsewhere.
    quotes, _ = vs.parse_quotes(vs.sample_quotes(100, 0.04, 0.0, seed=seed))
    fitted = vs.fit_market_surface(quotes, 100, 0.04, 0.0)
    true = vs.ssvi_surface(vs.SSVIParams(), 0.04, 0.0, fitted.moneyness, fitted.maturities)
    err = np.abs(fitted.local_vol - true.local_vol)
    assert fitted.calendar_violations == 0 and fitted.butterfly_violations == 0
    assert np.isfinite(fitted.local_vol).all()
    assert np.max(np.abs(fitted.implied_vol - true.implied_vol)) < 0.006
    assert err[0].max() < 0.05
    assert err[1:].max() < 0.03
    assert np.median(err) < 0.006
    assert np.all((fitted.fit_rmse > 0.0005) & (fitted.fit_rmse < 0.004))  # about the 0.2-point noise


def test_mixed_strike_ranges_keep_every_maturity_finely_gridded_and_the_summary_exact():
    # A LEAPS quoted 0.3-2.5 next to weeklies quoted 0.9-1.1: a fixed 41-column grid over the union left 4 columns
    # inside each weekly, the summary read them off that coarse grid (ATM vol off by 0.24 points) and skew and
    # butterfly were NaN for slices quoted exactly at 0.9 and 1.1.
    params, r = vs.SSVIParams(), 0.05
    ranges = {7: (0.9, 1.1, 9), 14: (0.9, 1.1, 9), 30: (0.85, 1.15, 13), 60: (0.8, 1.2, 17), 182: (0.6, 1.4, 17)}
    ranges[730] = (0.3, 2.5, 23)
    quotes = pd.concat(ssvi_quotes(params, r, 0.0, (d,), np.linspace(lo, hi, n)) for d, (lo, hi, n) in ranges.items())
    fitted = vs.fit_market_surface(quotes, 100, r, 0.0)
    assert fitted.moneyness[0] == pytest.approx(0.3) and fitted.moneyness[-1] == pytest.approx(2.5)
    assert np.isfinite(fitted.implied_vol).sum(axis=1).min() >= 15
    summary = vs.surface_summary(fitted)
    true = vs.ssvi_surface(params, r, 0.0, np.array([0.9, 1.0, 1.1]), fitted.maturities).implied_vol
    assert np.abs(summary["ATM implied vol"] - true[:, 1]).max() < 1e-4  # 0.01 vol points
    assert np.allclose(summary["Skew 90-110 (vol pts)"], (true[:, 0] - true[:, 2]) * 100, atol=0.01)
    assert np.allclose(summary["Butterfly (vol pts)"], (0.5 * (true[:, 0] + true[:, 2]) - true[:, 1]) * 100, atol=0.01)
    assert np.isfinite(summary["ATM local vol"]).all()
    assert len(fitted.moneyness) <= vs.MAX_GRID_POINTS


def test_identical_ranges_keep_the_plain_uniform_grid():
    params = vs.SSVIParams()
    fitted = vs.fit_market_surface(
        ssvi_quotes(params, 0.04, 0.0, SAMPLE_DAYS, np.linspace(0.8, 1.2, 9)), 100, 0.04, 0.0
    )
    assert np.allclose(fitted.moneyness, np.linspace(0.8, 1.2, 41))


def test_summary_is_nan_where_a_maturity_has_no_quotes():
    params = vs.SSVIParams()
    narrow = ssvi_quotes(params, 0.0, 0.0, (30, 60), np.linspace(0.95, 1.05, 7))
    wide = ssvi_quotes(params, 0.0, 0.0, (91, 182), np.linspace(0.6, 1.5, 19))
    summary = vs.surface_summary(vs.fit_market_surface(pd.concat([narrow, wide]), 100, 0.0, 0.0))
    assert summary["Skew 90-110 (vol pts)"][:2].isna().all() and np.isfinite(summary["Skew 90-110 (vol pts)"][2:]).all()
    assert np.isfinite(summary["ATM implied vol"]).all()


def test_implied_vol_is_blank_outside_each_maturitys_quotes_and_local_vol_needs_covered_neighbours():
    params = vs.SSVIParams()
    narrow = ssvi_quotes(params, 0.0, 0.0, (30, 60), np.linspace(0.95, 1.05, 7))
    wide = ssvi_quotes(params, 0.0, 0.0, (91, 182), np.linspace(0.6, 1.5, 19))
    fitted = vs.fit_market_surface(pd.concat([narrow, wide]), 100, 0.0, 0.0)
    m = fitted.moneyness
    assert m[0] == pytest.approx(0.6) and m[-1] == pytest.approx(1.5)
    outside = (m < 0.95 - 1e-9) | (m > 1.05 + 1e-9)
    assert np.isnan(fitted.implied_vol[:2, outside]).all() and np.isfinite(fitted.implied_vol[:2, ~outside]).all()
    assert np.isfinite(fitted.implied_vol[2:]).all()
    # 91 days needs the 60-day slice far outside its quotes in the wings: undefined there, not extrapolated
    far = (m < 0.9) | (m > 1.1)
    assert np.isnan(fitted.local_vol[2, far]).all() and np.isfinite(fitted.local_vol[2, ~outside]).all()
    assert fitted.calendar_violations == 0
    assert any("neighbouring maturity" in note for note in fitted.notes)
    assert fitted.quoted_points == np.isfinite(fitted.implied_vol).sum()


def test_quotes_and_fit_quality_are_kept_on_the_surface():
    quotes, _ = vs.parse_quotes(vs.sample_quotes())
    fitted = vs.fit_market_surface(quotes, 100, 0.04, 0.0)
    assert list(fitted.quotes.columns) == ["strike", "maturity", "iv", "moneyness", "fitted_iv"]
    assert len(fitted.quotes) == len(quotes)
    assert fitted.fit_rmse.shape == fitted.maturities.shape
    first = fitted.quotes[fitted.quotes["maturity"] == fitted.maturities[0]]
    rmse = np.sqrt(np.mean((first["iv"] - first["fitted_iv"]) ** 2))
    assert fitted.fit_rmse[0] == pytest.approx(rmse)
    summary = vs.surface_summary(fitted)
    assert np.allclose(summary["Fit RMSE (vol pts)"], fitted.fit_rmse * 100)
    assert "Fit RMSE (vol pts)" not in vs.surface_summary(vs.ssvi_surface(vs.SSVIParams(), 0.04, 0.0, M, T))


def test_surface_implied_vol_at_a_strike_and_expiry():
    params, r, q = vs.SSVIParams(), 0.03, 0.01
    fitted = vs.fit_market_surface(ssvi_quotes(params, r, q, SAMPLE_DAYS, M), 100, r, q)

    def exact(m, t):
        return float(np.sqrt(params.total_variance(np.log(m) - (r - q) * t, t)[0] / t))

    assert fitted.implied_vol_at(1.05, 365 / 365) == pytest.approx(exact(1.05, 1.0), abs=1e-9)  # quoted maturity
    between = fitted.implied_vol_at(0.9, 0.6)  # total variance interpolated linearly in T at fixed K/F
    assert between == pytest.approx(exact(0.9, 0.6), abs=0.002)
    assert np.isnan(fitted.implied_vol_at(1.0, 3.0)) and np.isnan(fitted.implied_vol_at(1.0, 0.05))
    assert np.isnan(fitted.implied_vol_at(1.0, 29 / 365))  # a day before the first expiry is outside
    assert np.isnan(fitted.implied_vol_at(1.6, 1.0))
    ssvi = vs.ssvi_surface(params, r, q, M, T)
    assert ssvi.implied_vol_at(1.1, 0.7) == pytest.approx(exact(1.1, 0.7), rel=1e-12)
    assert np.isnan(ssvi.implied_vol_at(1.1, 2.5))


def test_the_first_expiry_of_a_file_with_rounded_maturities_is_inside_the_surface():
    # The sample stores 30 days as 0.0822 years, a little after 30/365 = 0.082192: the sidebar's 30 days used to
    # fall outside the quoted region and the surface gave no implied vol for it.
    quotes, _ = vs.parse_quotes(vs.sample_quotes(100, 0.05, 0.0))
    fitted = vs.fit_market_surface(quotes, 100, 0.05, 0.0)
    assert fitted.maturities[0] > 30 / 365
    first = fitted.implied_vol_at(1.0, 30 / 365)
    assert first == pytest.approx(fitted.implied_vol_at(1.0, fitted.maturities[0]), abs=1e-12)
    assert 0.2 < first < 0.3
    assert np.isfinite(fitted.implied_vol_at(1.0, 730.4 / 365)) and np.isnan(fitted.implied_vol_at(1.0, 731 / 365))


def test_strikes_far_from_the_spot_are_reported():
    quotes, _ = vs.parse_quotes(vs.sample_quotes(5000, 0.05, 0.0))
    fitted = vs.fit_market_surface(quotes, 100, 0.05, 0.0)
    assert any("far from the spot 100.00" in note for note in fitted.notes)
    assert not vs.fit_market_surface(quotes, 5000, 0.05, 0.0).notes


def test_a_one_sided_chain_next_to_the_spot_is_not_flagged():
    # Out-of-the-money puts only (strikes 70-95 at spot 100) used to be told the spot was wrong.
    quotes, _ = vs.parse_quotes(vs.sample_quotes(100, 0.05, 0.0))
    puts = vs.fit_market_surface(quotes[quotes["strike"] < 100], 100, 0.05, 0.0)
    calls = vs.fit_market_surface(quotes[quotes["strike"] > 100], 100, 0.05, 0.0)
    assert not puts.notes and not calls.notes
    notes = vs.fit_market_surface(quotes, 175, 0.05, 0.0).notes  # nearest strike 130: 26% below the spot
    assert any("far from the spot 175.00" in note and "26% below it" in note for note in notes)


def test_fit_requires_two_usable_maturities():
    quotes = pd.DataFrame(
        {"strike": [90, 95, 100, 105, 110], "maturity": [0.5] * 5, "iv": [0.25, 0.22, 0.2, 0.19, 0.19]}
    )
    with pytest.raises(vs.QuoteParseError, match="two maturities"):
        vs.fit_market_surface(quotes, 100, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Summary and export
# ---------------------------------------------------------------------------


def test_summary_and_export():
    surface = vs.ssvi_surface(vs.SSVIParams(), 0.04, 0.0, M, T)
    summary = vs.surface_summary(surface)
    assert len(summary) == len(T)
    assert (summary["Skew 90-110 (vol pts)"] > 0).all()  # negative rho = downside skew
    frames = surface.to_frames(100.0)
    assert frames["LocalVol"].shape == (len(T), len(M))
    assert not np.allclose(frames["LocalVol"].to_numpy()[:, 0], frames["LocalVol"].to_numpy()[:, -1])


def test_summary_is_evaluated_at_the_strikes_it_names_not_read_off_the_grid():
    params, r = vs.SSVIParams(), 0.04
    coarse = vs.ssvi_surface(params, r, 0.0, np.linspace(0.7, 1.3, 4), T)  # no grid point at 0.9, 1 or 1.1
    exact = vs.ssvi_surface(params, r, 0.0, np.array([0.9, 1.0, 1.1]), T)
    summary = vs.surface_summary(coarse)
    assert np.allclose(summary["ATM implied vol"], exact.implied_vol[:, 1], rtol=0, atol=1e-12)
    assert np.allclose(summary["ATM local vol"], exact.local_vol[:, 1], rtol=0, atol=1e-12)
    skew = (exact.implied_vol[:, 0] - exact.implied_vol[:, 2]) * 100
    assert np.allclose(summary["Skew 90-110 (vol pts)"], skew, rtol=0, atol=1e-10)
    narrow = vs.surface_summary(vs.ssvi_surface(params, r, 0.0, np.linspace(0.95, 1.3, 9), T))
    assert narrow["Skew 90-110 (vol pts)"].isna().all() and np.isfinite(narrow["ATM implied vol"]).all()


def test_exported_grid_csv_labels_the_strike_axis():
    csv = vs.ssvi_surface(vs.SSVIParams(), 0.04, 0.0, M, T).to_frames(100.0)["ImpliedVol"].to_csv()
    assert csv.splitlines()[0].startswith("Maturity (years) \\ Strike,70.0,")


def test_summary_atm_local_vol_is_nan_where_local_vol_is_undefined():
    moneyness = np.linspace(0.7, 1.3, 41)
    moneyness[20] = 1.0
    maturities = np.linspace(0.08, 2, 30)
    surface = vs.ssvi_surface(vs.SSVIParams(rho=-0.9, eta=3.9), 0.05, 0.0, moneyness, maturities)
    assert np.isnan(surface.local_vol[:, 20]).all()
    # It used to interpolate across the gap or clamp to the nearest defined point (75%, 70%, 595%, ...).
    assert vs.surface_summary(surface)["ATM local vol"].isna().all()
    sound = vs.ssvi_surface(vs.SSVIParams(), 0.05, 0.0, moneyness, maturities)
    assert np.allclose(vs.surface_summary(sound)["ATM local vol"], sound.local_vol[:, 20])


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_quotes_normalises_formats():
    raw = vs.sample_quotes()
    raw = raw.rename(columns={"Maturity": "DTE", "ImpliedVol": "IV", "Strike": "strike"})
    raw["DTE"] = raw["DTE"] * 365
    raw["IV"] = raw["IV"] * 100
    raw = pd.concat([raw, raw.head(2), pd.DataFrame([{"strike": "abc", "DTE": 30, "IV": 20}])])
    clean, messages = vs.parse_quotes(raw)
    assert list(clean.columns) == ["strike", "maturity", "iv"]
    assert clean["iv"].max() < 1.0
    text = " ".join(messages)
    assert "days" in text and "percent" in text and "duplicate" in text and "Dropped" in text


def test_parse_quotes_states_how_each_column_was_read():
    _, messages = vs.parse_quotes(vs.sample_quotes())
    assert messages[0] == "Read strike from 'Strike', maturity from 'Maturity' (years), implied vol from 'ImpliedVol'."
    assert "decimals" in messages[1]


def test_implied_volatility_column_wins_over_a_volume_column():
    raw = vs.sample_quotes().rename(columns={"ImpliedVol": "Implied Volatility"})
    raw["Vol"] = np.random.default_rng(1).integers(1, 450, len(raw))  # trading volume, listed first
    raw = raw[["Strike", "Maturity", "Vol", "Implied Volatility"]]
    clean, _ = vs.parse_quotes(raw)
    assert clean["iv"].between(0.15, 0.6).all()


@pytest.mark.parametrize(
    "header",
    [
        "Maturity (years)",
        "Time to maturity",
        "Time to expiry",
        "Expiry (years)",
        "Maturity (yrs)",
        "T (years)",
        "Maturity_years",
        "Expiry",
        "time_to_maturity",
    ],
)
def test_maturity_headers_in_years_are_recognised(header):
    raw = vs.sample_quotes().rename(columns={"Maturity": header})
    clean, messages = vs.parse_quotes(raw)
    assert clean["maturity"].max() == pytest.approx(2.0)
    assert f"maturity from '{header}' (years)" in messages[0]


@pytest.mark.parametrize("header", ["Days", "DTE", "Days to maturity", "Days to Expiration", "Maturity (days)"])
def test_maturity_headers_in_days_are_recognised(header):
    raw = vs.sample_quotes().rename(columns={"Maturity": header})
    raw[header] = (raw[header] * 365).round()
    clean, messages = vs.parse_quotes(raw)
    assert clean["maturity"].max() == pytest.approx(2.0)
    assert "days" in messages[0]


def test_maturities_that_can_only_be_days_are_converted():
    raw = vs.sample_quotes(100, 0.04, 0.0)
    raw["Maturity"] = (raw["Maturity"] * 365).round()
    clean, messages = vs.parse_quotes(raw)
    assert clean["maturity"].max() == pytest.approx(2.0)
    assert any("only makes sense in days" in m for m in messages)
    vs.fit_market_surface(clean, 100, 0.04, 0.0)  # used to fail with a misleading 'common strike range' error


@pytest.mark.parametrize("header", ["IV (%)", "Implied Vol %", "iv_pct", "Implied volatility (percent)"])
def test_percent_in_the_header_sets_the_unit(header):
    raw = vs.sample_quotes().rename(columns={"ImpliedVol": header})
    raw[header] = raw[header] * 10  # percent vols of a low-vol underlying: median 2.4, which looks like decimals
    clean, messages = vs.parse_quotes(raw)
    assert clean["iv"].median() == pytest.approx(0.024, rel=0.05)
    assert any("from the column header" in m for m in messages)


def test_ambiguous_vol_units_are_called_out():
    raw = vs.sample_quotes()
    raw["ImpliedVol"] = raw["ImpliedVol"] * 10
    clean, messages = vs.parse_quotes(raw)
    assert clean["iv"].median() > 1  # read as decimals, as before ...
    assert any("ambiguous" in m and "IV (%)" in m for m in messages)  # ... but the user is told


def test_percent_signs_and_thousands_separators_are_stripped():
    raw = vs.sample_quotes(3500)
    raw["Strike"] = [f"{k:,.2f}" for k in raw["Strike"]]
    raw["ImpliedVol"] = [f"{v * 100:.2f}%" for v in raw["ImpliedVol"]]
    frame = pd.read_csv(io.StringIO(raw.to_csv(index=False)))
    clean, messages = vs.parse_quotes(frame)
    assert clean["strike"].min() == pytest.approx(2450.0) and clean["strike"].max() == pytest.approx(4550.0)
    assert clean["iv"].between(0.15, 0.6).all()
    assert any("% signs" in m for m in messages)


def test_a_column_without_numbers_is_named():
    raw = vs.sample_quotes()
    raw["ImpliedVol"] = "n/a"
    with pytest.raises(vs.QuoteParseError, match="column 'ImpliedVol' has no numeric values"):
        vs.parse_quotes(raw)


def test_expiry_dates_without_a_days_column_are_explained():
    raw = vs.sample_quotes().rename(columns={"Maturity": "Expiry"})
    raw["Expiry"] = "2026-12-18"
    with pytest.raises(vs.QuoteParseError, match=r"column 'Expiry' holds dates .* add a Days column"):
        vs.parse_quotes(raw)


def _with_dte_and_expiry(expiry_format: str | None) -> pd.DataFrame:
    raw = vs.sample_quotes(100, 0.05, 0.0)
    raw["DTE"] = (raw["Maturity"] * 365).round().astype(int)
    expiry = pd.Timestamp("2026-10-13") + pd.to_timedelta(raw["DTE"], unit="D")
    raw["Expiration"] = expiry.dt.strftime(expiry_format).astype(int if expiry_format == "%Y%m%d" else str)
    return raw[["Expiration", "DTE", "Strike", "ImpliedVol"]]


@pytest.mark.parametrize("expiry_format", ["%Y-%m-%d", "%Y%m%d"])
def test_a_days_column_wins_over_an_expiry_column(expiry_format):
    # Chain exports often carry the expiry date next to the days to expiry. The date column used to be read as
    # the maturity: date strings were refused, and YYYYMMDD integers became 55,510-year maturities.
    raw = _with_dte_and_expiry(expiry_format)
    frame = pd.read_csv(io.StringIO(raw.to_csv(index=False)))
    clean, messages = vs.parse_quotes(frame)
    assert clean["maturity"].min() == pytest.approx(30 / 365) and clean["maturity"].max() == pytest.approx(2.0)
    assert "maturity from 'DTE' (days" in messages[0]
    assert any("Ignored 'Expiration'" in m for m in messages)


@pytest.mark.parametrize("expiry_format", ["%Y-%m-%d", "%Y%m%d"])
def test_an_expiry_date_column_does_not_hide_a_maturity_in_years(expiry_format):
    raw = _with_dte_and_expiry(expiry_format)
    raw["TTM"] = (raw.pop("DTE") / 365).round(6)
    clean, messages = vs.parse_quotes(raw)
    assert clean["maturity"].max() == pytest.approx(2.0)
    assert "maturity from 'TTM' (years)" in messages[0]


def test_yyyymmdd_expiries_alone_are_refused_not_read_as_days():
    raw = _with_dte_and_expiry("%Y%m%d").drop(columns="DTE")
    with pytest.raises(vs.QuoteParseError, match=r"'Expiration' goes up to 20,281,012.*YYYYMMDD"):
        vs.parse_quotes(raw)


@pytest.mark.parametrize("header", ["Maturity", "Expiry", "T", "Tenor"])
def test_whole_number_maturities_that_look_like_days_are_converted(header):
    # A weekly chain at 7, 14, 21 and 28 days in a maturity column was read as 7-28 years without a word.
    quotes = ssvi_quotes(vs.SSVIParams(), 0.04, 0.0, (7, 14, 21, 28), np.linspace(0.9, 1.1, 9))
    raw = pd.DataFrame({"Strike": quotes["strike"], header: (quotes["maturity"] * 365).round(), "IV": quotes["iv"]})
    clean, messages = vs.parse_quotes(raw)
    assert clean["maturity"].max() == pytest.approx(28 / 365)
    assert any("whole numbers from 7 to 28, which look like days" in m for m in messages)


def test_whole_years_stay_years_when_the_header_says_so_and_are_flagged_otherwise():
    quotes = ssvi_quotes(vs.SSVIParams(), 0.04, 0.0, (365, 730, 1825, 3650), np.linspace(0.8, 1.2, 9))
    years = (quotes["maturity"]).round()
    explicit, messages = vs.parse_quotes(
        pd.DataFrame({"Strike": quotes["strike"], "Maturity (years)": years, "IV": quotes["iv"]})
    )
    assert explicit["maturity"].max() == pytest.approx(10.0)
    assert not any("days" in m for m in messages[1:])
    short = quotes[quotes["maturity"] < 3]
    short_years = short["maturity"].round()
    clean, messages = vs.parse_quotes(
        pd.DataFrame({"Strike": short["strike"], "Maturity": short_years, "IV": short["iv"]})
    )
    assert clean["maturity"].max() == pytest.approx(2.0)  # 1 and 2: read as years, but the user is told
    assert any("whole numbers up to 2, read as years" in m for m in messages)


def test_headers_wrapped_in_brackets_are_read():
    raw = vs.sample_quotes()
    raw["Maturity"] = (raw["Maturity"] * 365).round()
    clean, messages = vs.parse_quotes(
        raw.rename(columns={"Strike": "[STRIKE]", "Maturity": "[DTE]", "ImpliedVol": "[IV]"})
    )
    assert len(clean) == 104 and clean["maturity"].max() == pytest.approx(2.0)
    assert messages[0].startswith("Read strike from '[STRIKE]', maturity from '[DTE]' (days")


@pytest.mark.parametrize("header", ["IV (% p.a.)", "Vol (in %)", "Implied vol [%, mid]"])
def test_a_percent_sign_anywhere_in_the_header_sets_the_unit(header):
    raw = vs.sample_quotes().rename(columns={"ImpliedVol": header})
    raw[header] = raw[header] * 10  # median 2.4: read as decimals without the header unit
    clean, messages = vs.parse_quotes(raw)
    assert clean["iv"].median() == pytest.approx(0.024, rel=0.05)
    assert any("from the column header" in m for m in messages)


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"a": [1], "b": [2]}),
        pd.DataFrame({"Strike": [-1.0], "Maturity": [1.0], "ImpliedVol": [0.2]}),
        pd.DataFrame({"Strike": [100.0], "Maturity": [1.0]}),
    ],
)
def test_parse_quotes_rejects_unusable_files(frame):
    with pytest.raises(vs.QuoteParseError):
        vs.parse_quotes(frame)


# ---------------------------------------------------------------------------
# Option prices
# ---------------------------------------------------------------------------


def test_prices_are_inverted_into_the_same_vols():
    prices = vs.sample_quotes(100, 0.05, 0.01, prices=True)
    assert list(prices.columns) == ["Strike", "Days", "Type", "Price"]
    from_prices, messages = vs.parse_quotes(prices, 100, 0.05, 0.01)
    from_vols, _ = vs.parse_quotes(vs.sample_quotes(100, 0.05, 0.01))
    from_prices["maturity"] = from_prices["maturity"].round(4)  # the implied-vol sample rounds maturities
    merged = from_prices.merge(from_vols, on=["strike", "maturity"], suffixes=("_price", "_vol"))
    assert len(merged) == len(from_vols) == 104
    gap = np.abs(merged["iv_price"] - merged["iv_vol"])
    assert np.median(gap) < 1e-4  # both samples are rounded to 4 decimals ...
    assert gap.max() < 0.005  # ... which moves the vol of a 0.003 price (130 call, 30 days) by 0.35 points
    assert any("Black-Scholes-Merton" in m for m in messages)


def test_bid_ask_quotes_keep_the_out_of_the_money_side_and_drop_impossible_prices():
    r, q, spot, sigma = 0.03, 0.01, 100.0, 0.25
    rows = []
    for days in (30, 91, 182):
        t = days / 365
        for strike in range(80, 121, 5):
            for kind in ("C", "P"):
                value = bs.price(spot, strike, t, r, sigma, q, "call" if kind == "C" else "put")
                rows.append({"Strike": strike, "DTE": days, "C/P": kind, "Bid": value - 0.01, "Ask": value + 0.01})
    rows.append({"Strike": 125, "DTE": 30, "C/P": "C", "Bid": 150.0, "Ask": 151.0})  # above the spot: impossible
    rows.append({"Strike": 125, "DTE": 91, "C/P": "C", "Bid": 0.0, "Ask": 0.05})  # no bid
    frame = pd.DataFrame(rows)
    frame["Vol"] = 10  # trading volume next to prices
    clean, messages = vs.parse_quotes(frame, spot, r, q)
    text = " ".join(messages)
    assert "out-of-the-money" in text and "no-arbitrage" in text and "zero bid" in text and "trading volume" in text
    assert len(clean) == 27  # one option per strike and expiry
    assert np.allclose(clean["iv"], sigma, atol=0.01)
    fitted = vs.fit_market_surface(clean, spot, r, q)
    assert fitted.implied_vol_at(1.0, 0.4) == pytest.approx(sigma, abs=0.005)


def test_prices_need_a_type_column_and_market_inputs():
    frame = vs.sample_quotes(prices=True)
    with pytest.raises(vs.QuoteParseError, match="Type column"):
        vs.parse_quotes(frame.drop(columns="Type"), 100, 0.04, 0.0)
    with pytest.raises(vs.QuoteParseError, match="spot price"):
        vs.parse_quotes(frame)
