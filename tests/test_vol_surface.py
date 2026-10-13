import io
import re

import numpy as np
import pandas as pd
import pytest

from options_analyzer import black_scholes as bs
from options_analyzer import vol_surface as vs
from options_analyzer.implied_vol import implied_volatility

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


def ssvi_slice(params, t) -> vs.SVISlice:
    """The raw-SVI parameters of an SSVI slice."""
    theta = params.theta(t)
    phi = params.phi(theta)
    rho = params.rho
    return vs.SVISlice(theta * (1 - rho * rho) / 2, theta * phi / 2, rho, -rho / phi, np.sqrt(1 - rho * rho) / phi)


def vol_rmse(svi, y, iv, t) -> float:
    return float(np.sqrt(np.mean((np.sqrt(np.maximum(svi.total_variance(y)[0], 0.0) / t) - iv) ** 2)))


STEEP = vs.SSVIParams(0.159804, 0.469738, 3.961008, -0.866546, 0.482821, 0.438353)  # an equity index skew


def test_svi_fit_follows_a_steep_nearly_linear_skew():
    # Every starting point used to be a flat smile (b of 1e-10 to 1e-7): the slope was lost when the start's
    # rho was clipped to 0.99, and the optimiser could not leave it. The fit ended at rho = -0.999, s = 148 with
    # an RMSE of 3.45 vol points, while the generating smile is 0.30 points from these quotes.
    t = 30 / 365
    y = np.linspace(-0.110539, 0.103477, 17)
    iv = np.array(
        [
            0.284765,
            0.281186,
            0.277529,
            0.260996,
            0.261928,
            0.251624,
            0.252496,
            0.242422,
            0.2396,
            0.22634,
            0.219801,
            0.214376,
            0.205934,
            0.198332,
            0.192781,
            0.17986,
            0.171056,
        ]
    )  # SSVI quotes plus 0.3 vol-point noise
    assert all(b > 1e-4 for _, b, _, _, _ in vs._svi_start_points(y, iv, t))
    assert vol_rmse(vs.fit_svi(y, iv, t), y, iv, t) <= 1.5 * vol_rmse(ssvi_slice(STEEP, t), y, iv, t)


@pytest.mark.parametrize("bad_vol", [0.001, 0.002])
def test_one_very_low_implied_vol_does_not_abort_the_fit(bad_vol):
    # One quote at 0.1% next to normal vols made a start-point normal matrix exactly singular: numpy's
    # LinAlgError ('Singular matrix') aborted the whole fit (80 of the sample's 104 placements at 0.001).
    raw = vs.sample_quotes(100, 0.05, 0.0)
    clean_fit = vs.fit_market_surface(vs.parse_quotes(raw, 100, 0.05)[0], 100, 0.05, 0.0)
    raw.loc[2, "ImpliedVol"] = bad_vol  # strike 80 at the first maturity
    quotes, _ = vs.parse_quotes(raw, 100, 0.05)
    t = quotes["maturity"].iloc[0]
    y = np.log(quotes.loc[quotes["maturity"] == t, "strike"].to_numpy() / 100) - 0.05 * t
    assert len(vs._svi_start_points(y, quotes.loc[quotes["maturity"] == t, "iv"].to_numpy(), t)) == 3
    fitted = vs.fit_market_surface(quotes, 100, 0.05, 0.0)
    assert fitted.fit_rmse[0] > vs.POOR_FIT_RMSE  # the bad slice is reported, not hidden ...
    assert any("0.0822y" in note and "SVI smile" in note for note in fitted.notes)
    assert np.array_equal(fitted.implied_vol[1:], clean_fit.implied_vol[1:], equal_nan=True)  # ... the rest unchanged


def test_placeholder_implied_vols_are_dropped_with_a_message():
    # Some chains write 1e-05 where there is no implied vol; one per maturity made every fit fail.
    raw = vs.sample_quotes(100, 0.05, 0.0)
    raw.loc[raw.index % 13 == 2, "ImpliedVol"] = 1e-05
    clean, messages = vs.parse_quotes(raw, 100, 0.05)
    assert len(clean) == 104 - 8 and clean["iv"].min() > 0.1
    assert any("Dropped 8 quote(s) with an implied vol below 0.01%" in m and "placeholder" in m for m in messages)
    vs.fit_market_surface(clean, 100, 0.05, 0.0)


@pytest.mark.parametrize("header", ["IV (%)", "IV"])
def test_the_placeholder_threshold_is_0_01_percent_in_a_file_of_percentages(header):
    # The threshold was compared with the file's own numbers, so in percentages it was 0.0001% and the message
    # gave it as a bare "0.0001"; 0.005 (0.005%) was kept as a vol of 0.00005.
    raw = vs.sample_quotes(100, 0.05, 0.0).rename(columns={"ImpliedVol": header})
    raw[header] = (raw[header] * 100).round(2)  # percentages, named in the header or recognised by their size
    raw.loc[raw.index % 13 == 2, header] = 0.005
    raw.loc[raw.index % 13 == 5, header] = 1e-05
    clean, messages = vs.parse_quotes(raw, 100, 0.05)
    assert len(clean) == 104 - 16 and clean["iv"].min() > 0.1
    assert any("Dropped 16 quote(s) with an implied vol below 0.01%" in m for m in messages)


def test_svi_fit_reaches_the_generating_accuracy_on_steep_noisy_skews():
    # Steep equity skews (rho -0.9 to -0.7) quoted within one standard deviation with 0.2 vol-point noise: 8% of
    # them used to end in a flat local minimum, up to 4.2 vol points from the quotes.
    rng = np.random.default_rng(99)
    for _ in range(40):
        rho = rng.uniform(-0.9, -0.7)
        params = vs.SSVIParams(
            rng.uniform(0.12, 0.4),
            rng.uniform(0.15, 0.35),
            rng.uniform(0.5, 4),
            rho,
            rng.uniform(0.2, 0.8) * 2 / (1 + abs(rho)),
            rng.uniform(0.3, 0.5),
        )
        t = rng.choice((7, 14, 30, 60, 91, 182, 365)) / 365
        sd = np.sqrt(params.theta(t))
        y = np.linspace(-1.3 * sd, 0.8 * sd, rng.integers(9, 25))
        iv = np.sqrt(params.total_variance(y, t)[0] / t) + rng.normal(0, 0.002, len(y))
        assert vol_rmse(vs.fit_svi(y, iv, t), y, iv, t) <= 1.5 * vol_rmse(ssvi_slice(params, t), y, iv, t)


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


def test_a_steep_noisy_equity_surface_is_fitted_without_a_poor_fit_note():
    # The 30-day smile of this surface ended in a flat local minimum: fit RMSE 2.3 vol points, implied vol 3.5
    # and local vol 8.4 points off, and a note telling the user to check quotes that were fine.
    rng = np.random.default_rng(1071)
    rows = []
    for days in (14, 30, 60, 91, 182):
        t = days / 365
        for y in np.linspace(-1.2, 1.0, 17) * np.sqrt(STEEP.theta(t)):
            iv = np.sqrt(STEEP.total_variance(y, t)[0] / t) + rng.normal(0, 0.002)
            rows.append({"strike": 100 * np.exp(y + 0.05 * t), "maturity": t, "iv": float(iv)})
    fitted = vs.fit_market_surface(pd.DataFrame(rows), 100, 0.05, 0.0)
    true = vs.ssvi_surface(STEEP, 0.05, 0.0, fitted.moneyness, fitted.maturities)
    assert fitted.fit_rmse.max() < 0.003  # measured 0.14-0.21 vol points against 0.2 points of noise
    assert not any("SVI smile" in note for note in fitted.notes)
    assert np.nanmax(np.abs(fitted.implied_vol - true.implied_vol)) < 0.005  # measured 0.29 vol points
    assert np.nanmax(np.abs(fitted.local_vol - true.local_vol)[:-1]) < 0.015  # measured 0.77 vol points


def test_a_poor_fit_note_does_not_blame_quotes_that_svi_cannot_shape():
    # A humped (concave) smile is exact data that no SVI curve follows; the note used to say "check those quotes".
    params = vs.SSVIParams()
    quotes = ssvi_quotes(params, 0.0, 0.0, (91, 182, 365), np.linspace(0.8, 1.2, 9))
    humped = quotes["maturity"] == quotes["maturity"].min()
    quotes.loc[humped, "iv"] = 0.25 - 1.5 * np.log(quotes.loc[humped, "strike"] / 100) ** 2
    fitted = vs.fit_market_surface(quotes, 100, 0.0, 0.0)
    notes = [note for note in fitted.notes if "SVI smile" in note]
    assert fitted.fit_rmse[0] > vs.POOR_FIT_RMSE and len(notes) == 1
    assert "0.249y" in notes[0] and "a shape a single SVI curve cannot follow" in notes[0]
    assert "check those quotes" not in notes[0]


def _flat_last_day(days) -> pd.DataFrame:
    """Rounded SSVI quotes where the 366-day expiry has the same total variance as the 365-day one."""
    quotes = ssvi_quotes(vs.SSVIParams(), 0.0, 0.0, days, np.linspace(0.8, 1.2, 9), decimals=4)
    year, next_day = (quotes["maturity"] == round(d / 365, 10) for d in (365, 366))
    quotes.loc[next_day, "iv"] = np.round(quotes.loc[year, "iv"].to_numpy() * np.sqrt(365 / 366), 4)
    return quotes


def test_total_variance_flat_within_quote_precision_is_not_calendar_arbitrage():
    # From 365 to 366 days total variance changes by at most 2.4e-5, within the quotes' precision. As the last
    # maturity this counted 31 calendar-arbitrage points; on both rows local vol was up to 30 vol points too low
    # (3.7% at the money against a true 20.6%), because the unresolved one-day secant, about 0, set dw/dT.
    params = vs.SSVIParams()
    last = vs.fit_market_surface(_flat_last_day((30, 60, 91, 182, 365, 366)), 100, 0.0, 0.0)
    assert last.calendar_violations == 0 and last.butterfly_violations == 0
    assert np.isnan(last.local_vol[-1]).all() and np.isfinite(last.local_vol[:-1]).all()
    assert any("changes by less than the quotes' precision" in n and "not calendar arbitrage" in n for n in last.notes)
    inside = vs.fit_market_surface(_flat_last_day((30, 60, 91, 182, 365, 366, 730)), 100, 0.0, 0.0)
    assert inside.calendar_violations == 0 and np.isfinite(inside.local_vol).all()
    for fitted in (last, inside):
        true = vs.ssvi_surface(params, 0.0, 0.0, fitted.moneyness, fitted.maturities)
        error = np.abs(fitted.local_vol - true.local_vol)[4:6]  # the 365 and 366-day rows
        assert np.nanmax(error) < 0.02  # measured 1.6 vol points: the 182-365 day secant alone


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
    sparse = pd.concat([quotes, pd.DataFrame({"strike": [100.0], "maturity": [1.0], "iv": [0.2]})])
    with pytest.raises(vs.QuoteParseError, match=r"the quotes have 2 maturities, 1 of them with 5\+ strikes"):
        vs.fit_market_surface(sparse, 100, 0.0, 0.0)


@pytest.mark.parametrize("decimals", [3, 5])
def test_times_to_expiry_of_one_expiry_quoted_through_a_session_are_one_maturity(decimals):
    # Days to expiry taken from each quote's own time (29.86, 29.74, 29.96, ...) made every row its own maturity:
    # 51 one-strike slices, and the file was refused for lacking two maturities with five strikes.
    params, r = vs.SSVIParams(), 0.04
    rng = np.random.default_rng(5)
    rows = []
    for days in (30, 58, 93):
        for strike in np.arange(80, 120.01, 2.5):
            dte = round(days - rng.uniform(0, 0.27), decimals)  # up to 6.5 hours before the close
            t = dte / 365
            iv = np.sqrt(params.total_variance(np.log(strike / 100) - r * t, t)[0] / t)
            rows.append({"Strike": strike, "DTE": dte, "IV": float(iv)})
    clean, messages = vs.parse_quotes(pd.DataFrame(rows), 100.0, r)
    assert clean["maturity"].nunique() == 3
    assert any(re.match(r"Read \d+ times to expiry as 3 expiries: times within half a day", m) for m in messages)
    fitted = vs.fit_market_surface(clean, 100.0, r, 0.0)
    assert np.allclose(fitted.maturities * 365, (30, 58, 93), atol=0.3)
    true = vs.ssvi_surface(params, r, 0.0, fitted.moneyness, fitted.maturities)
    assert np.nanmax(np.abs(fitted.implied_vol - true.implied_vol)) < 5e-4  # total variance kept per quote
    assert fitted.calendar_violations == 0 and fitted.butterfly_violations == 0


def test_expiries_a_day_apart_stay_apart_and_a_chain_is_never_merged_over_more_than_half_a_day():
    days = np.array([1.0, 2.0, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2])
    labels = vs._expiries(days / 365)
    assert labels.tolist() == [0, 1, 2, 2, 2, 3, 3, 3, 4]  # each expiry spans at most half a day


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
    assert list(clean.columns) == ["strike", "maturity", "iv", "price_error"]
    assert clean["price_error"].isna().all()  # implied vols: the prices behind them are unknown
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


@pytest.mark.parametrize(
    "header, values, days_each, described",
    [
        ("Maturity (months)", (0.5, 1, 1.5, 2, 3), 365 / 12, "months, divided by 12"),
        ("Maturity (months)", (1, 2, 3), 365 / 12, "months, divided by 12"),
        ("Maturity_months", (1, 2, 3, 6, 9, 12), 365 / 12, "months, divided by 12"),
        ("Tenor (mo)", (1, 2, 3), 365 / 12, "months, divided by 12"),
        ("Maturity (weeks)", (1, 2, 4, 8), 7, "weeks of 7 days"),
        ("DTE (hours)", (24, 48, 72, 168), 1 / 24, "hours, converted to years with ACT/365"),
        ("Time to expiry (minutes)", (1440, 2880, 10080), 1 / 1440, "minutes, converted to years with ACT/365"),
    ],
)
def test_a_time_unit_stated_in_the_header_is_converted(header, values, days_each, described):
    # The header's unit was thrown away: months were read as years (0.5-3 months became 182-1,095 days) and hours
    # or minutes as days, with no warning and a message repeating the wrong unit.
    years = np.array(values, dtype=float) * days_each / 365
    rows = [
        {"Strike": 100 * m, header: v, "IV": np.sqrt(vs.SSVIParams().total_variance(np.log(m), t)[0] / t)}
        for v, t in zip(values, years, strict=True)
        for m in np.linspace(0.8, 1.2, 9)
    ]
    clean, messages = vs.parse_quotes(pd.DataFrame(rows), 100.0)
    assert np.allclose(np.sort(clean["maturity"].unique()), years, rtol=1e-12)
    assert f"maturity from '{header}' ({described}" in messages[0]
    assert not any(isinstance(m, vs.QuoteWarning) for m in messages)


def test_a_months_header_recovers_the_surface():
    params = vs.SSVIParams()
    months = (0.5, 1, 1.5, 2, 3)
    rows = []
    for mo in months:
        t = mo / 12
        for m in np.linspace(0.8, 1.2, 9):
            iv = np.sqrt(params.total_variance(np.log(m) - 0.05 * t, t)[0] / t)
            rows.append({"Strike": 100 * m, "Maturity (months)": mo, "IV": iv})
    fitted = vs.fit_market_surface(vs.parse_quotes(pd.DataFrame(rows), 100.0, 0.05)[0], 100.0, 0.05, 0.0)
    assert np.allclose(fitted.maturities, np.array(months) / 12)
    assert fitted.butterfly_violations == 0  # 66 points when the months were read as years
    lv_err, iv_err = relative_local_vol_error(fitted, params, 0.05, 0.0)
    assert iv_err.max() < 1e-6 and lv_err[1:-1].max() < 0.01


@pytest.mark.parametrize("header", ["Maturity (M)", "Tenor (m)", "DTE (m)"])
def test_the_ambiguous_unit_m_is_refused(header):
    rows = [{"Strike": k, header: v, "IV": 0.2} for v in (1, 2, 3) for k in range(90, 111, 5)]
    with pytest.raises(vs.QuoteParseError, match=rf"unit 'm' in '{re.escape(header)}' could mean months or minutes"):
        vs.parse_quotes(pd.DataFrame(rows))


@pytest.mark.parametrize("tenors", [("1M", "3M", "6M"), ("1m", "3m", "6m"), ("3 M", "6 M", "9 M"), ("1W", "1M", "3M")])
def test_tenor_strings_are_called_tenors_not_dates(tenors):
    # pd.to_datetime reads '1M' as a time on 1 January of the year 1, so tenors were reported as "dates".
    frame = pd.DataFrame(
        {"Strike": [90, 100, 110] * len(tenors), "Tenor": np.repeat(tenors, 3), "IV": [0.25, 0.2, 0.22] * len(tenors)}
    )
    with pytest.raises(vs.QuoteParseError) as error:
        vs.parse_quotes(frame)
    assert f"column 'Tenor' holds tenors (e.g. '{tenors[0]}')" in str(error.value)
    assert "dates" not in str(error.value) and "add a Days column" in str(error.value)


@pytest.mark.parametrize("times", [("5h", "10h"), ("16:00", "09:30"), ("4pm", "9:30 am")])
def test_times_of_day_are_not_called_dates(times):
    # pandas dates a time alone on 1 January of the year 1 ("5h") or today ("16:00").
    frame = pd.DataFrame({"Strike": [90, 100, 110] * 2, "Expiry": [times[0]] * 3 + [times[1]] * 3, "IV": [0.2] * 6})
    with pytest.raises(vs.QuoteParseError, match="has no numeric values"):
        vs.parse_quotes(frame)


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
    warnings = [m for m in messages if isinstance(m, vs.QuoteWarning)]
    assert len(warnings) == 1 and "ambiguous" in warnings[0] and "IV (%)" in warnings[0]  # ... but the user is warned


def test_percent_signs_and_thousands_separators_are_stripped():
    raw = vs.sample_quotes(3500)
    raw["Strike"] = [f"{k:,.2f}" for k in raw["Strike"]]
    raw["ImpliedVol"] = [f"{v * 100:.2f}%" for v in raw["ImpliedVol"]]
    frame = pd.read_csv(io.StringIO(raw.to_csv(index=False)))
    clean, messages = vs.parse_quotes(frame)
    assert clean["strike"].min() == pytest.approx(2450.0) and clean["strike"].max() == pytest.approx(4550.0)
    assert clean["iv"].between(0.15, 0.6).all()
    assert any("% signs" in m for m in messages)


def test_decimal_commas_are_read_as_decimal_points():
    # Every comma was deleted as a thousands separator: '25,3%' became 253% and '80,5' became 805, silently.
    raw = vs.sample_quotes(100, 0.04, 0.0)
    text = raw.assign(
        Strike=[f"{k + 0.5:.1f}".replace(".", ",") for k in raw["Strike"]],
        Maturity=[f"{t:.4f}".replace(".", ",") for t in raw["Maturity"]],
        ImpliedVol=[f"{v * 100:.2f}%".replace(".", ",") for v in raw["ImpliedVol"]],
    )
    frame = pd.read_csv(io.StringIO(text.to_csv(index=False)))  # quoted, comma-separated
    clean, messages = vs.parse_quotes(frame)
    expected, _ = vs.parse_quotes(raw.assign(Strike=raw["Strike"] + 0.5))
    assert np.allclose(clean.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-12, equal_nan=True)
    assert clean["strike"].min() == pytest.approx(70.5) and clean["iv"].max() < 0.6
    for column in ("Strike", "Maturity", "ImpliedVol"):
        example = text[column].iloc[0].rstrip("%")  # '70,5', '0,0822', '54,..'
        assert f"Read the commas in '{column}' as decimal points (e.g. '{example}')." in messages


@pytest.mark.parametrize("values", [["1.234,5", "1.300,0"], ["25.3", "25,3"], ["1,2,3", "4,5,6"]])
def test_commas_that_are_neither_thousands_nor_decimals_are_refused(values):
    frame = pd.DataFrame({"Strike": [90, 100] * 3, "Days": [30] * 6, "IV": values * 3})
    with pytest.raises(vs.QuoteParseError, match=r"column 'IV' has commas that are neither thousands separators"):
        vs.parse_quotes(frame)


@pytest.mark.parametrize("text, value", [("1,250", 1250.0), ("$2,450.00", 2450.0), ("-1,250.5", -1250.5)])
def test_thousands_separators_still_read_as_thousands(text, value):
    assert vs._numeric(pd.DataFrame({"x": [text, "7"]}), "x")[0].tolist() == [value, 7.0]


def test_a_column_without_numbers_is_named():
    raw = vs.sample_quotes()
    raw["ImpliedVol"] = "n/a"
    with pytest.raises(vs.QuoteParseError, match="column 'ImpliedVol' has no numeric values"):
        vs.parse_quotes(raw)


@pytest.mark.parametrize(
    "dates",
    [
        ("2026-11-20", "2026-12-18"),
        ("Nov 20", "Dec 18"),
        ("11/20", "12/18"),
        ("20 Nov", "18 Dec"),
        ("Nov-26", "Dec-26"),
        ("20NOV26", "18DEC26"),
    ],
)
def test_expiry_dates_without_a_days_column_are_explained(dates):
    # pandas gives a date without a year ('Nov 20') the year 1, so such dates got only "has no numeric values".
    raw = vs.sample_quotes().rename(columns={"Maturity": "Expiry"})
    raw["Expiry"] = np.where(raw.index % 2, *dates)
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


def _maturity_frame(header: str, values: tuple) -> pd.DataFrame:
    """SSVI implied vols with maturities ``values`` in a column ``header`` (generated as years up to 10, else days)."""
    years = np.array(values, dtype=float) if max(values) <= 10 else np.array(values, dtype=float) / 365
    quotes = []
    for value, t in zip(values, years, strict=True):
        for m in np.linspace(0.8, 1.2, 9):
            iv = np.sqrt(vs.SSVIParams().total_variance(np.log(m) - 0.04 * t, t)[0] / t)
            quotes.append({"Strike": 100 * m, header: value, "IV": float(iv)})
    return pd.DataFrame(quotes)


@pytest.mark.parametrize("header", ["Maturity", "Expiry", "T", "Tenor", "TTM", "Time to maturity"])
@pytest.mark.parametrize("values", [(1, 2, 3, 5, 7, 10), (7, 14, 21, 28), (1, 2, 5)])
def test_whole_number_maturities_that_could_be_years_or_days_are_refused(header, values):
    # A 1-10 year tenor grid was read as 1-10 days (and a surface built on it); a weekly chain at 7-28 days used
    # to be read as 7-28 years. Neither header says which, so the user is asked to.
    with pytest.raises(
        vs.QuoteParseError, match=r"could be years or days; name the column 'Days'.*'Maturity \(years\)'"
    ):
        vs.parse_quotes(_maturity_frame(header, values))
    years, _ = vs.parse_quotes(_maturity_frame("Maturity (years)", values))
    days, _ = vs.parse_quotes(_maturity_frame("Days", values))
    assert years["maturity"].max() == pytest.approx(max(values))
    assert days["maturity"].max() == pytest.approx(max(values) / 365)


def test_fractional_days_in_a_maturity_column_get_a_units_warning():
    # 6.65 to 27.65 days were read as 6.65 to 27.65 years with no word about units.
    clean, messages = vs.parse_quotes(_maturity_frame("Maturity", (6.65, 13.65, 20.65, 27.65)))
    warnings = [m for m in messages if isinstance(m, vs.QuoteWarning)]
    assert clean["maturity"].max() == pytest.approx(27.65)
    assert len(warnings) == 1 and "go up to 27.65, read as years; name the column 'Days'" in warnings[0]
    _, labelled = vs.parse_quotes(_maturity_frame("Maturity (years)", (6.65, 13.65, 20.65, 27.65)))
    _, short = vs.parse_quotes(vs.sample_quotes())  # fractions up to 2 years: nothing to warn about
    assert not any(isinstance(m, vs.QuoteWarning) for m in labelled + short)


NEAR_THE_MONEY = np.arange(0.95, 1.0501, 0.01)


def _days_chain(header: str, days=(2.65, 4.65, 9.65), moneyness=NEAR_THE_MONEY):
    """Rounded SSVI implied vols of options ``days`` calendar days long, with the days in a column ``header``."""
    quotes = ssvi_quotes(vs.SSVIParams(), 0.0, 0.0, days, moneyness, decimals=4)
    in_days = (quotes["maturity"] * 365).round(6)
    return pd.DataFrame({"Strike": quotes["strike"].round(2), header: in_days, "IV": quotes["iv"]})


@pytest.mark.parametrize("spot", [100.0, None])
def test_fractional_days_up_to_ten_get_a_units_warning(spot):
    # 2.65, 4.65 and 9.65 days under 'Maturity' were read as years with no word about units: only fractions
    # above 10 were flagged. Anything beyond the 3 years listed options rarely exceed now gets the warning.
    clean, messages = vs.parse_quotes(_days_chain("Maturity"), spot)
    warnings = [m for m in messages if isinstance(m, vs.QuoteWarning)]
    assert clean["maturity"].max() == pytest.approx(9.65)
    assert len(warnings) == 1
    assert "go up to 9.65, read as years; name the column 'Days' if they are days to expiry" in warnings[0]
    _, labelled = vs.parse_quotes(_days_chain("Maturity (years)"), spot)
    days, named = vs.parse_quotes(_days_chain("Days"), spot)
    assert not any(isinstance(m, vs.QuoteWarning) for m in labelled + named)
    assert days["maturity"].max() == pytest.approx(9.65 / 365)


def test_unlabelled_maturities_in_years_warn_only_beyond_three_years():
    # Noisy or steep smiles must not decide the unit (an earlier check misread dense noisy chains as days).
    steep = vs.SSVIParams(0.3, 0.25, 1.5, -0.9, 1.05, 0.5)  # a steep equity skew at the arbitrage-free limit
    rng = np.random.default_rng(0)
    for params, days, moneyness in (
        (steep, (91, 182, 365, 730, 1095), np.linspace(0.5, 1.5, 21)),
        (vs.SSVIParams(), (7, 30, 91, 182, 365, 730), np.linspace(0.9, 1.1, 41)),
    ):
        quotes = ssvi_quotes(params, 0.04, 0.0, days, moneyness, decimals=4)
        quotes["iv"] += rng.normal(0.0, 0.01, len(quotes))  # 1 vol point of noise
        for spot in (100.0, None):
            _, messages = vs.parse_quotes(quotes.rename(columns={"maturity": "Maturity"}), spot, 0.04)
            assert not any(isinstance(m, vs.QuoteWarning) for m in messages)
    longer = ssvi_quotes(vs.SSVIParams(), 0.04, 0.0, (182, 1825), np.linspace(0.8, 1.2, 9))
    _, messages = vs.parse_quotes(longer.rename(columns={"maturity": "Maturity"}))
    assert any(isinstance(m, vs.QuoteWarning) and "go up to 5" in m for m in messages)


def test_small_whole_days_get_one_units_warning():
    _, messages = vs.parse_quotes(_days_chain("T", days=(1, 2, 3, 4), moneyness=np.linspace(0.9, 1.1, 9)), 100.0)
    warnings = [m for m in messages if isinstance(m, vs.QuoteWarning)]
    assert len(warnings) == 1 and "'T' are whole numbers up to 4, read as years" in warnings[0]


def test_poor_fits_on_quotes_with_strike_arbitrage_point_at_the_maturity_unit():
    # The poor-fit note blamed noise or a shape SVI cannot follow; as years these quotes price a put below a
    # put at a lower strike, which no smile can follow, and the likely cause is days read as years.
    quotes, _ = vs.parse_quotes(_days_chain("Maturity"), 100.0)
    fitted = vs.fit_market_surface(quotes, 100.0, 0.0, 0.0)
    notes = [note for note in fitted.notes if "SVI smile" in note]
    assert len(notes) == 1 and "maturities 2.65y, 4.65y, 9.65y, where no arbitrage-free smile can follow" in notes[0]
    assert "a put costs less than one at a lower strike" in notes[0] and "the maturity unit" in notes[0]
    assert "noisy" not in notes[0]
    as_days = vs.fit_market_surface(vs.parse_quotes(_days_chain("Days"), 100.0)[0], 100.0, 0.0, 0.0)
    assert not any("SVI smile" in note for note in as_days.notes)


def _price_chain(header: str, rate: float, days=(2.65, 4.65, 9.65)) -> pd.DataFrame:
    """Out-of-the-money SSVI option prices for ``days`` calendar days, with the days in a column ``header``."""
    rows = []
    for row in ssvi_quotes(vs.SSVIParams(), rate, 0.0, days, NEAR_THE_MONEY).itertuples():
        kind = "call" if row.strike >= 100 * np.exp(rate * row.maturity) else "put"
        price = bs.price(100.0, row.strike, row.maturity, rate, row.iv, 0.0, kind)
        in_days = round(row.maturity * 365, 6)
        rows.append({"Strike": round(row.strike, 2), header: in_days, "Type": kind, "Price": round(float(price), 4)})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("rate", [0.0, 0.05])
def test_fractional_days_of_option_prices_get_a_units_warning(rate):
    _, messages = vs.parse_quotes(_price_chain("Maturity", rate), 100.0, rate)
    warnings = [m for m in messages if isinstance(m, vs.QuoteWarning)]
    assert len(warnings) == 1 and "go up to 9.65, read as years; name the column 'Days'" in warnings[0]
    days, named = vs.parse_quotes(_price_chain("Days", rate), 100.0, rate)
    assert not any(isinstance(m, vs.QuoteWarning) for m in named)
    assert days["iv"].median() == pytest.approx(0.25, abs=0.01)
    years = vs.sample_quotes(100, rate, 0.0, prices=True)
    years["Maturity"] = years.pop("Days") / 365
    _, quiet = vs.parse_quotes(years, 100.0, rate)
    assert not any(isinstance(m, vs.QuoteWarning) for m in quiet)


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
    assert any(isinstance(m, vs.QuoteWarning) and "whole numbers up to 2, read as years" in m for m in messages)


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


def _stacked_calls_and_puts(itm_vol) -> pd.DataFrame:
    """Calls and puts at every strike with a Type column: exact SSVI vols out of the money, ``itm_vol(iv)`` in it."""
    params, r = vs.SSVIParams(), 0.05
    rows = []
    for days in (16, 30, 44, 79, 107, 170):
        t = days / 365
        for strike in np.arange(70, 130.01, 2.5):
            iv = float(np.sqrt(params.total_variance(np.log(strike / 100) - r * t, t)[0] / t))
            for kind in ("Call", "Put"):
                otm = (kind == "Call") == (strike >= 100 * np.exp(r * t))
                rows.append(
                    {"Type": kind, "Strike": strike, "Days": days, "impliedVolatility": iv if otm else itm_vol(iv)}
                )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("itm_vol", [lambda iv: 1e-05, lambda iv: iv + 0.02], ids=["placeholder", "two-points-high"])
def test_an_implied_vol_file_with_a_type_column_keeps_the_out_of_the_money_side(itm_vol):
    # The Type column was ignored for implied vols: a call and a put at one strike were averaged as duplicates, so
    # placeholder in-the-money vols (1e-05) put the fit 32 vol points off, and in-the-money vols 2 points rich
    # (an American put read as European) moved every maturity's ATM vol 0.4 points with no note.
    clean, messages = vs.parse_quotes(_stacked_calls_and_puts(itm_vol), 100.0, 0.05)
    assert "call/put from 'Type'" in messages[0]
    assert "Used the out-of-the-money option at 150 strike(s) quoted for both calls and puts." in messages
    assert not any("duplicate" in m for m in messages)
    fitted = vs.fit_market_surface(clean, 100.0, 0.05, 0.0)
    true = vs.ssvi_surface(vs.SSVIParams(), 0.05, 0.0, fitted.moneyness, fitted.maturities)
    assert np.nanmax(np.abs(fitted.implied_vol - true.implied_vol)) < 1e-6


def test_an_implied_vol_file_keeps_rows_whose_type_is_blank():
    frame = _stacked_calls_and_puts(lambda iv: iv)
    frame.loc[frame.index % 7 == 0, "Type"] = None
    clean, _ = vs.parse_quotes(frame, 100.0, 0.05)
    assert len(clean) == 6 * 25  # one vol per expiry and strike: blank-type rows kept and averaged in
    no_spot, messages = vs.parse_quotes(frame)  # no forward to pick a side with: averaged, as before
    assert np.allclose(no_spot["iv"], clean["iv"]) and any("Averaged" in m for m in messages)


def _bid_ask_chain(**extra) -> pd.DataFrame:
    rows = []
    for days in (30, 60, 90, 180):
        t = days / 365
        for strike in range(80, 121, 5):
            kind = "call" if strike >= 100 else "put"
            value = bs.price(100.0, strike, t, 0.03, 0.25, 0.0, kind)
            rows.append({"Strike": strike, "DTE": days, "Type": kind, "Bid": value - 0.01, "Ask": value + 0.01})
    return pd.DataFrame(rows).assign(**extra)


@pytest.mark.parametrize("column", ["Mid", "Mark", "Price"])
@pytest.mark.parametrize("blank", [np.nan, "-"])
def test_an_empty_price_column_falls_back_to_bid_and_ask(column, blank):
    # An empty Mid next to valid Bid and Ask raised "column 'Mid' has no numeric values".
    expected, _ = vs.parse_quotes(_bid_ask_chain(), 100.0, 0.03)
    clean, messages = vs.parse_quotes(_bid_ask_chain(**{column: blank}), 100.0, 0.03)
    pd.testing.assert_frame_equal(clean, expected)
    assert f"option price as the bid/ask midpoint of 'Bid' and 'Ask' where '{column}' is blank" in messages[0]
    assert any("zero bid" in m for m in messages) and not any("missing, non-numeric" in m for m in messages)


def test_a_partly_blank_price_column_is_filled_from_bid_and_ask():
    chain = _bid_ask_chain()
    chain = chain[chain["Bid"] > 0].reset_index(drop=True)
    mid = ((chain["Bid"] + chain["Ask"]) / 2).where(chain.index % 2 == 0)
    clean, messages = vs.parse_quotes(chain.assign(Mid=mid), 100.0, 0.03)
    assert len(clean) == len(chain)  # the rows with a blank Mid used to be dropped
    assert "option price from 'Mid'" in messages[0] and "where 'Mid' is blank" in messages[0]
    assert np.allclose(clean["iv"], 0.25, atol=1e-3)


def test_an_empty_price_column_without_bid_and_ask_is_refused():
    chain = _bid_ask_chain(Mid=np.nan).drop(columns=["Bid", "Ask"])
    with pytest.raises(vs.QuoteParseError, match="no numeric option price in 'Mid'"):
        vs.parse_quotes(chain, 100.0, 0.03)
    last = vs.parse_quotes(chain.assign(Last=_bid_ask_chain()["Bid"] + 0.01), 100.0, 0.03)[1][0]
    assert "option price from 'Last' (last trade) where no other price is given" in last


def test_prices_need_a_type_column_and_market_inputs():
    frame = vs.sample_quotes(prices=True)
    with pytest.raises(vs.QuoteParseError, match="Type column"):
        vs.parse_quotes(frame.drop(columns="Type"), 100, 0.04, 0.0)
    with pytest.raises(vs.QuoteParseError, match="spot price"):
        vs.parse_quotes(frame)


@pytest.mark.parametrize(("vol", "T"), [(0.25, 2.65 / 365), (0.10, 0.05)])
def test_flat_smiles_have_no_strike_arbitrage(vol, T):
    # Calls from put-call parity cancelled to rounding noise far out of the money, which read as a call costing
    # more than a call at a lower strike, i.e. strike arbitrage on a perfectly flat smile.
    strikes = 100 * np.linspace(0.8, 1.2, 41)
    assert not vs._spread_arbitrage(strikes, np.full(41, vol), T, 100.0)
    assert vs._spread_arbitrage(strikes, np.clip(0.25 - 2.0 * np.log(strikes / 100), 0.01, None), 2.0, 100.0)


def _spiked_smiles(spike: float) -> pd.DataFrame:
    """Rounded SSVI vols at $1 strikes, 14-91 days, with the 30-day at-the-money vol ``spike`` too high."""
    params, r = vs.SSVIParams(), 0.05
    rows = []
    for days in (14, 30, 60, 91):
        t = days / 365
        for strike in range(90, 111):
            iv = round(float(np.sqrt(params.total_variance(np.log(strike / 100) - r * t, t)[0] / t)), 4)
            iv += spike if (days, strike) == (30, 100) else 0.0
            rows.append({"strike": float(strike), "maturity": t, "iv": iv})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("spike", [0.02, 0.03])
def test_butterfly_arbitrage_in_the_quotes_is_reported(spike):
    # The SVI fit smooths a rich at-the-money quote away, so the surface showed 0 butterfly-arbitrage points and
    # no note although the quoted 99/100/101 call butterfly costs less than nothing (-0.38 at 2 vol points).
    fitted = vs.fit_market_surface(_spiked_smiles(spike), 100.0, 0.05, 0.0)
    assert fitted.butterfly_violations == 0  # the fitted surface is still arbitrage-free ...
    notes = [note for note in fitted.notes if "static arbitrage" in note]
    assert len(notes) == 1  # ... and the quotes' arbitrage is reported
    assert "at maturities 0.0822y a butterfly of neighbouring strikes costs less than nothing" in notes[0]
    assert "e.g. 99/100/101 at 0.0822y" in notes[0] and "describe the fitted surface, not the quotes" in notes[0]
    assert "0.5 vol point off" in notes[0]


def test_quote_noise_within_half_a_vol_point_is_not_butterfly_arbitrage():
    def flagged(quotes) -> bool:
        return any("static arbitrage" in note for note in vs.fit_market_surface(quotes, 100.0, 0.05, 0.0).notes)

    assert not flagged(_spiked_smiles(0.0))
    for seed in range(4):  # the samples carry 0.2 vol points of noise, and some of their butterflies cost < 0
        for prices in (False, True):
            assert not flagged(
                vs.parse_quotes(vs.sample_quotes(100, 0.05, 0.0, seed=seed, prices=prices), 100, 0.05)[0]
            )
    rng = np.random.default_rng(7)
    assert not flagged(_spiked_smiles(0.0).assign(iv=lambda q: q["iv"] + rng.normal(0, 0.002, len(q))))


def test_butterfly_check_is_model_free_and_tolerates_repeated_strikes():
    strikes = np.arange(90.0, 111.0)
    flat = np.full(strikes.size, 0.2)
    assert vs._butterfly_arbitrage(strikes, flat, 30 / 365, 100.0) == (0, None)
    spiked = flat.copy()
    spiked[10] += 0.02  # 2 vol points rich at 100
    assert vs._butterfly_arbitrage(strikes, spiked, 30 / 365, 100.0) == (1, (99.0, 100.0, 101.0))
    assert vs._butterfly_arbitrage(np.r_[strikes, 100.0], np.r_[flat, 0.2], 30 / 365, 100.0) == (0, None)


def _tick_quoted_chain(
    tick: float = 0.05, half_spread=lambda price, tick: max(tick, 0.03 * price), spike=0.0, off_centre=0.0
):
    """Out-of-the-money bids and asks around Black-Scholes prices of SSVI vols, rounded outwards to ``tick``.

    Spot 100, rate 5%, strikes 70-130 in steps of 1, 7-91 days; arbitrage-free unless ``spike`` (vol points)
    is added to the 30-day at-the-money vol. Each spread is centred on the price, or with ``off_centre`` on a
    random point up to that fraction of the half-spread from it, and always brackets the price.
    """
    params, r = vs.SSVIParams(), 0.05
    rng = np.random.default_rng(0)
    rows = []
    for days in (7, 14, 30, 60, 91):
        t = days / 365
        for strike in range(70, 131):
            iv = np.sqrt(params.total_variance(np.log(strike / 100) - r * t, t)[0] / t)
            iv += spike if (days, strike) == (30, 100) else 0.0
            kind = "call" if strike >= 100 * np.exp(r * t) else "put"
            price = bs.price(100.0, strike, t, r, iv, 0.0, kind)
            half = half_spread(price, tick)
            centre = price + rng.uniform(-off_centre, off_centre) * half
            bid = min(np.floor((centre - half) / tick), np.floor(price / tick)) * tick
            ask = max(np.ceil((centre + half) / tick), np.ceil(price / tick)) * tick
            if bid > 0:
                rows.append({"Strike": strike, "Days": days, "Type": kind, "Bid": round(bid, 2), "Ask": round(ask, 2)})
    return pd.DataFrame(rows)


def _quote_arbitrage(frame: pd.DataFrame) -> list[str]:
    quotes, _ = vs.parse_quotes(frame, 100.0, 0.05)
    return [note for note in vs.fit_market_surface(quotes, 100.0, 0.05, 0.0).notes if "static arbitrage" in note]


def test_tick_rounded_quotes_of_an_arbitrage_free_chain_are_not_static_arbitrage():
    # Midpoints of bids and asks in 0.05 ticks move in steps of 0.025, more than half a vol point of a cheap
    # option's price: the 14-day 87/88/89 put butterfly cost -0.05 at the midpoints, and the arbitrage-free chain
    # was reported as static arbitrage although that butterfly costs +0.25 bought at the ask and sold at the bid.
    chain = _tick_quoted_chain()
    puts = chain[(chain["Days"] == 14) & chain["Strike"].between(87, 89)]
    mid = ((puts["Bid"] + puts["Ask"]) / 2).to_numpy()
    assert mid[0] - 2 * mid[1] + mid[2] == pytest.approx(-0.05)
    assert puts["Ask"].iloc[0] - 2 * puts["Bid"].iloc[1] + puts["Ask"].iloc[2] == pytest.approx(0.25)
    assert _quote_arbitrage(chain) == []
    mids = chain.assign(Mid=(chain["Bid"] + chain["Ask"]) / 2).drop(columns=["Bid", "Ask"])
    assert _quote_arbitrage(mids) == []
    quotes, _ = vs.parse_quotes(chain, 100.0, 0.05)
    assert np.allclose(np.sort(quotes["price_error"]), np.sort((chain["Ask"] - chain["Bid"]) / 2))
    assert np.allclose(vs.parse_quotes(mids, 100.0, 0.05)[0]["price_error"], 0.025)  # the step of the midpoints
    coarse = chain.assign(Last=(chain["Bid"] + chain["Ask"]) / 2).round({"Last": 1}).drop(columns=["Bid", "Ask"])
    assert _quote_arbitrage(coarse) == []  # last trades to the nearest 0.1 may each be 0.1 off


def test_a_butterfly_that_costs_less_than_nothing_at_the_bid_and_ask_is_still_reported():
    def one_tick(price, tick):  # spreads of one or two ticks
        return tick / 2

    assert _quote_arbitrage(_tick_quoted_chain(0.01, one_tick)) == []
    chain = _tick_quoted_chain(0.01, one_tick, spike=0.03)
    calls = chain[(chain["Days"] == 30) & chain["Strike"].between(99, 101)].set_index("Strike")
    assert calls.loc[99, "Ask"] - 2 * calls.loc[100, "Bid"] + calls.loc[101, "Ask"] < -0.2  # a real arbitrage
    (note,) = _quote_arbitrage(chain)
    assert "a butterfly of neighbouring strikes costs less than nothing" in note and "half its bid/ask spread" in note
    assert note.startswith("The quotes have static arbitrage") and "may not be in the market" not in note
    (with_mid,) = _quote_arbitrage(chain.assign(Mid=(chain["Bid"] + chain["Ask"]) / 2))  # the same with a Mid
    assert "e.g. 99/100/101 at 0.0822y" in with_mid


@pytest.mark.parametrize("column", ["Mid", "Mark"])
def test_a_price_column_next_to_the_bid_and_ask_may_be_off_by_half_the_spread(column):
    # A Mid or Mark next to Bid and Ask was allowed only the 0.025 step of its grid, so adding Mid = (Bid + Ask) / 2
    # to an arbitrage-free Bid/Ask file with wide spreads turned no note into 'static arbitrage': the 30-day 96/97/98
    # put butterfly costs -0.40 at the midpoints but +0.40 bought at the ask and sold at the bid.
    chain = _tick_quoted_chain(half_spread=lambda price, tick: max(tick, 0.1 * price), off_centre=0.9)
    puts = chain[(chain["Days"] == 30) & chain["Strike"].between(96, 98)]
    mid = ((puts["Bid"] + puts["Ask"]) / 2).to_numpy()
    assert mid[0] - 2 * mid[1] + mid[2] == pytest.approx(-0.4)
    assert puts["Ask"].iloc[0] - 2 * puts["Bid"].iloc[1] + puts["Ask"].iloc[2] == pytest.approx(0.4)
    assert _quote_arbitrage(chain) == []
    priced = chain.assign(**{column: ((chain["Bid"] + chain["Ask"]) / 2).round(3)})
    assert _quote_arbitrage(priced) == []
    quotes, messages = vs.parse_quotes(priced, 100.0, 0.05)
    assert f"option price from '{column}'" in messages[0]
    assert np.allclose(np.sort(quotes["price_error"]), np.sort((chain["Ask"] - chain["Bid"]) / 2))


def test_a_price_away_from_the_midpoint_may_be_off_by_as_much_as_reaches_its_bid_and_ask():
    frame = pd.DataFrame(
        {
            "Strike": [95.0, 100.0, 105.0, 110.0],
            "Days": 30,
            "Type": ["put", "call", "call", "call"],
            "Bid": [1.10, 2.40, 0.60, np.nan],
            "Ask": [1.30, 2.70, 0.50, np.nan],  # crossed at 105, missing at 110: the price keeps its step
            "Mark": [1.15, 2.55, 0.55, 0.15],
        }
    )
    quotes, _ = vs.parse_quotes(frame, 100.0, 0.05)
    assert np.allclose(quotes["price_error"], [0.15, 0.15, 0.05, 0.05])  # to the ask, half the spread, the step


def _vols_of_tick_rounded_midpoints(spot: float, tick: float, spike: float = 0.0) -> pd.DataFrame:
    """Implied vols (4 dp) of midpoints of bids and asks in ``tick``s around an arbitrage-free SSVI chain.

    Rate 5%, $0.50 strikes from 70% to 130% of ``spot``, 7-91 days; ``spike`` (vol points) is added to the
    30-day vol at the strike nearest the spot.
    """
    params, r = vs.SSVIParams(), 0.05
    rows = []
    for days in (7, 14, 30, 60, 91):
        t = days / 365
        for strike in np.arange(np.ceil(1.4 * spot), np.floor(2.6 * spot) + 1) / 2:
            iv = np.sqrt(params.total_variance(np.log(strike / spot) - r * t, t)[0] / t)
            kind = "call" if strike >= spot * np.exp(r * t) else "put"
            price = bs.price(spot, strike, t, r, iv, 0.0, kind)
            half = max(tick, 0.03 * price)
            bid, ask = np.floor((price - half) / tick) * tick, np.ceil((price + half) / tick) * tick
            if bid > 0:
                vol = implied_volatility((bid + ask) / 2, spot, strike, t, r, 0.0, kind)
                vol += spike if days == 30 and abs(strike - spot) <= 0.25 else 0.0
                rows.append({"Strike": strike, "Days": days, "IV": round(vol, 4)})
    return pd.DataFrame(rows)


def _vol_arbitrage(frame: pd.DataFrame, spot: float) -> list[str]:
    quotes, _ = vs.parse_quotes(frame, spot, 0.05)
    return [note for note in vs.fit_market_surface(quotes, spot, 0.05, 0.0).notes if "static arbitrage" in note]


def test_vols_of_tick_rounded_midpoints_on_a_low_priced_underlying_are_not_static_arbitrage():
    # The unknown price behind a quoted vol was allowed only 0.01% of the forward (0.0037 at spot 37.4), so vols
    # inverted from midpoints of an arbitrage-free chain in 0.05 ticks were reported as static arbitrage: the 30-day
    # 31.5/32/32.5 put butterfly costs -0.05 at the midpoints (+0.004 at the true prices).
    assert _vol_arbitrage(_vols_of_tick_rounded_midpoints(37.4, 0.05), 37.4) == []


def test_arbitrage_in_quoted_vols_within_the_rounding_of_their_prices_is_called_possible():
    # In 0.10 ticks the midpoints' rounding still shows as arbitrage beyond half a cent per price; it was reported
    # as certain although half a 0.05 tick per price removes it.
    (note,) = _vol_arbitrage(_vols_of_tick_rounded_midpoints(37.4, 0.10), 37.4)
    assert note.startswith("The quotes may have static arbitrage that the fitted surface smooths out")
    possible = "arbitrage is small enough that rounding the prices behind the quoted vols to 0.05 ticks could cause it"
    assert f"The {possible}, so it may not be in the market." in note
    assert "the unknown price behind each quoted implied vol to be off by 0.01% of the forward or 0.005" in note
    assert "bid/ask" not in note  # the allowances for prices do not apply to quoted vols
    # A 5 vol-point spike is beyond that rounding: the note says so, and names only the other maturities as possible.
    (note,) = _vol_arbitrage(_vols_of_tick_rounded_midpoints(37.4, 0.10, spike=0.05), 37.4)
    assert note.startswith("The quotes have static arbitrage") and "e.g. 37/37.5/38 at 0.0822y" in note
    assert f"At maturities 0.164y, 0.249y the {possible}" in note


def test_the_price_step_is_the_coarsest_usual_increment_of_every_price():
    assert vs._price_step(pd.Series([0.175, 0.225, 1.05, 12.3])) == 0.025
    assert vs._price_step(pd.Series([0.12, 0.13, np.nan, 0.0, -1.0])) == 0.01
    assert vs._price_step(pd.Series([1.2345, 2.0001])) == 0.0001
    assert vs._price_step(pd.Series([1.234567])) == 0.0
    assert vs._price_step(pd.Series([np.nan])) == 0.0


def test_vertical_spread_arbitrage_in_the_quotes_is_reported_even_when_the_fit_is_close():
    # Spread arbitrage only chose the wording of a poor-fit note, so it went unreported under 1 vol point of RMSE.
    quotes = _spiked_smiles(0.0)
    quotes.loc[(quotes["maturity"] == 30 / 365) & (quotes["strike"] == 110.0), "iv"] += 0.06
    fitted = vs.fit_market_surface(quotes, 100.0, 0.05, 0.0)
    assert fitted.fit_rmse[1] < vs.POOR_FIT_RMSE
    notes = [note for note in fitted.notes if "static arbitrage" in note]
    assert (
        len(notes) == 1 and "a call more than one at a lower strike, even if each quote is 1 vol point off" in notes[0]
    )
