"""The chain read quote by quote and the desk measures of a surface (options_analyzer.vol_surface.diagnostics; roadmap
SF-6 and SF-11, UI spec 3.4).

References are independent of the code under test where one exists: the synthetic generator's own forwards and ATM
vols (its model inputs), parsing the whole chain with ``parse_quotes`` (the fit before this module), SSVI's closed-form
ATM variance, Black-Scholes-Merton deltas from ``black_scholes``, a loop-by-loop transcription of the Cboe VIX white
paper, and the continuous model-free variance of the generator's own smile by quadrature.
"""

from __future__ import annotations

import doctest
import math
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from options_analyzer import black_scholes as bs
from options_analyzer import clock
from options_analyzer import vol_surface as vs
from options_analyzer.lets_be_rational import black
from options_analyzer.market_data import sample_chain, to_quotes
from options_analyzer.market_data.synthetic import DEFAULT_SSVI
from options_analyzer.vol_surface import diagnostics as dg

#: The valuation time the app's tests pin (tests/_apptest.py ASOF) and the index preset's market (ui/market.py).
AS_OF = datetime(2026, 10, 23, 16, 0)
SPOT, RATE, YIELD = 5800.0, 0.04, 0.013


def _fit(noise: bool = True, settings: dg.FitSettings = dg.DEFAULT_SETTINGS, **chain_options):
    chain = sample_chain(SPOT, RATE, YIELD, as_of=AS_OF, noise=noise, **chain_options)
    return chain, dg.fit_chain(chain.quotes, as_of=AS_OF, spot=SPOT, rate=RATE, dividend_yield=YIELD, settings=settings)


@pytest.fixture(scope="module")
def default():
    return _fit()


@pytest.fixture(scope="module")
def clean():
    return _fit(noise=False)


def _reconciles(counts: dict[str, int]) -> bool:
    return counts["quoted"] == counts["used"] + sum(counts[reason] for reason in dg.REASONS)


def test_the_default_screen_counts_reconcile_and_keep_what_the_parser_kept(default):
    chain, fit = default
    counts = fit.counts()
    assert _reconciles(counts)
    # Measured on the default chain (UI spec 3.4's example reads 5,020 → 2,363 with 147 zero bids: the generator
    # has changed since; these are the code's numbers, checked against the parser below).
    assert (counts["quoted"], counts["used"], counts["ITM side"], counts["zero bid"], counts["wide"]) == (
        5012,
        2357,
        2506,
        149,
        0,
    )
    # Every quote is either used or has exactly one reason; the out-of-the-money side of each strike is the one used.
    used = fit.quotes[fit.quotes["status"] == dg.USED]
    otm = (used["type"] == "call") == (used["strike"] >= used["forward"])
    assert otm.all()
    # The surface is the one parsing the whole chain gave before this module (ui/market.py until this change): the
    # same quotes, vols to rounding, forwards and fit.
    clean, _, forwards = vs.parse_quotes(to_quotes(chain.quotes, as_of=AS_OF), SPOT, RATE, YIELD, return_forwards=True)
    before = vs.fit_market_surface(clean, SPOT, RATE, YIELD, forwards=forwards)
    after = fit.surface
    np.testing.assert_array_equal(after.maturities, before.maturities)
    assert len(after.quotes) == len(before.quotes) == counts["used"]
    # LBR's batch inversion against the scalar solver: both converge to machine precision (measured 1.7e-15).
    np.testing.assert_allclose(after.quotes["iv"], before.quotes["iv"], rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.exp(after.log_forward_ratio), np.exp(before.log_forward_ratio), rtol=1e-14)
    # The SVI optimiser sees inputs 1e-15 apart; its fit moves the grid by 8e-11 at most (measured).
    np.testing.assert_allclose(after.implied_vol, before.implied_vol, rtol=0, atol=1e-9)


def test_each_filter_excludes_with_its_own_reason(default):
    _, fit = default
    base = fit.counts()
    checks = {
        "wide": dg.QuoteFilters(max_spread=1.0),
        "thin expiry": dg.QuoteFilters(min_strikes=200),
        "short expiry": dg.QuoteFilters(min_days=10.0),
        "weekly": dg.QuoteFilters(weeklies=False),
    }
    for reason, filters in checks.items():
        settings = dg.FitSettings(filters)
        if reason == "thin expiry":  # leaves too few expiries to fit: the screen itself is what is checked
            with pytest.raises(vs.QuoteParseError):
                _fit(settings=settings)
            continue
        counts = _fit(settings=settings)[1].counts()
        assert _reconciles(counts), reason
        assert counts[reason] > 0 and counts["used"] < base["used"], (reason, counts)
    # 1× the median spread leaves out the quotes wider than their expiry's median: about half (measured 1,146).
    assert _fit(settings=dg.FitSettings(dg.QuoteFilters(max_spread=1.0)))[1].counts()["wide"] == 1146
    # Weeklies off: the four SPXW Fridays go (20 November is the AM monthly), and the twelve monthlies and quarterlies
    # stay.
    weekly = _fit(settings=dg.FitSettings(dg.QuoteFilters(weeklies=False)))[1]
    assert (weekly.expiries["slice"] >= 0).sum() == 12
    assert set(weekly.expiries.loc[weekly.expiries["slice"] < 0, "root"]) == {"SPXW"}


def test_the_in_the_money_side_is_fitted_where_it_has_time_value():
    _, fit = _fit(settings=dg.FitSettings(dg.QuoteFilters(otm_only=False)))
    counts = fit.counts()
    assert _reconciles(counts) and counts["ITM side"] == 0
    # The in-the-money quotes whose time value is within their spread (the other side has a zero bid) carry no vol.
    assert counts["no time value"] > 0 and counts["used"] > 2357


def test_unconstrained_calibration_changes_nothing_on_an_arbitrage_free_chain(default):
    _, fit = default
    _, free = _fit(settings=dg.FitSettings(calibration="unconstrained"))
    # The default chain's arbitrage-free fit never binds (arbitrage cost 0 at every expiry), so both are one fit.
    assert np.all(fit.surface.arbitrage_cost == 0)
    np.testing.assert_allclose(free.surface.implied_vol, fit.surface.implied_vol, rtol=0, atol=1e-12)
    assert dg.FitSettings(calibration="unconstrained").changes() == ("unconstrained calibration (arbitrage allowed)",)


def test_on_the_noise_free_chain_the_table_recovers_the_generators_forwards_and_atm_vols(clean):
    """DT-4 acceptance 3 and SF-3: every ATMF within 0.1 vp of the generator's and every F within 2e-4 relative."""
    chain, fit = clean
    table = dg.expiry_table(fit, rate=RATE + 0.0045, dividend_yield=YIELD)
    truth = chain.expiries
    assert len(table) == len(truth) == 16
    np.testing.assert_allclose(table["forward"], truth["forward"], rtol=2e-4)
    assert np.max(np.abs(table["atmf"].to_numpy() - truth["atm_vol"].to_numpy())) < 0.001
    # Measured: 4.6e-6 relative in F and 0.0012 vp in ATMF, far inside the plan's bounds.
    assert np.max(np.abs(table["forward"] / truth["forward"] - 1)) < 1e-5
    # The box rate is the reference carry: no carry flag, and gaps within the quotes' precision.
    assert not table["flags"].str.contains("carry").any()
    assert np.max(np.abs(table["carry_gap_bp"])) < 1.0


def test_the_expiry_table_reads_the_fit(default):
    _, fit = default
    table = dg.expiry_table(fit, rate=RATE + 0.0045, dividend_yield=YIELD)
    assert list(table["used"]) == list(fit.expiries["used"])
    assert table["used"].sum() == fit.counts()["used"]
    assert table["in_band"].between(0, 1).all() and table["confidence"].between(0, 1).all()
    # 25-delta RR is negative (put skew) and the fitted RMSE is inside the half-spread at every expiry.
    assert (table["rr25"] < 0).all() and (table["rmse"] < 0.001).all()


def test_atmf_vol_is_ssvi_closed_form_and_interpolates_total_variance():
    params = vs.SSVIParams()
    surface = vs.ssvi_surface(params, 0.05, 0.01, np.linspace(0.7, 1.3, 41), np.array([7, 30, 365]) / 365)
    for days in (7, 30, 100, 365):
        T = days / 365
        assert dg.atmf_vol(surface, T) == pytest.approx(math.sqrt(float(params.theta(T)) / T), rel=1e-14)
    assert math.isnan(dg.atmf_vol(surface, 400 / 365))  # beyond the maturities: no extrapolation in time


def test_the_delta_strike_has_its_delta_by_black_scholes(default):
    """The strike delta_vol returns, priced by Black-Scholes-Merton at its own vol with the carry of the expiry's F
    and DF, has the target delta to 1e-9 (the root is solved to 1e-12 in log-moneyness)."""
    _, fit = default
    surface = fit.surface
    j = 3  # 20-Nov-26 AM
    T = float(surface.maturities[j])
    F = float(np.exp(surface.log_forward[j]))
    DF = float(surface.discount_factor[j])
    r = -math.log(DF) / T
    q = r - math.log(F / SPOT) / T
    for target in (0.25, -0.25, 0.10, -0.10, 0.50):
        sigma, y = dg.delta_vol(surface, T, target)
        K = F * math.exp(y)
        assert sigma == pytest.approx(surface.implied_vol_at(K / SPOT, T, extrapolate=True), rel=1e-12)
        kind = "call" if target > 0 else "put"
        delta = bs.greeks(SPOT, K, T, r, sigma, q, kind)["delta"]
        assert delta == pytest.approx(target, abs=1e-9), target


def test_the_30_day_tiles_match_the_code_and_where_they_differ_from_the_spec(default):
    """UI spec 3.4's example tiles, measured on today's code: ATMF 30D 15.38% (spec 15.38%), 25Δ RR −4.20 vp (spec
    −4.21, before SF-4's spread-weighted fit), BF +0.23 vp (+0.23), TERM +1.62 vp (+1.62), VIX-style 16.42% (16.43%)."""
    _, fit = default
    surface = fit.surface
    T30 = 30 / 365
    atmf = dg.atmf_vol(surface, T30)
    rr, bf = dg.risk_reversal(surface, T30)
    vix, pair = dg.vix_style_vol(fit, 30)
    assert round(atmf * 100, 2) == 15.38
    assert round(rr * 100, 2) == -4.20 and round(bf * 100, 2) == 0.23
    assert round((dg.atmf_vol(surface, 1.0) - atmf) * 100, 2) == 1.62
    assert round(vix * 100, 2) == 16.42 and pair == (3, 4)  # 20-Nov-26 AM (27.7 d) and 27-Nov-26 PM (34.9 d)
    # ATMF 30D is total variance linear in time between the two expiries around 30 days.
    T1, T2 = surface.maturities[3], surface.maturities[4]
    w1, w2 = (float(surface.svi[k].total_variance(0.0)[0]) for k in (3, 4))
    assert atmf == pytest.approx(math.sqrt((w1 + (w2 - w1) * (T30 - T1) / (T2 - T1)) / T30), rel=1e-14)


def test_forward_vols_rebuild_the_term_structure(default):
    _, fit = default
    surface = fit.surface
    for start, end, sigma in dg.forward_vols(surface):
        if math.isnan(sigma):
            assert end == 730  # beyond the last expiry (692 days)
            continue
        w1, w2 = (dg.atmf_vol(surface, d / 365) ** 2 * d / 365 for d in (start, end))
        assert w1 + sigma**2 * (end - start) / 365 == pytest.approx(w2, rel=1e-12)


def _white_paper_variance(quotes: pd.DataFrame, T: float, R: float) -> float:
    """The Cboe VIX white paper's steps 1 and 2 for one expiry, transcribed loop by loop (independent of the module)."""
    calls = {row.strike: row for row in quotes[quotes["type"] == "call"].itertuples()}
    puts = {row.strike: row for row in quotes[quotes["type"] == "put"].itertuples()}
    strikes = sorted(set(calls) & set(puts))
    best, k_star = None, None
    for k in strikes:
        c, p = calls[k], puts[k]
        if c.bid > 0 and p.bid > 0:
            diff = abs((c.bid + c.ask) / 2 - (p.bid + p.ask) / 2)
            if best is None or diff < best:
                best, k_star = diff, k
    c, p = calls[k_star], puts[k_star]
    F = k_star + math.exp(R * T) * ((c.bid + c.ask) / 2 - (p.bid + p.ask) / 2)
    k0 = max(k for k in strikes if k <= F)
    selected = {k0: ((calls[k0].bid + calls[k0].ask) / 2 + (puts[k0].bid + puts[k0].ask) / 2) / 2}
    zeros = 0
    for k in reversed([k for k in strikes if k < k0]):
        if puts[k].bid == 0:
            zeros += 1
            if zeros == 2:
                break
            continue
        zeros = 0
        selected[k] = (puts[k].bid + puts[k].ask) / 2
    zeros = 0
    for k in [k for k in strikes if k > k0]:
        if calls[k].bid == 0:
            zeros += 1
            if zeros == 2:
                break
            continue
        zeros = 0
        selected[k] = (calls[k].bid + calls[k].ask) / 2
    ks = sorted(selected)
    total = 0.0
    for n, k in enumerate(ks):
        if n == 0:
            dk = ks[1] - ks[0]
        elif n == len(ks) - 1:
            dk = ks[-1] - ks[-2]
        else:
            dk = (ks[n + 1] - ks[n - 1]) / 2
        total += dk / k**2 * math.exp(R * T) * selected[k]
    return 2 / T * total - (F / k0 - 1) ** 2 / T


def test_the_vix_style_vol_is_the_white_papers_formula(default):
    _, fit = default
    vix, (i, j) = dg.vix_style_vol(fit, 30)
    variances = []
    for label in (i, j):
        row = fit.expiries.iloc[label]
        quotes = fit.quotes[fit.quotes["label"] == label]
        T = float(row["maturity"])
        variances.append(_white_paper_variance(quotes, T, -math.log(float(row["discount_factor"])) / T))
    T1, T2 = (float(fit.expiries.iloc[k]["maturity"]) for k in (i, j))
    N1, N2, N30, N365 = T1 * 525_600, T2 * 525_600, 30 * 1440, 365 * 1440  # Cboe's minutes
    sigma2 = (T1 * variances[0] * (N2 - N30) / (N2 - N1) + T2 * variances[1] * (N30 - N1) / (N2 - N1)) * N365 / N30
    assert vix == pytest.approx(math.sqrt(sigma2), rel=1e-12)


@pytest.mark.oracle
@pytest.mark.parametrize(("noise", "tolerance"), [(False, 1e-4), (True, 3e-4)])
def test_the_vix_style_vol_is_the_model_free_variance_of_the_generators_smile(noise, tolerance):
    """Against the continuous model-free variance 2∫Q(K)/K² dK of the generator's own SSVI smile (quadrature of
    undiscounted Black prices) at the two expiries, interpolated in total variance to 30 days. Measured: 0.0030 vp
    below it on the noise-free chain (the strike grid and the wings cut at two zero bids) and 0.0095 vp with the
    quotes' noise; the tolerances are about three times that (0.01 vp and 0.03 vp)."""
    chain, fit = _fit(noise=noise)
    vix, (i, j) = dg.vix_style_vol(fit, 30)
    totals = []
    for label in (i, j):
        row = chain.expiries.iloc[label]
        F, T, tau = float(row["forward"]), float(row["maturity"]), float(row["variance_time"])

        def otm(K, F=F, T=T, tau=tau):
            w = float(DEFAULT_SSVI.total_variance(np.array([math.log(K / F)]), tau)[0][0])
            return black(F, K, math.sqrt(w / T), T, 1.0 if K >= F else -1.0) / K**2

        totals.append(2 * (quad(otm, 0.2 * F, F, limit=400)[0] + quad(otm, F, 3 * F, limit=400)[0]))
    T1, T2, target = float(chain.expiries.iloc[i]["maturity"]), float(chain.expiries.iloc[j]["maturity"]), 30 / 365
    w = totals[0] + (totals[1] - totals[0]) * (target - T1) / (T2 - T1)
    assert abs(vix - math.sqrt(w / target)) < tolerance


@pytest.mark.parametrize("noise", [False, True])
def test_on_a_flat_smile_the_vix_style_vol_is_the_atmf_vol(noise):
    """UI-26's acceptance: on a chain with no smile (SSVI with ρ = 0 and η → 0, in calendar time) the model-free
    30-day vol equals ATMF 30D to 0.1 vp (the plan's bound). Measured: 20.014% against 19.999% without noise, 20.011%
    against 19.998% with it, so 0.014 and 0.013 vp: the strike grid's discretisation, the same sign as on the skewed
    chain."""
    flat = vs.SSVIParams(0.20, 0.20, 1.5, 0.0, 1e-6, 0.5)
    _, fit = _fit(noise=noise, ssvi=flat, variance_clock=clock.VarianceClock())
    vix, _ = dg.vix_style_vol(fit, 30)
    atmf = dg.atmf_vol(fit.surface, 30 / 365)
    assert atmf == pytest.approx(0.20, abs=2e-5)
    assert abs(vix - atmf) < 0.001


def test_weeklies_and_the_implied_spot(default):
    assert not dg.is_weekly(date(2026, 11, 20)) and dg.is_weekly(date(2026, 11, 27))
    # Juneteenth 2027 is observed on Friday 18 June, so that month's standard expiry is Thursday the 17th.
    assert not dg.is_weekly(date(2027, 6, 17)) and dg.is_weekly(date(2027, 6, 18))
    chain, _ = default
    front = chain.expiries.iloc[0]
    # F·DF of the front expiry is S·e^(−qT): 5,798.55 here; parity reads it within the quotes' noise (0.03%).
    expected = SPOT * math.exp(-YIELD * float(front["maturity"]))
    assert dg.implied_spot(chain.quotes, as_of=AS_OF) == pytest.approx(expected, rel=3e-4)


def test_the_examples_in_the_docstrings_run():
    result = doctest.testmod(dg, optionflags=doctest.ELLIPSIS)
    assert result.attempted >= 6 and not result.failed
