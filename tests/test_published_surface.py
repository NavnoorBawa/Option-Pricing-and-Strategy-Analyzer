"""The published surface (options_analyzer.vol_surface.published, SF-11 core and RK-9a; UI spec 2.0, 2.7 item 6, 3.4).

References are independent of the code under test where one exists: the synthetic generator's own Black-76 prices
on its SSVI surface (the chain's true values), Black-Scholes-Merton deltas from ``black_scholes.greeks``, and the
martingale and total-probability identities that any risk-neutral density must meet.
"""

from __future__ import annotations

import math
import time
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import trapezoid
from scipy.special import ndtr

from options_analyzer import black_scholes as bs
from options_analyzer import vol_surface as vs
from options_analyzer.lets_be_rational import black
from options_analyzer.market_data import sample_chain, to_quotes
from options_analyzer.market_data.synthetic import DEFAULT_SSVI, SYNTHETIC_LABEL
from options_analyzer.vol_surface.published import ONE_SD, TWO_SD, PublishedSurface

#: The valuation time the app's tests pin (tests/_apptest.py ASOF) and the index preset's market (ui/market.py).
ASOF = datetime(2026, 10, 23, 16, 0)
SPOT, RATE, YIELD = 5800.0, 0.04, 0.013
DEC = date(2026, 12, 18)  # the Strategy tab's default expiry (UI spec 3.2): the AM monthly nearest 45 days


def _publish(noise: bool):
    """The chain, read and fitted as ui/market.py reads the index preset's, and its published surface."""
    chain = sample_chain(SPOT, RATE, YIELD, as_of=ASOF, noise=noise)
    quotes = to_quotes(chain.quotes, as_of=ASOF)
    clean, _, forwards = vs.parse_quotes(quotes, SPOT, RATE, YIELD, return_forwards=True)
    surface = vs.fit_market_surface(clean, SPOT, RATE, YIELD, forwards=forwards)
    expiries = chain.expiries[["expiry", "root", "settlement", "settles_at", "maturity"]].copy()
    index = [int(np.argmin(np.abs(surface.maturities - T))) for T in expiries["maturity"]]
    expiries["forward"] = np.exp(np.asarray(surface.log_forward)[index])
    expiries["discount_factor"] = np.asarray(surface.discount_factor)[index]
    published = PublishedSurface.build(surface, expiries, chain.quotes, spot=SPOT, as_of=ASOF, source=chain.label)
    return chain, published


@pytest.fixture(scope="module")
def clean():
    return _publish(noise=False)


@pytest.fixture(scope="module")
def default():
    return _publish(noise=True)


def _days(published: PublishedSurface) -> list[date]:
    return list(published.expiries["expiry"])


# --- the risk-neutral CDF --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["clean", "default"])
def test_the_cdf_is_a_distribution_function_on_every_expiry(which, request):
    _, published = request.getfixturevalue(which)
    for day in _days(published):
        F = published.forward(day)
        K = np.linspace(0.3 * F, 3.0 * F, 8001)  # far beyond every quoted wing (the widest reaches |ln K/F| ~ 1)
        cdf = published.cdf(K, day)
        assert np.all((cdf >= 0) & (cdf <= 1)), day
        assert np.all(np.diff(cdf) >= 0), day  # exactly, with no tolerance: see _distribution's survival form
        assert published.cdf(0.0, day) == 0.0 and published.cdf(-1.0, day) == 0.0
        assert np.all(published.pdf(K, day) >= 0), day


def test_the_cdf_matches_the_generators_own_prices(clean):
    # On the noise-free chain the true call prices are the generator's Black-76 values on its SSVI surface in
    # variance time. Their strike derivative (a central difference of 0.01 point: truncation h²C'''/6 ~ 1e-10, rounding
    # 1e-16 · C / h ~ 1e-11, both far below the tolerance) gives the true CDF, 1 + dC/dK / DF. The task's tolerance
    # is 0.5 percentage point; measured, the largest error over every listed strike of all 16 expiries is 0.018 pp,
    # what is left of fitting SVI to SSVI through quotes rounded to the tick.
    chain, published = clean
    worst = 0.0
    for _, row in chain.expiries.iterrows():
        day = row["expiry"].date()
        F, DF, tau = float(row["forward"]), float(row["discount_factor"]), float(row["variance_time"])

        def call(K, F=F, DF=DF, tau=tau):
            w = DEFAULT_SSVI.total_variance(np.log(K / F), tau)[0]
            d1 = -np.log(K / F) / np.sqrt(w) + 0.5 * np.sqrt(w)
            return DF * (F * ndtr(d1) - K * ndtr(d1 - np.sqrt(w)))

        K = published.strikes(day)
        h = 0.01
        true = 1.0 + (call(K + h) - call(K - h)) / (2 * h) / DF
        worst = max(worst, float(np.max(np.abs(published.cdf(K, day) - true))))
    assert worst < 0.005
    assert worst < 0.0005  # the measured 0.018 pp with room: a regression to a flat-vol CDF would miss by > 4 pp


def test_the_smile_moves_the_tail_probability_from_the_flat_vol_one(default):
    # UI spec 2.7 item 6: N(−d₂) at the 16Δ put's own vol gives 16.85% below it on the default chain, the smile
    # 11.91%. Measured here: 16.97% and 11.96% (the spec's figures came from an earlier fit of the same chain).
    _, published = default
    put = published.delta_strike(DEC, -0.16, "put")
    sigma, T, F = published.vol(put, DEC).vol, published.maturity(DEC), published.forward(DEC)
    flat = float(ndtr(-(math.log(F / put) - 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))))
    smile = published.cdf(put, DEC)
    assert flat == pytest.approx(0.1685, abs=0.005)
    assert smile == pytest.approx(0.1191, abs=0.005)


@pytest.mark.parametrize("which", ["clean", "default"])
def test_the_forward_is_the_mean_and_the_density_integrates_to_the_cdf(which, request):
    # Any risk-neutral distribution has the forward as its mean: E[S_T] = ∫₀^∞ (1 − CDF(K)) dK = F. The integral runs
    # on 400,001 points of ln K/F from −25 to +4 (the CDF is 0 and 1 to double precision at the ends), where the
    # trapezoid rule's error is below 1e-9 F; measured, every expiry gives F to 9e-10. Tolerance 1e-8 relative.
    # The analytic density integrates to the CDF's increase over ±14 ATM standard deviations to 1e-8.
    _, published = request.getfixturevalue(which)
    for day in _days(published):
        F = published.forward(day)
        K = F * np.exp(np.linspace(-25.0, 4.0, 400_001))
        cdf = published.cdf(K, day)
        assert cdf[0] < 1e-25 and cdf[-1] == 1.0
        assert trapezoid(1.0 - cdf, K) + K[0] == pytest.approx(F, rel=1e-8), day
        sd = math.sqrt(published.maturity(day)) * published.vol(F, day).vol
        K = F * np.exp(np.linspace(-14 * sd, 14 * sd, 40_001))
        increase = published.cdf(K[-1], day) - published.cdf(K[0], day)
        assert trapezoid(published.pdf(K, day), K) == pytest.approx(increase, abs=1e-8), day


def test_the_density_is_the_derivative_of_the_cdf(default):
    _, published = default
    for day in _days(published)[::3]:
        F = published.forward(day)
        K = np.linspace(0.8 * F, 1.15 * F, 41)
        h = 1e-5 * F  # truncation (h/sd)² ~ 1e-6 relative on the front expiry; rounding 1e-16 / h ~ 1e-15
        numeric = (published.cdf(K + h, day) - published.cdf(K - h, day)) / (2 * h)
        np.testing.assert_allclose(published.pdf(K, day), numeric, rtol=1e-5, atol=1e-12)


def test_the_cdf_uses_the_surfaces_own_vols(default):
    # The CDF's smile and the published vol are the same: sqrt(w/T) of the slice equals VolSurface.implied_vol_at
    # inside the reach. 1e-8 relative: the fit's maturities (from the quotes) and the chain's agree to about 1e-10.
    _, published = default
    for day in _days(published):
        F, T = published.forward(day), published.maturity(day)
        for k in np.linspace(-0.2, 0.1, 13):
            sigma = published.vol(F * math.exp(k), day).vol
            if np.isfinite(sigma):
                w = published._variance(day, np.array([k]))[0][0]
                assert math.sqrt(w / T) == pytest.approx(sigma, rel=1e-8)


# --- quantiles and ranges ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["clean", "default"])
def test_the_quantiles_invert_the_cdf(which, request):
    _, published = request.getfixturevalue(which)
    p = np.array([1e-6, 0.001, (1 - TWO_SD) / 2, (1 - ONE_SD) / 2, 0.5, (1 + ONE_SD) / 2, (1 + TWO_SD) / 2, 0.999])
    for day in _days(published):
        K = published.quantile(p, day)
        np.testing.assert_allclose(published.cdf(K, day), p, rtol=0, atol=1e-14)
        strikes = published.strikes(day)[::10]
        np.testing.assert_allclose(published.quantile(published.cdf(strikes, day), day), strikes, rtol=1e-9)
    assert np.isnan(published.quantile(0.0, DEC)) and np.isnan(published.quantile(1.0, DEC))


def test_the_risk_neutral_ranges(default):
    _, published = default
    one, two = published.rn_range(DEC, ONE_SD), published.rn_range(DEC, TWO_SD)
    F = published.forward(DEC)
    assert two[0] < one[0] < F < one[1] < two[1]
    # UI spec 2.7 item 6 reads 5,492–6,167 for the 68% range to 18-Dec on the default chain: within 10 points.
    assert one == pytest.approx((5492.0, 6167.0), abs=10.0)
    # Skew: the range reaches further below the forward than above it.
    assert F - two[0] > two[1] - F


def test_one_cdf_evaluation_and_the_ranges_are_fast(default):
    # Loose guards (shared CI runners are 2-3 times slower): measured on a shared 4-core machine, 15 µs per CDF
    # evaluation (43 µs for 301 strikes), and 0.4 ms for the 68% and 95% ranges together once the expiry's grid is
    # built (0.75 ms with it).
    _, published = default
    published.cdf(5800.0, DEC)
    start = time.perf_counter()
    for _ in range(100):
        published.cdf(5800.0, DEC)
    assert (time.perf_counter() - start) / 100 < 2e-3
    start = time.perf_counter()
    published.rn_range(DEC, ONE_SD), published.rn_range(DEC, TWO_SD)
    assert time.perf_counter() - start < 0.05


# --- strikes by delta (RK-9a) -----------------------------------------------------------------------------------------


def _bsm_delta(published: PublishedSurface, day: date, K: float, kind: str) -> float:
    """Spot delta from black_scholes at the expiry's carry (r from DF, q from F) and the strike's surface vol."""
    T, F, DF = published.maturity(day), published.forward(day), published.discount_factor(day)
    r = -math.log(DF) / T
    q = r - math.log(F / SPOT) / T
    return float(bs.greeks(SPOT, K, T, r, published.vol(K, day).vol, q, kind)["delta"])


@pytest.mark.parametrize(("target", "kind"), [(-0.16, "put"), (0.16, "call"), (-0.25, "put"), (0.10, "call")])
def test_the_delta_strike_is_the_listed_strike_nearest_the_target(default, target, kind):
    _, published = default
    for day in (_days(published)[1], DEC, _days(published)[8]):
        K = published.delta_strike(day, target, kind)
        listed = published.strikes(day)
        assert K in listed
        deltas = {k: _bsm_delta(published, day, k, kind) for k in listed if np.isfinite(published.vol(k, day).vol)}
        best = min(abs(d - target) for d in deltas.values())
        assert abs(deltas[K] - target) == pytest.approx(best, abs=1e-12)
        assert published.delta(K, day, kind) == pytest.approx(deltas[K], abs=1e-12)


def test_the_default_condor_shorts(default):
    # UI spec 3.2, RK-9a acceptance: the 16Δ shorts of the default iron condor on 18-Dec are 5,400 and 6,150.
    _, published = default
    assert published.delta_strike(DEC, -0.16, "put") == 5400.0
    assert published.delta_strike(DEC, 0.16, "call") == 6150.0


def test_the_delta_strike_refuses_a_target_of_the_wrong_sign(default):
    _, published = default
    with pytest.raises(ValueError, match="put delta"):
        published.delta_strike(DEC, 0.16, "put")
    with pytest.raises(ValueError, match="call delta"):
        published.delta_strike(DEC, -0.16, "call")


# --- quotes -----------------------------------------------------------------------------------------------------------


def test_quotes_round_trip_against_the_chain(default):
    chain, published = default
    table = chain.quotes
    for row in table.sample(400, random_state=1).itertuples():
        day = pd.Timestamp(row.expiry).date()
        quote = published.quote(day, row.strike, row.type)
        assert (quote.bid, quote.ask) == (row.bid, row.ask)
        assert quote.mid == pytest.approx((row.bid + row.ask) / 2, abs=1e-12)
        F, DF, T = published.forward(day), published.discount_factor(day), published.maturity(day)
        theta = 1.0 if row.type == "call" else -1.0
        for price, iv in ((quote.ask, quote.iv_ask), (quote.mid, quote.iv_mid), (quote.bid, quote.iv_bid)):
            if np.isfinite(iv):  # the vol re-prices the quote (Black-76 at the expiry's F and DF)
                assert DF * black(F, row.strike, iv, T, theta) == pytest.approx(price, rel=1e-9, abs=1e-9)
        if row.bid == 0:
            assert math.isnan(quote.iv_bid) and math.isnan(quote.iv_mid)
    assert len(published.quotes) == len(table)  # every expiry of the chain is listed


def test_listed_strikes_and_expiry_rows(default):
    chain, published = default
    for _, row in chain.expiries.iterrows():
        day = row["expiry"].date()
        mine = chain.quotes[chain.quotes["expiry"] == row["expiry"]]
        np.testing.assert_array_equal(published.strikes(day), np.unique(mine["strike"]))
        assert published.settlement(day) == row["settlement"]
        assert published.maturity(day) == pytest.approx(row["maturity"], abs=1e-12)
        # The parity forward the fit read back is the generator's within 2e-4 (DT-4 acceptance 3).
        assert published.forward(day) == pytest.approx(row["forward"], rel=2e-4)
    assert published.source == SYNTHETIC_LABEL


def test_unlisted_expiries_and_contracts_are_refused(default):
    _, published = default
    with pytest.raises(KeyError, match="not a listed expiry"):
        published.cdf(5800.0, date(2026, 12, 19))
    with pytest.raises(KeyError, match="no call"):
        published.quote(DEC, 5801.0, "call")


def test_the_vol_is_flagged_beyond_the_quotes(default):
    _, published = default
    F = published.forward(DEC)
    inside, wing = published.vol(F, DEC), published.vol(F * math.exp(0.2), DEC)
    assert np.isfinite(inside.vol) and not inside.extrapolated
    assert np.isfinite(wing.vol) and wing.extrapolated  # beyond the 18-Dec quotes (k up to +0.117), within reach
    assert math.isnan(published.vol(F * 3, DEC).vol)  # beyond the reach: no vol


def test_only_a_market_surface_is_published():
    surface = vs.ssvi_surface(DEFAULT_SSVI, 0.04, 0.0, np.linspace(0.7, 1.3, 11), np.array([0.25, 0.5]))
    with pytest.raises(ValueError, match="market surface"):
        PublishedSurface.build(surface, pd.DataFrame(), pd.DataFrame(), spot=100.0, as_of=ASOF, source="x")


def test_the_app_publishes_the_index_market():
    from options_analyzer.ui import market as mk
    from options_analyzer.ui import surface_state

    context = mk.index_market(ASOF)
    published = surface_state.publish(context)
    assert published.source == SYNTHETIC_LABEL and published.as_of == ASOF
    assert published.forward(DEC) == pytest.approx(float(context.expiry_row(DEC)["forward"]), rel=1e-15)
    assert published.vol(5800.0, DEC).vol == context.surface.implied_vol_at(
        1.0, published.maturity(DEC), extrapolate=True
    )
