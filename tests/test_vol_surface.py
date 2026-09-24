import numpy as np
import pandas as pd
import pytest

from options_analyzer import black_scholes as bs
from options_analyzer import vol_surface as vs

M = np.linspace(0.7, 1.3, 25)
T = np.linspace(0.1, 2.0, 12)


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


def test_market_fit_recovers_the_generating_surface():
    quotes, messages = vs.parse_quotes(vs.sample_quotes(100, 0.04, 0.0))
    assert messages == []
    fitted = vs.fit_market_surface(quotes, 100, 0.04, 0.0)
    true = vs.ssvi_surface(vs.SSVIParams(), 0.04, 0.0, fitted.moneyness, fitted.maturities)
    assert np.max(np.abs(fitted.implied_vol - true.implied_vol)) < 0.006  # quotes carry 0.2-pt noise
    assert np.nanmedian(np.abs(fitted.local_vol - true.local_vol)) < 0.01
    assert fitted.calendar_violations == 0


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


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"a": [1], "b": [2]}),
        pd.DataFrame({"Strike": [-1.0], "Maturity": [1.0], "ImpliedVol": [0.2]}),
    ],
)
def test_parse_quotes_rejects_unusable_files(frame):
    with pytest.raises(vs.QuoteParseError):
        vs.parse_quotes(frame)


def test_fit_requires_two_usable_maturities():
    quotes = pd.DataFrame(
        {"strike": [90, 95, 100, 105, 110], "maturity": [0.5] * 5, "iv": [0.25, 0.22, 0.2, 0.19, 0.19]}
    )
    with pytest.raises(vs.QuoteParseError, match="two maturities"):
        vs.fit_market_surface(quotes, 100, 0.0, 0.0)


def test_calendar_arbitrage_is_detected_not_patched():
    quotes, _ = vs.parse_quotes(vs.sample_quotes())
    # Make the longest maturity's total variance smaller than the previous one.
    last = quotes["maturity"].max()
    quotes.loc[quotes["maturity"] == last, "iv"] *= 0.6
    fitted = vs.fit_market_surface(quotes, 100, 0.04, 0.0)
    assert fitted.calendar_violations > 0
    assert np.isnan(fitted.local_vol).any()


def test_summary_and_export():
    surface = vs.ssvi_surface(vs.SSVIParams(), 0.04, 0.0, M, T)
    summary = vs.surface_summary(surface)
    assert len(summary) == len(T)
    assert (summary["Skew 90-110 (vol pts)"] > 0).all()  # negative rho = downside skew
    frames = surface.to_frames(100.0)
    assert frames["LocalVol"].shape == (len(T), len(M))
    assert not np.allclose(frames["LocalVol"].to_numpy()[:, 0], frames["LocalVol"].to_numpy()[:, -1])
