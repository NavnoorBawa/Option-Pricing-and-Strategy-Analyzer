"""Forwards and discount factors per expiry from put-call parity (roadmap item SF-3).

The references are the generators' own forwards and discount factors: closed-form Black-Scholes-Merton prices at a
known rate and dividend yield, and :func:`options_analyzer.market_data.sample_chain`, whose quotes are priced off the
box rate (45 bp above Treasuries) and continuous or cash dividends. The quotes are always read with a sidebar that is
wrong in the ways a desk's sidebar is: the Treasury rate and no dividend yield.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from options_analyzer import black_scholes as bs
from options_analyzer import vol_surface as vs
from options_analyzer.implied_vol import implied_volatility_batch
from options_analyzer.market_data import sample_chain, synthetic, to_quotes
from options_analyzer.vol_surface import forwards as fw

S, RATE, YIELD = 100.0, 0.045, 0.02
DAYS = (30, 91, 182, 365, 547, 730)
AS_OF = datetime(2026, 10, 26, 10, 0)  # a Monday, 10:00 ET, as in tests/test_market_data.py


def _ssvi_chain(*, tick: float = 0.05, half: float = 0.05, days=DAYS, kinds=("call", "put")) -> pd.DataFrame:
    """The plan's acceptance chain: S = 100, r = 4.5 %, q = 2 %, default SSVI, strikes 70-130 in steps of 2.5, mids
    rounded to ``tick`` and quoted ``half`` either side (``tick=0`` keeps the exact model values)."""
    params = vs.SSVIParams()
    rows = []
    for d in days:
        T = d / 365
        forward = S * math.exp((RATE - YIELD) * T)
        for strike in np.arange(70.0, 130.01, 2.5):
            iv = math.sqrt(params.total_variance(math.log(strike / forward), T)[0] / T)
            for kind in kinds:
                value = float(bs.price(S, strike, T, RATE, iv, YIELD, kind))
                mid = round(value / tick) * tick if tick else value
                bid, ask = (round(mid - half, 2), round(mid + half, 2)) if tick else (mid - half, mid + half)
                rows.append({"Strike": strike, "Days": d, "Type": kind, "Bid": bid, "Ask": ask})
    return pd.DataFrame(rows)


def _parsed(frame: pd.DataFrame, rate: float = RATE, dividend_yield: float = 0.0, **kwargs):
    clean, messages, table = vs.parse_quotes(frame, S, rate, dividend_yield, return_forwards=True, **kwargs)
    return clean, messages, table, vs.fit_market_surface(clean, S, rate, dividend_yield, forwards=table)


# --- implied_forwards on its own ------------------------------------------------------------------------------------


def _price_table(strikes, T: float, r: float, q: float, error: float = 0.01, sigma: float = 0.2) -> pd.DataFrame:
    kinds = ("call", "put")
    rows = [(K, T, kind, float(bs.price(S, K, T, r, sigma, q, kind)), error) for K in strikes for kind in kinds]
    return pd.DataFrame(rows, columns=["strike", "maturity", "type", "price", "price_error"])


def test_exact_prices_give_the_forward_and_discount_factor_to_rounding():
    table = fw.implied_forwards(_price_table(np.arange(60.0, 141.0, 5.0), 0.75, 0.05, 0.02), S, 0.04, 0.0)
    row = table.iloc[0]
    assert row["source"] == "parity" and row["pairs"] == 17
    assert row["forward"] == pytest.approx(S * math.exp(0.03 * 0.75), rel=1e-13)
    assert row["discount_factor"] == pytest.approx(math.exp(-0.05 * 0.75), rel=1e-13)
    assert row["rate"] == pytest.approx(0.05, abs=1e-12) and row["dividend_yield"] == pytest.approx(0.02, abs=1e-12)
    # The gap to the given carry (4 % - 0 %) in basis points of the forward: (3 % - 4 %) * 0.75 years.
    assert row["gap_bp"] == pytest.approx(-75.0, abs=1e-8)
    assert list(table.columns) == ["expiry", *fw.FORWARD_COLUMNS]


def test_an_outlier_pair_is_trimmed_and_moves_nothing():
    prices = _price_table(np.arange(80.0, 121.0, 2.5), 0.5, 0.05, 0.02)
    clean = fw.implied_forwards(prices, S, 0.05, 0.02).iloc[0]
    bad = prices.copy()
    bad.loc[(bad["strike"] == 102.5) & (bad["type"] == "call"), "price"] += 0.5  # 50 times its price error
    trimmed = fw.implied_forwards(bad, S, 0.05, 0.02).iloc[0]
    assert trimmed["pairs"] == clean["pairs"] - 1
    assert trimmed["forward"] == pytest.approx(clean["forward"], rel=1e-12)
    assert trimmed["discount_factor"] == pytest.approx(clean["discount_factor"], rel=1e-12)


def test_too_few_pairs_keep_the_given_rates_and_an_uncertain_slope_takes_the_pooled_spread():
    # In order of maturity: five strikes quoted on both sides; a wide span that measures the slope (r = 5 % against
    # a given 4 %); a two-point strike span with noisy prices, too narrow to measure it.
    rng = np.random.default_rng(1)
    few = _price_table(np.arange(90.0, 111.0, 5.0), 0.25, 0.05, 0.0)
    narrow = _price_table(np.arange(99.0, 101.01, 0.25), 1.0, 0.05, 0.0)
    narrow["price"] += rng.uniform(-0.01, 0.01, len(narrow))
    wide = _price_table(np.arange(60.0, 141.0, 2.5), 0.5, 0.05, 0.0)
    table = fw.implied_forwards(pd.concat([few, narrow, wide]), S, 0.04, 0.0)
    assert table["source"].tolist() == ["rates", "parity", "parity_forward"]
    assert table["discount_factor"].iloc[0] == pytest.approx(math.exp(-0.04 * 0.25), rel=1e-15)
    # The narrow expiry discounts at the given rate plus the spread measured on the wide one (100 bp).
    assert table["rate"].iloc[2] == pytest.approx(0.05, abs=1e-9)
    assert table["forward"].iloc[2] == pytest.approx(S * math.exp(0.05), abs=0.01)
    assert np.isnan(table["discount_se"].iloc[2]) and table["discount_se"].iloc[1] < fw.MAX_DISCOUNT_SE


def test_parity_off_gives_the_table_of_the_given_rates():
    table = fw.implied_forwards(_price_table(np.arange(80.0, 121.0, 5.0), 0.5, 0.05, 0.02), S, 0.03, 0.01, parity=False)
    assert table["source"].tolist() == ["rates"] and table["pairs"].tolist() == [0]
    assert table["forward"].iloc[0] == pytest.approx(S * math.exp(0.02 * 0.5), rel=1e-15)
    assert table["gap_bp"].iloc[0] == pytest.approx(0.0, abs=1e-9)


# --- The plan's acceptance tests (roadmap item SF-3) ----------------------------------------------------------------


def test_acceptance_1_forwards_and_an_arbitrage_free_surface_from_a_sidebar_without_the_yield():
    """S = 100, r = 4.5 %, q = 2 %, six expiries from 30 days to two years, strikes 70-130 in steps of 2.5, mids
    rounded to $0.05 and quoted $0.05 either side, read with the sidebar's q = 0.

    Every forward is within 2 bp (measured 0.89) and every discount factor within three of its own standard errors
    (measured errors up to 4.0e-4, standard errors 2.3e-4 to 4.6e-4). The plan asked for 2e-5, which these quotes
    cannot give: rounding each price to $0.05 on a $100 underlying leaves the slope of C - P that uncertain, and it
    costs the vols at most sigma * 4e-4, 0.01 vol point. No calendar or butterfly point and no static-arbitrage note
    (with the sidebar forward: 7 calendar and 12 butterfly points). The fit RMSE is below 0.4 vol point from the
    second expiry on (measured 0.11 at most) and 0.68 at 30 days, where wing quotes worth a few ticks are rounded by
    up to a third of their value: weighting quotes by their spread is roadmap item SF-4.
    """
    _, _, table, surface = _parsed(_ssvi_chain())
    T = table["maturity"].to_numpy()
    assert np.max(np.abs(np.log(table["forward"] / (S * np.exp((RATE - YIELD) * T))))) < 2e-4
    error = np.abs(table["discount_factor"] - np.exp(-RATE * T))
    assert (table["source"] == "parity").all() and (error <= 3 * table["discount_se"]).all()
    assert surface.calendar_violations == 0 and surface.butterfly_violations == 0
    assert not [note for note in surface.notes if "arbitrage" in note]
    assert surface.fit_rmse[1:].max() < 0.004 and surface.fit_rmse[0] < 0.0075
    by_rates = _parsed(_ssvi_chain(), forward="rates")[3]
    assert by_rates.calendar_violations + by_rates.butterfly_violations > 0  # what the sidebar forward gave


def test_acceptance_2_call_and_put_vols_agree_within_their_spreads():
    """At each strike quoted on both sides, the call and the put inverted at the parity forward and discount factor
    differ by less than the sum of their half-spreads in vol (half-spread over vega). Measured: at most 0.52 of it."""
    frame = _ssvi_chain()
    _, _, table, _ = _parsed(frame)
    frame = frame[frame["Bid"] > 0].assign(T=frame["Days"] / 365, mid=(frame["Bid"] + frame["Ask"]) / 2)
    for row in table.itertuples():
        quotes = frame[np.isclose(frame["T"], row.maturity)]
        r_T = -math.log(row.discount_factor) / row.maturity
        q_T = r_T - math.log(row.forward / S) / row.maturity
        K, T = quotes["Strike"].to_numpy(), quotes["T"].to_numpy()
        iv = implied_volatility_batch(quotes["mid"].to_numpy(), S, K, T, r_T, q_T, quotes["Type"].to_numpy())
        sd = iv * np.sqrt(T)
        d1 = np.log(row.forward / K) / sd + 0.5 * sd
        vega = row.discount_factor * row.forward * np.sqrt(T) * np.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
        sides = quotes.assign(iv=iv, half=(quotes["Ask"] - quotes["Bid"]) / 2 / vega)
        both = sides.pivot_table(index="Strike", columns="Type", values=["iv", "half"]).dropna()
        gap = (both[("iv", "call")] - both[("iv", "put")]).abs()
        assert (gap <= both[("half", "call")] + both[("half", "put")]).all(), row.maturity


def test_acceptance_3_a_one_sided_chain_falls_back_to_the_sidebar_and_says_so():
    clean, messages, table, surface = _parsed(_ssvi_chain(kinds=("call",)), dividend_yield=YIELD)
    assert (table["source"] == "rates").all() and "forward" not in clean
    assert any("sidebar rate and dividend yield: put-call parity needs a call and a put" in m for m in messages)
    assert surface.log_forward_ratio == pytest.approx((RATE - YIELD) * surface.maturities, abs=1e-15)


def test_acceptance_4_noise_free_quotes_are_still_recovered():
    """Exact model prices on both sides: the parity forward is the model's, and the SSVI vols come back to 1e-9."""
    _, _, table, surface = _parsed(_ssvi_chain(tick=0.0, half=0.01))
    T = table["maturity"].to_numpy()
    assert table["forward"].to_numpy() == pytest.approx(S * np.exp((RATE - YIELD) * T), rel=1e-12)
    true = vs.ssvi_surface(vs.SSVIParams(), RATE, YIELD, surface.moneyness, surface.maturities)
    assert np.nanmax(np.abs(surface.implied_vol - true.implied_vol)) < 1e-9
    assert np.max(surface.fit_rmse) < 1e-9


def test_acceptance_5_with_the_true_rate_and_yield_parity_changes_nothing():
    """Exact prices read with their own rate and dividend yield: the parity path gives the vols and the surface of the
    sidebar path (the one before parity forwards existed) to 1e-10."""
    frame = _ssvi_chain(tick=0.0, half=0.01)
    parity = _parsed(frame, dividend_yield=YIELD)
    rates = _parsed(frame, dividend_yield=YIELD, forward="rates")
    assert "forward" in parity[0] and "forward" not in rates[0]
    assert parity[0]["iv"].to_numpy() == pytest.approx(rates[0]["iv"].to_numpy(), abs=1e-10)
    assert np.allclose(parity[3].implied_vol, rates[3].implied_vol, rtol=0, atol=1e-10, equal_nan=True)
    assert np.allclose(parity[3].local_vol, rates[3].local_vol, rtol=0, atol=1e-8, equal_nan=True)


def test_the_synthetic_spx_chain_read_with_the_treasury_rate():
    """The panel's run (docs/reviews/panel-v3.0.md, quant developer, finding 2): the synthetic SPX chain with the
    generator's Treasury rate and yield. With the sidebar forward: 135 quotes dropped as outside the no-arbitrage
    bounds, 26 calendar violations and fits up to 1.33 vol points from the quotes. With parity forwards: none
    dropped, no violation, largest fit RMSE 0.068 vol point, ATM vols within 0.006 vol point and forwards within
    0.3 bp."""
    chain = sample_chain(5000.0, 0.04, 0.012, as_of=AS_OF)
    quotes = to_quotes(chain.quotes, as_of=AS_OF)
    clean, messages, table = vs.parse_quotes(quotes, 5000.0, 0.04, 0.012, return_forwards=True)
    surface = vs.fit_market_surface(clean, 5000.0, 0.04, 0.012, forwards=table)
    assert not [m for m in messages if "outside the no-arbitrage bounds" in m]
    assert surface.calendar_violations == 0 and surface.butterfly_violations == 0
    assert surface.fit_rmse.max() < 0.001
    assert len(surface.forwards) == len(chain.expiries) == len(surface.maturities)
    for i, row in enumerate(chain.expiries.itertuples()):
        assert math.exp(surface.log_forward[i]) == pytest.approx(row.forward, rel=5e-5)
        assert surface.implied_vol_at(row.forward / 5000.0, row.maturity) == pytest.approx(row.atm_vol, abs=2e-4)
    # The box rate is 45 bp above the Treasury rate the sidebar holds: the gap grows by 45 bp of the forward a year.
    assert table["gap_bp"].to_numpy() == pytest.approx(1e4 * synthetic.BOX_SPREAD * table["maturity"], abs=1.0)


def test_in_the_money_quotes_without_time_value_are_dropped():
    """An out-of-the-money option worth less than its half-spread has a negative bid and is dropped; the
    in-the-money option at its strike is then alone, and its time value (equal to the other's value, by parity) is
    within its own spread, so it is dropped too rather than inverted into a meaningless vol."""
    frame = _ssvi_chain(tick=0.0, half=0.05, days=(30, 91))
    no_bid = frame[frame["Bid"] <= 0]
    clean, messages = vs.parse_quotes(frame, S, RATE, YIELD)
    assert f"Dropped {len(no_bid)} quote(s) with a zero bid or an ask below the bid." in messages
    assert any(m.startswith(f"Dropped {len(no_bid)} in-the-money quote(s) whose time value") for m in messages)
    for row in no_bid.itertuples():
        assert not ((clean["strike"] == row.Strike) & np.isclose(clean["maturity"], row.Days / 365)).any()


def test_the_forward_method_is_checked():
    with pytest.raises(ValueError, match="forward must be one of 'parity', 'rates'"):
        vs.parse_quotes(_ssvi_chain(days=(30,)), S, RATE, 0.0, forward="sidebar")
