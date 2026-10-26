"""market_data: the canonical chain schema, the synthetic chain (DT-4) and the upload adapters (DT-5)."""

from __future__ import annotations

import io
import math
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

from options_analyzer import clock
from options_analyzer import vol_surface as vs
from options_analyzer.market_data import (
    CHAIN_COLUMNS,
    SYNTHETIC_LABEL,
    ChainSchemaError,
    occ_symbol,
    parse_occ_symbol,
    read_chain,
    sample_chain,
    synthetic,
    to_quotes,
    validate_chain,
)
from options_analyzer.market_data.adapters import clean_chain

FIXTURES = Path(__file__).parent / "fixtures"
AS_OF = datetime(2026, 10, 26, 10, 0)  # a Monday, 10:00 ET
SPOT, UST, YIELD = 6000.0, 0.04, 0.013
BOX = UST + synthetic.BOX_SPREAD
#: SPY-like cash dividends: quarterly ex-dates near the third Friday of the last month of each quarter.
SPY_DIVIDENDS = [(date(2026, 12, 18), 1.85), (date(2027, 3, 19), 1.75), (date(2027, 6, 18), 1.80)]
#: Cash dividends of 1% of the spot each quarter on a $100 ETF: the case where pricing the exercise just before an
#: ex-date at the expiry's own smile vol made calls lose value with maturity.
ONE_PERCENT = [(day, 1.0) for day in (date(2026, 12, 18), date(2027, 3, 19), date(2027, 6, 18), date(2027, 9, 17))]


@pytest.fixture(scope="module")
def index_chain() -> synthetic.SyntheticChain:
    return sample_chain(SPOT, UST, YIELD, as_of=AS_OF)


@pytest.fixture(scope="module")
def index_model() -> synthetic.SyntheticChain:
    return sample_chain(SPOT, UST, YIELD, as_of=AS_OF, noise=False)


@pytest.fixture(scope="module")
def etf_chain() -> synthetic.SyntheticChain:
    return sample_chain(600.0, UST, SPY_DIVIDENDS, as_of=AS_OF, style="etf")


@pytest.fixture(scope="module")
def etf_one_percent() -> synthetic.SyntheticChain:
    return sample_chain(100.0, UST, ONE_PERCENT, as_of=AS_OF, style="etf", noise=False)


def _with_expiry_data(chain: synthetic.SyntheticChain) -> pd.DataFrame:
    """The quotes with the model value and vol, and the forward, discount factor and maturity of their expiry."""
    columns = ["forward", "discount_factor", "maturity", "variance_time", "atm_vol", "rate"]
    expiries = chain.expiries.set_index(["expiry", "root"])[columns]
    return chain.quotes.join(chain.theoretical).join(expiries, on=["expiry", "root"])


def _black_vega(rows: pd.DataFrame) -> np.ndarray:
    """Black-76 vega per unit of vol: DF F n(d1) sqrt(T)."""
    root_t = np.sqrt(rows["maturity"].to_numpy())
    forward, sd = rows["forward"].to_numpy(), rows["iv"].to_numpy() * root_t
    d1 = np.log(forward / rows["strike"].to_numpy()) / sd + 0.5 * sd
    density = np.exp(-0.5 * d1**2) / math.sqrt(2.0 * math.pi)
    return rows["discount_factor"].to_numpy() * forward * density * root_t


def _price_arbitrage(quotes: pd.DataFrame) -> int:
    """Quote-level static-arbitrage flags, model-free, for European or American options.

    Per expiry, root and type, buying at the ask and selling at the bid: a call spread (put spread) that costs less
    than nothing, or a credit spread that takes in more than the strike difference (the most a vertical can pay,
    also with early exercise), and a butterfly of three neighbouring strikes that costs less than nothing.
    """
    flags = 0
    for (_, _, kind), group in quotes.groupby(["expiry", "root", "type"]):
        group = group.sort_values("strike")
        K, bid, ask = (group[c].to_numpy(float) for c in ("strike", "bid", "ask"))
        if kind == "put":  # puts rise with the strike: mirror them so that both read like calls
            K, bid, ask = -K[::-1], bid[::-1], ask[::-1]
        upper = np.triu(np.ones((len(K), len(K)), dtype=bool), 1)  # pairs i < j (K_i < K_j)
        flags += int(np.sum(upper & (bid[None, :] > ask[:, None] + 1e-9)))  # buy i, sell j: never pays less than 0
        width = K[None, :] - K[:, None]
        flags += int(np.sum(upper & (bid[:, None] - ask[None, :] > width + 1e-9)))  # sell i, buy j: costs <= width
        lam = (K[2:] - K[1:-1]) / (K[2:] - K[:-2])
        flags += int(np.sum(lam * ask[:-2] + (1 - lam) * ask[2:] - bid[1:-1] < -1e-9))
    return flags


def _calendar_arbitrage(quotes: pd.DataFrame) -> int:
    """Quote-level calendar flags for American options: buying the later expiry at its ask and selling the earlier
    one at its bid, same root, type and strike, must not take in money (an American option is worth at least as much
    with more time to run)."""
    flags = 0
    for _, group in quotes.groupby(["root", "type", "strike"]):
        group = group.sort_values("expiry")
        bid, ask = group["bid"].to_numpy(float), group["ask"].to_numpy(float)
        later = np.triu(np.ones((len(bid), len(bid)), dtype=bool), 1)  # pairs i < j (expiry_i < expiry_j)
        flags += int(np.sum(later & (bid[:, None] > ask[None, :] + 1e-9)))
    return flags


def _value_drops_with_maturity(chain: synthetic.SyntheticChain) -> int:
    """Pairs (same root, type and strike, expiry_i < expiry_j) whose model value falls with maturity."""
    rows = chain.quotes.join(chain.theoretical)
    drops = 0
    for _, group in rows.groupby(["root", "type", "strike"]):
        value = group.sort_values("expiry")["value"].to_numpy()
        later = np.triu(np.ones((len(value), len(value)), dtype=bool), 1)
        drops += int(np.sum(later & (value[None, :] < value[:, None] - 1e-9)))
    return drops


def _surface(chain: pd.DataFrame, as_of: datetime, rate: float, dividend_yield: float) -> vs.VolSurface:
    quotes, _ = vs.parse_quotes(to_quotes(chain, as_of=as_of), SPOT, rate, dividend_yield)
    return vs.fit_market_surface(quotes, SPOT, rate, dividend_yield)


def _arbitrage_notes(surface: vs.VolSurface) -> list[str]:
    return [note for note in surface.notes if "static arbitrage" in note or "no arbitrage-free smile" in note]


# --- OCC symbols and the schema -----------------------------------------------------------------------------------


def test_occ_symbols_round_trip():
    """DT-5 acceptance (2), and the padded 21-character form."""
    assert parse_occ_symbol("SPXW261102C07930000") == ("SPXW", date(2026, 11, 2), "call", 7930.0)
    assert parse_occ_symbol("spy   261218p00612500") == ("SPY", date(2026, 12, 18), "put", 612.5)
    assert occ_symbol("SPXW", date(2026, 11, 2), "call", 7930) == "SPXW261102C07930000"
    for strike in (0.5, 7.125, 612.5, 7930.0, 99_999.999):
        symbol = occ_symbol("XYZ1", date(2030, 2, 28), "put", strike)
        assert parse_occ_symbol(symbol).strike == pytest.approx(strike, abs=1e-9)
    for bad in ("SPXW261102X07930000", "SPXW2611020C7930000", "SPXW261302C07930000", "TOOLONGR261102C07930000"):
        with pytest.raises(ValueError):
            parse_occ_symbol(bad)
    with pytest.raises(ValueError, match=r"0\.001"):
        occ_symbol("SPX", date(2026, 11, 20), "call", 100.0001)
    with pytest.raises(ValueError, match="call' or 'put"):
        occ_symbol("SPX", date(2026, 11, 20), "straddle", 100.0)


def _one_contract(**changes) -> pd.DataFrame:
    row = {
        "symbol": "SPXW261102C07930000",
        "root": "SPXW",
        "expiry": pd.Timestamp("2026-11-02"),
        "settlement": "PM",
        "type": "call",
        "strike": 7930.0,
        "bid": 12.3,
        "ask": 12.9,
        "vendor_iv": np.nan,
    }
    frame = pd.DataFrame([row]).astype({"expiry": "datetime64[ns]"})
    for column, value in changes.items():
        frame[column] = [value]
    return frame


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("symbol", "SPXW261102P07930000", "disagree"),
        ("root", "spxw", "OCC root"),
        ("expiry", pd.Timestamp("2026-11-02 16:00"), "date at midnight"),
        ("settlement", "noon", "'AM' or 'PM'"),
        ("type", "c", "'call' or 'put'"),
        ("strike", 7930.0005, "0.001"),
        ("bid", -0.05, "at least 0"),
        ("ask", 0.0, "above 0"),
        ("bid", 13.0, "bid above the ask"),
        ("vendor_iv", -0.2, "negative"),
    ],
)
def test_schema_validation_names_the_bad_column(column, value, message):
    validate_chain(_one_contract())
    with pytest.raises(ChainSchemaError, match=f"{column}: .*{message}"):
        validate_chain(_one_contract(**{column: value}))


def test_schema_validation_checks_columns_and_types():
    with pytest.raises(ChainSchemaError, match="missing column"):
        validate_chain(_one_contract().drop(columns="ask"))
    with pytest.raises(ChainSchemaError, match="strike: expected floats"):
        validate_chain(_one_contract().astype({"strike": object}))
    with pytest.raises(ChainSchemaError, match="expiry: expected dates"):
        validate_chain(_one_contract().astype({"expiry": str}))
    doubled = pd.concat([_one_contract(), _one_contract()], ignore_index=True)
    with pytest.raises(ChainSchemaError, match=r"symbol: 1 row\(s\) repeated"):
        validate_chain(doubled)


# --- The synthetic chain (DT-4) -----------------------------------------------------------------------------------


def test_synthetic_chain_is_labelled_and_passes_schema_validation(index_chain, etf_chain):
    """Acceptance (5): every column of both styles passes the schema check."""
    for chain in (index_chain, etf_chain):
        assert chain.label == SYNTHETIC_LABEL == "Synthetic — not market data"
        assert chain.quotes.attrs["source"] == SYNTHETIC_LABEL
        assert list(chain.quotes.columns) == list(CHAIN_COLUMNS)
        validate_chain(chain.quotes)
        assert chain.theoretical.index.equals(chain.quotes.index)


def test_synthetic_chain_is_deterministic_per_seed(index_chain, etf_chain):
    """Acceptance (1): the same inputs give the same chain byte for byte; another seed moves the quotes only."""
    again = sample_chain(SPOT, UST, YIELD, as_of=AS_OF)
    pd.testing.assert_frame_equal(again.quotes, index_chain.quotes)
    pd.testing.assert_frame_equal(again.expiries, index_chain.expiries)
    assert again.quotes.to_csv(index=False) == index_chain.quotes.to_csv(index=False)
    other = sample_chain(SPOT, UST, YIELD, as_of=AS_OF, seed=1)
    assert other.quotes["symbol"].equals(index_chain.quotes["symbol"])
    assert not other.quotes["bid"].equals(index_chain.quotes["bid"])
    pd.testing.assert_frame_equal(other.theoretical, index_chain.theoretical)
    # Without noise the seed has no effect.
    quiet = [sample_chain(SPOT, UST, YIELD, as_of=AS_OF, noise=False, seed=seed).quotes for seed in (0, 7)]
    pd.testing.assert_frame_equal(*quiet)
    etf_again = sample_chain(600.0, UST, SPY_DIVIDENDS, as_of=AS_OF, style="etf")
    pd.testing.assert_frame_equal(etf_again.quotes, etf_chain.quotes)


def _check_listing(chain: synthetic.SyntheticChain) -> None:
    """Acceptance (4) and the strike grid: under 10,000 rows, at most 18 expiries and 243 strikes an expiry (the band
    limits of the module docstring); every strike on the exchange's interval; every strike at that interval near the
    money (within 1.9% of the forward, or 45 intervals if that is less, and at least 5 intervals)."""
    interval = synthetic.strike_interval(chain.spot, chain.style)
    q = chain.quotes
    assert len(q) < 10_000, (chain.style, chain.spot, len(q))
    assert len(chain.expiries) <= 18  # 6 weeklies, 6 monthlies and 6 quarterlies in months 7 to 24
    assert np.allclose(q["strike"] / interval, np.round(q["strike"] / interval), rtol=0, atol=1e-9)
    for row in chain.expiries.itertuples():
        strikes = q.loc[(q["expiry"] == row.expiry) & (q["root"] == row.root) & (q["type"] == "call"), "strike"]
        strikes = strikes.to_numpy()
        assert len(strikes) <= 243 and row.strikes == len(strikes)
        radius = max(5 * interval, min(0.019 * row.forward, 45 * interval))
        near = strikes[np.abs(strikes - row.forward) <= radius]
        assert len(near) >= 10 and np.allclose(np.diff(near), interval), (chain.spot, row.expiry)


#: SPX-like index levels and ETF prices; just below a round level is where a spot-relative step used to halve.
INDEX_SPOTS = [*range(3_000, 12_001, 100), 2_999, 5_999, 11_999]
ETF_SPOTS = [*range(50, 1_201, 50), 5, 24.99, 25, 599, 1_199]


@pytest.mark.parametrize(
    ("style", "spots"), [("index", INDEX_SPOTS), ("etf", ETF_SPOTS)], ids=["index-3000-12000", "etf-50-1200"]
)
def test_strike_grid_and_row_budget_over_spot_levels(style, spots, monkeypatch):
    """Acceptance (4) and the near-the-money strikes at every spot, not only at the round 6,000. The listing does not
    depend on prices, so the ETF's tree is replaced by a stub to keep the sweep fast."""
    monkeypatch.setattr(synthetic.binomial, "price", lambda *args: 0.0)
    rows = []
    for spot in spots:
        chain = sample_chain(float(spot), UST, YIELD, as_of=AS_OF, style=style, noise=False)
        _check_listing(chain)
        rows.append(len(chain.quotes))
        interval = synthetic.strike_interval(spot, style)
        if spot >= (3_000 if style == "index" else 200):  # coarser wings once the far strikes outnumber the limits
            longest = chain.quotes[chain.quotes["expiry"] == chain.quotes["expiry"].max()]
            assert np.diff(np.unique(longest["strike"])).max() > interval
    assert max(rows) < 10_000
    # The row count moves within a narrow range with the spot instead of doubling below a round level.
    if style == "index":
        assert min(rows) > 3_000 and max(rows) < 1.7 * min(rows)


@pytest.mark.slow
def test_row_budget_over_spot_levels_at_other_valuation_times(monkeypatch):
    """Acceptance (4) at two more valuation times (the day before Juneteenth and an expiry morning): SPX-like levels
    3,000 to 12,000 in steps of 100 and ETF prices $50 to $1,200 in steps of $25."""
    monkeypatch.setattr(synthetic.binomial, "price", lambda *args: 0.0)
    for as_of in (datetime(2026, 6, 18, 15, 30), datetime(2026, 11, 20, 9)):
        for style, spots in (("index", range(3_000, 12_001, 100)), ("etf", range(50, 1_201, 25))):
            for spot in spots:
                _check_listing(sample_chain(float(spot), UST, YIELD, as_of=as_of, style=style, noise=False))


def test_default_expiry_set_stays_under_10000_rows():
    """Acceptance (4) for the default 6,000 chain, at valuation times through a year, holidays and expiry days
    included, and for the default ETF chain with its binomial tree."""
    starts = [datetime(2026, 1, 2, 9, 45) + timedelta(days=28 * i) for i in range(14)]
    starts += [datetime(2026, 4, 2, 15, 0), datetime(2026, 11, 20, 9, 0), datetime(2026, 12, 24, 12, 59)]
    rows = []
    for start in starts:
        chain = sample_chain(SPOT, UST, YIELD, as_of=start, noise=False)
        _check_listing(chain)
        rows.append(len(chain.quotes))
    assert max(rows) < 10_000
    assert min(rows) > 4_000
    etf = sample_chain(600.0, UST, YIELD, as_of=datetime(2026, 6, 1, 10), style="etf", noise=False)
    _check_listing(etf)


def test_strike_interval_and_the_supported_spots():
    assert synthetic.strike_interval(6000.0) == synthetic.strike_interval(1_000.0) == 5.0
    assert synthetic.strike_interval(999.99) == synthetic.strike_interval(100.0) == 1.0
    assert synthetic.strike_interval(600.0, "etf") == synthetic.strike_interval(25.0, "etf") == 1.0
    assert synthetic.strike_interval(24.99, "etf") == synthetic.strike_interval(5.0, "etf") == 0.5
    for spot, style in ((99.9, "index"), (4.99, "etf"), (50_001.0, "index"), (math.nan, "etf"), (-1.0, "index")):
        with pytest.raises(ValueError, match="spot must be from"):
            sample_chain(spot, UST, YIELD, as_of=AS_OF, style=style)
    # At the lowest spots every expiry still lists at least ten strikes, some with a bid.
    for spot, style in ((100.0, "index"), (5.0, "etf")):
        chain = sample_chain(spot, UST, YIELD, as_of=AS_OF, style=style, noise=False)
        per_expiry = chain.quotes.groupby("expiry")
        assert per_expiry.size().min() >= 2 * 10
        assert (per_expiry["bid"].apply(lambda bids: (bids > 0).sum()) >= 4).all()


def test_expiries_weeklies_monthlies_and_settlement(index_chain, etf_chain):
    expiries = index_chain.expiries
    for row in expiries.itertuples():
        monthly = clock.previous_trading_day(synthetic._third_friday(row.expiry.year, row.expiry.month))
        expected = ("SPX", "AM") if row.expiry.date() == monthly else ("SPXW", "PM")
        assert (row.root, row.settlement) == expected, row.expiry
    # Juneteenth 2027 is the third Friday of June: that monthly expires on Thursday 17 June.
    assert pd.Timestamp("2027-06-17") in set(expiries["expiry"])
    # Weeklies for six weeks, then monthlies and quarterlies out to two years.
    assert (expiries["maturity"] * 365 < 45).sum() >= 6
    assert expiries["maturity"].max() == pytest.approx(1.89, abs=0.05)
    # Every expiry settles at its clock time: AM at 09:30, PM at 16:00 or 13:00 on the day after Thanksgiving.
    for row in expiries.itertuples():
        assert row.settles_at == clock.settlement_time(row.expiry.date(), row.settlement)
        assert row.maturity == clock.year_fraction(AS_OF, row.settles_at)
    assert expiries.loc[expiries["expiry"] == "2026-11-27", "settles_at"].iloc[0].hour == 13
    assert set(etf_chain.expiries["root"]) == {"SPY"} and set(etf_chain.expiries["settlement"]) == {"PM"}
    # A weekly that falls on Good Friday expires on the Thursday before.
    easter = sample_chain(SPOT, UST, YIELD, as_of=datetime(2026, 3, 23, 10), noise=False).expiries["expiry"]
    assert pd.Timestamp("2026-04-02") in set(easter) and pd.Timestamp("2026-04-03") not in set(easter)


def test_strikes_ticks_and_quote_shape(index_chain, etf_chain):
    q = index_chain.quotes
    # Calls and puts at every strike of every expiry.
    sides = q.groupby(["expiry", "root", "strike"])["type"].agg(lambda kinds: tuple(sorted(kinds)))
    assert (sides == ("call", "put")).all()
    # 5-point strikes near the money and coarser wings, every strike a multiple of 5.
    front = q[(q["expiry"] == q["expiry"].min()) & (q["type"] == "call")]["strike"].to_numpy()
    steps = np.diff(front)
    assert steps.min() == 5 and steps.max() == 50 and np.all(front % 5 == 0)
    forward = index_chain.expiries["forward"].iloc[0]
    assert np.all(steps[np.abs(front[1:] / forward - 1) < 0.02] == 5)
    assert np.all(steps[np.abs(front[1:] / forward - 1) > 0.085] == 50)
    longest = q[(q["expiry"] == q["expiry"].max()) & (q["type"] == "call")]["strike"].to_numpy()
    assert longest.min() < 0.45 * SPOT and longest.max() > 1.45 * SPOT  # wings out to -60% and +40% in ln K/F
    # Ticks: $0.05 below $3, $0.10 from $3 (SPX); pennies for the ETF.
    for column in ("bid", "ask"):
        cents = np.round(q[column].to_numpy() * 100).astype(int)
        assert np.allclose(q[column] * 100, cents, atol=1e-6)
        assert np.all(np.where(q[column] < 3, cents % 5, cents % 10) == 0)
        assert np.allclose(etf_chain.quotes[column] * 100, np.round(etf_chain.quotes[column] * 100), atol=1e-6)
    assert (q["ask"] > q["bid"]).all() and (etf_chain.quotes["ask"] > etf_chain.quotes["bid"]).all()
    # Zero-bid (offer-only) wings, and only on cheap options.
    zero_bid = q["bid"] == 0
    assert 0 < zero_bid.mean() < 0.1
    assert index_chain.theoretical.loc[zero_bid, "value"].max() < 1.0


def test_tick_rounding_uses_the_tick_of_the_rounded_price():
    """Bids round down and asks up to the nearest valid tick strictly beyond the price: SPX trades in $0.05 below $3
    and $0.10 from $3 (so $3.05 is never a price), SPY in pennies. Checked against every valid tick, by search."""
    assert synthetic._round_up(np.array([2.95, 2.999, 3.0, 3.001, 3.1]), "index").tolist() == [3.0, 3.0, 3.1, 3.1, 3.2]
    assert synthetic._round_down(np.array([3.0, 3.001, 3.1, 3.16]), "index").tolist() == [2.95, 3.0, 3.0, 3.1]
    prices = np.round(np.linspace(0.001, 12.0, 24_000), 6)
    for style, ticks in (
        ("index", np.round(np.concatenate([np.arange(60) * 0.05, 3.0 + np.arange(200) * 0.1]), 2)),
        ("etf", np.round(np.arange(1_500) * 0.01, 2)),
    ):
        up = ticks[np.searchsorted(ticks, prices + 1e-9, side="right")]
        down = ticks[np.maximum(np.searchsorted(ticks, prices - 1e-9, side="left") - 1, 0)]
        np.testing.assert_array_equal(synthetic._round_up(prices, style), up)
        np.testing.assert_array_equal(synthetic._round_down(prices, style), down)


def test_model_value_lies_inside_every_quote(index_chain, index_model, etf_chain):
    for chain in (index_chain, index_model, etf_chain):
        value = chain.theoretical["value"]
        assert (chain.quotes["bid"] <= value).all() and (value <= chain.quotes["ask"]).all()
    pd.testing.assert_frame_equal(index_chain.theoretical, index_model.theoretical)


def test_spreads_widen_with_moneyness_and_maturity(index_model):
    rows = _with_expiry_data(index_model)
    rows = rows[rows["bid"] > 0]
    vol_spread = (rows["ask"] - rows["bid"]).to_numpy() / _black_vega(rows)
    sds = np.abs(np.log(rows["strike"] / rows["forward"])) / (rows["atm_vol"] * np.sqrt(rows["maturity"]))
    near, far = vol_spread[(sds < 0.5).to_numpy()], vol_spread[((sds > 1.5) & (sds < 3)).to_numpy()]
    assert np.median(far) > 2 * np.median(near)
    atm = (sds < 0.25).to_numpy()
    by_maturity = pd.Series(vol_spread[atm], index=rows["maturity"][atm]).groupby(level=0).median()
    # The shortest expiries are left out: rounding to the tick widens their vol spread.
    short, long = (
        by_maturity[(by_maturity.index >= 0.05) & (by_maturity.index <= 0.25)],
        by_maturity[by_maturity.index > 1],
    )
    assert long.median() > 1.25 * short.median()


def test_index_values_are_black76_on_the_ssvi_surface_in_variance_time(index_model):
    """Put-call parity at the box rate, total variance from SSVI at the variance time, Black-76 prices."""
    rows = _with_expiry_data(index_model)
    k = np.log(rows["strike"] / rows["forward"]).to_numpy()
    w = synthetic.DEFAULT_SSVI.total_variance(k, rows["variance_time"].to_numpy())[0]
    assert np.allclose(rows["iv"] ** 2 * rows["maturity"], w, rtol=1e-12, atol=0)
    assert np.allclose(rows["rate"], BOX, rtol=0, atol=1e-15)  # flat Treasury curve plus 45 bp
    assert np.allclose(rows["discount_factor"], np.exp(-BOX * rows["maturity"]), rtol=1e-15)
    assert np.allclose(rows["forward"], SPOT * np.exp((BOX - YIELD) * rows["maturity"]), rtol=1e-14)
    calls = rows[rows["type"] == "call"].set_index(["expiry", "root", "strike"])
    puts = rows[rows["type"] == "put"].set_index(["expiry", "root", "strike"])
    strikes = calls.index.get_level_values("strike").to_numpy()
    parity = calls["discount_factor"] * (calls["forward"] - strikes)
    assert np.allclose(calls["value"] - puts["value"], parity, rtol=0, atol=1e-9 * SPOT)
    atm = index_model.expiries
    assert np.allclose(atm["atm_vol"] ** 2 * atm["maturity"], synthetic.DEFAULT_SSVI.theta(atm["variance_time"]))


@pytest.mark.oracle
def test_index_values_match_quantlib_black_calculator(index_model):
    ql = pytest.importorskip("QuantLib")
    rows = _with_expiry_data(index_model).iloc[::37]
    for row in rows.itertuples():
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if row.type == "call" else ql.Option.Put, row.strike)
        sd = row.iv * math.sqrt(row.maturity)
        theirs = ql.BlackCalculator(payoff, row.forward, sd, row.discount_factor).value()
        assert row.value == pytest.approx(theirs, rel=1e-11, abs=1e-11 * SPOT)


@pytest.mark.oracle
def test_etf_tree_values_match_quantlib_finite_differences():
    """With a continuous yield the ETF chain prices puts and calls on the binomial tree. Each value is the tree at the
    quote's time to expiry, and the tree agrees with QuantLib's finite-difference American engine (an independent
    method, on a 1,000 x 1,000 grid) within a cent at whole-day maturities."""
    ql = pytest.importorskip("QuantLib")
    chain = sample_chain(600.0, UST, 0.013, as_of=AS_OF, style="etf", noise=False)
    rows = _with_expiry_data(chain)
    rows = rows[(rows["maturity"] > 0.1) & (rows["maturity"] < 0.5)].iloc[::23]
    today = ql.Date(26, 10, 2026)
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    for row in rows.itertuples():
        args = (row.strike, row.maturity, row.rate, row.iv, 0.013, row.type, synthetic.TREE_STEPS, True, "lr")
        assert row.value == synthetic.binomial.price(600.0, *args)
        days = int(row.maturity * 365)
        expiry = today + ql.Period(days, ql.Days)
        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(600.0)),
            ql.YieldTermStructureHandle(ql.FlatForward(today, 0.013, day_count)),
            ql.YieldTermStructureHandle(ql.FlatForward(today, row.rate, day_count)),
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), row.iv, day_count)),
        )
        kind = ql.Option.Call if row.type == "call" else ql.Option.Put
        option = ql.VanillaOption(ql.PlainVanillaPayoff(kind, row.strike), ql.AmericanExercise(today, expiry))
        option.setPricingEngine(ql.FdBlackScholesVanillaEngine(process, 1000, 1000))
        tree = synthetic.binomial.price(
            600.0, row.strike, days / 365, row.rate, row.iv, 0.013, row.type, synthetic.TREE_STEPS, True, "lr"
        )
        assert tree == pytest.approx(option.NPV(), abs=0.01), (row.type, row.strike, days)


def test_etf_american_values_with_cash_dividends(etf_chain):
    """Escrowed dividends: the forward nets out their present value; calls and puts are worth at least the European
    value and their exercise value."""
    rows = _with_expiry_data(etf_chain)
    expiries = etf_chain.expiries
    paid = expiries["expiry"] >= pd.Timestamp("2026-12-18")
    assert (expiries.loc[~paid, "dividends_pv"] == 0).all() and (expiries.loc[paid, "dividends_pv"] > 1.8).all()
    assert np.allclose(
        expiries["forward"], (600.0 - expiries["dividends_pv"]) / expiries["discount_factor"], rtol=1e-14
    )
    F, K, DF = rows["forward"].to_numpy(), rows["strike"].to_numpy(), rows["discount_factor"].to_numpy()
    sd = rows["iv"].to_numpy() * np.sqrt(rows["maturity"].to_numpy())
    d1 = np.log(F / K) / sd + 0.5 * sd
    call = rows["type"].eq("call").to_numpy()
    european = np.where(call, DF * (F * ndtr(d1) - K * ndtr(d1 - sd)), DF * (K * ndtr(sd - d1) - F * ndtr(-d1)))
    value = rows["value"].to_numpy()
    assert np.all(value >= european - 2e-3)  # the tree's discretisation error is below a fifth of a cent
    assert np.all(value >= np.where(call, 600.0 - K, K - 600.0) - 1e-9)
    deep_put = (~call) & (K > 1.15 * 600) & (rows["maturity"] > 0.5).to_numpy()
    assert np.mean(value[deep_put] - european[deep_put]) > 0.5  # early exercise is worth something
    # A call just before an ex-date can be worth more than the European call to expiry (Black's approximation).
    assert np.any(call & (value > european + 0.05))


def test_deep_etf_puts_with_cash_dividends_are_worth_their_exercise_value(etf_chain, etf_one_percent):
    """Exercise pays the strike less the full share price. Exercising at the escrowed spot (the spot less the present
    value of the dividends to expiry) made every deep put worth at least ``K - S`` plus that present value.

    SPY-like dividends are smaller than the interest on the strike until the next ex-date, so a put deep in the money
    is exercised at once and worth exactly ``K - S``. With 1% of the spot a quarter, waiting for the next dividend
    pays near the money and exercising at once further out: the deep puts are worth ``K - S`` plus less than the
    present value of that one dividend, whatever the dividends after it.
    """
    for chain, spot in ((etf_chain, 600.0), (etf_one_percent, 100.0)):
        expiries = chain.expiries.set_index(["expiry", "root"])["dividends_pv"]
        rows = _with_expiry_data(chain).join(expiries, on=["expiry", "root"])
        puts = rows[rows["type"].eq("put") & (rows["dividends_pv"] > 0)]
        assert (puts["value"] >= puts["strike"] - spot - 1e-9).all()
        deep = puts[puts["strike"] >= 1.25 * spot]
        assert len(deep) > 20 and deep["expiry"].nunique() == puts["expiry"].nunique()
        premium = (deep["value"] - (deep["strike"] - spot)).to_numpy()
        if spot == 600.0:
            np.testing.assert_allclose(premium, 0.0, rtol=0, atol=1e-9)
        else:
            next_dividend = expiries[expiries > 0].min()  # the present value of the first dividend
            assert (premium >= 0).all() and (premium < next_dividend).all()
            assert (premium == 0).any() and (premium > 0.1).sum() >= 10  # exercised at once only beyond K = 154


def test_etf_puts_with_cash_dividends_are_the_escrowed_tree_at_the_clock_times(etf_one_percent):
    """Each put is Hull's escrowed tree at the quote's own clock times (the escrowed spot, the box rate and the model
    vol of its expiry, and every ex-date's time and present value from the clock), floored at the value of the same
    put with a shorter expiry, whose exercise it can copy. Without dividends the tree is the binomial engine's
    Leisen-Reimer tree to rounding."""
    spot, as_of = 100.0, clock.to_new_york(AS_OF)
    rows = _with_expiry_data(etf_one_percent).join(
        etf_one_percent.expiries.set_index(["expiry", "root"])[["settles_at", "dividends_pv"]], on=["expiry", "root"]
    )
    puts = rows[rows["type"].eq("put")].copy()
    for _, group in puts.groupby("expiry"):
        first = group.iloc[0]
        paid = []
        for ex_date, cash in ONE_PERCENT:
            drop = clock.to_new_york(datetime.combine(ex_date, clock.MARKET_OPEN))
            if drop <= first["settles_at"]:
                years = clock.year_fraction(as_of, drop)
                paid.append((years, cash * math.exp(-BOX * years)))
        escrowed = spot - sum(pv for _, pv in paid)
        assert first["dividends_pv"] == pytest.approx(spot - escrowed, rel=1e-14, abs=0)
        puts.loc[group.index, "tree"] = synthetic._escrowed_american_puts(
            escrowed, group["strike"].to_numpy(), first["maturity"], first["rate"], group["iv"].to_numpy(), paid
        )
    floored = puts.sort_values("expiry", kind="stable").groupby("strike")["tree"].cummax()
    np.testing.assert_allclose(puts["value"], floored.loc[puts.index], rtol=1e-13, atol=1e-12)
    lifted = puts["value"] - puts["tree"]
    assert 0 < lifted.max() < 0.05 and (lifted >= 0).all()  # the floor corrects the tree's time steps a little
    strikes, vols = np.array([60.0, 90.0, 100.0, 110.0, 160.0]), np.array([0.4, 0.25, 0.2, 0.18, 0.3])
    tree = synthetic._escrowed_american_puts(100.0, strikes, 1.3, BOX, vols, [])
    engine = [
        synthetic.binomial.price(100.0, K, 1.3, BOX, v, 0.0, "put", 201, True, "lr")
        for K, v in zip(strikes, vols, strict=True)
    ]
    np.testing.assert_allclose(tree, engine, rtol=0, atol=1e-10)


#: ETF chains with cash dividends, for QuantLib's date-based engines: (spot, dividends, the largest error allowed
#: against its escrowed-dividend engine, $).
_DIVIDEND_CHAINS = {
    "spy-like": (600.0, SPY_DIVIDENDS, 0.08),
    "one-percent": (100.0, ONE_PERCENT, 0.04),
    "1.7%-a-quarter": (600.0, [(date(2026, 12, 18), 10.0), (date(2027, 3, 19), 10.0)], 0.2),
}


def _whole_day_tree(row, spot: float, dividends) -> tuple[float, int, list[tuple[int, float]]]:
    """The escrowed tree for one put at whole-day times, its days to expiry and [(days to the ex-date, cash)]."""
    days = round(row.maturity * 365)
    paid = [((ex_date - AS_OF.date()).days, cash) for ex_date, cash in dividends if ex_date <= row.expiry.date()]
    pv = [cash * math.exp(-row.rate * d / 365) for d, cash in paid]
    times = [(d / 365, value) for (d, _), value in zip(paid, pv, strict=True)]
    strike, vol = np.array([row.strike]), np.array([row.iv])
    value = synthetic._escrowed_american_puts(spot - sum(pv), strike, days / 365, row.rate, vol, times)[0]
    return float(value), days, paid


def _whole_day_puts(chain: synthetic.SyntheticChain, spot: float, dividends) -> list[tuple]:
    """Puts with a dividend before expiry, at the strikes nearest 90%, 105% and 120% of the spot of every such expiry
    (out of the money, where early exercise starts to matter, and deep in the money), as (strike, days to expiry,
    box rate, model vol, [(days to the ex-date, cash)], half the quoted spread, the chain's value at those whole-day
    times: the tree, floored at the trees of the same put's shorter expiries)."""
    rows = _with_expiry_data(chain).join(
        chain.expiries.set_index(["expiry", "root"])["dividends_pv"], on=["expiry", "root"]
    )
    puts = rows[rows["type"].eq("put")]
    paying = puts[puts["dividends_pv"] > 0]
    nearest = [
        (group["strike"] - level * spot).abs().idxmin()
        for _, group in paying.groupby("expiry")
        for level in (0.9, 1.05, 1.2)
    ]
    sample = []
    for row in paying.loc[nearest].itertuples():
        value, days, paid = _whole_day_tree(row, spot, dividends)
        shorter = puts[(puts["strike"] == row.strike) & (puts["expiry"] < row.expiry)]
        value = max([value] + [_whole_day_tree(other, spot, dividends)[0] for other in shorter.itertuples()])
        sample.append((row.strike, days, row.rate, row.iv, paid, 0.5 * (row.ask - row.bid), value))
    return sample


def _quantlib_american_put(ql, spot, strike, days, rate, vol, paid, model) -> float:
    """QuantLib's finite-difference American put (800 x 800 grid) with cash dividends in ``model``."""
    today = ql.Date(AS_OF.day, AS_OF.month, AS_OF.year)
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, day_count)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, rate, day_count)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), vol, day_count)),
    )
    dividends = ql.DividendVector([today + d for d, _ in paid], [cash for _, cash in paid])
    option = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Put, strike), ql.AmericanExercise(today, today + days))
    scheme = ql.FdmSchemeDesc.Douglas()
    option.setPricingEngine(
        ql.FdBlackScholesVanillaEngine(process, dividends, 800, 800, 0, scheme, False, -ql.nullDouble(), model)
    )
    return option.NPV()


@pytest.mark.oracle
@pytest.mark.parametrize("case", list(_DIVIDEND_CHAINS))
def test_etf_puts_with_cash_dividends_match_quantlib_escrowed_finite_differences(case):
    """The chain's puts (Hull's escrowed tree with the shorter-expiry floor) against QuantLib's finite-difference
    engine in the same escrowed-dividend model (an independent method, on an 800 x 800 grid), at whole-day times:
    within half the quoted spread, and within $0.08, $0.04 and $0.20. The tree's 201 steps exercise up to a step
    after an ex-date, which costs most for deep puts at two years and large dividends: measured over every sixth put
    with a dividend before expiry, the largest errors are $0.062, $0.026 and $0.17."""
    ql = pytest.importorskip("QuantLib")
    spot, dividends, largest = _DIVIDEND_CHAINS[case]
    chain = sample_chain(spot, UST, dividends, as_of=AS_OF, style="etf", noise=False)
    for strike, days, rate, vol, paid, half_spread, value in _whole_day_puts(chain, spot, dividends):
        escrowed = ql.FdBlackScholesVanillaEngine.Escrowed
        theirs = _quantlib_american_put(ql, spot, strike, days, rate, vol, paid, escrowed)
        assert value == pytest.approx(theirs, abs=min(largest, half_spread)), (strike, days)


@pytest.mark.oracle
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="the escrowed-dividend model puts the vol on the spot less the dividends' present value, not on the share "
    "price; the planned dividends engine (spot-drop finite differences) replaces it",
)
def test_etf_puts_with_cash_dividends_match_a_spot_drop_model():
    """For the planned dividends engine: puts with a dividend before expiry within half the quoted spread of
    QuantLib's spot-drop finite-difference engine at the same vol. The escrowed model is up to $0.18 lower (about the
    gap between the two models' European values) and beyond half the spread on 40% of the SPY-like puts."""
    ql = pytest.importorskip("QuantLib")
    misses = []
    for spot, dividends, _ in _DIVIDEND_CHAINS.values():
        chain = sample_chain(spot, UST, dividends, as_of=AS_OF, style="etf", noise=False)
        for strike, days, rate, vol, paid, half_spread, value in _whole_day_puts(chain, spot, dividends):
            theirs = _quantlib_american_put(
                ql, spot, strike, days, rate, vol, paid, ql.FdBlackScholesVanillaEngine.Spot
            )
            if abs(value - theirs) > half_spread:
                misses.append((spot, strike, days, round(value - theirs, 4)))
    assert misses == []


@pytest.mark.parametrize("noise", [False, True], ids=["noise-free", "noisy"])
def test_quotes_have_no_static_arbitrage(noise, index_chain, index_model, etf_chain, etf_one_percent):
    """Acceptance (2): no quote-level arbitrage flags: the model-free vertical and butterfly check (both styles), the
    calendar check for the American ETF options (SPY-like dividends, and 1% of the spot a quarter), and the surface's
    own checks on the quotes (European index options)."""
    index = index_chain if noise else index_model
    assert _price_arbitrage(index.quotes) == 0
    etf = etf_chain if noise else sample_chain(600.0, UST, SPY_DIVIDENDS, as_of=AS_OF, style="etf", noise=False)
    one_percent = sample_chain(100.0, UST, ONE_PERCENT, as_of=AS_OF, style="etf") if noise else etf_one_percent
    for chain in (etf, one_percent):
        assert _price_arbitrage(chain.quotes) == 0
        assert _calendar_arbitrage(chain.quotes) == 0
        assert _value_drops_with_maturity(chain) == 0
    surface = _surface(index.quotes, AS_OF, BOX, YIELD)
    assert _arbitrage_notes(surface) == []


@pytest.mark.slow
@pytest.mark.parametrize(
    ("spot", "dividends"),
    [
        (600.0, 0.013),
        (600.0, [(date(2026, 12, 18), 6.0), (date(2027, 3, 19), 6.0)]),
        (600.0, [(date(2026, 12, 18), 10.0), (date(2027, 3, 19), 10.0)]),
        (50.0, [(date(2026, 12, 17), 1.0)]),
    ],
    ids=["yield", "1%-cash", "1.7%-cash", "2%-cash-low-price"],
)
def test_american_calendar_over_more_dividend_cases(spot, dividends):
    """The calendar check on the ETF chains with larger cash dividends and with a continuous yield, with and without
    noise: model values never fall with maturity, so the quotes cannot either."""
    for noise in (False, True):
        chain = sample_chain(spot, UST, dividends, as_of=AS_OF, style="etf", noise=noise)
        assert _value_drops_with_maturity(chain) == 0
        assert _calendar_arbitrage(chain.quotes) == 0
        assert _price_arbitrage(chain.quotes) == 0


def test_an_injected_calendar_spread_is_flagged(etf_one_percent):
    quotes = etf_one_percent.quotes.copy()
    calls = quotes[(quotes["type"] == "call") & (quotes["strike"] == 95.0)].sort_values("expiry")
    earlier, later = calls.index[3], calls.index[4]
    quotes.loc[earlier, ["bid", "ask"]] = quotes.loc[later, "ask"] + np.array([0.05, 0.10])
    assert _calendar_arbitrage(quotes) > 0


@pytest.mark.oracle
def test_etf_calls_with_cash_dividends_follow_blacks_approximation_on_the_surface(etf_one_percent):
    """Each call is worth the largest of its exercise value, the European to expiry and the European to just before
    each ex-date, the last priced at the forward and on the SSVI total variance of that moment (QuantLib's
    BlackCalculator for the European values; the ex-date quantities recomputed here from the clock)."""
    ql = pytest.importorskip("QuantLib")
    spot, box = 100.0, UST + synthetic.BOX_SPREAD
    as_of = clock.to_new_york(AS_OF)
    ex_dates = []
    for ex_date, cash in ONE_PERCENT:
        drop = clock.to_new_york(datetime.combine(ex_date, clock.MARKET_OPEN))
        t = clock.year_fraction(as_of, drop)
        ex_dates.append(
            (drop, cash * math.exp(-box * t), math.exp(-box * t), synthetic.DEFAULT_CLOCK.variance_time(as_of, drop))
        )
    rows = _with_expiry_data(etf_one_percent)
    rows = rows[rows["type"] == "call"].join(
        etf_one_percent.expiries.set_index(["expiry", "root"])["settles_at"], on=["expiry", "root"]
    )
    payoff = lambda K: ql.PlainVanillaPayoff(ql.Option.Call, K)  # noqa: E731
    before_ex_date = 0
    for row in rows.itertuples():
        candidates = [
            spot - row.strike,
            ql.BlackCalculator(
                payoff(row.strike), row.forward, row.iv * math.sqrt(row.maturity), row.discount_factor
            ).value(),
        ]
        for drop, _, discount, tau in ex_dates:
            if drop <= row.settles_at:
                forward = (spot - sum(pv for other, pv, _, _ in ex_dates if other < drop)) / discount
                w = synthetic.DEFAULT_SSVI.total_variance(math.log(row.strike / forward), tau)[0]
                candidates.append(ql.BlackCalculator(payoff(row.strike), forward, math.sqrt(w), discount).value())
        assert row.value == pytest.approx(max(candidates), rel=1e-11, abs=1e-12), (row.expiry, row.strike)
        before_ex_date += int(max(candidates[2:], default=-1.0) > max(candidates[:2]) + 1e-6)
    assert before_ex_date > 50  # the exercise just before an ex-date is what sets many of these values


def test_an_injected_butterfly_is_flagged(index_model, etf_chain):
    """Acceptance (2): the body of a butterfly bid above its wings is flagged by both checks."""
    quotes = index_model.quotes.copy()
    rows = _with_expiry_data(index_model)
    target = rows[(rows["expiry"] == "2026-12-18") & (rows["type"] == "call")]
    body = (target["strike"] - 1.02 * target["forward"]).abs().idxmin()
    quotes.loc[body, ["bid", "ask"]] += 20.0
    assert _price_arbitrage(quotes) > 0
    notes = _arbitrage_notes(_surface(quotes, AS_OF, BOX, YIELD))
    assert len(notes) == 1 and "butterfly" in notes[0] and "0.145y" in notes[0]
    etf = etf_chain.quotes.copy()
    body = etf[(etf["expiry"] == "2026-12-18") & (etf["type"] == "put") & (etf["strike"] == 590)].index
    etf.loc[body, ["bid", "ask"]] += 1.0
    assert _price_arbitrage(etf) > 0


#: The liquid core of a chain: quotes whose bid-ask half-spread is at most half a vol point. A desk fits an unweighted
#: smile to these until it weights each quote by its spread (SF-4).
MAX_VOL_HALF_SPREAD = 0.005
#: Valuation times for the fit: a Monday, the morning before a one-day and a six-hour expiry, and a late afternoon.
FIT_TIMES = (AS_OF, datetime(2026, 10, 29, 10, 0), datetime(2026, 10, 30, 10, 0), datetime(2026, 12, 7, 15, 30))


def _liquid_core(parsed: pd.DataFrame, spot: float, rate: float, dividend_yield: float) -> pd.DataFrame:
    """The parsed quotes whose half-spread in vol, the price error over the Black-Scholes vega at the quote's own
    implied vol, is at most :data:`MAX_VOL_HALF_SPREAD`. Chosen from the quotes and the sidebar inputs alone, never
    from the generator's forwards or vols."""
    K, T, iv = (parsed[column].to_numpy(float) for column in ("strike", "maturity", "iv"))
    sd = iv * np.sqrt(T)
    d1 = (np.log(spot / K) + (rate - dividend_yield) * T) / sd + 0.5 * sd
    vega = spot * np.exp(-dividend_yield * T) * np.exp(-0.5 * d1**2) / math.sqrt(2.0 * math.pi) * np.sqrt(T)
    return parsed[parsed["price_error"].to_numpy(float) <= MAX_VOL_HALF_SPREAD * vega].reset_index(drop=True)


def _atm_vol_misses(chain: synthetic.SyntheticChain, as_of: datetime, *, core: bool) -> list[str]:
    """The expiries whose fitted ATM vol is more than 0.1 vol point from the generator's, with the error."""
    parsed, _ = vs.parse_quotes(to_quotes(chain.quotes, as_of=as_of), SPOT, BOX, YIELD)
    if core:
        parsed = _liquid_core(parsed, SPOT, BOX, YIELD)
    surface = vs.fit_market_surface(parsed, SPOT, BOX, YIELD)
    assert len(surface.maturities) == len(chain.expiries)
    misses = []
    for row in chain.expiries.itertuples():
        error = surface.implied_vol_at(row.forward / SPOT, row.maturity) - row.atm_vol
        if not abs(error) <= 0.001:
            misses.append(f"{as_of:%Y-%m-%d %H:%M} {row.expiry:%Y-%m-%d}: {100 * error:+.3f} vol pt")
    return misses


@pytest.mark.parametrize("seed", [None, 0, 1, 2])
def test_fit_recovers_the_generator_atm_vols_from_the_liquid_core(seed):
    """Acceptance (3), the part the current fit can do: with the generator's rate and yield (the box rate and a
    continuous yield on a flat curve), parse_quotes and fit_market_surface on the liquid core of the chain recover
    every expiry's ATM vol within 0.1 vol point, for the model mids (seed None) and noisy quotes.

    The core (:func:`_liquid_core`) is chosen from the quotes alone. Measured over nine valuation times (these four;
    10:00 on 27 and 28 October; an hour before the 30 October close; half an hour before the 20 November AM
    settlement; noon on 1 March 2027) and five noise settings, the worst error is 0.038 vol point. Expiries within an
    hour of settlement keep fewer than five strikes in the core and drop out of the fit; none of these four times has
    one. The whole chain is the next test.
    """
    misses = []
    for as_of in FIT_TIMES:
        chain = sample_chain(SPOT, UST, YIELD, as_of=as_of, noise=seed is not None, seed=seed or 0)
        misses += _atm_vol_misses(chain, as_of, core=True)
    assert misses == []


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="needs spread-weighted calibration (SF-4): the unweighted SVI fit is pulled off the money by the cheap "
    "wing quotes, worth a few ticks with vol errors of several points",
)
def test_fit_recovers_the_generator_atm_vols_from_the_whole_chain():
    """Acceptance (3) on the whole chain, as uploaded: every quote with a bid, the generator's rate and yield.

    Measured on the four valuation times and the noise settings of the test above: the worst errors are 0.086, 0.86,
    1.9 and 0.095 vol point at the model mids, and 0.57, 1.4, 2.3 and 0.31 with noise, the largest at the expiries
    one and a quarter days and six hours away; 14 of the 16 runs miss 0.1 vol point somewhere, up to 7 of 17 expiries
    at once. Thirty minutes before an AM settlement the miss reaches 3.2 vol points.
    """
    misses = []
    for as_of in FIT_TIMES:
        for seed in (None, 0, 1, 2):
            chain = sample_chain(SPOT, UST, YIELD, as_of=as_of, noise=seed is not None, seed=seed or 0)
            misses += _atm_vol_misses(chain, as_of, core=False)
    assert misses == []


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="needs forwards and discount factors per expiry from put-call parity (SF-3): today the forward comes from "
    "the sidebar rate and dividend yield",
)
@pytest.mark.parametrize(
    ("rate_curve", "dividends"),
    [(UST, YIELD), (lambda T: 0.035 + 0.01 * T, [(date(2026, 12, 18), 20.0), (date(2027, 3, 19), 19.0)])],
    ids=["flat-yield", "curve-cash-dividends"],
)
def test_fit_recovers_forwards_and_atm_vols_with_the_sidebar_rate(rate_curve, dividends):
    """Acceptance (3), the part that needs SF-3: parsed with the sidebar's Treasury rate and no dividend yield, the
    surface fitted to the liquid core recovers each expiry's forward within 2e-4 and its ATM vol within 0.1 vol
    point."""
    chain = sample_chain(SPOT, rate_curve, dividends, as_of=AS_OF)
    parsed, _ = vs.parse_quotes(to_quotes(chain.quotes, as_of=AS_OF), SPOT, UST, 0.0)
    surface = vs.fit_market_surface(_liquid_core(parsed, SPOT, UST, 0.0), SPOT, UST, 0.0)
    log_forward = getattr(surface, "log_forward", None)  # SF-3 stores the forward of each maturity
    for i, row in enumerate(chain.expiries.itertuples()):
        forward = math.exp(log_forward[i]) if log_forward is not None else SPOT * math.exp(surface.carry * row.maturity)
        assert forward == pytest.approx(row.forward, rel=2e-4), row.expiry
        assert surface.implied_vol_at(row.forward / SPOT, row.maturity) == pytest.approx(row.atm_vol, abs=0.001)


def test_rate_curve_and_dividend_inputs():
    curve = sample_chain(SPOT, lambda T: 0.03 + 0.01 * T, YIELD, as_of=AS_OF, noise=False).expiries
    assert np.allclose(curve["rate"], 0.03 + 0.01 * curve["maturity"] + synthetic.BOX_SPREAD, rtol=0, atol=1e-15)
    no_box = sample_chain(SPOT, UST, YIELD, as_of=AS_OF, noise=False, box_spread=0.0).expiries
    assert np.allclose(no_box["rate"], UST)
    # A dividend that went ex before the valuation time is ignored; one after the last expiry changes nothing.
    paid = sample_chain(SPOT, UST, [(date(2026, 10, 1), 30.0), (date(2029, 1, 5), 30.0)], as_of=AS_OF, noise=False)
    assert (paid.expiries["dividends_pv"] == 0).all()
    with pytest.raises(ValueError, match="spot must be from"):
        sample_chain(-1.0, UST, YIELD, as_of=AS_OF)
    # A dividend yield may be any real number, numpy's included; text and booleans are refused.
    reference = sample_chain(SPOT, UST, YIELD, as_of=AS_OF, noise=False).quotes
    for value in (np.float64(YIELD), np.float32(YIELD)):
        assert np.allclose(sample_chain(SPOT, UST, value, as_of=AS_OF, noise=False).quotes["ask"], reference["ask"])
    pd.testing.assert_frame_equal(
        sample_chain(SPOT, UST, np.int64(0), as_of=AS_OF, noise=False).quotes,
        sample_chain(SPOT, UST, 0, as_of=AS_OF, noise=False).quotes,
    )
    for bad in ("0.013", True, None):
        with pytest.raises(TypeError, match="continuous yield"):
            sample_chain(SPOT, UST, bad, as_of=AS_OF)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="without events"):
        sample_chain(
            SPOT, UST, YIELD, as_of=AS_OF, variance_clock=clock.VarianceClock(events=[(date(2026, 11, 4), 0.02)])
        )
    for as_of in (datetime(1999, 12, 31, 10), datetime(2098, 1, 2, 10)):
        with pytest.raises(ValueError, match="as_of must be from 2000-01-01 to 2097-12-31"):
            sample_chain(SPOT, UST, YIELD, as_of=as_of)
    with pytest.raises(ValueError, match="style"):
        sample_chain(SPOT, UST, YIELD, as_of=AS_OF, style="future")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rate curve"):
        sample_chain(SPOT, lambda T: math.nan, YIELD, as_of=AS_OF)
    with pytest.raises(ValueError, match="dividend yield"):
        sample_chain(SPOT, UST, math.inf, as_of=AS_OF)
    with pytest.raises(ValueError, match="cash dividend"):
        sample_chain(SPOT, UST, [(date(2026, 12, 18), -1.0)], as_of=AS_OF)
    with pytest.raises(ValueError, match="pair"):
        sample_chain(SPOT, UST, [date(2026, 12, 18)], as_of=AS_OF)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="worth more than the spot"):
        sample_chain(100.0, UST, [(date(2026, 12, 18), 150.0)], as_of=AS_OF)


# --- Upload adapters (DT-5) ---------------------------------------------------------------------------------------

FIXTURE_LAYOUTS = {"long": "chain_long.csv", "yfinance": "chain_yfinance.csv", "cboe": "chain_cboe_wide.csv"}
FIXTURE_SPOT = 7915.0


@pytest.fixture(scope="module")
def fixture_chains() -> dict[str, tuple[pd.DataFrame, list[str]]]:
    return {layout: read_chain(FIXTURES / name, as_of=AS_OF) for layout, name in FIXTURE_LAYOUTS.items()}


def test_three_layouts_give_identical_cleaned_quotes(fixture_chains):
    """DT-5 acceptance (1): the same 40 fabricated contracts in three layouts give the same chain and quotes."""
    chains = {layout: chain for layout, (chain, _) in fixture_chains.items()}
    for chain in chains.values():
        validate_chain(chain)
        assert len(chain) == 36
    reference = chains["long"]
    for layout in ("yfinance", "cboe"):
        pd.testing.assert_frame_equal(chains[layout].drop(columns="vendor_iv"), reference.drop(columns="vendor_iv"))
        pd.testing.assert_frame_equal(to_quotes(chains[layout], as_of=AS_OF), to_quotes(reference, as_of=AS_OF))
    # The vendors' implied vols differ, as they do in practice; yfinance's placeholder is blanked.
    assert not np.allclose(chains["yfinance"]["vendor_iv"], reference["vendor_iv"], equal_nan=True)
    assert chains["yfinance"]["vendor_iv"].isna().sum() == 1 and reference["vendor_iv"].notna().all()


def test_expired_crossed_and_zero_bid_rows_are_dropped_with_one_message_each(fixture_chains):
    """DT-5 acceptance (3)."""
    for layout, (chain, messages) in fixture_chains.items():
        assert messages[0].startswith("Read 40 contract(s)"), layout
        expired = [m for m in messages if "expired" in m]
        crossed = [m for m in messages if "crossed" in m]
        zero_bid = [m for m in messages if "zero bid" in m]
        assert expired == ["Dropped 2 contract(s) that had settled by 2026-10-26 10:00 ET (expired)."]
        assert crossed == ["Dropped 1 contract(s) whose bid is above the ask (crossed)."]
        assert zero_bid == ["Dropped 1 contract(s) with a zero bid (offer only)."]
        assert not chain["symbol"].isin(["SPXW261106P07900000", "SPXW261102C08200000"]).any()
        assert (chain["expiry"] > pd.Timestamp(AS_OF.date())).all()
    assert any("Blanked 1 vendor implied vol" in m for m in fixture_chains["yfinance"][1])


def test_fit_is_unchanged_when_vendor_iv_is_shuffled(fixture_chains):
    """DT-5 acceptance (4): the vendor's implied vol is a diagnostic only."""
    rng = np.random.default_rng(4)
    for chain, spot in (
        (fixture_chains["cboe"][0], FIXTURE_SPOT),
        (sample_chain(SPOT, UST, YIELD, as_of=AS_OF).quotes, SPOT),
    ):
        fits = []
        for vendor_iv in (chain["vendor_iv"].to_numpy(), rng.permutation(chain["vendor_iv"]), rng.random(len(chain))):
            quotes, _ = vs.parse_quotes(to_quotes(chain.assign(vendor_iv=vendor_iv), as_of=AS_OF), spot, BOX, YIELD)
            fits.append(vs.fit_market_surface(quotes, spot, BOX, YIELD))
        for other in fits[1:]:
            np.testing.assert_array_equal(other.implied_vol, fits[0].implied_vol)
            np.testing.assert_array_equal(other.local_vol, fits[0].local_vol)
            assert other.svi == fits[0].svi and other.notes == fits[0].notes


def test_quotes_carry_clock_times_with_each_settlement_style(fixture_chains):
    chain, _ = fixture_chains["cboe"]
    quotes = to_quotes(chain, as_of=AS_OF)
    minutes = np.round(quotes["Maturity (years)"].to_numpy() * 525_600, 6)
    am = chain["root"].eq("SPX").to_numpy()
    assert set(chain.loc[am, "settlement"]) == {"AM"} and set(chain.loc[~am, "settlement"]) == {"PM"}
    # Monday 26 October 10:00 ET to Friday 20 November 09:30 ET (AM): 25 days less 30 minutes; to Monday 2 November
    # 16:00 ET (PM): 7 days and 6 hours. Wall-clock minutes: the end of daylight saving time on 1 November is not felt.
    assert set(minutes[am]) == {25 * 1_440 - 30}
    assert 7 * 1_440 + 360 in set(minutes[~am])
    parsed, messages = vs.parse_quotes(quotes, FIXTURE_SPOT, BOX, YIELD)
    assert not any(isinstance(message, vs.QuoteWarning) for message in messages)
    assert "maturity from 'Maturity (years)' (years)" in messages[0]
    assert sorted(parsed["maturity"].unique()) == pytest.approx(sorted(set(quotes["Maturity (years)"])), abs=1e-10)
    with pytest.raises(ValueError, match="had settled"):
        to_quotes(chain, as_of=datetime(2026, 11, 3, 10, 0))


def test_sources_and_layout_detection(fixture_chains):
    reference, _ = fixture_chains["yfinance"]
    path = FIXTURES / FIXTURE_LAYOUTS["yfinance"]
    data = path.read_bytes()
    for source in (str(path), data, io.BytesIO(data), io.StringIO(data.decode()), b"\xef\xbb\xbf" + data):
        pd.testing.assert_frame_equal(read_chain(source, as_of=AS_OF)[0], reference)
    # A table already read, with numbers rather than text, as yfinance returns it.
    lines = [line for line in data.decode().splitlines() if not line.startswith("#")]
    frame = pd.read_csv(io.StringIO("\n".join(lines)), index_col=0)
    pd.testing.assert_frame_equal(read_chain(frame, as_of=AS_OF, layout="yfinance")[0], reference)
    with pytest.raises(ValueError, match="looks like a yfinance-style export, not the cboe layout"):
        read_chain(path, as_of=AS_OF, layout="cboe")
    with pytest.raises(ValueError, match="not in a known chain layout"):
        read_chain(b"Strike,IV\n100,0.2\n", as_of=AS_OF)
    with pytest.raises(ValueError, match="layout must be"):
        read_chain(path, as_of=AS_OF, layout="excel")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="path, bytes"):
        read_chain(42, as_of=AS_OF)  # type: ignore[arg-type]


def test_cboe_layout_variants():
    """The older Cboe header (Vol, Open Int, dates as 11/02/2026) and a table pandas has already read."""
    text = (FIXTURES / FIXTURE_LAYOUTS["cboe"]).read_text(encoding="utf-8")
    reference, _ = read_chain(text.encode(), as_of=AS_OF)
    old = text.replace("Volume", "Vol").replace("Open Interest", "Open Int")
    for day, numeric in (("Mon Nov 02 2026", "11/02/2026"), ("Fri Nov 06 2026", "11/06/2026")):
        old = old.replace(day, numeric)
    pd.testing.assert_frame_equal(read_chain(old.encode(), as_of=AS_OF)[0], reference)
    lines = text.splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("Expiration Date"))
    frame = pd.read_csv(io.StringIO("\n".join(lines[header:])))  # repeated names become Bid.1, Ask.1, IV.1
    pd.testing.assert_frame_equal(read_chain(frame, as_of=AS_OF)[0], reference)


def test_long_layout_without_symbols_and_bad_rows():
    text = "\n".join(
        [
            "Expiry,Root,Type,Strike,Bid,Ask,Settlement",
            "2026-11-20,SPX,C,7900,101.5,103.0,",  # settlement from the root: AM
            "2026-11-20,SPXW,P,7900,88.0,89.5,PM",
            "2026-11-20,SPXW,call,,1.0,1.2,PM",  # no strike
            "2026-11-20,SPXW,put,7950,abc,1.2,PM",  # unreadable bid
            "2026-11-20,SPXW,put,7975,1.0,1.2,noon",  # unknown settlement
            "2026-11-20,SPXW,put,7990,1.0,0,PM",  # no offer
            "2026-11-20,SPXW,P,7900,88.1,89.6,PM",  # the same contract again
        ]
    )
    chain, messages = read_chain(text.encode(), as_of=AS_OF)
    assert chain[["symbol", "settlement"]].values.tolist() == [
        ["SPX261120C07900000", "AM"],
        ["SPXW261120P07900000", "PM"],
    ]
    assert chain["bid"].tolist() == [101.5, 88.0]
    assert "Dropped 3 contract(s) with a missing or unreadable symbol, expiry, type, strike, bid or ask." in messages
    assert "Dropped 1 contract(s) with no offer (an ask of zero)." in messages
    assert "Dropped 1 repeated contract(s), keeping the first quote of each." in messages
    # A symbol that disagrees with the row's own strike or type is dropped, not guessed at; blanks are not compared.
    mixed = "\n".join(
        [
            "symbol,root,expiry,type,strike,bid,ask",
            "SPXW261120C07900000,SPXW,2026-11-20,put,7900,1,2",
            "SPXW261120C07900000,SPXW,2026-11-20,call,7905,1,2",
            "SPXW261120C07910000,,,,,1,2",
        ]
    )
    chain, messages = read_chain(mixed.encode(), as_of=AS_OF)
    assert chain["symbol"].tolist() == ["SPXW261120C07910000"]
    assert "Dropped 2 contract(s) whose OCC symbol disagrees with their root, expiry, type or strike." in messages


@pytest.mark.parametrize(
    "bad_row",
    [
        "2026-11-20,^SPX,call,6000,1.0,1.2",  # a ticker, not an OCC root
        "2026-11-20,SPXWEEKLY,call,6000,1.0,1.2",  # seven characters
        "2026-11-20,SPX,call,-100,1.0,1.2",
        "2026-11-20,SPX,call,0,1.0,1.2",
        "2026-11-20,SPX,call,100.0004,1.0,1.2",  # off OCC's grid of 0.001
        "2026-11-20,SPX,call,1e6,1.0,1.2",  # beyond OCC's eight digits
        "1999-12-17,SPX,call,6000,1.0,1.2",  # before OCC's two-digit years and the NYSE calendar
        "2100-01-15,SPX,call,6000,1.0,1.2",  # OCC would write 2000
        "2101-01-20,SPX,call,6000,1.0,1.2",  # beyond the NYSE calendar too
        "2026-11-20,SPX,call,6000,-1.0,1.2",
        "2026-11-20,SPX,call,6000,1.0,-1.2",
    ],
)
def test_a_contract_an_occ_symbol_cannot_carry_is_dropped_not_fatal(bad_row):
    """One bad row is dropped with a message; the rest of the upload is read."""
    text = "\n".join(["expiry,root,type,strike,bid,ask", bad_row, "2026-11-20,SPX,put,6000,1.5,1.7"])
    chain, messages = read_chain(text.encode(), as_of=AS_OF)
    assert chain["symbol"].tolist() == ["SPX261120P06000000"]
    assert messages[0] == "Read 2 contract(s) in the app's long layout."
    assert len(messages) == 2 and messages[1].startswith("Dropped 1 contract(s) that an OCC symbol cannot carry")


def test_trailing_commas_encodings_and_empty_files():
    """Rows that end in a trailing comma (Cboe and long layouts), a Windows-1252 file (Excel on Western Windows), a
    file that is neither UTF-8 nor Windows-1252 (read as Latin-1), and a file with no table."""
    path = FIXTURES / FIXTURE_LAYOUTS["cboe"]
    reference, _ = read_chain(path, as_of=AS_OF)
    lines = path.read_text(encoding="utf-8").splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("Expiration Date"))
    for trailing in (lines[header + 1 :], lines[header:]):  # data rows only, or the header too
        text = "\n".join(lines[: len(lines) - len(trailing)] + [line + "," for line in trailing])
        pd.testing.assert_frame_equal(read_chain(text.encode(), as_of=AS_OF)[0], reference)
    windows = "\n".join(['"S&P 500 INDEX \u2013 SPX",Last: 7915.00', *lines[header - 1 :]]).encode("cp1252")
    with pytest.raises(UnicodeDecodeError):
        windows.decode("utf-8")
    pd.testing.assert_frame_equal(read_chain(windows, as_of=AS_OF)[0], reference)
    latin = b'"Index \x81\x8d",Last: 7915.00\n' + "\n".join(lines[header - 1 :]).encode()  # bytes cp1252 lacks
    pd.testing.assert_frame_equal(read_chain(latin, as_of=AS_OF)[0], reference)
    long = b"expiry,root,type,strike,bid,ask\n2026-11-20,SPX,call,6000,1.0,1.2,\n2026-11-20,SPX,put,6000,1.5,1.7,\n"
    chain, _ = read_chain(long, as_of=AS_OF)
    assert chain[["symbol", "bid", "ask"]].values.tolist() == [
        ["SPX261120C06000000", 1.0, 1.2],
        ["SPX261120P06000000", 1.5, 1.7],
    ]
    for empty in (b"", b"# Provenance: fabricated\n# nothing else\n"):
        with pytest.raises(ValueError, match="holds no table"):
            read_chain(empty, as_of=AS_OF)


def test_clean_chain_on_the_synthetic_chain(index_chain):
    later = datetime(2026, 11, 2, 10, 0)
    cleaned, messages = clean_chain(index_chain.quotes, as_of=later)
    validate_chain(cleaned)
    assert (cleaned["bid"] > 0).all()
    settled = index_chain.quotes["expiry"] <= pd.Timestamp("2026-10-30")
    assert messages[0] == f"Dropped {int(settled.sum())} contract(s) that had settled by 2026-11-02 10:00 ET (expired)."
    kept = index_chain.quotes[~settled & (index_chain.quotes["bid"] > 0)].reset_index(drop=True)
    pd.testing.assert_frame_equal(cleaned, kept)
    with pytest.raises(ValueError, match="lacks the column"):
        clean_chain(index_chain.quotes.drop(columns="settlement"), as_of=later)
