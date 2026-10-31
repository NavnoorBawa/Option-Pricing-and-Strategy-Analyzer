"""The Strategy tab's legs without Streamlit (ui/legs.py): listed expiries, strikes by smile delta (RK-9a), the
structure steppers (UI-8), legs priced on the published surface (BK-3), the package market, and the risk-neutral
probability, moments, ranges and straddle move (UI-13). The default chain: ``sample_chain`` at Friday 23 October 2026,
16:00 ET (``_apptest.ASOF``), fitted and published as the app publishes it."""

from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import lognorm

from options_analyzer import clock
from options_analyzer import strategies as strat
from options_analyzer.ui import legs as lg
from options_analyzer.ui import market as mk
from options_analyzer.ui import surface_state
from options_analyzer.vol_surface.published import ONE_SD, TWO_SD, Expiry

AS_OF = datetime(2026, 10, 23, 16, 0)
DEC = Expiry(date(2026, 12, 18), "AM")
#: The default condor (spec 3.2, default state): +1 5,300P / −1 5,400P / −1 6,150C / +1 6,250C on 18‑Dec‑26 AM.
CONDOR = [("put", 1, 5300.0), ("put", -1, 5400.0), ("call", -1, 6150.0), ("call", 1, 6250.0)]


@pytest.fixture(scope="module")
def published():
    return surface_state.publish(mk.index_market(AS_OF))


def _condor(published) -> pd.DataFrame:
    return lg.index_template("Iron Condor", published, DEC)


def _legs(table: pd.DataFrame) -> list[tuple]:
    return [(row[lg.TYPE], row[lg.QTY], row[lg.STRIKE], row[lg.EXPIRY]) for _, row in table.iterrows()]


def _own_delta(published, k: float, kind: str) -> float:
    """Spot delta at the strike's own surface vol, written out here: e^{−qT}N(d₁) = F·DF/S·N(d₁) for a call."""
    F, DF, T = published.forward(DEC), published.discount_factor(DEC), published.maturity(DEC)
    sigma = published.vol(k, DEC).vol
    d1 = math.log(F / k) / (sigma * math.sqrt(T)) + 0.5 * sigma * math.sqrt(T)
    carry = F * DF / published.spot
    return carry * ndtr(d1) if kind == "call" else -carry * ndtr(-d1)


# --- Listed expiries ---------------------------------------------------------------------------------------------


def test_expiry_labels_round_trip():
    for day, style in ((date(2026, 12, 18), "AM"), (date(2027, 1, 8), "PM")):
        assert lg.parse_expiry(lg.expiry_text(day, style)) == Expiry(day, style)
    assert lg.parse_expiry("18-dec-26 am") is None  # the settlement is upper case, as the editor writes it
    assert lg.parse_expiry("31-Feb-27 AM") is None and lg.parse_expiry(None) is None


def test_the_default_expiry_is_the_monthly_nearest_45_days(published):
    # Spec 3.2 item 1: 18-Dec-26 AM at 55.7 days beats 20-Nov-26 AM at 27.7 days.
    assert lg.default_expiry(published) == DEC
    assert published.maturity(DEC) * 365 == pytest.approx(55.729, abs=1e-3)
    assert published.maturity(DEC) == clock.time_to_expiry(AS_OF, DEC.day, style="AM")


# --- Strikes by smile delta (RK-9a) ------------------------------------------------------------------------------


def test_the_short_strikes_are_the_listed_strikes_nearest_16_delta_at_their_own_vols(published):
    # Spec 3.2's unit test: the listed shorts are 5,400 (put) and 6,150 (call); no other listed strike's own-vol delta
    # is nearer ∓0.16; the unsnapped strikes solve Δ = ∓0.16 to 1e−8 at their own smile vols. The deltas are written
    # out in this test (Black-76 on the expiry's parity forward and discount factor), not read from the library.
    table = _condor(published)
    assert _legs(table) == [(kind, float(q), k, "18‑Dec‑26 AM") for kind, q, k in CONDOR]
    strikes = published.strikes(DEC)
    for kind, target, short in (("put", -0.16, 5400.0), ("call", 0.16, 6150.0)):
        deltas = np.array([_own_delta(published, k, kind) for k in strikes])
        distance = np.abs(deltas - target)
        assert strikes[int(np.nanargmin(distance))] == short
        assert np.sum(distance < abs(_own_delta(published, short, kind) - target)) == 0
        exact = brentq(lambda k, kind=kind, target=target: _own_delta(published, k, kind) - target, 5000, 6500)
        assert abs(_own_delta(published, exact, kind) - target) < 1e-8
        assert abs(exact - short) < 25  # within half the 50-point interval listed this far out
    # The spec's deltas were −0.150 and +0.147 on the fit before SF-4; this fit gives −0.1506 and +0.1445.
    assert _own_delta(published, 5400.0, "put") == pytest.approx(-0.1506, abs=5e-5)
    assert _own_delta(published, 6150.0, "call") == pytest.approx(0.1445, abs=5e-5)


@pytest.mark.parametrize(
    ("name", "kwargs", "expected"),
    [
        # 25Δ shorts at 5,550 and 6,045; the call wing 6,095 snaps to 6,100 (5-point strikes stop at 6,065).
        ("Iron Condor", {"short_delta": 0.25, "wing": 50.0}, [5500.0, 5550.0, 6045.0, 6100.0]),
        ("Short Strangle", {}, [5400.0, 6150.0]),
        ("Iron Butterfly", {"wing": 150.0}, [5680.0, 5830.0, 5830.0, 5980.0]),
        ("Bull Call Spread", {"wing": 100.0}, [5830.0, 5930.0]),
        # 5,830 ± 200 and ± 400: 5,430 and 6,230 are not listed (50-point strikes out there), so the outer strikes
        # go to the first listed strike beyond, 5,400 and 6,250, never nearer the inner ones than the width.
        ("Iron Condor", {"strikes_by": "width", "width": 200.0}, [5400.0, 5630.0, 6030.0, 6250.0]),
    ],
)
def test_templates_place_their_strikes_by_delta_or_width_on_listed_strikes(published, name, kwargs, expected):
    # Structures with a leg at the money centre on the listed strike nearest the forward (5,830 for F 5,827.95), the
    # others put their shorts by delta; wings are set in points; by width every strike is ATMF ± n × the width.
    table = lg.index_template(name, published, DEC, **kwargs)
    assert sorted(table[lg.STRIKE]) == expected
    listed = set(published.strikes(DEC))
    assert all(k in listed for k in table[lg.STRIKE])


def test_a_wing_never_lands_on_its_short_on_any_listed_expiry(published):
    # A 25-point wing snapped to the nearest listed strike where SPX lists every 50 put the condor's call wing on its
    # short call (+1 and −1 6,150C) and the put wing 50 points out; with 10Δ shorts on 17-Mar-28 100-point wings did
    # the same on 7,500. Every wing is now the first listed strike at least the wing width beyond its short, or the
    # template is refused (no listed strike there). Across every listed expiry, short delta and wing of the menus:
    # no template holds a contract twice, and each wing lies at least the width beyond its short.
    sides = {"Iron Condor": ((0, 1), (3, 2)), "Jade Lizard": ((2, 1),), "Iron Butterfly": ((0, 1), (3, 2))}
    refused = 0
    for key in lg.listed(published):
        listed = set(published.strikes(key))
        for name, pairs in sides.items():
            for delta in lg.SHORT_DELTAS:
                for wing in lg.WING_POINTS:
                    try:
                        table = lg.index_template(name, published, key, short_delta=delta / 100, wing=float(wing))
                    except ValueError as exc:
                        assert "lists no strike" in str(exc)
                        refused += 1
                        continue
                    contracts = list(zip(table[lg.TYPE], table[lg.STRIKE], table[lg.EXPIRY], strict=True))
                    assert len(set(contracts)) == len(contracts), (key, name, delta, wing, contracts)
                    strikes = list(table[lg.STRIKE])
                    assert set(strikes) <= listed
                    for outer, inner in pairs:
                        side = 1 if outer > inner else -1
                        assert side * (strikes[outer] - strikes[inner]) >= wing, (key, name, delta, wing, strikes)
    assert refused < 10  # only far wings past the last listed strike are refused
    # The menu offers 1 to 4 × the 50-point interval listed out of the money (spec 3.2 item 1).
    assert lg.WING_POINTS == (50, 100, 150, 200)


def test_a_time_spread_puts_its_back_leg_on_the_listed_expiry_nearest_twice_the_front(published):
    table = lg.index_template("Long Call Calendar", published, DEC)
    assert list(table[lg.EXPIRY]) == ["18‑Dec‑26 AM", "19‑Feb‑27 AM"]  # 55.7 and 118.7 days
    assert list(table[lg.QTY]) == [-1.0, 1.0] and set(table[lg.STRIKE]) == {5830.0}


def test_index_templates_hold_option_legs_only():
    names = lg.index_templates()
    assert "Iron Condor" in names and "Covered Call" not in names and "Synthetic Long Stock" in names
    assert all(all(leg.is_option for leg in strat.build_template(n, 100, 10, 1.0).legs) for n in names)


# --- Structure steppers (UI-8) ------------------------------------------------------------------------------------


def test_shift_moves_every_strike_one_listed_interval(published):
    grid = lg.Grid(published)
    up = lg.shift(_condor(published), 1, grid)
    assert list(up[lg.STRIKE]) == [5350.0, 5450.0, 6200.0, 6250.0 + 50.0]
    assert list(lg.shift(up, -1, grid)[lg.STRIKE]) == [k for _, _, k in CONDOR]


def test_width_moves_the_wings_only_and_never_crosses_the_body(published):
    grid, forward = lg.Grid(published), published.forward(DEC)
    out = lg.widen(_condor(published), 1, grid, forward)
    assert list(out[lg.STRIKE]) == [5250.0, 5400.0, 6150.0, 6300.0]  # the body stays
    table = _condor(published)
    for _ in range(10):
        table = lg.widen(table, -1, grid, forward)
    # The wings stop one listed interval outside the shorts: 5,350 and 6,200, never on or past 5,400 and 6,150.
    assert list(table[lg.STRIKE]) == [5350.0, 5400.0, 6150.0, 6200.0]


@pytest.mark.parametrize(
    ("name", "wings", "out", "inward"),
    [
        # A strangle has no body inside it: Width moves both strikes, symmetrically, and stops short of crossing.
        ("Short Strangle", {5400.0: -1, 6150.0: 1}, [5350.0, 6200.0], [5450.0, 6100.0]),
        # A jade lizard's short put stands alone: only the long call wing moves (the short put and call stay).
        ("Jade Lizard", {6250.0: 1}, [5400.0, 6150.0, 6300.0], [5400.0, 6150.0, 6200.0]),
        # A butterfly's long wings move one listed interval (5 points near the money); its body stays.
        ("Long Call Butterfly", {5730.0: -1, 5930.0: 1}, [5725.0, 5830.0, 5935.0], [5735.0, 5830.0, 5925.0]),
        # A vertical's far strike is its wing; a straddle (one strike) has none.
        ("Bull Call Spread", {5930.0: 1}, [5830.0, 5935.0], [5830.0, 5925.0]),
        ("Long Straddle", {}, [5830.0, 5830.0], [5830.0, 5830.0]),
    ],
)
def test_width_moves_the_wings_of_each_structure(published, name, wings, out, inward):
    # Spec 3.2 item 5: Width −/+ moves the wings only, never crossing the body. The wings were the lowest and highest
    # strikes whatever their sign, so Width + moved a jade lizard's short put and only one side of a strangle.
    grid, forward = lg.Grid(published), published.forward(DEC)
    table = lg.index_template(name, published, DEC)
    assert lg.wings(table, forward) == wings
    assert list(lg.widen(table, 1, grid, forward)[lg.STRIKE]) == out
    assert list(lg.widen(table, -1, grid, forward)[lg.STRIKE]) == inward
    for _ in range(40):
        table = lg.widen(table, -1, grid, forward)
    strikes = sorted(table[lg.STRIKE])
    assert len(set(strikes)) == len(set(lg.index_template(name, published, DEC)[lg.STRIKE]))  # nothing merged


def test_roll_moves_every_leg_to_the_next_listed_expiry_and_flip_negates(published):
    grid = lg.Grid(published)
    order = lg.listed(published)
    rolled = lg.roll(_condor(published), 1, grid)
    after = order[order.index(DEC) + 1]
    assert set(rolled[lg.EXPIRY]) == {lg.expiry_text(*after)}
    assert all(k in set(published.strikes(after)) for k in rolled[lg.STRIKE])
    assert lg.cells(lg.roll(rolled, -1, grid)) == lg.cells(_condor(published))
    assert list(lg.flip(_condor(published))[lg.QTY]) == [-1.0, 1.0, 1.0, -1.0]
    assert list(lg.flip(_condor(published), [0])[lg.QTY]) == [-1.0, -1.0, -1.0, 1.0]
    assert len(lg.duplicate(_condor(published), 1)) == 5 and len(lg.remove(_condor(published), 1)) == 3
    # The textbook market lists no expiries: nothing rolls.
    textbook = lg.textbook_template("Bull Call Spread", 100.0, 10.0, 1.0)
    assert lg.cells(lg.roll(textbook, 1, lg.Grid(None, step=10.0))) == lg.cells(textbook)


# --- Legs priced on the published surface (BK-3) ------------------------------------------------------------------


def test_index_legs_take_surface_vols_chain_quotes_and_enter_at_mid(published):
    book = lg.price_index(_condor(published), published, AS_OF, name="Iron Condor")
    assert not book.errors and book.front == DEC
    for p, (kind, q, k) in zip(book.legs, CONDOR, strict=True):
        quote = published.quote(DEC, k, kind)
        assert (p.kind, p.quantity, p.strike, p.src) == (kind, q, k, lg.SURFACE)
        assert p.vol == p.fit == published.vol(k, DEC).vol
        assert p.entry == quote.mid and p.quote == quote
        assert p.maturity == clock.time_to_expiry(AS_OF, DEC.day, style="AM")
    # The analysis market: Black-Scholes-Merton with the front's parity carry is Black-76 on F and DF.
    market = book.market
    T = market.expiry
    assert market.spot * math.exp((market.rate - market.dividend_yield) * T) == pytest.approx(
        published.forward(DEC), rel=1e-12
    )
    assert math.exp(-market.rate * T) == pytest.approx(published.discount_factor(DEC), rel=1e-12)


def test_an_override_an_unlisted_date_and_a_settled_leg_are_tagged(published):
    table = _condor(published)
    table.loc[0, lg.IV] = 25.0  # an override in percent
    table.loc[1, lg.EXPIRY] = "17‑Dec‑26 AM"  # a Thursday: not listed, read from the surface in total variance
    book = lg.price_index(table, published, AS_OF)
    assert (book.legs[0].src, book.legs[0].vol) == (lg.OVERRIDE, 0.25)
    assert book.legs[0].fit == published.vol(5300.0, DEC).vol  # the surface's vol stays the leg's fit
    assert book.legs[1].src == lg.UNLISTED and book.legs[1].quote is None
    # One day after the 18-Dec settlement every leg is expired: out of every number, its value intrinsic.
    later = lg.price_index(_condor(published), published, datetime(2026, 12, 21, 16, 0))
    assert {p.src for p in later.legs} == {lg.EXPIRED} and later.strategy is None
    assert [p.theo for p in later.legs] == [0.0, 0.0, 0.0, 0.0]  # out of the money at the same spot


def test_invalid_rows_give_one_line_each(published):
    table = _condor(published)
    table.loc[0, lg.QTY] = 0.0
    table.loc[1, lg.EXPIRY] = "next month"
    table.loc[2, lg.STRIKE] = -5.0
    book = lg.price_index(table, published, AS_OF)
    assert [e.split(":")[0] for e in book.errors] == ["Row 1", "Row 2", "Row 3"]
    assert [p.row for p in book.legs] == [3]


def test_the_package_market_is_the_natural_bid_and_ask(published):
    # Spec 3.2 item 2: selling the condor receives Σ shorts' bids − Σ longs' asks and pays back Σ shorts' asks − Σ
    # longs' bids, per share: 20.80 / 23.90 (the spec's 20.80 / 24.00 was the generator before SF-4).
    book = lg.price_index(_condor(published), published, AS_OF)
    q = {(kind, k): published.quote(DEC, k, kind) for kind, _, k in CONDOR}
    bid = q["put", 5400.0].bid + q["call", 6150.0].bid - q["put", 5300.0].ask - q["call", 6250.0].ask
    ask = q["put", 5400.0].ask + q["call", 6150.0].ask - q["put", 5300.0].bid - q["call", 6250.0].bid
    package = lg.package(book)
    assert package.credit and (package.bid, package.ask) == (pytest.approx(bid), pytest.approx(ask))
    assert (round(package.bid, 2), round(package.ask, 2), round(package.mid, 2)) == (20.80, 23.90, 22.35)
    # Mid vs fit in vp of the package vega: the parallel vol move that takes the theo to the mid (−0.04 vp here).
    signed_mid, signed_theo = -package.mid, -package.theo
    assert package.mid_vs_fit == pytest.approx((signed_mid - signed_theo) / package.vega, rel=1e-12)
    assert package.vega < 0 and package.mid_vs_fit == pytest.approx(-0.0417, abs=5e-4)
    # Bought, the same legs are a debit quoted the same: its bid is Σ longs' bids − Σ shorts' asks, which are the
    # credit's shorts and longs.
    flipped = lg.package(lg.price_index(lg.flip(_condor(published)), published, AS_OF))
    assert not flipped.credit and (flipped.bid, flipped.ask) == (pytest.approx(bid), pytest.approx(ask))
    assert flipped.mid_vs_fit == pytest.approx(package.mid_vs_fit, rel=1e-12)  # the market's richness is the same


# --- Risk-neutral probability, moments, ranges and the straddle move (UI-13) ------------------------------------


def test_profit_intervals_and_probability_under_a_known_distribution():
    # A long straddle struck at 100 for 10 profits outside (90, 110): under a log-normal with median 100 and
    # σ√T = 0.2 the probability is 1 − (F(110) − F(90)), from SciPy.
    def pnl(spots):
        return np.abs(spots - 100.0) - 10.0

    intervals = lg.profit_intervals(pnl, [90.0, 110.0])
    assert intervals == [(0.0, 90.0), (110.0, math.inf)]
    law = lognorm(s=0.2, scale=100.0)
    assert lg.probability(law.cdf, intervals) == pytest.approx(1 - (law.cdf(110) - law.cdf(90)), rel=1e-12)
    # Moments by the quantile midpoint rule against SciPy's numerical expectation (4,000 points: 1e−3 relative).
    mean, sd = lg.rn_moments(pnl, law.ppf)
    assert mean == pytest.approx(law.expect(lambda s: abs(s - 100.0) - 10.0), rel=1e-3)
    second = law.expect(lambda s: (abs(s - 100.0) - 10.0) ** 2)
    assert sd == pytest.approx(math.sqrt(second - mean**2), rel=1e-3)


def test_the_flat_vol_cdf_is_the_log_normal_of_the_textbook_market():
    cdf = lg.lognormal_cdf(100.0, 105.127, 0.2, 1.0)
    law = lognorm(s=0.2, scale=105.127 * math.exp(-0.02))
    for x in (60.0, 100.0, 150.0):
        assert cdf(x) == pytest.approx(law.cdf(x), rel=1e-12)
    assert cdf(0.0) == 0.0


@pytest.mark.oracle
def test_the_default_condors_probability_of_profit_from_the_smile(published):
    # P(profit) at the front expiry = P(5,377.65 < S_T < 6,172.35) under the smile's risk-neutral law, here by
    # Breeden-Litzenberger on QuantLib's Black formula at the surface's vols (central differences of 0.01 points,
    # truncation error about 1e−9), against the library's analytic CDF: equal to 1e−7. 74.05%, against the spec's
    # 73.5% on the fit before SF-4 (within its 1 pp).
    ql = pytest.importorskip("QuantLib")
    F, DF, T = published.forward(DEC), published.discount_factor(DEC), published.maturity(DEC)

    def call(k):
        sigma = published.vol(k, DEC).vol
        return ql.blackFormula(ql.Option.Call, k, F, sigma * math.sqrt(T), DF)

    def cdf(k, h=0.01):
        return 1.0 + (call(k + h) - call(k - h)) / (2 * h) / DF

    book = lg.price_index(_condor(published), published, AS_OF)
    credit = -sum(p.quantity * p.entry for p in book.legs)
    assert credit == pytest.approx(22.35, abs=1e-9)
    low, high = 5400.0 - credit, 6150.0 + credit
    for k in (low, high):
        assert published.cdf(k, DEC) == pytest.approx(cdf(k), abs=1e-7)
    p = cdf(high) - cdf(low)
    assert p == pytest.approx(0.7405, abs=5e-5) and abs(p - 0.735) < 0.01
    # The ranges of the chart's legend: the quantiles at (1 ∓ 68.27%)/2 and (1 ∓ 95.45%)/2.
    for level, (lo, hi) in ((ONE_SD, (5490.69, 6161.92)), (TWO_SD, (4893.66, 6387.68))):
        bounds = published.rn_range(DEC, level)
        assert bounds == (pytest.approx(lo, abs=0.01), pytest.approx(hi, abs=0.01))
        assert cdf(bounds[0]) == pytest.approx((1 - level) / 2, abs=1e-7)
        assert cdf(bounds[1]) == pytest.approx((1 + level) / 2, abs=1e-7)


def test_the_straddle_implied_move_is_the_atm_straddle_at_mid_over_spot(published):
    # The 18-Dec 5,830 straddle (the listed strike nearest F 5,827.95) at mid: 279.65, a move of ±4.82% (the spec's
    # 280.30 and ±4.83% were the generator before SF-4).
    move, k = lg.straddle_move(published, DEC)
    call, put = published.quote(DEC, 5830.0, "call"), published.quote(DEC, 5830.0, "put")
    assert k == 5830.0 and call.mid + put.mid == pytest.approx(279.65, abs=1e-9)
    assert move == pytest.approx(279.65 / 5800.0, rel=1e-12)
