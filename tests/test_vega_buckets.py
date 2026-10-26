"""scenarios.vega_buckets and weighted_vega: where the vega sits, and what it is worth in one-month terms (RK-2)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.special import ndtri

from options_analyzer import scenarios as sc
from options_analyzer import strategies as S

MARKET = S.Market(100, 0.045, 0.18, 30 / 365, 0.013)
MULTIPLIER = 100
ONE_MONTH = 30 / 365
EXPIRY_LABELS = [
    "up_to_7d",
    "7d_to_14d",
    "14d_to_30d",
    "30d_to_60d",
    "60d_to_91d",
    "91d_to_182d",
    "182d_to_365d",
    "365d_to_730d",
    "over_730d",
]
DELTA_LABELS = ["10d_put", "25d_put", "atm", "25d_call", "10d_call"]
MONEYNESS_LABELS = [
    "up_to_80pct",
    "80pct_to_90pct",
    "90pct_to_95pct",
    "95pct_to_105pct",
    "105pct_to_110pct",
    "110pct_to_120pct",
    "over_120pct",
]


def _book() -> S.Strategy:
    """A representative 7-leg book: short front-month premium against long back-month vega, with a skew.

    Four legs sit exactly on default expiry edges (30, 182, 365 and 730 days).
    """
    return S.Strategy(
        "book",
        (
            S.Leg("put", -30, 98, 9 / 365, 0.20),
            S.Leg("call", -30, 102, 9 / 365, 0.16),
            S.Leg("call", -20, 100, 30 / 365, 0.175),
            S.Leg("put", -20, 100, 30 / 365, 0.175),
            S.Leg("put", 10, 95, 182 / 365, 0.22),
            S.Leg("call", 12, 100, 365 / 365, 0.18),
            S.Leg("call", 8, 105, 730 / 365, 0.18),
        ),
    )


def _cells(table):
    return table.drop(index="total", columns="total")


@pytest.mark.parametrize("strike_axis", ["delta", "moneyness"])
@pytest.mark.parametrize("edges", [sc.DEFAULT_EXPIRY_EDGES_DAYS, (30,), (5, 9, 10, 400)])
def test_buckets_sum_to_position_vega(strike_axis, edges):
    # RK-2 acceptance (1): the buckets sum to the position vega within 1e-10.
    book = _book()
    table = sc.vega_buckets(book, MARKET, edges, strike_axis, multiplier=MULTIPLIER)
    vega = S.position_risk(book, MARKET, multiplier=MULTIPLIER).vega
    cells = _cells(table)
    assert abs(cells.to_numpy().sum() - vega) <= 1e-10
    assert abs(table.loc["total", "total"] - vega) <= 1e-10
    assert np.allclose(table["total"].drop("total"), cells.sum(axis=1), rtol=0, atol=1e-12)
    assert np.allclose(table.loc["total"].drop("total"), cells.sum(axis=0), rtol=0, atol=1e-12)


def test_default_bucket_labels_are_stable():
    for axis, columns in (("delta", DELTA_LABELS), ("moneyness", MONEYNESS_LABELS)):
        table = sc.vega_buckets(_book(), MARKET, strike_axis=axis)
        assert list(table.index) == [*EXPIRY_LABELS, "total"]
        assert list(table.columns) == [*columns, "total"]
        assert (table.index.name, table.columns.name) == ("expiry_bucket", "strike_bucket")


def test_book_raw_and_one_month_vega():
    # The book looks long vega (+$387 per vol point) but loses money when one-month vol rises: its short
    # front-month legs dominate the one-month-equivalent vega. Measured values, locked against regressions.
    book = _book()
    raw = S.position_risk(book, MARKET, multiplier=MULTIPLIER).vega
    weighted = sc.weighted_vega(book, MARKET, multiplier=MULTIPLIER)
    sqrt_time = sc.weighted_vega(book, MARKET, scheme="sqrt_time", multiplier=MULTIPLIER)
    assert raw == pytest.approx(387.00, abs=0.005)
    assert weighted == pytest.approx(-18.857, abs=5e-4)
    assert sqrt_time == pytest.approx(-669.06, abs=0.005)
    assert raw > 0 > weighted


_UP = sc.StressScenario("1M vol +1%", 0.0, 1.01, 0.0, one_month_vol_shock=True)
_DOWN = sc.StressScenario("1M vol -1%", 0.0, 0.99, 0.0, one_month_vol_shock=True)


def _stress(strategy: S.Strategy, market: S.Market, multiplier: float = 1.0) -> tuple[float, float]:
    up, down = (row["P&L"] for row in sc.run_stress_tests(strategy, market, [_UP, _DOWN], multiplier=multiplier))
    return up, (up - down) / 2


def test_book_weighted_vega_is_the_one_month_stress_pnl():
    # RK-2 acceptance (2): weighted_vega equals the run_stress_tests P&L of
    # StressScenario(vol_multiplier=1.01, one_month_vol_shock=True) within 1%. Measured: -18.857 against -18.967
    # (0.58%); the difference is the volga term, which the symmetric +/-1% stress removes (to 1e-4).
    book = _book()
    weighted = sc.weighted_vega(book, MARKET, multiplier=MULTIPLIER)
    up, symmetric = _stress(book, MARKET, MULTIPLIER)
    assert weighted == pytest.approx(up, rel=0.01)
    assert weighted == pytest.approx(symmetric, rel=1e-4)


@pytest.mark.parametrize(
    "market", [S.Market(100, 0.05, 0.2, 0.25, 0.01), S.Market(100, 0.04, 0.3, 400 / 365)], ids=["3M", "400d"]
)
def test_template_weighted_vega_is_the_one_month_stress_pnl(market):
    # Where the legs' vegas nearly cancel the one-sided P&L is dominated by volga (a 3-month call spread: net
    # 0.0013 against a gross of 0.058, 1.1% apart), so the tolerance is 1% of the net plus 0.1% of the gross
    # (sum over legs of |weighted vega|); the symmetric stress agrees to 1e-4 of the net plus 1e-6 of the gross.
    for name in S.TEMPLATES:
        strategy = S.build_template(name, 100, 5, market.expiry)
        weighted = sc.weighted_vega(strategy, market)
        gross = sum(abs(sc.weighted_vega(S.Strategy("leg", (leg,)), market)) for leg in strategy.legs)
        up, symmetric = _stress(strategy, market)
        assert abs(weighted - up) <= 1e-2 * abs(weighted) + 1e-3 * gross, name
        assert abs(weighted - symmetric) <= 1e-4 * abs(weighted) + 1e-6 * gross, name


def _repriced_slope(strategy: S.Strategy, market: S.Market, shift_by_leg) -> float:
    """P&L per unit of a shock that moves each option leg's IV by ``shift_by_leg(sigma, tau)``, by explicit leg IVs.

    A symmetric difference at 1e-4 of the shock: its O(shock^2) error is about 1e-8 relative.
    """
    step = 1e-4

    def shocked(sign: float) -> float:
        legs = []
        for leg in strategy.legs:
            if leg.is_option:
                sigma = market.volatility if leg.volatility is None else leg.volatility
                tau = market.expiry if leg.expiry is None else leg.expiry
                leg = replace(leg, volatility=sigma + sign * step * shift_by_leg(sigma, tau))
            legs.append(leg)
        return float(S.position_value(S.Strategy("shocked", tuple(legs)), market))

    return (shocked(1.0) - shocked(-1.0)) / (2 * step)


def test_weighted_vega_schemes_match_explicit_repricing():
    # Independent of the damping code in position_value: move each leg's IV by hand. The relative scheme is the
    # P&L of a 1% relative one-month shock (leg IV moves 0.01 sigma w(tau)), the square-root-of-time scheme that of
    # one vol point of one-month IV (0.01 sqrt(1M / tau)).
    book = _book()
    decay = sc.DEFAULT_VOL_TENOR_DECAY
    relative = _repriced_slope(book, MARKET, lambda s, tau: 0.01 * s * min(1.0, (tau / ONE_MONTH) ** -decay))
    assert sc.weighted_vega(book, MARKET) == pytest.approx(relative, rel=1e-6)
    sqrt_time = _repriced_slope(book, MARKET, lambda s, tau: 0.01 * math.sqrt(ONE_MONTH / tau))
    assert sc.weighted_vega(book, MARKET, scheme="sqrt_time") == pytest.approx(sqrt_time, rel=1e-6)
    flat = _repriced_slope(book, MARKET, lambda s, tau: 0.01 * s)
    assert sc.weighted_vega(book, MARKET, decay=0.0) == pytest.approx(flat, rel=1e-6)


def test_one_month_legs_are_weighted_by_their_vol_only():
    # At a 30-day tenor (and below) the damping weight is 1: weighted vega = vega x sigma.
    for days in (1, 9, 30):
        leg = S.Leg("put", -3, 97, days / 365, 0.21)
        position = S.Strategy("leg", (leg,))
        vega = S.position_risk(position, MARKET).vega
        assert sc.weighted_vega(position, MARKET) == pytest.approx(vega * 0.21, rel=1e-14)


def _put_delta_strike(p: float, tau: float, vol: float, market: S.Market = MARKET) -> float:
    """Strike whose undiscounted put delta N(-d1) is p, from the closed form."""
    d1 = -float(ndtri(p))
    v = vol * math.sqrt(tau)
    return market.spot * math.exp(-d1 * v + (market.rate - market.dividend_yield) * tau + 0.5 * v * v)


@pytest.mark.parametrize(
    ("p", "bucket"),
    [
        (0.02, "10d_put"),
        (0.10, "10d_put"),
        (0.17, "10d_put"),
        (0.18, "25d_put"),
        (0.25, "25d_put"),
        (0.37, "25d_put"),
        (0.38, "atm"),
        (0.50, "atm"),
        (0.62, "atm"),
        (0.63, "25d_call"),
        (0.75, "25d_call"),
        (0.82, "25d_call"),
        (0.83, "10d_call"),
        (0.90, "10d_call"),
        (0.99, "10d_call"),
    ],
)
def test_delta_axis_uses_the_nearest_delta_pillar(p, bucket):
    for kind in ("call", "put"):  # a strike's bucket does not depend on the option type
        leg = S.Leg(kind, 1, _put_delta_strike(p, 60 / 365, 0.30), 60 / 365, 0.30)
        cells = _cells(sc.vega_buckets(S.Strategy("leg", (leg,)), MARKET))
        assert cells.loc["30d_to_60d", bucket] != 0.0
        assert np.count_nonzero(cells.to_numpy()) == 1


def test_delta_axis_uses_each_legs_own_iv():
    # A 25-delta put at its own 35% IV would be a 10-delta put at the 18% market vol.
    strike = _put_delta_strike(0.25, 45 / 365, 0.35)
    own = sc.vega_buckets(S.Strategy("own IV", (S.Leg("put", 1, strike, 45 / 365, 0.35),)), MARKET)
    market_vol = sc.vega_buckets(S.Strategy("market IV", (S.Leg("put", 1, strike, 45 / 365),)), MARKET)
    assert _cells(own).loc["30d_to_60d", "25d_put"] > 0
    assert _cells(market_vol).loc["30d_to_60d", "10d_put"] > 0


def test_a_single_leg_fills_exactly_one_cell():
    # RK-2 acceptance (3). (A deep in-the-money option half a day from expiry has a vega that underflows to exactly
    # zero; it then fills no cell.)
    days = [0.5, 7, 7.0001, 14, 21, 30, 45, 91, 182, 200, 365, 730, 1000]
    for axis in ("delta", "moneyness"):
        for kind in ("call", "put"):
            for strike in (60, 80, 90, 95, 99, 100, 105, 110, 120, 150):
                for d in days:
                    position = S.Strategy("leg", (S.Leg(kind, -2.5, strike, d / 365, 0.25),))
                    cells = _cells(sc.vega_buckets(position, MARKET, strike_axis=axis)).to_numpy()
                    vega = S.position_risk(position, MARKET).vega
                    assert np.count_nonzero(cells) == (1 if vega != 0 else 0), (axis, kind, strike, d)
                    assert cells.sum() == pytest.approx(vega, rel=1e-14)
                    assert d < 1 or vega != 0


def test_a_leg_on_a_bucket_edge_is_counted_once():
    # RK-2 acceptance (4): buckets are right-closed, so a leg exactly on an edge belongs to the bucket that edge
    # closes, also when its expiry was entered as days / 365 years.
    for edge, label in zip(sc.DEFAULT_EXPIRY_EDGES_DAYS, EXPIRY_LABELS, strict=False):
        position = S.Strategy("edge", (S.Leg("call", 1, 100, edge / 365),))
        cells = _cells(sc.vega_buckets(position, MARKET))
        assert np.count_nonzero(cells.to_numpy()) == 1
        assert cells.loc[label].sum() == pytest.approx(S.position_risk(position, MARKET).vega, rel=1e-14)
    for edge, label in zip(sc.DEFAULT_MONEYNESS_EDGES, MONEYNESS_LABELS, strict=False):
        position = S.Strategy("edge", (S.Leg("put", 1, edge * MARKET.spot),))
        cells = _cells(sc.vega_buckets(position, MARKET, strike_axis="moneyness"))
        assert np.count_nonzero(cells.to_numpy()) == 1
        assert cells[label].sum() != 0.0, (edge, label)


def test_book_lands_where_expected():
    table = sc.vega_buckets(_book(), MARKET, multiplier=MULTIPLIER)
    nonzero = {(i, j) for i in EXPIRY_LABELS for j in DELTA_LABELS if table.loc[i, j] != 0.0}
    assert nonzero == {
        ("7d_to_14d", "25d_put"),  # 9-day 98 put
        ("7d_to_14d", "25d_call"),  # 9-day 102 call
        ("14d_to_30d", "atm"),  # 30-day straddle
        ("91d_to_182d", "25d_put"),  # 182-day 95 put
        ("182d_to_365d", "atm"),  # 1-year 100 call
        ("365d_to_730d", "atm"),  # 2-year 105 call
    }
    assert table.loc["14d_to_30d", "atm"] == pytest.approx(
        2 * S.position_risk(S.Strategy("c", (_book().legs[2],)), MARKET, multiplier=MULTIPLIER).vega, rel=1e-14
    )


def test_shares_have_no_vega():
    book = _book()
    with_shares = S.Strategy("hedged", (*book.legs, S.Leg("stock", -40)))
    assert sc.vega_buckets(with_shares, MARKET).equals(sc.vega_buckets(book, MARKET))
    assert sc.weighted_vega(with_shares, MARKET) == sc.weighted_vega(book, MARKET)
    only_shares = sc.vega_buckets(S.Strategy("shares", (S.Leg("stock", 10),)), MARKET)
    assert (only_shares.to_numpy() == 0).all()


def test_multiplier_scales_buckets_and_weighted_vega():
    book = _book()
    one, many = sc.vega_buckets(book, MARKET), sc.vega_buckets(book, MARKET, multiplier=250)
    assert many.to_numpy() == pytest.approx(250 * one.to_numpy(), rel=1e-14)
    assert sc.weighted_vega(book, MARKET, multiplier=250) == pytest.approx(250 * sc.weighted_vega(book, MARKET))


def test_invalid_buckets_and_schemes_are_rejected():
    book = _book()
    with pytest.raises(ValueError, match="strike_axis"):
        sc.vega_buckets(book, MARKET, strike_axis="strike")
    with pytest.raises(ValueError, match="increase"):
        sc.vega_buckets(book, MARKET, expiry_edges=(30, 7))
    with pytest.raises(ValueError, match="positive"):
        sc.vega_buckets(book, MARKET, expiry_edges=())
    with pytest.raises(ValueError, match="positive"):
        sc.vega_buckets(book, MARKET, expiry_edges=(0, 7))
    with pytest.raises(ValueError, match="moneyness_edges"):
        sc.vega_buckets(book, MARKET, strike_axis="moneyness", moneyness_edges=(1.1, math.inf))
    with pytest.raises(ValueError, match="scheme"):
        sc.weighted_vega(book, MARKET, scheme="linear")
