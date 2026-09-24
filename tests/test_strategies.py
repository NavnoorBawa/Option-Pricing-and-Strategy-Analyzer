import numpy as np
import pytest
from scipy.stats import norm

from options_analyzer import black_scholes as bs
from options_analyzer import strategies as S

MARKET = S.Market(spot=100, rate=0.05, volatility=0.2, expiry=1.0, dividend_yield=0.0)
CALL = bs.price(100, 100, 1, 0.05, 0.2, 0, "call")
PUT = bs.price(100, 100, 1, 0.05, 0.2, 0, "put")


def test_every_template_builds_and_analyses():
    for name in S.TEMPLATES:
        strategy = S.build_template(name, 100, 10, MARKET.expiry)
        metrics = S.analyze(strategy, MARKET)
        assert np.isfinite(metrics.entry_cost)
        assert 0.0 <= metrics.probability_of_profit <= 1.0
        # P&L today at the current spot is zero because entry prices are model values.
        assert float(S.pnl(strategy, MARKET)) == pytest.approx(0.0, abs=1e-9)
        for be in metrics.breakevens:
            assert float(S.pnl(strategy, MARKET, be, metrics.horizon)) == pytest.approx(0.0, abs=1e-6)


def test_long_call_analytics_match_closed_form():
    m = S.analyze(S.build_template("Long Call", 100, 10), MARKET)
    be = 100 + CALL
    assert m.entry_cost == pytest.approx(CALL)
    assert m.breakevens == pytest.approx([be])
    assert np.isinf(m.max_profit) and m.max_loss == pytest.approx(CALL)
    d2 = (np.log(100 / be) + (0.05 - 0.02) * 1) / 0.2
    assert m.probability_of_profit == pytest.approx(norm.cdf(d2), abs=1e-12)
    # Risk-neutral expected payoff is the forward value of the premium.
    assert m.expected_pnl == pytest.approx(np.exp(0.05) * CALL - CALL, abs=1e-9)


def test_vertical_spread_bounds():
    m = S.analyze(S.build_template("Bull Call Spread", 100, 10), MARKET)
    debit = CALL - bs.price(100, 110, 1, 0.05, 0.2)
    assert m.entry_cost == pytest.approx(debit)
    assert m.max_profit == pytest.approx(10 - debit)
    assert m.max_loss == pytest.approx(debit)
    assert m.reward_to_risk == pytest.approx((10 - debit) / debit)


def test_straddle_has_two_breakevens_and_unlimited_profit():
    m = S.analyze(S.build_template("Long Straddle", 100, 10), MARKET)
    assert m.breakevens == pytest.approx([100 - CALL - PUT, 100 + CALL + PUT])
    assert np.isinf(m.max_profit)
    short = S.analyze(S.build_template("Short Straddle", 100, 10), MARKET)
    assert np.isinf(short.max_loss) and short.max_profit == pytest.approx(CALL + PUT)


def test_covered_call_payoff_and_greeks():
    strategy = S.build_template("Covered Call", 100, 10)
    spots = np.array([50.0, 100.0, 150.0])
    expected = np.minimum(spots, 100) - 100 + CALL
    assert S.pnl(strategy, MARKET, spots, 1.0) == pytest.approx(expected)
    g = S.position_greeks(strategy, MARKET)
    assert g["delta"] == pytest.approx(1 - bs.greeks(100, 100, 1, 0.05, 0.2)["delta"])


def test_iron_condor_is_symmetric_and_bounded():
    strategy = S.build_template("Iron Condor", 100, 5)
    assert strategy.strikes == [90, 95, 105, 110]
    m = S.analyze(strategy, MARKET)
    assert np.isfinite(m.max_profit) and np.isfinite(m.max_loss)
    assert m.max_profit + m.max_loss == pytest.approx(5.0)


def test_calendar_spread_uses_leg_expiries():
    strategy = S.build_template("Long Call Calendar", 100, 10, 0.25)
    assert [leg.expiry for leg in strategy.legs] == [0.25, 0.5]
    assert S.front_expiry(strategy, MARKET) == 0.25
    m = S.analyze(strategy, MARKET)
    assert np.isfinite(m.max_profit) and m.max_loss == pytest.approx(m.entry_cost, rel=1e-6)
    assert len(m.breakevens) == 2


def test_premium_override_and_fees_shift_pnl():
    leg = S.Leg("call", 1, 100, premium=CALL - 1.0)
    strategy = S.Strategy("cheap fill", (leg,), fees=0.01)
    assert S.entry_cost(strategy, MARKET) == pytest.approx(CALL - 0.99)
    assert float(S.pnl(strategy, MARKET)) == pytest.approx(0.99)


def test_dividends_keep_stock_and_options_consistent():
    market = S.Market(100, 0.05, 0.2, 1.0, dividend_yield=0.03)
    synthetic = S.Strategy("syn", (S.Leg("call", 1, 100), S.Leg("put", -1, 100)))
    stock = S.Strategy("stock", (S.Leg("stock", 1),))
    # Synthetic forward and stock have the same delta to first order after adjusting for dividends.
    assert S.position_greeks(synthetic, market)["delta"] == pytest.approx(np.exp(-0.03))
    assert S.position_greeks(stock, market)["delta"] == pytest.approx(1.0)


def test_template_width_validation():
    with pytest.raises(ValueError, match="too wide"):
        S.build_template("Iron Condor", 100, 50)
    with pytest.raises(ValueError):
        S.build_template("Bull Call Spread", 100, 0)


def test_leg_validation():
    with pytest.raises(ValueError):
        S.Leg("future", 1, 100)
    with pytest.raises(ValueError):
        S.Leg("call", 1, 0)
    with pytest.raises(ValueError):
        S.Leg("put", 1, 100, expiry=-1)
    with pytest.raises(ValueError):
        S.Market(0, 0.05, 0.2, 1)
    assert S.Leg("CALL", -2, 105).label() == "Short 2 × 105 Call"


def test_vectorised_scenario_valuation():
    strategy = S.build_template("Iron Butterfly", 100, 10)
    grid = S.position_value(
        strategy, MARKET, np.array([[90.0, 100.0, 110.0]]), 0.0, vol_multiplier=np.array([[0.5], [1.0], [2.0]])
    )
    assert grid.shape == (3, 3)
    assert grid[1, 1] == pytest.approx(S.entry_cost(strategy, MARKET))


def test_real_world_drift_moves_probability():
    strategy = S.build_template("Long Call", 100, 10)
    low = S.analyze(strategy, MARKET, drift=-0.10).probability_of_profit
    high = S.analyze(strategy, MARKET, drift=0.30).probability_of_profit
    assert low < high


def test_leg_table_rows():
    rows = S.leg_table(S.build_template("Collar", 100, 10), MARKET)
    assert [r["Type"] for r in rows] == ["Stock", "Put", "Call"]
    assert rows[0]["Entry price"] == 100


@pytest.mark.parametrize("name", list(S.TEMPLATES))
def test_financed_risk_neutral_expected_pnl_is_zero(name):
    # Martingale check of the whole valuation stack (options, total-return stock, multi-expiry legs).
    market = S.Market(100, 0.05, 0.25, 0.5, dividend_yield=0.02)
    metrics = S.analyze(S.build_template(name, 100, 10, market.expiry), market, carry=True)
    assert metrics.expected_pnl == pytest.approx(0.0, abs=1e-8)
