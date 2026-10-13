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


# --- Regression tests for findings of the independent review of the rebuilt engine ---------------


@pytest.mark.parametrize("quantity", [1, -1])
def test_far_out_of_the_money_option_with_tiny_premium(quantity):
    market = S.Market(100, 0.05, 0.10, 7 / 365)
    strategy = S.Strategy("lottery", (S.Leg("call", quantity, 110),))
    m = S.analyze(strategy, market)
    assert m.breakevens == pytest.approx([110.0])
    expected = 4.55e-12 if quantity > 0 else 1.0 - 4.55e-12
    assert m.probability_of_profit == pytest.approx(expected, rel=0.05, abs=1e-12)


def test_covered_call_profit_stays_capped_with_dividends():
    market = S.Market(100, 0.05, 0.2, 1.0, dividend_yield=0.02)
    m = S.analyze(S.build_template("Covered Call", 100, 10), market)
    call = bs.price(100, 100, 1, 0.05, 0.2, 0.02)
    dividends = 100 * (np.exp(0.05) - np.exp(0.03))
    assert m.max_profit == pytest.approx(call + dividends)  # K - S0 + C + D with K = S0
    assert np.isinf(S.analyze(S.build_template("Protective Put", 100, 10), market).max_profit)


@pytest.mark.parametrize("name", ["Covered Call", "Protective Put", "Collar", "Long Call", "Iron Condor"])
def test_position_theta_matches_finite_difference_with_dividends(name):
    market = S.Market(100, 0.05, 0.2, 1.0, dividend_yield=0.03)
    strategy = S.build_template(name, 100, 10)
    h = 1e-6
    numeric = float(S.position_value(strategy, market, None, h) - S.position_value(strategy, market)) / h
    assert S.position_greeks(strategy, market)["theta"] == pytest.approx(numeric, rel=1e-4, abs=1e-6)


def test_zero_pnl_plateau_reports_both_ends_and_exact_probability():
    rr = S.Strategy("zero-cost risk reversal", (S.Leg("put", -1, 90, premium=3.5), S.Leg("call", 1, 110, premium=3.5)))
    m = S.analyze(rr, MARKET)
    assert m.breakevens == pytest.approx([90.0, 110.0])
    d2 = (np.log(100 / 110) + (0.05 - 0.02) * 1) / 0.2
    assert m.probability_of_profit == pytest.approx(norm.cdf(d2), abs=1e-10)
    zero_debit = S.Strategy("free spread", (S.Leg("call", 1, 100, premium=5.0), S.Leg("call", -1, 110, premium=5.0)))
    assert S.analyze(zero_debit, MARKET).probability_of_profit == pytest.approx(norm.cdf(0.15), abs=1e-10)


def test_extreme_reached_only_as_spot_goes_to_infinity():
    market = S.Market(100, 0.05, 0.10, 30 / 365)
    strategy = S.Strategy("diagonal puts", (S.Leg("put", 2, 119, 400 / 365, 0.6), S.Leg("put", -2, 112)))
    m = S.analyze(strategy, market)
    assert m.max_loss == pytest.approx(42.905857, rel=1e-6)
    assert m.max_loss_spot is None


@pytest.mark.parametrize(("vol", "years"), [(0.2, 1.0), (1.5, 10.0), (5.0, 10.0)])
def test_expected_pnl_quadrature_at_extreme_volatility(vol, years):
    market = S.Market(100, 0.05, vol, years)
    call = bs.price(100, 100, years, 0.05, vol)
    m = S.analyze(S.build_template("Long Call", 100, 10), market)
    assert m.expected_pnl == pytest.approx(np.exp(0.05 * years) * call - call, rel=1e-9)


def test_riskless_box_has_no_loss():
    box = S.Strategy("box", (S.Leg("call", 1, 90), S.Leg("call", -1, 110), S.Leg("put", 1, 110), S.Leg("put", -1, 90)))
    m = S.analyze(box, MARKET)
    assert m.max_loss < 0 and m.reward_to_risk is None
    assert m.probability_of_profit == 1.0


def test_random_positions_match_monte_carlo():
    rng = np.random.default_rng(1)
    for _ in range(15):
        legs = []
        for _ in range(rng.integers(1, 4)):
            kind = str(rng.choice(["call", "put", "stock"]))
            legs.append(
                S.Leg(kind, float(rng.choice([-2, -1, 1, 2])), float(rng.uniform(70, 130)) if kind != "stock" else 0.0)
            )
        market = S.Market(100, 0.03, float(rng.uniform(0.1, 0.6)), float(rng.uniform(0.05, 1.5)), 0.02)
        strategy = S.Strategy("random", tuple(legs))
        m = S.analyze(strategy, market)
        z = rng.standard_normal(200_000)
        t = market.expiry
        spots = 100 * np.exp((0.03 - 0.02 - 0.5 * market.volatility**2) * t + market.volatility * np.sqrt(t) * z)
        outcomes = S.pnl(strategy, market, spots, t)
        assert m.probability_of_profit == pytest.approx((outcomes > 0).mean(), abs=0.005)
        assert m.expected_pnl == pytest.approx(outcomes.mean(), abs=5 * outcomes.std() / np.sqrt(z.size))


def test_time_spreads_require_an_expiry():
    with pytest.raises(ValueError, match="expiry"):
        S.build_template("Long Call Calendar", 100, 10)


# --- Second review round ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "vol", "days", "expected"),
    [("Bull Call Spread", 1.2, 3650, 1), ("Long Call Butterfly", 1.0, 3650, 2), ("Iron Condor", 3.0, 365, 2)],
)
def test_no_spurious_far_breakevens_at_extreme_volatility(name, vol, days, expected):
    # The grid used to reach 1e15 x spot, where rounding turned a constant P&L into fake sign changes.
    market = S.Market(100, 0.05, vol, days / 365)
    metrics = S.analyze(S.build_template(name, 100, 10), market)
    assert len(metrics.breakevens) == expected
    assert max(metrics.breakevens) < 200


def test_tiny_debit_far_out_is_not_a_breakeven():
    # A butterfly at 500% vol for 5 years costs ~1e-10: its far P&L is below the rounding tolerance.
    market = S.Market(100, 0.05, 5.0, 5.0)
    assert len(S.analyze(S.build_template("Long Call Butterfly", 100, 10), market).breakevens) == 2


@pytest.mark.parametrize(("q", "root"), [(0.01, 958.501), (0.005, 1798.998)])
def test_breakeven_beyond_the_grid_is_found(q, root):
    # A dividend-paying diagonal loses slowly far above the strikes; the upper breakeven used to be dropped.
    market = S.Market(100, 0.05, 0.2, 1.0, q)
    diagonal = S.build_template("Diagonal (Poor Man's Covered Call)", 100, 10, market.expiry)
    metrics = S.analyze(diagonal, market)
    assert len(metrics.breakevens) == 2 and np.isinf(metrics.max_loss)
    assert metrics.breakevens[1] == pytest.approx(root, abs=1e-3)
    assert float(S.pnl(diagonal, market, metrics.breakevens[1], metrics.horizon)) == pytest.approx(0, abs=1e-9)
    assert S.unbounded_loss_from_dividends(diagonal, market)


def test_unbounded_loss_note_only_for_dividend_drag():
    market = S.Market(100, 0.05, 0.2, 30 / 365, 0.01)
    naked = S.Strategy("naked", (S.Leg("call", -1, 105), S.Leg("put", 1, 95, 60 / 365)))
    assert np.isinf(S.analyze(naked, market).max_loss)
    assert not S.unbounded_loss_from_dividends(naked, market)


def test_riskless_conversion_has_no_dispersion():
    market = S.Market(5000, 0.05, 0.2, 0.5, 0.02)
    conversion = S.Strategy("conv", (S.Leg("stock", 1), S.Leg("put", 1, 5000), S.Leg("call", -1, 5000)))
    assert S.analyze(conversion, market).pnl_std < 1e-9


def test_leg_labels_show_expiries_only_when_they_tell_legs_apart():
    # Review finding: a calendar read "Short 1 × 100 Call, Long 1 × 100 Call", like a flat position.
    market = S.Market(100, 0.05, 0.2, 30 / 365)
    calendar = S.build_template("Long Call Calendar", 100, 10, market.expiry)
    assert S.leg_labels(calendar, market) == ["Short 1 × 100 Call (30d)", "Long 1 × 100 Call (60d)"]
    diagonal = S.build_template("Diagonal (Poor Man's Covered Call)", 100, 10, market.expiry)
    assert S.leg_labels(diagonal, market) == ["Long 1 × 90 Call (60d)", "Short 1 × 110 Call (30d)"]
    assert S.leg_labels(S.build_template("Bull Call Spread", 100, 10), market) == [
        "Long 1 × 100 Call",
        "Short 1 × 110 Call",
    ]
    # One option with its own expiry, IV and fill price, next to a share.
    custom = S.Strategy("custom", (S.Leg("stock", 1), S.Leg("call", -1, 105, 90 / 365, 0.25, 3.5)))
    assert S.leg_labels(custom, market) == ["Long 1 share", "Short 1 × 105 Call (90d, IV 25%, filled at $3.50)"]
    same_iv = S.Strategy("same", (S.Leg("put", 2, 95, volatility=0.2, premium=1.23456),))
    assert S.leg_labels(same_iv, market) == ["Long 2 × 95 Put (filled at $1.2346)"]


@pytest.mark.parametrize(
    ("strike", "text"),
    [
        (100.0, "100"),
        (102.5, "102.5"),
        (0.975, "0.975"),
        (1 - 2 * 0.025, "0.95"),
        (111.51, "111.51"),
        (12345.67, "12,345.67"),
        (12345.67 + 2 * 1234.57, "14,814.81"),
        (1234567.89, "1,234,567.89"),
        (1234567.891, "1,234,567.891"),
        (9999999.99 * 1.9, "18,999,999.981"),
    ],
)
def test_strikes_are_labelled_with_every_decimal(strike, text):
    # Round-2 verifier: labels used 6 significant digits, so a 12345.67 strike read "12345.7" and a 1234567.89
    # strike "1.23457e+06", unlike the legs table on the same page.
    assert S.strike_text(strike) == text
    assert S.Leg("put", 1, strike).label() == f"Long 1 × {text} Put"


def test_large_quantities_are_labelled_in_full():
    assert S.Leg("call", -1234567, 100).label() == "Short 1,234,567 × 100 Call"
    assert S.Leg("stock", 2.5).label() == "Long 2.5 shares"
