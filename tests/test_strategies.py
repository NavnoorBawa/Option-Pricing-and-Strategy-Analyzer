import re
import sys

import mpmath as mp
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


# --- Final review -------------------------------------------------------------------------------------------------


def _covered_call_reference(rate: float, vol: float, years: float) -> tuple[float, float]:
    """Breakeven and probability of profit of an at-the-money covered call (spot 100), in 50-digit arithmetic.

    Below the strike the P&L at expiry is S_T - (100 - C), so the breakeven is 100 - C with C the closed-form call
    value; the probability of profit is the log-normal mass above it under the risk-neutral drift.
    """
    with mp.workdps(50):
        s, k, t, r, sigma = mp.mpf(100), mp.mpf(100), mp.mpf(years), mp.mpf(rate), mp.mpf(vol)
        d1 = (mp.log(s / k) + (r + sigma**2 / 2) * t) / (sigma * mp.sqrt(t))
        call = s * mp.ncdf(d1) - k * mp.exp(-r * t) * mp.ncdf(d1 - sigma * mp.sqrt(t))
        breakeven = s - call
        mean = mp.log(s) + (r - sigma**2 / 2) * t
        return float(breakeven), float(1 - mp.ncdf((mp.log(breakeven) - mean) / (sigma * mp.sqrt(t))))


@pytest.mark.parametrize("years", [5.0, 10.0])
def test_tiny_breakeven_keeps_the_probability_of_profit_on_its_side(years):
    # Final review: at 500% volatility over ten years the covered call's breakeven (about 2e-13) collapsed onto
    # S = 0, so its probability of profit read 100% instead of about 5e-9. The P&L near S = 0 is S - (100 - C) with
    # 100 - C a difference of two numbers near 100, so the float breakeven carries a few per cent of rounding.
    market = S.Market(100, 0.05, 5.0, years)
    m = S.analyze(S.build_template("Covered Call", 100, 10), market)
    breakeven, pop = _covered_call_reference(0.05, 5.0, years)
    assert m.breakevens == pytest.approx([breakeven], rel=0.1)
    assert m.probability_of_profit == pytest.approx(pop, rel=0.05)
    # The probability is the log-normal mass above the breakeven the analysis found, not above S = 0.
    vol_t = 5.0 * np.sqrt(years)
    mean = np.log(100) + (0.05 - 0.5 * 25.0) * years
    assert m.probability_of_profit == pytest.approx(norm.sf((np.log(m.breakevens[0]) - mean) / vol_t), rel=1e-9)


@pytest.mark.parametrize(
    ("name", "rate"),
    [("Long Call Calendar", 0.05), ("Diagonal (Poor Man's Covered Call)", 0.05), ("Short Put", 0.0)],
)
def test_tiny_breakeven_or_zero_edge_is_not_placed_at_zero(name, rate):
    # Final review: these showed a probability of profit of >99.9% at 500% volatility over ten years, where the
    # position profits only above a price of about 2e-13 (true probability about 5e-9). The short put's P&L
    # near S = 0 is within rounding of zero; the edge of that zero region used to sit at S = 0 too.
    market = S.Market(100, rate, 5.0, 10.0)
    m = S.analyze(S.build_template(name, 100, 10, market.expiry), market)
    assert m.probability_of_profit < 1e-6
    assert 0 < m.breakevens[0] < 1e-10


# --- RK-9: strikes by delta, finite inputs, same-day expiries -------------------------------------------------------

CONDOR_MARKET = S.Market(100, 0.045, 0.18, 45 / 365, 0.013)


def test_iron_condor_with_16_delta_shorts():
    # RK-9 acceptance (1): an iron condor at 45 days and 18% vol with 16-delta shorts has a short put delta of
    # -0.16 and a short call delta of +0.16 (each within 1e-6) at the leg vols; the 5-delta wings likewise.
    market = CONDOR_MARKET
    condor = S.build_template("Iron Condor", 100, 0, strikes_by="delta", deltas=(0.16, 0.05), market=market)
    deltas = {}
    for leg in condor.legs:
        vol = market.volatility if leg.volatility is None else leg.volatility
        expiry = market.expiry if leg.expiry is None else leg.expiry
        g = bs.greeks(market.spot, leg.strike, expiry, market.rate, vol, market.dividend_yield, leg.kind)
        deltas[(leg.kind, leg.quantity)] = g["delta"]
    assert deltas[("put", -1)] == pytest.approx(-0.16, abs=1e-6)
    assert deltas[("call", -1)] == pytest.approx(0.16, abs=1e-6)
    assert deltas[("put", 1)] == pytest.approx(-0.05, abs=1e-6)
    assert deltas[("call", 1)] == pytest.approx(0.05, abs=1e-6)
    strikes = [leg.strike for leg in condor.legs]
    assert strikes == sorted(strikes)  # long put < short put < short call < long call
    assert [(leg.kind, leg.quantity) for leg in condor.legs] == [("put", 1), ("put", -1), ("call", -1), ("call", 1)]


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("days", [120 / 1440, 1, 7, 45, 365, 1825])
@pytest.mark.parametrize(("vol", "q", "r"), [(0.05, 0.0, 0.05), (0.2, 0.03, -0.01), (0.8, 0.03, 0.05)])
def test_strike_for_delta_round_trip(kind, days, vol, q, r):
    # The returned strike reproduces the target spot delta of black_scholes.greeks to 1e-12 (measured 1e-13).
    expiry = days / 365
    market = S.Market(100, r, vol, expiry, q)
    sign = 1 if kind == "call" else -1
    for d in (0.01, 0.05, 0.10, 0.16, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90):
        target = sign * d * np.exp(-q * expiry)
        strike = S.strike_for_delta(target, market, expiry, vol, kind)
        assert bs.greeks(100, strike, expiry, r, vol, q, kind)["delta"] == pytest.approx(target, abs=1e-12)


@pytest.mark.oracle
def test_strike_for_delta_matches_quantlib():
    # Independent oracle: QuantLib's BlackDeltaCalculator.strikeFromDelta with spot delta. QuantLib's inverse
    # normal CDF is accurate to about 1e-9, which bounds the agreement (measured 3.3e-9).
    ql = pytest.importorskip("QuantLib")
    for days, vol, q, r in [(1, 0.2, 0.0, 0.05), (45, 0.18, 0.013, 0.045), (365, 0.35, 0.03, -0.01), (1825, 0.8, 0, 0)]:
        expiry = days / 365
        market = S.Market(100, r, vol, expiry, q)
        for kind, ql_kind, sign in (("call", ql.Option.Call, 1), ("put", ql.Option.Put, -1)):
            for d in (0.05, 0.16, 0.25, 0.5, 0.75):
                target = sign * d * np.exp(-q * expiry)
                calc = ql.BlackDeltaCalculator(
                    ql_kind, ql.DeltaVolQuote.Spot, 100.0, np.exp(-r * expiry), np.exp(-q * expiry), vol * expiry**0.5
                )
                expected = calc.strikeFromDelta(float(target))
                assert S.strike_for_delta(target, market, expiry, vol, kind) == pytest.approx(expected, rel=1e-8)


def test_strike_for_delta_rejects_unattainable_deltas():
    market = S.Market(100, 0.05, 0.2, 1.0, 0.03)
    with pytest.raises(ValueError, match=r"call delta must lie in \(0, 0.97"):
        S.strike_for_delta(0.98, market, 1.0, 0.2, "call")  # above e^{-qT} = 0.9704
    with pytest.raises(ValueError, match="put delta"):
        S.strike_for_delta(0.16, market, 1.0, 0.2, "put")  # a put's delta is negative
    with pytest.raises(ValueError, match="call delta"):
        S.strike_for_delta(0.0, market, 1.0, 0.2, "call")
    with pytest.raises(ValueError, match="target"):
        S.strike_for_delta(float("nan"), market, 1.0, 0.2, "call")
    with pytest.raises(ValueError, match="expiry"):
        S.strike_for_delta(0.25, market, 0.0, 0.2, "call")
    with pytest.raises(ValueError, match="volatility"):
        S.strike_for_delta(0.25, market, 1.0, -0.2, "call")
    with pytest.raises(ValueError, match="option_type"):
        S.strike_for_delta(0.25, market, 1.0, 0.2, "stock")


def test_strike_for_delta_out_of_floating_point_range():
    # The strike is spot x e^{-d1 v + (r - q) T + v^2 / 2} with v = sigma sqrt(T); it used to raise OverflowError.
    market = S.Market(100, 0.05, 0.2, 1.0)
    for target, kind in ((0.5, "call"), (-0.5, "put")):  # v = 10 sqrt(30) = 54.8: e^1502
        with pytest.raises(ValueError, match=r"strike of a .*-delta .* e\^1502, is out of floating-point range"):
            S.strike_for_delta(target, market, 30.0, 10.0, kind)
    # Below the smallest normal float: a 1e-300-delta put (d1 = 37) with v = 37 and a carry of -50.
    with pytest.raises(ValueError, match=r"e\^-736.2, is out of floating-point range"):
        S.strike_for_delta(-1e-300, S.Market(100, -0.5, 0.2, 1.0), 100.0, 3.7, "put")
    with pytest.raises(ValueError, match="dividend discount factor"):
        S.strike_for_delta(0.5, S.Market(100, 0.05, 0.2, 1.0, -100.0), 10.0, 0.2, "call")


def test_strike_for_delta_gives_a_strike_or_a_value_error():
    # Property over a deliberately wide domain (rates and yields of +/-100%, 100 years, 2000% vol): any finite
    # inputs give a positive finite strike or a ValueError, never an OverflowError.
    hypothesis = pytest.importorskip("hypothesis")
    st = hypothesis.strategies

    @hypothesis.given(
        target=st.floats(-1.0, 1.0),
        rate=st.floats(-1.0, 1.0),
        q=st.floats(-1.0, 1.0),
        expiry=st.floats(1e-9, 100.0),
        vol=st.floats(1e-4, 20.0),
        kind=st.sampled_from(["call", "put"]),
    )
    def check(target, rate, q, expiry, vol, kind):
        try:
            strike = S.strike_for_delta(target, S.Market(100, rate, 0.2, 1.0, q), expiry, vol, kind)
        except ValueError:
            return
        assert sys.float_info.min <= strike <= sys.float_info.max

    check()


def test_strike_for_delta_round_trip_property():
    # Property over desk inputs (one minute to ten years, 1% to 300% vol, deltas from 1e-4 to 1 - 1e-4 of e^{-qT}):
    # the strike's delta is the target. The tolerance includes the delta's conditioning: a strike rounded to a float
    # moves it by up to phi(d1) eps / v, with v = sigma sqrt(T) (1e-10 at one minute and 1% vol).
    hypothesis = pytest.importorskip("hypothesis")
    st = hypothesis.strategies

    @hypothesis.given(
        fraction=st.floats(1e-4, 1 - 1e-4),
        rate=st.floats(-0.1, 0.2),
        q=st.floats(-0.1, 0.2),
        expiry=st.floats(1 / 525_600, 10.0),
        vol=st.floats(0.01, 3.0),
        kind=st.sampled_from(["call", "put"]),
    )
    def check(fraction, rate, q, expiry, vol, kind):
        target = (1 if kind == "call" else -1) * fraction * np.exp(-q * expiry)
        strike = S.strike_for_delta(target, S.Market(100, rate, 0.2, 1.0, q), expiry, vol, kind)
        delta = bs.greeks(100, strike, expiry, rate, vol, q, kind)["delta"]
        assert delta == pytest.approx(target, abs=1e-12 + 1e-15 / (vol * expiry**0.5))

    check()


def _width_offsets(name: str) -> list[int]:
    """Each leg's strike offset from the centre in strike steps, read from the width-mode build."""
    legs = S.build_template(name, 100, 5, 30 / 365).legs
    return [round((leg.strike - 100) / 5) if leg.is_option else 0 for leg in legs]


@pytest.mark.parametrize("name", [name for name, t in S.TEMPLATES.items() if t.uses_width])
def test_delta_strikes_for_every_template_with_a_width(name):
    # Strikes n steps above the centre are the calls with delta deltas[n-1], n steps below the puts with delta
    # -deltas[n-1], at each leg's own expiry (a diagonal's legs differ); legs at the centre keep the strike.
    market = S.Market(100, 0.04, 0.22, 30 / 365, 0.01)
    deltas = (0.30, 0.10)
    by_delta = S.build_template(name, 101, 0, 30 / 365, strikes_by="delta", deltas=deltas, market=market)
    by_width = S.build_template(name, 100, 5, 30 / 365)
    for leg, width_leg, n in zip(by_delta.legs, by_width.legs, _width_offsets(name), strict=True):
        assert (leg.kind, leg.quantity, leg.expiry, leg.volatility) == (
            width_leg.kind,
            width_leg.quantity,
            width_leg.expiry,
            width_leg.volatility,
        )
        if not leg.is_option:
            continue
        expiry = market.expiry if leg.expiry is None else leg.expiry
        if n == 0:
            assert leg.strike == 101
            continue
        kind = "call" if n > 0 else "put"
        delta = bs.greeks(100, leg.strike, expiry, 0.04, 0.22, 0.01, kind)["delta"]
        assert delta == pytest.approx(np.sign(n) * deltas[abs(n) - 1], abs=1e-12)


@pytest.mark.parametrize("name", [name for name, t in S.TEMPLATES.items() if not t.uses_width])
def test_delta_mode_leaves_templates_without_a_width_unchanged(name):
    by_delta = S.build_template(name, 97, 0, 0.1, strikes_by="delta", market=CONDOR_MARKET)
    assert by_delta == S.build_template(name, 97, 5, 0.1)


def test_delta_mode_errors():
    market = CONDOR_MARKET
    with pytest.raises(ValueError, match="needs the market"):
        S.build_template("Iron Condor", 100, 0, strikes_by="delta", deltas=(0.16, 0.05))
    with pytest.raises(ValueError, match="needs the deltas"):
        S.build_template("Iron Condor", 100, 0, strikes_by="delta", market=market)
    with pytest.raises(ValueError, match="needs 2 deltas"):
        S.build_template("Iron Condor", 100, 0, strikes_by="delta", deltas=0.16, market=market)
    with pytest.raises(ValueError, match="decrease"):
        S.build_template("Iron Condor", 100, 0, strikes_by="delta", deltas=(0.10, 0.16), market=market)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        S.build_template("Short Strangle", 100, 0, strikes_by="delta", deltas=1.2, market=market)
    with pytest.raises(ValueError, match="out of order"):
        # A 25-delta call sits near 103: below a centre strike of 110, the bull spread would be inverted.
        S.build_template("Bull Call Spread", 110, 0, strikes_by="delta", deltas=0.25, market=market)
    with pytest.raises(ValueError, match="strikes_by"):
        S.build_template("Iron Condor", 100, 5, strikes_by="moneyness")
    # The out-of-order strikes are given in cents (the 25-delta call strike is 104.96869671...).
    with pytest.raises(ValueError, match=re.escape("the delta strikes 120 and 104.97 are out of order")):
        S.build_template("Bull Call Spread", 120, 0, strikes_by="delta", deltas=0.25, market=market)
    # A single delta works where one step is enough.
    strangle = S.build_template("Short Strangle", 100, 0, strikes_by="delta", deltas=0.16, market=market)
    assert [round(leg.strike, 4) for leg in strangle.legs] == [
        round(S.strike_for_delta(-0.16, market, market.expiry, 0.18, "put"), 4),
        round(S.strike_for_delta(0.16, market, market.expiry, 0.18, "call"), 4),
    ]


@pytest.mark.parametrize(
    ("name", "kwargs", "message"),
    [
        ("Iron Condor", {"deltas": ("a", 0.1)}, "deltas must be numbers, got ('a', 0.1)"),
        ("Iron Condor", {"deltas": (0.16, None)}, "deltas must be numbers, got (0.16, None)"),
        ("Iron Condor", {"deltas": "0.16"}, "deltas must be numbers, got '0.16'"),
        ("Iron Condor", {"deltas": [[0.16, 0.05]]}, "deltas must be numbers, got [[0.16, 0.05]]"),
        ("Iron Condor", {"deltas": [0.16, [0.05, 0.01]]}, "deltas must be numbers"),
        ("Iron Condor", {"deltas": (0.16, float("nan"))}, "deltas are absolute values in (0, 1), got (0.16, nan)"),
        ("Iron Condor", {"deltas": np.array([0.16, 0.0])}, "deltas are absolute values in (0, 1), got (0.16, 0.0)"),
        ("Iron Condor", {"deltas": (0.16,)}, "needs 2 deltas (one per strike step out from the centre), got 1"),
        # Centre-only templates need no deltas, but deltas that are given are still checked.
        ("Long Straddle", {"deltas": ("a",)}, "deltas must be numbers"),
        ("Bull Call Spread", {"deltas": 0.25, "strike": float("nan")}, "strike must be a finite number, got nan"),
        ("Iron Condor", {"deltas": (0.16, 0.05), "strike": -100.0}, "strike must be positive, got -100"),
    ],
)
def test_delta_mode_names_the_bad_argument(name, kwargs, message):
    arguments = {"strike": 100.0, "width": 0.0, "strikes_by": "delta", "market": CONDOR_MARKET, **kwargs}
    with pytest.raises(ValueError, match=re.escape(message)):
        S.build_template(name, **arguments)


def test_deltas_without_delta_mode_are_rejected():
    # Deltas used to be ignored silently in the default width mode, which spaced the strikes by width instead.
    message = "deltas are only used with strikes_by='delta', got deltas=(0.16, 0.05) with strikes_by='width'"
    with pytest.raises(ValueError, match=re.escape(message)):
        S.build_template("Iron Condor", 100, 5, deltas=(0.16, 0.05), market=CONDOR_MARKET)
    with pytest.raises(ValueError, match="only used with strikes_by='delta'"):
        S.build_template("Long Straddle", 100, 5, strikes_by="width", deltas=0.16)
    # The market alone is harmless in width mode (a caller may pass it whichever mode is chosen) and is unused.
    by_width = S.build_template("Iron Condor", 100, 5, market=CONDOR_MARKET)
    assert by_width == S.build_template("Iron Condor", 100, 5)


def test_delta_mode_accepts_numpy_deltas():
    market = CONDOR_MARKET
    plain = S.build_template("Iron Condor", 100, 0, strikes_by="delta", deltas=(0.16, 0.05), market=market)
    array = np.array([0.16, 0.05])
    assert S.build_template("Iron Condor", 100, 0, strikes_by="delta", deltas=array, market=market) == plain
    single = S.build_template("Short Strangle", 100, 0, strikes_by="delta", deltas=np.float64(0.16), market=market)
    assert single == S.build_template("Short Strangle", 100, 0, strikes_by="delta", deltas=[0.16], market=market)


_MARKET_FIELDS = ("spot", "rate", "volatility", "expiry", "dividend_yield")
_LEG_FIELDS = ("quantity", "strike", "expiry", "volatility", "premium")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize("field", _MARKET_FIELDS)
def test_market_rejects_non_finite_fields(field, bad):
    # RK-9 acceptance (2): NaN or inf in each Market field raises ValueError naming the field.
    values = {"spot": 100.0, "rate": 0.05, "volatility": 0.2, "expiry": 1.0, "dividend_yield": 0.01, field: bad}
    with pytest.raises(ValueError, match=rf"market {field} must be a finite number"):
        S.Market(**values)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize("field", _LEG_FIELDS)
@pytest.mark.parametrize("kind", ["call", "stock"])
def test_leg_rejects_non_finite_fields(kind, field, bad):
    # RK-9 acceptance (2): NaN or inf in each Leg field raises ValueError naming the field (today quantity=inf,
    # for one, was accepted).
    values = {"kind": kind, "quantity": 1.0, "strike": 100.0, "expiry": 0.5, "volatility": 0.2, "premium": 3.0}
    values[field] = bad
    with pytest.raises(ValueError, match=rf"leg {field} must be a finite number"):
        S.Leg(**values)


def test_non_numbers_and_fees_are_rejected_by_name():
    with pytest.raises(ValueError, match="market rate must be a number"):
        S.Market(100, "5%", 0.2, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="leg strike must be a number"):
        S.Leg("put", 1, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="strategy fees must be a finite number"):
        S.Strategy("x", (S.Leg("call", 1, 100),), fees=float("nan"))
    # Finite values, and None for the optional leg fields, are still accepted as before.
    leg = S.Leg("call", -3, 105.5, None, None, None)
    assert (leg.expiry, leg.volatility, leg.premium) == (None, None, None)
    assert S.Market(100, -0.01, 0.2, 1e-6, -0.02).rate == -0.01


TWO_HOURS = 120 / 525_600


@pytest.mark.parametrize("spot", [100.0, 5000.0])
def test_a_position_two_hours_from_expiry_analyses_without_warnings(spot):
    # RK-9 acceptance (3): a leg 2 hours from expiry (T = 120 / 525,600 years) analyses without warnings, alone and
    # next to a 30-day leg. (pytest also turns any RuntimeWarning into an error for every test.)
    import warnings

    from options_analyzer import scenarios as sc

    market = S.Market(spot, 0.045, 0.15, 30 / 365, 0.013)
    k = [spot * f for f in (0.99, 0.996, 1.004, 1.01)]
    condor = S.Strategy(
        "0DTE condor",
        (
            S.Leg("put", 1, k[0], TWO_HOURS),
            S.Leg("put", -1, k[1], TWO_HOURS),
            S.Leg("call", -1, k[2], TWO_HOURS),
            S.Leg("call", 1, k[3], TWO_HOURS),
        ),
    )
    calendar = S.Strategy("0DTE vs 30d", (S.Leg("call", -1, spot, TWO_HOURS), S.Leg("call", 1, spot)))
    for strategy in (condor, calendar):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            metrics = S.analyze(strategy, market)
            labels = S.leg_labels(strategy, market)
            S.leg_table(strategy, market)
            risk = S.position_risk(strategy, market)
            sc.greek_ladder(strategy, market)
            sc.greek_ladder(strategy, market, roll_days=1)
            for axis in ("delta", "moneyness"):
                sc.vega_buckets(strategy, market, strike_axis=axis)
            sc.weighted_vega(strategy, market)
            sc.weighted_vega(strategy, market, scheme="sqrt_time")
            sc.scenario_cube(
                strategy, market, np.linspace(-0.02, 0.02, 9), [-0.5, 0, 0.5], [0, 1 / 24, 1], vol_mode="1m"
            )
            sc.run_stress_tests(strategy, market, sc.HISTORICAL_SCENARIOS, include_elapsed_time=True)
        assert metrics.horizon == TWO_HOURS
        assert 0 < metrics.probability_of_profit < 1
        for be in metrics.breakevens:
            assert float(S.pnl(strategy, market, be, metrics.horizon)) == pytest.approx(0.0, abs=1e-9 * spot)
        assert "(0.1d)" in labels[0]
        assert np.isfinite(risk.gamma) and risk.gamma != 0
    # The short-dated option is worth what Black-Scholes-Merton gives at that T (about S sigma sqrt(T) / sqrt(2 pi)).
    atm = float(S.position_value(S.Strategy("atm", (S.Leg("call", 1, spot, TWO_HOURS),)), market))
    assert atm == pytest.approx(bs.price(spot, spot, TWO_HOURS, 0.045, 0.15, 0.013), rel=1e-15)
    assert atm == pytest.approx(spot * 0.15 * np.sqrt(TWO_HOURS) / np.sqrt(2 * np.pi), rel=1e-2)
