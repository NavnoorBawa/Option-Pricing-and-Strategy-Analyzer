"""One-day decay by full revaluation, beside theta / 365 (improvement guide, section 2 item 1).

strategies.time_decay is checked against Black-Scholes-Merton prices written out in this file (at T and T - 1 day,
the intrinsic value at expiry), against QuantLib with its evaluation date moved by one day, and on the panel's
numbers (docs/reviews/panel-v3.0.md, market maker, finding 3). The Pricing tab's single-option decays are checked
against the same closed form, the trees against it at their measured accuracy, and the American engine against
QuantLib's QdFp engine. The rule that picks the headline is checked on both sides of its threshold.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.special import ndtr

from options_analyzer import american, binomial
from options_analyzer import black_scholes as bs
from options_analyzer import strategies as S
from options_analyzer.ui import format as fmt
from options_analyzer.ui import tab_pricing

DAY = 1.0 / 365.0
#: The panel's SPX-like market: spot 5,800, rate 4.3 %, dividend yield 1.3 %.
SPX = {"spot": 5800.0, "rate": 0.043, "dividend_yield": 0.013}


def _bsm(S0: float, K: float, T: float, r: float, sigma: float, q: float, kind: str) -> float:
    """Black-Scholes-Merton value written out here (not black_scholes.price), the intrinsic value at T <= 0."""
    phi = 1.0 if kind == "call" else -1.0
    if T <= 0:
        return max(phi * (S0 - K), 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S0 / K) + (r - q) * T) / v + 0.5 * v
    d2 = d1 - v
    return phi * (S0 * math.exp(-q * T) * ndtr(phi * d1) - K * math.exp(-r * T) * ndtr(phi * d2))


def _reference(strategy: S.Strategy, market: S.Market, days: float = 1.0) -> float:
    """Value ``days`` later minus value now, leg by leg: options at their own IV and expiry, shares with the
    dividends they accrue, S0 (e^{rt} - e^{(r-q)t})."""
    t = days * DAY
    r, q, S0 = market.rate, market.dividend_yield, market.spot
    total = 0.0
    for leg in strategy.legs:
        if leg.kind == "stock":
            total += leg.quantity * S0 * (math.exp(r * t) - math.exp((r - q) * t))
            continue
        T = market.expiry if leg.expiry is None else leg.expiry
        sigma = market.volatility if leg.volatility is None else leg.volatility
        later = _bsm(S0, leg.strike, T - t, r, sigma, q, leg.kind)
        total += leg.quantity * (later - _bsm(S0, leg.strike, T, r, sigma, q, leg.kind))
    return total


def _book() -> S.Strategy:
    """A multi-expiry book: a leg expiring tomorrow, legs with their own IVs (a skew), and shares."""
    return S.Strategy(
        "book",
        (
            S.Leg("put", -30, 98, 1 / 365, 0.20),
            S.Leg("call", -30, 102, 9 / 365, 0.16),
            S.Leg("call", -20, 100, 30 / 365, 0.175),
            S.Leg("put", 10, 95, 182 / 365, 0.22),
            S.Leg("call", 12, 100, 365 / 365),
            S.Leg("stock", 15),
        ),
    )


# --- strategies.time_decay --------------------------------------------------------------------------------------

MARKETS = {
    "45d": S.Market(100, 0.04, 0.25, 45 / 365, 0.015),
    "1d": S.Market(100, 0.05, 0.2, 1 / 365),
    "2h": S.Market(100, 0.05, 0.2, 2 / 24 / 365),
    "negative rate": S.Market(100, -0.01, 0.3, 30 / 365, 0.02),
}


@pytest.mark.parametrize("market_name", list(MARKETS))
@pytest.mark.parametrize("name", list(S.TEMPLATES))
def test_decay_is_the_value_one_day_later_minus_the_value_now(name, market_name):
    # The reference is computed above from the closed form, leg by leg. The two differ only by rounding: each leg's
    # values are below 3 x spot, so 1e-12 of the spot covers a few ulps of every term.
    market = MARKETS[market_name]
    strategy = S.build_template(name, 100, 5, market.expiry)
    assert S.time_decay(strategy, market) == pytest.approx(_reference(strategy, market), rel=1e-12, abs=1e-12 * 100)


def test_decay_of_a_book_with_a_leg_expiring_tomorrow():
    market = S.Market(100, 0.045, 0.18, 30 / 365, 0.013)
    book = _book()
    assert S.time_decay(book, market) == pytest.approx(_reference(book, market), rel=1e-12, abs=1e-10)
    # The front leg is worth its intrinsic value at today's spot after a day: a 98 put at 100 is worth nothing.
    front = S.Strategy("front", (book.legs[0],))
    assert S.time_decay(front, market) == pytest.approx(30 * _bsm(100, 98, 1 / 365, 0.045, 0.20, 0.013, "put"))


@pytest.mark.parametrize("hours", [2.0, 24.0])
@pytest.mark.parametrize(("kind", "strike"), [("call", 90.0), ("call", 110.0), ("put", 90.0), ("put", 110.0)])
def test_an_option_expiring_within_the_day_is_worth_its_intrinsic_value(kind, strike, hours):
    # In the money it ends at S - K or K - S, out of the money at zero, whatever the days that pass beyond its expiry.
    market = S.Market(100, 0.05, 0.4, hours / 24 / 365)
    option = S.Strategy("one", (S.Leg(kind, -3, strike),))
    intrinsic = max((100 - strike) if kind == "call" else (strike - 100), 0.0)
    expected = -3 * (intrinsic - float(S.position_value(S.Strategy("one", (S.Leg(kind, 1, strike),)), market)))
    for days in (1.0, 5.0):
        assert S.time_decay(option, market, days=days) == pytest.approx(expected, rel=1e-13, abs=1e-13)


def test_the_panels_numbers():
    # docs/reviews/panel-v3.0.md, market maker, finding 3, at its stated precision (cents): the panel used 18 %
    # volatility for the single options and the 0DTE straddle, and 16 % for the condor.
    D1 = S.Market(volatility=0.18, expiry=DAY, **SPX)
    atm = S.Strategy("put", (S.Leg("put", 1, 5800),))
    assert float(S.position_value(atm, D1)) == pytest.approx(21.56, abs=0.005)
    assert S.position_risk(atm, D1).theta == pytest.approx(-10.66, abs=0.005)
    assert S.time_decay(atm, D1) == pytest.approx(-21.56, abs=0.005)  # the whole premium
    otm = S.Strategy("put", (S.Leg("put", 1, 5750),))
    assert S.position_risk(otm, D1).theta == pytest.approx(-6.97, abs=0.005)
    assert S.time_decay(otm, D1) == pytest.approx(-5.19, abs=0.005)

    D7 = S.Market(volatility=0.16, expiry=7 * DAY, **SPX)
    condor = S.build_template("Iron Condor", 5800, 50, D7.expiry)  # the app's automatic width at these inputs
    assert [leg.strike for leg in condor.legs] == [5700, 5750, 5850, 5900]
    assert S.position_risk(condor, D7, multiplier=100).theta == pytest.approx(137.50, abs=0.005)
    assert S.time_decay(condor, D7, multiplier=100) == pytest.approx(151.97, abs=0.005)

    H2 = S.Market(volatility=0.18, expiry=2 / 24 / 365, **SPX)
    straddle = S.Strategy("short straddle", (S.Leg("call", -1, 5800), S.Leg("put", -1, 5800)))
    risk = S.position_risk(straddle, H2, multiplier=100)
    assert risk.theta == pytest.approx(7552, abs=0.5) and risk.value == pytest.approx(-1259, abs=0.5)
    # Repriced, the straddle gains what it is worth and no more: both legs expire at the money, worth nothing.
    assert S.time_decay(straddle, H2, multiplier=100) == pytest.approx(-risk.value, rel=1e-12)


def test_the_decay_tends_to_theta_over_a_short_time():
    # A one-sided difference of the value in time: (V(t + h) - V(t)) / h = theta + O(h). Over a thousandth of a day
    # the O(h) term of this book measured 9.2e-6 of its theta (9.2e-5 over a hundredth, 9.2e-7 over a ten-thousandth:
    # first order, as it should be); rounding, amplified by 1 / h, is about 1e-10 of it. Bound: 3e-5.
    market = S.Market(100, 0.045, 0.18, 45 / 365, 0.013)
    book = S.Strategy("book", tuple(leg for leg in _book().legs if leg.expiry is None or leg.expiry > 2 / 365))
    theta = S.position_risk(book, market).theta  # per calendar day
    h = 1e-3
    assert S.time_decay(book, market, days=h) / h == pytest.approx(theta, rel=3e-5)


@given(
    spot=st.floats(50, 200),
    strike=st.floats(50, 200),
    days=st.integers(2, 730),
    rate=st.floats(-0.02, 0.1),
    dividend=st.floats(0, 0.08),
    vol=st.floats(0.05, 1.0),
)
def test_call_minus_put_decays_like_the_forward(spot, strike, days, rate, dividend, vol):
    # Put-call parity holds at both dates, so C - P decays as S e^{-qT} - K e^{-rT} does, whatever the volatility.
    market = S.Market(spot, rate, vol, days * DAY, dividend)
    call = S.time_decay(S.Strategy("c", (S.Leg("call", 1, strike),)), market)
    put = S.time_decay(S.Strategy("p", (S.Leg("put", 1, strike),)), market)
    T = days * DAY

    def forward(t: float) -> float:
        return spot * math.exp(-dividend * t) - strike * math.exp(-rate * t)

    # Each decay is a difference of values below 4 x max(S, K): rounding of a few ulps of that.
    assert call - put == pytest.approx(forward(T - DAY) - forward(T), abs=1e-13 * max(spot, strike))


@given(quantity=st.floats(-1e4, 1e4), multiplier=st.floats(-1e3, 1e3), days=st.floats(0, 40))
def test_decay_is_linear_in_the_legs_and_the_multiplier(quantity, multiplier, days):
    market = S.Market(100, 0.045, 0.18, 30 / 365, 0.013)
    book = _book()
    legs = [S.Strategy("one", (leg,)) for leg in book.legs]
    whole = S.time_decay(book, market, days=days, multiplier=multiplier)
    parts = sum(S.time_decay(leg, market, days=days, multiplier=multiplier) for leg in legs)
    assert whole == pytest.approx(parts, rel=1e-12, abs=1e-12 * abs(multiplier) * 1e4)
    single = S.Strategy("x", (S.Leg("put", quantity, 98, 9 / 365),))
    unit = S.Strategy("x", (S.Leg("put", 1, 98, 9 / 365),))
    assert S.time_decay(single, market, days=days, multiplier=multiplier) == pytest.approx(
        quantity * multiplier * S.time_decay(unit, market, days=days), rel=1e-12, abs=1e-300
    )


def test_no_time_no_decay():
    assert S.time_decay(_book(), S.Market(100, 0.05, 0.2, 0.5), days=0) == 0.0


def test_the_legs_table_decays_add_up_to_the_position():
    market = S.Market(100, 0.045, 0.18, 30 / 365, 0.013)
    rows = S.leg_table(_book(), market)
    assert [row["1-day decay"] for row in rows] == pytest.approx(
        [S.time_decay(S.Strategy("one", (leg,)), market) for leg in _book().legs], rel=1e-15, abs=1e-15
    )
    assert math.fsum(row["1-day decay"] for row in rows) == pytest.approx(S.time_decay(_book(), market), rel=1e-14)
    # A share accrues its dividends, S0 (e^{rt} - e^{(r-q)t}): over a day q S0 / 365 (its theta / day) to first
    # order, and (2r - q) / 2 x 1 day = 1.05e-4 of it more.
    shares = rows[-1]
    assert shares["1-day decay"] == pytest.approx(15 * 100 * (math.exp(0.045 / 365) - math.exp(0.032 / 365)))
    assert shares["1-day decay"] / shares["Theta / day"] - 1 == pytest.approx((2 * 0.045 - 0.013) / 2 / 365, rel=1e-3)


def test_zero_dte_legs_raise_no_warning():
    market = S.Market(100, 0.05, 0.2, 2 / 24 / 365)
    strategy = S.build_template("Iron Condor", 100, 1, market.expiry)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isfinite(S.time_decay(strategy, market, days=0.5))
        S.leg_table(strategy, market)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"days": -1}, "days must be a non-negative number of calendar days"),
        ({"days": float("nan")}, "days must be a finite number"),
        ({"days": "1"}, "days must be a number"),
        ({"days": [1, 2]}, "days must be a single number"),
        ({"multiplier": float("inf")}, "multiplier must be a finite number"),
        ({"multiplier": None}, "multiplier must be a number"),
    ],
)
def test_invalid_arguments_are_rejected_by_name(kwargs, message):
    with pytest.raises(ValueError, match=message):
        S.time_decay(_book(), S.Market(100, 0.05, 0.2, 0.5), **kwargs)


def test_options_are_keyword_only():
    with pytest.raises(TypeError):
        S.time_decay(_book(), S.Market(100, 0.05, 0.2, 0.5), 1.0)  # type: ignore[misc]


@pytest.mark.oracle
def test_book_decay_matches_quantlib_one_day_later():
    # Independent oracle: QuantLib's AnalyticEuropeanEngine on each leg, with the evaluation date moved by one day.
    # With reference-date events included, the leg that expires on the new evaluation date is valued at its
    # intrinsic value, as ours is (by default QuantLib counts it as expired and values it at zero).
    ql = pytest.importorskip("QuantLib")
    market = S.Market(100, 0.045, 0.18, 30 / 365, 0.013)
    strategy = S.Strategy("book", (*_book().legs[:-1], S.Leg("call", 5, 97, 1 / 365, 0.3)))  # one 1-day leg ITM
    settings = ql.Settings.instance()
    today, day_count = ql.Date(2, 1, 2026), ql.Actual365Fixed()

    def value(evaluation) -> float:
        settings.evaluationDate = evaluation
        spot = ql.QuoteHandle(ql.SimpleQuote(market.spot))
        rates = ql.YieldTermStructureHandle(ql.FlatForward(evaluation, market.rate, day_count))
        dividends = ql.YieldTermStructureHandle(ql.FlatForward(evaluation, market.dividend_yield, day_count))
        total = 0.0
        for leg in strategy.legs:
            sigma = market.volatility if leg.volatility is None else leg.volatility
            vol = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(evaluation, ql.NullCalendar(), sigma, day_count))
            process = ql.BlackScholesMertonProcess(spot, dividends, rates, vol)
            kind = ql.Option.Call if leg.kind == "call" else ql.Option.Put
            expiry = today + round((market.expiry if leg.expiry is None else leg.expiry) * 365)
            option = ql.EuropeanOption(ql.PlainVanillaPayoff(kind, leg.strike), ql.EuropeanExercise(expiry))
            option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
            total += leg.quantity * option.NPV()
        return total

    events = settings.includeReferenceDateEvents
    try:
        settings.includeReferenceDateEvents = True
        expected = value(today + 1) - value(today)
    finally:
        settings.includeReferenceDateEvents = events
    # QuantLib and ours agree on prices to about 1e-12 relative (tests/test_position_risk.py); the decay is a
    # difference of values of order 1e3, so 1e-9 absolute.
    assert S.time_decay(strategy, market) == pytest.approx(expected, abs=1e-9)


# --- The headline rule (ui.format) -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("theta", "decay", "leads"),
    [
        (-90.0, -100.0, False),  # 10 % of the larger exactly: theta stays
        (-89.99, -100.0, True),
        (-100.0, -90.0, False),  # the comparison is symmetric: relative to the larger in size
        (-100.0, -89.99, True),
        (137.50, 151.97, False),  # the panel's 7-day condor, 9.5 % apart
        (-10.66, -21.56, True),  # the panel's 1-day put
        (0.0, 0.0, False),
        (1e-3, -1e-3, True),  # opposite signs always differ
    ],
)
def test_the_decay_leads_beyond_ten_percent_of_the_larger(theta, decay, leads):
    assert fmt.decay_leads(theta, decay) is leads
    assert fmt.decay_leads(theta, decay, to_expiry=True) is True


@pytest.mark.parametrize(("days", "leads"), [(2, True), (3, False), (45, False)])
def test_at_the_money_the_headline_switches_between_three_and_two_days(days, leads):
    # Spot = strike = 100, 20 % volatility, rate 5 %: the call's theta / 365 and decay differ by 8.7 % at 3 days and
    # 14.1 % at 2 days (the put's by 9.7 % and 15.2 %), so the rule's threshold lies between the two.
    T = days * DAY
    for kind in ("call", "put"):
        theta = bs.greeks(100, 100, T, 0.05, 0.2, 0.0, kind)["theta"] / 365
        decay = _bsm(100, 100, T - DAY, 0.05, 0.2, 0.0, kind) - _bsm(100, 100, T, 0.05, 0.2, 0.0, kind)
        assert fmt.decay_leads(theta, decay) is leads, kind


def test_the_decay_is_labelled_by_the_expiry():
    assert fmt.decay_label(False) == "1-day decay" and fmt.decay_label(True) == "Decay to expiry"


# --- Single options on the Pricing tab ---------------------------------------------------------------------------


@pytest.mark.parametrize("days", [1, 2, 7, 45, 365])
def test_closed_form_decay_over_a_strip_of_strikes(days):
    strikes = np.linspace(80, 120, 9)
    T = days * DAY
    for kind in ("call", "put"):
        decay = tab_pricing.closed_form_decay(100, strikes, T, 0.05, 0.2, 0.01, kind)
        args = (0.05, 0.2, 0.01, kind)
        expected = [_bsm(100, k, T - DAY, *args) - _bsm(100, k, T, *args) for k in strikes]
        np.testing.assert_allclose(decay, expected, rtol=1e-12, atol=1e-13)
    # A scalar in, a float out; at one day the call ends at its intrinsic value.
    one = tab_pricing.closed_form_decay(100.0, 95.0, DAY, 0.05, 0.2, 0.0, "call")
    assert isinstance(one, float) and one == pytest.approx(5.0 - _bsm(100, 95, DAY, 0.05, 0.2, 0.0, "call"))


@pytest.mark.parametrize(
    ("steps", "expiry", "method", "later"),
    [
        (501, 30 * DAY, "lr", 501),  # Leisen-Reimer keeps its steps
        (501, 7 * DAY, "crr", 431),  # 501 x 6/7 = 429.4: up to 430, then to the odd 431
        (500, 7 * DAY, "crr", 430),
        (500, 10 * DAY, "crr", 450),  # 450 exactly: not rounded up
        (10, 2 * DAY, "crr", 6),  # 5, then the even 6
        (11, 2 * DAY, "crr", 7),  # 5.5, up to 6, then the odd 7
        (10, 365.0001 * DAY, "crr", 10),
        (10, 1.000001 * DAY, "crr", 2),  # never fewer than one step, with the parity of the current tree
        (11, 1.000001 * DAY, "crr", 1),
    ],
)
def test_later_tree_steps(steps, expiry, method, later):
    assert tab_pricing.later_tree_steps(steps, expiry, method) == later
    if method == "crr":
        assert later % 2 == steps % 2 and (expiry - DAY) / later <= expiry / steps * (1 + 1e-12)  # not longer


def _tree_decay(K: float, days: int, kind: str, method: str, american_: bool, steps: int = 501) -> float:
    T = days * DAY
    used = binomial.price_with_greeks(100, K, T, 0.05, 0.2, 0.0, kind, steps, american_, method).steps
    later = tab_pricing.later_tree_steps(used, T, method)
    shorter = binomial.price(100, K, T - DAY, 0.05, 0.2, 0.0, kind, later, american_, method)
    return shorter - binomial.price(100, K, T, 0.05, 0.2, 0.0, kind, steps, american_, method)


@pytest.mark.parametrize("days", [2, 7, 30, 365, 3650])
@pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
def test_tree_decays_track_the_closed_form(K, days):
    # Measured over strikes 80 to 120 and 2 to 3,650 days at 501 steps, on decays of at least 1 cent a share
    # (later_tree_steps): Leisen-Reimer within 3.0e-6 of the closed form, relative, and Cox-Ross-Rubinstein with the
    # same time step within 1.5 %; below a cent, within 8.4e-9 and 1.6e-4 absolute on these cases. Bounds: three
    # times those.
    for kind in ("call", "put"):
        exact = _bsm(100, K, (days - 1) * DAY, 0.05, 0.2, 0.0, kind) - _bsm(100, K, days * DAY, 0.05, 0.2, 0.0, kind)
        lr = _tree_decay(K, days, kind, "lr", False)
        crr = _tree_decay(K, days, kind, "crr", False)
        assert lr == pytest.approx(exact, rel=9e-6, abs=3e-8), kind
        assert crr == pytest.approx(exact, rel=0.045, abs=5e-4 if abs(exact) < 0.01 else 0), kind


def test_crr_with_the_same_steps_would_not_cancel_its_error():
    # The case that made the rule: a 30-day in-the-money put (110 on 100) decays by +0.3149 cents a share. Two CRR
    # trees with 501 steps each put it at +0.3553 cents (13 % off); the same time step puts it within 2 %.
    exact = _bsm(100, 110, 29 * DAY, 0.05, 0.2, 0.0, "put") - _bsm(100, 110, 30 * DAY, 0.05, 0.2, 0.0, "put")
    same = binomial.price(100, 110, 29 * DAY, 0.05, 0.2, 0.0, "put", 501, False, "crr") - binomial.price(
        100, 110, 30 * DAY, 0.05, 0.2, 0.0, "put", 501, False, "crr"
    )
    assert abs(same / exact - 1) > 0.10
    assert _tree_decay(110, 30, "put", "crr", False) == pytest.approx(exact, rel=0.02)


@pytest.mark.parametrize("days", [2, 7, 30, 365])
def test_american_tree_decays_track_the_american_engine(days):
    # Early exercise has no closed form: the reference is the American engine at its high scheme (within 4e-10 of
    # QuantLib's QdFp, docs/methods/american_alo.md). The 501-step trees are first-order accurate with early
    # exercise; their decay errors measured 7.6e-6 (Leisen-Reimer) and 1.25e-4 (Cox-Ross-Rubinstein, same time step)
    # at most on these at-the-money puts, so the bounds are 2.5e-5 and 4e-4.
    def engine(T: float) -> float:
        return american.price(100, 100, T, 0.05, 0.2, 0.0, option_type="put", scheme="high")

    reference = engine((days - 1) * DAY) - engine(days * DAY)
    assert _tree_decay(100, days, "put", "lr", True) == pytest.approx(reference, abs=2.5e-5)
    assert _tree_decay(100, days, "put", "crr", True) == pytest.approx(reference, abs=4e-4)


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("kind", "K", "r", "q"),
    [("put", 100, 0.05, 0.0), ("put", 110, 0.05, 0.02), ("call", 100, 0.02, 0.06), ("call", 90, 0.03, 0.08)],
)
def test_american_engine_decays_match_quantlib(kind, K, r, q):
    # The Pricing tab's decay under the American engine is the engine's value one day shorter minus its value now.
    # Reference: QuantLib's QdFp engine at strict settings at both expiries. Measured worst difference over these
    # 16 cases 1.5e-9 (the call at 90 over a year); bound 5e-9.
    ql = pytest.importorskip("QuantLib")
    today, day_count = ql.Date(2, 1, 2026), ql.Actual365Fixed()
    ql.Settings.instance().evaluationDate = today
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(100.0)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, q, day_count)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), 0.25, day_count)),
    )

    def quantlib(days: int) -> float:
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if kind == "call" else ql.Option.Put, K)
        option = ql.VanillaOption(payoff, ql.AmericanExercise(today, today + days))
        option.setPricingEngine(ql.QdFpAmericanEngine(process, ql.QdFpTanhSinhIterationScheme(24, 64, 1e-12)))
        return option.NPV()

    for days in (2, 7, 30, 365):
        ours = american.price(100, K, (days - 1) * DAY, r, 0.25, q, option_type=kind) - american.price(
            100, K, days * DAY, r, 0.25, q, option_type=kind
        )
        assert ours == pytest.approx(quantlib(days - 1) - quantlib(days), abs=5e-9), days
