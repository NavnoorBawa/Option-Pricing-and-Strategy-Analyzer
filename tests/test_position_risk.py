"""strategies.position_risk: dollar and second-order Greeks of a whole position (RK-1).

The Greeks are checked three ways: against the raw first-order Greeks of position_greeks (unit conversions), against
central finite differences of position_value (every Greek, every template, so they are the derivatives of the
valuation the P&L uses), and against QuantLib's analytic European engine on a multi-expiry book.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

from options_analyzer import black_scholes as bs
from options_analyzer import strategies as S

MARKET = S.Market(100, 0.05, 0.2, 0.25, 0.01)
MARKETS = {
    "3M": MARKET,
    "45d": S.Market(100, 0.04, 0.25, 45 / 365, 0.015),
    "1Y, no dividends": S.Market(100, 0.05, 0.2, 1.0, 0.0),
}
FIELDS = [f.name for f in dataclasses.fields(S.PositionRisk)]


def _template(name: str, market: S.Market) -> S.Strategy:
    return S.build_template(name, 100, 5, market.expiry)


def _book() -> S.Strategy:
    """A multi-expiry book with its own leg IVs (a skew) and shares."""
    legs = (
        S.Leg("put", -30, 98, 9 / 365, 0.20),
        S.Leg("call", -30, 102, 9 / 365, 0.16),
        S.Leg("call", -20, 100, 30 / 365, 0.175),
        S.Leg("put", -20, 100, 30 / 365, 0.175),
        S.Leg("put", 10, 95, 182 / 365, 0.22),
        S.Leg("call", 12, 100, 365 / 365, 0.18),
        S.Leg("stock", 15),
    )
    return S.Strategy("book", legs)


def test_position_risk_converts_the_raw_greeks_to_desk_units():
    for name in S.TEMPLATES:
        strategy = _template(name, MARKET)
        raw = S.position_greeks(strategy, MARKET)
        risk = S.position_risk(strategy, MARKET)
        spot = MARKET.spot
        assert risk.spot == spot
        assert risk.value == pytest.approx(float(S.position_value(strategy, MARKET)), rel=1e-15)
        assert risk.delta == pytest.approx(raw["delta"], rel=1e-15, abs=1e-15)
        assert risk.dollar_delta == pytest.approx(raw["delta"] * spot, rel=1e-15, abs=1e-13)
        assert risk.gamma == pytest.approx(raw["gamma"], rel=1e-15, abs=1e-17)
        assert risk.gamma_1pct == pytest.approx(raw["gamma"] * spot * 0.01, rel=1e-15, abs=1e-17)
        assert risk.dollar_gamma_1pct == pytest.approx(0.5 * raw["gamma"] * (0.01 * spot) ** 2, rel=1e-15, abs=1e-17)
        assert risk.vega == pytest.approx(raw["vega"] / 100, rel=1e-15, abs=1e-15)
        assert risk.theta == pytest.approx(raw["theta"] / 365, rel=1e-15, abs=1e-15)
        assert risk.rho == pytest.approx(raw["rho"] / 100, rel=1e-15, abs=1e-15)


def test_position_greeks_keeps_its_keys_and_values():
    # Compatibility: the dict of first-order raw Greeks is what the UI and risk.py read.
    strategy = S.build_template("Collar", 100, 10)
    greeks = S.position_greeks(strategy, MARKET)
    assert list(greeks) == ["delta", "gamma", "theta", "vega", "rho"]
    put = bs.greeks(100, 90, 0.25, 0.05, 0.2, 0.01, "put")
    call = bs.greeks(100, 110, 0.25, 0.05, 0.2, 0.01, "call")
    assert greeks["delta"] == 1 + put["delta"] - call["delta"]
    assert greeks["vega"] == put["vega"] - call["vega"]


def _d1(f, h: float) -> float:
    """Fourth-order central first derivative of f at 0: error O(h^4)."""
    return (-f(2 * h) + 8 * f(h) - 8 * f(-h) + f(-2 * h)) / (12 * h)


def _d2(f, h: float) -> float:
    """Fourth-order central second derivative of f at 0: error O(h^4)."""
    return (-f(2 * h) + 16 * f(h) - 30 * f(0.0) + 16 * f(-h) - f(-2 * h)) / (12 * h * h)


def _finite_differences(strategy: S.Strategy, market: S.Market) -> dict[str, float]:
    """Every PositionRisk Greek in its desk unit from fourth-order central differences of position_value.

    Steps: 1e-3 x spot, 1e-3 of volatility (0.1 vol point), 1e-4 years, so the truncation error is about
    (step / scale)^4 ~ 1e-8 relative and rounding about 1e-10; time steps either side of t = 0 just lengthen or
    shorten every leg's remaining life.
    """
    h, k, e = 1e-3 * market.spot, 1e-3, 1e-4

    def v(ds=0.0, dv=0.0, dt_=0.0, dr=0.0):
        return float(S.position_value(strategy, market, market.spot + ds, dt_, vol_shift=dv, rate_shift=dr))

    return {
        "delta": _d1(lambda a: v(ds=a), h),
        "gamma": _d2(lambda a: v(ds=a), h),
        "vega": _d1(lambda a: v(dv=a), k) * 0.01,
        "theta": _d1(lambda a: v(dt_=a), e) / 365,
        "rho": _d1(lambda a: v(dr=a), k) * 0.01,
        "vanna": _d1(lambda a: _d1(lambda b: v(ds=a, dv=b), k), h) * 0.01,
        "volga": _d2(lambda a: v(dv=a), k) * 1e-4,
        "charm": _d1(lambda a: _d1(lambda b: v(ds=b, dt_=a), h), e) / 365,
        "veta": _d1(lambda a: _d1(lambda b: v(dv=b, dt_=a), k), e) * 0.01 / 365,
    }


def _gross(strategy: S.Strategy, market: S.Market) -> dict[str, float]:
    """Sum over legs of |leg Greek|: the scale against which a net Greek that nearly cancels is judged."""
    singles = [S.position_risk(S.Strategy("leg", (leg,)), market) for leg in strategy.legs]
    return {name: sum(abs(getattr(r, name)) for r in singles) for name in FIELDS if name not in ("spot", "value")}


@pytest.mark.parametrize("market_name", list(MARKETS))
@pytest.mark.parametrize("name", list(S.TEMPLATES))
def test_every_greek_is_the_derivative_of_position_value(name, market_name):
    # RK-1 acceptance (3): vanna, volga and charm (and every other Greek) match finite differences of
    # position_value to 1e-4 relative. A net Greek that cancels across legs (the gamma and volga of a synthetic
    # forward are exactly zero) is judged against the gross and the rounding of the differences instead:
    # |error| <= 1e-4 |net| + 1e-7 x (sum over legs of |Greek|) + 1e-10 (rounding, about 1e-12 here).
    market = MARKETS[market_name]
    strategy = _template(name, market)
    risk = S.position_risk(strategy, market)
    numeric = _finite_differences(strategy, market)
    gross = _gross(strategy, market)
    for greek, value in numeric.items():
        exact = getattr(risk, greek)
        assert abs(exact - value) <= 1e-4 * abs(exact) + 1e-7 * gross[greek] + 1e-10, (greek, exact, value)


def test_book_greeks_are_derivatives_of_position_value():
    market = S.Market(100, 0.045, 0.18, 30 / 365, 0.013)
    strategy = _book()
    risk = S.position_risk(strategy, market)
    for greek, value in _finite_differences(strategy, market).items():
        assert getattr(risk, greek) == pytest.approx(value, rel=1e-4), greek


@pytest.mark.oracle
def test_book_first_order_greeks_match_quantlib():
    # Independent oracle: QuantLib's AnalyticEuropeanEngine, leg by leg, summed and converted to desk units.
    ql = pytest.importorskip("QuantLib")
    market = S.Market(100, 0.045, 0.18, 30 / 365, 0.013)
    strategy = _book()
    today = ql.Date(2, 1, 2026)
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    spot = ql.SimpleQuote(market.spot)
    rates = ql.YieldTermStructureHandle(ql.FlatForward(today, market.rate, day_count))
    dividends = ql.YieldTermStructureHandle(ql.FlatForward(today, market.dividend_yield, day_count))
    totals = dict.fromkeys(("value", "delta", "gamma", "vega", "theta", "rho"), 0.0)
    for leg in strategy.legs:
        if not leg.is_option:
            totals["value"] += leg.quantity * market.spot
            totals["delta"] += leg.quantity
            totals["theta"] += leg.quantity * market.dividend_yield * market.spot / 365  # dividend accrual at t = 0
            continue
        days = round(leg.expiry * 365)
        vol = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), leg.volatility, day_count))
        process = ql.BlackScholesMertonProcess(ql.QuoteHandle(spot), dividends, rates, vol)
        kind = ql.Option.Call if leg.kind == "call" else ql.Option.Put
        option = ql.EuropeanOption(ql.PlainVanillaPayoff(kind, leg.strike), ql.EuropeanExercise(today + days))
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        totals["value"] += leg.quantity * option.NPV()
        totals["delta"] += leg.quantity * option.delta()
        totals["gamma"] += leg.quantity * option.gamma()
        totals["vega"] += leg.quantity * option.vega() / 100
        totals["theta"] += leg.quantity * option.theta() / 365
        totals["rho"] += leg.quantity * option.rho() / 100
    risk = S.position_risk(strategy, market)
    assert risk.value == pytest.approx(totals["value"], rel=1e-12)
    assert risk.delta == pytest.approx(totals["delta"], rel=1e-12)
    assert risk.dollar_delta == pytest.approx(totals["delta"] * market.spot, rel=1e-12)
    assert risk.gamma == pytest.approx(totals["gamma"], rel=1e-12)
    assert risk.gamma_1pct == pytest.approx(totals["gamma"] * market.spot * 0.01, rel=1e-12)
    assert risk.vega == pytest.approx(totals["vega"], rel=1e-12)
    assert risk.theta == pytest.approx(totals["theta"], rel=1e-10)
    assert risk.rho == pytest.approx(totals["rho"], rel=1e-12)


@pytest.mark.parametrize("t", [0.0, 0.1, 2.0])
def test_shares_have_delta_equal_to_quantity_and_no_curvature(t):
    risk = S.position_risk(S.Strategy("stock", (S.Leg("stock", -37.5),)), MARKET, 120.0, t)
    assert risk.delta == -37.5 and risk.dollar_delta == -37.5 * 120.0
    for name in ("gamma", "gamma_1pct", "dollar_gamma_1pct", "vega", "vanna", "volga", "charm", "veta"):
        assert getattr(risk, name) == 0.0, name


def test_multiplier_scales_every_field_but_spot():
    strategy = _book()
    one = S.position_risk(strategy, MARKET, 101.0, 0.01)
    many = S.position_risk(strategy, MARKET, 101.0, 0.01, multiplier=250.0)
    assert many.spot == one.spot == 101.0
    for name in FIELDS[1:]:
        assert getattr(many, name) == pytest.approx(250.0 * getattr(one, name), rel=1e-14), name


def test_expired_legs_keep_only_their_intrinsic_delta():
    # Past the expiry of every leg the position is a payoff: delta is the number of in-the-money options.
    strategy = S.build_template("Bull Call Spread", 100, 10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        risk = S.position_risk(strategy, MARKET, 105.0, MARKET.expiry + 0.01)
    assert risk.delta == 1.0
    assert risk.value == pytest.approx(5.0)
    for name in ("gamma", "vega", "theta", "vanna", "volga", "veta"):
        assert getattr(risk, name) == 0.0, name


def test_position_risk_is_frozen_and_uses_snake_case_fields():
    risk = S.position_risk(_book(), MARKET)
    assert FIELDS == [
        "spot",
        "value",
        "delta",
        "dollar_delta",
        "gamma",
        "gamma_1pct",
        "dollar_gamma_1pct",
        "vega",
        "theta",
        "rho",
        "vanna",
        "volga",
        "charm",
        "veta",
    ]
    assert all(isinstance(getattr(risk, name), float) for name in FIELDS)
    with pytest.raises(dataclasses.FrozenInstanceError):
        risk.delta = 0.0  # type: ignore[misc]


def test_dollar_gamma_is_the_convexity_of_a_one_percent_move():
    # For a long straddle the symmetric part of the P&L of +/-1% is the convexity term (to third order).
    strategy = S.build_template("Long Straddle", 100, 5)
    market = S.Market(100, 0.05, 0.2, 1.0)
    risk = S.position_risk(strategy, market)
    up, down, flat = (float(S.position_value(strategy, market, s)) for s in (101.0, 99.0, 100.0))
    assert (up + down) / 2 - flat == pytest.approx(risk.dollar_gamma_1pct, rel=1e-3)
    assert np.sign(risk.dollar_gamma_1pct) == 1
