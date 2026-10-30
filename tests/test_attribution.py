"""P&L explain (RK-3): the Greek waterfall, the Shapley split over factors and the P&L attribution test statistics.

The Taylor terms are checked against derivatives of Black-Scholes-Merton prices written out here in 40-digit mpmath,
the Shapley parts against the average of the 24 sequential attributions computed here (a different formula for the
same value), and the PLA statistics against SciPy's own Spearman and Kolmogorov-Smirnov functions. The acceptance
items of the plan (ACC 1 to 7) are each a test below, named after what they check.

The plan does not state the market its acceptance numbers were measured on; they reproduce on this one (GOLDEN):
spot and strikes around 100, vol 20 %, rate 4 %, no dividends, 30 days to expiry. The app's textbook market (rate
5 %, a year to expiry) does not reproduce them (see the golden stress day).
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import re
import warnings
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import stats

from options_analyzer import attribution as A
from options_analyzer import black_scholes as bs
from options_analyzer import strategies as S

GOLDEN = S.Market(100, 0.04, 0.20, 30 / 365)
STRADDLE = S.build_template("Short Straddle", 100, 5)
CONDOR = S.build_template("Iron Condor", 100, 5)  # 90 / 95 / 105 / 110
LOT = 100  # shares per contract
#: Markets for the checks over every template: the plan's, the app's textbook one (one year, r 5 %) and a 45-day
#: index-like one with dividends.
MARKETS = {
    "30d": GOLDEN,
    "1Y": S.Market(100, 0.05, 0.20, 1.0),
    "45d, q": S.Market(100, 0.043, 0.16, 45 / 365, 0.013),
}
#: Moves: (spot, vol shift, rate shift, days); the last one runs past the front expiry of every 30-day position.
MOVES = [(96, 0.06, 0.0, 1), (103, -0.02, 0.0025, 3), (88, 0.15, -0.005, 7), (101, 0.01, 0.0, 40)]


def _templates(market: S.Market):
    """Every template, centred on the market's spot with strikes 5 % apart (100 and 5 on the 100 markets)."""
    for name in S.TEMPLATES:
        yield name, S.build_template(name, market.spot, 0.05 * market.spot, market.expiry)


def _value(strategy: S.Strategy, market: S.Market, spot, t, vol_shift, rate_shift) -> float:
    return float(S.position_value(strategy, market, spot, t, vol_shift=vol_shift, rate_shift=rate_shift))


def _full(strategy: S.Strategy, market: S.Market, spot, vol_shift, rate_shift, days) -> float:
    """Full-revaluation P&L per unit, from position_value of the whole position (not leg by leg)."""
    return _value(strategy, market, spot, days / 365, vol_shift, rate_shift) - _value(strategy, market, None, 0, 0, 0)


# --- ACC 1, and order independence: the Shapley split ---------------------------------------------------------------


@pytest.mark.parametrize("market", MARKETS.values(), ids=MARKETS.keys())
def test_shapley_parts_add_up_to_the_full_revaluation_pnl(market):
    # ACC 1, within 1e-9 per unit. Rounding is all that separates them (measured worst 1.8e-15 over these 26
    # templates x 4 moves x 3 markets, positions worth up to about 100 per unit); the total is also the P&L of the
    # whole position revalued at once (worst 1.4e-14).
    for (name, strategy), (spot, vol, rate, days) in itertools.product(_templates(market), MOVES):
        explain = A.shapley_explain(strategy, market, spot, vol_shift=vol, rate_shift=rate, days=days)
        assert math.fsum(explain.terms.values()) == pytest.approx(explain.total, abs=1e-9), name
        assert explain.total == pytest.approx(_full(strategy, market, spot, vol, rate, days), abs=1e-9), name


def _sequential(strategy, market, spot, vol, rate, days, order) -> dict[str, float]:
    """Move the factors one after the other in ``order``; each gets the change in value when it moves."""
    end = {"spot": spot, "vol": vol, "time": days / 365, "rate": rate}
    state = {"spot": market.spot, "vol": 0.0, "time": 0.0, "rate": 0.0}
    parts, before = {}, _value(strategy, market, None, 0, 0, 0)
    for factor in order:
        state[factor] = end[factor]
        after = _value(strategy, market, state["spot"], state["time"], state["vol"], state["rate"])
        parts[factor], before = after - before, after
    return parts


@pytest.mark.parametrize("name", ["Short Straddle", "Iron Condor", "Risk Reversal", "Collar", "Long Call Calendar"])
def test_shapley_parts_are_the_average_of_the_24_sequential_attributions(name):
    # The Shapley value is the average contribution over every order of the factors: computed here that way, from
    # 24 x 4 revaluations of the whole position, against the module's 16 subsets with their weights (measured
    # agreement 6.2e-15 per unit; 1e-12 allowed).
    strategy = S.build_template(name, 100, 5, GOLDEN.expiry)
    move = (93, 0.08, 0.005, 2)
    orders = [_sequential(strategy, GOLDEN, *move, order) for order in itertools.permutations(A.SHAPLEY_FACTORS)]
    explain = A.shapley_explain(strategy, GOLDEN, move[0], vol_shift=move[1], rate_shift=move[2], days=move[3])
    for factor in A.SHAPLEY_FACTORS:
        average = math.fsum(parts[factor] for parts in orders) / len(orders)
        assert explain.terms[factor] == pytest.approx(average, abs=1e-12), factor


def test_a_sequential_attribution_depends_on_the_order_and_shapley_does_not():
    # The numbers of shapley_explain's docstring: on the golden stress day, one contract, the spot part of a
    # sequential attribution ranges from -$84.87 to -$59.72 over the 24 orders and the vol part from -$136.77 to
    # -$111.18; the Shapley parts sit inside those ranges and have no order to depend on.
    orders = [
        _sequential(STRADDLE, GOLDEN, 96, 0.06, 0.0, 1, order) for order in itertools.permutations(A.SHAPLEY_FACTORS)
    ]
    explain = A.shapley_explain(STRADDLE, GOLDEN, 96, vol_shift=0.06, days=1, multiplier=LOT)
    for factor, (low, high) in {"spot": (-84.87, -59.72), "vol": (-136.77, -111.18)}.items():
        parts = [LOT * order[factor] for order in orders]
        assert (min(parts), max(parts)) == pytest.approx((low, high), abs=0.005), factor
        assert low < explain.terms[factor] < high
    text = " ".join(A.shapley_explain.__doc__.split())
    assert "ranges from -$84.87 to -$59.72 per contract and the vol part from -$136.77 to -$111.18" in text


# --- ACC 2: the residual is third order ------------------------------------------------------------------------------


def _residuals(strategy, days: float, rate: float = 0.0) -> list[float]:
    """Unexplained P&L of a -4 % spot and +6 vol point move (and ``days``, ``rate``), halved three times."""
    return [
        A.taylor_explain(
            strategy, GOLDEN, 100 * (1 - 0.04 / 2**k), vol_shift=0.06 / 2**k, days=days / 2**k, rate_shift=rate / 2**k
        ).unexplained
        for k in range(4)
    ]


def test_the_second_order_residual_falls_eightfold_when_the_shocks_halve():
    # ACC 2: the iron condor's ratios lie in [7, 9]; they are the plan's measured 7.84, 8.10, 8.11 (to 0.005), and
    # a day of time decay halved along with the moves keeps them there (7.87, 8.11, 8.11): the theta term is exact
    # in time and charm and veta take the cross terms.
    for days, measured in [(0.0, (7.84, 8.10, 8.11)), (1.0, (7.87, 8.11, 8.11))]:
        residual = _residuals(CONDOR, days)
        ratios = [a / b for a, b in itertools.pairwise(residual)]
        assert all(7 <= ratio <= 9 for ratio in ratios), ratios
        assert ratios == pytest.approx(measured, abs=0.005)


def test_a_rate_move_leaves_second_order_terms_in_the_residual():
    # The expansion has rho but none of the rate's second-order terms (d2V/dr2, d2V/dSdr, d2V/dsigma dr), so with a
    # rate move the residual tends to fall 4x, not 8x, per halving: 7.64, 7.65, 7.29 with +25 bp (measured).
    residual = _residuals(CONDOR, 1.0, rate=0.0025)
    ratios = [a / b for a, b in itertools.pairwise(residual)]
    assert ratios == pytest.approx((7.64, 7.65, 7.29), abs=0.005)


# --- ACC 3, ACC 4 and the dummy property: no move, no part ----------------------------------------------------------


@pytest.mark.parametrize("market", MARKETS.values(), ids=MARKETS.keys())
def test_zero_shocks_give_all_terms_zero(market):
    # ACC 3, exactly: with nothing moving, every revaluation is today's value.
    for name, strategy in _templates(market):
        taylor = A.taylor_explain(strategy, market, multiplier=LOT)
        assert {**taylor.terms, "unexplained": taylor.unexplained, "total": taylor.total} == dict.fromkeys(
            (*A.TAYLOR_TERMS, "unexplained", "total"), 0.0
        ), name
        assert taylor.quality == 0.0
        shapley = A.shapley_explain(strategy, market, market.spot, vol_shift=0.0, rate_shift=0.0, days=0.0)
        assert dataclasses.astuple(shapley) == (0.0,) * 5, name
        # Plain floats, leg by leg too (not NumPy scalars, which print as np.float64(...)).
        legs = A.shapley_explain_by_leg(strategy, market, 101, days=1) + A.taylor_explain_by_leg(strategy, market, 101)
        assert all(type(value) is float for leg in legs for value in dataclasses.astuple(leg)), name


@pytest.mark.parametrize("spot", [70.0, 97.5, 100.0, 104.0, 140.0])
def test_a_pure_spot_shock_puts_all_the_shapley_pnl_on_spot(spot):
    # ACC 4: 100 % on spot. The other parts are exactly zero (their moves change nothing), and the spot part is the
    # total up to the rounding of the weights 1/4, 1/12, ... (measured 5e-16 relative).
    for name, strategy in _templates(GOLDEN):
        explain = A.shapley_explain(strategy, GOLDEN, spot, multiplier=LOT)
        assert (explain.vol, explain.time, explain.rate) == (0.0, 0.0, 0.0), name
        assert explain.spot == pytest.approx(explain.total, rel=1e-14, abs=1e-13), name


@given(
    name=st.sampled_from(sorted(S.TEMPLATES)),
    spot=st.floats(60.0, 150.0),
    vol=st.floats(-0.15, 0.5),
    rate=st.floats(-0.02, 0.02),
    days=st.floats(0.0, 60.0),
    still=st.sampled_from(A.SHAPLEY_FACTORS),
)
def test_a_factor_that_does_not_move_gets_nothing_and_the_parts_add_up(name, spot, vol, rate, days, still):
    # The Shapley "dummy" property for each factor in turn, on random moves of every template: a factor whose move
    # is zero gets exactly zero, and the parts still add up to the P&L (efficiency, within 1e-9 as in ACC 1).
    move = {"spot": spot, "vol": vol, "time": days, "rate": rate}
    move[still] = {"spot": GOLDEN.spot, "vol": 0.0, "time": 0.0, "rate": 0.0}[still]
    strategy = S.build_template(name, 100, 5, GOLDEN.expiry)
    explain = A.shapley_explain(
        strategy, GOLDEN, move["spot"], vol_shift=move["vol"], rate_shift=move["rate"], days=move["time"]
    )
    assert explain.terms[still] == 0.0
    assert math.fsum(explain.terms.values()) == pytest.approx(explain.total, abs=1e-9)


# --- ACC 5: one day of time alone ------------------------------------------------------------------------------------

#: A 1-day at-the-money index put (time_decay's example): worth 21.56, theta / 365 -10.66, decay -21.56.
SPX_1D = S.Market(5800, 0.043, 0.18, 1 / 365, 0.013)


@pytest.mark.parametrize("market", [*MARKETS.values(), SPX_1D], ids=[*MARKETS.keys(), "SPX 1 day"])
def test_one_day_of_time_alone_is_explained_by_theta_and_charm(market):
    # ACC 5: theta + charm within 1e-6 of position_value(t = 1/365) minus today's value, per unit (measured worst
    # 4.5e-13, on the index positions worth thousands). The theta term is that difference by repricing and the
    # charm term is zero without a spot move; nothing is left unexplained.
    for name, strategy in (*_templates(market), ("1-day put", S.Strategy("put", (S.Leg("put", 1, 5800),)))):
        if name == "1-day put" and market is not SPX_1D:
            continue
        explain = A.taylor_explain(strategy, market, days=1)
        decay = _value(strategy, market, None, 1 / 365, 0, 0) - _value(strategy, market, None, 0, 0, 0)
        assert explain.theta + explain.charm == pytest.approx(decay, abs=1e-6), name
        assert explain.charm == 0.0 and abs(explain.unexplained) <= 1e-6, name
        # The one-day decay the app shows (strategies.time_decay), as TaylorExplain.theta's docstring says.
        assert explain.theta == pytest.approx(S.time_decay(strategy, market), abs=1e-9), name


def test_theta_times_the_time_would_leave_the_time_convexity_unexplained():
    # Why the theta term reprices instead of using theta x days: theta / 365 misses (d2V/dt2) dt^2 / 2. On the
    # golden straddle that is 6.46e-4 per unit (0.84 % of its one-day decay of 0.0769); on time_decay's 1-day put it
    # is half the premium (theta / 365 -10.66 against the decay -21.56), which the theta term gets right.
    explain = A.taylor_explain(STRADDLE, GOLDEN, days=1)
    greek = S.position_risk(STRADDLE, GOLDEN).theta
    assert explain.theta - greek == pytest.approx(6.465e-4, abs=1e-6)
    assert (explain.theta - greek) / explain.theta == pytest.approx(0.0084, abs=1e-4)
    put = S.Strategy("put", (S.Leg("put", 1, 5800),))
    assert round(A.taylor_explain(put, SPX_1D, days=1).theta, 2) == -21.56
    assert round(S.position_risk(put, SPX_1D).theta, 2) == -10.66


# --- ACC 6: legs add up to the position ------------------------------------------------------------------------------


def _whole_shapley(strategy, market, spot, vol, rate, days) -> dict[str, float]:
    """Shapley parts of the whole position: 16 revaluations of all its legs at once, weights k! (3 - k)! / 4!."""
    end = {"spot": spot, "vol": vol, "time": days / 365, "rate": rate}
    today = {"spot": market.spot, "vol": 0.0, "time": 0.0, "rate": 0.0}

    def v(moved) -> float:
        state = {factor: (end if factor in moved else today)[factor] for factor in A.SHAPLEY_FACTORS}
        return _value(strategy, market, state["spot"], state["time"], state["vol"], state["rate"])

    parts = {}
    for factor in A.SHAPLEY_FACTORS:
        others = [f for f in A.SHAPLEY_FACTORS if f != factor]
        subsets = itertools.chain.from_iterable(itertools.combinations(others, k) for k in range(4))
        weight = {k: math.factorial(k) * math.factorial(3 - k) / 24 for k in range(4)}
        parts[factor] = math.fsum(weight[len(a)] * (v({*a, factor}) - v(set(a))) for a in subsets)
    return parts


@pytest.mark.parametrize("market", MARKETS.values(), ids=MARKETS.keys())
def test_leg_explains_add_up_to_the_position_explain(market):
    # ACC 6. The position's explains are the fsum of the legs' (exact by construction), so the sums of the legs are
    # also checked against the whole position done at once: its Greeks (position_risk of all the legs) times the
    # moves, its time decay and full P&L revalued at once, and the Shapley parts of 16 revaluations of all its legs.
    # Measured agreement 1.4e-14 per unit; 1e-11 allowed.
    for (name, strategy), (spot, vol, rate, days) in itertools.product(_templates(market), MOVES[:3]):
        kwargs = {"vol_shift": vol, "rate_shift": rate, "days": days}
        legs = A.taylor_explain_by_leg(strategy, market, spot, **kwargs)
        whole = A.taylor_explain(strategy, market, spot, **kwargs)
        risk = S.position_risk(strategy, market)
        d_spot, d_vol, d_rate = spot - market.spot, 100 * vol, 100 * rate
        reference = {
            "delta": risk.delta * d_spot,
            "gamma": 0.5 * risk.gamma * d_spot**2,
            "vega": risk.vega * d_vol,
            "theta": _value(strategy, market, None, days / 365, 0, 0) - _value(strategy, market, None, 0, 0, 0),
            "rho": risk.rho * d_rate,
            "vanna": risk.vanna * d_spot * d_vol,
            "volga": 0.5 * risk.volga * d_vol**2,
            "charm": risk.charm * d_spot * days,
            "veta": risk.veta * d_vol * days,
            "total": _full(strategy, market, spot, vol, rate, days),
        }
        reference["unexplained"] = reference["total"] - math.fsum(v for k, v in reference.items() if k != "total")
        for field, value in reference.items():
            assert math.fsum(getattr(leg, field) for leg in legs) == pytest.approx(value, abs=1e-11), (name, field)
            assert getattr(whole, field) == pytest.approx(value, abs=1e-11), (name, field)
        parts = A.shapley_explain_by_leg(strategy, market, spot, **kwargs)
        shapley = A.shapley_explain(strategy, market, spot, **kwargs)
        for factor, value in _whole_shapley(strategy, market, spot, vol, rate, days).items():
            assert math.fsum(getattr(p, factor) for p in parts) == pytest.approx(value, abs=1e-11), (name, factor)
            assert getattr(shapley, factor) == pytest.approx(value, abs=1e-11), (name, factor)


def test_per_leg_vol_shifts_move_each_leg_by_its_own_shift():
    # A skew move: the put's vol up 4 points and the call's down 1 on a risk reversal (short put, long call). Each
    # leg's explain is the one-leg position's under its own shift, and the stock leg of a collar ignores its entry.
    reversal = S.build_template("Risk Reversal", 100, 5)
    legs = A.taylor_explain_by_leg(reversal, GOLDEN, 98, vol_shift=(0.04, -0.01), days=1, multiplier=LOT)
    for leg, shift, explain in zip(reversal.legs, (0.04, -0.01), legs, strict=True):
        alone = A.taylor_explain(S.Strategy("leg", (leg,)), GOLDEN, 98, vol_shift=shift, days=1, multiplier=LOT)
        assert explain == alone
    collar = S.build_template("Collar", 100, 5)
    shifted = A.shapley_explain(collar, GOLDEN, 98, vol_shift=(0.3, 0.02, 0.01), days=1)
    assert shifted == A.shapley_explain(collar, GOLDEN, 98, vol_shift=(-0.1, 0.02, 0.01), days=1)


# --- ACC 7: the golden short straddle stress day --------------------------------------------------------------------


def test_golden_short_straddle_stress_day():
    # ACC 7: spot -4 % and implied vol +6 points over one day, one contract, on the plan's market. The full P&L is
    # the plan's -188.36. Its second-order residual, 27.04, is the P&L the terms leave unexplained *without* charm
    # and veta and with theta x days for the theta term: rebuilt here from this explain to 0.01. With charm (-0.46)
    # and veta (+2.26), and theta by repricing (+0.06 over theta / 365), $25.17 is left unexplained, 13.4 % of the
    # loss. (The app's textbook market, r 5 % and a year to expiry, gives -373.07 instead: not the plan's market.)
    explain = A.taylor_explain(STRADDLE, GOLDEN, 96, vol_shift=0.06, days=1, multiplier=LOT)
    assert explain.total == pytest.approx(-188.36, abs=0.01)
    assert explain.unexplained == pytest.approx(25.17, abs=0.01)
    greek_theta = S.position_risk(STRADDLE, GOLDEN, multiplier=LOT).theta
    plans = explain.unexplained + explain.charm + explain.veta + (explain.theta - greek_theta)
    assert plans == pytest.approx(27.04, abs=0.01)
    assert (round(explain.charm, 2), round(explain.veta, 2)) == (-0.46, 2.26)
    assert list(explain.terms) == list(A.TAYLOR_TERMS)
    assert explain.explained + explain.unexplained == pytest.approx(explain.total, abs=1e-12)
    assert explain.floor == pytest.approx(2.0, rel=1e-15)  # 1 bp of 2 x 100 shares x $100
    assert explain.quality == pytest.approx(25.17 / 188.36, abs=1e-4)
    textbook = A.taylor_explain(STRADDLE, S.Market(100, 0.05, 0.20, 1.0), 96, vol_shift=0.06, days=1, multiplier=LOT)
    assert textbook.total == pytest.approx(-373.07, abs=0.01)


# --- The terms against an independent reference ----------------------------------------------------------------------

mp.mp.dps = 40


def _mp_leg(leg: S.Leg, market: S.Market):
    """The leg's value as a function of (spot, vol shift, elapsed years, rate), in mpmath: BSM written out here."""
    strike = mp.mpf(leg.strike)
    expiry = mp.mpf(market.expiry if leg.expiry is None else leg.expiry)
    vol0 = mp.mpf(market.volatility if leg.volatility is None else leg.volatility)
    q, phi = mp.mpf(market.dividend_yield), 1 if leg.kind == "call" else -1

    def value(s, shift, t, r):
        tau, sigma = expiry - t, vol0 + shift
        d1 = (mp.log(s / strike) + (r - q + sigma**2 / 2) * tau) / (sigma * mp.sqrt(tau))
        d2 = d1 - sigma * mp.sqrt(tau)
        return phi * (s * mp.e ** (-q * tau) * mp.ncdf(phi * d1) - strike * mp.e ** (-r * tau) * mp.ncdf(phi * d2))

    return value


def _mp_terms(strategy, market, spot, shifts, rate_shift, days) -> dict[str, float]:
    """The explain's terms from mpmath derivatives at today's market (option legs only)."""
    totals = dict.fromkeys((*A.TAYLOR_TERMS, "total"), mp.mpf(0))
    s0, r0 = mp.mpf(market.spot), mp.mpf(market.rate)
    ds, dr, dt = mp.mpf(spot) - s0, mp.mpf(rate_shift), mp.mpf(days) / 365
    for leg, shift in zip(strategy.legs, shifts, strict=True):
        f, dv, at = _mp_leg(leg, market), mp.mpf(shift), (s0, mp.mpf(0), mp.mpf(0), r0)

        def d(*orders, f=f, at=at):
            return mp.diff(f, at, orders)

        today = f(*at)
        terms = {
            "delta": d(1, 0, 0, 0) * ds,
            "gamma": d(2, 0, 0, 0) * ds**2 / 2,
            "vega": d(0, 1, 0, 0) * dv,
            "theta": f(s0, 0, dt, r0) - today,
            "rho": d(0, 0, 0, 1) * dr,
            "vanna": d(1, 1, 0, 0) * ds * dv,
            "volga": d(0, 2, 0, 0) * dv**2 / 2,
            "charm": d(1, 0, 1, 0) * ds * dt,
            "veta": d(0, 1, 1, 0) * dv * dt,
            "total": f(s0 + ds, dv, dt, r0 + dr) - today,
        }
        for name, value in terms.items():
            totals[name] += leg.quantity * value
    return {name: float(value) for name, value in totals.items()}


ORACLE_CASES = {
    "golden straddle, +25 bp": (STRADDLE, GOLDEN, 96, (0.06, 0.06), 0.0025, 1),
    "risk reversal, skew move, dividends": (
        S.Strategy("rr", (S.Leg("put", -2, 90, volatility=0.24), S.Leg("call", 1, 110, 45 / 365, 0.17))),
        S.Market(100, 0.043, 0.20, 30 / 365, 0.013),
        103,
        (0.035, -0.01),
        -0.005,
        7,
    ),
    "one-year call, half a day": (S.Strategy("c", (S.Leg("call", 1, 95),)), MARKETS["1Y"], 99.5, (0.01,), 0.0, 0.5),
}


@pytest.mark.parametrize("case", ORACLE_CASES.values(), ids=ORACLE_CASES.keys())
def test_terms_match_high_precision_derivatives(case):
    # Every term, in the units it is reported in, against mpmath derivatives of the prices at 40 digits: catches a
    # wrong unit (vol points, per day, per 1 %) or sign convention, which an internal consistency check would not.
    # Measured worst gap 1.5e-14 per unit (2.7e-13 relative, on the smallest terms); allowed, whichever is larger,
    # 1e-11 per unit or 1e-12 relative.
    strategy, market, spot, shifts, rate, days = case
    explain = A.taylor_explain(strategy, market, spot, vol_shift=shifts, rate_shift=rate, days=days)
    reference = _mp_terms(strategy, market, spot, shifts, rate, days)
    for name, value in reference.items():
        got = explain.total if name == "total" else explain.terms[name]
        assert got == pytest.approx(value, rel=1e-12, abs=1e-11), name


def test_the_spot_rate_cross_term_left_in_the_residual():
    # taylor_explain's docstring: on the golden day with a 25 bp rate rise, the omitted (d2V/dSdr) dS dr is $1.14 a
    # contract (mpmath), larger than the rho term itself (-$0.05): a straddle's rho is nearly zero, its delta's
    # sensitivity to the rate is not.
    at = (mp.mpf(100), mp.mpf(0), mp.mpf(0), mp.mpf("0.04"))  # spot, vol shift, elapsed years, rate
    cross = sum(leg.quantity * mp.diff(_mp_leg(leg, GOLDEN), at, (1, 0, 0, 1)) for leg in STRADDLE.legs)
    assert float(LOT * cross * -4 * mp.mpf("0.0025")) == pytest.approx(1.14, abs=0.005)
    explain = A.taylor_explain(STRADDLE, GOLDEN, 96, vol_shift=0.06, rate_shift=0.0025, days=1, multiplier=LOT)
    assert round(explain.rho, 2) == -0.05
    text = " ".join(A.taylor_explain.__doc__.split())
    assert "with a 25 bp rate rise, $1.14 per contract, against a rho term of -$0.05" in text


# --- Quality, stock legs, expiring legs and the inputs ---------------------------------------------------------------


def test_quality_uses_the_floor_when_the_pnl_is_close_to_zero():
    # A 1 % spot rise that a 0.614-point vol fall offsets: the long straddle makes $0.0073 a contract and the Greeks
    # miss $0.144, 20 times the P&L, a meaningless ratio. Against the floor, 1 bp of the $20,000 notional ($2), the
    # quality reads 0.072.
    long = S.build_template("Long Straddle", 100, 5)
    explain = A.taylor_explain(long, GOLDEN, 101, vol_shift=-0.00614, multiplier=LOT)
    assert (explain.total, explain.unexplained) == pytest.approx((0.0073, 0.1440), abs=5e-5)
    assert explain.floor == pytest.approx(2.0, rel=1e-15)
    assert explain.quality == pytest.approx(explain.unexplained / 2.0, rel=1e-15)
    given = A.taylor_explain(long, GOLDEN, 101, vol_shift=-0.00614, multiplier=LOT, floor=0.05)
    assert given.floor == 0.05 and given.quality == pytest.approx(explain.unexplained / 0.05, rel=1e-15)
    # A P&L above the floor is its own scale; each leg's floor is its own notional, $1 here.
    assert A.taylor_explain(long, GOLDEN, 101, vol_shift=-0.00614, multiplier=LOT, floor=1e-4).quality == (
        pytest.approx(explain.unexplained / explain.total, rel=1e-15)
    )
    assert [leg.floor for leg in A.taylor_explain_by_leg(long, GOLDEN, 101, multiplier=LOT)] == pytest.approx([1, 1])


def test_stock_legs_have_delta_and_dividend_accrual_only():
    # A covered call on a dividend payer: the share's explain is its spot move (delta 1) and its dividends (theta),
    # with no vol or rate terms and nothing unexplained.
    market = S.Market(100, 0.05, 0.20, 30 / 365, 0.03)
    stock, call = A.taylor_explain_by_leg(S.build_template("Covered Call", 100, 5), market, 97, vol_shift=0.02, days=2)
    accrued = 100 * (math.exp(0.05 * 2 / 365) - math.exp(0.02 * 2 / 365))
    assert stock.delta == -3.0 and stock.theta == pytest.approx(accrued, rel=1e-12)
    assert stock.total == pytest.approx(-3.0 + accrued, rel=1e-12)
    others = {name: value for name, value in stock.terms.items() if name not in ("delta", "theta")}
    assert others == dict.fromkeys(others, 0.0)
    assert abs(stock.unexplained) < 1e-12 and call.vega != 0.0


def test_a_leg_that_expires_within_the_period_shows_in_the_quality():
    # A 30/60-day calendar over 40 days: the front leg is settled at its intrinsic value, which no expansion of
    # today's value follows. The Greeks miss 31 % of that leg's P&L ($45.39 of $145.13) against 1.3 % of the back
    # leg's, and the quality says so; the Shapley split still adds up.
    calendar = S.build_template("Long Call Calendar", 100, 5, 30 / 365)
    front, back = A.taylor_explain_by_leg(calendar, GOLDEN, 101, days=40, multiplier=LOT)
    assert (front.unexplained, front.total) == pytest.approx((-45.39, 145.13), abs=0.005)
    assert (front.quality, back.quality) == pytest.approx((0.313, 0.013), abs=0.0005)
    shapley = A.shapley_explain(calendar, GOLDEN, 101, days=40, multiplier=LOT)
    assert math.fsum(shapley.terms.values()) == pytest.approx(shapley.total, abs=1e-9)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"spot": -1.0}, "spot must be a non-negative price"),
        ({"spot": math.nan}, "spot must be a finite number"),
        ({"days": -1}, "days must be a non-negative number of calendar days"),
        ({"days": math.inf}, "days must be a finite number"),
        ({"rate_shift": "25bp"}, "rate_shift must be a number"),
        ({"vol_shift": (0.01,)}, r"vol_shift needs one shift per leg \(2\), got 1"),
        ({"vol_shift": (0.01, math.nan)}, "vol_shift must be a finite number"),
        ({"vol_shift": -0.2}, r"vol_shift -0.2 takes the implied vol of leg 1 \(Short 1 × 100 Call, 0.2\) to zero"),
        ({"multiplier": math.nan}, "multiplier must be a finite number"),
        ({"floor": 0.0}, "floor must be a positive amount of P&L"),
    ],
)
def test_bad_inputs_are_refused_naming_the_argument(kwargs, message):
    for explain in (A.taylor_explain, A.taylor_explain_by_leg, A.shapley_explain, A.shapley_explain_by_leg):
        if "floor" in kwargs and explain.__name__.startswith("shapley"):
            continue
        with pytest.raises(ValueError, match=message):
            explain(STRADDLE, GOLDEN, **kwargs)


def test_no_runtime_warning_at_the_edges():
    # Spot to zero, a vol shift to just above zero, a zero multiplier: finite results, no warning (RuntimeWarning is
    # an error in this suite anyway; the explicit filter documents the intent).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        crash = A.taylor_explain(STRADDLE, GOLDEN, 0.0, vol_shift=-0.1999, days=1)
        assert all(math.isfinite(value) for value in dataclasses.astuple(crash))
        nothing = A.taylor_explain(STRADDLE, GOLDEN, 96, vol_shift=0.06, multiplier=0.0)
        assert dataclasses.astuple(nothing) == (0.0,) * 13


# --- PLA statistics --------------------------------------------------------------------------------------------------


def test_pla_metrics_match_scipy():
    # Against SciPy's own implementations (spearmanr: Pearson correlation of average ranks; ks_2samp: the two-sample
    # KS statistic) on 250-day series, a near copy (RTPL = HPL + noise) and an unrelated one, with ties: to 1e-12.
    rng = np.random.default_rng(3)
    hpl = np.round(rng.standard_t(4, 250) * 1000, -1)  # ties: rounded to $10
    for rtpl in (np.round(hpl + rng.normal(0, 300, 250), -1), rng.normal(0, 1000, 250)):
        metrics = A.pla_metrics(hpl, rtpl)
        assert metrics.spearman == pytest.approx(stats.spearmanr(hpl, rtpl).statistic, abs=1e-12)
        assert metrics.ks == pytest.approx(stats.ks_2samp(hpl, rtpl).statistic, abs=1e-12)
        assert metrics.observations == 250
        # MAR32.39-32.41: the empirical CDFs step by 0.004 (1 / 250), so the statistic is a multiple of it.
        assert metrics.ks / 0.004 == pytest.approx(round(metrics.ks / 0.004), abs=1e-9)


def test_pla_metrics_by_hand():
    # Ranks (2, 1, 3, 4) and (3, 1, 2, 4): Spearman 1 - 6 x 2 / (4 x 15) = 0.8. The CDFs differ most from 0.6 to
    # 0.65, where HPL has 3 of its 4 observations at or below and RTPL 1 of 4: KS = 0.5.
    metrics = A.pla_metrics([0.5, -1.0, 0.6, 2.0], [0.7, -1.0, 0.65, 2.0])
    assert metrics == A.PLAMetrics(spearman=pytest.approx(0.8, abs=1e-15), ks=0.5, observations=4)
    assert A.pla_metrics([1, 2, 3], [3, 2, 1]).spearman == pytest.approx(-1.0, abs=1e-15)
    assert A.pla_metrics([1, 2, 3], [1, 2, 3]) == A.PLAMetrics(1.0, 0.0, 3)


def test_pla_metrics_of_a_constant_series_has_no_correlation():
    # A constant series has no ranking: NaN, without the warning SciPy's spearmanr raises. Every RTPL value lies
    # below the constant HPL, so the distributions are as far apart as they can be: KS = 1.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metrics = A.pla_metrics([5.0, 5.0, 5.0], [1.0, 2.0, 3.0])
    assert math.isnan(metrics.spearman) and metrics.ks == 1.0


@pytest.mark.parametrize(
    ("hpl", "rtpl", "message"),
    [
        ([1, 2, 3], [1, 2], "hpl and rtpl must cover the same days, got 3 and 2 observations"),
        ([1], [1], "needs at least 2 days, got 1"),
        ([[1, 2], [3, 4]], [1, 2], r"hpl must be one-dimensional, got shape \(2, 2\)"),
        ([1, math.nan], [1, 2], "hpl must be finite numbers"),
        ([1, 2], ["a", "b"], "rtpl must be a sequence of numbers"),
    ],
)
def test_pla_metrics_refuse_bad_series(hpl, rtpl, message):
    with pytest.raises(ValueError, match=message):
        A.pla_metrics(hpl, rtpl)


# --- The Methodology quotes the golden case -------------------------------------------------------------------------


def test_the_methodology_quotes_the_golden_stress_day():
    path = Path(A.__file__).parent / "ui" / "methodology" / "risk.md"
    body = " ".join(path.read_text(encoding="utf-8").split())
    explain = A.taylor_explain(STRADDLE, GOLDEN, 96, vol_shift=0.06, days=1, multiplier=LOT)
    shapley = A.shapley_explain(STRADDLE, GOLDEN, 96, vol_shift=0.06, days=1, multiplier=LOT)
    assert rf"loses \${-explain.total:.2f}" in body
    assert rf"leave \${explain.unexplained:.2f} unexplained ({100 * explain.quality:.1f} %)" in body
    shares = re.findall(r"spot −\\\$([\d.]+), vol −\\\$([\d.]+) and time \+\\\$([\d.]+)", body)
    assert shares == [(f"{-shapley.spot:.2f}", f"{-shapley.vol:.2f}", f"{shapley.time:.2f}")]
    # The largest piece of the residual: zomma's term, from the engine's zomma (checked against mpmath in
    # tests/test_black_scholes.py), per contract.
    zomma = sum(
        leg.quantity * bs.greeks(100, leg.strike, GOLDEN.expiry, 0.04, 0.20, 0.0, leg.kind)["zomma"]
        for leg in STRADDLE.legs
    )
    assert rf"is +\${0.5 * zomma * 4**2 * 0.06 * LOT:.2f})" in body
    sequential = [
        _sequential(STRADDLE, GOLDEN, 96, 0.06, 0.0, 1, order)["spot"]
        for order in itertools.permutations(A.SHAPLEY_FACTORS)
    ]
    assert rf"anything from −\${-LOT * min(sequential):.2f} to −\${-LOT * max(sequential):.2f}" in body


def _dollars(value: float) -> str:
    return f"{value:.2f}".replace("-", "−")


def test_the_method_note_quotes_the_golden_stress_day():
    # docs/methods/pnl_explain.md: its table of terms and its Shapley parts are this explain's, to the cent.
    note = (Path(__file__).parents[1] / "docs" / "methods" / "pnl_explain.md").read_text(encoding="utf-8")
    explain = A.taylor_explain(STRADDLE, GOLDEN, 96, vol_shift=0.06, days=1, multiplier=LOT)
    for name, value in {**explain.terms, "unexplained": explain.unexplained}.items():
        assert f"| {name} | {_dollars(value).replace('−0.00', '0.00')} |" in note, name
    assert f"| **full P&L** | **{_dollars(explain.total)}** |" in note
    assert f"The quality is {explain.quality:.3f}." in note
    shapley = A.shapley_explain(STRADDLE, GOLDEN, 96, vol_shift=0.06, days=1, multiplier=LOT)
    parts = rf"spot −\${-shapley.spot:.2f}, vol −\${-shapley.vol:.2f}, time +\${shapley.time:.2f} and rate 0"
    assert parts in " ".join(note.split())
