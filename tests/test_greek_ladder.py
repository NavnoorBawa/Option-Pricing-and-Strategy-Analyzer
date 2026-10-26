"""scenarios.greek_ladder: full-revaluation spot ladder with the position's risk at each rung (RK-1)."""

from __future__ import annotations

import dataclasses
import math
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from options_analyzer import black_scholes as bs
from options_analyzer import scenarios as sc
from options_analyzer import strategies as S

MARKETS = {
    "3M": S.Market(100, 0.05, 0.2, 0.25, 0.01),
    "45d": S.Market(100, 0.04, 0.25, 45 / 365, 0.015),
}
COLUMNS = [
    "shock",
    "spot_change",
    "spot",
    "pnl",
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
NUMERIC = COLUMNS[3:]  # the columns that add up over legs


def _template(name: str, market: S.Market) -> S.Strategy:
    return S.build_template(name, 100, 5, market.expiry)


def test_default_ladder_shape_and_columns():
    market = MARKETS["3M"]
    ladder = sc.greek_ladder(_template("Iron Condor", market), market)
    assert list(ladder.columns) == COLUMNS
    assert list(ladder["shock"]) == list(sc.DEFAULT_LADDER_SHOCKS)
    assert ladder["spot"].to_numpy() == pytest.approx(100 * (1 + np.array(sc.DEFAULT_LADDER_SHOCKS) / 100))
    assert isinstance(ladder.index, pd.RangeIndex)


@pytest.mark.parametrize("name", list(S.TEMPLATES))
def test_zero_row_is_position_risk_with_zero_pnl(name):
    # RK-1 acceptance (1): the 0% row equals position_risk to 1e-12, with P&L exactly 0.
    market = MARKETS["3M"]
    strategy = _template(name, market)
    ladder = sc.greek_ladder(strategy, market, multiplier=100)
    row = ladder.loc[ladder["shock"] == 0].iloc[0]
    assert row["pnl"] == 0.0
    risk = S.position_risk(strategy, market, multiplier=100)
    for field in dataclasses.fields(S.PositionRisk):
        assert row[field.name] == pytest.approx(getattr(risk, field.name), rel=1e-12, abs=1e-12), field.name


def _richardson(ladder: pd.DataFrame, column: str) -> float:
    """Central difference of ``column`` over +/-1%, with the +/-2% rows cancelling its h^3 error.

    D(h) = (f(h) - f(-h)) / 2 = f'h + f'''h^3 / 6 + O(h^5), so (8 D(1%) - D(2%)) / 6 = f' x 1% + O(h^5).
    """
    by_shock = ladder.set_index("shock")[column]
    d1 = (by_shock[1] - by_shock[-1]) / 2
    d2 = (by_shock[2] - by_shock[-2]) / 2
    return (8 * d1 - d2) / 6


@pytest.mark.parametrize("market_name", list(MARKETS))
@pytest.mark.parametrize("name", list(S.TEMPLATES))
def test_ladder_differences_match_dollar_delta_and_one_percent_gamma(name, market_name):
    # RK-1 acceptance (2): the central difference of ladder P&L over +/-1% matches $delta x 0.01, and that of delta
    # matches 1% gamma. A plain +/-1% difference carries the Taylor remainder f'''h^2/6 relative to f', which is
    # measured at up to 1.4e-2 (3-month butterflies) and 0.30 (45-day iron butterfly, net delta 2e-4) and so cannot
    # meet 1e-3; the ladder's own +/-2% rows cancel that term (Richardson). Tolerance: 1e-3 of the net Greek, or
    # 1e-5 of the gross (the sum over legs of |Greek|) where the legs nearly cancel. Measured: at most 2.1e-4 of the
    # net (3M) and 1.4e-5 of the gross (45d).
    market = MARKETS[market_name]
    strategy = _template(name, market)
    ladder = sc.greek_ladder(strategy, market)
    risk = S.position_risk(strategy, market)
    singles = [S.position_risk(S.Strategy("leg", (leg,)), market) for leg in strategy.legs]
    gross_delta = sum(abs(r.dollar_delta) for r in singles) * 0.01
    gross_gamma = sum(abs(r.gamma_1pct) for r in singles)
    pnl_slope, delta_slope = _richardson(ladder, "pnl"), _richardson(ladder, "delta")
    assert abs(pnl_slope - risk.dollar_delta * 0.01) <= 1e-3 * abs(risk.dollar_delta * 0.01) + 1e-5 * gross_delta
    assert abs(delta_slope - risk.gamma_1pct) <= 1e-3 * abs(risk.gamma_1pct) + 1e-5 * gross_gamma + 1e-14


@pytest.mark.parametrize("name", ["Long Call", "Short Put", "Covered Call", "Long Straddle", "Risk Reversal"])
def test_plain_one_percent_difference_where_the_taylor_remainder_is_small(name):
    # The literal +/-1% identity holds to 1e-3 where the third derivative is small relative to delta: one-year
    # options and the positions built from them.
    market = S.Market(100, 0.05, 0.2, 1.0)
    strategy = _template(name, market)
    ladder = sc.greek_ladder(strategy, market, (-1, 0, 1)).set_index("shock")
    risk = S.position_risk(strategy, market)
    assert (ladder.loc[1, "pnl"] - ladder.loc[-1, "pnl"]) / 2 == pytest.approx(risk.dollar_delta * 0.01, rel=1e-3)


@pytest.mark.parametrize("name", list(S.TEMPLATES))
def test_ladder_of_a_sum_of_legs_is_the_sum_of_their_ladders(name):
    # RK-1 acceptance (4).
    market = MARKETS["45d"]
    strategy = _template(name, market)
    total = sc.greek_ladder(strategy, market, roll_days=3)
    parts = [sc.greek_ladder(S.Strategy("leg", (leg,)), market, roll_days=3) for leg in strategy.legs]
    summed = sum(part[NUMERIC] for part in parts)
    for column in NUMERIC:
        scale = sum(part[column].abs() for part in parts)
        assert np.all(np.abs(total[column] - summed[column]) <= 1e-12 * np.maximum(scale, 1.0)), column
    for column in ("shock", "spot_change", "spot"):
        assert total[column].tolist() == parts[0][column].tolist()


@pytest.mark.parametrize("roll_days", [0, 10, 400])
def test_stock_only_ladder(roll_days):
    # RK-1 acceptance (5): for stock only, delta = quantity and gamma = vega = 0 in every row.
    market = MARKETS["3M"]
    ladder = sc.greek_ladder(S.Strategy("shares", (S.Leg("stock", 40),)), market, roll_days=roll_days)
    assert (ladder["delta"] == 40.0).all()
    assert (ladder["gamma"] == 0.0).all() and (ladder["vega"] == 0.0).all()
    if roll_days == 0:
        # A share's P&L is exactly quantity x spot change.
        assert ladder["pnl"].to_numpy() == pytest.approx(40 * 100 * ladder["spot_change"].to_numpy(), rel=1e-14)


def test_pnl_is_an_independent_full_revaluation():
    # Re-price every leg with black_scholes directly at the shocked spot and the rolled time.
    market = MARKETS["3M"]
    strategy = S.Strategy(
        "book",
        (S.Leg("call", 2, 105, 60 / 365, 0.18), S.Leg("put", -3, 95, 20 / 365, 0.24), S.Leg("call", -1, 100)),
    )
    roll = 5.5
    ladder = sc.greek_ladder(strategy, market, roll_days=roll, multiplier=100)

    def book(spot, t):
        total = 0.0
        for leg in strategy.legs:
            expiry = market.expiry if leg.expiry is None else leg.expiry
            vol = market.volatility if leg.volatility is None else leg.volatility
            total += leg.quantity * bs.price(spot, leg.strike, expiry - t, 0.05, vol, 0.01, leg.kind)
        return 100 * total

    expected = [book(spot, roll / 365) - book(100.0, 0.0) for spot in ladder["spot"]]
    assert ladder["pnl"].to_numpy() == pytest.approx(expected, rel=1e-12, abs=1e-10)
    # The same numbers as the time-decay P&L grid at that date.
    grid = sc.pnl_grid(strategy, market, ladder["spot_change"].to_numpy(), days=roll, multiplier=100)
    assert ladder["pnl"].to_numpy() == pytest.approx(grid[0], rel=1e-12, abs=1e-10)


def test_rolled_ladder_reports_the_risk_at_the_later_date():
    market = MARKETS["3M"]
    strategy = _template("Long Straddle", market)
    ladder = sc.greek_ladder(strategy, market, (-2, 0, 2), roll_days=30)
    row = ladder.set_index("shock").loc[0]
    later = S.position_risk(strategy, market, 100.0, 30 / 365)
    assert row["gamma"] == later.gamma and row["theta"] == later.theta
    assert row["pnl"] < 0  # a long straddle bleeds time value at an unchanged spot


def test_sigma_shocks():
    market = MARKETS["45d"]
    strategy = _template("Short Strangle", market)
    ladder = sc.greek_ladder(strategy, market, (-3, -1, 0, 1, 3), unit="sigma", sigma_days=5)
    sigma = 0.25 * math.sqrt(5 / 365)
    assert ladder["spot_change"].to_numpy() == pytest.approx(np.array([-3, -1, 0, 1, 3]) * sigma, rel=1e-15)
    assert ladder["spot"].to_numpy() == pytest.approx(100 * (1 + ladder["spot_change"].to_numpy()), rel=1e-15)
    one_day = sc.greek_ladder(strategy, market, (1,), unit="sigma")
    assert one_day["spot_change"].iloc[0] == pytest.approx(0.25 / math.sqrt(365), rel=1e-15)


def test_invalid_ladders_are_rejected():
    market = MARKETS["3M"]
    strategy = _template("Long Call", market)
    with pytest.raises(ValueError, match="unit"):
        sc.greek_ladder(strategy, market, unit="bp")
    with pytest.raises(ValueError, match="zero or below"):
        sc.greek_ladder(strategy, market, (-100,))
    with pytest.raises(ValueError, match="roll_days"):
        sc.greek_ladder(strategy, market, roll_days=-1)
    with pytest.raises(ValueError, match="roll_days"):
        sc.greek_ladder(strategy, market, roll_days=math.nan)
    with pytest.raises(ValueError, match="sigma_days"):
        sc.greek_ladder(strategy, market, unit="sigma", sigma_days=0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"shocks": (1, math.nan)}, "shock must be a finite number, got nan"),
        ({"shocks": (math.inf,)}, "shock must be a finite number, got inf"),
        ({"shocks": (-math.inf,)}, "shock must be a finite number, got -inf"),
        ({"shocks": (1, "2")}, "shock must be a number, got '2'"),
        ({"shocks": (None,)}, "shock must be a number, got None"),
        ({"shocks": ([1, 2],)}, "shock must be a single number"),
        ({"shocks": 5}, "shocks must be a sequence of numbers, got 5"),
        ({"roll_days": np.array([1.0, 2.0])}, "roll_days must be a single number"),
        ({"roll_days": math.inf}, "roll_days must be a finite number"),
        ({"roll_days": "1"}, "roll_days must be a number"),
        ({"sigma_days": math.nan, "unit": "sigma"}, "sigma_days must be a finite number"),
        ({"sigma_days": np.array([1.0]), "unit": "sigma"}, "sigma_days must be a single number"),
    ],
)
def test_non_numbers_are_rejected_by_name(kwargs, message):
    # Every argument that is not a single finite number raises ValueError naming it (a NaN shock used to be reported
    # as taking spot below zero, an array roll_days raised TypeError).
    market = MARKETS["3M"]
    with pytest.raises(ValueError, match=re.escape(message)):
        sc.greek_ladder(_template("Long Call", market), market, **kwargs)


def test_numpy_shocks_and_days_are_accepted():
    market = MARKETS["3M"]
    strategy = _template("Long Straddle", market)
    plain = sc.greek_ladder(strategy, market, (-1, 0, 1), roll_days=2)
    from_numpy = sc.greek_ladder(strategy, market, np.array([-1.0, 0.0, 1.0]), roll_days=np.float64(2.0))
    assert np.array_equal(plain[NUMERIC].to_numpy(), from_numpy[NUMERIC].to_numpy())


def test_ladder_through_expiry_has_no_warnings():
    # Rolling past a leg's expiry settles it at intrinsic value; nothing divides by a zero time to expiry.
    market = S.Market(100, 0.05, 0.2, 7 / 365)
    strategy = S.build_template("Long Call Calendar", 100, 5, market.expiry)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ladder = sc.greek_ladder(strategy, market, roll_days=7)
    assert np.isfinite(ladder[NUMERIC].to_numpy()).all()
