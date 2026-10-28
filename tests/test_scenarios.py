import math
import re
from dataclasses import replace
from datetime import date
from typing import NamedTuple

import numpy as np
import pytest

from options_analyzer import scenarios as sc
from options_analyzer import strategies as S

MARKET = S.Market(100, 0.05, 0.2, 0.25, 0.01)


def test_zero_shock_scenario_has_zero_pnl():
    strategy = S.build_template("Iron Condor", 100, 5)
    rows = sc.run_stress_tests(strategy, MARKET, [sc.StressScenario("none", 0.0, 1.0, 0.0)], multiplier=100)
    assert rows[0]["P&L"] == pytest.approx(0.0, abs=1e-9)


def test_crash_hurts_long_stock_and_helps_long_put():
    crash = [sc.StressScenario("crash", -0.3, 2.0, 0.0)]
    stock = S.Strategy("stock", (S.Leg("stock", 1),))
    put = S.build_template("Long Put", 100, 10)
    assert sc.run_stress_tests(stock, MARKET, crash)[0]["P&L"] == pytest.approx(-30.0)
    assert sc.run_stress_tests(put, MARKET, crash)[0]["P&L"] > 20


def test_vol_shock_damping_reduces_long_dated_vega_impact():
    market = S.Market(100, 0.05, 0.2, 2.0)
    straddle = S.build_template("Long Straddle", 100, 10)
    spike = [sc.StressScenario("spike", 0.0, 3.0, 0.0, one_month_vol_shock=True)]
    full = sc.run_stress_tests(straddle, market, spike, vol_tenor_decay=0.0)[0]["P&L"]
    damped = sc.run_stress_tests(straddle, market, spike, vol_tenor_decay=0.25)[0]["P&L"]
    assert 0 < damped < full


def test_elapsed_time_expires_short_options():
    strategy = S.build_template("Short Straddle", 100, 10)
    long_episode = [sc.StressScenario("flat", 0.0, 1.0, 0.0, days=365)]
    rows = sc.run_stress_tests(strategy, MARKET, long_episode, include_elapsed_time=True)
    # After expiry at an unchanged spot the short straddle keeps its premium (up to the forward drift).
    assert rows[0]["P&L"] > 0


def test_presets_are_well_formed():
    for scenario in (*sc.HISTORICAL_SCENARIOS, *sc.HYPOTHETICAL_SCENARIOS):
        assert -1 < scenario.spot_change < 1
        assert scenario.vol_multiplier > 0
        assert scenario.description


def test_pnl_grids():
    strategy = S.build_template("Long Call", 100, 10)
    spot_moves = np.linspace(-0.2, 0.2, 5)
    grid = sc.pnl_grid(strategy, MARKET, spot_moves, np.array([-0.5, 0.0, 0.5]), days=0)
    assert grid.shape == (3, 5)
    assert grid[1, 2] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(grid, axis=1) > 0)  # long call gains with spot
    assert np.all(np.diff(grid[:, 2]) > 0)  # and with implied vol
    decay = sc.pnl_grid(strategy, MARKET, spot_moves, days=np.array([0, 30, 60, 91]))
    assert decay.shape == (4, 5)
    assert decay[-1] == pytest.approx(S.pnl(strategy, MARKET, 100 * (1 + spot_moves), 91 / 365))
    assert decay[1, 2] < 0  # theta


def test_hypothetical_iv_shocks_are_literal_and_match_the_heatmap():
    # Review finding: the heatmap and the stress table disagreed for the same IV shock.
    market = S.Market(100, 0.05, 0.2, 1.0)
    straddle = S.build_template("Long Straddle", 100, 10)
    spike = next(x for x in sc.HYPOTHETICAL_SCENARIOS if x.name == "Vol spike (IV +50%)")
    row = sc.run_stress_tests(straddle, market, [spike], multiplier=100)[0]
    cell = sc.pnl_grid(straddle, market, np.array([0.0]), np.array([0.5]), 0, multiplier=100)[0, 0]
    assert row["P&L"] == pytest.approx(cell)
    assert not row["IV shock damped"]


def test_legs_expiring_during_an_episode_are_flagged():
    strategy = S.build_template("Long Call Calendar", 100, 10, 0.1)
    episode = [sc.StressScenario("long", -0.1, 1.0, 0.0, days=60)]
    rows = sc.run_stress_tests(strategy, MARKET, episode, include_elapsed_time=True)
    assert rows[0]["Legs expired"] == 1


def test_pnl_grid_accepts_a_single_day_and_rejects_ambiguous_shapes():
    market = S.Market(100, 0.05, 0.2, 30 / 365)
    calendar = S.build_template("Long Call Calendar", 100, 10, market.expiry)
    assert sc.pnl_grid(calendar, market, np.array([0.0, 0.1]), days=5).shape == (1, 2)
    with pytest.raises(ValueError, match="single number"):
        sc.pnl_grid(calendar, market, np.array([0.0]), np.array([0.0]), days=np.array([5]))


# --- Stress scenarios pinned to their sources ----------------------------------------------------------------------


class Source(NamedTuple):
    """A sourced scenario's numbers as its sources print them, typed here and not copied from the module."""

    start: str  # date of the first close (ISO)
    end: str  # date of the last close
    equity: tuple[str, str]  # equity index closes
    vol: tuple[str, str, str]  # volatility index and its closes
    rate: tuple[float, float]  # policy rate (%) at the start and the end; the midpoint of a target range
    rate_text: str  # how the description states the rate


#: S&P 500 closes: S&P Dow Jones Indices (and FRED SP500 for 2018 on). VIX and VXO closes: FRED VIXCLS and VXOCLS. Fed
#: funds target: FRED DFEDTAR, and from 16 Dec 2008 the range DFEDTARL to DFEDTARU. Fed scenario: Federal Reserve, 2026
#: Stress Test Scenarios (Feb 2026), table 2.A for the Q4 2025 jump-off (its VIX of 26.4 is the quarter's highest
#: close, 26.42 on 20 Nov 2025 in FRED VIXCLS) and table 4.A for the trough of the Dow Jones Total Stock Market Index
#: (Q3 2026), the VIX peak (Q2 2026) and the 3-month Treasury rate (Q2 2026 on).
SOURCES = {
    "Black Monday (1987)": Source(
        "1987-10-16", "1987-10-19", ("282.70", "224.84"), ("VXO", "36.37", "150.19"), (7.3125, 7.3125),
        "Fed funds target unchanged at 7.3125%",
    ),
    "Dot-com bust (2000-02)": Source(
        "2000-03-24", "2002-10-09", ("1,527.46", "776.76"), ("VIX", "23.31", "42.13"), (6.00, 1.75),
        "Fed funds target 6.00% → 1.75%",
    ),
    "Global Financial Crisis (2007-09)": Source(
        "2007-10-09", "2009-03-09", ("1,565.15", "676.53"), ("VIX", "16.12", "49.68"), (4.75, 0.125),
        "Fed funds target 4.75% → 0–0.25% (midpoint 0.125%)",
    ),
    "Volmageddon (2018)": Source(
        "2018-02-02", "2018-02-05", ("2,762.13", "2,648.94"), ("VIX", "17.31", "37.32"), (1.375, 1.375),
        "Fed funds target unchanged at 1.25–1.50%",
    ),
    "COVID crash (2020)": Source(
        "2020-02-19", "2020-03-23", ("3,386.15", "2,237.40"), ("VIX", "14.38", "61.59"), (1.625, 0.125),
        "Fed funds target 1.50–1.75% → 0–0.25%",
    ),
    "2022 rate shock": Source(
        "2022-01-03", "2022-10-12", ("4,796.56", "3,577.03"), ("VIX", "16.60", "33.57"), (0.125, 3.125),
        "Fed funds target 0–0.25% → 3.00–3.25%",
    ),
    "Tariff shock (Apr 2025)": Source(
        "2025-04-02", "2025-04-08", ("5,670.97", "4,982.77"), ("VIX", "21.51", "52.33"), (4.375, 4.375),
        "Fed funds target unchanged at 4.25–4.50%",
    ),
    "Fed severely adverse (2026)": Source(
        "2025-12-31", "2026-09-30", ("67,502", "28,490"), ("VIX", "26.4", "72.0"), (3.7, 0.1),
        "3-month Treasury rate falls 3.7% → 0.1%",
    ),
}  # fmt: skip

_PRESETS = {x.name: x for x in (*sc.HISTORICAL_SCENARIOS, *sc.HYPOTHETICAL_SCENARIOS)}


def _level(text: str) -> float:
    return float(text.replace(",", ""))


def _percent(change: float) -> str:
    """A change as the descriptions quote it, with the minus sign U+2212: ``−20.5%``."""
    return f"{change:+.1%}".replace("-", "−")


def test_every_sourced_scenario_has_its_sources():
    sourced = {x.name for x in _PRESETS.values() if x.vol_index}
    assert sourced == set(SOURCES)
    assert {x.name for x in sc.HISTORICAL_SCENARIOS} <= sourced
    assert all(_PRESETS[name].one_month_vol_shock for name in sourced)


@pytest.mark.parametrize("name", list(SOURCES))
def test_scenarios_are_pinned_to_their_sourced_numbers(name):
    # Validator finding 3: the Fed scenario's vol multiplier was 4.0 ("the VIX peaks at 72, about 4x") against the
    # Fed's 72.0 / 26.4 = 2.73, and three historical multipliers disagreed with the levels their descriptions quoted
    # (COVID 4.3 against 62 / 14 = 4.43). Every number is now checked against its source.
    scenario, source = _PRESETS[name], SOURCES[name]
    (e0, e1), (index, v0, v1), (r0, r1) = source.equity, source.vol, source.rate
    assert (scenario.vol_index, scenario.vol_start, scenario.vol_end) == (index, _level(v0), _level(v1))
    # The multiplier is the ratio of the closes itself, not a rounded copy of it.
    assert scenario.vol_multiplier == _level(v1) / _level(v0)
    # The equity change is quoted and applied to 0.1 %, so it is within 0.05 % of spot of the exact change.
    assert scenario.spot_change == round(_level(e1) / _level(e0) - 1, 3)
    # Rates in percent with at most four decimals: the change in decimal agrees to rounding error.
    assert scenario.rate_change == pytest.approx((r1 - r0) / 100, abs=1e-15)
    assert scenario.days == (date.fromisoformat(source.end) - date.fromisoformat(source.start)).days
    # The description states the same numbers.
    for text in (f"{e0} → {e1} ({_percent(scenario.spot_change)})", index, f"{v0} → {v1}", source.rate_text):
        assert text in scenario.description, (text, scenario.description)


def test_the_fed_scenario_states_its_jump_off_and_its_timing():
    # The Fed's VIX peaks in Q2 2026 and its equities bottom in Q3 2026; the scenario applies both at once and says so.
    fed = _PRESETS["Fed severely adverse (2026)"]
    assert fed.vol_multiplier == pytest.approx(2.727, abs=5e-4)
    for text in ("Q4 2025 jump-off", "Q3 2026 trough", "Q2 2026 peak", "applies both at once"):
        assert text in fed.description


def _put(spot: float, strike: float, tau: float, rate: float, vol: float, dividend: float) -> float:
    """Black-Scholes-Merton put, in closed form with math.erfc (independent of options_analyzer.black_scholes)."""
    d1 = (math.log(spot / strike) + (rate - dividend + vol**2 / 2) * tau) / (vol * math.sqrt(tau))
    d2 = d1 - vol * math.sqrt(tau)
    # N(-d) = erfc(d / sqrt 2) / 2
    return (
        strike * math.exp(-rate * tau) * math.erfc(d2 / math.sqrt(2)) / 2
        - spot * math.exp(-dividend * tau) * math.erfc(d1 / math.sqrt(2)) / 2
    )


@pytest.mark.parametrize("name", list(SOURCES))
def test_stress_pnl_reprices_with_the_sourced_shocks(name):
    # A one-month put takes the whole one-month vol shock (the damping weight min(1, (T / 1M)^-0.25) is 1), so its
    # stress P&L is the closed-form put at the sourced spot, vol and rate minus today's. Both sides are double-precision
    # Black-Scholes on P&L of 5 to 55: the worst measured difference is 1.4e-14 (4 ulps of Black Monday's 20.8);
    # 1e-12 leaves room for another platform's erfc.
    source, tau = SOURCES[name], 30 / 365
    (e0, e1), (_, v0, v1), (r0, r1) = source.equity, source.vol, source.rate
    market = S.Market(100, 0.04, 0.25, tau, 0.015)
    put = S.Strategy("put", (S.Leg("put", 1, 100),))
    pnl = sc.run_stress_tests(put, market, [_PRESETS[name]])[0]["P&L"]
    shocked = _put(
        100 * (1 + round(_level(e1) / _level(e0) - 1, 3)),
        100,
        tau,
        0.04 + (r1 - r0) / 100,
        0.25 * _level(v1) / _level(v0),
        0.015,
    )
    assert pnl == pytest.approx(shocked - _put(100, 100, tau, 0.04, 0.25, 0.015), abs=1e-12)


def test_ui_text_of_the_presets_uses_the_minus_sign():
    # One minus sign (U+2212) in UI text: the names and descriptions are shown in the Risk tab.
    for scenario in _PRESETS.values():
        for text in (scenario.name, scenario.description):
            assert not re.search(r"(?<![\w])-\d|->", text), text


_FED = _PRESETS["Fed severely adverse (2026)"]


def _sourced(**levels):
    return sc.StressScenario.sourced("x", 0.0, 0.0, 0, "", vol_index="VIX", **levels)


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: replace(_FED, vol_multiplier=4.0), "vol_multiplier 4.0 is not vol_end / vol_start = 2.72727"),
        (lambda: replace(_FED, vol_end=75.0), "vol_multiplier 2.7272727272727275 is not vol_end / vol_start = 2.8409"),
        (lambda: replace(_FED, one_month_vol_shock=False), "a shock calibrated on the VIX is a one-month vol shock"),
        (lambda: replace(_FED, vol_index=""), "vol_start and vol_end need the vol_index they are levels of"),
        (lambda: replace(_FED, vol_start=None), "Fed severely adverse (2026): vol_start must be a number, got None"),
        (lambda: sc.StressScenario("x", 0, 1, 0, vol_end=20.0), "x: vol_start and vol_end need the vol_index"),
        (lambda: _sourced(vol_start=0.0, vol_end=20), "x: vol_start and vol_end must be positive, got 0 and 20"),
        (lambda: _sourced(vol_start=14, vol_end=-1), "vol_start and vol_end must be positive, got 14 and -1"),
        (lambda: _sourced(vol_start="14", vol_end=20), "x: vol_start must be a number, got '14'"),
        (lambda: _sourced(vol_start=14, vol_end=math.inf), "x: vol_end must be a finite number, got inf"),
        (lambda: _sourced(vol_start=np.array([14.0, 15.0]), vol_end=20), "x: vol_start must be a single number"),
    ],
    ids=["drifted multiplier", "drifted level", "not one-month", "no index", "no start", "levels without index",
         "zero start", "negative end", "string start", "infinite end", "array start"],
)  # fmt: skip
def test_a_sourced_scenario_refuses_levels_that_disagree_with_its_shock(make, message):
    # The levels are the only copy of the shock: a multiplier that drifts from them cannot be built.
    with pytest.raises(ValueError, match=re.escape(message)):
        make()


def test_a_sourced_scenario_keeps_its_levels_and_reprices_like_a_plain_one():
    scenario = sc.StressScenario.sourced("x", -0.1, 0.0, 0, "", vol_index="VIX", vol_start=20.0, vol_end=30.0)
    assert (scenario.vol_multiplier, scenario.one_month_vol_shock) == (1.5, True)
    plain = sc.StressScenario("x", -0.1, 1.5, 0.0, one_month_vol_shock=True)
    straddle = S.build_template("Long Straddle", 100, 10)
    assert sc.run_stress_tests(straddle, MARKET, [scenario]) == sc.run_stress_tests(straddle, MARKET, [plain])


# --- RK-1: scenario cube and pnl_grid volatility modes -------------------------------------------------------------

SPOT_MOVES = np.linspace(-0.2, 0.2, 9)
VOL_SHOCKS = np.array([-0.3, -0.1, 0.0, 0.1, 0.5])
DAYS = np.array([0, 1, 7.5, 30])


def _multi_expiry_book() -> S.Strategy:
    return S.Strategy(
        "book",
        (
            S.Leg("call", -2, 100, 10 / 365, 0.22),
            S.Leg("put", 1, 95, 30 / 365),
            S.Leg("call", 1, 105, 180 / 365, 0.18),
            S.Leg("put", -1, 90, 2.0, 0.25),
            S.Leg("stock", 0.5),
        ),
    )


@pytest.mark.parametrize("vol_mode", sc.VOL_MODES)
def test_scenario_cube_slices_are_pnl_grids(vol_mode):
    # RK-1 acceptance (6): scenario_cube[0, zero-vol index, :] equals today's pnl_grid; every day's slice is the
    # volatility x spot grid of that day.
    strategy = _multi_expiry_book()
    cube = sc.scenario_cube(strategy, MARKET, SPOT_MOVES, VOL_SHOCKS, DAYS, vol_mode=vol_mode, multiplier=100)
    assert cube.shape == (len(DAYS), len(VOL_SHOCKS), len(SPOT_MOVES))
    zero = int(np.flatnonzero(VOL_SHOCKS == 0)[0])
    today = sc.pnl_grid(strategy, MARKET, SPOT_MOVES, days=0, multiplier=100)
    assert cube[0, zero] == pytest.approx(today[0], rel=1e-12, abs=1e-10)
    for d, day in enumerate(DAYS):
        grid = sc.pnl_grid(strategy, MARKET, SPOT_MOVES, VOL_SHOCKS, float(day), 100, vol_mode=vol_mode)
        assert cube[d] == pytest.approx(grid, rel=1e-12, abs=1e-10)


def test_one_month_mode_equals_relative_on_a_single_30_day_expiry():
    # RK-1 acceptance (6): the damping weight min(1, (tau / 1M)^-decay) is 1 for a 30-day leg, and stays 1 as the
    # leg ages (a shorter tenor's shock is never amplified).
    market = S.Market(100, 0.05, 0.2, 30 / 365, 0.01)
    strategy = S.build_template("Iron Condor", 100, 5)
    one_month = sc.scenario_cube(strategy, market, SPOT_MOVES, VOL_SHOCKS, DAYS, vol_mode="1m")
    relative = sc.scenario_cube(strategy, market, SPOT_MOVES, VOL_SHOCKS, DAYS, vol_mode="relative")
    assert one_month == pytest.approx(relative, rel=1e-12, abs=1e-12)


def _repriced(strategy: S.Strategy, market: S.Market, spot_change: float, vols: list[float]) -> float:
    """P&L with each option leg's IV set explicitly, independently of the vol_mode code path."""
    legs = tuple(
        replace(leg, volatility=vol) if leg.is_option else leg for leg, vol in zip(strategy.legs, vols, strict=True)
    )
    shocked = float(S.position_value(S.Strategy("shocked", legs), market, market.spot * (1 + spot_change)))
    return shocked - float(S.position_value(strategy, market))


@pytest.mark.parametrize(("vol_mode", "shock"), [("relative", 0.3), ("absolute", 0.04), ("1m", 0.5), ("1m", -0.25)])
def test_vol_modes_move_each_leg_as_documented(vol_mode, shock):
    strategy = _multi_expiry_book()
    vols = []
    for leg in strategy.legs:
        sigma = MARKET.volatility if leg.volatility is None else leg.volatility
        tau = MARKET.expiry if leg.expiry is None else leg.expiry
        if vol_mode == "relative":
            vols.append(sigma * (1 + shock))
        elif vol_mode == "absolute":
            vols.append(sigma + shock)
        else:
            vols.append(sigma * (1 + shock * min(1.0, (tau / (30 / 365)) ** -sc.DEFAULT_VOL_TENOR_DECAY)))
    cube = sc.scenario_cube(strategy, MARKET, [0.05], [shock], vol_mode=vol_mode)
    assert cube[0, 0, 0] == pytest.approx(_repriced(strategy, MARKET, 0.05, vols), rel=1e-12, abs=1e-12)


def test_one_month_shocks_move_long_dated_legs_less():
    market = S.Market(100, 0.05, 0.2, 2.0)
    straddle = S.build_template("Long Straddle", 100, 10)
    damped = sc.pnl_grid(straddle, market, [0.0], [1.0], vol_mode="1m")[0, 0]
    literal = sc.pnl_grid(straddle, market, [0.0], [1.0])[0, 0]
    assert 0 < damped < literal
    # The same damping as a one-month stress scenario.
    stress = sc.StressScenario("2x 1M vol", 0.0, 2.0, 0.0, one_month_vol_shock=True)
    assert damped == pytest.approx(sc.run_stress_tests(straddle, market, [stress])[0]["P&L"], rel=1e-12)


def test_pnl_grid_default_is_the_relative_mode():
    strategy = _multi_expiry_book()
    default = sc.pnl_grid(strategy, MARKET, SPOT_MOVES, VOL_SHOCKS, 3, 100)
    assert np.array_equal(default, sc.pnl_grid(strategy, MARKET, SPOT_MOVES, VOL_SHOCKS, 3, 100, vol_mode="relative"))


def test_stock_only_grids_fill_every_row():
    # A share does not depend on implied vol; the grid used to come back with a single row.
    shares = S.Strategy("shares", (S.Leg("stock", 10),))
    grid = sc.pnl_grid(shares, MARKET, SPOT_MOVES, VOL_SHOCKS)
    assert grid.shape == (len(VOL_SHOCKS), len(SPOT_MOVES))
    assert grid == pytest.approx(np.broadcast_to(10 * 100 * SPOT_MOVES, grid.shape), abs=1e-12)
    cube = sc.scenario_cube(shares, MARKET, SPOT_MOVES, VOL_SHOCKS, DAYS)
    assert cube.shape == (len(DAYS), len(VOL_SHOCKS), len(SPOT_MOVES))
    assert cube[0] == pytest.approx(grid, abs=1e-12)


def test_scenario_cube_rejects_unknown_modes_and_shapes():
    strategy = S.build_template("Long Call", 100, 10)
    with pytest.raises(ValueError, match="vol_mode"):
        sc.scenario_cube(strategy, MARKET, SPOT_MOVES, VOL_SHOCKS, vol_mode="log")
    with pytest.raises(ValueError, match="vol_mode"):
        sc.pnl_grid(strategy, MARKET, SPOT_MOVES, VOL_SHOCKS, vol_mode="1M")
    with pytest.raises(ValueError, match="one-dimensional"):
        sc.scenario_cube(strategy, MARKET, np.zeros((2, 2)), VOL_SHOCKS)


_DAYS_RULE = "days must be finite and non-negative"
_SPOT_RULE = "spot_changes must be finite and at least -1"


@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    [
        (([0.0], [0.0], [-10]), {}, f"{_DAYS_RULE} (calendar days elapsed), got -10"),
        (([0.0], [0.0], [0, np.nan]), {}, f"{_DAYS_RULE} (calendar days elapsed), got nan"),
        (([0.0], [0.0], np.inf), {}, f"{_DAYS_RULE} (calendar days elapsed), got inf"),
        (([0.0, np.inf], [0.0]), {}, f"{_SPOT_RULE} (a -100% move takes spot to zero), got inf"),
        (([-1.5], [0.0]), {}, f"{_SPOT_RULE} (a -100% move takes spot to zero), got -1.5"),
        (([0.0], [0.0, np.nan]), {}, "vol_shocks must be finite, got nan"),
        ((["0.1"], [0.0]), {}, "spot_changes must be numbers, got ['0.1']"),
        (([0.0], [0.0, None]), {}, "vol_shocks must be numbers, got [0.0, None]"),
        (([0.0], [0.0], [[1, 2], [3]]), {}, "days must be numbers"),
        (([0.0], [0.1]), {"vol_mode": "1m", "vol_tenor_decay": np.nan}, "vol_tenor_decay must be a finite number"),
    ],
)
def test_scenario_cube_rejects_invalid_axes_by_name(args, kwargs, message):
    # Negative days used to return the P&L of time running backwards, and NaN days failed inside black_scholes
    # without naming the argument.
    strategy = S.build_template("Long Straddle", 100, 5)
    with pytest.raises(ValueError, match=re.escape(message)):
        sc.scenario_cube(strategy, MARKET, *args, **kwargs)


def test_pnl_grid_rejects_invalid_arguments_by_name():
    strategy = S.build_template("Long Straddle", 100, 5)
    # vol_mode is checked whether or not there are vol_changes (it used to be ignored without them).
    with pytest.raises(ValueError, match="vol_mode"):
        sc.pnl_grid(strategy, MARKET, [0.0, 0.1], vol_mode="bogus")
    with pytest.raises(ValueError, match=_DAYS_RULE):
        sc.pnl_grid(strategy, MARKET, [0.0, 0.1], days=[0, -1])
    with pytest.raises(ValueError, match=_DAYS_RULE):
        sc.pnl_grid(strategy, MARKET, [0.0, 0.1], [0.0], days=np.nan)
    with pytest.raises(ValueError, match="vol_changes must be finite"):
        sc.pnl_grid(strategy, MARKET, [0.0, 0.1], [np.inf])
    with pytest.raises(ValueError, match=_SPOT_RULE):
        sc.pnl_grid(strategy, MARKET, [np.nan])


def test_a_minus_100_percent_move_is_valued_at_zero_spot():
    # The boundary itself is valid: at zero spot a straddle is worth its discounted strike (the put's intrinsic).
    strategy = S.build_template("Long Straddle", 100, 5)
    base = float(S.position_value(strategy, MARKET))
    expected = 100 * np.exp(-MARKET.rate * MARKET.expiry) - base
    assert sc.pnl_grid(strategy, MARKET, [-1.0])[0, 0] == pytest.approx(expected, rel=1e-14)
    assert sc.scenario_cube(strategy, MARKET, [-1.0], [0.0])[0, 0, 0] == pytest.approx(expected, rel=1e-14)


_CONDOR = S.build_template("Iron Condor", 100, 5)
#: Every scenario function that takes a dollar multiplier, called on an iron condor.
_WITH_MULTIPLIER = {
    "pnl_grid": lambda m: sc.pnl_grid(_CONDOR, MARKET, SPOT_MOVES, VOL_SHOCKS, multiplier=m),
    "pnl_grid by date": lambda m: sc.pnl_grid(_CONDOR, MARKET, SPOT_MOVES, days=DAYS, multiplier=m),
    "scenario_cube": lambda m: sc.scenario_cube(_CONDOR, MARKET, SPOT_MOVES, VOL_SHOCKS, DAYS, multiplier=m),
    "greek_ladder": lambda m: sc.greek_ladder(_CONDOR, MARKET, multiplier=m),
    "vega_buckets": lambda m: sc.vega_buckets(_CONDOR, MARKET, multiplier=m),
    "weighted_vega": lambda m: sc.weighted_vega(_CONDOR, MARKET, multiplier=m),
    "run_stress_tests": lambda m: sc.run_stress_tests(_CONDOR, MARKET, sc.HYPOTHETICAL_SCENARIOS, multiplier=m),
}


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (np.nan, "multiplier must be a finite number, got nan"),
        (np.inf, "multiplier must be a finite number, got inf"),
        (-np.inf, "multiplier must be a finite number, got -inf"),
        ("100", "multiplier must be a number, got '100'"),
        (None, "multiplier must be a number, got None"),
        (np.array([100.0, 200.0]), "multiplier must be a single number"),
    ],
    ids=["nan", "inf", "-inf", "string", "None", "array"],
)
@pytest.mark.parametrize("function", list(_WITH_MULTIPLIER))
def test_multiplier_must_be_a_finite_number(function, bad, message):
    # A NaN or infinite multiplier used to give an all-NaN table (and a RuntimeWarning in vega_buckets).
    with pytest.raises(ValueError, match=re.escape(message)):
        _WITH_MULTIPLIER[function](bad)


@pytest.mark.parametrize("function", list(_WITH_MULTIPLIER))
def test_multiplier_accepts_numpy_and_negative_numbers(function):
    # Validation keeps every finite number: numpy scalars, integers and a negative multiplier (the reversed position).
    def as_array(value):
        if isinstance(value, list):  # run_stress_tests
            return np.array([row["P&L"] for row in value])
        return np.asarray(value.to_numpy() if hasattr(value, "to_numpy") else value, dtype=float)

    call = _WITH_MULTIPLIER[function]
    one = as_array(call(1.0))
    assert np.array_equal(as_array(call(np.float64(250.0))), as_array(call(250)))
    reversed_ = as_array(call(-1))
    if function == "greek_ladder":  # the shock, spot_change and spot columns do not scale
        one, reversed_ = one[:, 3:], reversed_[:, 3:]
    assert np.array_equal(reversed_, -one)
