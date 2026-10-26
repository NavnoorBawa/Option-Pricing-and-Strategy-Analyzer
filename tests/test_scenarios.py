from dataclasses import replace

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


def test_scenario_data_matches_sources():
    by_name = {x.name: x for x in (*sc.HISTORICAL_SCENARIOS, *sc.HYPOTHETICAL_SCENARIOS)}
    assert by_name["Dot-com bust (2000-02)"].rate_change == pytest.approx(-0.0425)  # Fed funds 6.00% -> 1.75%
    fed = by_name["Fed severely adverse (2026)"]
    assert (fed.spot_change, fed.rate_change) == (pytest.approx(-0.58), pytest.approx(-0.036))
    assert all(x.one_month_vol_shock for x in sc.HISTORICAL_SCENARIOS)


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
