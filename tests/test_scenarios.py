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
