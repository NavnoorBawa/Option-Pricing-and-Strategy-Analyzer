"""Scenario analysis: mark-to-market P&L under spot, volatility, rate and time shocks.

Every number here comes from re-pricing each leg of the position with
Black-Scholes-Merton at the shocked inputs and the remaining time to expiry -
the same valuation used for the payoff diagram, so the tabs of the app are
mutually consistent. Scenario P&L is measured against the position's
*current* model value, so it isolates market risk (fill prices and fees do not
enter; they are part of the payoff analysis instead).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .strategies import Market, Strategy, position_value

__all__ = [
    "DEFAULT_VOL_TENOR_DECAY",
    "HISTORICAL_SCENARIOS",
    "HYPOTHETICAL_SCENARIOS",
    "StressScenario",
    "pnl_grid",
    "run_stress_tests",
]

DAYS_PER_YEAR = 365.0


@dataclass(frozen=True)
class StressScenario:
    """An instantaneous shock, optionally spread over ``days`` of elapsed time."""

    name: str
    spot_change: float  # relative, e.g. -0.20 = -20 %
    vol_multiplier: float  # implied vol is multiplied by this factor
    rate_change: float  # absolute, e.g. -0.015 = -150 bp
    days: int = 0  # calendar days the episode lasted
    description: str = ""


#: Historical analogues, built from S&P 500 closes (Yardeni bull/bear tables,
#: closing basis), the VIX / VXO and the Fed funds target (FRED) over each
#: episode. The implied-vol multiplier is the ratio of the VIX at the end of
#: the window to its level at the start, i.e. a shock to one-month implied
#: vol; see :func:`run_stress_tests` for how it is damped for longer tenors.
#: Rate changes are policy-rate moves over the window.
HISTORICAL_SCENARIOS: tuple[StressScenario, ...] = (
    StressScenario(
        "Black Monday (1987)",
        -0.205,
        4.1,
        -0.005,
        1,
        "19 Oct 1987: S&P 500 -20.5% in one session; VXO 36 -> 150.",
    ),
    StressScenario(
        "Dot-com bust (2000-02)",
        -0.491,
        1.8,
        -0.0525,
        929,
        "24 Mar 2000 - 9 Oct 2002: S&P 500 -49.1%; VIX ~23 -> ~42; Fed funds 6.50% -> 1.25%.",
    ),
    StressScenario(
        "Global Financial Crisis (2007-09)",
        -0.568,
        3.1,
        -0.046,
        517,
        "9 Oct 2007 - 9 Mar 2009: S&P 500 -56.8%; VIX 16 -> 50 (peak close 80.9 in Nov 2008); "
        "Fed funds 4.75% -> 0-0.25%.",
    ),
    StressScenario(
        "Volmageddon (2018)",
        -0.041,
        2.16,
        0.0,
        1,
        "5 Feb 2018: S&P 500 -4.1%; VIX 17.3 -> 37.3, its largest one-day jump at the time.",
    ),
    StressScenario(
        "COVID crash (2020)",
        -0.339,
        4.3,
        -0.015,
        33,
        "19 Feb - 23 Mar 2020: S&P 500 -33.9% in 23 sessions; VIX 14 -> 62 (record close 82.7 on 16 Mar); "
        "Fed cut 150 bp.",
    ),
    StressScenario(
        "2022 rate shock",
        -0.254,
        2.0,
        0.03,
        282,
        "3 Jan - 12 Oct 2022: S&P 500 -25.4%; VIX 17 -> 34; Fed funds +300 bp.",
    ),
    StressScenario(
        "Tariff shock (Apr 2025)",
        -0.121,
        2.4,
        0.0,
        6,
        "2 - 8 Apr 2025: S&P 500 -12.1% in four sessions; VIX 22 -> 52.",
    ),
)

HYPOTHETICAL_SCENARIOS: tuple[StressScenario, ...] = (
    StressScenario("Rally +15%, IV -30%", 0.15, 0.7, 0.0, 0, "Relief rally with implied volatility collapsing."),
    StressScenario(
        "Vol crush (IV -40%)",
        0.0,
        0.6,
        0.0,
        0,
        "Implied volatility falls 40% with spot unchanged (e.g. after earnings).",
    ),
    StressScenario("Vol spike (IV +50%)", 0.0, 1.5, 0.0, 0, "Implied volatility rises 50% with spot unchanged."),
    StressScenario("Rates +200 bp", 0.0, 1.0, 0.02, 0, "Parallel 2-point rise in the risk-free rate."),
    StressScenario(
        "Fed severely adverse (2026)",
        -0.54,
        4.0,
        -0.039,
        274,
        "Federal Reserve 2026 supervisory stress test: equities about -54% over three quarters, "
        "VIX peaking near 72, short rates falling to about 0.1%.",
    ),
    StressScenario("Gap down -10%", -0.10, 1.4, 0.0, 0, "Overnight gap lower with a volatility pickup."),
    StressScenario("Gap up +10%", 0.10, 0.9, 0.0, 0, "Overnight gap higher with a mild volatility drop."),
)


#: Exponent for damping implied-vol shocks with tenor. Crisis data put the
#: 1M-vs-3M decay of additive VIX shocks at about 0.20 (2008) to 0.25 (2020).
DEFAULT_VOL_TENOR_DECAY = 0.25


def run_stress_tests(
    strategy: Strategy,
    market: Market,
    scenarios: tuple[StressScenario, ...] | list[StressScenario],
    include_elapsed_time: bool = False,
    multiplier: float = 1.0,
    vol_tenor_decay: float = DEFAULT_VOL_TENOR_DECAY,
) -> list[dict]:
    """P&L of the position under each scenario (in $ after ``multiplier``).

    The implied-vol multiplier is the shock to one-month vol; with
    ``vol_tenor_decay > 0`` it is damped for longer-dated legs (a 1-year
    option's IV moves roughly half as much as one-month IV in a crisis).
    With ``include_elapsed_time`` the scenario's duration also passes, so
    options decay (and expire, if the episode outlasts them).
    """
    base = float(position_value(strategy, market))
    rows = []
    for sc in scenarios:
        t = sc.days / DAYS_PER_YEAR if include_elapsed_time else 0.0
        value = position_value(
            strategy,
            market,
            spot=market.spot * (1.0 + sc.spot_change),
            t=t,
            vol_multiplier=sc.vol_multiplier,
            rate_shift=sc.rate_change,
            vol_tenor_decay=vol_tenor_decay,
        )
        change = multiplier * (float(value) - base)
        rows.append(
            {
                "Scenario": sc.name,
                "Spot change": sc.spot_change,
                "IV multiplier": sc.vol_multiplier,
                "Rate change (bp)": sc.rate_change * 1e4,
                "Days elapsed": sc.days if include_elapsed_time else 0,
                "P&L": change,
                "Description": sc.description,
            }
        )
    return rows


def pnl_grid(
    strategy: Strategy,
    market: Market,
    spot_changes: np.ndarray,
    vol_changes: np.ndarray | None = None,
    days: np.ndarray | float = 0.0,
    multiplier: float = 1.0,
) -> np.ndarray:
    """P&L on a grid.

    * with ``vol_changes`` (relative IV changes, e.g. 0.25 = +25 %) the result
      has shape ``(len(vol_changes), len(spot_changes))`` at a fixed ``days``;
    * otherwise ``days`` must be an array and the result has shape
      ``(len(days), len(spot_changes))`` at unchanged implied volatility.
    """
    spots = market.spot * (1.0 + np.asarray(spot_changes, dtype=float))[None, :]
    base = float(position_value(strategy, market))
    if vol_changes is not None:
        vol_mult = (1.0 + np.asarray(vol_changes, dtype=float))[:, None]
        t = float(days) / DAYS_PER_YEAR
        values = position_value(strategy, market, spots, t, vol_multiplier=vol_mult)
    else:
        t = np.asarray(days, dtype=float)[:, None] / DAYS_PER_YEAR
        values = position_value(strategy, market, spots, t)
    return multiplier * (np.asarray(values, dtype=float) - base)
