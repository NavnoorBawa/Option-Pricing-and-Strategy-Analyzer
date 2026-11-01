"""The Risk tab in the desk layout (UI spec 3.3, BK-1, UI-10): the position line, the six pinned tiles, the report
selector and its six reports, on the synthetic index at the pinned valuation time and on the textbook market.

Under the index the tab analyses the Strategy tab's default, the spec's iron condor: +1 5,300P / −1 5,400P / −1 6,150C
/ +1 6,250C on 18-Dec-26 AM, 1 × 100, each leg at its own vol. Numbers are checked against Black-Scholes-Merton
written here from the textbook formulas (scipy's normal distribution, calls and puts), each leg at the vol the
published surface gives at its strike and expiry (``surface_state.publish(context).vol(K, expiry)``) and at the carry
of that expiry's parity forward and discount factor (``market.carry``): not against the package's pricer.

The spec's example numbers (3.3, its measured goldens) are of this position at this valuation time. Those this build
reproduces are asserted as the spec writes them: vega −$306 with 1M −$225, Θ +$41 with BE 0.77% vs 0.81%, and the
worst grid cell at +10% · −10 vp. The others differ, and the spec does not record how they were computed (its $Δ is
of 3.37 index units of delta, against the 3.25 of the legs at their own surface vols here): the tab's $Δ −$18,871
against the spec's −$19,525, Γ P&L 1% −$69 against −$68, the worst grid −$7,683 against −$7,663, the max loss
−$7,765 against −$7,760, the FINRA requirement −$3,141 against −$3,160. The goldens below are this build's, each
recomputed here.
Under the textbook preset the Strategy tab's default is still version 2's 100 / 110 call spread at the flat sidebar
vol, and its goldens are unchanged.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from _apptest import ASOF, make_app
from scipy.optimize import brentq
from scipy.stats import norm
from streamlit.testing.v1 import AppTest

from options_analyzer import scenarios
from options_analyzer import strategies as strat
from options_analyzer.ui import legs as lg
from options_analyzer.ui import market as mk
from options_analyzer.ui import surface_state, tab_risk, whatif
from options_analyzer.ui.format import signed
from options_analyzer.ui.tab_risk import GridAxes
from options_analyzer.ui.tab_strategy import Position
from options_analyzer.vol_surface.published import Expiry

#: The default Strategy position under the index preset at ASOF: 1 × 100 of the spec's iron condor (quantity, kind,
#: strike) on the AM monthly nearest 45 days.
EXPIRY = Expiry(date(2026, 12, 18), "AM")
LEGS = ((1.0, "put", 5300.0), (-1.0, "put", 5400.0), (-1.0, "call", 6150.0), (1.0, "call", 6250.0))
UNITS = 100.0
#: Friday 23-Oct 16:00 to Friday 30-Oct 16:00 (T+5) and to Monday 26-Oct 16:00 (the next session), in years.
WEEK, WEEKEND = 7 / 365, 3 / 365
#: One month, the reference tenor of the 1M-scaled vol shocks, and their damping exponent.
ONE_MONTH, DECAY = 30 / 365, 0.25
#: The no-break space that keeps a leg's quantity with its strike in the position line.
NBSP = "\u00a0"


def _ok(at: AppTest) -> AppTest:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    return at


def _risk(**query) -> AppTest:
    """The Risk tab on the synthetic index at ASOF (or on the textbook market with ``preset="Textbook"``)."""
    query.setdefault("preset", None)
    if query["preset"] is None:
        query.setdefault("asof", ASOF)
    at = make_app(tab="Risk", **query)
    at.run(timeout=240)
    return _ok(at)


def _tiles(at: AppTest) -> dict:
    labels = ("$Δ", "Γ P&L 1%", "Vega ($/vp)", "Θ ($/day)", "Worst grid", "Worst stress")
    return {m.label: m for m in at.metric if m.label in labels}


def _chart(at: AppTest, key: str) -> dict:
    return next(json.loads(c.proto.spec) for c in at.get("plotly_chart") if c.proto.id.endswith(key))


def _frame(at: AppTest, column: str):
    return next(d.value for d in at.dataframe if column in d.value.columns)


def _line(at: AppTest) -> str:
    return next(m.value for m in at.markdown if "max loss" in m.value)


def _shown(text: str) -> str:
    """A tile's dollar amount as Python formats it: ``−$18,871`` is ``-18,871``, ``+$41`` is ``41``."""
    return text.replace("$", "").lstrip("+").replace("−", "-")


# --- an independent Black-Scholes-Merton ------------------------------------------------------------------------------


def _d1(s, k, t, r, q, v):
    return (math.log(s / k) + (r - q + 0.5 * v * v) * t) / (v * math.sqrt(t))


def _price(kind, s, k, t, r, q, v):
    """A European call or put; its intrinsic value once expired (``t`` ≤ 0)."""
    if t <= 0:
        return max(s - k, 0.0) if kind == "call" else max(k - s, 0.0)
    d1 = _d1(s, k, t, r, q, v)
    d2 = d1 - v * math.sqrt(t)
    if kind == "call":
        return s * math.exp(-q * t) * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)
    return k * math.exp(-r * t) * norm.cdf(-d2) - s * math.exp(-q * t) * norm.cdf(-d1)


def _greeks(kind, s, k, t, r, q, v):
    """Delta, gamma, vega (per 1.00) and theta (∂V/∂t per year) of a call or a put, from the closed forms."""
    d1 = _d1(s, k, t, r, q, v)
    d2 = d1 - v * math.sqrt(t)
    eq, er = math.exp(-q * t), math.exp(-r * t)
    gamma = eq * norm.pdf(d1) / (s * v * math.sqrt(t))
    vega = s * eq * norm.pdf(d1) * math.sqrt(t)
    decay = -s * eq * norm.pdf(d1) * v / (2 * math.sqrt(t))
    if kind == "call":
        return eq * norm.cdf(d1), gamma, vega, decay - r * k * er * norm.cdf(d2) + q * s * eq * norm.cdf(d1)
    return -eq * norm.cdf(-d1), gamma, vega, decay + r * k * er * norm.cdf(-d2) - q * s * eq * norm.cdf(-d1)


def _damp(tau: float) -> float:
    """The share of a one-month vol shock that a leg with ``tau`` years left takes: min(1, (τ / 1M)^−0.25)."""
    return min(1.0, (max(tau, 1e-8) / ONE_MONTH) ** -DECAY)


class Index:
    """The market at ASOF, read from the published surface: spot, the 18-Dec-26 AM expiry's years to settlement, the
    carry of its parity forward and discount factor, each leg's surface vol at its strike and the ATMF vol."""

    def __init__(self) -> None:
        context = mk.index_market(datetime.fromisoformat(ASOF))
        published = surface_state.publish(context)
        self.context, self.published = context, published
        self.s = context.spot
        self.t = published.maturity(EXPIRY)
        self.forward = published.forward(EXPIRY)
        self.r, self.q = mk.carry(self.forward, published.discount_factor(EXPIRY), self.s, self.t)
        self.vols = tuple(published.vol(k, EXPIRY).vol for _, _, k in LEGS)
        self.atm = published.vol(self.forward, EXPIRY).vol

    def value(self, s=None, t=None, r=None, vols=None, legs=LEGS) -> float:
        """The position's value ($, 1 × 100): each leg at its own vol."""
        s, t, r = self.s if s is None else s, self.t if t is None else t, self.r if r is None else r
        vols = self.vols if vols is None else vols
        return UNITS * sum(n * _price(kind, s, k, t, r, self.q, v) for (n, kind, k), v in zip(legs, vols, strict=True))

    def greeks(self, s=None, t=None, vols=None) -> np.ndarray:
        """Delta, gamma, vega and theta of the position ($ units, 1 × 100)."""
        s, t, vols = self.s if s is None else s, self.t if t is None else t, self.vols if vols is None else vols
        totals = np.zeros(4)
        for (n, kind, k), v in zip(LEGS, vols, strict=True):
            totals += n * np.array(_greeks(kind, s, k, t, self.r, self.q, v))
        return UNITS * totals

    def credit(self) -> float:
        """The structure's credit per share at the chain's mids, the legs' entries."""
        return -sum(n * self.published.quote(EXPIRY, k, kind).mid for n, kind, k in LEGS)


@pytest.fixture(scope="module")
def index() -> Index:
    return Index()


# --- the report selector and what each report computes ----------------------------------------------------------------


def test_the_selector_has_six_reports_and_ladder_is_the_default():
    at = _risk()
    pills = at.get("button_group")
    report = next(p for p in pills if p.proto.id.endswith("-report") or "report" in p.proto.id)
    assert [o.content for o in report.proto.options] == list(tab_risk.REPORTS)
    assert _frame(at, "Move") is not None  # the Ladder's table


def test_only_the_selected_report_computes(monkeypatch):
    # ?report=VaR runs the two simulations (with implied vol moving and frozen, Milestone 1's spot-only VaR beside
    # the spot-plus-vol one) and no other report's engine: the ladder never, the scenario cube once (the WORST GRID
    # tile's grid), no backtest.
    from options_analyzer import backtest, risk
    from options_analyzer.ui import cache

    calls = {"greek_ladder": 0, "scenario_cube": 0, "value_at_risk": 0, "rolling_backtest": 0}

    def counted(module, name):
        real = getattr(module, name)

        def wrapper(*args, **kwargs):
            calls[name] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(module, name, wrapper)

    for module, name in ((scenarios, "greek_ladder"), (scenarios, "scenario_cube"), (risk, "value_at_risk")):
        counted(module, name)
    counted(backtest, "rolling_backtest")
    cache.value_at_risk.clear()
    _risk(report="VaR")
    assert calls == {"greek_ladder": 0, "scenario_cube": 1, "value_at_risk": 2, "rolling_backtest": 0}


# --- the tiles --------------------------------------------------------------------------------------------------------


def test_the_default_tiles_are_the_closed_form_values(index):
    # $Δ = Δ·S, Γ P&L 1% = ½Γ(0.01S)², vega per vol point, Θ = ∂V/∂t ÷ 365, each leg at its own surface vol (12.06%
    # for the 6,250 call to 20.56% for the 5,300 put; the ATMF vol, 15.52%, is no leg's). Tolerance: the tile's own
    # rounding (whole dollars). Vega and Θ are the spec's; $Δ and Γ P&L 1% differ from it (module docstring).
    at = _risk()
    tiles = _tiles(at)
    delta, gamma, vega, theta = index.greeks()
    expected = {
        "$Δ": delta * index.s,
        "Γ P&L 1%": 0.5 * gamma * (0.01 * index.s) ** 2,
        "Vega ($/vp)": vega / 100,
        "Θ ($/day)": theta / 365,
    }
    for label, value in expected.items():
        assert tiles[label].value == tab_risk.dollars(value), label
    assert [tiles[k].value for k in ("$Δ", "Γ P&L 1%", "Vega ($/vp)", "Θ ($/day)")] == [
        "−$18,871",
        "−$69",
        "−$306",
        "+$41",
    ]
    assert [round(v, 4) for v in index.vols] == [0.2056, 0.1963, 0.1270, 0.1206] and round(index.atm, 4) == 0.1552
    # At the ATMF vol, the one vol the footer named before, the four legs read another position: $Δ −$12,363.
    flat = index.greeks(vols=(index.atm,) * 4)
    assert tab_risk.dollars(flat[0] * index.s) == "−$12,363" != tiles["$Δ"].value
    # The vega tile's 1M slot: each leg's vega weighted √(1M/τ), the spec's −$225.
    one_month = tab_risk.dollars(vega / 100 * math.sqrt(ONE_MONTH / index.t))
    assert tiles["Vega ($/vp)"].delta == f"1M {one_month}".replace("$", "\\$") == "1M −\\$225"  # Markdown
    # No statistic of a simulation or a history in a pinned tile (spec section 5 item 19).
    assert not any("VaR" in m.label or "ES " in m.label for m in tiles.values())


def test_the_delta_tile_counts_es_contracts_at_the_december_forward(index):
    at = _risk()
    delta = index.greeks()[0]
    forward = float(index.context.expiry_row(date(2026, 12, 18))["forward"])
    contracts = delta / (50.0 * forward / index.s)
    # The spec's −0.07 ES is of its −3.37 index units; the legs at their own vols hold −3.25.
    assert _tiles(at)["$Δ"].delta == f"{contracts:+.2f} ES".replace("-", "−") == "−0.06 ES"
    assert f"Δ {signed(delta, 2)} index units" in _tiles(at)["$Δ"].help  # −3.25: 100 × −0.0325
    assert tab_risk.es_contract(datetime(2026, 10, 23, 16)) == date(2026, 12, 18)


def test_the_theta_tile_gives_the_breakeven_move_against_the_implied_move(index):
    # BE = 1% × √(|Θ| ÷ |Γ P&L 1%|), defined since Γ < 0 < Θ (the short condor earns theta); implied = σ ATMF ÷ √365
    # at the front expiry, both per calendar day. The ATMF vol is the surface's at K = F. Both are the spec's.
    at = _risk()
    _, gamma, _, theta = index.greeks()
    be = 0.01 * math.sqrt(abs(theta / 365) / abs(0.5 * gamma * (0.01 * index.s) ** 2))
    assert _tiles(at)["Θ ($/day)"].delta == f"BE {be:.2%} vs {index.atm / math.sqrt(365):.2%}" == "BE 0.77% vs 0.81%"
    # Θ to the next session (Mon 26-Oct 16:00, three calendar days) by full revaluation, in the help: +$124.69
    # (the spec's +$126 differs).
    decay = index.value(t=index.t - WEEKEND) - index.value()
    assert f"{abs(decay):,.2f}" == "124.69" and f"{abs(decay):,.2f}" in _tiles(at)["Θ ($/day)"].help


def _desk_grid(index: Index, mode: str = "vp", vols=None, t=None) -> dict:
    """The desk grid (spot move %, vol shock) → P&L, each leg's vol shocked by the mode: vol points added, a share of
    its own vol, or a share of one-month vol damped by its own time left."""
    vols = index.vols if vols is None else vols
    t = index.t if t is None else t
    today = index.value()
    grid = {}
    for dv in tab_risk.DESK_VOL[mode]:
        if mode == "vp":
            shocked = [v + dv / 100 for v in vols]
        elif mode == "relative":
            shocked = [v * (1 + dv / 100) for v in vols]
        else:
            shocked = [v * (1 + dv / 100 * _damp(t)) for v in vols]
        for x in tab_risk.DESK_SPOT:
            grid[(x, dv)] = index.value(s=index.s * (1 + x / 100), t=t, vols=shocked) - today
    return grid


def _stresses(index: Index, s=None, t=None, vols=None) -> dict:
    """Every named stress, instantaneous, from the state (s, t, vols): spot moved, each leg's vol multiplied (a
    one-month shock damped by the leg's own time left), the rate shifted; against today's value at that state."""
    s, t, vols = index.s if s is None else s, index.t if t is None else t, index.vols if vols is None else vols
    base = index.value(s=s, t=t, vols=vols)
    out = {}
    for sc in tab_risk.ALL_STRESSES:
        damp = _damp(t) if sc.one_month_vol_shock else 1.0
        shocked = [v * (1 + (sc.vol_multiplier - 1) * damp) for v in vols]
        out[sc.name] = index.value(s * (1 + sc.spot_change), t, index.r + sc.rate_change, shocked) - base
    return out


def test_the_worst_grid_and_worst_stress_tiles(index):
    at = _risk()
    tiles = _tiles(at)
    grid = _desk_grid(index)
    worst = min(grid.values())
    # The spec's cell (+10%, −10 vp), now −$7,683: spot at 6,380, past both calls, every leg's vol 10 points down.
    assert tiles["Worst grid"].value == tab_risk.dollars(worst) == "−$7,683"
    (x, dv) = min(grid, key=grid.get)
    assert (x, dv) == (10.0, -10.0)
    assert tiles["Worst grid"].delta == f"{tab_risk.shock_label(x / 100)} · {tab_risk.vol_label(dv, 'vp')}"
    assert tiles["Worst grid"].delta == "+10% · −10 vp"
    assert sum(round(v) == round(worst) for v in grid.values()) == 1 and "one of" not in tiles["Worst grid"].help
    # The worst stress: each leg's vol × the VIX ratio, damped by its 55.7 days to (55.7 / 30)^−0.25 = 0.857 of it.
    # The financial crisis (−$7,749.58) and the dot-com bust (−$7,749.00) are 58 cents apart, both near the max loss.
    stresses = _stresses(index)
    name, loss = min(stresses.items(), key=lambda item: item[1])
    assert name == "Global Financial Crisis (2007–09)" and stresses["Dot-com bust (2000–02)"] - loss < 1.0
    assert tiles["Worst stress"].value == tab_risk.dollars(loss) == "−$7,750"
    assert tiles["Worst stress"].delta == name.split(" (")[0]
    assert f"{name}, {tab_risk.exact(loss)}".replace("$", "\\$") in tiles["Worst stress"].help


def test_ties_in_the_worst_grid_go_to_the_smallest_spot_move(index):
    # The 5,815 / 5,915 call spread on 20-Nov-26 AM (the index's default before the iron condor, by link), each leg
    # at its own surface vol there (15.44% and 14.19%): far enough down it is worth nothing to the dollar, so four
    # cells equal its loss (−5%, −7.5% and −10% at −10 vp, −10% at −5 vp). The tile names the one with the smallest
    # spot move, not the first in row order.
    key = Expiry(date(2026, 11, 20), "AM")
    at = _risk(strategy="Bull Call Spread", strategy_expiry="2026-11-20 AM")
    t = index.published.maturity(key)
    r, q = mk.carry(index.published.forward(key), index.published.discount_factor(key), index.s, t)
    legs = ((1.0, "call", 5815.0), (-1.0, "call", 5915.0))
    vols = [index.published.vol(k, key).vol for _, _, k in legs]

    def value(s, dv):
        return UNITS * sum(n * _price(kind, s, k, t, r, q, v + dv) for (n, kind, k), v in zip(legs, vols, strict=True))

    today = value(index.s, 0.0)
    grid = {(x, dv): value(index.s * (1 + x / 100), dv / 100) - today for x in tab_risk.DESK_SPOT
            for dv in tab_risk.DESK_VOL["vp"]}  # fmt: skip
    worst = min(grid.values())
    ties = [cell for cell, value in grid.items() if round(value) == round(worst)]
    x, dv = min(ties, key=lambda cell: (abs(cell[0]), abs(cell[1])))
    assert len(ties) == 4 and (x, dv) == (-5.0, -10.0)
    tile = _tiles(at)["Worst grid"]
    assert tile.value == tab_risk.dollars(worst)
    assert tile.delta == f"{tab_risk.shock_label(x / 100)} · {tab_risk.vol_label(dv, 'vp')}" == "−5% · −10 vp"
    assert "one of 4 cells equal to the dollar" in tile.help
    assert _line(at).startswith(
        f"**Bull Call Spread** — +1{NBSP}5,815C / −1{NBSP}5,915C · 1 × 100 · 20‑Nov‑26 AM (27d 17h) · "
    )


def test_the_position_line_names_legs_expiry_size_and_max_loss(index):
    at = _risk()
    # The condor's maximum loss at expiry is a wing's width less the credit taken at the chain's mids (the legs'
    # entries): 100 × (100 − 22.35). The spec's −$7,760 would be a credit of 22.40.
    credit = index.credit()
    loss = UNITS * (max(5400.0 - 5300.0, 6250.0 - 6150.0) - credit)
    assert round(credit, 2) == 22.35
    # One expiry for the four legs: named once, with its settlement and the time left (55.73 days, floored). Each
    # leg's quantity keeps to its strike (a no-break space), so a phone breaks the line between legs.
    assert _line(at) == (
        f"**Iron Condor** — +1{NBSP}5,300P / −1{NBSP}5,400P / −1{NBSP}6,150C / +1{NBSP}6,250C · 1 × 100 · "
        f"18‑Dec‑26 AM ({math.floor(index.t * 365)}d) · max loss at expiry −\\${loss:,.0f}"
    )
    assert "−\\$7,765" in _line(at)
    textbook = _line(_risk(preset="Textbook"))
    assert "· 365d ·" in textbook  # the textbook market counts days, not dates


def test_a_position_on_two_expiries_names_each_leg_and_expiry_with_its_settlement(index):
    # A call calendar under the index: the front leg on the default expiry, the back leg on the AM monthly nearest
    # twice its maturity (19-Feb-27, 118.7 days), both at the listed strike nearest the December forward. The line
    # read "18‑Dec‑26 (55d), 19‑Feb‑27 (118d)" before, the dates read back from years without their settlements.
    k = 5 * round(index.forward / 5)
    line = _line(_risk(strategy="Long Call Calendar"))
    back = index.published.maturity(Expiry(date(2027, 2, 19), "AM"))
    assert line.startswith(
        f"**Long Call Calendar** — −1{NBSP}{k:,.0f}C 18‑Dec‑26 AM / +1{NBSP}{k:,.0f}C 19‑Feb‑27 AM · 1 × 100 · "
        f"18‑Dec‑26 AM ({math.floor(index.t * 365)}d), 19‑Feb‑27 AM ({math.floor(back * 365)}d) · max loss at expiry "
    )
    assert k == 5830


@pytest.mark.parametrize("strategy", ["Long Call Calendar", "Diagonal (Poor Man's Covered Call)"])
def test_the_position_line_reads_the_strategy_tabs_maximum_loss(strategy):
    # With the index's dividend yield (1.27%) the long back call gains less than $1 per $1 rise, so the loss is
    # unbounded only far above the strikes (the calendar's passes its debit near 3.4 × spot). The Strategy tile reads
    # the worst P&L within the RN 99.9% range, the debit (−$8,500; −$51,065 for the diagonal); the line read
    # "Unlimited" for the same position. Both now read the same loss over the same range, and it is the debit: the
    # calls' value at the range's low end, 3,868, is below half a dollar a contract.
    tiles = {
        m.label: m
        for m in _ok(make_app(preset=None, asof=ASOF, strategy=strategy, tab="Strategy").run(timeout=240)).metric
    }
    tile, premium = tiles["Max loss"], tiles["Net premium"]
    assert tile.proto.delta == "RN 99.9% range" and premium.proto.delta == "debit"
    assert tile.value == "−" + premium.value
    line = _line(_risk(strategy=strategy))
    assert line.endswith(f"max loss at expiry {tile.value.replace('$', chr(92) + '$')} in the RN 99.9% range")


def test_tiles_follow_the_what_if_state(index):
    # ?ws=-2&wv=3&wd=5: spot −2%, +3 vol points on every leg's own vol, Friday 23-Oct 16:00 to Friday 30-Oct 16:00.
    at = _risk(ws=-2, wv=3, wd=5)
    s, t, vols = index.s * 0.98, index.t - WEEK, [v + 0.03 for v in index.vols]
    delta = index.greeks(s, t, vols)[0]
    assert _tiles(at)["$Δ"].value == tab_risk.dollars(delta * s)
    pnl = index.value(s, t, vols=vols) - index.value()
    assert f"what-if P&L {tab_risk.dollars(pnl)} vs value now".replace("$", "\\$") in _line(at)
    assert any("T+5 · Fri 30‑Oct‑26" in s.label for s in at.slider if s.key == "wd")


@pytest.mark.parametrize("mode", ["relative", "1M-scaled"])
def test_the_what_if_vol_shift_reaches_each_leg_by_its_mode(index, mode):
    # +10% in each mode: relative scales each leg's own vol by 1.10 (the 5,300 put 20.56% to 22.62%, the 6,250 call
    # 12.06% to 13.26%); 1M-scaled by 1 + 10% × (55.7 / 30)^−0.25, 8.57% of each. The tiles and the what-if P&L
    # are those of the legs so moved, against their values today.
    at = _risk(wv=10, wvmode=mode)
    weight = 1.0 if mode == "relative" else _damp(index.t)
    vols = [v * (1 + 0.10 * weight) for v in index.vols]
    delta, _, vega, _ = index.greeks(vols=vols)
    tiles = _tiles(at)
    assert tiles["$Δ"].value == tab_risk.dollars(delta * index.s)
    assert tiles["Vega ($/vp)"].value == tab_risk.dollars(vega / 100)
    pnl = index.value(vols=vols) - index.value()
    assert f"what-if P&L {tab_risk.dollars(pnl)} vs value now".replace("$", "\\$") in _line(at)
    # Not the other mode's shift: relative moves each leg by all of the 10%, 1M-scaled by 8.57% of it.
    other = 1.0 if mode != "relative" else _damp(index.t)
    pnl = index.value(vols=[v * (1 + 0.10 * other) for v in index.vols]) - index.value()
    assert f"what-if P&L {tab_risk.dollars(pnl)} vs value now".replace("$", "\\$") not in _line(at)


def test_one_scenario_reads_against_the_value_now_and_against_the_entry(index):
    # ?ws=-2&wv=3&wd=5&wr=25 on the default iron condor: the Strategy chart marked "Scenario 5,684: −$400.61" (against
    # the entry at mid) and the Risk line read "what-if P&L −$388" (against today's value), both under the same strip
    # and neither saying against what. Both now name their basis, and Risk shows both; they differ by the T+0 P&L, the
    # entry at the chain's mids against the legs' value now (written here from the closed form).
    query = {"ws": -2, "wv": 3, "wd": 5, "wr": 25}
    s, t, r, vols = index.s * 0.98, index.t - WEEK, index.r + 0.0025, [v + 0.03 for v in index.vols]
    later = index.value(s, t, r, vols)
    now, entry = later - index.value(), later + UNITS * index.credit()
    assert round(now - entry, 2) == round(-index.value() - UNITS * index.credit(), 2) == 12.77  # the T+0 edge
    line = _line(_risk(**query)).replace("\\$", "$")
    assert line.endswith(f"what-if P&L {tab_risk.dollars(now)} vs value now, {tab_risk.dollars(entry)} vs entry")
    assert (tab_risk.dollars(now), tab_risk.dollars(entry)) == ("−$388", "−$401")
    strategy = _chart(_ok(make_app(preset=None, asof=ASOF, tab="Strategy", **query).run(timeout=240)), "payoff_chart")
    marker = next(t for t in strategy["data"] if (t.get("name") or "").startswith("Scenario"))
    assert marker["y"] == [pytest.approx(entry, abs=1e-6)]
    labels = [a["text"] for a in strategy["layout"]["annotations"] if "Scenario" in a.get("text", "")]
    labels += [str(marker.get("text", "")), str(marker.get("name", ""))]
    assert any(text.endswith(f"{signed(entry, 2, '$')} vs entry") for text in labels), labels


# --- the reports ------------------------------------------------------------------------------------------------------


def test_the_ladder_zero_row_equals_the_tiles_and_rolls_to_the_next_session(index):
    at = _risk()
    ladder = _frame(at, "Move")
    zero = ladder[ladder["Move"].str.startswith("0")].iloc[0]
    tiles = _tiles(at)
    assert zero["P&L"] == 0
    for column, label in (("$Δ", "$Δ"), ("Γ P&L 1%", "Γ P&L 1%"), ("Vega", "Vega ($/vp)"), ("Θ", "Θ ($/day)")):
        assert f"{zero[column]:,.0f}" == _shown(tiles[label].value), column
    assert zero["Move"] == "0 · now"
    # Each row is the position revalued at its spot (a percent move), each leg at its own vol: to the dollar, the
    # column's precision (components.table rounds what it sends).
    for x, (_, row) in zip(scenarios.DEFAULT_LADDER_SHOCKS, ladder.iterrows(), strict=True):
        assert row["P&L"] == round(index.value(s=index.s * (1 + x / 100)) - index.value()), row["Move"]
    rolled = _frame(_risk(lad_roll="Next session"), "Move")
    zero = rolled[rolled["Move"].str.startswith("0")].iloc[0]
    assert zero["Move"] == "0 · 26-Oct-26"  # Friday 16:00 to Monday 16:00: "now" read as no decay
    assert _frame(_risk(lad_roll="+1w"), "Move")["Move"].str.fullmatch("0 · 30-Oct-26").any()
    decay = index.value(t=index.t - WEEKEND) - index.value()
    assert zero["P&L"] == round(decay) == 125  # the column's whole dollars
    # Under a what-if the zero row is the what-if state (spot 5,684 on Fri 30-Oct), not "now", while the line reads
    # its P&L against the value now; the help says which clock its σ steps count.
    shifted = _risk(ws=-2, wd=5)
    moves = _frame(shifted, "Move")["Move"]
    assert "0 · what-if" in set(moves) and "0 · now" not in set(moves)
    move_help = json.loads(next(d for d in shifted.dataframe if "Move" in d.value.columns).proto.columns)["Move"]
    assert "a calendar day as Θ counts it (the VaR's day is a trading day, σ·√(1/252))" in move_help["help"]


def test_small_greek_columns_keep_their_cents_as_the_tiles_do():
    # The textbook market's single contract of a 100/110 call spread: Γ P&L 1% −$0.05 and Θ −$0.14 a day, which
    # whole dollars would show as 0 beside tiles that read −$0.05 and −$0.14.
    at = _risk(preset="Textbook")
    tiles = _tiles(at)
    ladder = next(d for d in at.dataframe if "Move" in d.value.columns)
    zero = ladder.value[ladder.value["Move"].str.startswith("0")].iloc[0]
    for column, label in (("Γ P&L 1%", "Γ P&L 1%"), ("Θ", "Θ ($/day)")):
        assert tab_risk.dollars(zero[column]) == tiles[label].value, column
    config = json.loads(ladder.proto.columns)
    # A column has one precision: vega reaches −$14 at +20%, so it stays in whole dollars (−2 beside −$2.05).
    assert ladder.value["Vega"].abs().max() >= 10 and config["Vega"]["type_config"]["format"] == "%,.0f"
    assert config["Γ P&L 1%"]["label"] == "Γ P&L 1% ($)" and config["Θ"]["label"] == "Θ ($/day)"
    assert config["Θ"]["type_config"]["format"] == "%,.2f" and config["$Δ"]["type_config"]["format"] == "%,.0f"
    # Under the index the same columns stay in whole dollars (the condor's Θ reaches +$43, its Γ P&L 1% −$69).
    index_ladder = next(d for d in _risk().dataframe if "Move" in d.value.columns)
    assert json.loads(index_ladder.proto.columns)["Θ"]["type_config"]["format"] == "%,.0f"


def test_the_theta_tile_without_a_breakeven_names_the_implied_move():
    # The textbook 100/110 call spread over a year: Θ −$0.14 and Γ P&L 1% −$0.05 have one sign (the carry of the
    # 5% rate outweighs the decay), so no daily move pays for a day of theta. The slot gives the implied daily move,
    # 20% ÷ √365 = 1.05%, rather than "BE – vs 1.05%".
    tile = _tiles(_risk(preset="Textbook"))["Θ ($/day)"]
    assert tile.value == "−$0.14" and _tiles(_risk(preset="Textbook"))["Γ P&L 1%"].value == "−$0.05"
    assert f"{0.20 / math.sqrt(365):.2%}" == "1.05%"
    assert tile.delta == "1.05% implied move"


def test_a_what_if_step_and_the_var_report_are_fast():
    # Spec 3.3's budgets, as loose guards (timing assertions only as such): a what-if step updates the tiles in 1.0 s
    # and the VaR report renders in 2.5 s. Measured on a shared 4-core machine: 0.10 to 0.14 s a step (the tiles'
    # 63-cell grid and 14 stresses), 0.35 s for the VaR report with both simulations of 100,000 paths uncached.
    # Three times the budget, for a loaded CI runner.
    import time

    from options_analyzer.ui import cache

    at = _risk()
    for step in (-1.0, -2.0):
        start = time.perf_counter()
        _ok(at.number_input(key="ws").set_value(step).run(timeout=240))
        assert time.perf_counter() - start < 3.0
    cache.value_at_risk.clear()
    start = time.perf_counter()
    _ok(at.pills(key="report").set_value("VaR").run(timeout=240))
    assert time.perf_counter() - start < 7.5


def test_the_ladder_in_sigma_and_second_order(index):
    # A σ step is a daily standard deviation of spot at the front expiry's ATMF vol (15.52%), the market's one vol,
    # which the help names: no leg of the condor is at it.
    at = _risk(lad_axis="σ", lad_order="+2nd")
    ladder = _frame(at, "Move")
    assert len(ladder) == len(tab_risk.LADDER_SIGMA)
    assert {"Vanna", "Volga", "Charm"} <= set(ladder.columns)
    sd = index.atm * math.sqrt(1 / 365)
    spots = [index.s * math.exp(k * sd) for k in tab_risk.LADDER_SIGMA]
    assert ladder["Spot"].tolist() == [round(s, 2) for s in spots]  # the level column's cents
    assert ladder["Move"].iloc[0] == f"−3σ · {signed(100 * math.expm1(-3 * sd), 2)}%" == "−3σ · −2.41%"
    named = f"{index.atm:.2%}, the 18‑Dec‑26 AM ATMF vol"
    assert named == "15.52%, the 18‑Dec‑26 AM ATMF vol"
    assert named in json.loads(next(d for d in at.dataframe if "Move" in d.value.columns).proto.columns)["Move"]["help"]
    assert any(named in c.proto.help for c in at.get("button_group") if c.proto.id.endswith("lad_axis"))
    textbook = _risk(preset="Textbook", lad_axis="σ")
    assert any("20.00%, the sidebar vol" in c.proto.help for c in textbook.get("button_group"))


def test_the_stress_table_is_sorted_worst_first_with_readable_labels(index):
    at = _risk(report="Stress")
    table = _frame(at, "Scenario")
    assert table["P&L"].is_monotonic_increasing
    assert len(table) == len(tab_risk.ALL_STRESSES)
    assert "IV damped" not in table.columns and "IV shock damped" not in table.columns
    # P&L second, Source last: a phone shows the first three columns, and P&L sat seventh, off-screen.
    assert list(table.columns) == ["Scenario", "P&L", "Spot", "1M vol", "Rate", "Days", "Source"]
    # Each P&L is the condor's, each leg's vol multiplied by the stress (damped by its own 55.7 days when the stress
    # is a one-month shock), in the order of the independent values.
    stresses = _stresses(index)
    assert table["Scenario"].tolist() == sorted(stresses, key=stresses.get)
    assert table["P&L"].tolist() == [round(stresses[n]) for n in table["Scenario"]]  # the column's dollars
    chart = _chart(at, "stress_chart")
    bars = chart["data"][0]
    assert list(bars["y"]) == table["Scenario"].tolist()
    # Each amount in the column right of the plot, in the bars' order, signed with U+2212, whole dollars from $100:
    # on the bars they were clipped or ran over the scenario names on a phone (tests/test_risk_e2e.py measures it).
    notes = chart["layout"]["annotations"]
    amounts = [n for n in notes if n["xanchor"] == "right"]
    # From the amounts themselves, not the table's whole dollars: the rate shock's −$57.08 reads to the cent.
    pnl = [stresses[n] for n in table["Scenario"]]
    assert [n["y"] for n in amounts] == table["Scenario"].tolist()
    assert [n["text"] for n in amounts] == [signed(v, 0 if abs(v) >= 100 else 2, "$") for v in pnl]
    assert all(n["xref"] == "paper" and n["x"] == 1.0 for n in amounts)
    # Each name above its bar at the plot's left edge, not as a tick label left of the plot: on a phone the tick
    # labels took 150 px and left the bars about 70.
    names = [n for n in notes if n["xanchor"] == "left"]
    assert [(n["y"], n["text"]) for n in names] == [(name, name) for name in table["Scenario"]]
    assert all(n["x"] == 0.0 and n["yshift"] > 0 for n in names)
    assert chart["layout"]["yaxis"]["showticklabels"] is False
    assert "text" not in bars


def _one_month_vol(index: Index) -> float:
    """The surface's one-month ATMF vol at ASOF, located at the forward of the front expiry's carry."""
    forward_ratio = math.exp((index.r - index.q) * ONE_MONTH)
    return float(index.context.surface.implied_vol_at(forward_ratio, ONE_MONTH, extrapolate=True))


def test_the_stress_tables_one_month_vol_is_at_the_what_if_state(index):
    # The 1M vol column is the stress's multiplier on the 1M ATMF vol the stress meets, the what-if's. With +3 vp it
    # read today's 15.38% under the index (+7.69 vp for the vol spike, as at rest) but the shifted sidebar vol under
    # the textbook market (+11.50 vp).
    one_month = _one_month_vol(index)
    for query, sigma in (({}, one_month), ({"wv": 3}, one_month + 0.03)):
        table = _frame(_risk(report="Stress", **query), "Scenario").set_index("Scenario")
        assert table.loc["Vol spike (IV +50%)", "1M vol"] == round(100 * 0.5 * sigma, 2)  # the column's 0.01 vp
    assert round(100 * 0.5 * one_month, 2) == 7.69 and round(100 * 0.5 * (one_month + 0.03), 2) == 9.19
    textbook = _frame(_risk(report="Stress", preset="Textbook", wv=3), "Scenario").set_index("Scenario")
    assert textbook.loc["Vol spike (IV +50%)", "1M vol"] == round(100 * 0.5 * 0.23, 2)
    # 1M-scaled: the one-month vol takes the whole shift (the damping is 1 at one month), whatever the front's tenor.
    scaled = _frame(_risk(report="Stress", preset="Textbook", wv=10, wvmode="1M-scaled"), "Scenario")
    assert scaled.set_index("Scenario").loc["Vol spike (IV +50%)", "1M vol"] == round(100 * 0.5 * 0.22, 2)


@pytest.mark.parametrize("mode", list(tab_risk.DESK_VOL))
def test_the_desk_grid_axes_and_its_cells(index, mode):
    at = _risk(report="Scenarios", sc_volmode=mode)
    heat = _chart(at, "sc_heatmap_True")["data"][0]
    unit = "vp" if mode == "vp" else "%"
    assert all(label.endswith(unit) for label in heat["y"])
    assert len(heat["x"]) == 9 and len(heat["y"]) == 7
    assert all(re.fullmatch(r"[0-9,]+ \([+−]?[0-9.]+%\)", label) for label in heat["x"]), heat["x"]
    z = _array(heat["z"])
    zero = "0 vp" if mode == "vp" else "0%"
    assert z[heat["y"].index(zero), heat["x"].index("5,800 (0%)")] == 0.0
    # Every cell: each leg's own vol shocked by the mode (1M-scaled damped by the legs' 55.7 days), today.
    grid = _desk_grid(index, mode)
    expected = [[grid[(x, dv)] for x in tab_risk.DESK_SPOT] for dv in tab_risk.DESK_VOL[mode]]
    assert z == pytest.approx(np.array(expected), abs=1e-6)


def test_the_spot_by_date_grid_uses_trading_dates(index):
    heat = _chart(_risk(report="Scenarios", sc_axes="Spot × Date"), "sc_heatmap_False")["data"][0]
    assert not any("Day " in label for label in heat["y"])
    # 39 sessions to the 18-Dec expiry: today, every ten sessions, then each of the last five. Thanksgiving (26-Nov)
    # is no session: T+30 is Monday 7-Dec, where counting it would make it Friday 4-Dec.
    assert heat["y"] == [
        "T+0 · Fri 23‑Oct",
        "T+10 · Fri 06‑Nov",
        "T+20 · Fri 20‑Nov",
        "T+30 · Mon 07‑Dec",
        "T+35 · Mon 14‑Dec",
        "T+36 · Tue 15‑Dec",
        "T+37 · Wed 16‑Dec",
        "T+38 · Thu 17‑Dec",
        "T+39 · Fri 18‑Dec exp",
    ]
    z = _array(heat["z"])
    today = index.value()
    spots = [index.s * (1 + x / 100) for x in tab_risk.DESK_SPOT]
    assert z[0] == pytest.approx([index.value(s=s) - today for s in spots], abs=1e-6)
    # The last row is the expiry's close, after the AM settlement at the open: every leg at its intrinsic value.
    assert z[-1] == pytest.approx([index.value(s=s, t=0.0) - today for s in spots], abs=1e-6)


def test_a_long_dated_spot_by_date_grid_keeps_a_few_rows_with_their_years():
    # The textbook's 365-day option is about 250 sessions away: rows 50 sessions apart, then the last five, the dates
    # with their years since the grid crosses one (it drew 55 rows 5 sessions apart, overprinted).
    heat = _chart(_risk(report="Scenarios", sc_axes="Spot × Date", preset="Textbook"), "sc_heatmap_False")["data"][0]
    assert 6 <= len(heat["y"]) <= 11
    assert re.fullmatch(r"T\+0 · \w{3} \d\d‑\w{3}‑\d\d", heat["y"][0]), heat["y"][0]
    assert heat["y"][1].startswith("T+50 ·")


def test_the_finra_grid_and_its_illustrative_requirement(index):
    at = _risk(report="Scenarios", sc_grid="Portfolio margin")
    heat = _chart(at, "sc_heatmap_margin")["data"][0]
    assert len(heat["x"]) == 10
    assert heat["x"][0].endswith("(−8%)") and heat["x"][-1].endswith("(+6%)")
    # Ten points from −8% to +6% (FINRA 4210(g)(2)(F), SPX), vols unchanged: the condor loses most at +6%, where its
    # short call spread is nearer (the spec's −$3,160 differs).
    today = index.value()
    points = np.linspace(-0.08, 0.06, 10)
    pnl = [index.value(s=index.s * (1 + x)) - today for x in points]
    assert _array(heat["z"])[0] == pytest.approx(pnl, abs=1e-6)
    worst = min(pnl)
    assert pnl.index(worst) == 9
    line = next(m.value for m in at.markdown if "Requirement" in m.value)
    assert "illustrative" in line and "4210(g)(7)(B)" in line
    assert f"{tab_risk.dollars(worst)} at +6%".replace("$", "\\$") in line
    assert "−\\$3,141 at +6%" in line and f"\\${-worst:,.0f}, the greater" in line
    # The minimum: four listed contracts at $0.375 × 100 each, $150.00, which the largest loss exceeds.
    assert "\\$150.00" in line


def _dgt_var(delta: float, gamma: float, theta: float, s: float, sigma: float, confidence: float = 0.99) -> float:
    """VaR (a loss, positive) of θh/252 + Δx + ½Γx² over one trading day, x ~ N(0, (Sσ√(1/252))²), found as the P&L
    level q whose loss region has probability 1 − confidence: for Γ < 0 the two tails beyond the roots of the
    quadratic, from the Normal distribution directly (no chi-square)."""
    sd = s * sigma * math.sqrt(1 / 252)
    decay = theta / 252

    def tail(q: float) -> float:
        # P&L(x) ≤ q  ⇔  ½Γx² + Δx + (θh − q) ≤ 0, outside the roots when Γ < 0.
        a, b, c = 0.5 * gamma, delta, decay - q
        root = math.sqrt(b * b - 4 * a * c)
        x1, x2 = sorted(((-b - root) / (2 * a), (-b + root) / (2 * a)))
        return norm.cdf(x1 / sd) + norm.sf(x2 / sd) - (1 - confidence)

    assert gamma < 0
    top = decay - delta**2 / (2 * gamma)  # the vertex: the largest P&L, far beyond the distribution here
    bottom = decay - abs(delta) * 12 * sd + 0.5 * gamma * (12 * sd) ** 2  # twelve standard deviations out
    return -brentq(tail, bottom, top - 1e-6 * abs(top), xtol=1e-10)


def test_the_var_report(index):
    from dataclasses import replace

    from options_analyzer import risk

    at = _risk(report="VaR")
    hist = _chart(at, "var_hist")
    assert hist["layout"]["yaxis"]["title"]["text"] == "Probability (%)"
    cards = {m.label: m for m in at.metric}
    for label in ("VaR 99% 1D · MC", "ES 97.5% 1D · MC", "Spot-only VaR", "Spot-only ES"):
        assert cards[label].value.startswith("−"), label
    # The simulations are the library's (tests/test_risk.py checks the engine) on the four legs built here at their
    # own surface vols, the spot paths at the ATMF vol; the DGT VaR is solved here from the Normal distribution.
    legs = tuple(strat.Leg(kind, n, k, index.t, v) for (n, kind, k), v in zip(LEGS, index.vols, strict=True))
    condor = strat.Strategy("Iron Condor", legs)
    market = strat.Market(index.s, index.r, index.atm, index.t, index.q)
    moving = risk.VaRConfig(0.99, 1, 100_000, seed=7, vol_of_vol=1.0, spot_vol_correlation=-0.7)
    full = risk.value_at_risk(condor, market, moving, UNITS)
    spot = risk.value_at_risk(condor, market, replace(moving, vol_of_vol=0.0), UNITS)
    es = (risk.tail_measures(full.pnl, 0.975)[1], risk.tail_measures(spot.pnl, 0.975)[1])
    delta, gamma, _, theta = index.greeks()
    dgt = _dgt_var(delta, gamma, theta, index.s, index.atm)
    expected = {
        "VaR 99% 1D · MC": -full.var,
        "ES 97.5% 1D · MC": -es[0],
        "Spot-only VaR": -spot.var,
        "Spot-only ES": -es[1],
        "DGT VaR 99% 1D": -dgt,
    }
    # Every gap between the cards is $10 or more, so they are in whole dollars (the VaR's 95% simulation interval
    # is −$471.51 to −$487.45: no cent of it is known), each amount to the cent in its help.
    assert [round(v, 2) for v in expected.values()] == [-479.95, -492.32, -713.95, -727.60, -725.40]
    assert {label: cards[label].value for label in expected} == {k: tab_risk.dollars(v) for k, v in expected.items()}
    assert [cards[label].value for label in expected] == ["−$480", "−$492", "−$714", "−$728", "−$725"]
    for label, amount in expected.items():
        assert tab_risk.exact(amount).replace("$", "\\$") in cards[label].help, label
    assert "95% simulation interval −\\$471.51 to −\\$487.45" in cards["VaR 99% 1D · MC"].help
    # The gaps from the amounts shown, so that the cards add up on screen: −$480 − (−$714), −$492 − (−$728) and
    # −$714 − (−$725), the comparator taken like for like (implied vol frozen in both).
    assert cards["VaR 99% 1D · MC"].delta == "+\\$234 vs spot only"
    assert cards["ES 97.5% 1D · MC"].delta == "+\\$236 vs spot only"
    gap = tab_risk.shown(-spot.var) - tab_risk.shown(-dgt)
    assert cards["Non-linearity"].value == tab_risk.dollars(gap) == "+$11"
    assert cards["Non-linearity"].delta == "spot only − DGT"
    # The spot vol is named where it is used: the front expiry's ATMF vol, not a leg's.
    assert "at 15.52%, the 18‑Dec‑26 AM ATMF vol" in cards["DGT VaR 99% 1D"].help
    assert "Spot moves at 15.52%, the 18‑Dec‑26 AM ATMF vol; each leg re-priced at its own vol" in cards[
        "VaR 99% 1D · MC"
    ].help.replace("\\", "")
    basis = next(c.value for c in at.caption if "Monte Carlo" in c.value)
    assert basis == "Normal Monte Carlo, 100,000 paths, seed 7; spot vol 15.52%, vol of vol 100%, correlation −0.70."
    # The models simulate from today's vols: no statistic of a history, so none is tagged SYN HIST; the line under
    # the report points to the backtests on the synthetic history.
    assert not any("SYN HIST" in label for label in cards)
    line = next(c.value for c in at.caption if "SYN HIST" in c.value)
    assert line.startswith("SYN HIST · ") and "Methodology" in line
    # The default model with the default vol dynamics: the Methodology's backtest of it, 31 exceptions against 20.2.
    assert "31 exceptions at 99% vs 20.2 expected, amber" in line
    # The bars are shares of all outcomes in percent: they add up to the share inside the plotted range.
    bars = _bars(hist)
    assert 99.0 < bars.sum() <= 100.0 + 1e-9


def _amount(text: str) -> float:
    """A tile's dollar amount as a number: ``−$1,922`` is −1,922."""
    return float(text.replace("−", "-").replace("$", "").replace(",", ""))


def _bars(fig: dict) -> np.ndarray:
    return _array(fig["data"][0]["y"])


def _array(value) -> np.ndarray:
    """A trace's data array; Plotly sends numeric NumPy arrays as base64 ``bdata`` (with a shape for 2-D)."""
    import base64

    if isinstance(value, dict) and "bdata" in value:
        array = np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"])
        shape = value.get("shape")
        if shape is not None:
            array = array.reshape([int(n) for n in str(shape).split(",")] if isinstance(shape, str) else shape)
        return array
    if isinstance(value, list) and value and isinstance(value[0], (list, dict)):
        return np.array([_array(row) for row in value], dtype=float)
    return np.asarray(value, dtype=float)


def test_legs_that_settle_within_the_horizon_are_counted_and_leave_no_expansion():
    # A 1-day textbook call spread settles on day 1, the horizon itself: the caption counts its two legs (ported
    # from the earlier Risk tab's test). Two days out (1.38 trading days) it outlives the horizon.
    at = _risk(report="VaR", preset="Textbook", days=1)
    basis = next(c.value for c in at.caption if "Monte Carlo" in c.value)
    assert basis.endswith(" 2 leg(s) settle before the horizon.")
    assert not any("settle before" in c.value for c in _risk(report="VaR", preset="Textbook", days=2).caption)
    # A 3-day spread at a 5-day horizon: the expansion in today's Greeks is evaluated past expiry, where it read
    # "DGT VaR −$91.46" beside a full VaR of −$39.80 (the whole debit). The comparator, its gap and the
    # delta-normal VaR read – and say why.
    at = _risk(report="VaR", preset="Textbook", days=3)
    at.number_input(key="var_horizon").set_value(5).run(timeout=240)
    cards = {m.label: m for m in _ok(at).metric}
    assert cards["DGT VaR 99% 5D"].value == "–" and cards["Non-linearity"].value == "–"
    assert "2 leg(s) settle within the 5-day horizon" in cards["DGT VaR 99% 5D"].help
    stats = next(t.value for t in at.table if "Statistic" in t.value.columns).set_index("Statistic")["Value"]
    assert stats["Delta-normal VaR 99%"] == "–" and cards["VaR 99% 5D · MC"].value == stats["Worst path"]
    assert cards["Non-linearity"].delta is None or cards["Non-linearity"].delta == ""


def test_the_quoted_backtest_is_the_one_day_var_only():
    # The Methodology's backtest is of the 1-day VaR 99%: at five days the same model is not the one backtested
    # (it read "This model on 2,020 synthetic days: 31 exceptions" beside a 5D VaR).
    at = _risk(report="VaR", preset="Textbook")
    assert "This model on 2,020 synthetic days" in next(c.value for c in at.caption if "SYN HIST" in c.value)
    at.number_input(key="var_horizon").set_value(5).run(timeout=240)
    assert "Not backtested with this model, horizon" in next(c.value for c in _ok(at).caption if "SYN HIST" in c.value)


def test_the_ladder_chart_has_a_tick_on_the_loss_side():
    # Plotly's own step was 5,000 for P&L from −$4,096 to +$5,871: no tick below zero, no loss readable. The condor's
    # ladder runs from −$7,646 (+20%) to +$120 (−1%): ticks down from 0 in the loss, none above it, which the data
    # reach by less than a tenth of the range (charts_risk.signed_ticks).
    at = _risk()
    ladder = _frame(at, "Move")
    yaxis = _chart(at, "ladder_chart")["layout"]["yaxis"]
    ticks = yaxis["tickvals"]
    low, high = ladder["P&L"].min(), ladder["P&L"].max()
    assert (low, high) == (-7646, 120) and high < 0.1 * (high - low)
    assert min(ticks) < 0 == max(ticks) and min(ticks) >= low and len(ticks) >= 3
    assert yaxis["tickformat"] == ",.0f"


@pytest.mark.parametrize(
    "query",
    [
        {"strike": "20"},  # the 20 / 22 call spread on spot 100: both legs deep in the money
        {"strike": "500", "strategy": "Long Call"},  # far out of the money
        {"strike": "20", "strategy": "Iron Condor"},
    ],
)
def test_a_flat_ladder_draws_a_flat_chart(query):
    # The P&L column is float noise of about 1e-10 here (the table shows 0): its ticks (0, 1e-9, 2e-9) fitted no count
    # of decimals up to 6, and the default report raised StopIteration. Flat to the cent, the chart takes Plotly's own
    # ticks in whole dollars.
    at = _risk(preset="Textbook", **query)
    ladder = _frame(at, "Move")
    assert np.allclose(ladder["P&L"], 0.0, atol=0.005)
    yaxis = _chart(at, "ladder_chart")["layout"]["yaxis"]
    assert "tickvals" not in yaxis and yaxis["tickformat"] == ",.0f"


def test_ladder_ticks_below_a_cent_are_flat():
    from options_analyzer.ui import charts_risk

    yaxis = charts_risk.ladder_chart([1.0, 2.0], [-1.2e-8, 4e-8], "P&L ($)", spot=1.0).layout.yaxis
    assert yaxis.tickvals is None and yaxis.tickformat == ",.0f"
    assert charts_risk.signed_ticks([-1e-12, 3e-12]) is None  # five ticks that all read 0.0, before


def test_small_var_cards_are_to_the_cent_and_add_up_on_screen():
    # The textbook contract's VaR is about $55: whole dollars read −$54 beside a spot-only −$55 and "+$0.58 vs spot
    # only". Below $1,000 the six cards are to the cent and each gap is the difference of two amounts shown.
    cards = {m.label: m for m in _risk(report="VaR", preset="Textbook").metric}
    amount = {k: _amount(cards[k].value) for k in ("VaR 99% 1D · MC", "Spot-only VaR", "DGT VaR 99% 1D")}
    assert all(re.fullmatch(r"−\$\d+\.\d\d", cards[k].value) for k in amount), [cards[k].value for k in amount]
    vs_spot = signed(amount["VaR 99% 1D · MC"] - amount["Spot-only VaR"], 2, "$")
    assert cards["VaR 99% 1D · MC"].delta == vs_spot.replace("$", "\\$") + " vs spot only"  # Markdown
    assert cards["Non-linearity"].value == signed(amount["Spot-only VaR"] - amount["DGT VaR 99% 1D"], 2, "$")


def test_the_tabs_doctests_hold():
    import doctest

    runner = doctest.DocTestRunner()
    for test in doctest.DocTestFinder().find(tab_risk):
        runner.run(test)
    assert runner.tries >= 9 and runner.failures == 0


def test_the_quoted_backtest_is_the_methodologys_and_follows_the_vol_dynamics():
    # One source: the numbers the VaR report quotes are those of the Methodology's Backtesting section (whose own
    # numbers tests/test_backtest.py's acceptance 7 measures).
    from pathlib import Path

    text = " ".join((Path(tab_risk.__file__).parent / "methodology" / "risk.md").read_text(encoding="utf-8").split())
    assert f"{tab_risk.SYN_BACKTEST_DAYS:,} days ({tab_risk.SYN_BACKTEST_EXPECTED:g} exceptions expected" in text
    (moving, moving_zone), (frozen, frozen_zone) = tab_risk.SYN_BACKTEST[True], tab_risk.SYN_BACKTEST[False]
    assert f"implied vol frozen has {frozen} exceptions ({frozen_zone})" in text
    assert f"default with implied-vol moves {moving} ({moving_zone})" in text
    at = _risk(report="VaR")
    at.toggle(key="var_volmoves").set_value(False).run()
    assert "50 exceptions at 99% vs 20.2 expected, red" in next(
        c.value for c in _ok(at).caption if "SYN HIST" in c.value
    )
    at.segmented_control(key="var_model").set_value("Student-t").run()
    assert "Not backtested" in next(c.value for c in _ok(at).caption if "SYN HIST" in c.value)


def _bucket(kind: str, k: float, index: Index, v: float) -> str:
    """A leg's strike bucket: the delta bucket whose undiscounted put delta N(−d1) is nearest the leg's own, at its
    own vol."""
    put_delta = norm.cdf(-_d1(index.s, k, index.t, index.r, index.q, v))
    centres = {"10Δp": 0.10, "25Δp": 0.25, "ATM": 0.50, "25Δc": 0.75, "10Δc": 0.90}
    return min(centres, key=lambda name: abs(centres[name] - put_delta))


def test_the_vega_report_buckets_add_up_to_the_vega_tile(index):
    at = _risk(report="Vega")
    table = _frame(at, "Expiry")
    total = table[table["Expiry"] == "Total"]["Total"].iloc[0]
    assert f"{total:,.0f}" == _shown(_tiles(at)["Vega ($/vp)"].value) == "-306"
    # Only the buckets the heatmap draws, and the total: the 18-Dec legs listed nine rows of zeros around them.
    assert table["Expiry"].tolist() == ["30–60d", "Total"]
    assert list(table.columns) == ["Expiry", "10Δp", "25Δp", "ATM", "25Δc", "10Δc", "Total"]
    # The 18-Dec legs (55.7 days) sit in 30–60d. By delta at their own vols, the 5,300 and 5,400 puts (put deltas
    # N(−d1) 0.11 and 0.15) both go to 10Δp and the 6,150 and 6,250 calls (0.86 and 0.93) both to 10Δc: each pair is
    # a long wing and a short body, whose vegas net to −$102 and −$204.
    buckets = {k: _bucket(kind, k, index, v) for (_, kind, k), v in zip(LEGS, index.vols, strict=True)}
    assert buckets == {5300.0: "10Δp", 5400.0: "10Δp", 6150.0: "10Δc", 6250.0: "10Δc"}
    vegas = {k: n * _greeks(kind, index.s, k, index.t, index.r, index.q, v)[2] for (n, kind, k), v in
             zip(LEGS, index.vols, strict=True)}  # fmt: skip
    row = table[table["Expiry"] == "30–60d"].iloc[0]
    for column in ("10Δp", "25Δp", "ATM", "25Δc", "10Δc"):
        expected = sum(vega for k, vega in vegas.items() if buckets[k] == column)
        assert row[column] == round(expected), column  # the column's whole dollars
    assert row["10Δc"] < row["10Δp"] < 0


def test_the_explain_report_explains_the_what_if_move(index):
    at = _risk(report="Explain", ws=-2, wv=3, wd=5)
    table = _frame(at, "Term")
    s, t, vols = index.s * 0.98, index.t - WEEK, [v + 0.03 for v in index.vols]
    total = index.value(s, t, vols=vols) - index.value()
    full = table[table["Term"] == "Full revaluation"]["P&L"].iloc[0]
    assert full == round(total, 2)  # the column's cents
    terms = table[table["Term"] != "Full revaluation"]["P&L"]
    assert terms.sum() == pytest.approx(full, abs=len(terms) * 0.005)  # each row rounded to the cent
    # The vega term is each leg's vega times its own +3 vp, at today's market.
    vega = index.greeks()[2]
    assert table.set_index("Term").loc["Vega", "P&L"] == round(vega * 0.03, 2)
    shapley = _frame(_risk(report="Explain", ws=-2, wv=3, wd=5, ex_method="Shapley"), "Term")
    assert shapley["Term"].tolist() == ["Spot", "Vol", "Time", "Rate", "Full revaluation"]
    # With every amount below $1,000 the waterfall's labels are to the cent, as the table is.
    labels = [label for label in _chart(at, "explain_waterfall")["data"][0]["text"] if label]
    assert abs(full) < 1000 and all(re.fullmatch(r"[+−]\$[\d,]+\.\d\d", label) for label in labels), labels
    # No shift set: the what-if has nothing to explain, so the report explains the next session (Friday 16:00 to
    # Monday 16:00, three calendar days, nothing else moving) and says that Θ carries it all.
    quiet = _risk(report="Explain")
    assert any(c.value.startswith("No what-if shift is set: only time moves") for c in quiet.caption)
    decay = index.value(t=index.t - WEEKEND) - index.value()
    full = _frame(quiet, "Term").set_index("Term").loc["Full revaluation", "P&L"]
    assert full == round(decay, 2)  # the column's cents
    # Only time moves, so every other term is zero and carries no label; the time decay by repricing is exact.
    labels = _chart(quiet, "explain_waterfall")["data"][0]["text"]
    assert decay > 0 and [label for label in labels if label] == [f"+${decay:.2f}"] * 2  # Θ and the total
    assert any("unexplained (0.0% of the P&L)" in c.value for c in quiet.caption)


def test_the_footer_states_each_legs_vol(index):
    # The footer named one vol for every leg, the front expiry's ATMF vol ("BSM European legs at 15.52% (the
    # contract's surface vol)"), at which no leg of the condor is valued. It now gives the legs' own vols, read off
    # the surface at their strikes, and the sidebar vol on the textbook market.
    footer = [c.value for c in _risk().tabs[2].caption if c.value.startswith("Full revaluation")]
    low, high = min(index.vols), max(index.vols)
    assert footer == [
        f"Full revaluation · BSM European legs at their surface IVs, {low:.2%} to {high:.2%} · sticky strike · 4 legs "
        "· 1 × 100"
    ]
    assert f"{index.atm:.2%}" not in footer[0] and "12.06% to 20.56%" in footer[0]
    textbook = [c.value for c in _risk(preset="Textbook").tabs[2].caption if c.value.startswith("Full revaluation")]
    assert textbook == [
        "Full revaluation · BSM European legs at the sidebar vol, 20.00% · sticky strike · 2 legs · 1 × 100"
    ]


def test_the_footer_counts_ivs_typed_over_the_surface(index):
    # A typed IV replaces a leg's surface vol (legs.price_index marks it "override"): the footer counts it and its
    # range takes it in. Under the textbook market a leg's typed IV shows beside the sidebar vol.
    table = lg.index_template("Iron Condor", index.published, EXPIRY)
    table.loc[0, lg.IV] = 25.0
    book = lg.price_index(table, index.published, index.context.as_of, name="Iron Condor")
    assert tab_risk.leg_vols(Position(book.strategy, book.market, 1, 100.0, book)) == (
        f"their surface IVs (1 typed over), {min(index.vols):.2%} to 25.00%"
    )
    table.loc[:, lg.IV] = 18.0
    book = lg.price_index(table, index.published, index.context.as_of, name="Iron Condor")
    assert tab_risk.leg_vols(Position(book.strategy, book.market, 1, 100.0, book)) == "their typed IVs, 18.00%"
    market = strat.Market(100.0, 0.05, 0.20, 1.0)
    rows = lg.textbook_template("Bull Call Spread", 100.0, 10.0, 1.0)
    rows.loc[1, lg.IV] = 22.5
    book = lg.price_textbook(rows, market, 100.0)
    typed = tab_risk.leg_vols(Position(book.strategy, market, 1, 100.0, book))
    assert typed == "20.00% to 22.50% (1 typed, sidebar 20.00%)"


def test_each_leg_moves_by_its_own_vol_and_its_own_time_left(index):
    # A calendar on two listed expiries, each leg at its own surface vol: −1 5,830C 18-Dec-26 AM (55.7 days) and
    # +1 5,830C 19-Feb-27 AM (118.7 days), valued at the December carry as the Strategy tab values every leg (the
    # carry is the market's: legs.price_index). A one-month vol shock moves each leg by the share its own time left
    # gives it, at the what-if date as at rest.
    back = Expiry(date(2027, 2, 19), "AM")
    expiries = (index.t, index.published.maturity(back))
    vols = (index.published.vol(5830.0, EXPIRY).vol, index.published.vol(5830.0, back).vol)
    calendar = ((-1.0, "call", 5830.0), (1.0, "call", 5830.0))
    legs = tuple(strat.Leg(kind, n, k, T, v) for (n, kind, k), T, v in zip(calendar, expiries, vols, strict=True))
    market = strat.Market(index.s, index.r, index.atm, index.t, index.q)
    position = Position(strat.Strategy("Long Call Calendar", legs), market, 1, 100.0)
    inputs = SimpleNamespace(context=index.context)

    def value(s, taus, sigmas, r=index.r):
        return UNITS * sum(
            n * _price(kind, s, k, T, r, index.q, v) for (n, kind, k), T, v in zip(calendar, taus, sigmas, strict=True)
        )

    # The what-if: +10% 1M-scaled and T+5. Each leg keeps its own time left, 7 days less, and takes 10% × its damping.
    shifts = whatif.Shifts(vol=10.0, mode="1M-scaled", sessions=5)
    book = tab_risk.make_book(position, inputs, shifts)
    taus = [T - WEEK for T in expiries]
    moved = [v * (1 + 0.10 * _damp(tau)) for v, tau in zip(vols, taus, strict=True)]
    assert [leg.expiry for leg in book.strategy.legs] == pytest.approx(taus, abs=1e-12)
    assert [leg.volatility for leg in book.strategy.legs] == pytest.approx(moved, abs=1e-12)
    assert _damp(taus[0]) > _damp(taus[1])  # the front leg moves more
    # The stresses at the what-if state: each one-month shock damped by each leg's own time left at that date.
    table = tab_risk.stress_frame(book, tab_risk.STRESS_GROUPS[:2], False, True, None).set_index("Scenario")
    base = value(index.s, taus, moved)
    for sc in tab_risk.ALL_STRESSES:
        sigmas = [v * (1 + (sc.vol_multiplier - 1) * (_damp(tau) if sc.one_month_vol_shock else 1.0))
                  for v, tau in zip(moved, taus, strict=True)]  # fmt: skip
        expected = value(index.s * (1 + sc.spot_change), taus, sigmas, index.r + sc.rate_change) - base
        assert table.loc[sc.name, "P&L"] == pytest.approx(expected, abs=1e-6), sc.name
    assert tab_risk.tiles(book).worst_stress == pytest.approx(table["P&L"].min(), abs=1e-9)
    # Today's desk grid in the 1M-scaled mode and the explain of a stress damp by each leg's time left today.
    today = tab_risk.make_book(position, inputs, whatif.Shifts())
    grid = tab_risk.desk_grid(today, "1M-scaled")
    for i, shock in enumerate(tab_risk.DESK_VOL["1M-scaled"]):
        sigmas = [v * (1 + shock / 100 * _damp(T)) for v, T in zip(vols, expiries, strict=True)]
        for j, x in enumerate(tab_risk.DESK_SPOT):
            expected = value(index.s * (1 + x / 100), expiries, sigmas) - value(index.s, expiries, vols)
            assert grid[i, j] == pytest.approx(expected, abs=1e-6), (shock, x)
    covid = next(sc for sc in tab_risk.ALL_STRESSES if sc.name.startswith("COVID"))
    move = tab_risk.explain_target(today, covid.name)
    assert move["vols"] == pytest.approx(
        [v * (1 + (covid.vol_multiplier - 1) * _damp(T)) for v, T in zip(vols, expiries, strict=True)], abs=1e-12
    )


@pytest.mark.parametrize("report", tab_risk.REPORTS)
@pytest.mark.parametrize("preset", [None, "Textbook"])
def test_every_report_runs_with_short_captions(report, preset):
    at = _risk(report=report, preset=preset)
    captions = [c.value for c in at.tabs[2].caption]  # the Risk tab body (the share note is not its own)
    assert all(len(c) <= 140 for c in captions), [c for c in captions if len(c) > 140]
    # The removed views stay removed (spec 3.3 Removed).
    keys = {c.proto.id.rsplit("-", 1)[-1] for c in at.get("plotly_chart")}
    assert not keys & {"time_decay", "greek_profiles", "risk_heatmap", "date_heatmap"}


def test_without_a_position_the_tab_says_so_and_opens_strategy():
    page = AppTest.from_string(
        "from options_analyzer.ui import tab_risk\ntab_risk.render(None, None)", default_timeout=60
    ).run()
    assert page.info[0].value.startswith("The Strategy tab's legs are invalid")
    page.button(key="risk_open_strategy").click().run()
    assert page.session_state["tab"] == "Strategy"


# --- UI-10: a scenario cell into the what-if strip ------------------------------------------------------------------

AXES = GridAxes(
    tab_risk.DESK_SPOT,
    tuple((v, 3) for v in tab_risk.DESK_VOL["vp"]),
    "absolute",
    tuple(tab_risk.shock_label(x / 100) for x in tab_risk.DESK_SPOT),
)


@pytest.fixture
def state(monkeypatch):
    import streamlit as st

    store: dict = {}
    monkeypatch.setattr(st, "session_state", store)
    return store


def test_a_heatmap_point_sets_spot_vol_and_date(state):
    # The overlay's points run row by row: row 4 (+2 vp), column 6 (+5%) is point 4 × 9 + 6.
    tab_risk.on_scenario_select({"selection": {"points": [{"point_index": 42, "x": "6,090 (+5%)"}]}}, AXES)
    assert state == {"wvmode": "absolute", "ws": 5.0, "wv": 2.0, "wd": 3}


def test_a_table_cell_sets_spot_vol_and_date(state):
    tab_risk.on_scenario_select({"selection": {"cells": [(1, "+5%")]}}, AXES)
    assert state == {"wvmode": "absolute", "ws": 5.0, "wv": -5.0, "wd": 3}
    state.clear()
    tab_risk.on_scenario_select({"selection": {"cells": [(1, "Vol shock (vp)")]}}, AXES)  # the row label
    tab_risk.on_scenario_select({"selection": {"points": [], "cells": []}}, AXES)  # a cleared selection
    assert state == {}


def test_a_relative_grid_sets_the_relative_mode(state):
    axes = GridAxes(tab_risk.DESK_SPOT, tuple((v, 0) for v in tab_risk.DESK_VOL["relative"]), "relative")
    tab_risk.on_scenario_select({"selection": {"points": [{"point_index": 0}]}}, axes)
    assert state == {"wvmode": "relative", "ws": -10.0, "wv": -50.0, "wd": 0}


def test_the_strips_cell_is_outlined_in_amber():
    at = _risk(report="Scenarios", ws=5, wv=2)
    shapes = _chart(at, "sc_heatmap_True")["layout"]["shapes"]
    amber = [s for s in shapes if s["line"]["dash"] == "solid"]
    assert len(amber) == 1 and amber[0]["x0"] == 5.5 and amber[0]["y0"] == 3.5  # column +5%, row +2 vp
    resting = _chart(_risk(report="Scenarios"), "sc_heatmap_True")["layout"]["shapes"]
    assert all(s["line"]["dash"] == "dot" for s in resting)  # the worst cell only


# --- Milestone 1, kept: implied-vol moves in the VaR by default, beside the spot-only VaR; decay beside theta -------

#: The review panel's VaR case (market maker, finding 2): a 45-day at-the-money short straddle on an index, one
#: contract of 100 shares, on the textbook market's typed inputs.
SPX_STRADDLE = {
    "preset": "Textbook",
    "spot": 5800,
    "strike": 5800,
    "days": 45,
    "vol": 16,
    "rate": 4.3,
    "div": 1.3,
    "strategy": "Short Straddle",
}


def _var_configs(monkeypatch) -> list:
    """The configuration of every VaR simulation from now on, in order (the cache is cleared, so none hides there)."""
    from options_analyzer import risk
    from options_analyzer.ui import cache

    configs = []
    real = risk.value_at_risk
    monkeypatch.setattr(
        risk, "value_at_risk", lambda s, m, config, n=1.0: configs.append(config) or real(s, m, config, n)
    )
    cache.value_at_risk.clear()
    return configs


def test_var_simulates_index_vol_dynamics_beside_the_spot_only_var(monkeypatch):
    # The panel's numbers: VaR $2,833 with implied vol frozen, $3,862 with index dynamics (vol of vol 100%, spot-vol
    # correlation -0.7), 27% more. Both show, signed as P&L, and the headline cards say what the vol moves add.
    from dataclasses import replace

    from options_analyzer import risk
    from options_analyzer.strategies import Market, build_template

    configs = _var_configs(monkeypatch)
    at = _risk(report="VaR", **SPX_STRADDLE)
    moving = risk.VaRConfig(0.99, 1, 100_000, seed=7, vol_of_vol=1.0, spot_vol_correlation=-0.7)
    assert configs == [replace(moving, vol_of_vol=0.0), moving]
    cards = {m.label: m for m in at.metric}
    assert cards["VaR 99% 1D · MC"].value == "−$3,862" and cards["Spot-only VaR"].value == "−$2,833"
    assert cards["VaR 99% 1D · MC"].delta == r"−\$1,029 vs spot only"  # Markdown, as the delta renders it
    # ES at 97.5% (FRTB's level) off the same simulation: the mean of the 2,500 worst of 100,000.
    straddle = build_template("Short Straddle", 5800, 580, 45 / 365)
    run = risk.value_at_risk(straddle, Market(5800, 0.043, 0.16, 45 / 365, 0.013), moving, 100)
    es = -np.sort(run.pnl)[:2_500].mean()
    assert cards["ES 97.5% 1D · MC"].value == tab_risk.dollars(-es)
    assert at.toggle(key="var_volmoves").value is True
    assert at.number_input(key="var_vov").value == 100.0 and at.slider(key="var_corr").value == -0.7


def test_the_spot_only_run_keeps_every_setting_but_the_vol_moves(monkeypatch):
    from dataclasses import replace

    from options_analyzer import risk

    configs = _var_configs(monkeypatch)
    at = _risk(report="VaR")
    at.number_input(key="var_horizon").set_value(5)
    at.select_slider(key="var_paths").set_value(10_000)
    at.number_input(key="var_vov").set_value(150.0)
    at.slider(key="var_corr").set_value(-0.5)
    at.number_input(key="var_seed").set_value(11)
    at.slider(key="var_dof").set_value(4)
    at.segmented_control(key="var_model").set_value("Student-t").run()
    _ok(at)
    config = risk.VaRConfig(
        0.99, 5, 10_000, distribution="student_t", dof=4.0, seed=11, vol_of_vol=1.5, spot_vol_correlation=-0.5
    )
    assert configs[-2:] == [replace(config, vol_of_vol=0.0, spot_vol_correlation=-0.7), config]
    assert [m.label for m in at.metric if "VaR" in m.label or "ES" in m.label][:4] == [
        "VaR 99% 5D · MC",
        "ES 97.5% 5D · MC",
        "Spot-only VaR",
        "Spot-only ES",
    ]
    # The spot-only pair's labels are short enough for one line of six cards; their help names level and horizon.
    spot_only = {m.label: m for m in at.metric}
    assert spot_only["Spot-only VaR"].help.startswith("VaR 99% 5D of the same spot paths")
    assert spot_only["Spot-only ES"].help.startswith("ES 97.5% 5D, implied vol frozen")


def test_vol_moves_switched_off_show_the_spot_only_run_from_the_cache(monkeypatch):
    configs = _var_configs(monkeypatch)
    at = _risk(report="VaR", **SPX_STRADDLE)
    at.toggle(key="var_volmoves").set_value(False).run()
    _ok(at)
    assert len(configs) == 2  # the frozen run came from the cache
    labels = [m.label for m in at.metric if "VaR" in m.label or "ES" in m.label or m.label == "Non-linearity"]
    assert labels == ["VaR 99% 1D · MC", "ES 97.5% 1D · MC", "DGT VaR 99% 1D", "Non-linearity"]
    assert next(m for m in at.metric if m.label == "VaR 99% 1D · MC").value == "−$2,833"


def test_the_theta_tile_gives_the_decay_to_expiry_of_a_one_day_position():
    # A 1-day short straddle: repricing gains its whole value, $4,359.89, where theta / 365 says $2,179.76 a day.
    from options_analyzer import black_scholes as bs

    tile = _tiles(_risk(**(SPX_STRADDLE | {"days": 1, "vol": 18})))["Θ ($/day)"]
    value = 100 * sum(bs.price(5800, 5800, 1 / 365, 0.043, 0.18, 0.013, kind) for kind in ("call", "put"))
    assert value == pytest.approx(4359.89, abs=0.005)
    assert tile.value == "+$2,180" and tile.delta == r"Decay to expiry +\$4,360"
    assert _tiles(_risk(**SPX_STRADDLE))["Θ ($/day)"].delta.startswith("BE ")  # 45 days: theta leads


def test_every_risk_chart_has_the_desk_config():
    for query in ({}, {"report": "Scenarios"}, {"report": "Stress"}, {"report": "VaR"}, {"report": "Vega"},
                  {"report": "Explain", "ws": 1}, {"report": "Scenarios", "sc_grid": "Portfolio margin"}):  # fmt: skip
        for chart in _risk(**query).get("plotly_chart"):
            key = chart.proto.id.rsplit("-", 1)[-1]
            config = json.loads(chart.proto.config)
            assert config["toImageButtonOptions"]["filename"] == f"options-analyzer-{key}", key
            assert len(chart.proto.alt) >= 20 and not chart.proto.alt[0].islower(), (key, chart.proto.alt)
