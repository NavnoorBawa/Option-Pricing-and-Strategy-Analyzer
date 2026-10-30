"""The what-if strip: spot, vol, date and rate shifts for the Strategy and Risk tabs (UI spec 3.2 item 3, UI-4).

Four shifts, each bound to the URL so that a shared link reproduces the scenario:

* ``ws``, the spot shift in percent, in steps of 0.5: ``Spot −2.0% → 5,684.00``;
* ``wv``, the vol shift, read by its mode ``wvmode``: ``absolute`` (the default) adds vol points to every leg's vol,
  ``relative`` scales it, and ``1M-scaled`` scales it by a shock to one-month vol damped for longer tenors, RK-1's
  ``vol_mode`` ``"1m"`` (:func:`options_analyzer.scenarios.scenario_cube`). Its label works an example at the
  reference expiry's ATMF vol: ``Vol +3.00 vp: ATMF 15.52% → 18.52%``;
* ``wd``, the date as a whole number of trading sessions after the valuation date (``clock.is_trading_day``), labelled
  ``Date T+10 · Fri 06‑Nov‑26 (41d left)``; the scenario's time is the valuation time's wall-clock time on that date,
  so Friday to Monday is 3 calendar days of decay, as theta counts it;
* ``wr``, the rate shift in basis points, raising every expiry's rate with spot and dividend yield held (spec 2.0):
  ``Rate +25 bp: 4.75% → 5.00%``.

While any shift is non-zero the strip leads with an amber ``SCENARIO`` badge and draws an amber rule at its left
(``static/desk.css``), and :func:`note` gives the muted line the other tabs end their context line with. The pure
functions (:func:`apply`, :func:`leg_vol`, :func:`session_dates`, :func:`scenario_time`) hold the semantics and are
unit-tested; :func:`strip` only draws the widgets. Undo (UI-24) is not here.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta

import streamlit as st

from .. import clock
from ..scenarios import DEFAULT_VOL_TENOR_DECAY
from ..strategies import VOL_SHOCK_REFERENCE_TENOR, Market
from .format import MINUS, bp, level, pct, session_label, signed, time_left, vol, vol_change

#: The vol modes, as the strip's control shows them and a link holds them (``?wvmode=relative``): a bound segmented
#: control writes its options' labels to the URL, so the labels are the link's values.
VOL_MODES = ("absolute", "relative", "1M-scaled")
#: The URL keys of the four shifts, and of the vol mode.
SHIFT_KEYS = ("ws", "wv", "wd", "wr")
#: Smallest vol a shifted leg keeps (decimal), as ``strategies.position_value`` floors a shocked vol.
VOL_FLOOR = 1e-8
#: Limits of the typed shifts: spot −50% to +100%; vol −50 to +100 vp (absolute) or −90% to +300% (relative);
#: rate −500 to +500 bp.
SPOT_LIMITS, VP_LIMITS, PCT_LIMITS, BP_LIMITS = (-50.0, 100.0), (-50.0, 100.0), (-90.0, 300.0), (-500, 500)


@dataclass(frozen=True)
class Shifts:
    """The what-if shifts: ``spot`` in percent, ``vol`` in vol points (absolute) or percent (relative, 1M-scaled),
    ``mode`` (one of :data:`VOL_MODES`), ``sessions`` (T+N) and ``rate`` in basis points."""

    spot: float = 0.0
    vol: float = 0.0
    mode: str = "absolute"
    sessions: int = 0
    rate: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in VOL_MODES:
            raise ValueError(f"mode must be one of {tuple(VOL_MODES)}, got {self.mode!r}")
        if self.sessions < 0:
            raise ValueError(f"sessions must be 0 or more, got {self.sessions!r}")

    @property
    def active(self) -> bool:
        """Whether any shift is non-zero (the mode alone moves nothing)."""
        return bool(self.spot or self.vol or self.sessions or self.rate)


def leg_vol(sigma: float, tau: float, shifts: Shifts, *, decay: float = DEFAULT_VOL_TENOR_DECAY) -> float:
    """A leg's vol under the vol shift, by its mode, floored at :data:`VOL_FLOOR`.

    * ``absolute``: σ + wv/100 (wv in vol points);
    * ``relative``: σ · (1 + wv/100);
    * ``1M-scaled``: σ · (1 + wv/100 · min(1, (τ / 1M)^−decay)), with 1M = 30 days: a shock to one-month vol, damped
      for a leg with longer to run and never amplified for a shorter one (RK-1's ``"1m"`` mode, as
      ``strategies.position_value`` applies it with ``vol_tenor_decay``).

    Parameters
    ----------
    sigma : float
        The leg's vol before the shift (decimal).
    tau : float
        Years the leg has left to run at the scenario date.
    shifts : Shifts
        The what-if shifts.
    decay : float, default scenarios.DEFAULT_VOL_TENOR_DECAY
        The tenor damping's exponent, 0.25.

    Returns
    -------
    float
        The shifted vol (decimal).

    Examples
    --------
    >>> round(leg_vol(0.1552, 0.15, Shifts(vol=3.0)), 6)
    0.1852
    >>> round(leg_vol(0.1552, 0.15, Shifts(vol=10.0, mode="relative")), 6)
    0.17072
    >>> round(leg_vol(0.20, 1.0, Shifts(vol=10.0, mode="1M-scaled")), 6)  # a one-year leg moves 0.53 of the shock
    0.210709
    """
    move = shifts.vol / 100.0
    if shifts.mode == "absolute":
        shifted = sigma + move
    elif shifts.mode == "relative":
        shifted = sigma * (1.0 + move)
    else:
        ratio = max(tau, 1e-8) / VOL_SHOCK_REFERENCE_TENOR
        shifted = sigma * (1.0 + move * min(1.0, ratio**-decay))
    return max(shifted, VOL_FLOOR)


@functools.lru_cache(maxsize=64)
def _sessions(start: date, n: int) -> tuple[date, ...]:
    days, day = [start], start
    while len(days) <= n:
        day += timedelta(days=1)
        if clock.is_trading_day(day):
            days.append(day)
    return tuple(days)


def session_dates(start: date, n: int) -> tuple[date, ...]:
    """The valuation date and the next ``n`` NYSE trading days (``clock.is_trading_day``): T+0 to T+N.

    Parameters
    ----------
    start : date
        The valuation date (T+0, whether or not it is a trading day).
    n : int
        Sessions after it.

    Returns
    -------
    tuple of date
        ``n + 1`` dates.

    Examples
    --------
    >>> [d.isoformat() for d in session_dates(date(2026, 11, 25), 2)]  # Thanksgiving, 26 November, is skipped
    ['2026-11-25', '2026-11-27', '2026-11-30']
    """
    return _sessions(start, int(n))


def sessions_to(start: date, end: date) -> int:
    """Trading sessions after ``start`` up to and including ``end``: the largest N of a scenario date on or before
    ``end`` (0 when ``end`` is not after ``start``).

    Examples
    --------
    >>> sessions_to(date(2026, 10, 23), date(2026, 11, 6))
    10
    """
    count, day = 0, start
    while day < end:
        day += timedelta(days=1)
        count += clock.is_trading_day(day)
    return count


def scenario_time(as_of: datetime, sessions: int) -> datetime:
    """The scenario's valuation time: the wall-clock time of ``as_of`` on session T+``sessions`` (New York).

    Examples
    --------
    >>> scenario_time(datetime(2026, 10, 23, 16, 0), 1)  # Friday's close to Monday's
    datetime.datetime(2026, 10, 26, 16, 0)
    """
    day = session_dates(as_of.date(), sessions)[-1]
    return datetime.combine(day, as_of.time(), as_of.tzinfo)


def apply(market: Market, shifts: Shifts, *, as_of: datetime) -> Market:
    """The market under the shifts: spot × (1 + ws/100), the rate + wr/10,000 with the dividend yield held, the vol
    by :func:`leg_vol` at the market's own expiry, and the time to expiry less the calendar time to the scenario
    date (minutes / 525,600, as ``clock.year_fraction``), floored at one minute.

    Parameters
    ----------
    market : strategies.Market
        Today's market (its ``expiry`` is the default leg's years to expiry from ``as_of``).
    shifts : Shifts
        The what-if shifts.
    as_of : datetime
        The valuation time, from which T+N counts.

    Returns
    -------
    strategies.Market
        The shifted market. Each leg with its own vol and expiry goes through :func:`leg_vol` with its own ``tau``.

    Examples
    --------
    >>> shifts = Shifts(spot=-2.0, vol=3.0, rate=25.0)
    >>> m = apply(Market(5800.0, 0.0475, 0.1552, 0.15), shifts, as_of=datetime(2026, 10, 23, 16))
    >>> round(m.spot, 2), round(m.volatility, 4), round(m.rate, 4)
    (5684.0, 0.1852, 0.05)
    """
    elapsed = clock.year_fraction(as_of, scenario_time(as_of, shifts.sessions)) if shifts.sessions else 0.0
    tau = max(market.expiry - elapsed, 1.0 / clock.MINUTES_PER_YEAR)
    return replace(
        market,
        spot=market.spot * (1.0 + shifts.spot / 100.0),
        rate=market.rate + shifts.rate / 1e4,
        volatility=leg_vol(market.volatility, tau, shifts),
        expiry=tau,
    )


def note(shifts: Shifts) -> str:
    """The muted line that Pricing, IV and Surface end their context line with while a shift is set (spec 2.0):
    ``Scenario on Strategy, Risk: spot −2.0% · vol +3.00 vp · T+5``; empty while every shift is zero.

    Examples
    --------
    >>> note(Shifts(spot=-2.0, vol=3.0, sessions=5))
    'Scenario on Strategy, Risk: spot −2.0% · vol +3.00 vp · T+5'
    >>> note(Shifts())
    ''
    """
    if not shifts.active:
        return ""
    parts = []
    if shifts.spot:
        parts.append(f"spot {signed(shifts.spot, 1)}%")
    if shifts.vol:
        unit = " vp" if shifts.mode == "absolute" else "%" + (" 1M-scaled" if shifts.mode == "1M-scaled" else "")
        parts.append(f"vol {signed(shifts.vol, 2 if shifts.mode == 'absolute' else 1)}{unit}")
    if shifts.sessions:
        parts.append(f"T+{shifts.sessions}")
    if shifts.rate:
        parts.append(f"rate {bp(shifts.rate / 1e4)}")
    return "Scenario on Strategy, Risk: " + " · ".join(parts)


# --- the strip ------------------------------------------------------------------------------------------------------


def _number(text: object, default: float) -> float:
    try:
        value = float(str(text))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _value(key: str, default: float, limits: tuple[float, float]) -> float:
    """A shift's value before its widget is drawn this run, for the widget's label: the session's, else the link's
    (which the bound widget reads on the session's first run; outside the limits it ignores it, so this does too)."""
    if key in st.session_state:
        return float(st.session_state[key])
    value = _number(st.query_params.get(key), default)
    return value if limits[0] <= value <= limits[1] else default


def current() -> Shifts:
    """The shifts in the session (else in the link), for a tab that shows the market but notes the scenario."""
    mode = st.session_state.get("wvmode") or st.query_params.get("wvmode") or "absolute"
    mode = mode if mode in VOL_MODES else "absolute"
    vol_limits = VP_LIMITS if mode == "absolute" else PCT_LIMITS
    return Shifts(
        spot=_value("ws", 0.0, SPOT_LIMITS),
        vol=_value("wv", 0.0, vol_limits),
        mode=mode,
        sessions=max(0, int(_value("wd", 0, (0, 10_000)))),
        rate=float(_value("wr", 0, BP_LIMITS)),
    )


def _reset() -> None:
    for key in SHIFT_KEYS:
        st.session_state[key] = 0 if key in ("wd", "wr") else 0.0


def _vol_label(shifts: Shifts, atm_vol: float, tenor: float) -> str:
    shifted = leg_vol(atm_vol, tenor, shifts)
    if shifts.mode == "absolute":
        return f"Vol {vol_change(shifts.vol / 100.0)}: ATMF {vol(atm_vol)} → {vol(shifted)}"
    scope = " at 1M" if shifts.mode == "1M-scaled" else ""
    return f"Vol {signed(shifts.vol, 1)}%{scope}: ATMF {vol(atm_vol)} → {vol(shifted)}"


def strip(
    *,
    spot: float,
    atm_vol: float,
    tenor: float,
    rate: float,
    as_of: datetime,
    horizon: datetime,
    dynamics: str = "sticky strike",
) -> Shifts:
    """Draw the what-if strip and return its shifts (spec 3.2 item 3). Keyed ``whatif``; one per page.

    Parameters
    ----------
    spot : float
        Today's spot, for the spot label's worked example.
    atm_vol : float
        The reference expiry's ATMF vol (decimal), for the vol label's example.
    tenor : float
        The reference expiry's years to expiry, for the 1M-scaled example's damping.
    rate : float
        The reference expiry's rate (decimal), for the rate label's example.
    as_of : datetime
        The valuation time (New York wall time): T+0.
    horizon : datetime
        The last date the date control reaches: the position's front settlement. Its days left show with each date.
    dynamics : str, default "sticky strike"
        The smile dynamics the scenario uses, named in the caption under the strip.

    Returns
    -------
    Shifts
        The shifts as set.
    """
    shifts = current()
    row = st.container(key="whatif", horizontal=True, gap="small", vertical_alignment="bottom")
    lead = row.empty()  # the SCENARIO badge leads the strip; filled once the widgets have their values
    ws = row.number_input(
        f"Spot {signed(shifts.spot, 1)}% → {level(spot * (1 + shifts.spot / 100))}",
        min_value=SPOT_LIMITS[0],
        max_value=SPOT_LIMITS[1],
        value=0.0,
        step=0.5,
        format="%.1f",
        key="ws",
        bind="query-params",
        persist_state="session",
        width=190,
        help="Spot move in percent of today's level, in steps of 0.5%. Every leg is revalued at the moved spot "
        f"under {dynamics} dynamics.",
    )
    vol_limits = VP_LIMITS if shifts.mode == "absolute" else PCT_LIMITS
    wv = row.number_input(
        _vol_label(shifts, atm_vol, tenor),
        min_value=vol_limits[0],
        max_value=vol_limits[1],
        value=0.0,
        step=0.5 if shifts.mode == "absolute" else 5.0,
        format="%.2f" if shifts.mode == "absolute" else "%.1f",
        key="wv",
        bind="query-params",
        persist_state="session",
        width=250,
        help="vp: vol points added to every leg's vol (+3.00 takes 15.52% to 18.52%). Relative: percent of each leg's "
        "vol (+10% takes 15.52% to 17.07%). 1M-scaled: a relative shock to one-month vol, damped for longer legs by "
        f"(τ / 1M)^{MINUS}{DEFAULT_VOL_TENOR_DECAY:g}, as a vol index moves more than a one-year vol.",
    )
    mode = row.segmented_control(
        "Vol shift",
        list(VOL_MODES),
        default="absolute",
        required=True,
        key="wvmode",
        bind="query-params",
        persist_state="session",
        help="How the vol shift reads: vol points (vp), relative to each leg's vol, or relative to one-month vol and "
        "damped with tenor (1M-scaled).",
    )
    last = max(1, sessions_to(as_of.date(), horizon.date()))
    dates = session_dates(as_of.date(), last)

    # A slider of whole sessions, not a select_slider of dated labels: a bound select_slider writes its options' labels
    # to the URL, and the link must hold the integer (?wd=5). The label names the date.
    shown = min(shifts.sessions, last)
    day = dates[shown]
    left = time_left(clock.year_fraction(datetime.combine(day, as_of.time()), horizon) * clock.MINUTES_PER_YEAR)
    wd = row.slider(
        f"Date {session_label(shown, day)}" + (f" ({left} left)" if left != "expired" else ""),
        min_value=0,
        max_value=last,
        value=0,
        step=1,
        format="T+%d",
        key="wd",
        bind="query-params",
        persist_state="session",
        width=300,
        help="Trading sessions after the valuation date (NYSE holidays skipped), at the valuation time of day. Time "
        "decay counts calendar days: Friday to Monday is three. The time left runs to the position's front "
        "settlement.",
    )
    wr = row.number_input(
        f"Rate {bp(shifts.rate / 1e4)}: {pct(rate)} → {pct(rate + shifts.rate / 1e4)}",
        min_value=BP_LIMITS[0],
        max_value=BP_LIMITS[1],
        value=0,
        step=5,
        key="wr",
        bind="query-params",
        persist_state="session",
        width=190,
        help="Basis points added to every expiry's rate, with spot and the dividend yield held, so each forward moves "
        "by e^(Δr·T): the convention of the position's rho.",
    )
    shifts = Shifts(float(ws), float(wv), str(mode or "absolute"), int(wd), float(wr))
    row.button(
        "Reset",
        key="whatif_reset",
        on_click=_reset,
        disabled=not shifts.active,
        help="Set the spot, vol, date and rate shifts back to zero (the vol mode stays).",
    )
    if shifts.active:
        lead.container(key="whatif_scenario", width="content").badge("SCENARIO", color="yellow")
    st.caption(
        f"{dynamics.capitalize()}: leg vols {'fixed by strike' if dynamics == 'sticky strike' else 'move with spot'}."
    )
    return shifts
