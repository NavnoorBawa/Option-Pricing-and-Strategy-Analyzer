"""clock: New York valuation times, settlement times, the NYSE calendar and the variance clock (DT-1)."""

from __future__ import annotations

import csv
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from dateutil.easter import easter as dateutil_easter

from options_analyzer import black_scholes as bs
from options_analyzer import clock

FIXTURES = Path(__file__).parent / "fixtures"
MINUTE = 1 / 525_600


def _published_calendar() -> dict[str, dict[int, list[date]]]:
    """The NYSE holiday and early-close dates for 2026-2028 transcribed from NYSE's published calendar."""
    lines = (FIXTURES / "nyse_holidays_2026_2028.csv").read_text(encoding="utf-8").splitlines()
    rows = list(csv.DictReader(line for line in lines if not line.startswith("#")))
    calendar: dict[str, dict[int, list[date]]] = {"holiday": {}, "early_close": {}}
    for row in rows:
        day = date.fromisoformat(row["date"])
        calendar[row["kind"]].setdefault(day.year, []).append(day)
    return calendar


# --- Times to expiry -------------------------------------------------------------------------------------------


def test_ten_am_on_a_pm_expiry_day_is_exactly_360_minutes():
    """Acceptance (1): T = 360 / 525,600 exactly, for naive New York time and for the same instant in UTC."""
    expiry = date(2026, 11, 2)
    assert clock.time_to_expiry(datetime(2026, 11, 2, 10, 0), expiry) == 360 / 525_600
    utc = datetime(2026, 11, 2, 15, 0, tzinfo=timezone.utc)  # 10:00 EST
    assert clock.time_to_expiry(utc, expiry) == 360 / 525_600
    summer = datetime(2026, 7, 17, 14, 0, tzinfo=timezone.utc)  # 10:00 EDT
    assert clock.time_to_expiry(summer, date(2026, 7, 17)) == 360 / 525_600


@pytest.mark.parametrize("expiry", [date(2026, 11, 20), date(2027, 3, 19), date(2028, 6, 16)])
def test_am_and_pm_settlement_are_390_minutes_apart(expiry):
    """Acceptance (2): the open (09:30) and the close (16:00) of a monthly expiry day."""
    am, pm = clock.settlement_time(expiry, "AM"), clock.settlement_time(expiry, "PM")
    assert (am.hour, am.minute, pm.hour, pm.minute) == (9, 30, 16, 0)
    assert clock.year_fraction(am, pm) * 525_600 == 390
    valuation = datetime(2026, 10, 26, 12, 0)
    gap = clock.time_to_expiry(valuation, expiry, style="PM") - clock.time_to_expiry(valuation, expiry, style="AM")
    assert gap == pytest.approx(390 * MINUTE, abs=1e-15)


def test_settlement_roots_and_holiday_expiries():
    assert [clock.settlement_style(root) for root in ("SPX", "SPXW", "NDX", "NDXP", "RUT", "RUTW", "VIX", "VIXW")] == [
        "AM",
        "PM",
        "AM",
        "PM",
        "AM",
        "PM",
        "AM",
        "AM",
    ]
    assert clock.settlement_style(" spy ") == "PM"
    # The April 2025 monthly fell on Good Friday (18 April): it settled at Thursday's open.
    assert clock.settlement_time(date(2025, 4, 18), "AM") == datetime(2025, 4, 17, 9, 30, tzinfo=clock.NEW_YORK)
    # A PM expiry on the day after Thanksgiving settles at the 13:00 early close.
    assert clock.settlement_time(date(2026, 11, 27)) == datetime(2026, 11, 27, 13, 0, tzinfo=clock.NEW_YORK)
    with pytest.raises(ValueError, match="'AM' or 'PM'"):
        clock.settlement_time(date(2026, 11, 20), "noon")  # type: ignore[arg-type]


def test_minutes_follow_the_new_york_wall_clock_across_daylight_saving_changes():
    """Cboe's VIX convention: every calendar day is 1,440 minutes, although 1 November 2026 lasts 25 hours."""
    start, end = datetime(2026, 10, 31, 16, 0), datetime(2026, 11, 2, 16, 0)
    assert clock.year_fraction(start, end) == 2_880 / 525_600
    elapsed = clock.to_new_york(end) - clock.to_new_york(start)  # the same tzinfo: wall-clock arithmetic too
    utc_elapsed = clock.to_new_york(end).astimezone(timezone.utc) - clock.to_new_york(start).astimezone(timezone.utc)
    assert elapsed == timedelta(days=2) and utc_elapsed == timedelta(days=2, hours=1)
    assert clock.year_fraction(end, start) == -2_880 / 525_600


def test_to_new_york_reads_naive_times_as_new_york_and_converts_aware_ones():
    naive = clock.to_new_york(datetime(2026, 3, 9, 9, 30))
    assert naive.utcoffset() == timedelta(hours=-4)  # daylight saving time began on 8 March 2026
    tokyo = datetime(2026, 3, 9, 22, 30, tzinfo=timezone(timedelta(hours=9)))
    assert clock.to_new_york(tokyo) == naive
    with pytest.raises(TypeError, match="datetime"):
        clock.to_new_york(date(2026, 3, 9))  # type: ignore[arg-type]


def test_black_scholes_at_the_clock_time_equals_the_hand_formula():
    """Acceptance (7): a call 30 days and 6 hours from settlement, priced by hand with T = minutes / 525,600."""
    valuation, expiry = datetime(2026, 10, 21, 10, 0), date(2026, 11, 20)
    T = clock.time_to_expiry(valuation, expiry)
    assert T == (30 * 1_440 + 360) / 525_600
    S, K, r, q, sigma = 6_000.0, 6_100.0, 0.042, 0.013, 0.17

    def cdf(x: float) -> float:
        return 0.5 * math.erfc(-x / math.sqrt(2.0))

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    hand = S * math.exp(-q * T) * cdf(d1) - K * math.exp(-r * T) * cdf(d2)
    assert bs.price(S, K, T, r, sigma, q, "call") == pytest.approx(hand, rel=1e-13)


# --- NYSE calendar ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("year", [2026, 2027, 2028])
def test_holidays_and_early_closes_equal_the_published_nyse_calendar(year):
    """Acceptance (3): the rule-based calendar against NYSE's published list (tests/fixtures)."""
    published = _published_calendar()
    assert list(clock.nyse_holidays(year)) == published["holiday"][year]
    assert list(clock.nyse_early_closes(year)) == published["early_close"][year]
    for day in published["early_close"][year]:
        assert clock.market_close(day).hour == 13
    for day in published["holiday"][year]:
        assert not clock.is_trading_day(day)


def test_easter_matches_dateutil_for_every_gregorian_year_it_covers():
    """dateutil implements Easter independently (the Gregorian method of Oudin, 1940)."""
    mismatches = [year for year in range(1583, 4100) if clock.easter(year) != dateutil_easter(year)]
    assert mismatches == []


@pytest.mark.oracle
def test_holidays_2000_to_2100_match_quantlib_nyse_calendar():
    """QuantLib's UnitedStates(NYSE) calendar, with its own list of unscheduled closures, year by year."""
    ql = pytest.importorskip("QuantLib")
    calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
    for year in range(clock.FIRST_YEAR, clock.LAST_YEAR + 1):
        theirs = calendar.holidayList(ql.Date(1, 1, year), ql.Date(31, 12, year), False)
        assert list(clock.nyse_holidays(year)) == [date(d.year(), d.month(), d.dayOfMonth()) for d in theirs], year


def test_holiday_rules():
    """Observance: Saturday to Friday (not New Year's Day), Sunday to Monday; Juneteenth only from 2022."""
    assert date(2021, 12, 31) not in clock.nyse_holidays(2021)  # New Year's Day 2022 was a Saturday
    assert date(2021, 12, 24) in clock.nyse_holidays(2021)  # Christmas on a Saturday
    assert date(2022, 6, 20) in clock.nyse_holidays(2022)  # Juneteenth on a Sunday
    assert not any(day.month == 6 and day.day >= 18 for day in clock.nyse_holidays(2021))
    assert date(2001, 9, 11) in clock.nyse_holidays(2001)
    assert date(2001, 9, 11) not in clock.nyse_holidays(2001, unscheduled=False)
    for year in range(clock.FIRST_YEAR, clock.LAST_YEAR + 1):
        holidays = clock.nyse_holidays(year, unscheduled=False)
        assert all(day.weekday() < 5 and day.year == year for day in holidays)
        assert len(holidays) in ((8, 9, 10) if year < 2022 else (9, 10))
    assert clock.previous_trading_day(date(2026, 11, 22)) == date(2026, 11, 20)  # a Sunday
    with pytest.raises(ValueError, match="closed"):
        clock.market_close(date(2026, 12, 25))
    with pytest.raises(ValueError, match="2000 to 2100"):
        clock.nyse_holidays(1999)
    with pytest.raises(ValueError, match="2000 to 2100"):
        clock.is_trading_day(date(2101, 3, 1))


# --- Variance clock --------------------------------------------------------------------------------------------

#: Moments from 2025 to 2028 on a one-minute grid, as offsets in minutes from the start of 2025.
_BASE = datetime(2025, 1, 1)
_SPAN = 4 * 525_600


def _moment(minutes: int) -> datetime:
    return _BASE + timedelta(minutes=minutes)


def test_with_unit_weights_the_variance_clock_is_calendar_time():
    """Acceptance (4): exactly equal, for intervals across weekends, holidays and daylight saving changes."""
    hypothesis = pytest.importorskip("hypothesis")
    st = hypothesis.strategies
    unit = clock.VarianceClock()

    @hypothesis.given(st.integers(0, _SPAN), st.integers(0, _SPAN))
    def check(a, b):
        start, end = _moment(a), _moment(b)
        assert unit.variance_time(start, end) == clock.year_fraction(start, end)
        assert unit.total_variance(0.2, start, end) == 0.2**2 * clock.year_fraction(start, end)

    check()


def test_variance_time_adds_up_over_consecutive_intervals():
    """Acceptance (5): variance_time(t0, t2) = variance_time(t0, t1) + variance_time(t1, t2) to 1e-15."""
    hypothesis = pytest.importorskip("hypothesis")
    st = hypothesis.strategies

    @hypothesis.given(
        st.lists(st.integers(0, _SPAN), min_size=3, max_size=3),
        st.floats(0.0, 2.0),
        st.floats(0.0, 2.0),
        st.integers(0, 59),
    )
    def check(minutes, weekend, holiday, seconds):
        t0, t1, t2 = (_moment(m) + timedelta(seconds=seconds) for m in minutes)  # in any order
        variance_clock = clock.VarianceClock(weekend, holiday, [(date(2026, 4, 30), 0.04)])
        whole = variance_clock.variance_time(t0, t2)
        assert whole == pytest.approx(
            variance_clock.variance_time(t0, t1) + variance_clock.variance_time(t1, t2), abs=1e-15
        )
        events = variance_clock.event_variance(t0, t1) + variance_clock.event_variance(t1, t2)
        assert variance_clock.event_variance(t0, t2) == pytest.approx(events, abs=1e-15)

    check()


def test_total_variance_is_diffusion_plus_events():
    """Acceptance (6): sigma^2 tau_trading + sum of (pi/2) m^2 over the events in (t0, t1], to 1e-15."""
    events = [
        (date(2026, 10, 29), 0.06),  # earnings before the open: 09:30 ET
        (datetime(2026, 11, 4, 14, 0), 0.012),  # a central-bank decision
        (datetime(2026, 11, 20, 16, 5), 0.05),  # after the expiry settles
    ]
    variance_clock = clock.VarianceClock(weekend_weight=0.3, holiday_weight=0.5, events=events)
    t0 = datetime(2026, 10, 26, 10, 0)
    t1 = clock.settlement_time(date(2026, 11, 20))
    sigma = 0.18
    tau = variance_clock.variance_time(t0, t1)
    expected = sigma**2 * tau + math.pi / 2 * (0.06**2 + 0.012**2)
    assert variance_clock.total_variance(sigma, t0, t1) == pytest.approx(expected, abs=1e-15)
    effective = variance_clock.effective_vol(sigma, t0, t1)
    assert effective**2 * clock.year_fraction(t0, t1) == pytest.approx(expected, abs=1e-15)
    # An event exactly at the end of an interval belongs to it, not to the next one.
    at_open = datetime(2026, 10, 29, 9, 30)
    assert variance_clock.event_variance(t0, at_open) == pytest.approx(math.pi / 2 * 0.06**2, abs=1e-18)
    assert variance_clock.event_variance(at_open, datetime(2026, 10, 30)) == 0.0


def test_zero_weights_count_trading_days_over_252():
    """With weekends and holidays at weight 0 the clock is the classic trading days / 252."""
    zero = clock.VarianceClock(weekend_weight=0.0, holiday_weight=0.0)
    # Close of Friday 20 November 2026 to the close of Friday 4 December: ten weekdays, one of them Thanksgiving
    # (a full day of weight 0) and one the early close after it (a full day of weight 1: the weight is per day).
    tau = zero.variance_time(datetime(2026, 11, 20, 16, 0), datetime(2026, 12, 4, 16, 0))
    assert tau == pytest.approx(9 / 252, abs=1e-15)
    assert zero.mean_weight == 252 / 365
    weekend = zero.variance_time(datetime(2026, 10, 24), datetime(2026, 10, 26))
    assert weekend == 0.0


def test_variance_clock_checks_its_inputs():
    with pytest.raises(ValueError, match="weekend_weight"):
        clock.VarianceClock(weekend_weight=-0.1)
    with pytest.raises(ValueError, match="holiday_weight"):
        clock.VarianceClock(holiday_weight=float("nan"))
    with pytest.raises(ValueError, match="expected absolute move"):
        clock.VarianceClock(events=[(date(2026, 11, 2), -0.01)])
    with pytest.raises(ValueError, match="pair"):
        clock.VarianceClock(events=[date(2026, 11, 2)])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="date or a datetime"):
        clock.VarianceClock(events=[("2026-11-02", 0.01)])  # type: ignore[list-item]
    variance_clock = clock.VarianceClock()
    with pytest.raises(ValueError, match="sigma"):
        variance_clock.total_variance(-0.2, datetime(2026, 11, 2), datetime(2026, 11, 3))
    with pytest.raises(ValueError, match="end after start"):
        variance_clock.effective_vol(0.2, datetime(2026, 11, 3), datetime(2026, 11, 2))
    with pytest.raises(ValueError, match="2000 to 2100"):
        variance_clock.variance_time(datetime(1999, 12, 30), datetime(2000, 1, 3))
    assert hash(clock.VarianceClock(events=[(date(2026, 11, 2), 0.01)])) is not None  # frozen and hashable
