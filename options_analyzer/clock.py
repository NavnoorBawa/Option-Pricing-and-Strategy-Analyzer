"""Valuation clock: New York time, settlement times, the NYSE calendar and a variance clock.

Conventions
-----------
* Times are timezone-aware datetimes in America/New_York (:mod:`zoneinfo`; on Windows the ``tzdata`` package
  supplies the time-zone database). A naive datetime is read as New York wall-clock time.
* A time to expiry is counted in minutes of New York wall-clock time and divided by 525,600, the minutes in a
  365-day year: Cboe's convention for the VIX (every calendar day has 1,440 minutes, so a change to or from
  daylight saving time neither adds nor removes an hour). Times are counted exactly, in whole microseconds, and
  rounded to a float once.
* PM-settled options settle at the close, 16:00 ET (13:00 on NYSE early-close days); AM-settled options (the SPX,
  NDX and RUT monthlies, and VIX) settle at the open, 09:30 ET, on a value computed from that morning's opening
  prices. An expiry that falls on a weekend or an NYSE holiday settles on the trading day before it.
* NYSE holidays come from the exchange's rules (Good Friday from the date of Easter, and the observance rules for
  holidays on a weekend) for 2000 to 2100, plus the unscheduled closures since 2000. Early closes follow the rules
  NYSE applies today; NYSE did not always close early on those days before 2025.

:class:`VarianceClock` measures time by the variance it carries: weekend and holiday minutes count at a weight
relative to a trading day's, and scheduled events (earnings, central-bank meetings) add a jump in variance.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from fractions import Fraction
from functools import lru_cache
from typing import Literal
from zoneinfo import ZoneInfo

__all__ = [
    "AM_SETTLED_ROOTS",
    "MINUTES_PER_YEAR",
    "NEW_YORK",
    "UNSCHEDULED_CLOSURES",
    "VarianceClock",
    "easter",
    "is_trading_day",
    "market_close",
    "nyse_early_closes",
    "nyse_holidays",
    "previous_trading_day",
    "settlement_style",
    "settlement_time",
    "time_to_expiry",
    "to_new_york",
    "year_fraction",
]

NEW_YORK = ZoneInfo("America/New_York")
MINUTES_PER_YEAR = 525_600  # 365 x 1,440
MINUTES_PER_DAY = 1_440
_MICROSECOND = timedelta(microseconds=1)
_MICROSECONDS_PER_YEAR = MINUTES_PER_YEAR * 60_000_000
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)
#: Years for which the NYSE calendar is generated.
FIRST_YEAR, LAST_YEAR = 2000, 2100
#: Option roots that settle at the open (AM): the standard monthlies of SPX, NDX and RUT (their PM-settled
#: weeklies trade as SPXW, NDXP and RUTW), and every VIX option, monthly (VIX) or weekly (VIXW), which settles
#: on the special opening quotation of the VIX.
AM_SETTLED_ROOTS = frozenset({"SPX", "NDX", "RUT", "VIX", "VIXW"})
#: Days since 2000 on which NYSE closed for the whole day outside its holiday rules: the attacks of 11 September
#: 2001, the national days of mourning for Presidents Reagan, Ford, G. H. W. Bush and Carter, and Hurricane Sandy.
UNSCHEDULED_CLOSURES: tuple[date, ...] = (
    date(2001, 9, 11),
    date(2001, 9, 12),
    date(2001, 9, 13),
    date(2001, 9, 14),
    date(2004, 6, 11),
    date(2007, 1, 2),
    date(2012, 10, 29),
    date(2012, 10, 30),
    date(2018, 12, 5),
    date(2025, 1, 9),
)
#: Days of each kind in the 252-trading-day year that normalises :class:`VarianceClock` (252 + 104 + 9 = 365).
TRADING_DAYS_PER_YEAR = 252
WEEKEND_DAYS_PER_YEAR = 104
HOLIDAYS_PER_YEAR = 9

SettlementStyle = Literal["AM", "PM"]


# ---------------------------------------------------------------------------
# New York time
# ---------------------------------------------------------------------------


def to_new_york(moment: datetime) -> datetime:
    """``moment`` as a timezone-aware datetime in New York; a naive datetime is read as New York wall-clock time.

    Parameters
    ----------
    moment : datetime
        Any datetime (a :class:`pandas.Timestamp` is one).

    Returns
    -------
    datetime
        The same instant with ``tzinfo`` America/New_York.

    Examples
    --------
    >>> from datetime import timezone
    >>> to_new_york(datetime(2026, 11, 2, 15, 0, tzinfo=timezone.utc)).isoformat()
    '2026-11-02T10:00:00-05:00'
    >>> to_new_york(datetime(2026, 11, 2, 10, 0)).isoformat()
    '2026-11-02T10:00:00-05:00'
    """
    if not isinstance(moment, datetime):
        raise TypeError(f"expected a datetime, got {type(moment).__name__}")
    if moment.tzinfo is None or moment.utcoffset() is None:
        return moment.replace(tzinfo=NEW_YORK)
    return moment.astimezone(NEW_YORK)


def _wall_clock(moment: datetime) -> datetime:
    """New York wall-clock time without a time zone: differences between two of them count every day as 1,440 min."""
    return to_new_york(moment).replace(tzinfo=None)


def _microseconds(start: datetime, end: datetime) -> int:
    """Whole microseconds of wall-clock time from ``start`` to ``end`` (both wall-clock times), exactly."""
    return (end - start) // _MICROSECOND


def year_fraction(start: datetime, end: datetime) -> float:
    """Time from ``start`` to ``end`` in years: New York wall-clock minutes divided by 525,600.

    Cboe's convention for the VIX: every calendar day counts 1,440 minutes, also across a change to or from
    daylight saving time. Negative when ``end`` is before ``start``. The time is counted in whole microseconds and
    divided once, so the result is the correctly rounded value of the exact quotient.

    Parameters
    ----------
    start, end : datetime
        The two moments (naive ones are New York wall-clock time; see :func:`to_new_york`).

    Returns
    -------
    float
        Years (minutes / 525,600).

    Examples
    --------
    Six hours on an expiry day, from 10:00 to the 16:00 close:

    >>> year_fraction(datetime(2026, 11, 2, 10, 0), datetime(2026, 11, 2, 16, 0)) == 360 / 525_600
    True
    """
    return _microseconds(_wall_clock(start), _wall_clock(end)) / _MICROSECONDS_PER_YEAR


# ---------------------------------------------------------------------------
# NYSE calendar
# ---------------------------------------------------------------------------


def easter(year: int) -> date:
    """Easter Sunday of ``year`` in the Gregorian calendar (the anonymous Gregorian algorithm of Meeus, Jones and
    Butcher).

    Examples
    --------
    >>> easter(2026), easter(2027), easter(2028)
    (datetime.date(2026, 4, 5), datetime.date(2027, 3, 28), datetime.date(2028, 4, 16))
    """
    a, b, c = year % 19, year // 100, year % 100
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    weekday = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * weekday) // 451
    month, day = divmod(h + weekday - 7 * m + 114, 31)
    day += 1
    return date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The ``n``-th ``weekday`` (Monday = 0) of a month; ``n = -1`` for the last one."""
    if n > 0:
        first = date(year, month, 1)
        return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    last = date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: date) -> date | None:
    """NYSE's observance rule: a Saturday holiday closes the Friday before, a Sunday holiday the Monday after.

    New Year's Day on a Saturday is not observed (the Friday before ends the previous year's accounting period).
    """
    if day.weekday() == 5:
        return None if (day.month, day.day) == (1, 1) else day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _check_year(year: int) -> None:
    if not FIRST_YEAR <= year <= LAST_YEAR:
        raise ValueError(f"the NYSE calendar covers {FIRST_YEAR} to {LAST_YEAR}, got the year {year}")


@lru_cache(maxsize=256)
def _rule_holidays(year: int) -> tuple[date, ...]:
    _check_year(year)
    days = [
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),  # Martin Luther King, Jr. Day: third Monday in January
        _nth_weekday(year, 2, 0, 3),  # Washington's Birthday: third Monday in February
        easter(year) - timedelta(days=2),  # Good Friday
        _nth_weekday(year, 5, 0, -1),  # Memorial Day: last Monday in May
        _observed(date(year, 6, 19)) if year >= 2022 else None,  # Juneteenth National Independence Day
        _observed(date(year, 7, 4)),  # Independence Day
        _nth_weekday(year, 9, 0, 1),  # Labor Day: first Monday in September
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving Day: fourth Thursday in November
        _observed(date(year, 12, 25)),  # Christmas Day
    ]
    return tuple(sorted(day for day in days if day is not None))


def nyse_holidays(year: int, *, unscheduled: bool = True) -> tuple[date, ...]:
    """Weekdays of ``year`` on which NYSE is closed all day, in date order.

    The ten holidays of NYSE's rules (Martin Luther King, Jr. Day since 1998 and Juneteenth since 2022 included),
    with a holiday on a Saturday observed the Friday before and one on a Sunday the Monday after, except that New
    Year's Day on a Saturday is not observed. Good Friday is two days before Easter (:func:`easter`).

    Parameters
    ----------
    year : int
        2000 to 2100.
    unscheduled : bool, default True
        Include the days NYSE closed outside its rules (:data:`UNSCHEDULED_CLOSURES`).

    Returns
    -------
    tuple of date

    Raises
    ------
    ValueError
        If ``year`` is outside 2000 to 2100.

    Examples
    --------
    New Year's Day 2028 is a Saturday and is not observed; Independence Day is a Tuesday:

    >>> [str(day) for day in nyse_holidays(2028)][:4]
    ['2028-01-17', '2028-02-21', '2028-04-14', '2028-05-29']
    >>> date(2025, 1, 9) in nyse_holidays(2025), date(2025, 1, 9) in nyse_holidays(2025, unscheduled=False)
    (True, False)
    """
    holidays = _rule_holidays(int(year))
    if not unscheduled:
        return holidays
    extra = tuple(day for day in UNSCHEDULED_CLOSURES if day.year == year)
    return tuple(sorted(holidays + extra)) if extra else holidays


@lru_cache(maxsize=256)
def nyse_early_closes(year: int) -> tuple[date, ...]:
    """Days of ``year`` on which NYSE closes early, at 13:00 ET, under its current rules, in date order.

    The day before Independence Day (3 July) and Christmas Eve when they are trading days, and the day after
    Thanksgiving.

    Parameters
    ----------
    year : int
        2000 to 2100.

    Returns
    -------
    tuple of date

    Examples
    --------
    >>> [str(day) for day in nyse_early_closes(2026)]
    ['2026-11-27', '2026-12-24']
    """
    _check_year(year)
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    candidates = (date(year, 7, 3), thanksgiving + timedelta(days=1), date(year, 12, 24))
    return tuple(day for day in candidates if is_trading_day(day))


@lru_cache(maxsize=256)
def _closed(year: int) -> frozenset[date]:
    return frozenset(nyse_holidays(year))


def is_trading_day(day: date) -> bool:
    """Whether NYSE is open on ``day``: a weekday that is not a holiday or an unscheduled closure.

    Examples
    --------
    >>> is_trading_day(date(2026, 4, 3)), is_trading_day(date(2026, 4, 2))  # Good Friday, the Thursday before
    (False, True)
    """
    if isinstance(day, datetime):
        day = day.date()
    return day.weekday() < 5 and day not in _closed(day.year)


def previous_trading_day(day: date) -> date:
    """``day`` itself if NYSE is open then, else the last trading day before it.

    Examples
    --------
    >>> previous_trading_day(date(2026, 4, 3))  # Good Friday
    datetime.date(2026, 4, 2)
    """
    if isinstance(day, datetime):
        day = day.date()
    while not is_trading_day(day):
        day -= timedelta(days=1)
    return day


def market_close(day: date) -> datetime:
    """NYSE's close on trading day ``day``: 16:00 ET, or 13:00 ET on an early-close day.

    Raises
    ------
    ValueError
        If NYSE is closed on ``day``.

    Examples
    --------
    >>> market_close(date(2026, 11, 27)).isoformat()  # the day after Thanksgiving
    '2026-11-27T13:00:00-05:00'
    """
    if isinstance(day, datetime):
        day = day.date()
    if not is_trading_day(day):
        raise ValueError(f"NYSE is closed on {day:%A %Y-%m-%d}")
    close = EARLY_CLOSE if day in nyse_early_closes(day.year) else MARKET_CLOSE
    return datetime.combine(day, close, tzinfo=NEW_YORK)


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------


def settlement_style(root: str) -> SettlementStyle:
    """``"AM"`` for the option roots in :data:`AM_SETTLED_ROOTS`, else ``"PM"`` (weeklies, equities and ETFs).

    Examples
    --------
    >>> settlement_style("SPX"), settlement_style("SPXW"), settlement_style("spy")
    ('AM', 'PM', 'PM')
    """
    return "AM" if str(root).strip().upper() in AM_SETTLED_ROOTS else "PM"


def settlement_time(expiry: date, style: SettlementStyle = "PM") -> datetime:
    """When an option expiring on ``expiry`` settles, in New York time.

    PM settlement is at the close (16:00 ET, 13:00 ET on an early-close day, see :func:`market_close`); AM
    settlement is at the open, 09:30 ET, when the opening prices that fix the settlement value print. An expiry on
    a weekend or an NYSE holiday settles on the trading day before it (:func:`previous_trading_day`), as the
    exchanges move an expiration that falls on a holiday.

    Parameters
    ----------
    expiry : date
        The expiration date (a datetime's date is used).
    style : {"PM", "AM"}, default "PM"
        Settlement style; :func:`settlement_style` gives it for an option root.

    Returns
    -------
    datetime
        Timezone-aware, America/New_York.

    Examples
    --------
    AM and PM settlement of the 20 November 2026 monthly expiry are 390 minutes apart:

    >>> settlement_time(date(2026, 11, 20), "AM").isoformat()
    '2026-11-20T09:30:00-05:00'
    >>> settlement_time(date(2026, 11, 20), "PM").isoformat()
    '2026-11-20T16:00:00-05:00'
    """
    kind = str(style).strip().upper()
    if kind not in ("AM", "PM"):
        raise ValueError(f"settlement style must be 'AM' or 'PM', got {style!r}")
    day = previous_trading_day(expiry)
    if kind == "AM":
        return datetime.combine(day, MARKET_OPEN, tzinfo=NEW_YORK)
    return market_close(day)


def time_to_expiry(valuation: datetime, expiry: date, *, style: SettlementStyle = "PM") -> float:
    """Years from ``valuation`` to the settlement of an option expiring on ``expiry`` (minutes / 525,600).

    Parameters
    ----------
    valuation : datetime
        The valuation time (a naive datetime is New York wall-clock time).
    expiry : date
        The expiration date.
    style : {"PM", "AM"}, default "PM"
        Settlement style (see :func:`settlement_time`).

    Returns
    -------
    float
        Years; zero or negative once the option has settled.

    Examples
    --------
    At 10:00 ET on the expiry day, a PM-settled option has 360 minutes left:

    >>> time_to_expiry(datetime(2026, 11, 2, 10, 0), date(2026, 11, 2)) == 360 / 525_600
    True
    """
    return year_fraction(valuation, settlement_time(expiry, style))


# ---------------------------------------------------------------------------
# Variance clock
# ---------------------------------------------------------------------------


def _event_time(when: date | datetime) -> datetime:
    """An event's wall-clock time: a datetime as given, a bare date at that day's open (09:30 ET)."""
    if isinstance(when, datetime):
        return _wall_clock(when)
    if isinstance(when, date):
        return datetime.combine(when, MARKET_OPEN)
    raise TypeError(f"an event time must be a date or a datetime, got {type(when).__name__}")


def _non_negative(name: str, value: float) -> float:
    value = float(value)
    if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return value


@dataclass(frozen=True)
class VarianceClock:
    """Time measured by the variance it carries: quieter weekends and holidays, and jumps at scheduled events.

    Every New York calendar day carries a weight spread evenly over its 1,440 minutes: 1 for a trading day,
    ``weekend_weight`` for a Saturday or Sunday and ``holiday_weight`` for an NYSE holiday. The variance time
    between two moments is the weighted count of minutes between them divided by 525,600 and by the mean weight
    of a 252-trading-day year (252 trading days, 104 weekend days and 9 holidays), so that such a year counts as
    one year whatever the weights: with both weights 1 the variance time is calendar time, and with both weights 0
    it is the classic trading days / 252.

    Each event ``(when, move)`` adds ``(pi/2) move**2`` of total variance at ``when`` (a datetime, or a date for
    that day's open, 09:30 ET): ``move`` is the expected absolute log-return the event causes, and a normal move
    with that mean absolute size has variance ``(pi/2) move**2``, since ``E|X| = s sqrt(2/pi)`` for
    ``X ~ N(0, s**2)``. Fat-tailed moves have more variance for the same mean absolute size, so this is an
    approximation.

    Parameters
    ----------
    weekend_weight : float, default 1.0
        Variance of a weekend day relative to a trading day (non-negative).
    holiday_weight : float, default 1.0
        Variance of an NYSE holiday relative to a trading day (non-negative).
    events : sequence of (date or datetime, float), default ()
        Scheduled events and their expected absolute moves (decimal: 0.05 is 5%).

    Examples
    --------
    From a Friday close to the Monday close with weekends at a quarter of a trading day: one trading day and two
    quarter days make 1.5 weighted days, against three calendar days. Divided by the mean weight of a calendar day,
    287/365, that is 1.9077 days of variance time:

    >>> clock = VarianceClock(weekend_weight=0.25)
    >>> friday, monday = datetime(2026, 10, 23, 16, 0), datetime(2026, 10, 26, 16, 0)
    >>> round(clock.variance_time(friday, monday) * 365, 4)
    1.9077
    >>> round(clock.effective_vol(0.20, friday, monday), 4)  # 20% in variance time is 15.95% in calendar time
    0.1595
    """

    weekend_weight: float = 1.0
    holiday_weight: float = 1.0
    events: Sequence[tuple[date | datetime, float]] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "weekend_weight", _non_negative("weekend_weight", self.weekend_weight))
        object.__setattr__(self, "holiday_weight", _non_negative("holiday_weight", self.holiday_weight))
        events = []
        for event in self.events:
            try:
                when, move = event
            except (TypeError, ValueError):
                raise ValueError(
                    f"each event must be a (date or datetime, expected absolute move) pair, got {event!r}"
                ) from None
            events.append((_event_time(when), _non_negative("an event's expected absolute move", move)))
        object.__setattr__(self, "events", tuple(sorted(events)))

    def _exact_mean_weight(self) -> Fraction:
        """:attr:`mean_weight` as an exact fraction of the two weights (which are binary fractions)."""
        weekend = WEEKEND_DAYS_PER_YEAR * Fraction(self.weekend_weight)
        holiday = HOLIDAYS_PER_YEAR * Fraction(self.holiday_weight)
        return (TRADING_DAYS_PER_YEAR + weekend + holiday) / 365

    @property
    def mean_weight(self) -> float:
        """Mean weight of a calendar day in a year of 252 trading days, 104 weekend days and 9 holidays."""
        return float(self._exact_mean_weight())

    def variance_time(self, start: datetime, end: datetime) -> float:
        """Variance time from ``start`` to ``end`` in years (negative when ``end`` is before ``start``).

        The minutes of each kind of day between the two moments (trading days, weekend days, holidays), each times
        its weight, divided by 525,600 and by :attr:`mean_weight`; events are not included (see
        :meth:`total_variance`).

        The minutes are counted exactly, in whole microseconds, and the weighted sum is formed and divided in exact
        rational arithmetic, so the result is the correctly rounded value of an exactly additive quantity. Over
        consecutive intervals, ``variance_time(a, c)`` therefore equals ``variance_time(a, b) + variance_time(b, c)``
        to within four units of 2**-53 (about 4.4e-16) times the largest of the three in size: a relative, not an
        absolute, tolerance, as no float sum of numbers of a few years can be exact to 1e-15 absolute.

        Parameters
        ----------
        start, end : datetime
            The two moments (naive ones are New York wall-clock time; see :func:`to_new_york`).

        Returns
        -------
        float
            Years of variance time.

        Raises
        ------
        ValueError
            If the interval reaches outside the years 2000 to 2100 of the NYSE calendar.

        Examples
        --------
        Thanksgiving (26 November 2026) at half a trading day's variance: the 48 hours from the Wednesday close to
        the Friday close are a third of Wednesday, all of Thursday at 0.5 and two thirds of Friday, 1.5 days of
        variance, which :attr:`mean_weight` (360.5 / 365) scales so that a year still counts as one:

        >>> clock = VarianceClock(holiday_weight=0.5)
        >>> tau = clock.variance_time(datetime(2026, 11, 25, 16, 0), datetime(2026, 11, 27, 16, 0))
        >>> tau == 1.5 / 365 / clock.mean_weight, round(tau * 365, 6)
        (True, 1.518724)
        """
        a, b = _wall_clock(start), _wall_clock(end)
        if b < a:
            return -self.variance_time(b, a)
        weekend = holiday = 0  # microseconds
        day = a.date()
        while day <= b.date():
            midnight = datetime.combine(day, time())
            overlap = _microseconds(max(a, midnight), min(b, midnight + timedelta(days=1)))
            if overlap > 0:
                if day.weekday() >= 5:
                    weekend += overlap
                elif not is_trading_day(day):
                    holiday += overlap
            day += timedelta(days=1)
        trading = _microseconds(a, b) - weekend - holiday
        weighted = trading + Fraction(self.weekend_weight) * weekend + Fraction(self.holiday_weight) * holiday
        return float(weighted / (_MICROSECONDS_PER_YEAR * self._exact_mean_weight()))

    def event_variance(self, start: datetime, end: datetime) -> float:
        """Total variance of the events after ``start`` and up to ``end``: the sum of ``(pi/2) move**2``.

        Negative when ``end`` is before ``start`` (the events after ``end`` and up to ``start``), so that it adds
        up over consecutive intervals like :meth:`variance_time`.

        Parameters
        ----------
        start, end : datetime
            The two moments (naive ones are New York wall-clock time). An event exactly at ``end`` is included, one
            exactly at ``start`` is not.

        Returns
        -------
        float
            Total variance (decimal squared: 0.01 is a standard deviation of 10%).

        Examples
        --------
        An earnings release with an expected absolute move of 5%, before the open on 29 October 2026 (an event given
        as a date is at that day's 09:30 ET open):

        >>> clock = VarianceClock(events=[(date(2026, 10, 29), 0.05)])
        >>> round(clock.event_variance(datetime(2026, 10, 26, 10, 0), datetime(2026, 10, 30, 16, 0)), 6)
        0.003927
        >>> clock.event_variance(datetime(2026, 10, 29, 9, 30), datetime(2026, 10, 30, 16, 0))  # at the start: out
        0.0
        """
        a, b = _wall_clock(start), _wall_clock(end)
        sign = 1.0
        if b < a:
            a, b, sign = b, a, -1.0
        return sign * math.fsum(0.5 * math.pi * move * move for when, move in self.events if a < when <= b)

    def total_variance(self, sigma: float, start: datetime, end: datetime) -> float:
        """``sigma**2 * variance_time(start, end) + event_variance(start, end)``.

        Parameters
        ----------
        sigma : float
            Volatility per year of variance time (decimal, non-negative).
        start, end : datetime
            The two moments (naive ones are New York wall-clock time).

        Returns
        -------
        float
            Total variance of the log-return from ``start`` to ``end`` (decimal squared).

        Examples
        --------
        Twenty percent a year over the 30 calendar days to the 20 November 2026 close (22 trading days and 8
        weekend days, at a quarter of a trading day each: 24 / 287 years of variance time), plus an event with an
        expected absolute move of 4% on 4 November:

        >>> clock = VarianceClock(weekend_weight=0.25, events=[(date(2026, 11, 4), 0.04)])
        >>> start, end = datetime(2026, 10, 21, 16, 0), datetime(2026, 11, 20, 16, 0)
        >>> diffusion = 0.2**2 * clock.variance_time(start, end)
        >>> total = clock.total_variance(0.2, start, end)
        >>> round(diffusion, 6), round(total - diffusion, 6)  # (pi/2) 0.04**2 = 0.002513
        (0.003345, 0.002513)
        """
        sigma = _non_negative("sigma", sigma)
        return sigma * sigma * self.variance_time(start, end) + self.event_variance(start, end)

    def effective_vol(self, sigma: float, start: datetime, end: datetime) -> float:
        """The calendar-time volatility with the same total variance: ``sqrt(total_variance / year_fraction)``.

        Use it with a time to expiry from :func:`year_fraction` (or :func:`time_to_expiry`) in Black-Scholes.

        Parameters
        ----------
        sigma : float
            Volatility per year of variance time (decimal, non-negative).
        start, end : datetime
            The two moments, ``end`` after ``start``.

        Returns
        -------
        float
            Volatility per calendar year (decimal).

        Raises
        ------
        ValueError
            If ``end`` is not after ``start``.

        Examples
        --------
        With unit weights and no events the clock is calendar time, so the effective volatility is ``sigma``:

        >>> VarianceClock().effective_vol(0.2, datetime(2026, 10, 26, 10, 0), datetime(2026, 11, 20, 16, 0))
        0.2
        """
        years = year_fraction(start, end)
        if not years > 0:
            raise ValueError("the effective volatility needs end after start")
        return math.sqrt(self.total_variance(sigma, start, end) / years)
