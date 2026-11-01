"""A chain read quote by quote, and the desk's measures of a fitted surface (roadmap items SF-6 and SF-11).

:func:`fit_chain` turns an option chain in the canonical schema (``market_data.schema``: one row per contract with
its bid and ask) into a fitted surface and says what happened to every quote:

1. each expiry's forward and discount factor from put-call parity on its two-sided quotes
   (:func:`.forwards.implied_forwards`, as :func:`.quotes.parse_quotes` reads them);
2. the Black-76 implied vols of every bid, ask and mid at that forward, in one batch (Let's Be Rational);
3. the screen (:class:`QuoteFilters`, with defaults modelled on LSEG's volatility-surface service): each quote is
   ``used`` or excluded for the first of :data:`REASONS` that applies, so the counts always reconcile,
   quoted = used + the exclusions by reason (:meth:`ChainFit.counts`);
4. the spread-weighted SVI surface fitted to the used quotes' mid vols (:func:`.surface.fit_market_surface`).

On the default synthetic chain (5,800, 23 October 2026 16:00 ET) this gives the surface that parsing the whole chain
with :func:`parse_quotes` gave before, to the last digit of every used vol: the default screen drops exactly the
in-the-money side and the zero bids that the parser dropped.

The measures are the ones a desk reads off a surface, all in forward log-moneyness ``y = ln K/F`` and total implied
variance ``w = σ²T``, interpolated linearly in ``T`` at fixed ``y`` between the fitted expiries (no extrapolation in
time): the ATMF vol at a constant maturity (:func:`atmf_vol`), the vol at a delta (:func:`delta_vol`; spot delta
e^{−qT}N(d₁), the app's convention), 25-delta risk reversals and butterflies, forward vols, a vol grid by delta
(:func:`delta_grid`), an expiry table (:func:`expiry_table`), Mixon's normalised skew, and the model-free 30-day vol
of the VIX method (:func:`vix_style_vol`).
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

from .. import clock
from ..lets_be_rational import implied_black_volatility
from .forwards import carry_gaps, implied_forwards
from .local_vol import _market_values
from .quotes import MATURITY_TOLERANCE, QuoteParseError, _forward_message
from .surface import CALIBRATIONS, MIN_STRIKES_PER_SLICE, VolSurface, fit_market_surface
from .svi import durrleman_g

__all__ = [
    "CONSTANT_MATURITIES",
    "DEFAULT_SETTINGS",
    "GRID_DELTAS",
    "REASONS",
    "USED",
    "ChainFit",
    "FitSettings",
    "QuoteFilters",
    "atmf_vol",
    "delta_grid",
    "delta_vol",
    "expiry_table",
    "fit_chain",
    "forward_vols",
    "implied_spot",
    "slice_table",
    "slice_violations",
    "smile_variance",
    "vix_style_vol",
]

#: The status of a quote that the fit uses.
USED = "used"
#: Why a quote is left out, in the order the screen tests them (a quote takes the first that applies):
#:
#: * ``expired``: settled at or before the valuation time;
#: * ``crossed``: bid above the ask, or no offer;
#: * ``zero bid``: an offer only, so no mid (always left out: half the offer is not a price);
#: * ``short expiry``: fewer calendar days to settlement than :attr:`QuoteFilters.min_days`;
#: * ``weekly``: not a standard monthly expiry, with :attr:`QuoteFilters.weeklies` off;
#: * ``ITM side``: in the money against the expiry's forward, with :attr:`QuoteFilters.otm_only` on;
#: * ``no time value``: in the money with no more time value than half its spread (with OTM only off), or a mid at
#:   intrinsic;
#: * ``below floor`` / ``above cap``: a mid below the option's lower no-arbitrage bound, or at or above its upper one
#:   (the forward for a call, the strike for a put): no implied vol;
#: * ``wide``: a bid/ask spread wider than :attr:`QuoteFilters.max_spread` times the median of its expiry;
#: * ``thin expiry``: its expiry has fewer than :attr:`QuoteFilters.min_strikes` strikes left.
REASONS = (
    "expired",
    "crossed",
    "zero bid",
    "short expiry",
    "weekly",
    "ITM side",
    "no time value",
    "below floor",
    "above cap",
    "wide",
    "thin expiry",
)
#: The constant maturities of the term structure and the grid, in calendar days.
CONSTANT_MATURITIES = (7, 14, 30, 60, 91, 182, 365, 730)
#: The grid's deltas, spot convention: puts from 10 to 45 delta, 50 delta (a call), calls from 45 to 10.
GRID_DELTAS = (*(-d / 100 for d in range(10, 50, 5)), 0.50, *(d / 100 for d in range(45, 5, -5)))
_DAYS = 365.0
#: Points of the y-grid on which :func:`delta_vol` brackets its root, and how far out it looks (|y| at most this many
#: ATM standard deviations, and the surface's wing reach for a market surface).
_DELTA_GRID_POINTS, _DELTA_SDS = 401, 8.0


@dataclass(frozen=True)
class QuoteFilters:
    """Which quotes the fit uses (UI spec 3.4; defaults after LSEG IPA's volatility-surface filters).

    Attributes
    ----------
    otm_only : bool, default True
        Fit the out-of-the-money side only (puts below the forward, calls at and above it). Off, the in-the-money
        side is fitted too where it has more time value than half its spread, and a strike with both sides averages
        their vols.
    max_spread : float, default 4.0
        A quote whose bid/ask spread ($) is wider than this many times its expiry's median spread is left out
        (``math.inf``: no limit). In dollars, as LSEG measures it: on the default chain the widest out-of-the-money
        spread is 2.3 times its expiry's median, so 4 drops nothing there, while 1 drops half the quotes.
    min_strikes : int, default 6
        An expiry with fewer strikes left is left out whole (one SVI smile has five parameters).
    min_days : float, default 0.0
        Expiries closer than this many calendar days are left out. LSEG's default is 2; here it is 0, because every
        listed expiry of the published surface prices the sidebar's contract, and one left out would have no vol.
    weeklies : bool, default True
        Fit the expiries that are not standard monthlies (third Fridays, or the trading day before when that Friday
        is a holiday).
    """

    otm_only: bool = True
    max_spread: float = 4.0
    min_strikes: int = 6
    min_days: float = 0.0
    weeklies: bool = True

    def changes(self) -> tuple[str, ...]:
        """Each setting away from its default, in words (``wide above 1× the median spread``)."""
        default, out = QuoteFilters(), []
        if self.otm_only != default.otm_only:
            out.append("in-the-money quotes fitted too")
        if self.max_spread != default.max_spread:
            out.append(
                "no spread limit" if math.isinf(self.max_spread) else f"wide above {self.max_spread:g}× the median"
            )
        if self.min_strikes != default.min_strikes:
            out.append(f"at least {self.min_strikes} strikes per expiry")
        if self.min_days != default.min_days:
            out.append(f"expiries from {self.min_days:g} days")
        if self.weeklies != default.weeklies:
            out.append("weeklies left out")
        return tuple(out)


@dataclass(frozen=True)
class FitSettings:
    """Everything that decides the published surface besides the quotes: the screen, where each expiry's forward
    comes from (``"parity"`` or ``"rates"``, the given rate and dividend yield) and how the smiles are calibrated
    (``"arbitrage_free"`` or ``"unconstrained"``)."""

    filters: QuoteFilters = field(default_factory=QuoteFilters)
    forward: str = "parity"
    calibration: str = "arbitrage_free"

    def __post_init__(self) -> None:
        if self.forward not in ("parity", "rates"):
            raise ValueError(f"forward must be 'parity' or 'rates', got {self.forward!r}")
        if self.calibration not in CALIBRATIONS:
            raise ValueError(f"calibration must be one of {', '.join(CALIBRATIONS)}, got {self.calibration!r}")

    def changes(self) -> tuple[str, ...]:
        """Each setting away from its default, in words; empty at the defaults (the FIT chip's help)."""
        out = list(self.filters.changes())
        if self.forward != "parity":
            out.append("forwards from the given rate and yield, not put-call parity")
        if self.calibration != "arbitrage_free":
            out.append("unconstrained calibration (arbitrage allowed)")
        return tuple(out)

    @property
    def is_default(self) -> bool:
        return not self.changes()


#: The settings the published surface uses unless the user changes them.
DEFAULT_SETTINGS = FitSettings()


def is_weekly(day: date) -> bool:
    """Whether an expiry date is not a standard monthly one: the third Friday of its month, or the trading day before
    it when that Friday is an exchange holiday (as Good Friday or Juneteenth moves it).

    >>> is_weekly(date(2026, 11, 20)), is_weekly(date(2026, 11, 27)), is_weekly(date(2027, 6, 17))
    (False, True, False)
    """
    first = date(day.year, day.month, 1)
    third_friday = date(day.year, day.month, 1 + (4 - first.weekday()) % 7 + 14)
    standard = third_friday if clock.is_trading_day(third_friday) else clock.previous_trading_day(third_friday)
    return day != standard


@dataclass(frozen=True, eq=False)  # holds tables: compared by identity
class ChainFit:
    """A chain screened quote by quote and the surface fitted to the quotes it kept (see :func:`fit_chain`).

    Attributes
    ----------
    quotes : pandas.DataFrame
        One row per contract of the chain, in its order, with its canonical columns and ``label`` (its expiry's
        row in ``expiries``), ``maturity`` (years), ``forward``, ``discount_factor``, ``mid``, the Black-76 vols
        ``iv_bid``, ``iv_ask`` and ``iv_mid`` (NaN where there is none: a zero bid, a price outside the bounds),
        ``status`` (:data:`USED` or one of :data:`REASONS`) and ``fit_iv``, the fitted smile's vol at its strike
        (NaN where the surface does not reach).
    surface : VolSurface
        The SVI surface fitted to the used quotes.
    expiries : pandas.DataFrame
        One row per expiry of the chain, by maturity: ``expiry`` (date), ``root``, ``settlement``, ``maturity``,
        ``forward``, ``discount_factor``, ``source`` (of the forward), ``quoted`` and ``used`` (quotes), and
        ``slice``, its row in the surface's maturities (−1 where it was left out).
    forwards : pandas.DataFrame
        :func:`implied_forwards`' table for every expiry (``expiry`` is the label).
    settings : FitSettings
    messages : tuple of str
        How the chain was read.
    inversion_ms : float
        Wall time of the batch inversion of every bid, ask and mid, in milliseconds.
    """

    quotes: pd.DataFrame
    surface: VolSurface
    expiries: pd.DataFrame
    forwards: pd.DataFrame
    settings: FitSettings
    messages: tuple[str, ...]
    inversion_ms: float

    def counts(self) -> dict[str, int]:
        """``quoted``, ``used`` and the count of each reason (zeros included), with quoted = used + the reasons."""
        status = self.quotes["status"].value_counts()
        out = {"quoted": len(self.quotes), "used": int(status.get(USED, 0))}
        out.update({reason: int(status.get(reason, 0)) for reason in REASONS})
        return out

    def expiry_quotes(self, label: int) -> pd.DataFrame:
        """The quotes of one expiry (its ``label``), by strike, calls before puts."""
        return self.quotes[self.quotes["label"] == label]


def _first(status: np.ndarray, rows: np.ndarray, reason: str) -> None:
    """Give ``reason`` to the quotes in ``rows`` that have no status yet (in place)."""
    status[(status == "") & rows] = reason


def _check_enough_expiries(status: np.ndarray, label: np.ndarray, strike: np.ndarray, n_expiries: int) -> None:
    """Raise :class:`QuoteParseError` unless two or more expiries keep :data:`.surface.MIN_STRIKES_PER_SLICE`
    usable strikes, saying what was left and what the screen set aside.

    Worded around the surface the user asked for: :func:`.surface.fit_market_surface` would refuse the same quotes
    for local volatility's sake, and an upload with no usable quote (a header-only file, one whose bids are all
    missing) read as "local volatility needs at least two maturities ...; the quotes have 0 maturities".
    """
    used = status == USED
    usable = [np.unique(strike[used & (label == i)]).size for i in range(n_expiries)]
    enough = sum(n >= MIN_STRIKES_PER_SLICE for n in usable)
    if enough >= 2:
        return
    need = f"a surface needs at least two expiries with {MIN_STRIKES_PER_SLICE}+ usable strikes each"
    if not len(status):
        raise QuoteParseError(f"{need}; after reading, no contract is left to fit")
    text = (
        f"{need}; after reading and screening, the chain has {n_expiries} expir{'y' if n_expiries == 1 else 'ies'}, "
        f"{enough} of them with {MIN_STRIKES_PER_SLICE}+ usable strikes"
    )
    set_aside = {reason: int(np.sum(status == reason)) for reason in REASONS}
    reasons = [f"{count:,} {reason}" for reason, count in set_aside.items() if count]
    if reasons:
        text += f" (the screen set aside {sum(set_aside.values()):,} of {len(status):,} quotes: {', '.join(reasons)})"
    raise QuoteParseError(text)


def fit_chain(
    chain: pd.DataFrame,
    *,
    as_of: datetime,
    spot: float,
    rate: float,
    dividend_yield: float,
    settings: FitSettings = DEFAULT_SETTINGS,
    exercise: str = "european",
) -> ChainFit:
    """Screen a chain quote by quote and fit the arbitrage-checked SVI surface to the quotes it keeps.

    Parameters
    ----------
    chain : pandas.DataFrame
        The chain in the canonical schema (``market_data.CHAIN_COLUMNS``): ``root``, ``expiry``, ``settlement``,
        ``type``, ``strike``, ``bid``, ``ask``.
    as_of : datetime
        The valuation time (a naive datetime is New York wall time): each expiry's time runs to its settlement,
        in minutes / 525,600.
    spot : float
        The underlying's level (the surface's moneyness is K/S; parity gives the forwards).
    rate, dividend_yield : float
        The fallback carry: an expiry whose call/put pairs cannot give its forward takes ``S e^{(r − q)T}`` and
        ``e^{−rT}``, as does every expiry with ``settings.forward == "rates"`` or American ``exercise`` (parity holds
        only for European options).
    settings : FitSettings, default DEFAULT_SETTINGS
        The screen, the forwards and the calibration.
    exercise : {"european", "american"}, default "european"

    Returns
    -------
    ChainFit

    Raises
    ------
    options_analyzer.vol_surface.QuoteParseError
        If fewer than two expiries keep five or more usable strikes after the screen; the message counts them and
        the quotes the screen set aside, by reason.

    Examples
    --------
    >>> from options_analyzer.market_data import sample_chain
    >>> as_of = datetime(2026, 10, 23, 16, 0)
    >>> chain = sample_chain(5800.0, 0.04, 0.013, as_of=as_of)
    >>> fit = fit_chain(chain.quotes, as_of=as_of, spot=5800.0, rate=0.0445, dividend_yield=0.013)
    >>> counts = fit.counts()
    >>> counts["quoted"], counts["used"], counts["ITM side"], counts["zero bid"], counts["wide"]
    (5012, 2357, 2506, 149, 0)
    >>> counts["quoted"] == counts["used"] + sum(counts[reason] for reason in REASONS)
    True
    """
    filters = settings.filters
    rows = chain.reset_index(drop=True)
    days = [d.date() if isinstance(d, (pd.Timestamp, datetime)) else d for d in rows["expiry"]]
    keys = list(zip(days, rows["settlement"].astype(str), strict=True))
    maturity_of: dict[tuple[date, str], float] = {}
    for key in dict.fromkeys(keys):
        try:
            maturity_of[key] = clock.time_to_expiry(as_of, key[0], style=key[1])  # type: ignore[arg-type]
        except ValueError:  # outside the NYSE calendar
            maturity_of[key] = math.nan
    ordered = sorted(maturity_of, key=lambda key: (maturity_of[key], key))
    label_of = {key: i for i, key in enumerate(ordered)}
    label = np.array([label_of[key] for key in keys])
    T = np.array([maturity_of[key] for key in keys])
    strike = rows["strike"].to_numpy(float)
    bid, ask = rows["bid"].to_numpy(float), rows["ask"].to_numpy(float)
    call = (rows["type"] == "call").to_numpy()
    theta = np.where(call, 1.0, -1.0)
    status = np.full(len(rows), "", dtype=object)
    _first(status, ~(T > 0), "expired")
    _first(status, (ask <= 0) | (bid > ask), "crossed")
    _first(status, bid <= 0, "zero bid")
    two_sided = status == ""

    # 1. Forwards from the two-sided quotes of each expiry (the rows parse_quotes reads: no zero bids).
    parity = settings.forward == "parity" and exercise == "european"
    prices = pd.DataFrame(
        {
            "strike": strike[two_sided],
            "maturity": T[two_sided],
            "type": rows["type"].to_numpy()[two_sided],
            "price": 0.5 * (bid + ask)[two_sided],
            "price_error": 0.5 * (ask - bid)[two_sided],
        }
    )
    table = implied_forwards(prices, spot, rate, dividend_yield, expiry=label[two_sided], parity=parity)
    by_label = table.set_index("expiry")
    forward = np.empty(len(rows))
    discount = np.empty(len(rows))
    for i, key in enumerate(ordered):
        mine = label == i
        if i in by_label.index:
            forward[mine], discount[mine] = by_label.at[i, "forward"], by_label.at[i, "discount_factor"]
        else:  # no two-sided quote at all: the fallback carry
            forward[mine] = spot * math.exp((rate - dividend_yield) * maturity_of[key])
            discount[mine] = math.exp(-rate * maturity_of[key])

    # 2. Every bid, ask and mid as a Black-76 vol at the expiry's forward, in one batch.
    mid = 0.5 * (bid + ask)
    started = time.perf_counter()
    with np.errstate(all="ignore"):
        safe_T = np.where(T > 0, T, np.nan)
        vols = implied_black_volatility(
            np.concatenate([bid, ask, mid]) / np.tile(discount, 3),
            np.tile(forward, 3),
            np.tile(strike, 3),
            np.tile(safe_T, 3),
            np.tile(theta, 3),
        )
    inversion_ms = (time.perf_counter() - started) * 1e3
    iv_bid, iv_ask, iv_mid = np.split(np.asarray(vols, dtype=float), 3)
    iv_bid = np.where(bid > 0, iv_bid, np.nan)
    iv_ask = np.where(ask > 0, iv_ask, np.nan)
    iv_mid = np.where(bid > 0, iv_mid, np.nan)

    # 3. The screen.
    _first(status, filters.min_days > T * _DAYS, "short expiry")
    if not filters.weeklies:
        weekly = {key: is_weekly(key[0]) for key in ordered}
        _first(status, np.array([weekly[key] for key in keys], dtype=bool), "weekly")
    in_the_money = call != (strike >= forward)
    if filters.otm_only:
        _first(status, in_the_money, "ITM side")
    intrinsic = discount * np.maximum(theta * (forward - strike), 0.0)
    _first(status, in_the_money & (mid - intrinsic <= 0.5 * (ask - bid)), "no time value")
    _first(status, iv_mid == 0, "no time value")
    _first(status, np.isinf(iv_mid), "above cap")
    _first(status, np.isnan(iv_mid), "below floor")
    if math.isfinite(filters.max_spread):
        spread = ask - bid
        for i in range(len(ordered)):
            mine = (label == i) & (status == "")
            if mine.any():
                _first(status, mine & (spread > filters.max_spread * np.median(spread[mine])), "wide")
    for i in range(len(ordered)):
        mine = (label == i) & (status == "")
        if mine.any() and np.unique(strike[mine]).size < filters.min_strikes:
            _first(status, mine, "thin expiry")
    status[status == ""] = USED
    used = status == USED
    _check_enough_expiries(status, label, strike, len(ordered))

    # 4. The fit: one point per expiry and strike (a call and a put at one strike, with OTM only off, are averaged).
    points = pd.DataFrame(
        {
            "label": label[used],
            "strike": strike[used],
            "maturity": np.round(T[used], 10),
            "iv": iv_mid[used],
            "price_error": 0.5 * (ask - bid)[used],
            "forward": forward[used],
            "discount_factor": discount[used],
        }
    )
    points = points.groupby(["label", "strike"], as_index=False).mean()
    clean = points.sort_values(["maturity", "strike"]).reset_index(drop=True).drop(columns="label")
    if not parity or not len(table) or (table["source"] == "rates").all():
        clean = clean.drop(columns=["forward", "discount_factor"])  # the fit then uses the rate and yield itself
    fitted_labels = sorted(points["label"].unique())
    kept = table[table["expiry"].isin(fitted_labels)].drop(columns="expiry")
    surface = fit_market_surface(
        clean, spot, rate, dividend_yield, forwards=kept.reset_index(drop=True), calibration=settings.calibration
    )

    # Each expiry's slice of the surface, and each quote's fitted vol.
    slices = np.full(len(ordered), -1)
    for i, key in enumerate(ordered):
        if i in fitted_labels:
            nearest = int(np.argmin(np.abs(surface.maturities - maturity_of[key])))
            if abs(surface.maturities[nearest] - maturity_of[key]) <= 1e-8:
                slices[i] = nearest
    fit_iv = np.full(len(rows), np.nan)
    log_forward = np.log(spot) + np.asarray(surface.log_forward_ratio)
    for i, j in enumerate(slices):
        if j < 0:
            continue
        mine = label == i
        y = np.log(strike[mine]) - log_forward[j]
        w = surface.svi[j].total_variance(y)[0]
        lo, hi = surface.quoted_y[j]  # type: ignore[index]
        reach = float(surface._reach()[j])
        inside = ((y >= lo - 1e-9) & (y <= hi + 1e-9)) | (np.abs(y) <= reach)
        fit_iv[mine] = np.where(inside & (w > 0), np.sqrt(np.maximum(w, 0.0) / surface.maturities[j]), np.nan)

    quotes = rows.assign(
        label=label,
        maturity=T,
        forward=forward,
        discount_factor=discount,
        mid=mid,
        iv_bid=iv_bid,
        iv_ask=iv_ask,
        iv_mid=iv_mid,
        status=status,
        fit_iv=fit_iv,
    )
    first = {key: i for i, key in reversed(list(enumerate(keys)))}  # each expiry's first row, for its root
    expiries = pd.DataFrame(
        {
            "expiry": [key[0] for key in ordered],
            "root": [str(rows["root"].iloc[first[key]]) for key in ordered],
            "settlement": [key[1] for key in ordered],
            "maturity": [maturity_of[key] for key in ordered],
            "forward": [float(forward[label == i][0]) for i in range(len(ordered))],
            "discount_factor": [float(discount[label == i][0]) for i in range(len(ordered))],
            "source": [str(by_label.at[i, "source"]) if i in by_label.index else "rates" for i in range(len(ordered))],
            "quoted": np.bincount(label, minlength=len(ordered)),
            "used": np.bincount(label[used], minlength=len(ordered)),
            "slice": slices,
        }
    )
    messages = [_forward_message(table, settings.forward, exercise) if len(table) else "No two-sided quote."]
    messages.append(
        f"Inverted {3 * len(rows):,} bid, ask and mid prices of {len(rows):,} quotes into Black-76 vols at each "
        f"expiry's forward and discount factor in {inversion_ms:.0f} ms (batch Let's Be Rational)."
    )
    return ChainFit(quotes, surface, expiries, table, settings, tuple(messages), inversion_ms)


def implied_spot(chain: pd.DataFrame, *, as_of: datetime) -> float:
    """The underlying's level implied by put-call parity at the front expiry whose pairs give it: ``F·DF``, which is
    the spot up to the dividend yield over that expiry (``S = F·DF·e^{qT}``: 0.02% for a 1.3% yield a week out). For
    an uploaded chain whose file does not state the spot. NaN where no expiry has enough call/put pairs.

    Examples
    --------
    >>> from options_analyzer.market_data import sample_chain
    >>> as_of = datetime(2026, 10, 23, 16, 0)
    >>> round(implied_spot(sample_chain(5800.0, 0.04, 0.013, as_of=as_of).quotes, as_of=as_of), 1)
    5798.7
    """
    rows = chain[(chain["bid"] > 0) & (chain["ask"] >= chain["bid"])]
    if rows.empty:
        return math.nan
    keys = list(zip((pd.Timestamp(d).date() for d in rows["expiry"]), rows["settlement"].astype(str), strict=True))
    maturity = {}
    for key in dict.fromkeys(keys):
        try:
            maturity[key] = clock.time_to_expiry(as_of, key[0], style=key[1])  # type: ignore[arg-type]
        except ValueError:
            maturity[key] = math.nan
    T = np.array([maturity[key] for key in keys])
    live = T > 0
    prices = pd.DataFrame(
        {
            "strike": rows["strike"].to_numpy(float)[live],
            "maturity": T[live],
            "type": rows["type"].to_numpy()[live],
            "price": (0.5 * (rows["bid"] + rows["ask"])).to_numpy(float)[live],
            "price_error": (0.5 * (rows["ask"] - rows["bid"])).to_numpy(float)[live],
        }
    )
    if prices.empty:
        return math.nan
    guess = float(np.median(prices["strike"]))
    table = implied_forwards(prices, guess, 0.0, 0.0)
    fitted = table[table["source"] != "rates"]
    if fitted.empty:
        return math.nan
    front = fitted.iloc[0]
    return float(front["forward"] * front["discount_factor"])


# --------------------------------------------------------------------------------------------------------------------
# The smile at any maturity
# --------------------------------------------------------------------------------------------------------------------


def log_forward_ratio(surface: VolSurface, T: float) -> float:
    """``ln(F/S)`` at ``T``: ``carry T`` for SSVI, the fitted forwards' ``ln(F/S)`` linear in ``T`` for a market
    surface."""
    if surface.log_forward_ratio is None:
        return surface.carry * T
    return float(np.interp(T, surface.maturities, surface.log_forward_ratio))


def discount_factor(surface: VolSurface, T: float, rate: float) -> float:
    """The discount factor to ``T``: ``e^{−rate T}`` for SSVI, the fitted discount factors' rate ``−ln(DF)/T`` linear
    in ``T`` for a market surface."""
    if surface.discount_factor is None:
        return math.exp(-rate * T)
    rates = -np.log(surface.discount_factor) / surface.maturities
    return math.exp(-float(np.interp(T, surface.maturities, rates)) * T)


def smile_variance(surface: VolSurface, T: float, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Total implied variance ``w`` and its y-derivatives at ``T`` and forward log-moneyness ``y``.

    SSVI is read at ``T`` itself, within the maturities of its grid. A market surface is read from its SVI slices:
    the one fitted at ``T`` (within half a day, its smile in vol), else the two around it, linear in ``T`` at fixed
    ``y`` (as :meth:`VolSurface.implied_vol_at` interpolates); NaN outside the fitted maturities. The slices are
    read at every ``y`` (their wings beyond the quotes included): callers bound ``y``.
    """
    y = np.asarray(y, dtype=float)
    Ts = surface.maturities
    nan = np.full(y.shape, np.nan)
    if not Ts[0] - MATURITY_TOLERANCE <= T <= Ts[-1] + MATURITY_TOLERANCE:
        return nan, nan, nan
    if surface.ssvi is not None:
        w, w_y, w_yy, _ = surface.ssvi.total_variance(y, T)
        return np.asarray(w, dtype=float), np.asarray(w_y, dtype=float), np.asarray(w_yy, dtype=float)
    if not surface.svi:
        return nan, nan, nan
    nearest = int(np.argmin(np.abs(Ts - T)))
    if abs(Ts[nearest] - T) <= MATURITY_TOLERANCE:
        w, w_y, w_yy = surface.svi[nearest].total_variance(y)
        scale = T / Ts[nearest]  # the same smile in vol: total variance in proportion to time
        return w * scale, w_y * scale, w_yy * scale
    j = int(np.searchsorted(Ts, T)) - 1
    a = (T - Ts[j]) / (Ts[j + 1] - Ts[j])
    lo, hi = surface.svi[j].total_variance(y), surface.svi[j + 1].total_variance(y)
    w, w_y, w_yy = ((1 - a) * p + a * q for p, q in zip(lo, hi, strict=True))
    return w, w_y, w_yy


def _reach(surface: VolSurface, T: float) -> float:
    """How far from the forward (|y|) the smile at ``T`` is read: the surface's extrapolation reach."""
    return surface.reach_at(T)


def atmf_vol(surface: VolSurface, T: float) -> float:
    """The at-the-money-forward vol (K = F) at maturity ``T`` years; NaN outside the fitted maturities.

    >>> from options_analyzer.vol_surface import SSVIParams, ssvi_surface
    >>> s = ssvi_surface(SSVIParams(), 0.05, 0.0, np.linspace(0.7, 1.3, 41), np.linspace(0.1, 2.0, 20))
    >>> round(atmf_vol(s, 1.0), 6) == round(float(np.sqrt(SSVIParams().theta(1.0))), 6)
    True
    """
    w = float(smile_variance(surface, T, np.zeros(1))[0][0])
    return math.sqrt(w / T) if w > 0 else math.nan


def delta_vol(surface: VolSurface, T: float, delta: float, *, rate: float = 0.0) -> tuple[float, float]:
    """The vol and forward log-moneyness ``y`` of the strike whose spot delta, at its own vol, is ``delta``.

    Spot delta is ``e^{−qT} N(d₁)`` for a call (``delta`` in (0, 1)) and ``−e^{−qT} N(−d₁)`` for a put (in (−1, 0)),
    with ``e^{−qT} = F·DF/S`` (:func:`discount_factor`; ``rate`` is the discount rate of an SSVI surface) and
    ``d₁ = −y/√w + √w/2``. The root is bracketed on a grid of ``y`` within the surface's reach and refined with
    Brent's method to 1e−12 in ``y``.

    Returns
    -------
    (float, float)
        ``(vol, y)``; NaN where no strike within the reach has that delta.
    """
    if not (0 < abs(delta) < 1):
        raise ValueError(f"delta must lie in (−1, 0) or (0, 1), got {delta!r}")
    w0 = float(smile_variance(surface, T, np.zeros(1))[0][0])
    if not w0 > 0:
        return math.nan, math.nan
    carry_factor = math.exp(log_forward_ratio(surface, T)) * discount_factor(surface, T, rate)
    target = abs(delta) / carry_factor
    if not 0 < target < 1:
        return math.nan, math.nan
    d1_target = float(ndtri(target)) if delta > 0 else -float(ndtri(target))
    span = min(_DELTA_SDS * math.sqrt(w0), _reach(surface, T)) if surface.ssvi is None else _DELTA_SDS * math.sqrt(w0)

    def gap(y):
        w = smile_variance(surface, T, np.atleast_1d(y))[0]
        with np.errstate(all="ignore"):
            root = np.sqrt(w)
            return -np.atleast_1d(y) / root + 0.5 * root - d1_target

    grid = np.linspace(-span, span, _DELTA_GRID_POINTS)
    values = gap(grid)
    change = np.nonzero(
        np.isfinite(values[:-1]) & np.isfinite(values[1:]) & (np.sign(values[:-1]) != np.sign(values[1:]))
    )
    if not change[0].size:
        return math.nan, math.nan
    i = int(change[0][0])
    y = brentq(lambda v: float(gap(v)[0]), grid[i], grid[i + 1], xtol=1e-12)
    w = float(smile_variance(surface, T, np.array([y]))[0][0])
    return math.sqrt(w / T), float(y)


def risk_reversal(surface: VolSurface, T: float, *, rate: float = 0.0, delta: float = 0.25) -> tuple[float, float]:
    """``(RR, BF)`` at ``T``: σ(Δc) − σ(Δp) and ½(σ(Δc) + σ(Δp)) − σ(ATMF), for ``delta`` (default 25)."""
    call = delta_vol(surface, T, delta, rate=rate)[0]
    put = delta_vol(surface, T, -delta, rate=rate)[0]
    return call - put, 0.5 * (call + put) - atmf_vol(surface, T)


def forward_vols(surface: VolSurface, days=CONSTANT_MATURITIES) -> list[tuple[int, int, float]]:
    """ATMF forward vols between consecutive constant maturities: ``(from, to, vol)`` with
    ``vol = sqrt((w₂ − w₁) / (T₂ − T₁))`` (NaN where either end is not fitted or the variance falls)."""
    out = []
    for start, end in itertools.pairwise(days):
        T1, T2 = start / _DAYS, end / _DAYS
        w1, w2 = (float(smile_variance(surface, T, np.zeros(1))[0][0]) for T in (T1, T2))
        variance = (w2 - w1) / (T2 - T1)
        out.append((start, end, math.sqrt(variance) if variance > 0 else math.nan))
    return out


def delta_grid(surface: VolSurface, *, rate: float = 0.0, days=CONSTANT_MATURITIES) -> pd.DataFrame:
    """Implied vols by delta (columns, :data:`GRID_DELTAS`) at constant maturities (rows, in calendar days)."""
    rows = {d: [delta_vol(surface, d / _DAYS, delta, rate=rate)[0] for delta in GRID_DELTAS] for d in days}
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(GRID_DELTAS))


# --------------------------------------------------------------------------------------------------------------------
# The expiry table
# --------------------------------------------------------------------------------------------------------------------


def _call_delta(surface: VolSurface, T: float, y: np.ndarray, rate: float) -> np.ndarray:
    w = smile_variance(surface, T, y)[0]
    with np.errstate(all="ignore"):
        root = np.sqrt(w)
        d1 = -y / root + 0.5 * root
    return math.exp(log_forward_ratio(surface, T)) * discount_factor(surface, T, rate) * ndtr(d1)


def slice_table(
    surface: VolSurface, *, rate: float, dividend_yield: float, min_strikes: int = QuoteFilters().min_strikes
) -> pd.DataFrame:
    """One row per maturity of a surface, in numbers (the Surface tab formats them).

    Columns: ``slice`` (the row of the surface's maturities), ``maturity`` (years), ``forward``,
    ``discount_factor``, ``carry_gap_bp`` (the forward against the reference carry ``rate`` and ``dividend_yield``:
    ``ln(F/S) − (r − q)T`` in basis points of the forward), ``atmf``, ``rr25`` and ``bf25`` (25-delta risk reversal
    σ(25Δc) − σ(25Δp) and butterfly ½(σ(25Δc) + σ(25Δp)) − σ(ATMF)), ``rmse`` (the fit's, vol), ``in_band`` (share
    of its quotes whose fitted vol is within their half-spread in vol), ``used`` (quotes fitted), ``confidence``
    (share of the call-delta interval [0.10, 0.90] that the quoted strikes span, after ORATS's ``confidence``),
    ``skew_90_110`` (σ(K/F 90%) − σ(110%)), ``mixon`` ((σ25Δp − σ25Δc) / σATMF, Mixon's normalised skew) and
    ``flags``: ``carry`` (the reference carry misses a parity forward by more than 5 bp and half the ATM spread,
    SF-3's rule), ``thin`` (fewer than twice ``min_strikes`` strikes quoted), ``cal`` and ``bfly`` (calendar or
    butterfly arbitrage on the slice's grid), ``extrap`` (a 25-delta strike beyond the quotes). Quote columns are NaN
    for a surface without quotes (SSVI); an SSVI "quoted range" is its K/S grid.
    """
    Ts = surface.maturities
    spot = float(surface.spot) if surface.spot is not None else 1.0
    calendar = butterfly = np.zeros((len(Ts), 1), dtype=bool)
    if surface.svi and surface.quoted_y is not None:
        values = _market_values(surface.svi, Ts, surface.quoted_y, surface._log_fs(), surface.moneyness)
        calendar, butterfly = values.calendar_bad, values.butterfly_bad
    forwards = surface.forwards
    gap_flag: dict[int, bool] = {}
    if forwards is not None and len(forwards):
        reference = (rate - dividend_yield) * forwards["maturity"].to_numpy(float)
        table = forwards.assign(gap_bp=1e4 * (np.log(forwards["forward"].to_numpy(float) / spot) - reference))
        flags = carry_gaps(table)
        for T, flag in zip(table["maturity"].to_numpy(float), flags, strict=True):
            gap_flag[int(np.argmin(np.abs(Ts - T)))] = bool(flag)
    rows = []
    for j, T in enumerate(Ts):
        T = float(T)
        log_fs = log_forward_ratio(surface, T)
        if surface.quoted_y is not None:
            lo, hi = (float(v) for v in surface.quoted_y[j])
        else:
            lo, hi = float(np.log(surface.moneyness[0]) - log_fs), float(np.log(surface.moneyness[-1]) - log_fs)
        atmf = atmf_vol(surface, T)
        call, y_call = delta_vol(surface, T, 0.25, rate=rate)
        put, y_put = delta_vol(surface, T, -0.25, rate=rate)
        ends = _call_delta(surface, T, np.array([lo, hi]), rate)
        span = float(np.clip(ends[0], 0.10, 0.90) - np.clip(ends[1], 0.10, 0.90)) / 0.80
        used, in_band, strikes = math.nan, math.nan, math.inf
        if surface.quotes is not None:
            mine = surface.quotes[np.isclose(surface.quotes["maturity"], T, rtol=0.0, atol=1e-9)]
            used, strikes = len(mine), mine["strike"].nunique()
            half = mine["vol_half_spread"].to_numpy(float) if "vol_half_spread" in mine else np.full(len(mine), np.nan)
            known = np.isfinite(half) & (half > 0)
            if known.any():
                miss = np.abs(mine["fitted_iv"] - mine["iv"]).to_numpy(float)[known]
                in_band = float(np.mean(miss <= half[known]))
        w90, w110 = smile_variance(surface, T, np.log([0.9, 1.1]))[0]
        flags = []
        if gap_flag.get(j, False):
            flags.append("carry")
        if strikes < 2 * min_strikes:
            flags.append("thin")
        if calendar[j].any():
            flags.append("cal")
        if butterfly[j].any():
            flags.append("bfly")
        if any(not (lo - 1e-9 <= y <= hi + 1e-9) for y in (y_call, y_put) if np.isfinite(y)):
            flags.append("extrap")
        rows.append(
            {
                "slice": j,
                "maturity": T,
                "forward": spot * math.exp(log_fs),
                "discount_factor": discount_factor(surface, T, rate),
                "carry_gap_bp": 1e4 * (log_fs - (rate - dividend_yield) * T) if forwards is not None else math.nan,
                "atmf": atmf,
                "rr25": call - put,
                "bf25": 0.5 * (call + put) - atmf,
                "rmse": float(surface.fit_rmse[j]) if surface.fit_rmse is not None else math.nan,
                "in_band": in_band,
                "used": used,
                "confidence": max(span, 0.0),
                "skew_90_110": (math.sqrt(max(w90, 0.0) / T) - math.sqrt(max(w110, 0.0) / T)) if w90 > 0 else math.nan,
                "mixon": (put - call) / atmf,
                "flags": " ".join(flags),
            }
        )
    return pd.DataFrame(rows)


def expiry_table(fit: ChainFit, *, rate: float, dividend_yield: float) -> pd.DataFrame:
    """:func:`slice_table` of a chain's fitted surface, one row per fitted expiry with its ``label``, ``expiry``
    (date) and ``settlement``, and the quote columns measured on the chain's own bid/ask vols: ``used`` (quotes the
    fit used) and ``in_band`` (share of them whose fitted vol lies within their bid/ask in vol).

    The reference carry is the rate and yield the forwards take when they do not come from parity (the sidebar's,
    or the index's Treasury rate plus the box spread): a ``carry`` flag says the quotes resolve a forward it misses."""
    table = slice_table(
        fit.surface, rate=rate, dividend_yield=dividend_yield, min_strikes=fit.settings.filters.min_strikes
    ).set_index("slice")
    rows = []
    for label, row in fit.expiries.iterrows():
        j = int(row["slice"])
        if j < 0:
            continue
        quotes = fit.quotes[(fit.quotes["label"] == label) & (fit.quotes["status"] == USED)]
        vol_half = 0.5 * (quotes["iv_ask"] - quotes["iv_bid"]).to_numpy(float)
        miss = np.abs(quotes["fit_iv"] - quotes["iv_mid"]).to_numpy(float)
        measured = table.loc[j].to_dict()
        measured.update(
            label=label,
            expiry=row["expiry"],
            settlement=row["settlement"],
            slice=j,
            used=len(quotes),
            in_band=float(np.mean(miss <= vol_half)) if len(quotes) else math.nan,
        )
        rows.append(measured)
    columns = ["label", "expiry", "settlement", "slice", *table.columns]
    return pd.DataFrame(rows, columns=columns)


def slice_violations(surface: VolSurface, j: int) -> list[tuple[float, str]]:
    """Where slice ``j`` of a market surface has arbitrage on the surface's grid: ``(K/S, "cal" or "bfly")`` for each
    grid point with calendar arbitrage (total variance falling to the next expiry) or butterfly arbitrage
    (Durrleman's g < 0); empty for SSVI, which has none within its parameter bounds."""
    if not surface.svi or surface.quoted_y is None:
        return []
    values = _market_values(surface.svi, surface.maturities, surface.quoted_y, surface._log_fs(), surface.moneyness)
    out = [(float(m), "cal") for m in surface.moneyness[values.calendar_bad[j]]]
    return out + [(float(m), "bfly") for m in surface.moneyness[values.butterfly_bad[j]]]


def durrleman(surface: VolSurface, T: float, y) -> np.ndarray:
    """Durrleman's ``g(y)`` of the smile at ``T``: a butterfly-arbitrage-free smile has ``g >= 0`` everywhere."""
    w, w_y, w_yy = smile_variance(surface, T, y)
    with np.errstate(all="ignore"):
        return durrleman_g(np.asarray(y, dtype=float), w, w_y, w_yy)


# --------------------------------------------------------------------------------------------------------------------
# The model-free 30-day vol (the VIX method)
# --------------------------------------------------------------------------------------------------------------------


def _expiry_variance(quotes: pd.DataFrame, T: float, discount: float) -> float:
    """The VIX method's variance of one expiry (Cboe VIX white paper, 2019, steps 1 and 2).

    ``F`` from the strike where the call and put mids differ least: ``F = K* + e^{RT} (C − P)``; ``K₀`` the first
    strike at or below ``F``; then the out-of-the-money mids (puts below ``K₀``, calls above, their average at
    ``K₀``), each side moving outward from ``K₀`` and stopping after two consecutive zero bids, zero bids skipped;
    ``σ² = (2/T) Σ ΔK/K² e^{RT} Q(K) − (1/T)(F/K₀ − 1)²`` with ``ΔK`` half the distance between the neighbouring
    strikes (the distance to the one neighbour at either end).
    """
    calls = quotes[quotes["type"] == "call"].set_index("strike")
    puts = quotes[quotes["type"] == "put"].set_index("strike")
    both = calls.index.intersection(puts.index)
    two_sided = [k for k in both if calls.at[k, "bid"] > 0 and puts.at[k, "bid"] > 0]
    if not two_sided:
        return math.nan
    gaps = {k: calls.at[k, "mid"] - puts.at[k, "mid"] for k in two_sided}
    k_star = min(two_sided, key=lambda k: (abs(gaps[k]), k))
    growth = 1.0 / discount  # e^{RT}
    forward = k_star + growth * gaps[k_star]
    strikes = np.sort(both.to_numpy(float))
    below = strikes[strikes <= forward]
    if not below.size:
        return math.nan
    k0 = float(below[-1])
    chosen: dict[float, float] = {}
    for frame, order in ((puts, strikes[strikes < k0][::-1]), (calls, strikes[strikes > k0])):
        zeros = 0
        for k in order:
            if frame.at[k, "bid"] <= 0:
                zeros += 1
                if zeros == 2:
                    break
                continue
            zeros = 0
            chosen[float(k)] = float(frame.at[k, "mid"])
    chosen[k0] = 0.5 * (float(calls.at[k0, "mid"]) + float(puts.at[k0, "mid"]))
    ks = np.array(sorted(chosen))
    q = np.array([chosen[k] for k in ks])
    dk = np.empty_like(ks)
    if ks.size == 1:
        return math.nan
    dk[1:-1] = 0.5 * (ks[2:] - ks[:-2])
    dk[0], dk[-1] = ks[1] - ks[0], ks[-1] - ks[-2]
    return float(2.0 / T * np.sum(dk / ks**2 * growth * q) - (forward / k0 - 1.0) ** 2 / T)


def vix_style_vol(fit: ChainFit, days: float = 30.0) -> tuple[float, tuple[int, int] | None]:
    """The model-free vol to ``days`` by the VIX method (Cboe VIX white paper): the variance of the two listed
    expiries around ``days`` from their out-of-the-money mids (:func:`_expiry_variance`), interpolated in total
    variance (in minutes, as Cboe does), as an annualised vol.

    Every listed expiry is a candidate, weeklies included, as Cboe uses the SPXW Friday expiries; the two must lie
    on either side of ``days`` (Cboe's window is 23 to 37 days; here any pair around it). The chain's own quotes are
    read, before the fit's screen, since the method is model-free.

    Returns
    -------
    (float, (int, int) or None)
        The vol (NaN where no pair of expiries brackets ``days``) and the labels of the two expiries used.
    """
    target = days / _DAYS
    table = fit.expiries
    T = table["maturity"].to_numpy(float)
    near = np.nonzero(target >= T)[0]
    far = np.nonzero(target < T)[0]
    if not near.size or not far.size:
        return math.nan, None
    i, j = int(near[-1]), int(far[0])
    variances = []
    for label in (i, j):
        quotes = fit.quotes[fit.quotes["label"] == label]
        variances.append(_expiry_variance(quotes, T[label], float(table.at[label, "discount_factor"])))
    w1, w2 = T[i] * variances[0], T[j] * variances[1]
    w = w1 + (w2 - w1) * (target - T[i]) / (T[j] - T[i])
    return (math.sqrt(w / target) if w > 0 else math.nan), (i, j)
