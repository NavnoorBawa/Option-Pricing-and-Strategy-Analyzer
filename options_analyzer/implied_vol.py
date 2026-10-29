"""Robust Black-Scholes-Merton implied volatility.

The European solver

1. checks the quote against the model-free no-arbitrage bounds (a price
   outside them has *no* implied volatility, and saying so is better than
   returning a meaningless number);
2. converts in-the-money quotes to the equivalent out-of-the-money option via
   put-call parity, which carries the same time value but is far better
   conditioned numerically. Subtracting the intrinsic value leaves that time
   value with the absolute rounding of the whole price, so an in-the-money
   quote within ``1e-12·max(S, K)`` of the lower bound (the margin the bound
   check already allows below it), like an out-of-the-money quote of zero, has
   no time value to identify a volatility from: it is rejected with the reason
   ``"no_time_value"`` instead of being given whatever volatility its last
   digits imply (a 2-day call struck at 60, spot 100, rate 4 %: 0 % at its
   intrinsic value, 94.5 % at 1e-13 above it). An out-of-the-money quote is all
   time value, keeps its full relative precision and is inverted however small
   it is;
3. inverts the normalised out-of-the-money price with Peter Jäckel's "Let's Be
   Rational" (:mod:`options_analyzer.lets_be_rational`), which reaches machine
   precision in two Householder steps, and falls back to Brent's method on a
   bracket of the same normalised price (guaranteed to converge once a bracket
   exists) for any quote where those steps do not report convergence.

:func:`implied_volatility_batch` does the same for arrays of quotes in one
vectorised pass, with NaN and a reason for each rejected quote, and the vega at
each volatility found.

:func:`implied_volatility_american` inverts a binomial tree (Leisen-Reimer by
default) for American quotes (listed single-stock options), with American
bounds; :func:`implied_volatility_alo` inverts the Andersen-Lake-Offengenden
engine (:mod:`options_analyzer.american`) with the same bounds. Both report a
quote no more than ``1e-12·max(S, K)`` above the option's zero-volatility value
as having no time value, as the European solver does in the money; they apply
the margin out of the money too, because their values near zero volatility are
accurate only in absolute terms (see :func:`implied_volatility_alo`).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from . import american, binomial
from . import black_scholes as bs
from . import lets_be_rational as lbr

__all__ = [
    "REJECTION_REASONS",
    "TIME_VALUE_MARGIN",
    "ImpliedVolBatch",
    "ImpliedVolError",
    "american_bounds",
    "implied_volatility",
    "implied_volatility_alo",
    "implied_volatility_american",
    "implied_volatility_batch",
    "lowest_tree_volatility",
    "no_arbitrage_bounds",
]

MIN_VOL = 1e-6  # lower bracket of the American solver (the tree needs a positive volatility)
MAX_VOL = 20.0
_VOL_FLOOR = 1e-12  # European volatilities below this are reported as 0 (and it is the Brent fallback's lower bracket)
#: The margin around the lower bound, relative to ``max(S, K)``. A European quote up to this far below the bound is
#: taken as on it (the bound carries rounding too), and an in-the-money quote no more than this far above it (for an
#: American option, above its zero-volatility value) has no time value. Subtracting the intrinsic value leaves the
#: time value with the absolute rounding of the price and of the intrinsic value, a few ε·max(S, K) (ε = 2.2e-16);
#: at this margin that is a few 2.2e-4 of the time value, and so deep in the wing the volatility's relative error is
#: that divided by about 2·ln(1/β) = 55 (β = 1e-12, the normalised time value). Measured on 20,000 random quotes
#: priced in double precision (tests/test_implied_vol.py): every quote with more time value inverts to within 7.7e-6
#: (relative) of the volatility that priced it, and the errors fall in proportion to the time value. With less, the
#: last digits of the price decide the volatility: a 2-day call struck at 60, spot 100, rate 4 %, gives 0 % at its
#: intrinsic value, 94.5 % at 1e-13 above it and 230 % at half a cent above it.
TIME_VALUE_MARGIN = 1e-12

#: The codes of :attr:`ImpliedVolError.reason`, with what each means.
REJECTION_REASONS = {
    "not_finite": "an input, or the discounted spot S e^(-qT) or strike K e^(-rT), is not a finite number",
    "not_positive": "the spot or the strike is not positive",
    "expired": "the option has expired",
    "below_lower_bound": "the price is below the no-arbitrage lower bound",
    "no_time_value": "the price has no time value to identify a volatility from: it is at most 1e-12·max(S, K) above "
    "the option's value at zero volatility (in the money, the discounted intrinsic value; for an American option, "
    "typically the value of exercising now) and not below it by more than the solver tolerates, where its last digits "
    "(for an American option, the engine's error near zero volatility) would decide the volatility; or, for a "
    "European option out of the money, it is zero (or underflows to zero once divided by √(S e^(-qT)·K e^(-rT)))",
    "at_or_above_upper_bound": "the price is at or above the no-arbitrage upper bound",
    "above_max_vol": f"the implied volatility exceeds {MAX_VOL:.0%}",
    "below_tree_floor": "the price is not above the value at the lowest volatility the binomial tree can represent",
    "invalid": "the quote cannot be inverted (the default code, where no specific one applies)",
}
_ABOVE_MAX_VOL = f"implied volatility exceeds {MAX_VOL:.0%}; check the quote and inputs"


class ImpliedVolError(ValueError):
    """Raised when a quote admits no implied volatility.

    The message explains why; :attr:`reason` is a short code for it, one of :data:`REJECTION_REASONS`.
    """

    def __init__(self, message: str, reason: str = "invalid") -> None:
        super().__init__(message)
        self.reason = reason

    def __reduce__(self):  # keep the reason when pickled (multiprocessing, caches)
        return type(self), (str(self), self.reason)


def _exp(value: float) -> float:
    """``math.exp``, with overflow giving ``inf`` (as NumPy does) instead of an exception."""
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def _discounted(S: float, K: float, T: float, r: float, q: float) -> tuple[float, float]:
    """``(S e^(-qT), K e^(-rT))``. The batch solver repeats this arithmetic exactly (libm's exp, not NumPy's)."""
    return S * _exp(-q * T), K * _exp(-r * T)


def no_arbitrage_bounds(S: float, K: float, T: float, r: float, q: float = 0.0, option_type: str = "call"):
    """Model-free bounds ``(lower, upper)`` for a European option price."""
    phi = bs.option_sign(option_type)
    fwd_s, pv_k = _discounted(S, K, T, r, q)
    lower = max(phi * (fwd_s - pv_k), 0.0)
    upper = fwd_s if phi > 0 else pv_k
    return lower, upper


def _check_contract(price, S, K, T, r, q) -> tuple[float, float, float, float, float, float]:
    """The checks every solver makes before looking at bounds; returns the inputs as floats."""
    values = (float(price), float(S), float(K), float(T), float(r), float(q))
    names = ("price", "spot", "strike", "time to expiry", "rate", "dividend yield")
    for name, value in zip(names, values, strict=True):
        if not math.isfinite(value):
            raise ImpliedVolError(f"{name} must be a finite number", "not_finite")
    _, S, K, T, _, _ = values
    if S <= 0 or K <= 0:
        raise ImpliedVolError("spot and strike must be positive", "not_positive")
    if T <= 0:
        raise ImpliedVolError("the option has expired: implied volatility is undefined at T = 0", "expired")
    return values


def _check_european_quote(price, S, K, T, r, q, phi: int) -> tuple[float, ...]:
    """Raise the :class:`ImpliedVolError` of a quote outside the European bounds; return the floats and forwards.

    Returns ``(price, S, K, T, r, q, fwd_s, pv_k)`` as floats, ``fwd_s = S e^(-qT)`` and ``pv_k = K e^(-rT)``.
    Where one of these overflows (``-rT`` or ``-qT`` beyond about 709, less for a large spot or strike) the bounds
    and the normalised price become ``inf - inf`` or 0, so the quote is rejected as not finite rather than given a
    meaningless volatility.
    """
    price, S, K, T, r, q = _check_contract(price, S, K, T, r, q)
    fwd_s, pv_k = _discounted(S, K, T, r, q)
    if not (math.isfinite(fwd_s) and math.isfinite(pv_k)):
        raise ImpliedVolError(
            f"the discounted spot S e^(-qT) ({fwd_s:.6g}) or strike K e^(-rT) ({pv_k:.6g}) overflows; "
            "check the rate, dividend yield and time to expiry",
            "not_finite",
        )
    lower = max(phi * (fwd_s - pv_k), 0.0)
    upper = fwd_s if phi > 0 else pv_k
    if price < lower - TIME_VALUE_MARGIN * max(S, K):
        raise ImpliedVolError(
            f"price {price:.6g} is below the no-arbitrage lower bound {lower:.6g} "
            "(discounted intrinsic value); no volatility can produce it",
            "below_lower_bound",
        )
    if price >= upper:
        raise ImpliedVolError(
            f"price {price:.6g} is at or above the no-arbitrage upper bound {upper:.6g}; "
            "no finite volatility can produce it",
            "at_or_above_upper_bound",
        )
    return price, S, K, T, r, q, fwd_s, pv_k


def implied_volatility(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = "call",
    *,
    tol: float = 1e-12,
) -> float:
    """Volatility that reproduces ``price`` under Black-Scholes-Merton.

    Raises :class:`ImpliedVolError` with an explanation when the quote is
    outside the no-arbitrage bounds or implies a volatility above
    ``MAX_VOL`` (2,000 %), when the discounted spot ``S e^(-qT)`` or
    strike ``K e^(-rT)`` overflows (reason ``"not_finite"``), and when the
    quote has no time value (reason ``"no_time_value"``): in the money
    (``ln(F/K)`` of the option's sign), a price within ``1e-12·max(S, K)``
    of the lower bound, where the last digits of the price would decide the
    volatility; out of the money, a price of zero (or one so small that it
    underflows once divided by ``√(S e^(-qT)·K e^(-rT))``, below about
    ``5e-324`` of that). The result is accurate to machine precision (Let's
    Be Rational); ``tol`` is the volatility tolerance of the Brent fallback,
    used only where Let's Be Rational does not report convergence. A
    volatility below ``1e-12`` (an out-of-the-money quote with that little
    time value) is returned as 0.

    Examples
    --------
    >>> price = bs.price(100, 110, 0.5, 0.03, 0.25, 0.0, "call")
    >>> round(implied_volatility(price, 100, 110, 0.5, 0.03, 0.0, "call"), 14)
    0.25
    >>> try:  # a 2-day call at its intrinsic value, 100 - 60
    ...     implied_volatility(40.0, 100, 60, 2 / 365, 0.0)
    ... except ImpliedVolError as error:
    ...     print(error.reason)
    no_time_value
    """
    phi = bs.option_sign(option_type)
    price, S, K, T, r, q, fwd_s, pv_k = _check_european_quote(price, S, K, T, r, q, phi)

    # Work with the out-of-the-money side: same time value, better conditioning. In the money (ln(F/K) of the
    # option's sign) that means subtracting the intrinsic value, which leaves the time value with the absolute
    # rounding of the whole price; out of the money the price is all time value, with its full relative precision.
    # (A hair from the forward, where the intrinsic value is below the margin too, the two sides differ only for
    # quotes below the margin, which mean a total volatility below about 2.5e-12: no time value on the in-the-money
    # side, a volatility of about that, 0 below 1e-12, on the other.)
    x = _log_moneyness(S, K, T, r, q)
    in_the_money = phi * x > 0
    otm_price = price - phi * _forward_intrinsic(fwd_s, pv_k, x) if in_the_money else price
    if otm_price <= (TIME_VALUE_MARGIN * max(S, K) if in_the_money else 0.0):
        raise _no_time_value(price, S, K, phi, fwd_s, pv_k, in_the_money, underflow=False)
    # The undiscounted out-of-the-money price over √(FK); with time value left, both forwards are positive (and finite).
    beta = otm_price / (math.sqrt(fwd_s) * math.sqrt(pv_k))
    if beta == 0.0:  # a time value below about 5e-324·√(FK): the division leaves none
        raise _no_time_value(price, S, K, phi, fwd_s, pv_k, in_the_money, underflow=True)
    try:
        s, converged = lbr._lets_be_rational_s(beta, -abs(x))
    except (ArithmeticError, ValueError):  # math over- or underflows on extreme inputs
        converged = False
    if converged:
        return _accept(s / math.sqrt(T))
    return _brent_implied_volatility(beta, -abs(x), T, tol)


def _log_moneyness(S: float, K: float, T: float, r: float, q: float) -> float:
    """``ln(F/K) = ln(S/K) + (r - q)T``, accurate in relative terms also near the forward.

    Out of the money with ``σ√T`` small, the volatility's relative error equals that of ``ln(F/K)``, so
    ``ln(S e^(-qT)/(K e^(-rT)))``, whose rounding is about ``ε`` absolute, would cost about ``ε/|ln(F/K)|`` (1e-13
    at 0.2 % from the forward). Within a factor of two ``S - K`` is exact, and ``log1p`` keeps the relative accuracy.
    Where ``S/K`` under- or overflows, ``ln S - ln K``.
    """
    return lbr._log_ratio_s(S, K) + (r - q) * T


def _forward_intrinsic(fwd_s: float, pv_k: float, x: float) -> float:
    """``S e^(-qT) - K e^(-rT) = K e^(-rT)·(e^x - 1)``, without the cancellation of the difference near the forward.

    The difference of the two rounded values is off by ``ε·S``, which near the money is a large part of a small
    out-of-the-money price (low volatility, short expiry).
    """
    return pv_k * math.expm1(x) if abs(x) < 1 else fwd_s - pv_k


def _no_time_value(
    price: float, S: float, K: float, phi: float, fwd_s: float, pv_k: float, in_the_money: bool, *, underflow: bool
) -> ImpliedVolError:
    """The :class:`ImpliedVolError` of a European quote with no time value (reason ``"no_time_value"``).

    ``underflow``: the time value is positive but its normalised value is 0. :func:`implied_volatility_batch`
    builds its messages with this function from the same floats, so the two paths say the same.
    """
    if underflow:
        return ImpliedVolError(
            f"the time value of price {price:.6g} underflows to zero once divided by √(S e^(-qT)·K e^(-rT)) = "
            f"{math.sqrt(fwd_s) * math.sqrt(pv_k):.6g}: no volatility can be identified from it",
            "no_time_value",
        )
    if in_the_money:
        lower = max(phi * (fwd_s - pv_k), 0.0)  # as _check_european_quote
        return ImpliedVolError(
            f"price {price:.6g} is within {TIME_VALUE_MARGIN * max(S, K):.2g} (1e-12·max(S, K)) of the no-arbitrage "
            f"lower bound {lower:.6g} (discounted intrinsic value), so it has no time value to identify a volatility "
            "from: its last digits would decide it",
            "no_time_value",
        )
    return ImpliedVolError(
        f"price {price:.6g} of an out-of-the-money option, which is worth only its time value, has no time value: "
        "no volatility can be identified from it",
        "no_time_value",
    )


def _accept(vol: float) -> float:
    if not vol <= MAX_VOL:
        raise ImpliedVolError(_ABOVE_MAX_VOL, "above_max_vol")
    return vol if vol >= _VOL_FLOOR else 0.0  # zero to machine precision


def _brent_implied_volatility(beta: float, x: float, T: float, tol: float) -> float:
    """The bracketing solver: Brent's method on ``b(x, σ√T) = β``, the normalised out-of-the-money price (``x ≤ 0``).

    :func:`lets_be_rational.normalised_black` stays accurate in relative terms far out of the money, where
    ``bs.price`` cancels, and is defined (without warnings) wherever ``β`` and ``x`` are.
    """
    sqrt_t = math.sqrt(T)

    def objective(vol: float) -> float:
        return float(lbr.normalised_black(x, vol * sqrt_t)) - beta

    hi = _upper_bracket(objective)
    if objective(_VOL_FLOOR) > 0.0:
        return 0.0  # the time value is below what a volatility of 1e-12 produces: zero to machine precision
    return float(brentq(objective, _VOL_FLOOR, hi, xtol=tol, rtol=4 * np.finfo(float).eps, maxiter=500))


def _upper_bracket(objective) -> float:
    """Smallest of 1, 2, 4, ..., ``MAX_VOL`` where the (increasing) objective is non-negative."""
    hi = 1.0
    while objective(hi) < 0.0:
        if hi >= MAX_VOL:
            raise ImpliedVolError(_ABOVE_MAX_VOL, "above_max_vol")
        hi = min(2.0 * hi, MAX_VOL)
    return hi


class ImpliedVolBatch(NamedTuple):
    """Result of :func:`implied_volatility_batch` with ``return_reasons=True``; each field has the input shape."""

    vol: np.ndarray
    """Implied volatility (decimal), NaN where the quote was rejected."""
    reason: np.ndarray
    """The :data:`REJECTION_REASONS` code of each rejected quote, ``""`` where it was solved."""
    message: np.ndarray
    """The :class:`ImpliedVolError` message :func:`implied_volatility` gives for the quote, ``""`` where solved."""
    vega: np.ndarray
    """Black-Scholes-Merton vega at the implied volatility, ``∂price/∂σ = S e^(-qT)·φ(d₁)·√T``, in price units per
    unit of volatility (1.00 = 100 vol points; divide by 100 for the vega per vol point); NaN where the quote was
    rejected. A price error ``δ`` moves the volatility by about ``δ/vega``, so a bid-ask spread divided by vega is
    the spread in volatility, and its inverse (or inverse square) weights the quote in a fit to implied volatilities:
    the wider the spread in volatility, the less the quote counts."""


# Index 0 is a solved quote; the others are the European rejection codes.
_CODES = (
    "",
    "not_finite",
    "not_positive",
    "expired",
    "below_lower_bound",
    "at_or_above_upper_bound",
    "above_max_vol",
    "no_time_value",
)


def _option_signs(option_type: ArrayLike) -> np.ndarray:
    """+1/-1 for each "call"/"put" (any spelling :func:`black_scholes.option_sign` accepts)."""
    kinds = np.asarray(option_type)
    if kinds.ndim == 0:
        return np.asarray(float(bs.option_sign(kinds.item())))
    unique, inverse = np.unique(kinds.astype(str), return_inverse=True)
    signs = [bs.option_sign(str(kind)) for kind in unique]  # str: the error shows 'xyz', not np.str_('xyz')
    return np.array(signs, dtype=float)[inverse].reshape(kinds.shape)


def _libm(function, values: np.ndarray) -> np.ndarray:
    """A ``math`` function element by element.

    NumPy's exp, log1p and expm1 can differ from libm's in the last bit, which would let a quote within an ulp of a
    bound, or without time value, be judged differently by the batch than by :func:`implied_volatility`.
    """
    return np.fromiter(map(function, values.tolist()), dtype=float, count=values.size)


def _log_ratio_libm(S: np.ndarray, K: np.ndarray) -> np.ndarray:
    """``lets_be_rational._log_ratio_s`` on arrays, with the same libm calls, hence the same bits."""
    ratio = S / K
    near = (ratio >= 0.5) & (ratio <= 2)
    far = ~near & (ratio > 0) & (ratio < np.inf)
    extreme = ~near & ~far  # S/K under- or overflows
    out = np.empty_like(ratio)
    out[near] = _libm(math.log1p, (S[near] - K[near]) / K[near])
    out[far] = _libm(math.log, ratio[far])
    out[extreme] = _libm(math.log, S[extreme]) - _libm(math.log, K[extreme])
    return out


# implied_volatility_batch hands a quote to implied_volatility when its normalised out-of-the-money price is within
# _NEAR_MAXIMUM (relative) of the maximum e^(-|x|/2), or its volatility above _NEAR_MAX_VOL·MAX_VOL. The margins
# are far wider than the few ulps by which NumPy's exp differs from libm's (and the cond·ε by which the two paths'
# volatilities differ there), and such quotes are rare: none of the 10,000 in benchmarks/bench_iv.py.
_NEAR_MAXIMUM = 1e-12
_NEAR_MAX_VOL = 0.99


def implied_volatility_batch(
    prices: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: ArrayLike = "call",
    *,
    return_reasons: bool = False,
) -> np.ndarray | ImpliedVolBatch:
    """Black-Scholes-Merton implied volatilities of many quotes at once.

    The inputs broadcast against each other. Each quote gets the volatility (and the checks and error) of
    :func:`implied_volatility`, computed in one vectorised Let's Be Rational pass. The bound and time-value checks
    repeat the scalar path's arithmetic exactly. :func:`implied_volatility` itself solves the rare quote whose outcome
    the vectorised pass could judge differently: where the Householder steps do not report convergence (the Brent
    fallback), the out-of-the-money price is within ``1e-12`` (relative) of its maximum, or the volatility is within
    1 % of ``MAX_VOL``. Near those edges a last-bit difference between NumPy's ``exp`` and the C library's, which the
    scalar path uses, would decide between a volatility and "exceeds 2000%".

    Parameters
    ----------
    prices : array_like
        Option prices (premiums, discounted).
    S, K : array_like
        Spot and strike.
    T : array_like
        Time to expiry in years.
    r, q : array_like
        Continuously compounded rate and dividend yield (decimals).
    option_type : str or array_like of str, default "call"
        ``"call"``/``"put"`` (or ``"c"``/``"p"``), one for all quotes or one per quote; another value raises
        ``ValueError`` as :func:`implied_volatility` does.
    return_reasons : bool, default False
        Also return, per quote, why it was rejected, and the vega at each volatility found.

    Returns
    -------
    numpy.ndarray or ImpliedVolBatch
        The implied volatilities (decimal) with NaN for rejected quotes, in the broadcast shape of the inputs; with
        ``return_reasons`` an :class:`ImpliedVolBatch` ``(vol, reason, message, vega)`` whose ``reason`` holds the
        :attr:`ImpliedVolError.reason` code, ``message`` the error message that :func:`implied_volatility`
        raises for that quote and ``vega`` the Black-Scholes-Merton vega at ``vol`` (per unit of volatility).

    Examples
    --------
    >>> prices = [bs.price(100, k, 0.5, 0.03, 0.25, 0.0, "call") for k in (80, 100, 125)]
    >>> implied_volatility_batch(prices, 100, [80, 100, 125], 0.5, 0.03).round(14)
    array([0.25, 0.25, 0.25])
    >>> result = implied_volatility_batch([5.0, 101.0], 100, 90, 1.0, 0.05, return_reasons=True)
    >>> result.vol
    array([nan, nan])
    >>> result.reason.tolist()
    ['below_lower_bound', 'at_or_above_upper_bound']
    >>> result = implied_volatility_batch([40.0, 10.0], 100, [60, 100], 1.0, 0.0, return_reasons=True)
    >>> result.reason.tolist(), result.vega.round(4)  # at the money: vega 100·φ(σ√T/2)·√T at σ = 25.13 %
    (['no_time_value', ''], array([    nan, 39.5805]))
    """
    phi = _option_signs(option_type)
    arrays = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (prices, S, K, T, r, q)), phi)
    shape = arrays[0].shape
    price, S, K, T, r, q, phi = (np.ravel(a) for a in arrays)
    vol = np.full(price.size, np.nan)
    code = np.zeros(price.size, dtype=np.int8)  # index into _CODES
    messages: dict[int, str] = {}
    # For the messages of quotes with no time value: in the money, and positive time value that β = 0 lost.
    in_the_money, underflow = np.zeros(price.size, dtype=bool), np.zeros(price.size, dtype=bool)
    log_moneyness, root_fk = np.full(price.size, np.nan), np.full(price.size, np.nan)  # for the vega

    with np.errstate(all="ignore"):
        # The scalar path's checks in its order, so that each quote gets the same (first) reason.
        finite = np.isfinite(price) & np.isfinite(S) & np.isfinite(K) & np.isfinite(T) & np.isfinite(r) & np.isfinite(q)
        code[~finite] = _CODES.index("not_finite")
        code[finite & ((S <= 0) | (K <= 0))] = _CODES.index("not_positive")
        code[(code == 0) & (T <= 0)] = _CODES.index("expired")
        rows = np.flatnonzero(code == 0)
        p, s_, k_, t_, r_, q_, ph = (a[rows] for a in (price, S, K, T, r, q, phi))
        fwd_s = s_ * _libm(_exp, -q_ * t_)  # as _discounted
        pv_k = k_ * _libm(_exp, -r_ * t_)
        overflow = ~(np.isfinite(fwd_s) & np.isfinite(pv_k))
        below = ~overflow & (p < np.maximum(ph * (fwd_s - pv_k), 0.0) - TIME_VALUE_MARGIN * np.maximum(s_, k_))
        above = ~overflow & ~below & (p >= np.where(ph > 0, fwd_s, pv_k))
        code[rows[overflow]] = _CODES.index("not_finite")
        code[rows[below]] = _CODES.index("below_lower_bound")
        code[rows[above]] = _CODES.index("at_or_above_upper_bound")

        keep = ~overflow & ~below & ~above
        rows, p, s_, k_, t_, r_, q_, ph = (a[keep] for a in (rows, p, s_, k_, t_, r_, q_, ph))
        fwd_s, pv_k = fwd_s[keep], pv_k[keep]
        x = _log_ratio_libm(s_, k_) + (r_ - q_) * t_  # as _log_moneyness
        itm = ph * x > 0
        intrinsic = fwd_s - pv_k  # as _forward_intrinsic
        small = np.abs(x) < 1
        intrinsic[small] = pv_k[small] * _libm(math.expm1, x[small])
        otm_price = np.where(itm, p - ph * intrinsic, p)
        root = np.sqrt(fwd_s) * np.sqrt(pv_k)
        beta = otm_price / root
        # As implied_volatility: in the money no more than the margin of time value, out of the money none (or none
        # left once divided by √(FK)).
        flat = otm_price <= np.where(itm, TIME_VALUE_MARGIN * np.maximum(s_, k_), 0.0)
        no_time_value = flat | (beta == 0)
        code[rows[no_time_value]] = _CODES.index("no_time_value")
        in_the_money[rows], underflow[rows] = itm, ~flat & (beta == 0)
        live = ~no_time_value
        rows, t_, x, root, beta = rows[live], t_[live], x[live], root[live], beta[live]
        log_moneyness[rows], root_fk[rows] = x, root
        ax = np.abs(x)
        s, converged = lbr._lets_be_rational_v(beta, -ax)
        sigma = s / np.sqrt(t_)
        # implied_volatility decides the rows whose outcome it could see differently (see the docstring).
        scalar = ~converged | (beta >= (1 - _NEAR_MAXIMUM) * np.exp(-0.5 * ax)) | ~(sigma <= _NEAR_MAX_VOL * MAX_VOL)
        solved = ~scalar
        vol[rows[solved]] = np.where(sigma[solved] >= _VOL_FLOOR, sigma[solved], 0.0)

    for i in rows[scalar].tolist():  # the scalar path, with its Brent fallback
        try:
            vol[i] = implied_volatility(price[i], S[i], K[i], T[i], r[i], q[i], "call" if phi[i] > 0 else "put")
        except ImpliedVolError as error:
            vol[i] = np.nan
            code[i] = _CODES.index(error.reason)
            messages[i] = str(error)

    vol_shaped = vol.reshape(shape)
    if not return_reasons:
        return vol_shaped
    message = np.full(price.size, "", dtype=object)
    for i in np.flatnonzero(code).tolist():
        if i in messages:  # from implied_volatility itself
            message[i] = messages[i]
        elif _CODES[code[i]] == "no_time_value":  # the same floats as implied_volatility, hence the same message
            quote, spot, strike, sign = float(price[i]), float(S[i]), float(K[i]), int(phi[i])
            discounted = _discounted(spot, strike, float(T[i]), float(r[i]), float(q[i]))
            itm_i, underflow_i = bool(in_the_money[i]), bool(underflow[i])
            message[i] = str(_no_time_value(quote, spot, strike, sign, *discounted, itm_i, underflow=underflow_i))
        else:
            try:  # a bound check: the same checks on the same floats as implied_volatility, hence the same message
                _check_european_quote(price[i], S[i], K[i], T[i], r[i], q[i], int(phi[i]))
            except ImpliedVolError as error:
                message[i] = str(error)
    # Vega S e^(-qT)·φ(d₁)·√T = √(S e^(-qT)·K e^(-rT))·b'(x, σ√T)·√T, b' the normalised vega (the same for calls and
    # puts), at the volatility found; NaN stays NaN where there is none.
    vega = np.full(price.size, np.nan)
    found = np.flatnonzero(~np.isnan(vol))
    with np.errstate(all="ignore"):
        sqrt_t = np.sqrt(T[found])
        vega[found] = root_fk[found] * sqrt_t * lbr.normalised_vega(log_moneyness[found], vol[found] * sqrt_t)
    reason = np.asarray(_CODES, dtype=object)[code]
    return ImpliedVolBatch(vol_shaped, reason.reshape(shape), message.reshape(shape), vega.reshape(shape))


def american_bounds(S: float, K: float, T: float, r: float, q: float = 0.0, option_type: str = "call"):
    """Bounds ``(lower, upper)`` for an American option.

    Lower: immediate exercise and the European floor. Upper: :func:`binomial.american_upper_bound`, the most
    exercise can be worth, ``K max(1, e^(-rT))`` for a put and ``S max(1, e^(-qT))`` for a call (more than
    the strike or the spot when the rate or the dividend yield is negative).
    """
    phi = bs.option_sign(option_type)
    euro_lower, _ = no_arbitrage_bounds(S, K, T, r, q, option_type)
    return max(phi * (S - K), euro_lower, 0.0), binomial.american_upper_bound(S, K, T, r, q, option_type)


def lowest_tree_volatility(T: float, r: float, q: float = 0.0, steps: int = 501, method: str = "lr") -> float:
    """Lowest volatility the American solver uses on the given tree.

    A Cox-Ross-Rubinstein tree is arbitrage-free only when ``sigma sqrt(dt) > |r - q| dt``, so it cannot
    represent volatilities below ``|r - q| sqrt(dt)``; Leisen-Reimer works down to :data:`MIN_VOL`.
    """
    if str(method).lower() != "crr" or T <= 0:
        return MIN_VOL
    return max(MIN_VOL, abs(r - q) * math.sqrt(T / max(int(steps), 1)) * (1.0 + 1e-6))


def implied_volatility_american(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = "call",
    *,
    steps: int = 501,
    method: str = "lr",
    tol: float = 1e-8,
) -> float:
    """Volatility that reproduces an American ``price`` on a binomial tree (Leisen-Reimer by default).

    The American value is increasing in volatility, so Brent's method on a
    bracket is again guaranteed to converge. Raises :class:`ImpliedVolError`
    for quotes outside the American bounds or below the zero-volatility value
    (which already includes the value of optimal early exercise). On a CRR
    tree the search starts at :func:`lowest_tree_volatility`.

    A quote no more than ``1e-12·max(S, K)`` above the tree's value at :data:`MIN_VOL` (the zero-volatility value,
    for example the value of exercising at once), or up to ``1e-10·max(S, K)`` below it (the tree's own error
    there), has no time value (reason ``"no_time_value"``), as in :func:`implied_volatility`: every volatility up
    to some level gives a value that close to it (deep in the money exactly that value, at every volatility at
    which exercising at once stays optimal), so the quote cannot single one out. The margin applies out of the
    money too, where the tree is accurate only in absolute terms (see :func:`implied_volatility_alo`).

    A CRR tree cannot represent volatilities that low, so there the quote is also compared with the exact
    zero-volatility value (the engine's, as in :func:`implied_volatility_alo`). As on the Leisen-Reimer tree, the
    quote has no time value if it is no more than the margin above that value, or if the tree's value at its lowest
    volatility is that value to the tree's error, ``1e-10·max(S, K)`` (deep in the money, where exercising at once
    stays optimal up to there). Any other quote not above the value at the tree's lowest volatility (to the same
    margin) has its volatility somewhere below what the tree represents: it is rejected with the reason
    ``"below_tree_floor"``.
    """
    price, S, K, T, r, q = _check_american_quote(price, S, K, T, r, q, option_type)
    scale = max(S, K)

    def objective(vol: float) -> float:
        return binomial.price(S, K, T, r, vol, q, option_type, steps, american=True, method=method) - price

    lowest = lowest_tree_volatility(T, r, q, steps, method)
    floor = objective(lowest)
    if floor >= -TIME_VALUE_MARGIN * scale:  # no more than the margin above the value at the lowest volatility
        zero = price + floor  # the zero-volatility value, to the tree's error, on a tree that reaches MIN_VOL
        if lowest > MIN_VOL:
            # A CRR tree's floor. The engine's exact zero-volatility value takes about 0.1 ms, only for quotes this low.
            exact = float(american.price(S, K, T, r, 0.0, q, option_type=option_type))
            if zero > exact + 1e-10 * scale:  # the floor has time value
                if price > exact + TIME_VALUE_MARGIN * scale:  # the volatility is somewhere below the floor
                    where = "below" if floor > 0.0 else f"no more than {TIME_VALUE_MARGIN * scale:.2g} above"
                    raise ImpliedVolError(
                        f"price {price:.6g} is {where} {zero:.6g}, the value at {lowest:.4%} volatility, the lowest "
                        f"a {steps}-step CRR tree can represent; use more steps or the Leisen-Reimer tree",
                        "below_tree_floor",
                    )
                zero = exact
        if price >= zero - 1e-10 * scale:
            raise _american_no_time_value(price, zero, S, K, T, r, q, option_type)
        raise _below_zero_volatility(price, zero)
    hi = _upper_bracket(objective)
    return float(brentq(objective, lowest, hi, xtol=tol, maxiter=200))


def implied_volatility_alo(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = "call",
    *,
    scheme: str = "accurate",
    tol: float = 1e-10,
) -> float:
    """Volatility that reproduces an American ``price`` with the Andersen-Lake-Offengenden engine
    (:func:`american.price` with the given ``scheme``).

    The quote is checked against the American bounds first, with the messages of
    :func:`implied_volatility_american`. The engine values zero volatility exactly (the best deterministic
    exercise date), so the bracket starts at 0. A quote no more than ``1e-12·max(S, K)`` above that value, or up to
    ``1e-10·max(S, K)`` below it, has no time value (reason ``"no_time_value"``, with the tree solver's message),
    and one further below is rejected. Brent's method then solves on ``[0, hi]``, ``hi`` the first of 1, 2, 4, ...
    up to :data:`MAX_VOL` where the value reaches the quote.

    Unlike :func:`implied_volatility`, which inverts a tiny out-of-the-money European quote to full relative
    precision, both American solvers apply the margin out of the money too: near zero volatility their engines are
    accurate only in absolute terms, far more coarsely than a quote's rounding. This engine values a total
    volatility ``σ√T`` below 1e-8 at zero volatility (an error up to about 4e-9·S), and the 501-step tree values a
    one-year at-the-money put (rate 5 %) at 0.046 % volatility at 1.3e-13 where this engine gives 7.8e-5. Just above
    the margin a volatility is only as good as the engine there: that put quoted 1e-9 above its zero-volatility
    value of 0 gives about 0.0005 % here (the digits after that move with the NumPy and SciPy versions) and 0.06 %
    on the tree.

    Examples
    --------
    >>> from options_analyzer import american
    >>> quote = american.price(100, 100, 1.0, 0.05, 0.2, 0.0, option_type="put")
    >>> round(implied_volatility_alo(quote, 100, 100, 1.0, 0.05, 0.0, "put"), 10)
    0.2
    """
    price, S, K, T, r, q = _check_american_quote(price, S, K, T, r, q, option_type)

    def objective(vol: float) -> float:
        return float(american.price(S, K, T, r, vol, q, option_type=option_type, scheme=scheme)) - price

    floor = objective(0.0)
    if floor >= -TIME_VALUE_MARGIN * max(S, K):
        if floor <= 1e-10 * max(S, K):
            raise _american_no_time_value(price, price + floor, S, K, T, r, q, option_type)  # as on the tree
        raise _below_zero_volatility(price, price + floor)
    hi = _upper_bracket(objective)
    return float(brentq(objective, 0.0, hi, xtol=tol, rtol=4 * np.finfo(float).eps, maxiter=200))


def _check_american_quote(price, S, K, T, r, q, option_type: str) -> tuple[float, float, float, float, float, float]:
    """Raise the :class:`ImpliedVolError` of a quote outside the American bounds (:func:`american_bounds`); return
    the inputs as floats."""
    price, S, K, T, r, q = _check_contract(price, S, K, T, r, q)
    lower, upper = american_bounds(S, K, T, r, q, option_type)
    if price < lower - TIME_VALUE_MARGIN * max(S, K):
        raise ImpliedVolError(
            f"price {price:.6g} is below the American lower bound {lower:.6g} (immediate exercise value); "
            "no volatility can produce it",
            "below_lower_bound",
        )
    if price >= upper:
        raise ImpliedVolError(
            f"price {price:.6g} is at or above the American upper bound {upper:.6g}", "at_or_above_upper_bound"
        )
    return price, S, K, T, r, q


def _american_no_time_value(
    price: float, zero: float, S: float, K: float, T: float, r: float, q: float, option_type: str
) -> ImpliedVolError:
    """The :class:`ImpliedVolError` of an American quote no more than the margin above ``zero``, the option's value at
    zero volatility (reason ``"no_time_value"``)."""
    margin = f"{TIME_VALUE_MARGIN * max(S, K):.2g} (1e-12·max(S, K))"
    if american_bounds(S, K, T, r, q, option_type)[0] == 0.0:  # worth 0 at zero volatility (out of the money)
        # The value rises from 0 at any positive volatility, but the engines do not resolve it that close to 0.
        return ImpliedVolError(
            f"price {price:.6g} is no more than {margin} above 0, the option's value at zero volatility, and the "
            "American engines are not that accurate near zero volatility, so no volatility can be identified from it",
            "no_time_value",
        )
    return ImpliedVolError(
        f"price {price:.6g} is no more than {margin} above the zero-volatility American value {zero:.6g}, which "
        "every volatility up to some level gives to within that margin, so it has no time value to identify a "
        "volatility from",
        "no_time_value",
    )


def _below_zero_volatility(price: float, floor: float) -> ImpliedVolError:
    return ImpliedVolError(
        f"price {price:.6g} is below the zero-volatility American value {floor:.6g}; no volatility can produce it",
        "below_lower_bound",
    )
