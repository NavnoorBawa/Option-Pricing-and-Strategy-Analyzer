"""Robust Black-Scholes-Merton implied volatility.

The European solver

1. checks the quote against the model-free no-arbitrage bounds (a price
   outside them has *no* implied volatility, and saying so is better than
   returning a meaningless number);
2. converts in-the-money quotes to the equivalent out-of-the-money option via
   put-call parity, which carries the same time value but is far better
   conditioned numerically;
3. inverts the normalised out-of-the-money price with Peter Jäckel's "Let's Be
   Rational" (:mod:`options_analyzer.lets_be_rational`), which reaches machine
   precision in two Householder steps, and falls back to Brent's method on a
   bracket of the same normalised price (guaranteed to converge once a bracket
   exists) for any quote where those steps do not report convergence.

:func:`implied_volatility_batch` does the same for arrays of quotes in one
vectorised pass, with NaN and a reason for each rejected quote.

:func:`implied_volatility_american` inverts a binomial tree (Leisen-Reimer by
default) for American quotes (listed single-stock options), with American
bounds; :func:`implied_volatility_alo` inverts the Andersen-Lake-Offengenden
engine (:mod:`options_analyzer.american`) with the same bounds.
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

#: The codes of :attr:`ImpliedVolError.reason`, with what each means.
REJECTION_REASONS = {
    "not_finite": "an input, or the discounted spot S e^(-qT) or strike K e^(-rT), is not a finite number",
    "not_positive": "the spot or the strike is not positive",
    "expired": "the option has expired",
    "below_lower_bound": "the price is below the no-arbitrage lower bound",
    "at_or_above_upper_bound": "the price is at or above the no-arbitrage upper bound",
    "above_max_vol": f"the implied volatility exceeds {MAX_VOL:.0%}",
    "below_tree_floor": "the price is below the value at the lowest volatility the binomial tree can represent",
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
    if price < lower - 1e-12 * max(S, K):
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
    ``MAX_VOL`` (2,000 %), and when the discounted spot ``S e^(-qT)`` or
    strike ``K e^(-rT)`` overflows (reason ``"not_finite"``). The result
    is accurate to machine precision (Let's Be Rational); ``tol`` is the
    volatility tolerance of the Brent fallback, used only where Let's Be
    Rational does not report convergence.

    Examples
    --------
    >>> price = bs.price(100, 110, 0.5, 0.03, 0.25, 0.0, "call")
    >>> round(implied_volatility(price, 100, 110, 0.5, 0.03, 0.0, "call"), 14)
    0.25
    """
    phi = bs.option_sign(option_type)
    price, S, K, T, r, q, fwd_s, pv_k = _check_european_quote(price, S, K, T, r, q, phi)

    # Work with the out-of-the-money side: same time value, better conditioning.
    x = _log_moneyness(S, K, T, r, q)
    otm_phi = -1 if x >= 0 else 1
    otm_price = price if otm_phi == phi else price - phi * _forward_intrinsic(fwd_s, pv_k, x)
    if otm_price <= 0.0:
        return 0.0  # no time value: the quote sits exactly on the lower bound
    # The undiscounted out-of-the-money price over √(FK); with time value left, both forwards are positive (and finite).
    beta = otm_price / (math.sqrt(fwd_s) * math.sqrt(pv_k))
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


# Index 0 is a solved quote; the others are the European rejection codes.
_CODES = ("", "not_finite", "not_positive", "expired", "below_lower_bound", "at_or_above_upper_bound", "above_max_vol")


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
    :func:`implied_volatility`, computed in one vectorised Let's Be Rational pass. The bound checks repeat the scalar
    path's arithmetic exactly. :func:`implied_volatility` itself solves the rare quote whose outcome the vectorised
    pass could judge differently: where the Householder steps do not report convergence (the Brent fallback), the
    out-of-the-money price is within ``1e-12`` (relative) of its maximum, or the volatility is within 1 % of
    ``MAX_VOL``. Near those edges a last-bit difference between NumPy's ``exp`` and the C library's, which the
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
        Also return, per quote, why it was rejected.

    Returns
    -------
    numpy.ndarray or ImpliedVolBatch
        The implied volatilities (decimal) with NaN for rejected quotes, in the broadcast shape of the inputs; with
        ``return_reasons`` an :class:`ImpliedVolBatch` ``(vol, reason, message)`` whose ``reason`` holds the
        :attr:`ImpliedVolError.reason` code and ``message`` the error message that :func:`implied_volatility`
        raises for that quote.

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
    """
    phi = _option_signs(option_type)
    arrays = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (prices, S, K, T, r, q)), phi)
    shape = arrays[0].shape
    price, S, K, T, r, q, phi = (np.ravel(a) for a in arrays)
    vol = np.full(price.size, np.nan)
    code = np.zeros(price.size, dtype=np.int8)  # index into _CODES
    messages: dict[int, str] = {}

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
        below = ~overflow & (p < np.maximum(ph * (fwd_s - pv_k), 0.0) - 1e-12 * np.maximum(s_, k_))
        above = ~overflow & ~below & (p >= np.where(ph > 0, fwd_s, pv_k))
        code[rows[overflow]] = _CODES.index("not_finite")
        code[rows[below]] = _CODES.index("below_lower_bound")
        code[rows[above]] = _CODES.index("at_or_above_upper_bound")

        keep = ~overflow & ~below & ~above
        rows, p, s_, k_, t_, r_, q_, ph = (a[keep] for a in (rows, p, s_, k_, t_, r_, q_, ph))
        fwd_s, pv_k = fwd_s[keep], pv_k[keep]
        x = _log_ratio_libm(s_, k_) + (r_ - q_) * t_  # as _log_moneyness
        intrinsic = fwd_s - pv_k  # as _forward_intrinsic
        small = np.abs(x) < 1
        intrinsic[small] = pv_k[small] * _libm(math.expm1, x[small])
        otm_phi = np.where(x >= 0, -1.0, 1.0)
        otm_price = np.where(otm_phi == ph, p, p - ph * intrinsic)
        vol[rows] = 0.0  # no time value; overwritten below where there is some
        live = otm_price > 0
        rows, t_, ax = rows[live], t_[live], np.abs(x[live])
        beta = otm_price[live] / (np.sqrt(fwd_s[live]) * np.sqrt(pv_k[live]))
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
            continue
        try:  # a bound check: the same checks on the same floats as implied_volatility, hence the same message
            _check_european_quote(price[i], S[i], K[i], T[i], r[i], q[i], int(phi[i]))
        except ImpliedVolError as error:
            message[i] = str(error)
    reason = np.asarray(_CODES, dtype=object)[code]
    return ImpliedVolBatch(vol_shaped, reason.reshape(shape), message.reshape(shape))


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
    """
    price, S, K, T, r, q = _check_american_quote(price, S, K, T, r, q, option_type)
    scale = max(S, K)

    def objective(vol: float) -> float:
        return binomial.price(S, K, T, r, vol, q, option_type, steps, american=True, method=method) - price

    lowest = lowest_tree_volatility(T, r, q, steps, method)
    floor = objective(lowest)
    if floor >= 0.0:
        if floor <= 1e-10 * scale:
            # The quote equals the zero-volatility value (e.g. immediate exercise is optimal): every
            # volatility up to some level reproduces it, so report zero like the European solver does.
            return 0.0
        if lowest > MIN_VOL:
            raise ImpliedVolError(
                f"price {price:.6g} is below {price + floor:.6g}, the value at {lowest:.4%} volatility, the lowest "
                f"a {steps}-step CRR tree can represent; use more steps or the Leisen-Reimer tree",
                "below_tree_floor",
            )
        raise _below_zero_volatility(price, price + floor)
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
    exercise date), so the bracket starts at 0; a quote at that value is reported as zero volatility, and one
    below it is rejected. Brent's method then solves on ``[0, hi]``, ``hi`` the first of 1, 2, 4, ... up to
    :data:`MAX_VOL` where the value reaches the quote.

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
    if floor >= 0.0:
        if floor <= 1e-10 * max(S, K):
            return 0.0  # the zero-volatility value, as in implied_volatility_american
        raise _below_zero_volatility(price, price + floor)
    hi = _upper_bracket(objective)
    return float(brentq(objective, 0.0, hi, xtol=tol, rtol=4 * np.finfo(float).eps, maxiter=200))


def _check_american_quote(price, S, K, T, r, q, option_type: str) -> tuple[float, float, float, float, float, float]:
    """Raise the :class:`ImpliedVolError` of a quote outside the American bounds (:func:`american_bounds`); return
    the inputs as floats."""
    price, S, K, T, r, q = _check_contract(price, S, K, T, r, q)
    lower, upper = american_bounds(S, K, T, r, q, option_type)
    if price < lower - 1e-12 * max(S, K):
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


def _below_zero_volatility(price: float, floor: float) -> ImpliedVolError:
    return ImpliedVolError(
        f"price {price:.6g} is below the zero-volatility American value {floor:.6g}; no volatility can produce it",
        "below_lower_bound",
    )
