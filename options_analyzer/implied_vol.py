"""Robust Black-Scholes-Merton implied volatility.

The solver

1. checks the quote against the model-free no-arbitrage bounds (a price
   outside them has *no* implied volatility, and saying so is better than
   returning a meaningless number);
2. converts in-the-money quotes to the equivalent out-of-the-money option via
   put-call parity, which carries the same time value but is far better
   conditioned numerically;
3. brackets the root and solves with Brent's method, which is guaranteed to
   converge once a bracket exists (unlike an undamped Newton iteration).

:func:`implied_volatility_american` inverts the Leisen-Reimer binomial tree
for American quotes (listed single-stock options), with American bounds.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from . import binomial
from . import black_scholes as bs

__all__ = [
    "ImpliedVolError",
    "american_bounds",
    "implied_volatility",
    "implied_volatility_american",
    "no_arbitrage_bounds",
]

MIN_VOL = 1e-6  # lower bracket of the American solver (the tree needs a positive volatility)
MAX_VOL = 20.0
_VOL_FLOOR = 1e-12  # lower bracket of the European solver


class ImpliedVolError(ValueError):
    """Raised when a quote admits no implied volatility."""


def no_arbitrage_bounds(S: float, K: float, T: float, r: float, q: float = 0.0, option_type: str = "call"):
    """Model-free bounds ``(lower, upper)`` for a European option price."""
    phi = bs.option_sign(option_type)
    fwd_s = S * math.exp(-q * T)
    pv_k = K * math.exp(-r * T)
    lower = max(phi * (fwd_s - pv_k), 0.0)
    upper = fwd_s if phi > 0 else pv_k
    return lower, upper


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
    ``MAX_VOL`` (2,000 %).
    """
    phi = bs.option_sign(option_type)
    for name, value in (("price", price), ("spot", S), ("strike", K), ("time to expiry", T)):
        if not np.isfinite(value):
            raise ImpliedVolError(f"{name} must be a finite number")
    if S <= 0 or K <= 0:
        raise ImpliedVolError("spot and strike must be positive")
    if T <= 0:
        raise ImpliedVolError("the option has expired: implied volatility is undefined at T = 0")

    lower, upper = no_arbitrage_bounds(S, K, T, r, q, option_type)
    scale = max(S, K)
    if price < lower - 1e-12 * scale:
        raise ImpliedVolError(
            f"price {price:.6g} is below the no-arbitrage lower bound {lower:.6g} "
            "(discounted intrinsic value); no volatility can produce it"
        )
    if price >= upper:
        raise ImpliedVolError(
            f"price {price:.6g} is at or above the no-arbitrage upper bound {upper:.6g}; "
            "no finite volatility can produce it"
        )

    # Work with the out-of-the-money side: same time value, better conditioning.
    fwd_s = S * math.exp(-q * T)
    pv_k = K * math.exp(-r * T)
    otm_type = "put" if fwd_s >= pv_k else "call"
    otm_price = price if bs.option_sign(otm_type) == phi else price - phi * (fwd_s - pv_k)
    if otm_price <= 0.0:
        return 0.0  # no time value: the quote sits exactly on the lower bound

    def objective(vol: float) -> float:
        return bs.price(S, K, T, r, vol, q, otm_type) - otm_price

    hi = _upper_bracket(objective)
    if objective(_VOL_FLOOR) > 0.0:
        return 0.0  # the time value is below what a volatility of 1e-12 produces: zero to machine precision
    return float(brentq(objective, _VOL_FLOOR, hi, xtol=tol, rtol=4 * np.finfo(float).eps, maxiter=500))


def _upper_bracket(objective) -> float:
    """Smallest of 1, 2, 4, ..., ``MAX_VOL`` where the (increasing) objective is non-negative."""
    hi = 1.0
    while objective(hi) < 0.0:
        if hi >= MAX_VOL:
            raise ImpliedVolError(f"implied volatility exceeds {MAX_VOL:.0%}; check the quote and inputs")
        hi = min(2.0 * hi, MAX_VOL)
    return hi


def american_bounds(S: float, K: float, T: float, r: float, q: float = 0.0, option_type: str = "call"):
    """Bounds ``(lower, upper)`` for an American option: at least immediate exercise and the European floor."""
    phi = bs.option_sign(option_type)
    euro_lower, _ = no_arbitrage_bounds(S, K, T, r, q, option_type)
    return max(phi * (S - K), euro_lower, 0.0), (S if phi > 0 else K)


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
    tol: float = 1e-8,
) -> float:
    """Volatility that reproduces an American ``price`` on a Leisen-Reimer tree.

    The American value is increasing in volatility, so Brent's method on a
    bracket is again guaranteed to converge. Raises :class:`ImpliedVolError`
    for quotes outside the American bounds or below the zero-volatility value
    (which already includes the value of optimal early exercise).
    """
    for name, value in (("price", price), ("spot", S), ("strike", K), ("time to expiry", T)):
        if not np.isfinite(value):
            raise ImpliedVolError(f"{name} must be a finite number")
    if S <= 0 or K <= 0:
        raise ImpliedVolError("spot and strike must be positive")
    if T <= 0:
        raise ImpliedVolError("the option has expired: implied volatility is undefined at T = 0")
    lower, upper = american_bounds(S, K, T, r, q, option_type)
    scale = max(S, K)
    if price < lower - 1e-12 * scale:
        raise ImpliedVolError(
            f"price {price:.6g} is below the American lower bound {lower:.6g} (immediate exercise value); "
            "no volatility can produce it"
        )
    if price >= upper:
        raise ImpliedVolError(f"price {price:.6g} is at or above the American upper bound {upper:.6g}")

    def objective(vol: float) -> float:
        return binomial.price(S, K, T, r, vol, q, option_type, steps, american=True, method="lr") - price

    floor = objective(MIN_VOL)
    if floor >= 0.0:
        if floor <= 1e-10 * scale:
            # The quote equals the zero-volatility value (e.g. immediate exercise is optimal): every
            # volatility up to some level reproduces it, so report zero like the European solver does.
            return 0.0
        raise ImpliedVolError(
            f"price {price:.6g} is below the zero-volatility American value {price + floor:.6g}; "
            "no volatility can produce it"
        )
    hi = _upper_bracket(objective)
    return float(brentq(objective, MIN_VOL, hi, xtol=tol, maxiter=200))
