"""Black-Scholes-Merton pricing and analytical Greeks for European options.

Conventions
-----------
* ``S`` spot, ``K`` strike, ``T`` time to expiry in years, ``r`` continuously
  compounded risk-free rate, ``sigma`` annualised volatility and ``q``
  continuous dividend yield. Rates and volatilities are decimals (0.05 = 5 %).
* Greeks are returned in *raw* model units, i.e. the exact partial
  derivatives: vega per 1.00 of volatility, rho per 1.00 of rate, theta per
  year. :data:`GREEKS` records the conventional display scaling (theta per
  calendar day, vega per volatility point, rho per 1 % rate move, ...).
* Time-decay Greeks (theta, charm, veta, color) are derivatives with respect to
  the *passage of calendar time*, i.e. ``-d/dT``, which is the market
  convention (a long option has negative theta).
* Every function broadcasts over NumPy arrays and returns a Python ``float``
  when all inputs are scalars.

Degenerate inputs are handled with their exact limits instead of producing
NaNs: at ``T <= 0`` the option is worth its intrinsic value, and with zero
volatility (or zero spot) the value is the discounted forward intrinsic value
``max(phi * (S e^{-qT} - K e^{-rT}), 0)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr

__all__ = [
    "GREEKS",
    "GreekInfo",
    "greeks",
    "option_sign",
    "price",
]

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _npdf(x: np.ndarray) -> np.ndarray:
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


def option_sign(option_type: str) -> int:
    """Return ``+1`` for a call and ``-1`` for a put (case-insensitive)."""
    kind = str(option_type).strip().lower()
    if kind in ("call", "c"):
        return 1
    if kind in ("put", "p"):
        return -1
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def _prepare(S, K, T, r, sigma, q):
    arrays = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q)))
    S, K, T, r, sigma, q = arrays
    if np.any(np.isnan(S) | np.isnan(K) | np.isnan(T) | np.isnan(r) | np.isnan(sigma) | np.isnan(q)):
        raise ValueError("inputs must not be NaN")
    if np.any(S < 0) or np.any(K < 0):
        raise ValueError("spot and strike must be non-negative")
    if np.any(sigma < 0):
        raise ValueError("volatility must be non-negative")
    return S, K, T, r, sigma, q


def _scalar_or_array(x: np.ndarray):
    return float(x) if np.ndim(x) == 0 else x


def price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: str = "call",
):
    """Black-Scholes-Merton value of a European call or put.

    Examples
    --------
    Hull, *Options, Futures and Other Derivatives*, Example 15.6:

    >>> round(price(42, 40, 0.5, 0.10, 0.20, option_type="call"), 2)
    4.76
    >>> round(price(42, 40, 0.5, 0.10, 0.20, option_type="put"), 2)
    0.81
    """
    phi = option_sign(option_type)
    S, K, T, r, sigma, q = _prepare(S, K, T, r, sigma, q)
    tau = np.maximum(T, 0.0)
    df_r = np.exp(-r * tau)
    df_q = np.exp(-q * tau)
    deterministic = np.maximum(phi * (S * df_q - K * df_r), 0.0)

    vol_sqrt_t = sigma * np.sqrt(tau)
    live = (vol_sqrt_t > 0) & (S > 0) & (K > 0)
    if not np.any(live):
        return _scalar_or_array(deterministic)

    s = np.where(live, S, 1.0)
    k = np.where(live, K, 1.0)
    v = np.where(live, vol_sqrt_t, 1.0)
    d1 = (np.log(s / k) + (r - q) * tau) / v + 0.5 * v
    d2 = d1 - v
    value = phi * (s * df_q * ndtr(phi * d1) - k * df_r * ndtr(phi * d2))
    # Cancellation can leave values a hair below the no-arbitrage floor.
    value = np.maximum(value, deterministic)
    return _scalar_or_array(np.where(live, value, deterministic))


GREEK_ORDER = (
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "epsilon",
    "vanna",
    "charm",
    "vomma",
    "veta",
    "speed",
    "zomma",
    "color",
    "ultima",
)


def greeks(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: str = "call",
) -> dict:
    """Analytical Black-Scholes-Merton Greeks up to third order.

    Returns a dict keyed by the names in :data:`GREEKS` (raw model units, see
    the module docstring). Formulas follow Haug, *The Complete Guide to Option
    Pricing Formulas* (2nd ed.) with every time derivative taken with respect
    to calendar time (note that the veta and color formulas printed on
    Wikipedia are derivatives with respect to time-to-expiry, i.e. have the
    opposite sign). All Greeks are verified against high-precision finite
    differences in the test-suite.

    In the degenerate cases (expired, zero volatility or zero spot) the value
    is a deterministic function of the inputs; its first-order sensitivities
    are returned and all curvature Greeks are zero.
    """
    phi = option_sign(option_type)
    S, K, T, r, sigma, q = _prepare(S, K, T, r, sigma, q)
    tau = np.maximum(T, 0.0)
    df_r = np.exp(-r * tau)
    df_q = np.exp(-q * tau)
    vol_sqrt_t = sigma * np.sqrt(tau)
    live = (vol_sqrt_t > 0) & (S > 0) & (K > 0)

    # Deterministic limit: V = max(phi (S e^{-q tau} - K e^{-r tau}), 0). Exactly at the kink the
    # derivative is undefined; use the average of the one-sided derivatives (e.g. delta = +/-1/2),
    # which keeps put-call parity (delta_call - delta_put = e^{-q tau}) intact.
    moneyness = phi * (S * df_q - K * df_r)
    itm = np.where(moneyness > 0, 1.0, np.where(moneyness == 0, 0.5, 0.0))
    det = {name: np.zeros_like(S) for name in GREEK_ORDER}
    det["delta"] = phi * df_q * itm
    det["theta"] = phi * (q * S * df_q - r * K * df_r) * itm * (tau > 0)
    det["rho"] = phi * K * tau * df_r * itm
    det["epsilon"] = -phi * S * tau * df_q * itm

    if not np.any(live):
        return {name: _scalar_or_array(det[name]) for name in GREEK_ORDER}

    s = np.where(live, S, 1.0)
    k = np.where(live, K, 1.0)
    t = np.where(live, tau, 1.0)
    sig = np.where(live, sigma, 1.0)
    v = np.where(live, vol_sqrt_t, 1.0)
    sqrt_t = np.sqrt(t)
    b = r - q  # cost of carry

    d1 = (np.log(s / k) + b * t) / v + 0.5 * v
    d2 = d1 - v
    n1 = _npdf(d1)
    cdf1 = ndtr(phi * d1)
    cdf2 = ndtr(phi * d2)

    gamma = df_q * n1 / (s * v)
    vega = s * df_q * n1 * sqrt_t
    out = {
        "delta": phi * df_q * cdf1,
        "gamma": gamma,
        "theta": (-s * df_q * n1 * sig / (2.0 * sqrt_t) - phi * r * k * df_r * cdf2 + phi * q * s * df_q * cdf1),
        "vega": vega,
        "rho": phi * k * t * df_r * cdf2,
        "epsilon": -phi * s * t * df_q * cdf1,
        "vanna": -df_q * n1 * d2 / sig,
        "charm": phi * q * df_q * cdf1 - df_q * n1 * (2.0 * b * t - d2 * v) / (2.0 * t * v),
        "vomma": vega * d1 * d2 / sig,
        "veta": s * df_q * n1 * sqrt_t * (q + b * d1 / v - (1.0 + d1 * d2) / (2.0 * t)),
        "speed": -gamma / s * (d1 / v + 1.0),
        "zomma": gamma * (d1 * d2 - 1.0) / sig,
        "color": df_q * n1 / (2.0 * s * t * v) * (2.0 * q * t + 1.0 + (2.0 * b * t - d2 * v) * d1 / v),
        "ultima": -vega / sig**2 * (d1 * d2 * (1.0 - d1 * d2) + d1**2 + d2**2),
    }
    return {name: _scalar_or_array(np.where(live, out[name], det[name])) for name in GREEK_ORDER}


@dataclass(frozen=True)
class GreekInfo:
    """Display metadata for a Greek."""

    symbol: str
    label: str
    scale: float
    unit: str
    description: str
    order: int


#: Display metadata. Multiply a raw Greek by ``scale`` to express it in ``unit``.
GREEKS: dict[str, GreekInfo] = {
    "delta": GreekInfo("Δ", "Delta", 1.0, "per $1 move in spot", "∂V/∂S", 1),
    "gamma": GreekInfo("Γ", "Gamma", 1.0, "Δ per $1 move in spot", "∂²V/∂S²", 1),
    "theta": GreekInfo("Θ", "Theta", 1.0 / 365.0, "per calendar day", "∂V/∂t", 1),
    "vega": GreekInfo("ν", "Vega", 0.01, "per 1 vol point", "∂V/∂σ", 1),
    "rho": GreekInfo("ρ", "Rho", 0.01, "per 1% rate move", "∂V/∂r", 1),
    "epsilon": GreekInfo("ε", "Epsilon (dividend rho)", 0.01, "per 1% dividend-yield move", "∂V/∂q", 1),
    "vanna": GreekInfo("", "Vanna", 0.01, "Δ per 1 vol point", "∂²V/∂S∂σ", 2),
    "charm": GreekInfo("", "Charm", 1.0 / 365.0, "Δ per calendar day", "∂Δ/∂t", 2),
    "vomma": GreekInfo("", "Vomma (volga)", 1e-4, "vega per 1 vol point", "∂²V/∂σ²", 2),
    "veta": GreekInfo("", "Veta", 0.01 / 365.0, "vega per calendar day", "∂²V/∂σ∂t", 2),
    "speed": GreekInfo("", "Speed", 1.0, "Γ per $1 move in spot", "∂³V/∂S³", 3),
    "zomma": GreekInfo("", "Zomma", 0.01, "Γ per 1 vol point", "∂³V/∂S²∂σ", 3),
    "color": GreekInfo("", "Color", 1.0 / 365.0, "Γ per calendar day", "∂³V/∂S²∂t", 3),
    "ultima": GreekInfo("", "Ultima", 1e-6, "vomma per 1 vol point", "∂³V/∂σ³", 3),
}
