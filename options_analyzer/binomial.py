"""Recombining binomial trees for European and American options.

Two lattices are provided:

* ``"crr"`` - Cox, Ross & Rubinstein (1979): ``u = exp(sigma sqrt(dt))``,
  ``d = 1/u``. Converges to Black-Scholes at rate O(1/n) with the familiar
  odd/even oscillation.
* ``"lr"`` - Leisen & Reimer (1996) with the Peizer-Pratt (method 2)
  inversion. The tree is centred on the strike, so convergence is smooth and
  roughly O(1/n^2); an odd number of steps is used (an even request is rounded
  up by one).

Backward induction is vectorised over the nodes of each time step, so a
2,000-step American tree prices in a few milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import black_scholes as bs

__all__ = ["METHODS", "TreeResult", "price", "price_with_greeks"]

METHODS = {"crr": "Cox-Ross-Rubinstein", "lr": "Leisen-Reimer"}

_MAX_STEPS = 20_000


@dataclass(frozen=True)
class TreeResult:
    """Price and tree Greeks. ``theta`` is per year of calendar time."""

    price: float
    delta: float
    gamma: float
    theta: float
    steps: int
    method: str
    american: bool
    #: ``(times, spot)`` of the early-exercise boundary for American options
    #: (the highest exercised node for puts, the lowest for calls). Empty when
    #: early exercise is never optimal.
    exercise_boundary: tuple[np.ndarray, np.ndarray] | None = None


def _peizer_pratt(z: float, n: int) -> float:
    """Peizer-Pratt method-2 inversion used by the Leisen-Reimer tree."""
    denom = n + 1.0 / 3.0 + 0.1 / (n + 1.0)
    return 0.5 + np.copysign(0.5, z) * np.sqrt(1.0 - np.exp(-((z / denom) ** 2) * (n + 1.0 / 6.0)))


def _lattice(S, K, T, r, sigma, q, steps, method):
    dt = T / steps
    growth = np.exp((r - q) * dt)
    if method == "crr":
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (growth - d) / (u - d)
        if not 0.0 < p < 1.0:
            raise ValueError(
                f"The CRR tree is not arbitrage-free for these inputs (risk-neutral "
                f"probability {p:.4f} is outside (0, 1)). Increase the number of steps, "
                "raise the volatility, or use the Leisen-Reimer tree."
            )
    elif method == "lr":
        vst = sigma * np.sqrt(T)
        d1 = (np.log(S / K) + (r - q) * T) / vst + 0.5 * vst
        d2 = d1 - vst
        p = _peizer_pratt(d2, steps)
        p_star = _peizer_pratt(d1, steps)
        eps = 1e-12
        if eps < p < 1.0 - eps and eps < p_star < 1.0 - eps:
            u = growth * p_star / p
            d = (growth - p * u) / (1.0 - p)
        else:
            # The option is so far in or out of the money relative to its volatility that the
            # Peizer-Pratt probabilities round to 0 or 1. Fall back to the drift-centred
            # (Jarrow-Rudd, risk-neutral) lattice, which stays arbitrage-free here.
            centre = (r - q - 0.5 * sigma**2) * dt
            u = np.exp(centre + sigma * np.sqrt(dt))
            d = np.exp(centre - sigma * np.sqrt(dt))
            p = (growth - d) / (u - d)
            if not 0.0 < p < 1.0:
                raise ValueError("no arbitrage-free lattice exists for these inputs; increase the number of steps")
    else:
        raise ValueError(f"method must be one of {sorted(METHODS)}, got {method!r}")
    return dt, u, d, p


def _deterministic_american(S, K, T, r, q, phi):
    """American value with zero volatility: best discounted exercise on a fine grid."""
    t = np.linspace(0.0, T, 4001)
    values = np.exp(-r * t) * np.maximum(phi * (S * np.exp((r - q) * t) - K), 0.0)
    return float(values.max())


def price_with_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    steps: int = 500,
    american: bool = False,
    method: str = "crr",
) -> TreeResult:
    """Price an option on a binomial tree and read Delta/Gamma/Theta off the tree.

    Delta and gamma use the node values at steps 1 and 2 (Hull, ch. 21). Theta
    uses the middle node at step 2 for CRR (where ``u d = 1``) and a one-day
    re-pricing for Leisen-Reimer, whose middle node is not at the current spot.
    """
    phi = bs.option_sign(option_type)
    method = str(method).lower()
    if method not in METHODS:
        raise ValueError(f"method must be one of {sorted(METHODS)}, got {method!r}")
    steps = int(steps)
    if not 1 <= steps <= _MAX_STEPS:
        raise ValueError(f"steps must be between 1 and {_MAX_STEPS}")
    if S <= 0 or K <= 0:
        raise ValueError("spot and strike must be positive")
    if sigma < 0:
        raise ValueError("volatility must be non-negative")

    if T <= 0 or sigma == 0:
        g = bs.greeks(S, K, T, r, sigma, q, option_type)
        value = bs.price(S, K, T, r, sigma, q, option_type)
        if american and T > 0:
            value = max(value, _deterministic_american(S, K, T, r, q, phi))
        elif american:
            value = max(value, max(phi * (S - K), 0.0))
        return TreeResult(value, g["delta"], g["gamma"], g["theta"], steps, method, american, None)

    # Greeks need the node values at steps 1 and 2 before the terminal layer.
    steps = max(steps, 3)
    if method == "lr" and steps % 2 == 0:
        steps += 1

    dt, u, d, p = _lattice(S, K, T, r, sigma, q, steps, method)
    disc = np.exp(-r * dt)
    log_u, log_d = np.log(u), np.log(d)

    def spots(i: int) -> np.ndarray:
        j = np.arange(i + 1)  # number of up-moves
        return S * np.exp(j * log_u + (i - j) * log_d)

    values = np.maximum(phi * (spots(steps) - K), 0.0)
    level2 = level1 = None
    boundary_t: list[float] = []
    boundary_s: list[float] = []
    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            s_i = spots(i)
            exercise = np.maximum(phi * (s_i - K), 0.0)
            exercised = (exercise > values) & (exercise > 0)
            values = np.where(exercised, exercise, values)
            if exercised.any():
                boundary_t.append(i * dt)
                boundary_s.append(s_i[exercised].max() if phi < 0 else s_i[exercised].min())
        if i == 2:
            level2 = values.copy()
        elif i == 1:
            level1 = values.copy()

    value = float(values[0])
    s1 = spots(1)
    s2 = spots(2)
    delta = float((level1[1] - level1[0]) / (s1[1] - s1[0]))
    gamma = float(
        ((level2[2] - level2[1]) / (s2[2] - s2[1]) - (level2[1] - level2[0]) / (s2[1] - s2[0]))
        / (0.5 * (s2[2] - s2[0]))
    )
    if method == "crr":
        theta = float((level2[1] - value) / (2.0 * dt))
    else:
        bump = min(1.0 / 365.0, 0.5 * T)
        later = price(S, K, T - bump, r, sigma, q, option_type, steps, american, method)
        theta = (later - value) / bump

    boundary = None
    if american:
        order = np.argsort(boundary_t)
        boundary = (np.asarray(boundary_t)[order], np.asarray(boundary_s)[order])
    return TreeResult(value, delta, gamma, theta, steps, method, american, boundary)


def price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    steps: int = 500,
    american: bool = False,
    method: str = "crr",
) -> float:
    """Binomial-tree option value (see :func:`price_with_greeks`)."""
    phi = bs.option_sign(option_type)
    method = str(method).lower()
    if method not in METHODS:
        raise ValueError(f"method must be one of {sorted(METHODS)}, got {method!r}")
    steps = int(steps)
    if not 1 <= steps <= _MAX_STEPS:
        raise ValueError(f"steps must be between 1 and {_MAX_STEPS}")
    if S <= 0 or K <= 0:
        raise ValueError("spot and strike must be positive")
    if sigma < 0:
        raise ValueError("volatility must be non-negative")
    if T <= 0 or sigma == 0:
        value = bs.price(S, K, T, r, sigma, q, option_type)
        if american:
            extra = _deterministic_american(S, K, T, r, q, phi) if T > 0 else max(phi * (S - K), 0.0)
            value = max(value, extra)
        return float(value)
    if method == "lr" and steps % 2 == 0:
        steps += 1

    dt, u, d, p = _lattice(S, K, T, r, sigma, q, steps, method)
    disc = np.exp(-r * dt)
    log_u, log_d = np.log(u), np.log(d)
    j = np.arange(steps + 1)
    values = np.maximum(phi * (S * np.exp(j * log_u + (steps - j) * log_d) - K), 0.0)
    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            j = np.arange(i + 1)
            exercise = np.maximum(phi * (S * np.exp(j * log_u + (i - j) * log_d) - K), 0.0)
            values = np.maximum(values, exercise)
    return float(values[0])
