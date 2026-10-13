"""Recombining binomial trees for European and American options.

Two lattices are provided:

* ``"crr"`` - Cox, Ross & Rubinstein (1979): ``u = exp(sigma sqrt(dt))``,
  ``d = 1/u``. Converges to Black-Scholes at rate O(1/n) with the familiar
  odd/even oscillation.
* ``"lr"`` - Leisen & Reimer (1996) with the Peizer-Pratt (method 2)
  inversion. The tree is centred on the strike, so convergence is smooth:
  roughly O(1/n^2) for European exercise and first order with early exercise.
  An odd number of steps is used (an even request is rounded up by one, or
  down at the 20,000-step limit).

Calls are priced as the symmetric put (McDonald & Schroder, 1998):
``C(S, K, T; r, q) = P(K, S, T; q, r)``. On the lattice this is an exact change
of numeraire (the call tree is the put tree turned upside down, node for node,
so both give the same value), but a put's payoff is bounded by its strike. Very
wide lattices (high volatility, long expiry, many steps) have to cap their node
spots at both ends to stay finite, and a call valued on capped nodes would lose
most of its value; the put is unaffected.

Backward induction is vectorised over the nodes of each time step: a
2,000-step American tree prices in tens of milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import black_scholes as bs

__all__ = ["METHODS", "TreeResult", "american_upper_bound", "extrapolated_price", "price", "price_with_greeks"]

METHODS = {"crr": "Cox-Ross-Rubinstein", "lr": "Leisen-Reimer"}

_MAX_STEPS = 20_000
#: Node spots are kept between e^-700 and e^700 (about 1e-304 and 1e304) so that very wide lattices (high
#: volatility, long expiry, many steps) neither overflow nor round to zero. Only puts are valued on the lattice
#: (calls by put-call symmetry): a put is worth nothing at the top nodes and, to within 1e-304, its whole strike
#: at the bottom ones, so the caps do not change any value. Capped levels are computed from their logarithms, so
#: a capped node never feeds the spots of an earlier level.
_MAX_LOG_SPOT = 700.0
_MAX_SPOT = float(np.exp(_MAX_LOG_SPOT))
#: Bumps for the finite-difference vega (half a volatility point) and rho / dividend rho (25 basis points).
_VOL_BUMP = 0.005
_RATE_BUMP = 0.0025


@dataclass(frozen=True)
class TreeResult:
    """Price and tree Greeks in model units, as in :func:`black_scholes.greeks`.

    ``theta`` is per year of calendar time; ``vega``, ``rho`` and ``epsilon``
    (dividend rho) are per 1.00 of volatility, rate and dividend yield.
    """

    price: float
    delta: float
    gamma: float
    theta: float
    steps: int
    method: str
    american: bool
    #: ``(times, spot)`` of the early-exercise boundary for American options: the critical spot price below
    #: which a put (above which a call) is exercised. At each time step it is interpolated between the last
    #: exercised node and its neighbour, where the exercise gain changes sign; consecutive steps are averaged
    #: because even and odd steps sit on interleaved spot grids, and the result is fitted to be monotone in
    #: time, as the exact boundary is. Only steps where the boundary lies inside the lattice are used, so it
    #: is empty when early exercise is never optimal.
    exercise_boundary: tuple[np.ndarray, np.ndarray] | None = None
    #: Bump-and-reprice sensitivities: central differences on trees with the same steps and lattice type.
    vega: float = float("nan")
    rho: float = float("nan")
    epsilon: float = float("nan")


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


def _deterministic_american(S, K, T, r, q, phi) -> tuple[float, float]:
    """American value with zero volatility and the optimal exercise time (best discounted exercise on a grid)."""
    t = np.linspace(0.0, T, 4001)
    values = np.exp(-r * t) * np.maximum(phi * (S * np.exp((r - q) * t) - K), 0.0)
    best = int(np.argmax(values))
    return float(values[best]), float(t[best])


def _validate(S, K, sigma, option_type, steps, method) -> tuple[int, str, int]:
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
    if method == "lr" and steps % 2 == 0:
        steps = steps + 1 if steps < _MAX_STEPS else steps - 1
    return phi, method, steps


def _spots(S: float, log_u: float, log_d: float, i: int, ups_log: np.ndarray | None = None) -> np.ndarray:
    """Spot at every node of level ``i``, capped at both ends so that extreme lattices stay finite and positive.

    ``ups_log`` optionally holds ``j (log u - log d)`` for ``j = 0, 1, ...`` (at least ``i + 1`` entries).
    """
    if ups_log is None:
        ups_log = np.arange(i + 1) * (log_u - log_d)
    log_spot = (np.log(S) + i * log_d) + ups_log[: i + 1]
    return np.exp(np.clip(log_spot, -_MAX_LOG_SPOT, _MAX_LOG_SPOT))


def _induct_put(S, K, T, r, sigma, q, steps, american, method, keep_levels=False):
    """Backward induction for a put. Returns ``(value, levels, boundary, (log_u, log_d))``.

    With ``keep_levels``, ``levels`` holds the node values at steps 0-2 and ``boundary`` the critical prices.
    """
    dt, u, d, p = _lattice(S, K, T, r, sigma, q, steps, method)
    disc = np.exp(-r * dt)
    log_u, log_d = np.log(u), np.log(d)
    ups_log = np.arange(steps + 1) * (log_u - log_d)
    spot = _spots(S, log_u, log_d, steps, ups_log)
    values = np.maximum(K - spot, 0.0)
    # Whether every node of the lattice lies strictly inside the caps (and so is a normal float).
    log_s = np.log(S)
    uncapped = log_s + steps * min(log_d, 0.0) > -_MAX_LOG_SPOT and log_s + steps * max(log_u, 0.0) < _MAX_LOG_SPOT
    inv_d = 1.0 / d
    levels = {}
    boundary_steps: list[int] = []
    boundary_spots: list[float] = []
    # With r <= 0 and q >= 0 a put is never exercised early: on the risk-neutral lattice its continuation value is
    # at least K e^(-r tau) - S e^(-q tau) >= K - S (equal deep in the money at r = 0, where only rounding decides
    # the sign of the exercise gain). Through C(S, K; r, q) = P(K, S; q, r) this also covers a call with r >= 0 and
    # q <= 0 (whose swapped lattice gains -S (1 - e^(-r dt)) at its tiny bottom spots, which can round above zero).
    # Elsewhere a gain within rounding of zero (1e-12 of the strike) is not counted as exercise either, so a rate
    # within rounding of zero behaves like a zero rate.
    record_boundary = keep_levels and not (r <= 0.0 and q >= 0.0)
    tolerance = 1e-12 * K
    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            # Level-i spots are level i+1's divided by d (one multiply per node) while no node is capped.
            # Otherwise they come from their logarithms: dividing capped spots would carry the cap back towards
            # the root (a bottom node held at e^-700, or one rounded to zero, would make exercising worth the
            # whole strike at nodes far above it). The root is exact, so immediate exercise there is worth
            # exactly the intrinsic value.
            if not i:
                spot = np.array([float(S)])
            elif uncapped:
                spot = spot[:-1] * inv_d
            else:
                spot = _spots(S, log_u, log_d, i, ups_log)
            exercise = K - spot
            if record_boundary:
                gain = exercise - values
                exercised = gain > tolerance
                # Record the boundary only where it is interior to the level: if the top node is exercised
                # the boundary lies beyond the lattice (the spot started inside the exercise region).
                if exercised.any() and not exercised[-1]:
                    k = int(np.flatnonzero(exercised)[-1])
                    # The exercise gain is positive at node k and not at node k+1: take the spot where its
                    # linear interpolant crosses zero (exercising a put is only ever optimal below K). A gain at
                    # node k+1 inside the tolerance keeps the crossing between the two nodes.
                    weight = min(gain[k] / (gain[k] - gain[k + 1]), 1.0)
                    boundary_steps.append(i)
                    boundary_spots.append(min(spot[k] + weight * (spot[k + 1] - spot[k]), K))
            values = np.maximum(values, exercise)
        if keep_levels and i <= 2:
            levels[i] = values.copy()
    return float(values[0]), levels, _boundary_curve(boundary_steps, boundary_spots, dt), (log_u, log_d)


def _boundary_curve(steps: list[int], spots: list[float], dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Smooth the per-step critical prices of a put (``steps`` arrive in descending order).

    Even and odd steps sit on interleaved spot grids, so the per-step estimates alternate around the true
    boundary; the mean of each pair of consecutive steps cancels that odd/even alternation. A put's exercise
    region can only grow as expiry approaches (more time to expiry is never worth less), so the boundary is
    non-decreasing in time: the small ripple left by lattices whose levels do not interleave exactly (such
    as Leisen-Reimer away from the money) is removed by the least-squares non-decreasing fit.
    """
    i = np.asarray(steps[::-1], dtype=int)
    b = np.asarray(spots[::-1], dtype=float)
    if i.size < 2:
        return np.empty(0), np.empty(0)
    pair = np.diff(i) == 1
    return ((i[:-1] + 0.5) * dt)[pair], _non_decreasing_fit((0.5 * (b[:-1] + b[1:]))[pair])


def _non_decreasing_fit(y: np.ndarray) -> np.ndarray:
    """Least-squares non-decreasing fit (pool-adjacent-violators algorithm)."""
    means: list[float] = []
    sizes: list[int] = []
    for value in y:
        means.append(float(value))
        sizes.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            size = sizes[-2] + sizes[-1]
            means[-2:] = [(means[-2] * sizes[-2] + means[-1] * sizes[-1]) / size]
            sizes[-2:] = [size]
    return np.repeat(means, sizes)


def _induct(S, K, T, r, sigma, q, phi, steps, american, method, keep_levels=False):
    """Backward induction. Returns ``(value, levels, (boundary_t, boundary_s), (log_u, log_d))``.

    A call is the put on swapped inputs, ``C(S, K; r, q) = P(K, S; q, r)``. Node ``m`` (up-moves) of step
    ``i`` of the call lattice is node ``i - m`` of that put's lattice, whose spot is ``K S / x``, and the call
    is worth ``x / S`` times the put there, so the call's node values and boundary follow exactly.
    """
    if phi < 0:
        return _induct_put(S, K, T, r, sigma, q, steps, american, method, keep_levels)
    value, put_levels, (times, put_boundary), (log_u_put, log_d_put) = _induct_put(
        K, S, T, q, sigma, r, steps, american, method, keep_levels
    )
    log_u, log_d = -log_d_put, -log_u_put
    levels = {}
    for i, put_values in put_levels.items():
        ups = np.arange(i + 1)
        levels[i] = np.exp(ups * log_u + (i - ups) * log_d) * put_values[::-1]
    valid = put_boundary > K * S / _MAX_SPOT  # the call's critical price K S / b stays on the capped lattice
    return value, levels, (times[valid], K * S / put_boundary[valid]), (log_u, log_d)


def _difference(f, x: float, h: float, base: float, lower: float = -np.inf) -> float:
    """Central difference of ``f`` at ``x``; one-sided where a bumped tree is invalid (e.g. CRR with p > 1)."""

    def value(y: float) -> float | None:
        if y <= lower:
            return None
        try:
            return f(y)
        except ValueError:
            return None

    up, down = value(x + h), value(x - h)
    if up is not None and down is not None:
        return (up - down) / (2.0 * h)
    if up is not None:
        return (up - base) / h
    if down is not None:
        return (base - down) / h
    return float("nan")


def _bumped_greeks(S, K, T, r, sigma, q, option_type, steps, american, method, base) -> tuple[float, float, float]:
    """Vega, rho and dividend rho by repricing on trees with the same steps and lattice type."""

    def tree(r_: float = r, sigma_: float = sigma, q_: float = q) -> float:
        return price(S, K, T, r_, sigma_, q_, option_type, steps, american, method)

    vega = _difference(lambda v: tree(sigma_=v), sigma, min(_VOL_BUMP, 0.5 * sigma), base, lower=0.0)
    rho = _difference(lambda x: tree(r_=x), r, _RATE_BUMP, base)
    epsilon = _difference(lambda x: tree(q_=x), q, _RATE_BUMP, base)
    return vega, rho, epsilon


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
    uses the middle node at step 2 for CRR (where ``u d = 1``). The
    Leisen-Reimer middle node is not at the current spot, so its theta is a
    central difference in time to expiry on the same tree, whose value is
    smooth in ``T``. Vega, rho and dividend rho are central differences on
    trees with the same number of steps and lattice type (half a volatility
    point and 25 basis points either side), so they include early exercise.
    """
    phi, method, steps = _validate(S, K, sigma, option_type, steps, method)

    if T <= 0 or sigma == 0:
        g = bs.greeks(S, K, T, r, sigma, q, option_type)
        value = bs.price(S, K, T, r, sigma, q, option_type)
        delta, gamma, theta = g["delta"], g["gamma"], g["theta"]
        vega, rho, epsilon = g["vega"], g["rho"], g["epsilon"]
        if american:
            if T > 0:
                early, when = _deterministic_american(S, K, T, r, q, phi)
            else:
                early, when = max(phi * (S - K), 0.0), 0.0
            if early > value:
                # Exercise at a fixed date w before expiry: the value phi (S e^{-q w} - K e^{-r w}) no longer
                # depends on the time to expiry (theta 0) and is linear in the spot.
                value, delta, gamma, theta = early, float(phi * np.exp(-q * when)), 0.0, 0.0
                vega = 0.0
                rho = float(phi * K * when * np.exp(-r * when))
                epsilon = float(-phi * S * when * np.exp(-q * when))
        return TreeResult(value, delta, gamma, theta, steps, method, american, None, vega, rho, epsilon)

    # Greeks need the node values at steps 1 and 2 before the terminal layer.
    if steps < 3:
        steps = 3

    value, levels, boundary, (log_u, log_d) = _induct(
        S, K, T, r, sigma, q, phi, steps, american, method, keep_levels=True
    )
    level1, level2 = levels[1], levels[2]
    s1 = _spots(S, log_u, log_d, 1)
    s2 = _spots(S, log_u, log_d, 2)
    delta = float((level1[1] - level1[0]) / (s1[1] - s1[0]))
    gamma = float(
        ((level2[2] - level2[1]) / (s2[2] - s2[1]) - (level2[1] - level2[0]) / (s2[1] - s2[0]))
        / (0.5 * (s2[2] - s2[0]))
    )
    if method == "crr":
        theta = float((level2[1] - value) / (2.0 * T / steps))
    else:
        h = 0.01 * min(1.0 / 365.0, T)
        shorter = price(S, K, T - h, r, sigma, q, option_type, steps, american, method)
        longer = price(S, K, T + h, r, sigma, q, option_type, steps, american, method)
        theta = (shorter - longer) / (2.0 * h)

    vega, rho, epsilon = _bumped_greeks(S, K, T, r, sigma, q, option_type, steps, american, method, value)
    return TreeResult(
        value, delta, gamma, theta, steps, method, american, boundary if american else None, vega, rho, epsilon
    )


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
    phi, method, steps = _validate(S, K, sigma, option_type, steps, method)
    if T <= 0 or sigma == 0:
        value = bs.price(S, K, T, r, sigma, q, option_type)
        if american:
            extra = _deterministic_american(S, K, T, r, q, phi)[0] if T > 0 else max(phi * (S - K), 0.0)
            value = max(value, extra)
        return float(value)
    return _induct(S, K, T, r, sigma, q, phi, steps, american, method)[0]


def extrapolated_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    american: bool = True,
    steps: int = 2001,
) -> float:
    """Converged value: Richardson extrapolation of Leisen-Reimer trees with ``steps`` and twice as many.

    With early exercise the Leisen-Reimer error is usually close to ``c / n``, so ``(n2 V2 - n1 V1) / (n2 - n1)``
    removes its leading term: at the defaults the result agrees with the 20,000-step references in the tests to
    about 1e-5. Where the tree's error is irregular (e.g. a put at very low volatility with a high rate, or
    several hundred percent volatility over years) the extrapolation is only about as accurate as the finer
    tree. The result is kept within the no-arbitrage bounds of an American option (see :func:`_american_bounds`).
    European values use the closed form.
    """
    european = float(bs.price(S, K, T, r, sigma, q, option_type))
    if not american:
        return european
    phi, _, n1 = _validate(S, K, sigma, option_type, steps, "lr")
    _, _, n2 = _validate(S, K, sigma, option_type, min(2 * n1, _MAX_STEPS), "lr")
    v1 = price(S, K, T, r, sigma, q, option_type, n1, True, "lr")
    if n2 <= n1:
        value = v1
    else:
        v2 = price(S, K, T, r, sigma, q, option_type, n2, True, "lr")
        value = (n2 * v2 - n1 * v1) / (n2 - n1)
    lower, upper = _american_bounds(S, K, T, r, sigma, q, phi, european)
    return float(max(min(value, upper), lower))


def american_upper_bound(S: float, K: float, T: float, r: float, q: float = 0.0, option_type: str = "call") -> float:
    """Model-free upper bound of an American option: the most exercise can be worth.

    A put pays at most its strike, received at once or, when the rate is negative, at expiry:
    ``K max(1, e^(-rT))``. A call is bounded as the symmetric put, ``C(S, K; r, q) = P(K, S; q, r)``:
    ``S max(1, e^(-qT))``. The value approaches the bound as volatility grows without limit.
    """
    if bs.option_sign(option_type) > 0:
        S, K, r, q = K, S, q, r
    return float(K * max(1.0, np.exp(-r * T)))


def _american_bounds(S, K, T, r, sigma, q, phi, european) -> tuple[float, float]:
    """No-arbitrage bounds ``(lower, upper)`` of an American option.

    Lower: the European value and the intrinsic value. Upper: :func:`american_upper_bound` and the perpetual
    American value, which no finite expiry can exceed (finite for a put when ``r > 0``, and for a call, the
    symmetric put, when ``q > 0``). The lower bound wins if rounding ever puts the two in the wrong order.
    """
    lower = max(european, phi * (S - K))
    upper = american_upper_bound(S, K, T, r, q, "call" if phi > 0 else "put")
    if phi > 0:
        S, K, r, q = K, S, q, r
    if r > 0 and sigma > 0:
        upper = min(upper, _perpetual_put(S, K, r, q, sigma))
    return lower, max(upper, lower)


def _perpetual_put(S: float, K: float, r: float, q: float, sigma: float) -> float:
    """Perpetual American put (Merton, 1973) for ``r > 0``: exercised at or below ``b = K beta / (beta - 1)``,
    worth ``(K - b) (S / b)^beta`` above it, where ``beta < 0`` solves
    ``sigma^2 beta (beta - 1) / 2 + (r - q) beta - r = 0``."""
    a = (r - q) / sigma**2 - 0.5
    c = 2.0 * r / sigma**2
    root = np.sqrt(a * a + c)
    beta = -a - root if a >= 0 else -c / (root - a)  # the same root, without cancellation when a < 0
    if not beta < 0:
        return float("inf")
    log_s_over_b = np.log(S / K) + np.log1p(-1.0 / beta)
    if log_s_over_b <= 0:
        return float(K - S)
    return float(K / (1.0 - beta) * np.exp(beta * log_s_over_b))
