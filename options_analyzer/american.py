"""American options by spectral collocation of the exercise boundary (Andersen, Lake & Offengenden, 2016).

The engine follows the design of QuantLib's ``QdFpAmericanEngine`` (BSD-3-Clause, by Klaus Spanderen, 2022), with
every integral taken on a cosine map. No QuantLib code is copied: the method is written from the papers below, and
QuantLib's engine served as the reference design and, in the oracle tests, as the reference implementation.

Model and method
----------------
Black-Scholes-Merton dynamics with a constant rate ``r``, dividend yield ``q`` and volatility ``sigma``. An American
put is exercised once the spot falls to the exercise boundary ``B(tau)`` (``tau`` is the time to expiry), and its
value is the European value plus the early-exercise premium (Kim, 1990)::

    V(S) = v(S) + int_0^T [r K e^{-r t} N(-d-(t, S / B(T - t))) - q S e^{-q t} N(-d+(t, S / B(T - t)))] dt,

with ``d+-(t, z) = (ln z + (r - q) t) / (sigma sqrt(t)) +- sigma sqrt(t) / 2``. The boundary solves a fixed-point
equation ``B(tau) = K e^{-(r - q) tau} N(tau, B) / D(tau, B)`` whose ``N`` and ``D`` are integrals over the whole
boundary before ``tau``:

* the *cdf form* (the paper's FP-B), used when ``|r - q| >= 1e-3``, has normal distribution functions in ``N`` and
  ``D``; the *density form* (FP-A), used otherwise, has normal densities, and converges where the cdf form stalls
  (``r`` close to ``q``);
* ``B`` is represented by ``H = ln(B / X)^2`` on ``n + 1`` Chebyshev-Lobatto nodes in ``z = 2 sqrt(tau / T) - 1``,
  where ``X = K min(1, r / q)`` is the boundary at expiry: ``H`` is smooth in ``sqrt(tau)`` although ``B`` is not;
* the iteration starts from Li's (2010) QD+ approximation of the boundary, takes one Jacobi-Newton step and then
  ``m - 1`` plain fixed-point steps;
* every integral, over ``u`` in ``[0, tau]`` for the boundary and over ``t`` in ``[0, T]`` for the premium, is a
  Gauss-Legendre rule in ``theta`` after the cosine map ``u = tau (1 - cos theta) / 2``. The map makes both ends of
  the interval smooth (``sqrt(u)`` and ``sqrt(tau - u)`` are ``sqrt(tau)`` times ``sin(theta / 2)`` and
  ``cos(theta / 2)``), so the rule converges spectrally.

A scheme fixes ``(l, m, n, p)``: ``l`` quadrature points for the boundary integrals, ``m`` iterations, ``n + 1``
collocation nodes and ``p`` quadrature points for the premium (:data:`SCHEMES`).

Calls are priced as puts by McDonald and Schroder's (1998) symmetry ``C(S, K; r, q) = P(K, S; q, r)``. The
boundary of a put is proportional to its strike, so the normalised boundary ``B / K`` is cached per
``(T, r, q, sigma, scheme)``: a strike strip at one expiry (or the spots of a profile) solves one boundary.

Regimes (Healy, 2021; Andersen and Lake, 2021)
----------------------------------------------
* ``r <= 0`` and ``q >= r`` (a put; ``q <= 0`` and ``r >= q`` for a call): early exercise is never optimal, and the
  value is the European one.
* ``q < r < 0`` for a put (``r < q < 0`` for a call): the exercise region lies between two boundaries, which this
  engine (like QuantLib's) does not solve. The value falls back to :func:`binomial.extrapolated_price`, and
  :func:`price_details` flags it.
* Otherwise one boundary, solved as above. ``sigma = 0`` is valued exactly (the best deterministic exercise date).

References
----------
* L. Andersen and M. Lake (2021), "Fast American option pricing: the double-boundary case", *Wilmott*
  2021(116), 30-41, doi:10.1002/wilm.10969.
* L. Andersen, M. Lake and D. Offengenden (2016), "High-performance American option pricing", *Journal of
  Computational Finance* 20(1), 39-87, doi:10.21314/JCF.2016.312.
* J. Healy (2021), "Pricing American options under negative rates", *Journal of Computational Finance* 25(1),
  doi:10.21314/JCF.2021.004, arXiv:2109.15157.
* I. J. Kim (1990), "The analytic valuation of American options", *Review of Financial Studies* 3(4), 547-572,
  doi:10.1093/rfs/3.4.547.
* M. Li (2010), "Analytical approximations for the critical stock prices of American options: a performance
  comparison", *Review of Derivatives Research* 13(1), 75-99, doi:10.1007/s11147-009-9044-3.
* R. McDonald and M. Schroder (1998), "A parity result for American options", *Journal of Computational Finance*
  1(3), 5-13, doi:10.21314/JCF.1998.010.
* K. Spanderen (2022), QuantLib ``QdFpAmericanEngine``, https://github.com/lballabio/QuantLib (BSD-3-Clause).

The method, its error and its limits are written up in ``docs/methods/american_alo.md``.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr

from . import binomial
from . import black_scholes as bs

__all__ = [
    "SCHEMES",
    "AmericanPrice",
    "Scheme",
    "exercise_boundary",
    "exercise_regime",
    "price",
    "price_details",
]

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
#: Below this ``|r - q|`` the density form of the fixed-point equation replaces the cdf form (as in QuantLib).
_DENSITY_FORM_BELOW = 1e-3
#: QD+ initial guess: Halley steps per node and their relative tolerance.
_QD_PLUS_STEPS = 12
_QD_PLUS_TOLERANCE = 1e-9
#: Normalised boundaries kept in the cache (one per expiry, rate, dividend yield, volatility and scheme).
_CACHE_SIZE = 4096
#: Steps of the Leisen-Reimer tree in the double-boundary fallback (Richardson with twice as many).
_TREE_STEPS = 2001


@dataclass(frozen=True)
class Scheme:
    """Discretisation of the engine.

    Attributes
    ----------
    name : str
        Label shown in results.
    l : int
        Gauss-Legendre points of each boundary integral (``N`` and ``D`` at one node).
    m : int
        Iterations: one Jacobi-Newton step, then ``m - 1`` plain fixed-point steps.
    n : int
        Degree of the Chebyshev interpolant of the boundary (``n + 1`` collocation nodes).
    p : int
        Gauss-Legendre points of the early-exercise premium integral.
    """

    name: str
    l: int  # noqa: E741  (the paper's name)
    m: int
    n: int
    p: int

    def __post_init__(self) -> None:
        if min(self.l, self.m, self.n, self.p) < 1:
            raise ValueError("every scheme parameter (l, m, n, p) must be at least 1")


#: Named schemes. Measured on the 4,495 puts of Healy (2021, Table 5) against QuantLib's QdFp high-precision
#: engine (see docs/methods/american_alo.md and tests/test_american.py).
SCHEMES: dict[str, Scheme] = {
    "fast": Scheme("fast", l=8, m=3, n=7, p=24),
    "accurate": Scheme("accurate", l=24, m=8, n=14, p=48),
    "high": Scheme("high", l=64, m=16, n=32, p=128),
}


@dataclass(frozen=True)
class AmericanPrice:
    """An American value with what produced it (see :func:`price_details`).

    Attributes
    ----------
    value : float
        The American value, in the currency of ``S`` and ``K``.
    european : float
        The European (BSM) value of the same option.
    premium : float
        ``value - european``, the early-exercise premium.
    method : str
        ``"alo"`` (spectral collocation), ``"european"`` (early exercise never optimal), ``"tree"`` (the
        double-boundary fallback), ``"exact"`` (zero volatility or a zero spot) or ``"intrinsic"`` (expired).
    fallback : bool
        True when the engine could not price the option and :func:`binomial.extrapolated_price` did.
    critical_spot : float
        The exercise boundary today: a put is exercised at or below it, a call at or above it. ``0`` (put) or
        ``inf`` (call) when early exercise is never optimal; NaN where it is not computed (``tree``, ``exact``,
        ``intrinsic``).
    fixed_point : str or None
        ``"cdf"`` or ``"density"``: the form of the boundary equation, for ``method == "alo"``.
    scheme : str
        The scheme's name.
    """

    value: float
    european: float
    premium: float
    method: str
    fallback: bool
    critical_spot: float
    fixed_point: str | None
    scheme: str


# ------------------------------------------------------------------------------------------------- regimes


def exercise_regime(r: float, q: float = 0.0, *, option_type: str = "call") -> str:
    """Shape of the early-exercise region of an American option in the Black-Scholes-Merton model.

    Parameters
    ----------
    r, q : float
        Continuously compounded risk-free rate and dividend yield (decimals, per year).
    option_type : {"call", "put"}
        The option.

    Returns
    -------
    str
        ``"european"`` when early exercise is never optimal (a put with ``r <= 0`` and ``r <= q``; a call with
        ``q <= 0`` and ``q <= r``; Healy 2021, Propositions 1 and 2), ``"double"`` when the exercise region lies
        between two boundaries (a put with ``q < r < 0``; a call with ``r < q < 0``), and ``"single"`` otherwise.

    Examples
    --------
    >>> exercise_regime(0.05, 0.02, option_type="put")
    'single'
    >>> exercise_regime(0.05, 0.0, option_type="call")
    'european'
    >>> exercise_regime(-0.01, -0.02, option_type="put")
    'double'
    """
    if bs.option_sign(option_type) > 0:
        r, q = q, r
    return _put_regime(float(r), float(q))


def _put_regime(r: float, q: float) -> str:
    if r <= 0.0 and r <= q:
        return "european"
    if q < r < 0.0:
        return "double"
    return "single"


def _put_x(r: float, q: float) -> float:
    """``X / K``: the normalised boundary of a put at expiry (``r > 0``, or ``r = 0`` with ``q < 0``)."""
    return min(1.0, r / q) if q > 0.0 else 1.0


# ------------------------------------------------------------------------------------- scheme constants


def _chebyshev_matrix(x: np.ndarray, n: int) -> np.ndarray:
    """Values at ``x`` of the degree-``n`` polynomial through the Chebyshev-Lobatto nodes, as a matrix.

    Row ``k`` maps the values at the ``n + 1`` nodes ``z_i = -cos(i pi / n)`` to the interpolant at ``x[k]``.
    """
    nodes = -np.cos(np.arange(n + 1) * np.pi / n)
    degrees = np.arange(n + 1)
    at_nodes = np.cos(degrees[None, :] * np.arccos(nodes[:, None]))
    at_x = np.cos(degrees[None, :] * np.arccos(np.clip(x.reshape(-1), -1.0, 1.0))[:, None])
    return np.linalg.solve(at_nodes.T, at_x.T).T


@dataclass(frozen=True)
class _Grid:
    """Constants of a scheme that do not depend on the option: nodes, quadrature rules and interpolation matrices."""

    z: np.ndarray  # collocation nodes in z, z[0] = -1 (tau = 0) ... z[n] = 1 (tau = T)
    tau_share: np.ndarray  # tau_i / T at the nodes 1..n
    u_share: np.ndarray  # u / tau at the boundary quadrature points (cosine map), shape (l,)
    weight: np.ndarray  # du / tau weights: (pi / 2) w_j sin(theta_j) / 2
    density_weight: np.ndarray  # du / sqrt(tau (tau - u)) weights: (pi / 2) w_j sin(theta_j / 2)
    boundary_matrix: np.ndarray  # (n, l, n + 1): H at the quadrature points of nodes 1..n from H at the nodes
    t_share: np.ndarray  # t / T at the premium quadrature points, shape (p,)
    premium_weight: np.ndarray  # dt / T weights
    premium_matrix: np.ndarray  # (p, n + 1): H at tau = T - t from H at the nodes


@functools.cache
def _grid(scheme: Scheme) -> _Grid:
    n = scheme.n
    z = -np.cos(np.arange(n + 1) * np.pi / n)
    x, w = np.polynomial.legendre.leggauss(scheme.l)
    theta = 0.5 * np.pi * (x + 1.0)
    half = 0.5 * np.pi * w
    # At node i, u = tau_i sin^2(theta / 2), so z(u) = 2 sqrt(u / T) - 1 = (1 + z_i) sin(theta / 2) - 1.
    z_points = (1.0 + z[1:, None]) * np.sin(0.5 * theta)[None, :] - 1.0
    xp, wp = np.polynomial.legendre.leggauss(scheme.p)
    phi = 0.5 * np.pi * (xp + 1.0)
    # The premium needs B at tau = T - t = T cos^2(phi / 2), so z = 2 cos(phi / 2) - 1.
    return _Grid(
        z=z,
        tau_share=(0.5 * (1.0 + z[1:])) ** 2,
        u_share=np.sin(0.5 * theta) ** 2,
        weight=0.5 * half * np.sin(theta),
        density_weight=half * np.sin(0.5 * theta),
        boundary_matrix=_chebyshev_matrix(z_points, n).reshape(n, scheme.l, n + 1),
        t_share=np.sin(0.5 * phi) ** 2,
        premium_weight=0.25 * np.pi * wp * np.sin(phi),
        premium_matrix=_chebyshev_matrix(2.0 * np.cos(0.5 * phi) - 1.0, n),
    )


def _scheme(scheme: str | Scheme) -> Scheme:
    if isinstance(scheme, Scheme):
        return scheme
    try:
        return SCHEMES[str(scheme).strip().lower()]
    except KeyError:
        raise ValueError(f"scheme must be one of {sorted(SCHEMES)} or a Scheme, got {scheme!r}") from None


# ----------------------------------------------------------------------------------------------- boundary


def _npdf(x: np.ndarray) -> np.ndarray:
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


def _qd_plus(tau: np.ndarray, r: float, q: float, sigma: float, x_max: float) -> np.ndarray:
    """Li's (2010) QD+ approximation of the normalised put boundary ``B / K`` at ``tau`` (the initial guess).

    The QD+ boundary solves ``F(b) = (1 - e^{-q tau} N(-d+)) b - beta (1 - b - p(b)) + alpha Theta(b) / e^{-r tau}
    = 0``, where ``p`` and ``Theta`` are the European put and its calendar-time theta at spot ``b`` and unit strike,
    and ``alpha``, ``beta`` and ``lambda`` are Li's coefficients (as in QuantLib's QdPlus engine). Expanded, ``F`` is
    ``c1 b - beta - c2 b N(-d+) + c3 N(-d-) - c4 b n(d+)``, which Halley's method solves from ``b = X`` on every
    node at once. A node that does not converge keeps its last iterate; the fixed-point iterations correct it.
    """
    sigma2 = sigma * sigma
    v = sigma * np.sqrt(tau)
    dr, dq = np.exp(-r * tau), np.exp(-q * tau)
    rt = r * tau
    # r / (1 - e^{-r tau}) = 1 / (tau g(r tau)) with g(x) = (1 - e^{-x}) / x, which is 1 at x = 0.
    g = np.where(rt == 0.0, 1.0, -np.expm1(-rt) / np.where(rt == 0.0, 1.0, rt))
    ddr = 1.0 / (tau * g)
    omega = 2.0 * (r - q) / sigma2
    root = np.sqrt((omega - 1.0) ** 2 + 8.0 * ddr / sigma2)
    lam = 0.5 * (-(omega - 1.0) - root)
    lam_prime = 2.0 * ddr * ddr / (sigma2 * root)
    denominator = 2.0 * lam + omega - 1.0
    alpha = 2.0 * dr / (sigma2 * denominator)
    beta = alpha * (ddr + lam_prime / denominator) - lam
    a = alpha / dr
    c1 = 1.0 + beta
    c2 = dq * (c1 + a * q)
    c3 = dr * (beta + a * r)
    c4 = a * dq * 0.5 * sigma2 / v
    e1 = dq * (1.0 - a * (r - q)) / v
    e2 = a * dq / (2.0 * tau)
    shift = (r - q) * tau / v + 0.5 * v  # d+ = ln(b) / v + shift
    x_min = 1e4 * np.finfo(float).eps * x_max
    b = np.full_like(tau, x_max)
    for _ in range(_QD_PLUS_STEPS):
        dp = np.log(b) / v + shift
        dm = dp - v
        cdf_dp, pdf_dp = ndtr(-dp), _npdf(dp)
        f = c1 * b - beta - c2 * b * cdf_dp + c3 * ndtr(-dm) - c4 * b * pdf_dp
        slope = e1 + e2 * dm
        f1 = c1 - c2 * cdf_dp + pdf_dp * slope
        f2 = pdf_dp / (b * v) * (c2 - dp * slope + e2)
        with np.errstate(divide="ignore", invalid="ignore"):  # a flat F gives no step: the node keeps b
            new = np.maximum(x_min, b - f / f1 / (1.0 - 0.5 * f * f2 / (f1 * f1)))
        new = np.where(np.isfinite(new), new, b)
        done = np.max(np.abs(new - b)) <= _QD_PLUS_TOLERANCE * x_max
        b = new
        if done:
            break
    return np.minimum(b, x_max)


@functools.lru_cache(maxsize=_CACHE_SIZE)
def _put_boundary(T: float, r: float, q: float, sigma: float, scheme: Scheme) -> tuple[np.ndarray, str]:
    """``H = ln(B / X)^2`` of the normalised put boundary at the scheme's nodes, and the fixed-point form used.

    For ``r > 0`` (or ``r = 0 > q``) and ``sigma > 0``. The array is read-only: it is shared through the cache.
    """
    grid = _grid(scheme)
    x_max = _put_x(r, q)
    log_x = math.log(x_max)
    density = abs(r - q) < _DENSITY_FORM_BELOW
    tau = T * grid.tau_share  # nodes 1..n
    sqrt_tau = np.sqrt(tau)
    v = sigma * sqrt_tau
    carry = r - q
    # Quadrature points: u = tau a_j and tau - u = tau (1 - a_j).
    u = tau[:, None] * grid.u_share[None, :]
    rest = tau[:, None] - u
    v_rest = sigma * np.sqrt(rest)
    drift_rest = carry * rest
    w_r = tau[:, None] * grid.weight[None, :] * np.exp(r * u)
    w_q = tau[:, None] * grid.weight[None, :] * np.exp(q * u)
    if density:
        dw = (sqrt_tau / sigma)[:, None] * grid.density_weight[None, :]
        dw_r, dw_q = dw * np.exp(r * u), dw * np.exp(q * u)
    growth = np.exp(-carry * tau)

    b0 = _qd_plus(tau, r, q, sigma, x_max)
    h = np.zeros(scheme.n + 1)
    h[1:] = np.log(b0 / x_max) ** 2
    for iteration in range(scheme.m):
        root_h = np.sqrt(h[1:])
        root_points = np.sqrt(np.maximum(grid.boundary_matrix @ h, 0.0))
        # ln(B(tau_i) / B(u)) = sqrt(H(u)) - sqrt(H(tau_i)), and ln(B(tau_i) / K) = ln X - sqrt(H(tau_i)).
        dp = (root_points - root_h[:, None] + drift_rest) / v_rest + 0.5 * v_rest
        dm = dp - v_rest
        dp0 = (log_x - root_h + carry * tau) / v + 0.5 * v
        dm0 = dp0 - v
        pdf_dp0, pdf_dm0 = _npdf(dp0), _npdf(dm0)
        if density:
            cdf_dp = ndtr(dp)
            numerator = pdf_dm0 / v + r * np.sum(dw_r * _npdf(dm), axis=1)
            denominator = pdf_dp0 / v + ndtr(dp0) + q * np.sum(dw_q * _npdf(dp) + w_q * cdf_dp, axis=1)
        else:
            numerator = ndtr(dm0) + r * np.sum(w_r * ndtr(dm), axis=1)
            denominator = ndtr(dp0) + q * np.sum(w_q * ndtr(dp), axis=1)
        b = x_max * np.exp(-root_h)
        # A node whose update is not a positive number (a vanishing denominator, a Newton step through zero)
        # keeps its value for this iteration.
        with np.errstate(divide="ignore", invalid="ignore"):
            f = growth * numerator / denominator
            if iteration == 0:
                # Jacobi-Newton: the derivative of f in B(tau_i) through the terms outside the integrals.
                if density:
                    d_num = -pdf_dm0 * dm0 / (b * v * v)
                    d_den = -pdf_dp0 * dp0 / (b * v * v) + pdf_dp0 / (b * v)
                else:
                    d_num = pdf_dm0 / (b * v)
                    d_den = pdf_dp0 / (b * v)
                slope = growth * (d_num / denominator - d_den * numerator / (denominator * denominator))
                newton = b - (f - b) / (slope - 1.0)
                f = np.where(np.isfinite(newton) & (newton > 0.0), newton, f)
        new = np.where(np.isfinite(f) & (f > 0.0), f, b)
        h[1:] = (np.log(new) - log_x) ** 2
    h.flags.writeable = False
    return h, "density" if density else "cdf"


def _boundary_at(h: np.ndarray, z: np.ndarray, n: int, x_max: float) -> np.ndarray:
    """The normalised boundary ``X exp(-sqrt(H(z)))`` at points ``z`` in [-1, 1]."""
    values = _chebyshev_matrix(z, n) @ h
    return x_max * np.exp(-np.sqrt(np.maximum(values, 0.0)))


def _premium(s: np.ndarray, T: float, r: float, q: float, sigma: float, h: np.ndarray, scheme: Scheme) -> np.ndarray:
    """Early-exercise premium of puts with normalised spots ``s = S / K`` (in units of the strike)."""
    grid = _grid(scheme)
    x_max = _put_x(r, q)
    t = T * grid.t_share
    log_b = math.log(x_max) - np.sqrt(np.maximum(grid.premium_matrix @ h, 0.0))
    vt = sigma * np.sqrt(t)
    dp = (np.log(s)[:, None] - log_b[None, :] + (r - q) * t) / vt + 0.5 * vt
    dm = dp - vt
    weight = T * grid.premium_weight
    rate_leg = (weight * r * np.exp(-r * t)) @ ndtr(-dm).T
    dividend_leg = (weight * q * np.exp(-q * t)) @ ndtr(-dp).T
    return rate_leg - s * dividend_leg


# -------------------------------------------------------------------------------------------------- prices


def _deterministic_put(S: np.ndarray, K: np.ndarray, T: float, r: float, q: float) -> np.ndarray:
    """Zero-volatility American put: the best of exercising now, at expiry, or at the stationary date of
    ``K e^{-r t} - S e^{-q t}`` (Healy 2021; QuantLib's QdPlus engine does the same)."""

    def gain(t: np.ndarray | float) -> np.ndarray:
        return K * np.exp(-r * t) - S * np.exp(-q * t)

    best = np.maximum(np.maximum(gain(0.0), gain(T)), 0.0)
    if r != q and r * q > 0.0:
        with np.errstate(divide="ignore", invalid="ignore"):
            t_star = np.log(r * K / (q * S)) / (r - q)
        inside = np.isfinite(t_star) & (t_star > 0.0) & (t_star < T)
        best = np.where(inside, np.maximum(best, gain(np.where(inside, t_star, 0.0))), best)
    return best


def _european_put(S: np.ndarray, K: np.ndarray, T: float, r: float, q: float, sigma: float) -> np.ndarray:
    """Black-Scholes-Merton put for positive ``S``, ``K``, ``T`` and ``sigma`` (as :func:`black_scholes.price`,
    without its input checks, which the hot path has already made)."""
    v = sigma * math.sqrt(T)
    dr, dq = math.exp(-r * T), math.exp(-q * T)
    dp = (np.log(S / K) + (r - q) * T) / v + 0.5 * v
    value = K * dr * ndtr(v - dp) - S * dq * ndtr(-dp)
    return np.maximum(value, K * dr - S * dq)  # cancellation can leave a hair below the no-arbitrage floor


def _price_puts(
    S: np.ndarray, K: np.ndarray, T: float, r: float, q: float, sigma: float, scheme: Scheme
) -> tuple[np.ndarray, np.ndarray, str, float, str | None]:
    """American puts sharing ``(T, r, q, sigma)``.

    Returns ``(value, european, method, b, form)``: ``b`` is the exercise boundary today over the strike (0 when
    early exercise is never optimal, NaN when the method does not compute it) and ``form`` the fixed-point form.
    """
    intrinsic = np.maximum(K - S, 0.0)
    if T <= 0.0:
        return intrinsic, intrinsic, "intrinsic", math.nan, None
    regime = _put_regime(r, q)
    live = (K > 0.0) & (S > 0.0)
    if regime != "single" or sigma == 0.0 or not live.all():
        european = np.asarray(bs.price(S, K, T, r, sigma, q, "put"), dtype=float).reshape(S.shape)
    else:
        european = _european_put(S, K, T, r, q, sigma)
    if regime == "european":
        return np.maximum(european, intrinsic), european, "european", 0.0, None
    if sigma == 0.0:
        value = _deterministic_put(S, K, T, r, q)
        return np.maximum(value, intrinsic), european, "exact", math.nan, None
    # A zero spot stays at zero: the put pays its whole strike, at once (r >= 0) or at expiry (r < 0).
    value = np.where(S == 0.0, K * max(1.0, math.exp(-r * T)), intrinsic)
    if regime == "double":
        for i in np.flatnonzero(live):
            value[i] = binomial.extrapolated_price(S[i], K[i], T, r, sigma, q, "put", True, _TREE_STEPS)
        return value, european, "tree", math.nan, None
    h, form = _put_boundary(T, r, q, sigma, scheme)
    b_today = _put_x(r, q) * math.exp(-math.sqrt(h[-1]))
    s = np.where(live, S / np.where(live, K, 1.0), 0.0)
    holding = live & (s > b_today)  # below the boundary the put is exercised: its value is K - S
    if holding.all():
        premium = K * _premium(s, T, r, q, sigma, h, scheme)
        value = np.maximum(european + np.maximum(premium, 0.0), intrinsic)
    elif holding.any():
        premium = K[holding] * _premium(s[holding], T, r, q, sigma, h, scheme)
        value[holding] = np.maximum(european[holding] + np.maximum(premium, 0.0), intrinsic[holding])
    return value, european, "alo", b_today, form


_NAMES = ("S", "K", "T", "r", "sigma", "q")


def _check(values: tuple[np.ndarray, ...] | tuple[float, ...]) -> None:
    """NaN or infinity names its field; spot, strike and volatility must be non-negative."""
    for name, x in zip(_NAMES, values, strict=True):
        if not np.all(np.isfinite(x)):
            raise ValueError(f"{name} must be finite (no NaN or infinity)")
    if np.any(np.less(values[0], 0.0)) or np.any(np.less(values[1], 0.0)):
        raise ValueError("spot and strike must be non-negative")
    if np.any(np.less(values[4], 0.0)):
        raise ValueError("volatility must be non-negative")


def _price_group(S, K, T, r, sigma, q, phi, scheme):
    """Value options of one type that share ``(T, r, sigma, q)``: a call is the put ``P(K, S; q, r)``."""
    if phi > 0:
        return _price_puts(K, S, T, q, r, sigma, scheme)
    return _price_puts(S, K, T, r, q, sigma, scheme)


def price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    *,
    option_type: str = "call",
    scheme: str | Scheme = "accurate",
) -> float | np.ndarray:
    """American option value by the Andersen-Lake-Offengenden spectral collocation engine.

    Parameters
    ----------
    S, K : array_like
        Spot and strike (non-negative, same currency).
    T : array_like
        Time to expiry in years.
    r, q : array_like
        Continuously compounded risk-free rate and dividend yield (decimals, per year).
    sigma : array_like
        Volatility (decimal, per square-root year).
    option_type : {"call", "put"}
        The option.
    scheme : {"fast", "accurate", "high"} or Scheme
        Discretisation (:data:`SCHEMES`). On the 4,495-put grid of Healy (2021, Table 5), "accurate" is within
        5e-6 RMSE of QuantLib's QdFp high-precision engine (docs/methods/american_alo.md).

    Returns
    -------
    float or numpy.ndarray
        The value, in the currency of ``S`` and ``K``; a float when every input is a scalar. Inputs broadcast;
        options that share ``(T, r, sigma, q)`` share one exercise boundary.

    Notes
    -----
    In the double-boundary regime (:func:`exercise_regime`) the value comes from
    :func:`binomial.extrapolated_price`; :func:`price_details` reports it.

    Examples
    --------
    The example of Andersen et al. (2016): S = K = 100, r = q = 5 %, sigma = 25 %, one year. Its early-exercise
    premium over the European value is 0.10695270275 (QuantLib's QdFp engine at its tightest settings agrees to
    1e-11).

    >>> round(price(100, 100, 1.0, 0.05, 0.25, 0.05, option_type="put"), 6)
    9.569445

    A strike strip at one expiry solves one exercise boundary:

    >>> np.round(price(100, [90, 100, 110], 0.5, 0.05, 0.3, 0.02, option_type="put"), 4).tolist()
    [3.5577, 7.7297, 13.795]
    """
    phi = bs.option_sign(option_type)
    chosen = _scheme(scheme)
    if all(isinstance(x, (int, float, np.number)) for x in (S, K, T, r, sigma, q)):
        scalars = tuple(float(x) for x in (S, K, T, r, sigma, q))
        _check(scalars)
        S1, K1 = np.array([scalars[0]]), np.array([scalars[1]])
        return float(_price_group(S1, K1, *scalars[2:], phi, chosen)[0][0])
    arrays = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q)))
    _check(tuple(arrays))
    shape = arrays[0].shape
    S_, K_, T_, r_, sigma_, q_ = (np.ascontiguousarray(x).reshape(-1) for x in arrays)
    out = np.empty(S_.shape)
    if not S_.size:
        return out.reshape(shape)
    if all(np.ndim(x) == 0 for x in (T, r, sigma, q)):  # a strip or a profile: one boundary
        out[:] = _price_group(S_, K_, float(T_[0]), float(r_[0]), float(sigma_[0]), float(q_[0]), phi, chosen)[0]
    else:
        keys = np.stack([T_, r_, sigma_, q_], axis=1)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        for g, (t, rate, vol, div) in enumerate(unique):
            rows = np.flatnonzero(inverse.reshape(-1) == g)
            out[rows] = _price_group(S_[rows], K_[rows], float(t), float(rate), float(vol), float(div), phi, chosen)[0]
    return float(out[0]) if not shape else out.reshape(shape)


def price_details(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    option_type: str = "call",
    scheme: str | Scheme = "accurate",
) -> AmericanPrice:
    """The value of one American option, with the engine that produced it and the exercise boundary today.

    Parameters are those of :func:`price`, as scalars.

    Returns
    -------
    AmericanPrice
        ``value`` equals :func:`price`; ``method`` and ``fallback`` say which engine was used, ``critical_spot``
        where exercise starts today.

    Examples
    --------
    >>> details = price_details(100, 100, 1.0, -0.01, 0.2, -0.02, option_type="put")
    >>> details.method, details.fallback, round(details.value, 4)
    ('tree', True, 7.6253)
    >>> details = price_details(100, 100, 1.0, 0.05, 0.25, 0.05, option_type="put")
    >>> details.method, details.fixed_point, round(details.critical_spot, 3)
    ('alo', 'density', 64.792)
    """
    phi = bs.option_sign(option_type)
    chosen = _scheme(scheme)
    if any(np.ndim(x) for x in (S, K, T, r, sigma, q)):
        raise ValueError("price_details takes scalars; use price for arrays")
    scalars = tuple(float(x) for x in (S, K, T, r, sigma, q))
    _check(scalars)
    S1, K1 = np.array([scalars[0]]), np.array([scalars[1]])
    value, european, method, b, form = _price_group(S1, K1, *scalars[2:], phi, chosen)
    # A put is exercised at or below K b. The call is its symmetric put, with spot K and strike S, exercised when
    # K <= S b, that is at or above S = K / b (infinity when b = 0: never).
    critical = scalars[1] * b if phi < 0 else (math.inf if b == 0.0 else scalars[1] / b)
    return AmericanPrice(
        value=float(value[0]),
        european=float(european[0]),
        premium=float(value[0] - european[0]),
        method=method,
        fallback=method == "tree",
        critical_spot=critical,
        fixed_point=form,
        scheme=chosen.name,
    )


def exercise_boundary(
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    option_type: str = "call",
    scheme: str | Scheme = "accurate",
    tau: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The early-exercise boundary: the critical spot at each time to expiry.

    Parameters
    ----------
    K, T, r, sigma, q : float
        Strike, time to expiry (years), rate, volatility and dividend yield, as in :func:`price`.
    option_type : {"call", "put"}
        A put is exercised when the spot is at or below the boundary, a call when it is at or above it.
    scheme : {"fast", "accurate", "high"} or Scheme
        Discretisation (:data:`SCHEMES`).
    tau : array_like, optional
        Times to expiry in ``[0, T]`` (years) at which to evaluate it. By default 101 points from 0 to ``T``,
        uniform in ``sqrt(tau)``, so that they are densest near expiry, where the boundary moves fastest.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``(tau, B)``: times to expiry in years, ascending, and the boundary in the currency of ``K``. ``B(0)`` is
        ``K min(1, r / q)`` for a put and ``K max(1, r / q)`` for a call. Where early exercise is never optimal,
        ``B`` is 0 for a put and infinity for a call.

    Raises
    ------
    ValueError
        In the double-boundary regime (:func:`exercise_regime`), which this engine does not solve, or when ``K``,
        ``T`` or ``sigma`` is not positive.

    Examples
    --------
    >>> tau, b = exercise_boundary(100, 1.0, 0.05, 0.25, 0.05, option_type="put")
    >>> round(float(b[0]), 6), round(float(b[-1]), 3), bool(np.all(np.diff(b) <= 0))
    (100.0, 64.792, True)
    """
    phi = bs.option_sign(option_type)
    chosen = _scheme(scheme)
    K, T, r, sigma, q = (float(x) for x in (K, T, r, sigma, q))
    for name, x in (("K", K), ("T", T), ("r", r), ("sigma", sigma), ("q", q)):
        if not math.isfinite(x):
            raise ValueError(f"{name} must be finite (no NaN or infinity)")
    if K <= 0.0 or T <= 0.0 or sigma <= 0.0:
        raise ValueError("the exercise boundary needs a positive strike, time to expiry and volatility")
    if tau is None:
        times = T * np.linspace(0.0, 1.0, 101) ** 2
    else:
        times = np.asarray(tau, dtype=float).reshape(-1)
        if not np.all(np.isfinite(times)) or np.any(times < 0.0) or np.any(times > T):
            raise ValueError("tau must lie in [0, T]")
    put_r, put_q = (q, r) if phi > 0 else (r, q)
    regime = _put_regime(put_r, put_q)
    if regime == "double":
        raise ValueError(
            "two exercise boundaries (a put with q < r < 0, a call with r < q < 0): this engine solves one; "
            "price() values the option on a tree instead"
        )
    if regime == "european":
        return times, np.full(times.shape, np.inf if phi > 0 else 0.0)
    h, _ = _put_boundary(T, put_r, put_q, sigma, chosen)
    b = _boundary_at(h, 2.0 * np.sqrt(times / T) - 1.0, chosen.n, _put_x(put_r, put_q))
    return times, K / b if phi > 0 else K * b
