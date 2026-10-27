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
  ``D``; the *density form* (FP-A), used otherwise, has normal densities and converges faster when ``r`` is close to
  ``q`` (at ``r = q`` the value settles to 1e-12 in 8 iterations, with the cdf form in 14). The ``fixed_point``
  option picks one form for every input, so that bumped prices near ``|r - q| = 1e-3`` do not jump;
* ``B`` is represented by ``H = ln(B / X)^2`` on ``n + 1`` Chebyshev-Lobatto nodes in ``z = 2 sqrt(tau / T) - 1``,
  where ``X = K min(1, r / q)`` is the boundary at expiry: ``H`` is smooth in ``sqrt(tau)`` although ``B`` is not;
* the iteration starts from Li's (2010) QD+ approximation of the boundary, takes one Jacobi-Newton step and then
  ``m - 1`` plain fixed-point steps;
* every integral, over ``u`` in ``[0, tau]`` for the boundary and over ``t`` in ``[0, T]`` for the premium, is a
  Gauss-Legendre rule in ``theta`` after the cosine map ``u = tau (1 - cos theta) / 2``. The map makes both ends of
  the interval smooth (``sqrt(u)`` and ``sqrt(tau - u)`` are ``sqrt(tau)`` times ``sin(theta / 2)`` and
  ``cos(theta / 2)``), so the rule converges spectrally. Discount and growth factors enter as ``e^{-r (tau - u)}``
  and the like, never above 1, so that no rate or expiry overflows them.

A scheme fixes ``(l, m, n, p)``: ``l`` quadrature points for the boundary integrals, ``m`` iterations, ``n + 1``
collocation nodes and ``p`` quadrature points for the premium (:data:`SCHEMES`). The error of a scheme grows with
the stiffness ``max(T c^2 / sigma^2, T max(|r|, |q|))``, ``c = max(|r|, |q|, |r - q|)``: beyond 5 the "fast"
scheme hands the option to "accurate", and beyond 25 "accurate" hands it to "high". Every value is then kept
within the no-arbitrage bounds (the European, intrinsic and best deterministic-exercise values below it; the strike
and the perpetual put above it).

Calls are priced as puts by McDonald and Schroder's (1998) symmetry ``C(S, K; r, q) = P(K, S; q, r)``. The
boundary of a put is proportional to its strike, so the normalised boundary ``B / K`` is cached per
``(T, r, q, sigma, scheme, form)``: a strike strip at one expiry (or the spots of a profile) solves one boundary.

Regimes (Healy, 2021; Andersen and Lake, 2021)
----------------------------------------------
* ``r <= 0`` and ``q >= r`` (a put; ``q <= 0`` and ``r >= q`` for a call): early exercise is never optimal, and the
  value is the European one.
* ``q < r < 0`` for a put (``r < q < 0`` for a call): the exercise region lies between two boundaries, which this
  engine (like QuantLib's) does not solve. The value falls back to :func:`binomial.extrapolated_price`, and
  :func:`price_details` flags it.
* Otherwise one boundary, solved as above. ``sigma = 0`` is valued exactly (the best deterministic exercise date),
  and so are a total volatility ``sigma sqrt(T)`` below 1e-8 and an expiry so short that the premium is below
  1e-16 of the strike (see :func:`price`).

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
* R. C. Merton (1973), "Theory of rational option pricing", *Bell Journal of Economics and Management Science*
  4(1), 141-183, doi:10.2307/3003143 (the perpetual put).
* K. Spanderen (2022), QuantLib ``QdFpAmericanEngine``, https://github.com/lballabio/QuantLib (BSD-3-Clause).

The method, its error and its limits are written up in ``docs/methods/american_alo.md``.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple, cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import log_ndtr, ndtr

from . import binomial
from . import black_scholes as bs

__all__ = [
    "GREEK_BUMPS",
    "GREEK_NAMES",
    "SCHEMES",
    "AmericanGreeks",
    "AmericanPrice",
    "Scheme",
    "exercise_boundary",
    "exercise_regime",
    "greeks",
    "price",
    "price_details",
]

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
#: Below this ``|r - q|`` the density form of the fixed-point equation replaces the cdf form (as in QuantLib).
_DENSITY_FORM_BELOW = 1e-3
#: The forms of the fixed-point equation that ``fixed_point`` accepts.
_FORMS = ("cdf", "density")
#: QD+ initial guess: Halley steps per node and their relative tolerance.
_QD_PLUS_STEPS = 12
_QD_PLUS_TOLERANCE = 1e-9
#: Normalised boundaries kept in the cache (one per expiry, rate, dividend yield, volatility, scheme and form).
_CACHE_SIZE = 4096
#: Steps of the Leisen-Reimer tree in the fallback (Richardson with twice as many, made odd: 2,001 and 4,003).
_TREE_STEPS = 2001
#: A boundary whose last fixed-point iteration still moved ``ln B`` by more than this at some node has not converged,
#: and the option is valued on the tree instead. On the 4,495-put grid the largest last step is 5.5e-4 (fast),
#: 1.2e-5 (accurate) and 2.8e-7 (high); the iterations that failed in tests/test_american.py moved it by 4e-2 to 1.3.
_MAX_LAST_STEP = 1e-2
#: The smallest normal double. Rates and yields smaller in magnitude count as zero (they move no value by more than
#: 1e-307 of the strike per year, and subnormal arithmetic derails the boundary iteration); a put with
#: ``X / K = r / q`` below it is exercised only at spots under 2.2e-308 K, and is valued as European.
_TINY = float(np.finfo(float).tiny)
#: Below this total volatility ``sigma sqrt(T)`` the option is valued at zero volatility, floored at its European
#: value. The error is of the order of vega times sigma, which for a European option is at most
#: ``S e^{-qT} sigma sqrt(T) / sqrt(2 pi)``, 4e-9 ``S e^{-qT}`` here; the tests find the engine and this closed form
#: within 4e-9 ``S`` of each other across the threshold. The engine is no more accurate there: its boundary layers
#: are thinner than any of its quadrature rules can resolve.
_MIN_TOTAL_VOL = 1e-8
#: Below this ``max(|r|, |q|) T`` the early-exercise premium, at most ``(|r| K + |q| S) T e^{max(|r|, |q|) T}``,
#: is below rounding, and the value is ``max(European, intrinsic)``.
_NEGLIGIBLE_CARRY = 1e-16
#: Stiffness ``max(T c^2 / sigma^2, T max(|r|, |q|))`` above which a named scheme hands the option to the next one.
#: Measured over 2,500 random puts (expiries 0.01 to 100 years, volatilities 1 % to 150 %, rates 0 to 30 %,
#: dividend yields -15 % to 35 %) against a 65-node scheme: up to a stiffness of 5, fast is within 3.9e-4 of
#: ``max(1, V)`` (beyond 8 it reaches 3e-2) and accurate within 7.6e-6; up to 25 accurate is within 2.7e-5.
_HAND_OVER = {"fast": (5.0, "accurate"), "accurate": (25.0, "high")}
#: Upper and lower no-arbitrage bounds this close (relative to the upper one) pin the value: no tree is built.
_BOUNDS_MEET = 1e-14


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
        fallback), ``"exact"`` (a closed form: zero or negligible volatility, or a negligible premium, see
        :func:`price`) or ``"intrinsic"`` (expired).
    fallback : bool
        True when the engine could not price the option and :func:`binomial.extrapolated_price` did.
    fallback_reason : str or None
        Why it could not: ``"double boundary"`` (the regime of :func:`exercise_regime`, which the engine does not
        solve) or ``"not converged"`` (the boundary iteration was still moving, see ``last_step``).
    critical_spot : float
        The exercise boundary today: a put is exercised at or below it, a call at or above it. ``0`` (put) or
        ``inf`` (call) when early exercise is never optimal; NaN where it is not computed (``tree``, ``exact``,
        ``intrinsic``).
    fixed_point : str or None
        ``"cdf"`` or ``"density"``: the form of the boundary equation, when a boundary was solved.
    last_step : float
        Convergence check: the largest change of ``ln B`` over the collocation nodes in the last fixed-point
        iteration, when a boundary was solved (NaN otherwise). Above 1e-2 the boundary is rejected.
    scheme : str
        The name of the scheme that solved the boundary: the one asked for, or the next one when "fast" or
        "accurate" handed the option over (see :func:`price`).
    """

    value: float
    european: float
    premium: float
    method: str
    fallback: bool
    fallback_reason: str | None
    critical_spot: float
    fixed_point: str | None
    last_step: float
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


def _flush(x: float) -> float:
    """A rate or yield, with magnitudes below the smallest normal double (:data:`_TINY`) set to zero."""
    return 0.0 if abs(x) < _TINY else x


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
    return np.asarray(np.linalg.solve(at_nodes.T, at_x.T).T)


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


def _form(fixed_point: str | None) -> str | None:
    if fixed_point is None:
        return None
    form = str(fixed_point).strip().lower()
    if form not in _FORMS:
        raise ValueError(f"fixed_point must be 'cdf', 'density' or None, got {fixed_point!r}")
    return form


def _stiffness(T: float, r: float, q: float, sigma: float) -> float:
    """``max(T c^2 / sigma^2, T max(|r|, |q|))`` with ``c = max(|r|, |q|, |r - q|)``: drift against diffusion over
    the life of the option, and discounting over it. The boundary moves on the time scales ``sigma^2 / c^2`` and
    ``1 / max(|r|, |q|)``; the stiffer the option, the closer to expiry (``z = -1``) the interpolant must bend."""
    carry = max(abs(r), abs(q))
    drift = max(carry, abs(r - q))
    return max(T * (drift / sigma) ** 2, T * carry)


def _solving_scheme(scheme: Scheme, T: float, r: float, q: float, sigma: float) -> Scheme:
    """The scheme that solves the boundary: a named "fast" or "accurate" scheme hands over beyond its stiffness
    limit (:data:`_HAND_OVER`); any other scheme is used as given."""
    stiffness = _stiffness(T, r, q, sigma)
    while scheme.name in _HAND_OVER and scheme is SCHEMES[scheme.name]:
        limit, successor = _HAND_OVER[scheme.name]
        if stiffness <= limit:
            break
        scheme = SCHEMES[successor]
    return scheme


# ----------------------------------------------------------------------------------------------- boundary


def _npdf(x: np.ndarray) -> np.ndarray:
    return np.asarray(_INV_SQRT_2PI * np.exp(-0.5 * x * x))


def _qd_plus(tau: np.ndarray, r: float, q: float, sigma: float, x_max: float) -> np.ndarray:
    """Li's (2010) QD+ approximation of the normalised put boundary ``B / K`` at ``tau`` (the initial guess).

    The QD+ boundary solves ``F(b) = (1 - e^{-q tau} N(-d+)) b - beta (1 - b - p(b)) + alpha Theta(b) / e^{-r tau}
    = 0``, where ``p`` and ``Theta`` are the European put and its calendar-time theta at spot ``b`` and unit strike,
    and ``alpha``, ``beta`` and ``lambda`` are Li's coefficients (as in QuantLib's QdPlus engine). Expanded, ``F`` is
    ``c1 b - beta - c2 b N(-d+) + c3 N(-d-) - c4 b n(d+)``, which Halley's method solves from ``b = X`` on every
    node at once. A node that does not converge keeps its last iterate, and a node whose coefficients are not
    finite (rates over centuries, volatilities near zero) starts from ``X``; the fixed-point iterations correct it.
    """
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        sigma2 = sigma * sigma
        v = sigma * np.sqrt(tau)
        # e^{-r tau} <= 1 (r >= 0 here); e^{-q tau} is capped at e^700, which only a guess for q tau < -700 feels.
        dr, dq = np.exp(-r * tau), np.exp(np.minimum(-q * tau, 700.0))
        rt = r * tau
        # r / (1 - e^{-r tau}) = 1 / (tau g(r tau)) with g(x) = (1 - e^{-x}) / x, which is 1 at x = 0.
        g = np.where(rt == 0.0, 1.0, -np.expm1(-rt) / np.where(rt == 0.0, 1.0, rt))
        ddr = 1.0 / (tau * g)
        omega = 2.0 * (r - q) / sigma2
        root = np.sqrt((omega - 1.0) ** 2 + 8.0 * ddr / sigma2)
        lam = 0.5 * (-(omega - 1.0) - root)
        lam_prime = 2.0 * ddr * ddr / (sigma2 * root)
        denominator = 2.0 * lam + omega - 1.0
        a = 2.0 / (sigma2 * denominator)  # alpha / e^{-r tau}
        beta = a * dr * (ddr + lam_prime / denominator) - lam
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
            new = np.maximum(x_min, b - f / f1 / (1.0 - 0.5 * f * f2 / (f1 * f1)))  # a flat F gives no step
            new = np.where(np.isfinite(new), new, b)
            done = np.max(np.abs(new - b)) <= _QD_PLUS_TOLERANCE * x_max
            b = new
            if done:
                break
    return np.asarray(np.minimum(b, x_max))


@functools.lru_cache(maxsize=_CACHE_SIZE)
def _put_boundary(
    T: float, r: float, q: float, sigma: float, scheme: Scheme, form: str | None = None
) -> tuple[np.ndarray, str, float]:
    """``H = ln(B / X)^2`` of the normalised put boundary at the scheme's nodes, the fixed-point form used and the
    last iteration's largest change of ``ln B`` (the convergence check).

    For ``r > 0`` (or ``r = 0 > q``) and ``sigma > 0``. ``form`` is "cdf", "density" or None (by ``|r - q|``).
    The array is read-only: it is shared through the cache.

    With ``a = max(q, 0)``, the fixed point ``B = K e^{-(r - q) tau} N / D`` is solved as
    ``B = K e^{(q - a) tau} (e^{-r tau} N) / (e^{-a tau} D)``: every exponential factor is then at most 1.
    """
    grid = _grid(scheme)
    x_max = _put_x(r, q)
    log_x = math.log(x_max)
    density = abs(r - q) < _DENSITY_FORM_BELOW if form is None else form == "density"
    tau = T * grid.tau_share  # nodes 1..n
    sqrt_tau = np.sqrt(tau)
    v = sigma * sqrt_tau
    carry = r - q
    a = max(q, 0.0)
    # Quadrature points: u = tau a_j and tau - u = tau (1 - a_j).
    u = tau[:, None] * grid.u_share[None, :]
    rest = tau[:, None] - u
    v_rest = sigma * np.sqrt(rest)
    drift_rest = carry * rest
    scale_r, scale_q, growth = np.exp(-r * tau), np.exp(-a * tau), np.exp((q - a) * tau)  # each at most 1
    discount_r = np.exp(-r * rest)  # e^{r u} e^{-r tau}
    discount_q = np.exp(q * u - a * tau[:, None])  # e^{q u} e^{-a tau}
    w_r = tau[:, None] * grid.weight[None, :] * discount_r
    w_q = tau[:, None] * grid.weight[None, :] * discount_q
    if density:
        dw = (sqrt_tau / sigma)[:, None] * grid.density_weight[None, :]
        dw_r, dw_q = dw * discount_r, dw * discount_q

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
            numerator = scale_r * pdf_dm0 / v + r * np.sum(dw_r * _npdf(dm), axis=1)
            denominator = scale_q * (pdf_dp0 / v + ndtr(dp0)) + q * np.sum(dw_q * _npdf(dp) + w_q * cdf_dp, axis=1)
        else:
            numerator = scale_r * ndtr(dm0) + r * np.sum(w_r * ndtr(dm), axis=1)
            denominator = scale_q * ndtr(dp0) + q * np.sum(w_q * ndtr(dp), axis=1)
        b = x_max * np.exp(-root_h)
        # A node whose update is not a positive number (a vanishing denominator, a Newton step through zero)
        # keeps its value for this iteration.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            f = growth * numerator / denominator
            if iteration == 0:
                # Jacobi-Newton: the derivative of f in B(tau_i) through the terms outside the integrals.
                if density:
                    d_num = -pdf_dm0 * dm0 / (b * v * v)
                    d_den = -pdf_dp0 * dp0 / (b * v * v) + pdf_dp0 / (b * v)
                else:
                    d_num = pdf_dm0 / (b * v)
                    d_den = pdf_dp0 / (b * v)
                d_num, d_den = scale_r * d_num, scale_q * d_den
                slope = growth * (d_num / denominator - d_den * numerator / (denominator * denominator))
                newton = b - (f - b) / (slope - 1.0)
                f = np.where(np.isfinite(newton) & (newton > 0.0), newton, f)
        new = np.where(np.isfinite(f) & (f > 0.0), f, b)
        h[1:] = (np.log(new) - log_x) ** 2
    # sqrt(H) = ln(X / B): its change is the change of ln B the last iteration made.
    last_step = float(np.max(np.abs(np.sqrt(h[1:]) - root_h)))
    h.flags.writeable = False
    return h, "density" if density else "cdf", last_step


def _boundary_at(h: np.ndarray, z: np.ndarray, n: int, x_max: float) -> np.ndarray:
    """The normalised boundary ``X exp(-sqrt(H(z)))`` at points ``z`` in [-1, 1]."""
    values = _chebyshev_matrix(z, n) @ h
    return np.asarray(x_max * np.exp(-np.sqrt(np.maximum(values, 0.0))))


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
    if q >= 0.0:
        dividend_leg = (weight * q * np.exp(-q * t)) @ ndtr(-dp).T
    else:  # e^{-q t} N(-d+) is at most B / S, but e^{-q t} alone can overflow: take it in logarithms
        dividend_leg = np.sum(weight * q * np.exp(-q * t + log_ndtr(-dp)), axis=1)
    return np.asarray(rate_leg - s * dividend_leg)


# -------------------------------------------------------------------------------------------------- prices


def _forward_put(S: np.ndarray, K: np.ndarray, t: float | np.ndarray, r: float, q: float) -> np.ndarray:
    """``max(K e^{-r t} - S e^{-q t}, 0)``. Past ``-q t = 700``, where ``e^{-q t}`` nears overflow, it is taken as
    ``K e^{-r t} (1 - e^{ln(S / K) + (r - q) t})^+`` (0 when ``K = 0``)."""
    if np.max(-q * np.asarray(t)) <= 700.0:
        return np.asarray(np.maximum(K * np.exp(-r * t) - S * np.exp(-q * t), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):  # ln 0; ln 0 - ln 0 where S = K = 0 (masked)
        exponent = np.minimum(np.log(S) - np.log(K) + (r - q) * t, 0.0)
        value = K * np.exp(-r * t) * -np.expm1(exponent)
    return np.where(K > 0.0, value, 0.0)


def _deterministic_put(S: np.ndarray, K: np.ndarray, T: float, r: float, q: float) -> np.ndarray:
    """Zero-volatility American put: the best of exercising now, at expiry, or at the stationary date of
    ``K e^{-r t} - S e^{-q t}`` (Healy 2021; QuantLib's QdPlus engine does the same). It is also a lower bound at any
    volatility: exercising at a fixed date ``t`` is worth ``e^{-rt} E[(K - S_t)^+] >= (K e^{-rt} - S e^{-qt})^+``."""
    best = np.maximum(np.maximum(K - S, 0.0), _forward_put(S, K, T, r, q))
    if r != q and ((r > 0.0 and q > 0.0) or (r < 0.0 and q < 0.0)):
        # The gain is stationary at t* = ln(r K / (q S)) / (r - q), taken in logarithms so that no ratio overflows.
        with np.errstate(divide="ignore", invalid="ignore"):  # a zero spot or strike has no stationary date
            t_star = (math.log(abs(r)) - math.log(abs(q)) + np.log(K) - np.log(S)) / (r - q)
        inside = np.isfinite(t_star) & (t_star > 0.0) & (t_star < T)
        best = np.where(inside, np.maximum(best, _forward_put(S, K, np.where(inside, t_star, 0.0), r, q)), best)
    return np.asarray(best)


def _european_put(S: np.ndarray, K: np.ndarray, T: float, r: float, q: float, sigma: float) -> np.ndarray:
    """Black-Scholes-Merton put for non-negative ``S``, ``K`` and ``sigma`` and positive ``T`` (the value of
    :func:`black_scholes.price`), with the discount factors taken into the logarithms of the probabilities,
    ``K e^{-rT + ln N(-d-)}`` and ``S e^{-qT + ln N(-d+)}``, so that neither a large ``-q T`` nor a subnormal
    ``sigma`` overflows (``S e^{-qT} N(-d+)`` is at most ``K e^{-rT}``)."""
    forward = _forward_put(S, K, T, r, q)
    v = sigma * math.sqrt(T)
    live = (S > 0.0) & (K > 0.0)
    if v == 0.0 or not live.any():
        return forward
    s, k = np.where(live, S, 1.0), np.where(live, K, 1.0)
    with np.errstate(over="ignore"):  # a total volatility near zero sends d+- to +-infinity
        dp = (np.log(s) - np.log(k) + (r - q) * T) / v + 0.5 * v
    dm = dp - v
    value = k * np.exp(log_ndtr(-dm) - r * T) - s * np.exp(log_ndtr(-dp) - q * T)
    return np.where(live, np.maximum(value, forward), forward)  # cancellation can leave a hair below the floor


def _perpetual_put(s: np.ndarray, r: float, q: float, sigma: float) -> np.ndarray:
    """Perpetual American put over its strike (Merton 1973), for ``r > 0`` and normalised spots ``s = S / K``:
    exercised at or below ``b = beta / (beta - 1)`` and worth ``(1 - b) (s / b)^beta`` above it, where ``beta < 0``
    solves ``sigma^2 beta (beta - 1) / 2 + (r - q) beta - r = 0``. No finite expiry is worth more."""
    a = (r - q) / sigma**2 - 0.5
    c = 2.0 * r / sigma**2
    root = math.hypot(a, math.sqrt(c))
    beta = -a - root if a >= 0.0 else -c / (root - a)  # the same root, without cancellation when a < 0
    b = 1.0 / (1.0 - 1.0 / beta)
    if not b > 0.0:  # beta rounds to -0 (r / sigma^2 below 1e-308): the boundary is 0 and the bound the strike
        return np.ones_like(s)
    with np.errstate(divide="ignore"):  # a zero spot is exercised (below b)
        log_ratio = np.maximum(np.log(s) - math.log(b), 0.0)
    return np.where(s > b, (1.0 - b) * np.exp(beta * log_ratio), 1.0 - s)


def _upper_put(s: np.ndarray, K: np.ndarray, T: float, r: float, q: float, sigma: float) -> np.ndarray:
    """No-arbitrage upper bound of American puts with normalised spots ``s = S / K``: the strike, received at once
    or, when ``r < 0``, at expiry, and for ``r > 0`` the perpetual put (as in :func:`binomial._american_bounds`)."""
    if r > 0.0:
        return np.asarray(K * np.minimum(1.0, _perpetual_put(s, r, q, sigma)))
    return K * math.exp(-r * T)


class _Puts(NamedTuple):
    """American puts that share ``(T, r, q, sigma)``, and how they were valued (see :class:`AmericanPrice`)."""

    value: np.ndarray
    european: np.ndarray
    method: str
    boundary: float  # the exercise boundary today over the strike: 0 when never exercised, NaN when not computed
    scheme: str  # the name of the scheme that solved the boundary (the one asked for when none was solved)
    form: str | None = None
    reason: str | None = None
    last_step: float = math.nan


def _tree_puts(
    value: np.ndarray, live: np.ndarray, S: np.ndarray, K: np.ndarray, T: float, r: float, sigma: float, q: float
) -> np.ndarray:
    """The fallback: :func:`binomial.extrapolated_price` for the live rows of ``value`` (changed in place). It costs
    two Leisen-Reimer trees (2,001 and 4,003 steps), about 60 ms, per option."""
    for i in np.flatnonzero(live):
        value[i] = binomial.extrapolated_price(S[i], K[i], T, r, sigma, q, "put", True, _TREE_STEPS)
    return value


def _price_puts(
    S: np.ndarray, K: np.ndarray, T: float, r: float, q: float, sigma: float, scheme: Scheme, form: str | None
) -> _Puts:
    """American puts sharing ``(T, r, q, sigma)``."""
    intrinsic = np.maximum(K - S, 0.0)
    if T <= 0.0:
        return _Puts(intrinsic, intrinsic, "intrinsic", math.nan, scheme.name)
    r, q = _flush(r), _flush(q)
    regime = _put_regime(r, q)
    live = (K > 0.0) & (S > 0.0)
    european = _european_put(S, K, T, r, q, sigma)
    # Where the put is exercised it is worth its intrinsic value, never less than the European one. Only a
    # boundary placed too high could make the two differ there (it happens just before expiry at extreme
    # volatilities, by a few 1e-9). A zero spot stays at zero: the put pays K at once (r >= 0) or at expiry (r < 0).
    floor = np.maximum(european, intrinsic)
    if regime == "european":
        return _Puts(floor, european, "european", 0.0, scheme.name)
    if sigma * math.sqrt(T) < _MIN_TOTAL_VOL:
        return _Puts(np.maximum(_deterministic_put(S, K, T, r, q), floor), european, "exact", math.nan, scheme.name)
    if max(abs(r), abs(q)) * T < _NEGLIGIBLE_CARRY:
        return _Puts(floor, european, "exact", math.nan, scheme.name)
    s = np.where(live, S / np.where(live, K, 1.0), 0.0)
    upper = _upper_put(s, K, T, r, q, sigma)
    # Where the bounds meet (the European value reaches the strike's worth at extreme total volatility) there is
    # nothing to solve; the tree is not even built there (it has no arbitrage-free lattice at such volatilities).
    open_rows = live & (upper - floor > _BOUNDS_MEET * upper)
    if regime == "double":
        if not open_rows.any():
            return _Puts(floor, european, "exact", math.nan, scheme.name)
        value = _tree_puts(floor, open_rows, S, K, T, r, sigma, q)
        return _Puts(value, european, "tree", math.nan, scheme.name, None, "double boundary")
    if _put_x(r, q) < _TINY:
        return _Puts(floor, european, "european", 0.0, scheme.name)
    solver = _solving_scheme(scheme, T, r, q, sigma)
    h, used_form, last_step = _put_boundary(T, r, q, sigma, solver, form)
    if last_step > _MAX_LAST_STEP:
        if not open_rows.any():
            return _Puts(floor, european, "exact", math.nan, solver.name, used_form, None, last_step)
        value = _tree_puts(floor, open_rows, S, K, T, r, sigma, q)
        return _Puts(value, european, "tree", math.nan, solver.name, used_form, "not converged", last_step)
    b_today = _put_x(r, q) * math.exp(-math.sqrt(h[-1]))
    holding = live & (s > b_today)  # at or below the boundary the put is exercised
    value = floor
    if holding.any():
        # Kept within the no-arbitrage bounds: the best deterministic exercise date (it includes the intrinsic
        # value) below, the strike and the perpetual put above. Moving a value onto a bound it crosses never takes
        # it farther from the true value, which lies between them.
        k, s_held, lower = K[holding], s[holding], floor[holding]
        if q > r:  # the gain K e^{-rt} - S e^{-qt} can peak inside (0, T); otherwise it peaks now or at expiry
            lower = np.maximum(lower, _deterministic_put(s_held, np.ones_like(s_held), T, r, q) * k)
        held = european[holding] + np.maximum(k * _premium(s_held, T, r, q, sigma, h, solver), 0.0)
        value[holding] = np.maximum(np.minimum(held, upper[holding]), lower)
    return _Puts(value, european, "alo", b_today, solver.name, used_form, None, last_step)


_NAMES = ("S", "K", "T", "r", "sigma", "q")


def _check(values: Sequence[float | np.ndarray]) -> None:
    """NaN or infinity names its field; spot, strike and volatility must be non-negative."""
    for name, x in zip(_NAMES, values, strict=True):
        if not np.all(np.isfinite(x)):
            raise ValueError(f"{name} must be finite (no NaN or infinity)")
    if np.any(np.less(values[0], 0.0)) or np.any(np.less(values[1], 0.0)):
        raise ValueError("spot and strike must be non-negative")
    if np.any(np.less(values[4], 0.0)):
        raise ValueError("volatility must be non-negative")


def _price_group(
    S: np.ndarray,
    K: np.ndarray,
    T: float,
    r: float,
    sigma: float,
    q: float,
    phi: int,
    scheme: Scheme,
    form: str | None,
) -> _Puts:
    """Value options of one type that share ``(T, r, sigma, q)``: a call is the put ``P(K, S; q, r)``."""
    if phi > 0:
        return _price_puts(K, S, T, q, r, sigma, scheme, form)
    return _price_puts(S, K, T, r, q, sigma, scheme, form)


def _groups(*columns: np.ndarray) -> list[tuple[tuple[float, ...], np.ndarray]]:
    """The distinct rows of ``columns`` (equal-length 1-d arrays) and the indices that share each: one sort, then a
    split where the key changes, so the cost is O(N log N) however many groups there are."""
    order = np.lexsort(columns[::-1])
    keys = np.stack(columns, axis=1)[order]
    starts = np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1
    firsts = np.concatenate(([0], starts))
    return [
        (tuple(float(x) for x in keys[first]), rows)
        for first, rows in zip(firsts, np.split(order, starts), strict=True)
    ]


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
    fixed_point: str | None = None,
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
        5e-6 RMSE of QuantLib's QdFp high-precision engine (docs/methods/american_alo.md). Outside the domain where
        they were validated, "fast" and "accurate" hand the option to the next scheme (see Notes).
    fixed_point : {"cdf", "density"} or None
        The form of the boundary equation. None (the default) takes the density form when ``|r - q| < 1e-3`` and
        the cdf form otherwise, as QuantLib does; the two converge to the same boundary, but with different
        discretisation errors, so the value jumps where the choice changes (by up to 4e-4 with "fast" and 1e-6
        with "accurate" at ten years). Fix the form to bump ``r`` or ``q`` across that line.

    Returns
    -------
    float or numpy.ndarray
        The value, in the currency of ``S`` and ``K``; a float when every input is a scalar. Inputs broadcast;
        options that share ``(T, r, sigma, q)`` share one exercise boundary.

    Notes
    -----
    * Hand-over: the error of a scheme grows with the stiffness ``max(T c^2 / sigma^2, T max(|r|, |q|))``,
      ``c = max(|r|, |q|, |r - q|)`` (long expiries, high rates or dividend yields at low volatility). Beyond a
      stiffness of 5 "fast" uses "accurate", and beyond 25 "accurate" uses "high"; :func:`price_details` names the
      scheme used. The value jumps by the error of the scheme handed over, at most about 4e-4 ``max(1, V)``.
    * Every value lies within the no-arbitrage bounds: at least the European value, the intrinsic value and the
      best deterministic exercise date; for a put at most the strike and, for ``r > 0``, the perpetual put (a call
      by symmetry). A value outside them is moved onto the nearer bound, which is never farther from the truth.
    * Closed forms (``method = "exact"``): ``sigma = 0`` is the best deterministic exercise date; below a total
      volatility ``sigma sqrt(T)`` of 1e-8 the same, floored at the European value (an error of the order of
      vega times sigma, about 4e-9 ``S``); when ``max(|r|, |q|) T < 1e-16``, ``max(European, intrinsic)``, exact
      to rounding.
    * In the double-boundary regime (:func:`exercise_regime`), and when the boundary iteration has not converged
      (its last step moved ``ln B`` by more than 1e-2 at some node, which happens only far outside the validated
      domain, for instance decades of expiry with a dividend yield of -30 %), the value comes from
      :func:`binomial.extrapolated_price`; :func:`price_details` reports it. That fallback values each option on
      its own, at about 60 ms per option: a 41-strike strip takes about 2.5 s.

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
    form = _form(fixed_point)
    inputs = (S, K, T, r, sigma, q)
    if all(isinstance(x, (int, float, np.number)) for x in inputs):
        S0, K0, T0, r0, sigma0, q0 = (float(cast(float, x)) for x in inputs)
        _check((S0, K0, T0, r0, sigma0, q0))
        puts = _price_group(np.array([S0]), np.array([K0]), T0, r0, sigma0, q0, phi, chosen, form)
        return float(puts.value[0])
    arrays = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in inputs))
    _check(arrays)
    shape = arrays[0].shape
    S_, K_, T_, r_, sigma_, q_ = (np.ascontiguousarray(x).reshape(-1) for x in arrays)
    out = np.empty(S_.shape)
    if not S_.size:
        return out.reshape(shape)
    if all(np.ndim(x) == 0 for x in (T, r, sigma, q)):  # a strip or a profile: one boundary
        group = (float(T_[0]), float(r_[0]), float(sigma_[0]), float(q_[0]))
        out[:] = _price_group(S_, K_, *group, phi, chosen, form).value
    else:
        for (T1, r1, sigma1, q1), rows in _groups(T_, r_, sigma_, q_):
            out[rows] = _price_group(S_[rows], K_[rows], T1, r1, sigma1, q1, phi, chosen, form).value
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
    fixed_point: str | None = None,
) -> AmericanPrice:
    """The value of one American option, with the engine that produced it and the exercise boundary today.

    Parameters are those of :func:`price`, as scalars.

    Returns
    -------
    AmericanPrice
        ``value`` equals :func:`price`; ``method`` and ``fallback`` say which engine was used, ``scheme`` which
        discretisation, ``critical_spot`` where exercise starts today.

    Examples
    --------
    >>> details = price_details(100, 100, 1.0, -0.01, 0.2, -0.02, option_type="put")
    >>> details.method, details.fallback_reason, round(details.value, 4)
    ('tree', 'double boundary', 7.6253)
    >>> details = price_details(100, 100, 1.0, 0.05, 0.25, 0.05, option_type="put")
    >>> details.method, details.fixed_point, round(details.critical_spot, 3), details.last_step < 1e-9
    ('alo', 'density', 64.792, True)
    >>> price_details(100, 100, 30.0, 0.1, 0.1, 0.0, option_type="put", scheme="fast").scheme
    'high'
    """
    phi = bs.option_sign(option_type)
    chosen = _scheme(scheme)
    form = _form(fixed_point)
    if any(np.ndim(x) for x in (S, K, T, r, sigma, q)):
        raise ValueError("price_details takes scalars; use price for arrays")
    S0, K0, T0, r0, sigma0, q0 = (float(x) for x in (S, K, T, r, sigma, q))
    _check((S0, K0, T0, r0, sigma0, q0))
    puts = _price_group(np.array([S0]), np.array([K0]), T0, r0, sigma0, q0, phi, chosen, form)
    # A put is exercised at or below K b. The call is its symmetric put, with spot K and strike S, exercised when
    # K <= S b, that is at or above S = K / b (infinity when b = 0: never).
    b = puts.boundary
    critical = K0 * b if phi < 0 else (math.inf if b == 0.0 else K0 / b)
    return AmericanPrice(
        value=float(puts.value[0]),
        european=float(puts.european[0]),
        premium=float(puts.value[0] - puts.european[0]),
        method=puts.method,
        fallback=puts.method == "tree",
        fallback_reason=puts.reason,
        critical_spot=critical,
        fixed_point=puts.form,
        last_step=puts.last_step,
        scheme=puts.scheme,
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
    fixed_point: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The early-exercise boundary: the critical spot at each time to expiry.

    Parameters
    ----------
    K, T, r, sigma, q : float
        Strike, time to expiry (years), rate, volatility and dividend yield, as in :func:`price`.
    option_type : {"call", "put"}
        A put is exercised when the spot is at or below the boundary, a call when it is at or above it.
    scheme : {"fast", "accurate", "high"} or Scheme
        Discretisation (:data:`SCHEMES`), handed over as in :func:`price`. Between its nodes the interpolant of
        the boundary ripples: with "fast" a put's boundary can rise with time to expiry by up to 8e-3 of its value
        at volatilities up to 60 % (2e-2 at 150 %; "accurate" 1e-7 and 3e-5); use "accurate" or "high" to plot it.
    tau : array_like, optional
        Times to expiry in ``[0, T]`` (years) at which to evaluate it. By default 101 points from 0 to ``T``,
        uniform in ``sqrt(tau)``, so that they are densest near expiry, where the boundary moves fastest.
    fixed_point : {"cdf", "density"} or None
        The form of the boundary equation, as in :func:`price`.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``(tau, B)``: times to expiry in years, ascending, and the boundary in the currency of ``K``. ``B(0)`` is
        ``K min(1, r / q)`` for a put and ``K max(1, r / q)`` for a call. Where early exercise is never optimal,
        ``B`` is 0 for a put and infinity for a call.

    Raises
    ------
    ValueError
        In the double-boundary regime (:func:`exercise_regime`), which this engine does not solve; when the
        boundary iteration has not converged (see :func:`price`); or when ``K``, ``T`` or ``sigma`` is not
        positive.

    Examples
    --------
    >>> tau, b = exercise_boundary(100, 1.0, 0.05, 0.25, 0.05, option_type="put")
    >>> round(float(b[0]), 6), round(float(b[-1]), 3), bool(np.all(np.diff(b) <= 0))
    (100.0, 64.792, True)
    """
    phi = bs.option_sign(option_type)
    chosen = _scheme(scheme)
    form = _form(fixed_point)
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
    put_r, put_q = (_flush(q), _flush(r)) if phi > 0 else (_flush(r), _flush(q))
    regime = _put_regime(put_r, put_q)
    if regime == "double":
        raise ValueError(
            "two exercise boundaries (a put with q < r < 0, a call with r < q < 0): this engine solves one; "
            "price() values the option on a tree instead"
        )
    if regime == "european" or _put_x(put_r, put_q) < _TINY:  # never exercised, or only below 2.2e-308 K
        return times, np.full(times.shape, np.inf if phi > 0 else 0.0)
    if sigma * math.sqrt(T) < _MIN_TOTAL_VOL:  # at zero volatility a put is exercised at or below X at any tau
        x = _put_x(put_r, put_q)
        return times, np.full(times.shape, K / x if phi > 0 else K * x)
    solver = _solving_scheme(chosen, T, put_r, put_q, sigma)
    h, _, last_step = _put_boundary(T, put_r, put_q, sigma, solver, form)
    if last_step > _MAX_LAST_STEP:
        raise ValueError(
            f"the boundary iteration has not converged (its last step moved ln B by {last_step:.1e}); "
            "price() values the option on a tree instead"
        )
    b = _boundary_at(h, 2.0 * np.sqrt(times / T) - 1.0, solver.n, _put_x(put_r, put_q))
    return times, K / b if phi > 0 else K * b


# ------------------------------------------------------------------------------------------------- Greeks

#: Bumps of :func:`greeks`: spot and time to expiry relative to their values; volatility, rate and dividend yield
#: absolute (decimals). With its scheme and the form of its boundary equation held fixed, the engine is smooth in
#: every input within a regime, so differences this small are stable: at S = K = 100, one year, r = 5 % and
#: sigma = 20 % the put's Greeks agree with bumps ten times smaller to 4e-7 of each Greek (and with bumps ten times
#: larger to 1.4e-5).
GREEK_BUMPS = {"spot": 1e-4, "expiry": 1e-4, "volatility": 1e-4, "rate": 1e-4}
#: The Greeks of :func:`greeks`, in the order of :data:`black_scholes.GREEKS`.
GREEK_NAMES = ("delta", "gamma", "theta", "vega", "rho", "epsilon")


@dataclass(frozen=True)
class AmericanGreeks(Mapping[str, float]):
    """First-order Greeks of an American option (see :func:`greeks`), read like a dict: ``g["delta"]``.

    Attributes
    ----------
    delta, gamma, theta, vega, rho, epsilon : float
        In the raw units of :func:`black_scholes.greeks`.
    basis : str
        ``"differences"`` (finite differences of :func:`price`), ``"closed form"`` (an option valued as European:
        the Black-Scholes-Merton Greeks) or ``"exercise value"`` (an option exercised at once: the Greeks of its
        intrinsic value, delta +-1 and the others zero).
    one_sided : tuple of str
        The Greeks taken from one side only, because the bump on the other side changed how the option is valued.
    """

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    epsilon: float
    basis: str = "differences"
    one_sided: tuple[str, ...] = ()

    def __getitem__(self, name: str) -> float:
        if name not in GREEK_NAMES:
            raise KeyError(name)
        return cast(float, getattr(self, name))

    def __iter__(self) -> Iterator[str]:
        return iter(GREEK_NAMES)

    def __len__(self) -> int:
        return len(GREEK_NAMES)


def _exercised(S: float, details: AmericanPrice, phi: int) -> bool:
    """Whether the engine exercises the option at once at spot ``S``: at or below the critical spot of a put, at
    or above that of a call."""
    if details.method != "alo":
        return False
    return phi * (S - details.critical_spot) >= 0.0


def greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    option_type: str = "call",
    scheme: str | Scheme = "accurate",
) -> AmericanGreeks:
    """First-order Greeks of an American option, by finite differences of :func:`price` where they are needed.

    Parameters are those of :func:`price_details`, as scalars, with ``S``, ``K``, ``T`` and ``sigma`` positive.

    * An option valued as European (``method == "european"``, early exercise never optimal) has the closed-form
      Black-Scholes-Merton Greeks. At the edge of that regime (a rate and dividend yield of zero, for instance,
      where a 1 bp move one way makes early exercise pay) they are the one-sided derivatives within it.
    * An option exercised at once (at or beyond its critical spot) is worth its intrinsic value nearby, so its
      delta is -1 (put) or +1 (call) and the other Greeks are zero, without the rounding of differences of a value
      as large as the strike.
    * Otherwise each Greek is a central difference with the bumps of :data:`GREEK_BUMPS`: ``S (1 +- 1e-4)``,
      ``T (1 +- 1e-4)``, ``sigma +- 1e-4`` (at most half the volatility), ``r +- 1e-4`` and ``q +- 1e-4``. Every
      bumped value is priced with the scheme and the form of the boundary equation that solved the option at its
      own inputs, held fixed (a :class:`Scheme` of one's own is never handed over), so that no difference
      straddles a hand-over or a change of form, where the value jumps (see :func:`price`). Where one bump
      changes how the option is valued (its ``method``, or whether it is exercised at once), the derivative is
      taken on the other side alone, from values at a quarter, a half and the whole bump on it, extrapolated
      (if both bumps change it, the central difference stands): for example the dividend rho of a call at a
      negative rate without dividends, whose lower bump lies in the double-boundary regime. Where the option
      falls back to the tree (:func:`price_details`), the differences are those of tree values, which are far
      less smooth.

    Returns
    -------
    AmericanGreeks
        ``delta``, ``gamma``, ``theta``, ``vega``, ``rho`` and ``epsilon`` (dividend rho) in the raw units of
        :func:`black_scholes.greeks`: per unit of spot, per year of calendar time (``theta = -dV/dT``) and per
        unit (1.00 = 100 %) of volatility, rate and dividend yield; :data:`black_scholes.GREEKS` gives the display
        scales. ``basis`` and ``one_sided`` say how they were found.

    Examples
    --------
    A call without dividends is never exercised early, so its Greeks are Black-Scholes-Merton's:

    >>> g = greeks(100, 100, 1.0, 0.05, 0.2, 0.0, option_type="call")
    >>> round(g["delta"], 6), round(g["vega"], 5), g.basis
    (0.636831, 37.52403, 'closed form')
    >>> g = greeks(100, 100, 1.0, 0.05, 0.2, 0.0, option_type="put")
    >>> round(g["delta"], 6), round(g["gamma"], 6), round(g["theta"] / 365, 5), g.basis
    (-0.411059, 0.022989, -0.00613, 'differences')
    >>> greeks(100, 100, 1.0, -0.01, 0.2, 0.0, option_type="call").one_sided
    ('epsilon',)
    """
    S, K, T, r, sigma, q = (float(x) for x in (S, K, T, r, sigma, q))
    if min(S, K, T, sigma) <= 0.0:
        raise ValueError("finite-difference Greeks need a positive spot, strike, time to expiry and volatility")
    phi = bs.option_sign(option_type)
    chosen = _scheme(scheme)
    details = price_details(S, K, T, r, sigma, q, option_type=option_type, scheme=chosen)
    if details.method == "european":
        exact = bs.greeks(S, K, T, r, sigma, q, option_type)
        return AmericanGreeks(*(float(exact[name]) for name in GREEK_NAMES), basis="closed form")
    if _exercised(S, details, phi):
        return AmericanGreeks(float(phi), 0.0, 0.0, 0.0, 0.0, 0.0, basis="exercise value")
    held, form = chosen, None
    if details.method == "alo":  # a boundary was solved: hold its scheme (after any hand-over) and its form
        named = chosen is SCHEMES.get(chosen.name)
        held, form = dataclasses.replace(SCHEMES[details.scheme] if named else chosen), details.fixed_point
    inputs = {"S": S, "T": T, "r": r, "sigma": sigma, "q": q}
    v0 = details.value

    def value(name: str, x: float) -> tuple[float, bool]:
        """The value with input ``name`` at ``x``, and whether it is valued as at the unbumped inputs."""
        c = inputs | {name: x}
        d = price_details(
            c["S"], K, c["T"], c["r"], c["sigma"], c["q"], option_type=option_type, scheme=held, fixed_point=form
        )
        return d.value, d.method == details.method and not _exercised(c["S"], d, phi)

    one_sided = []

    def slopes(name: str, h: float) -> tuple[float, float]:
        """First and second derivatives in input ``name`` from bumps of ``+-h``."""
        x = inputs[name]
        (up, up_same), (down, down_same) = value(name, x + h), value(name, x - h)
        if up_same == down_same:  # both on the side of the inputs (or, on a sliver narrower than 2 h, neither)
            return (up - down) / (2.0 * h), (up - 2.0 * v0 + down) / h**2
        # One side alone: values at a quarter, a half and the whole bump on it. Near the edge of a regime the value
        # bends sharply, so the one-sided differences over h / 2 and h / 4 (second order for the slope, first for
        # the curvature) are extrapolated (Richardson) to third and second order.
        side = 1.0 if up_same else -1.0
        far = up if up_same else down
        half, quarter = value(name, x + 0.5 * side * h)[0], value(name, x + 0.25 * side * h)[0]
        one_sided.append(name)
        first = (side * (4.0 * half - 3.0 * v0 - far) / h, side * (4.0 * quarter - 3.0 * v0 - half) / (0.5 * h))
        second = ((v0 - 2.0 * half + far) / (0.5 * h) ** 2, (v0 - 2.0 * quarter + half) / (0.25 * h) ** 2)
        return (4.0 * first[1] - first[0]) / 3.0, 2.0 * second[1] - second[0]

    delta, gamma = slopes("S", GREEK_BUMPS["spot"] * S)
    theta = 0.0 - slopes("T", GREEK_BUMPS["expiry"] * T)[0]  # theta = -dV/dT, and +0.0 where dV/dT = 0
    vega = slopes("sigma", min(GREEK_BUMPS["volatility"], 0.5 * sigma))[0]
    rho = slopes("r", GREEK_BUMPS["rate"])[0]
    epsilon = slopes("q", GREEK_BUMPS["rate"])[0]
    names = {"S": ("delta", "gamma"), "T": ("theta",), "sigma": ("vega",), "r": ("rho",), "q": ("epsilon",)}
    sided = tuple(greek for name in one_sided for greek in names[name])
    return AmericanGreeks(delta, gamma, theta, vega, rho, epsilon, one_sided=sided)
