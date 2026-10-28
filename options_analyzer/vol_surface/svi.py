"""Gatheral's raw SVI smile at one maturity, and its least-squares calibration to implied vols.

:class:`SVISlice` is the smile ``w(y) = a + b (rho (y - m) + sqrt((y - m)^2 + s^2))`` in total implied variance at
forward log-moneyness ``y``, with analytic ``dw/dy`` and ``d2w/dy2``; :func:`fit_svi` fits it to one maturity's
implied vols, from Zeliade's quasi-explicit starting points (:func:`_svi_start_points`), each quote weighted by how
precisely it is known (roadmap item SF-4), and by default kept free of butterfly arbitrage (Durrleman's
:func:`durrleman_g` non-negative) and of calendar arbitrage against the previous, shorter maturity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, minimize

QUOTE_PRECISION = 1e-4  # implied-vol precision of a quote: differences below it are noise
#: An arbitrage-free fit holds Durrleman's g at least this high on its constraint grid, so that the surface's own
#: check (g <= 0 is butterfly arbitrage) never sees a rounding-level violation.
G_MARGIN = 1e-3
#: The constraint grids start sparse: this many points over the quoted range widened by CONSTRAINT_REACH of its width
#: on each side (and on the calendar overlap); the dense checks add the points where a solution still fails.
CONSTRAINT_POINTS = 40
CONSTRAINT_REACH = 0.10
#: Extrapolated vols are read beyond the quotes out to |k| = |ln K/F| of WING_SDS at-the-money standard deviations
#: sqrt(w(0)), at most WING_CAP (roadmap item SF-8; :func:`wing_reach`), and the butterfly constraint holds out to
#: there too, on WING_POINTS points in each wing. A cap of 1.5 alone is a hundred standard deviations for a six-hour
#: expiry: holding the density positive that far bent the short smiles 0.3 vol point off at the money. Four
#: standard deviations lie well beyond a 1-delta strike (about 2.3); on 200 random noisy SSVI surfaces the worst
#: expiry's RMSE then grows by 6 % under the constraints, against 12 % at five and 14 % at six.
WING_CAP = 1.5
WING_SDS = 4.0
WING_POINTS = 15
#: A fit meets the constraints only if it does on a dense grid as well (DENSE_POINTS over the constrained range, and
#: points a fraction of s apart around the smile's vertex m, where g can turn within a width of s): points where it
#: fails are added to the constraint grid and the fit solved again (at most CUTTING_ROUNDS times).
DENSE_POINTS = 2001
CUTTING_ROUNDS = 4
#: The calendar constraint holds where the two slices' quoted ranges overlap, widened by this fraction of their
#: widths. Zero, as the roadmap's prototype found: over 200 random arbitrage-free SSVI surfaces with 0.2 vol point of
#: noise, holding it 25 % beyond the quotes (where the surface's calendar check also compares slices for local vol)
#: forced each noisy slice's extrapolated wings onto the next one and left 36 of them with fits 1.1 to 300 times
#: worse than unconstrained; on the quoted overlap alone no fit got worse by more than 10 %.
CALENDAR_REACH = 0.0


#: Weights of the quotes on the fit (:func:`spread_weights`): one over the half-spread in vol, at most this many times
#: the median weight, and with half-spreads below a vol floor counted at the floor.
MAX_WEIGHT = 5.0
VOL_SPREAD_FLOOR = QUOTE_PRECISION


def spread_weights(half_spread) -> np.ndarray | None:
    """Weights of quotes on an SVI fit from their half bid/ask spreads in implied vol (roadmap item SF-4).

    ``1 / max(half_spread, VOL_SPREAD_FLOOR)``, divided by the median and capped at :data:`MAX_WEIGHT`: a quote known
    to a tenth of a vol point counts more than a wing quote whose spread spans five, without letting the tightest
    few decide the whole smile. A quote whose spread is unknown (NaN) gets the median weight; with no spread known
    at all the result is None (equal weights), and equal spreads give weights of exactly 1.

    Examples
    --------
    >>> spread_weights([0.001, 0.002, 0.004]).tolist()
    [2.0, 1.0, 0.5]
    >>> spread_weights([0.003, 0.003]).tolist(), spread_weights([np.nan, np.nan])
    ([1.0, 1.0], None)
    """
    half_spread = np.asarray(half_spread, dtype=float)
    known = np.isfinite(half_spread)
    if not known.any():
        return None
    weights = 1.0 / np.maximum(np.where(known, half_spread, np.nan), VOL_SPREAD_FLOOR)
    median = np.nanmedian(weights)
    return np.minimum(np.where(known, weights / median, 1.0), MAX_WEIGHT)


def wing_reach(fit: SVISlice) -> float:
    """How far beyond the quotes a smile is read, in ``|ln K/F|``: :data:`WING_SDS` at-the-money standard deviations
    ``sqrt(w(0))``, at most :data:`WING_CAP`."""
    return float(min(WING_CAP, WING_SDS * np.sqrt(max(float(fit.total_variance(0.0)[0]), 0.0))))


def _vols(fit: SVISlice, y: np.ndarray, T: float) -> np.ndarray:
    return np.sqrt(np.maximum(fit.total_variance(y)[0], 0.0) / T)


def durrleman_g(y, w, w_y, w_yy):
    """Durrleman's butterfly-arbitrage function (the local-variance denominator)."""
    return 1.0 - (y / w) * w_y + 0.25 * (-0.25 - 1.0 / w + y**2 / w**2) * w_y**2 + 0.5 * w_yy


@dataclass(frozen=True)
class SVISlice:
    """Gatheral's raw SVI smile ``w(y) = a + b (rho (y - m) + sqrt((y - m)^2 + s^2))`` at one maturity."""

    a: float
    b: float
    rho: float
    m: float
    s: float

    def total_variance(self, y):
        """``(w, dw/dy, d2w/dy2)`` at forward log-moneyness ``y`` (all analytic)."""
        z = np.asarray(y, dtype=float) - self.m
        root = np.sqrt(z * z + self.s * self.s)
        w = self.a + self.b * (self.rho * z + root)
        w_y = self.b * (self.rho + z / root)
        w_yy = self.b * self.s * self.s / root**3
        return w, w_y, w_yy

    def parameter_derivatives(self, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The derivatives of ``(w, dw/dy, d2w/dy2)`` at ``y`` with respect to ``(a, b, rho, m, s)``: three arrays
        of shape ``(len(y), 5)``, for the gradients of the arbitrage constraints (:func:`fit_svi`)."""
        z = np.asarray(y, dtype=float) - self.m
        s2 = self.s * self.s
        root = np.sqrt(z * z + s2)
        r3, r5 = root**3, root**5
        zeros, ones = np.zeros_like(z), np.ones_like(z)
        dw = np.column_stack(
            [ones, self.rho * z + root, self.b * z, -self.b * (self.rho + z / root), self.b * self.s / root]
        )
        dw_y = np.column_stack(
            [zeros, self.rho + z / root, self.b * ones, -self.b * s2 / r3, -self.b * z * self.s / r3]
        )
        dw_yy = np.column_stack(
            [zeros, s2 / r3, zeros, 3.0 * self.b * s2 * z / r5, self.b * (2.0 * self.s / r3 - 3.0 * self.s**3 / r5)]
        )
        return dw, dw_y, dw_yy

    @property
    def min_variance(self) -> float:
        return self.a + self.b * self.s * np.sqrt(1.0 - self.rho * self.rho)


_RHO_FACE = 0.99  # |rho| of the starting points that sit on the edge of the admissible region


def _svi_start_points(y, iv, T, n_best: int = 3, weights=None) -> list[np.ndarray]:
    """Quasi-explicit starting points (Zeliade): for fixed ``(m, s)`` SVI is linear in ``(a, c = b s, d = rho c)``.

    On a grid of ``(m, s)`` the weighted linear least-squares problem is solved subject to ``|d| < c``: the
    unconstrained solution where it satisfies that, otherwise the better of the two edges ``d = +-0.99 c``.
    Clipping an unconstrained solution instead (``c`` to about 0, or ``rho`` alone to 0.99) kept almost none of
    a steep, nearly linear skew's slope, and every start was a flat smile the optimiser could not leave.
    """
    w = iv * iv * T
    weight = 1.0 / (2.0 * iv * T)  # d(iv) = dw / (2 iv T): weighted variance residuals approximate vol residuals
    if weights is not None:  # and each quote's own weight on its vol residual
        weight = weight * weights
    target = w * weight
    lo, hi = y.min(), y.max()
    width = max(hi - lo, 1e-4)
    m_grid = np.linspace(lo - 0.5 * width, hi + 0.5 * width, 25)
    s_grid = width * np.geomspace(0.003, 2.0, 24)
    m_all, s_all = (g.ravel() for g in np.meshgrid(m_grid, s_grid, indexing="ij"))
    z = (y[None, :] - m_all[:, None]) / s_all[:, None]
    root = np.sqrt(z * z + 1.0)
    ones = np.ones_like(z)

    def solve(columns):  # weighted least squares at every grid point: coefficients and cost
        design = np.stack(columns, axis=-1) * weight[None, :, None]
        normal = np.einsum("gni,gnj->gij", design, design)
        # A ridge relative to each grid point's own matrix. Far from the quotes z and sqrt(z^2 + 1) are almost
        # collinear, and next to one heavily weighted quote (an implied vol of 0.1%) an absolute ridge vanished:
        # one exactly singular grid point aborted the whole batched solve. Such a point now gets some
        # coefficients, and its cost keeps it from being picked as a start.
        scale = np.trace(normal, axis1=1, axis2=2)[:, None, None] / len(columns)
        normal = normal + 1e-12 * scale * np.eye(len(columns))
        coef = np.linalg.solve(normal, np.einsum("gni,n->gi", design, target)[..., None])[..., 0]
        return coef, np.sum((np.einsum("gni,gi->gn", design, coef) - target) ** 2, axis=1)

    coef, cost = solve([ones, z, root])
    a, d, c = coef.T
    inside = (c > 0) & (np.abs(d) < _RHO_FACE * c)
    a, c = np.where(inside, a, np.mean(w)), np.where(inside, c, 0.0)  # flat smile until an edge does better
    rho = np.where(inside, d / np.where(inside, c, 1.0), 0.0)
    cost = np.where(inside, cost, np.sum((np.mean(w) * weight - target) ** 2))
    for sign in (1.0, -1.0):
        coef, cost_edge = solve([ones, sign * _RHO_FACE * z + root])
        a_edge, c_edge = coef.T
        better = (c_edge > 0) & (cost_edge < cost)
        a, c, cost = np.where(better, a_edge, a), np.where(better, c_edge, c), np.where(better, cost_edge, cost)
        rho = np.where(better, sign * _RHO_FACE, rho)
    b = np.minimum(np.maximum(c, 1e-12) / s_all, 2.0 / (1.0 + np.abs(rho)))  # Lee's wing bound
    a = np.maximum(a, -b * s_all * np.sqrt(1.0 - rho * rho))  # keep the minimum variance non-negative
    params = np.column_stack([a, b, rho, m_all, s_all])
    fitted = a[:, None] + b[:, None] * s_all[:, None] * (rho[:, None] * z + root)
    cost = np.sum(((fitted - w[None, :]) * weight[None, :]) ** 2, axis=1)
    return [params[i] for i in np.argsort(cost)[:n_best]]


_RHO_MAX = 0.999
_LOG_S_RANGE = (-12.0, 6.0)  # s from e^-12 to e^6 times the quoted range of y: a kink to a parabola


def fit_svi(
    y,
    iv,
    T: float,
    weights=None,
    previous: tuple[SVISlice, tuple[float, float]] | None = None,
    *,
    arbitrage_free: bool = True,
) -> SVISlice:
    """Least-squares raw-SVI fit to implied vols ``iv`` at forward log-moneyness ``y`` and maturity ``T``.

    Returns the fit; :func:`fit_svi_pair` also returns the fit without the arbitrage constraints, to show what they
    cost.

    The residuals are implied-vol errors, each multiplied by its quote's ``weights`` (default 1; see
    :func:`spread_weights`). ``b >= 0``, ``|rho| < 1``, an ``s`` from
    a kink to a parabola and a vertex ``m`` within a few quoted ranges of the
    quotes hold by construction (``b = |beta|``, ``rho = 0.999 tanh(r)``,
    ``ln s`` and ``m`` mapped through ``tanh`` onto their ranges), with no
    parameter whose gradient is cut to zero; a negative minimum variance and a
    breach of Lee's wing bound ``b (1 + |rho|) <= 2`` are penalised. Several
    quasi-explicit starting points (:func:`_svi_start_points`) are refined and
    the best fit is kept. Noise-free SVI (hence SSVI) quotes are recovered to
    solver precision.

    With ``arbitrage_free`` (the default) the fit is then checked for static arbitrage, on
    :data:`DENSE_POINTS` points: butterfly arbitrage where Durrleman's ``g`` (:func:`durrleman_g`) is not positive,
    over the quoted range widened by :data:`CONSTRAINT_REACH` of its width on each side and out to the smile's
    :func:`wing_reach` (where extrapolated vols are read); and calendar arbitrage where the total variance falls
    below that of ``previous``, the fit of the previous (shorter) maturity with its quoted range of ``y``, by more
    than the quotes' precision, on the overlap of the two quoted ranges only (enforcing it beyond the quotes
    over-constrains the smile). Where either fails, the nearest smile that holds ``g`` at :data:`G_MARGIN` or above
    and ``w`` at or above the previous slice's is found by sequential quadratic programming with analytic gradients,
    from that optimum and, if it ends 10 % of RMSE worse, from the quasi-explicit starts; so on arbitrage-free quotes
    the result is the unconstrained fit itself.
    """
    return fit_svi_pair(y, iv, T, weights, previous, arbitrage_free=arbitrage_free)[0]


def fit_svi_pair(
    y,
    iv,
    T: float,
    weights=None,
    previous: tuple[SVISlice, tuple[float, float]] | None = None,
    *,
    arbitrage_free: bool = True,
) -> tuple[SVISlice, SVISlice]:
    """:func:`fit_svi` and the fit without the arbitrage constraints: ``(fit, unconstrained)``, the same object
    twice where the unconstrained fit has no arbitrage or ``arbitrage_free`` is False."""
    y = np.asarray(y, dtype=float)
    iv = np.asarray(iv, dtype=float)
    weight = None if weights is None else np.asarray(weights, dtype=float)
    lo, hi = y.min(), y.max()
    width = max(hi - lo, 1e-4)
    centre, span = 0.5 * (lo + hi), 3.5 * width
    log_s_mid = np.log(width) + 0.5 * (_LOG_S_RANGE[0] + _LOG_S_RANGE[1])
    log_s_half = 0.5 * (_LOG_S_RANGE[1] - _LOG_S_RANGE[0])
    penalty = 10.0 * np.sqrt(len(y))
    vega = 1.0 / (2.0 * np.sqrt(np.mean(iv * iv * T) * T))  # variance -> vol units for the penalties

    def unpack(x):
        a, beta, r, mu, sigma = x
        m = centre + span * np.tanh(mu)
        return a, abs(beta), _RHO_MAX * np.tanh(r), m, np.exp(log_s_mid + log_s_half * np.tanh(sigma))

    def residuals(x):
        a, b, rho, m, s = unpack(x)
        z = y - m
        root = np.sqrt(z * z + s * s)
        w = np.maximum(a + b * (rho * z + root), 1e-300)
        wmin = a + b * s * np.sqrt(1.0 - rho * rho)
        lee = b * (1.0 + abs(rho)) - 2.0
        errors = np.sqrt(w / T) - iv
        if weight is not None:
            errors = errors * weight
        return np.concatenate([errors, [penalty * vega * max(-wmin, 0.0), penalty * max(lee, 0.0)]])

    def jacobian(x):
        _, beta, r, mu, sigma = x
        a, b, rho, m, s = unpack(x)
        z = y - m
        root = np.sqrt(z * z + s * s)
        w = a + b * (rho * z + root)
        scale = np.where(w > 0, 0.5 / np.sqrt(np.maximum(w, 1e-300) * T), 0.0)
        if weight is not None:
            scale = scale * weight
        dw = np.column_stack([np.ones_like(z), rho * z + root, b * z, -b * (rho + z / root), b * s / root])
        jac = np.vstack([dw * scale[:, None], np.zeros((2, 5))])
        q = np.sqrt(1.0 - rho * rho)
        if a + b * s * q < 0:
            jac[-2] = -penalty * vega * np.array([1.0, s * q, -b * s * rho / q, 0.0, b * q])
        if b * (1.0 + abs(rho)) > 2.0:
            jac[-1] = penalty * np.array([0.0, 1.0 + abs(rho), b * np.sign(rho), 0.0, 0.0])
        chain = [
            1.0,
            1.0 if beta >= 0 else -1.0,
            _RHO_MAX * (1.0 - np.tanh(r) ** 2),
            span * (1.0 - np.tanh(mu) ** 2),
            s * log_s_half * (1.0 - np.tanh(sigma) ** 2),
        ]
        return jac * np.array(chain)[None, :]

    def solve(x0, budget):
        return least_squares(
            residuals, x0, jac=jacobian, method="lm", x_scale="jac", ftol=1e-10, xtol=1e-12, gtol=1e-12, max_nfev=budget
        )

    def inside(value):
        return np.arctanh(np.clip(value, -0.999, 0.999))

    starts = [
        np.array(
            [
                a,
                b,
                inside(rho / _RHO_MAX),
                inside((m - centre) / span),
                inside((np.log(s) - log_s_mid) / log_s_half),
            ]
        )
        for a, b, rho, m, s in _svi_start_points(y, iv, T, weights=weight)
    ]
    # Race the starting points on a small budget, then finish the best one in chunks. Noisy quotes leave SVI
    # with a flat valley along which the fit hardly improves; stop once a chunk gains less than a thousandth
    # of the quote precision in implied-vol RMSE.
    best = min((solve(x0, 20) for x0 in starts), key=lambda r: r.cost)
    n_obs = len(y)
    for _ in range(8):
        if best.status != 0:
            break
        step = solve(best.x, 50)
        gain = np.sqrt(2.0 * best.cost / n_obs) - np.sqrt(2.0 * min(step.cost, best.cost) / n_obs)
        best = step if step.cost <= best.cost else best
        if gain < 1e-3 * QUOTE_PRECISION:
            break
    unconstrained = SVISlice(*map(float, unpack(best.x)))
    if not arbitrage_free:
        return unconstrained, unconstrained
    constraints = _Constraints.build(lo, hi, T, previous, wing_reach(unconstrained))
    if constraints.met(unconstrained):
        return unconstrained, unconstrained
    # The nearest arbitrage-free smile: sequential quadratic programming on the raw parameters with the constraints
    # as inequalities, from the unconstrained optimum and from the quasi-explicit starts, keeping the best that meets
    # them. A penalty method (constraint rows added to the least squares) was tried first: on 200 random
    # arbitrage-free SSVI surfaces with 0.2 vol point of noise it left some fits 6 times worse than unconstrained;
    # this keeps every maturity within 10 %. The unconstrained optimum alone, as the start, left one at 24 %.
    weights = np.ones_like(y) if weight is None else weight

    def cost(fit: SVISlice) -> float:
        return float(np.sum((weights * (_vols(fit, y, T) - iv)) ** 2))

    nearest = constraints.nearest(y, iv, T, weight, unconstrained)
    # From the unconstrained optimum the search usually ends next to it; only a fit that lost more than 10 % of
    # RMSE (21 % of squared error) is worth the other starts.
    if nearest is not None and cost(nearest) <= 1.21 * cost(unconstrained):
        return nearest, unconstrained
    starts = (SVISlice(*map(float, x)) for x in _svi_start_points(y, iv, T, weights=weight))
    feasible = [fit for fit in (nearest, *(constraints.nearest(y, iv, T, weight, start) for start in starts)) if fit]
    if not feasible:
        return unconstrained, unconstrained  # no start reached an arbitrage-free smile: the surface's checks report it
    return min(feasible, key=cost), unconstrained


@dataclass(frozen=True, eq=False)  # holds arrays: compared by identity
class _Constraints:
    """The static-arbitrage constraints of one slice's fit (see :func:`fit_svi`): Durrleman's g on the sparse
    ``grid`` and densely over ``span``; the total variance at or above the previous slice's (``previous``) on the
    sparse ``overlap`` and densely over ``overlap_span``."""

    T: float
    span: tuple[float, float]  # where the density must stay positive: the quotes widened, and the wings
    grid: np.ndarray
    previous: SVISlice | None
    overlap_span: tuple[float, float] | None  # the overlap of the two slices' quoted ranges
    overlap: np.ndarray

    @classmethod
    def build(cls, lo: float, hi: float, T: float, previous, wing: float = 0.0) -> _Constraints:
        reach = CONSTRAINT_REACH * max(hi - lo, 1e-4)
        grid = np.linspace(lo - reach, hi + reach, CONSTRAINT_POINTS)
        if grid[0] > -wing:
            grid = np.concatenate([np.linspace(-wing, grid[0], WING_POINTS, endpoint=False), grid])
        if grid[-1] < wing:
            grid = np.concatenate([grid, np.linspace(wing, grid[-1], WING_POINTS, endpoint=False)[::-1]])
        fit, overlap_span, overlap = None, None, np.empty(0)
        if previous is not None:
            fit, (previous_lo, previous_hi) = previous
            reach_now, reach_before = (
                CALENDAR_REACH * max(b - a, 1e-4) for a, b in ((lo, hi), (previous_lo, previous_hi))
            )
            start, end = (
                max(lo - reach_now, previous_lo - reach_before),
                min(hi + reach_now, previous_hi + reach_before),
            )
            if start < end:
                overlap_span = (float(start), float(end))
                overlap = np.linspace(start, end, CONSTRAINT_POINTS)
        return cls(float(T), (float(grid[0]), float(grid[-1])), grid, fit, overlap_span, overlap)

    def _w_previous(self, y: np.ndarray) -> np.ndarray:
        return self.previous.total_variance(y)[0] if self.previous is not None and len(y) else np.zeros(len(y))

    def nearest(
        self, y: np.ndarray, iv: np.ndarray, T: float, weight: np.ndarray | None, start: SVISlice
    ) -> SVISlice | None:
        """The SVI smile nearest the quotes (weighted vol least squares) that meets the constraints, by SLSQP from
        ``start``, or None where the solver ends without meeting them.

        Inequalities: ``g >= G_MARGIN`` on the grid, ``w >= w_previous`` on the overlap, a non-negative minimum
        variance, and Lee's bound ``b (1 + |rho|) <= 2`` as its two smooth halves ``b (1 +- rho) <= 2``. A solution
        that meets them on the sparse grids but not on the dense checks (:meth:`_failing`, :meth:`_falling`) has the
        points where it fails added to the grids and is solved again from there (cutting planes): with all 520
        points from the start, the solver's quadratic subproblems took 15 of the 16 seconds of the slowest of 200
        noisy surfaces.
        """
        weights = np.ones_like(y) if weight is None else weight
        lo, hi = y.min(), y.max()
        width = max(hi - lo, 1e-4)
        centre = 0.5 * (lo + hi)
        variance_scale = float(np.mean(iv * iv * T))
        # The solver works in parameters of order one: a in units of the quotes' total variance, b in that per unit
        # of the quoted width, m and s in units of the width. In raw units a six-hour expiry's variances (1e-5) left
        # the finite differences and the line search so badly scaled that one fit collapsed to w = 0.
        scale = np.array([variance_scale, variance_scale / width, 1.0, width, width])
        offset = np.array([0.0, 0.0, 0.0, centre, 0.0])

        def raw(x):
            return x * scale + offset

        def cost(x):
            w = np.maximum(SVISlice(*raw(x)).total_variance(y)[0], 1e-300)
            return float(np.mean((weights * (np.sqrt(w / T) - iv)) ** 2)) * 1e4  # in squared vol points

        def cost_gradient(x):
            fit = SVISlice(*raw(x))
            w = np.maximum(fit.total_variance(y)[0], 1e-300)
            vol = np.sqrt(w / T)
            # d/dp mean((wt (vol - iv))^2) = mean(2 wt^2 (vol - iv) dvol/dw dw/dp), dvol/dw = 1 / (2 sqrt(w T)).
            factor = 2.0 * weights**2 * (vol - iv) / (2.0 * np.sqrt(w * T))
            return (factor @ fit.parameter_derivatives(y)[0]) / len(y) * 1e4 * scale

        def inequalities(x, grid, overlap):
            p = raw(x)
            fit = SVISlice(*p)
            w, w_y, w_yy = fit.total_variance(grid)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                g = durrleman_g(grid, np.maximum(w, 1e-300), w_y, w_yy)
            # Bounded, so that the solver stays finite where w is near zero on the way.
            g = np.clip(np.nan_to_num(g, nan=-1.0), -1e3, 1e3)
            calendar = (fit.total_variance(overlap)[0] - self._w_previous(overlap)) / variance_scale
            a, b, rho, _, s = p
            minimum = (a + b * s * np.sqrt(max(1.0 - rho * rho, 0.0))) / variance_scale
            return np.concatenate([g - G_MARGIN, calendar, [minimum, 2.0 - b * (1.0 + rho), 2.0 - b * (1.0 - rho)]])

        def inequality_jacobian(x, grid, overlap):
            p = raw(x)
            fit = SVISlice(*p)
            w, w_y, _ = fit.total_variance(grid)
            w = np.maximum(w, 1e-300)
            dw, dw_y, dw_yy = fit.parameter_derivatives(grid)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                # g = 1 - (y/w) w' + (-1/4 - 1/w + y^2/w^2) w'^2 / 4 + w''/2, differentiated in w, w' and w''.
                g_w = grid * w_y / w**2 + 0.25 * (1.0 / w**2 - 2.0 * grid**2 / w**3) * w_y**2
                g_wy = -grid / w + 0.5 * (-0.25 - 1.0 / w + grid**2 / w**2) * w_y
                rows_g = g_w[:, None] * dw + g_wy[:, None] * dw_y + 0.5 * dw_yy
            rows_g = np.clip(np.nan_to_num(rows_g, nan=0.0), -1e8, 1e8)
            rows_calendar = fit.parameter_derivatives(overlap)[0] / variance_scale
            _, b, rho, _, s = p
            q = np.sqrt(max(1.0 - rho * rho, 1e-300))
            rows_bounds = np.array(
                [
                    [
                        1.0 / variance_scale,
                        s * q / variance_scale,
                        -b * s * rho / q / variance_scale,
                        0.0,
                        b * q / variance_scale,
                    ],
                    [0.0, -(1.0 + rho), -b, 0.0, 0.0],
                    [0.0, -(1.0 - rho), b, 0.0, 0.0],
                ]
            )
            return np.vstack([rows_g, rows_calendar, rows_bounds]) * scale

        bounds = [(None, None), (0.0, None), (-_RHO_MAX, _RHO_MAX), (lo - 3.5 * width, hi + 3.5 * width)]
        bounds.append((width * np.exp(_LOG_S_RANGE[0]), width * np.exp(_LOG_S_RANGE[1])))
        low = np.array([-np.inf if b[0] is None else b[0] for b in bounds])
        high = np.array([np.inf if b[1] is None else b[1] for b in bounds])
        x = (np.clip([start.a, start.b, start.rho, start.m, start.s], low, high) - offset) / scale
        scaled_bounds = list(zip((low - offset) / scale, (high - offset) / scale, strict=True))
        grid, overlap = self.grid, self.overlap
        for _ in range(CUTTING_ROUNDS):
            result = minimize(
                cost,
                x,
                jac=cost_gradient,
                method="SLSQP",
                bounds=scaled_bounds,
                constraints=[
                    {"type": "ineq", "fun": inequalities, "jac": inequality_jacobian, "args": (grid, overlap)}
                ],
                options={"maxiter": 300, "ftol": 1e-12},
            )
            x = result.x
            fit = SVISlice(*map(float, raw(x)))
            # A smile whose variance has collapsed far below the quotes' is a failed search, not a fit.
            if float(np.min(fit.total_variance(y)[0])) < 1e-3 * variance_scale:
                return None
            butterfly, calendar = self._failing(fit), self._falling(fit)
            if not (butterfly.size or calendar.size):
                return fit
            grid, overlap = np.union1d(grid, butterfly), np.union1d(overlap, calendar)
        return None

    def _dense(self, fit: SVISlice) -> np.ndarray:
        """Where the density is checked: :data:`DENSE_POINTS` over the span, and 161 points s / 10 apart around the
        vertex m (within the span)."""
        dense = np.linspace(*self.span, DENSE_POINTS)
        vertex = fit.m + fit.s * np.linspace(-8.0, 8.0, 161)
        return np.concatenate([dense, vertex[(vertex >= self.span[0]) & (vertex <= self.span[1])]])

    def _failing(self, fit: SVISlice) -> np.ndarray:
        """The points of the dense check where ``w <= 0`` or ``g <= 0``."""
        y = self._dense(fit)
        w, w_y, w_yy = fit.total_variance(y)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = durrleman_g(y, w, w_y, w_yy)
        return y[(w <= 0) | ~(g > 0)]

    def _falling(self, fit: SVISlice) -> np.ndarray:
        """The points of a dense check of the overlap (:data:`DENSE_POINTS`) where the total variance falls below the
        previous slice's by more than the quotes' precision: each slice's quotes are precise to
        :data:`QUOTE_PRECISION` in vol, ``2 sqrt(w T)`` of it in total variance (as the surface's own calendar check,
        ``local_vol._time_derivative``; this slice's ``T`` bounds both)."""
        if self.overlap_span is None:
            return np.empty(0)
        y = np.linspace(*self.overlap_span, DENSE_POINTS)
        w_previous = self._w_previous(y)
        tolerance = 2.0 * np.sqrt(np.maximum(w_previous, 0.0) * self.T) * QUOTE_PRECISION
        return y[fit.total_variance(y)[0] - w_previous < -tolerance]

    def met(self, fit: SVISlice) -> bool:
        """No butterfly arbitrage over the span (g > 0) and no fall below the previous slice beyond the precision."""
        return not any(self.broken(fit))

    def broken(self, fit: SVISlice) -> tuple[bool, bool]:
        """``(butterfly, calendar)``: whether ``fit`` has butterfly arbitrage on the dense check (g <= 0, or w <= 0)
        and whether its total variance falls below the previous slice's by more than the quotes' precision."""
        return bool(self._failing(fit).size), bool(self._falling(fit).size)


def static_arbitrage(
    fit: SVISlice, quoted: tuple[float, float], T: float, previous: tuple[SVISlice, tuple[float, float]] | None
) -> tuple[bool, bool]:
    """``(butterfly, calendar)``: whether ``fit`` breaks the constraints :func:`fit_svi` holds an arbitrage-free fit
    to: butterfly arbitrage over its ``quoted`` range of ``y`` widened by :data:`CONSTRAINT_REACH` and out to its
    :func:`wing_reach`, calendar arbitrage against ``previous`` on the overlap of the two quoted ranges."""
    return _Constraints.build(quoted[0], quoted[1], T, previous, wing_reach(fit)).broken(fit)
