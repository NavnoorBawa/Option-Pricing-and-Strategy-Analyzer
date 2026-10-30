"""Gatheral's raw SVI smile at one maturity, and its least-squares calibration to implied vols.

:class:`SVISlice` is the smile ``w(y) = a + b (rho (y - m) + sqrt((y - m)^2 + s^2))`` in total implied variance at
forward log-moneyness ``y``, with analytic ``dw/dy`` and ``d2w/dy2``; :func:`fit_svi` fits it to one maturity's
implied vols, from Zeliade's quasi-explicit starting points (:func:`_svi_start_points`), each quote weighted by how
precisely it is known (roadmap item SF-4), and by default kept free of butterfly arbitrage (Durrleman's
:func:`durrleman_g` non-negative) and of calendar arbitrage against the previous, shorter maturity.
"""

from __future__ import annotations

import warnings
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
#: expiry's RMSE then grows by 8.6 % under the constraints (7.4 % at five standard deviations, 9.5 % at six, all with
#: BLAS threads: the reach is a choice of how far SVI's straight wings are trusted, hardly a cost of the fit).
WING_CAP = 1.5
WING_SDS = 4.0
WING_POINTS = 15
#: A fit meets the constraints only if it does on a dense grid as well (DENSE_POINTS over the constrained range, and
#: points a fraction of s apart around the smile's vertex m, where g can turn within a width of s): the points where
#: it fails are added to the constraint grid and the fit solved again (at most CUTTING_ROUNDS times), all of them
#: where there are at most THIN_ABOVE, else CUT_POINTS evenly spaced ones and the worst of each stretch of them
#: (:func:`_thinned`).
DENSE_POINTS = 2001
CUTTING_ROUNDS = 4
THIN_ABOVE = 200
CUT_POINTS = 5
#: The solver's iterations for one slice, over all its starts and cutting-plane rounds (SLSQP_BUDGET), for a whole
#: surface (SURFACE_BUDGET, shared by its slices in order of maturity, so a slice gets at most what is left of it), and
#: for one solve (SLSQP_MAXITER, with SLSQP_FTOL its tolerance on the mean squared error in squared vol points); a start
#: that has not met the constraints when either budget runs out fails, and a slice whose every start fails keeps its
#: unconstrained fit with a note (see :func:`fit_svi`). The reviewer's slowest case, 20 expiries of 41 strikes with 8
#: vol points of noise over seeds 5-8, took 2.8-14 s with BLAS threads and 3.4-5.0 s of processor time with one thread
#: before the budgets, the thinned cutting planes and the evaluation of the smile once per solver step; it takes 1.8-2.9
#: s with threads on a lightly loaded shared four-core machine (up to 10 s under a load of 8) and 2.4-4.5 s with one
#: thread after them (0.5-0.8 s unconstrained). It spends 2,761-3,872 iterations in all (at most 600 on a slice), about
#: 0.8-1.1 ms each with one thread: SURFACE_BUDGET caps any surface at about 3-4.5 s of solver time with one thread,
#: whatever its noise; the slices it leaves unsolved are noted. No slice of the bundled sample quotes, of the synthetic
#: SPX chains or of the 200 random surfaces of tests/test_svi_constraints.py runs out of either budget (the most a slice
#: used is 419 iterations, a surface 940), so their fits are as they were.
SLSQP_BUDGET = 600
SURFACE_BUDGET = 4000
SLSQP_MAXITER = 300
SLSQP_FTOL = 1e-12
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
    *,
    weights=None,
    previous: tuple[SVISlice, tuple[float, float]] | None = None,
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
    the best fit is kept. A penalty is not a hard bound: fitted to 25 quotes of a
    raw SVI smile whose wing slope is beyond 2, the fit's slope ends above it, by
    up to 6e-5 for true slopes of 2.2 to 3 and 2e-4 to 0.0024 for slopes of 5.7 to
    10 (2.0024074 for ``b = 4``, ``rho = -0.99``). So the fit's ``b`` is then
    projected onto the bound (:func:`_within_lee`), with ``a``, ``rho``, ``m`` and
    ``s`` kept: every fit's wing slope is at most 2, with or without the arbitrage
    constraints. The projection is not re-optimised (the Levenberg-Marquardt
    solver takes no inequalities); it moved those fits' RMSE by 0.005 vol point at
    most, except on that steepest smile, from 93.395 to 93.457 vol points.
    Noise-free SVI (hence SSVI) quotes are recovered to solver precision.

    With ``arbitrage_free`` (the default) the fit is then checked for static arbitrage, on
    :data:`DENSE_POINTS` points: butterfly arbitrage where Durrleman's ``g`` (:func:`durrleman_g`) is not positive,
    over the quoted range widened by :data:`CONSTRAINT_REACH` of its width on each side and out to the smile's
    :func:`wing_reach` (where extrapolated vols are read); and calendar arbitrage where the total variance falls
    below that of ``previous``, the fit of the previous (shorter) maturity with its quoted range of ``y``, by more
    than the quotes' precision, on the overlap of the two quoted ranges only (enforcing it beyond the quotes
    over-constrains the smile). Where either fails, the nearest smile that holds ``g`` at :data:`G_MARGIN` or above
    and ``w`` at or above the previous slice's is found by sequential quadratic programming with analytic gradients,
    from that optimum and, if it ends 10 % of RMSE worse, from the quasi-explicit starts in turn until one ends within
    those 10 %, and if none does, from the previous slice shifted up and from a nearly flat smile above it (which
    meets the constraints); the best of them is kept. So on arbitrage-free quotes the result is the unconstrained fit
    itself. Where no start reaches an arbitrage-free smile (within :data:`CUTTING_ROUNDS` rounds each and
    :data:`SLSQP_BUDGET` iterations in all, or what is left of a surface's :data:`SURFACE_BUDGET`), the unconstrained
    fit is returned, arbitrage and all:
    :func:`static_arbitrage` tells, and :func:`.surface.fit_market_surface` says so in a note.
    """
    return fit_svi_pair(y, iv, T, weights=weights, previous=previous, arbitrage_free=arbitrage_free)[0]


def fit_svi_pair(
    y,
    iv,
    T: float,
    *,
    weights=None,
    previous: tuple[SVISlice, tuple[float, float]] | None = None,
    arbitrage_free: bool = True,
    budget: _Budget | None = None,
) -> tuple[SVISlice, SVISlice]:
    """:func:`fit_svi` and the fit without the arbitrage constraints: ``(fit, unconstrained)``, the same object
    twice where the unconstrained fit has no arbitrage, where ``arbitrage_free`` is False, or where no start reached
    an arbitrage-free smile (then :func:`static_arbitrage` of the fit is not ``(False, False)``).

    ``budget`` is the solver iterations left for a whole surface (:data:`SURFACE_BUDGET`, shared by its slices): the
    arbitrage-free search spends at most :data:`SLSQP_BUDGET` of them, or what is left where that is less, and takes
    what it spent off it. Without it the search has :data:`SLSQP_BUDGET` to itself."""
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
    unconstrained = _within_lee(SVISlice(*map(float, unpack(best.x))))
    if not arbitrage_free:
        return unconstrained, unconstrained
    constraints = _Constraints.build(lo, hi, T, previous, wing_reach(unconstrained))
    if constraints.met(unconstrained):
        return unconstrained, unconstrained
    allowed = SLSQP_BUDGET if budget is None else max(min(SLSQP_BUDGET, budget.left), 0)
    left = _Budget(allowed)
    fit = _arbitrage_free(constraints, y, iv, T, weight, unconstrained, previous, left)
    if budget is not None:
        budget.left -= allowed - left.left
    if fit is None:
        return unconstrained, unconstrained  # no start reached an arbitrage-free smile: the surface notes it
    return _within_lee(fit), unconstrained


def _arbitrage_free(
    constraints: _Constraints,
    y: np.ndarray,
    iv: np.ndarray,
    T: float,
    weight: np.ndarray | None,
    unconstrained: SVISlice,
    previous: tuple[SVISlice, tuple[float, float]] | None,
    budget: _Budget,
) -> SVISlice | None:
    """The search of :func:`fit_svi_pair` for the arbitrage-free smile nearest the quotes, from its starts in turn,
    spending at most ``budget``; None where no start reaches one."""
    # The nearest arbitrage-free smile: sequential quadratic programming on the raw parameters with the constraints
    # as inequalities, from the unconstrained optimum and from the quasi-explicit starts, keeping the best that meets
    # them. A penalty method (constraint rows added to the least squares) was tried first: on 200 random
    # arbitrage-free SSVI surfaces with 0.2 vol point of noise it left some fits 6 times worse than unconstrained;
    # this keeps every maturity within 10 %. The unconstrained optimum alone, as the start, left one at 24 %.
    weights = np.ones_like(y) if weight is None else weight

    def cost(fit: SVISlice) -> float:
        return float(np.sum((weights * (_vols(fit, y, T) - iv)) ** 2))

    nearest = constraints.nearest(y, iv, T, weight, unconstrained, budget)
    # From the unconstrained optimum the search usually ends next to it; only a fit that lost more than 10 % of
    # RMSE (21 % of squared error) is worth the other starts.
    if nearest is not None and cost(nearest) <= 1.21 * cost(unconstrained):
        return nearest
    feasible = [nearest] if nearest is not None else []
    for start in _svi_start_points(y, iv, T, weights=weight):
        fit = constraints.nearest(y, iv, T, weight, SVISlice(*map(float, start)), budget)
        feasible += [fit] if fit is not None else []
        if fit is not None and cost(fit) <= 1.21 * cost(unconstrained):  # as good as the first is asked to be
            break
    if not any(cost(fit) <= 1.21 * cost(unconstrained) for fit in feasible):
        # Two starts inside the calendar constraint, tried only where those above found nothing within the 10 %, so
        # that wherever they do the fit is as before. The previous slice raised to this one's quotes (and a little
        # more): where these quotes lie below the previous smile across the whole overlap, every other start ended
        # with the solver's "inequality constraints incompatible" and the slice kept its arbitrage. A nearly flat
        # smile above the previous one, which meets both constraints: on a 7-day slice with 8 vol points of noise
        # and no previous slice, every other start ended the same way.
        extra = [constraints.flat(y, iv, T)]
        if previous is not None:
            extra.insert(0, _raised(previous[0], y, iv, T))
        for start in extra:
            fit = constraints.nearest(y, iv, T, weight, start, budget)
            feasible += [fit] if fit is not None else []
    return min(feasible, key=cost) if feasible else None


def _smile(p, points: np.ndarray, *, derivatives: bool = False) -> tuple[np.ndarray, ...]:
    """``(w, dw/dy, d2w/dy2)`` of the raw SVI parameters ``p = (a, b, rho, m, s)`` at ``points``, and with
    ``derivatives`` the three arrays of :meth:`SVISlice.parameter_derivatives` after them: the same arithmetic,
    element by element, without building an :class:`SVISlice`, for the solver's many calls."""
    a, b, rho, m, s = p
    z = points - m
    s2 = s * s
    root = np.sqrt(z * z + s2)
    values: tuple[np.ndarray, ...] = (a + b * (rho * z + root), b * (rho + z / root), b * s * s / root**3)
    if not derivatives:
        return values
    r3, r5 = root**3, root**5
    dw, dw_y, dw_yy = np.empty((len(z), 5)), np.empty((len(z), 5)), np.empty((len(z), 5))
    dw[:, 0], dw[:, 1], dw[:, 2], dw[:, 3], dw[:, 4] = 1.0, rho * z + root, b * z, -b * (rho + z / root), b * s / root
    dw_y[:, 0], dw_y[:, 1], dw_y[:, 2] = 0.0, rho + z / root, b
    dw_y[:, 3], dw_y[:, 4] = -b * s2 / r3, -b * z * s / r3
    dw_yy[:, 0], dw_yy[:, 1], dw_yy[:, 2] = 0.0, s2 / r3, 0.0
    dw_yy[:, 3], dw_yy[:, 4] = 3.0 * b * s2 * z / r5, b * (2.0 * s / r3 - 3.0 * s**3 / r5)
    return (*values, dw, dw_y, dw_yy)


def _raised(previous: SVISlice, y: np.ndarray, iv: np.ndarray, T: float) -> SVISlice:
    """The previous slice's smile raised by the median gap between it and the quotes' total variance (where they lie
    above it) plus 1 % of their mean: a start of :func:`fit_svi_pair` that meets the calendar constraint."""
    gap = float(np.median(iv * iv * T - previous.total_variance(y)[0]))
    shift = max(gap, 0.0) + 0.01 * float(np.mean(iv * iv * T))
    return SVISlice(previous.a + shift, previous.b, previous.rho, previous.m, previous.s)


def _within_lee(fit: SVISlice) -> SVISlice:
    """``fit`` with ``b`` cut to Lee's bound ``2 / (1 + |rho|)`` where it exceeds it, and ``a`` raised as much as keeps
    the minimum variance from falling below zero; ``fit`` itself where it is within the bound. The unconstrained
    solver's bound is a penalty, which leaves slopes up to 2.0024 on quotes far steeper than Lee allows (see
    :func:`fit_svi`); SLSQP's is an inequality met to its tolerance, a slope above 2 by rounding at most."""
    bound = 2.0 / (1.0 + abs(fit.rho))
    if fit.b <= bound:
        return fit
    a = max(fit.a, -bound * fit.s * np.sqrt(1.0 - fit.rho * fit.rho)) if fit.min_variance >= 0 else fit.a
    return SVISlice(float(a), float(bound), fit.rho, fit.m, fit.s)


@dataclass(eq=False)
class _Budget:
    """The solver iterations left for one slice's arbitrage-free fit (:data:`SLSQP_BUDGET`), or for a whole surface's
    (:data:`SURFACE_BUDGET`, see :func:`fit_svi_pair`)."""

    left: int


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

    def flat(self, y: np.ndarray, iv: np.ndarray, T: float) -> SVISlice:
        """A start that meets the constraints: an almost flat smile at the quotes' mean total variance, or a little
        above the previous slice's highest on the overlap where that is higher."""
        level = float(np.mean(iv * iv * T))
        if self.overlap_span is not None:
            level = max(level, 1.01 * float(self._w_previous(np.linspace(*self.overlap_span, DENSE_POINTS)).max()))
        width = max(float(np.ptp(y)), 1e-4)
        b = 1e-3 * level / width
        return SVISlice(level - b * width, b, 0.0, float(np.mean(y)), width)

    def nearest(
        self, y: np.ndarray, iv: np.ndarray, T: float, weight: np.ndarray | None, start: SVISlice, budget: _Budget
    ) -> SVISlice | None:
        """The SVI smile nearest the quotes (weighted vol least squares) that meets the constraints, by SLSQP from
        ``start``, or None where the solver ends without meeting them.

        Inequalities: ``g >= G_MARGIN`` on the grid, ``w >= w_previous`` on the overlap, a non-negative minimum
        variance, and Lee's bound ``b (1 + |rho|) <= 2`` as its two smooth halves ``b (1 +- rho) <= 2``. A solution
        that meets them on the sparse grids but not on the dense checks (:meth:`_failing`, :meth:`_falling`) has a
        few of the points where it fails (:func:`_thinned`) added to the grids and is solved again from there
        (cutting planes): with all 520 points from the start, the solver's quadratic subproblems took 15 of the 16
        seconds of the slowest of 200 noisy surfaces. It spends at most what is left of ``budget`` (iterations).
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

        n_y = len(y)
        # SLSQP asks for the cost and the constraints at the same point, and then for both gradients at it: the
        # smile is evaluated once per point, at the quotes and the constraint points together (the same arithmetic
        # as SVISlice.total_variance and parameter_derivatives, element by element).
        last: dict[bool, tuple[bytes, np.ndarray, tuple]] = {}

        def smile(x, points, derivatives: bool):
            key = x.tobytes()
            if derivatives in last and last[derivatives][0] == key and last[derivatives][1] is points:
                return last[derivatives][2]
            values = _smile(raw(x), points, derivatives=derivatives)
            last[derivatives] = (key, points, values)
            return values

        def cost(x, points, n_grid, w_previous):
            w = np.maximum(smile(x, points, False)[0][:n_y], 1e-300)
            return float(np.mean((weights * (np.sqrt(w / T) - iv)) ** 2)) * 1e4  # in squared vol points

        def cost_gradient(x, points, n_grid, w_previous):
            values = smile(x, points, True)
            w, dw = np.maximum(values[0][:n_y], 1e-300), values[3][:n_y]
            vol = np.sqrt(w / T)
            # d/dp mean((wt (vol - iv))^2) = mean(2 wt^2 (vol - iv) dvol/dw dw/dp), dvol/dw = 1 / (2 sqrt(w T)).
            factor = 2.0 * weights**2 * (vol - iv) / (2.0 * np.sqrt(w * T))
            return (factor @ dw) / n_y * 1e4 * scale

        def inequalities(x, points, n_grid, w_previous):
            p = raw(x)
            values = smile(x, points, False)
            on_grid = slice(n_y, n_y + n_grid)
            grid = points[on_grid]
            w, w_y, w_yy = (v[on_grid] for v in values)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                g = durrleman_g(grid, np.maximum(w, 1e-300), w_y, w_yy)
            # Bounded, so that the solver stays finite where w is near zero on the way.
            g = np.clip(np.where(np.isnan(g), -1.0, g), -1e3, 1e3)
            calendar = (values[0][n_y + n_grid :] - w_previous) / variance_scale
            a, b, rho, _, s = p
            minimum = (a + b * s * np.sqrt(max(1.0 - rho * rho, 0.0))) / variance_scale
            return np.concatenate([g - G_MARGIN, calendar, [minimum, 2.0 - b * (1.0 + rho), 2.0 - b * (1.0 - rho)]])

        def inequality_jacobian(x, points, n_grid, w_previous):
            p = raw(x)
            values = smile(x, points, True)
            on_grid = slice(n_y, n_y + n_grid)
            grid = points[on_grid]
            w, w_y = np.maximum(values[0][on_grid], 1e-300), values[1][on_grid]
            dw, dw_y, dw_yy = (v[on_grid] for v in values[3:])
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                # g = 1 - (y/w) w' + (-1/4 - 1/w + y^2/w^2) w'^2 / 4 + w''/2, differentiated in w, w' and w''.
                g_w = grid * w_y / w**2 + 0.25 * (1.0 / w**2 - 2.0 * grid**2 / w**3) * w_y**2
                g_wy = -grid / w + 0.5 * (-0.25 - 1.0 / w + grid**2 / w**2) * w_y
                rows_g = g_w[:, None] * dw + g_wy[:, None] * dw_y + 0.5 * dw_yy
            rows_g = np.clip(np.where(np.isnan(rows_g), 0.0, rows_g), -1e8, 1e8)
            rows_calendar = values[3][n_y + n_grid :] / variance_scale
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
            if budget.left <= 0:
                return None
            # The quotes and the constraint points, and the previous slice's variance there (once per round).
            args = (np.concatenate([y, grid, overlap]), len(grid), self._w_previous(overlap))
            with warnings.catch_warnings():
                # SciPy before 1.16 (the Fortran SLSQP) warns when a step leaves the bounds and is clipped back,
                # which is the intended handling of a bound.
                warnings.filterwarnings("ignore", "Values in x were outside bounds", RuntimeWarning)
                result = minimize(
                    cost,
                    x,
                    args=args,
                    jac=cost_gradient,
                    method="SLSQP",
                    bounds=scaled_bounds,
                    constraints=[{"type": "ineq", "fun": inequalities, "jac": inequality_jacobian, "args": args}],
                    options={"maxiter": min(SLSQP_MAXITER, budget.left), "ftol": SLSQP_FTOL},
                )
            budget.left -= max(int(result.nit), 1)
            x = result.x
            fit = SVISlice(*map(float, raw(x)))
            # A smile whose variance has collapsed far below the quotes' is a failed search, not a fit.
            if float(np.min(fit.total_variance(y)[0])) < 1e-3 * variance_scale:
                return None
            butterfly, calendar = self._failing(fit, thin=True), self._falling(fit, thin=True)
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

    def _failing(self, fit: SVISlice, *, thin: bool = False) -> np.ndarray:
        """The points of the dense check where ``w <= 0`` or ``g <= 0``; with ``thin``, those the cutting planes add
        (:func:`_thinned`)."""
        y = np.sort(self._dense(fit))
        w, w_y, w_yy = fit.total_variance(y)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = durrleman_g(y, w, w_y, w_yy)
        bad = (w <= 0) | ~(g > 0)
        return _thinned(y, bad, np.where(w > 0, np.nan_to_num(g, nan=-np.inf), -np.inf)) if thin else y[bad]

    def _falling(self, fit: SVISlice, *, thin: bool = False) -> np.ndarray:
        """The points of a dense check of the overlap (:data:`DENSE_POINTS`) where the total variance falls below the
        previous slice's by more than the quotes' precision: each slice's quotes are precise to
        :data:`QUOTE_PRECISION` in vol, ``2 sqrt(w T)`` of it in total variance (as the surface's own calendar check,
        ``local_vol._time_derivative``; this slice's ``T`` bounds both). With ``thin``, those the cutting planes
        add (:func:`_thinned`)."""
        if self.overlap_span is None:
            return np.empty(0)
        y = np.linspace(*self.overlap_span, DENSE_POINTS)
        w_previous = self._w_previous(y)
        tolerance = 2.0 * np.sqrt(np.maximum(w_previous, 0.0) * self.T) * QUOTE_PRECISION
        margin = fit.total_variance(y)[0] - w_previous + tolerance
        return _thinned(y, margin < 0, margin) if thin else y[margin < 0]

    def met(self, fit: SVISlice) -> bool:
        """No butterfly arbitrage over the span (g > 0) and no fall below the previous slice beyond the precision."""
        return not any(self.broken(fit))

    def broken(self, fit: SVISlice) -> tuple[bool, bool]:
        """``(butterfly, calendar)``: whether ``fit`` has butterfly arbitrage on the dense check (g <= 0, or w <= 0)
        and whether its total variance falls below the previous slice's by more than the quotes' precision."""
        return bool(self._failing(fit).size), bool(self._falling(fit).size)


def _thinned(y: np.ndarray, bad: np.ndarray, score: np.ndarray) -> np.ndarray:
    """The points of sorted ``y`` that the cutting planes of :meth:`_Constraints.nearest` add: every ``bad`` point
    where there are at most :data:`THIN_ABOVE` of them, else of each run of consecutive ones :data:`CUT_POINTS`
    evenly spaced points (its ends among them) and the one with the lowest ``score``.

    On noisy quotes adding every failing point grew the constraint grid to 1,000-2,000 points (the dense check is
    2,001), and the solver's quadratic subproblems with it, which BLAS threads then slowed further: 20 expiries of 41
    strikes with 8 vol points of noise took 12-14 s. The quotes the app is meant for fail at a few to 150 points, where
    all are added as before. Three points per run were too few: on a slice whose previous smile bends sharply, the
    stretch where the calendar constraint failed narrowed by only a quarter each round and was still there after
    :data:`CUTTING_ROUNDS` rounds, from every start.
    """
    if bad.sum() <= THIN_ABOVE:
        return y[bad]
    edges = np.flatnonzero(np.diff(np.concatenate([[0], bad.astype(np.int8), [0]])))
    picked: list[int] = []
    for start, end in zip(edges[::2], edges[1::2], strict=True):  # each run is y[start:end]
        picked += np.linspace(start, end - 1, min(end - start, CUT_POINTS)).round().astype(int).tolist()
        picked.append(start + int(np.argmin(score[start:end])))
    return np.unique(y[picked])


def static_arbitrage(
    fit: SVISlice,
    quoted: tuple[float, float],
    T: float,
    previous: tuple[SVISlice, tuple[float, float]] | None,
    *,
    reach: float | None = None,
) -> tuple[bool, bool]:
    """``(butterfly, calendar)``: whether ``fit`` breaks the constraints :func:`fit_svi` holds an arbitrage-free fit
    to: butterfly arbitrage over its ``quoted`` range of ``y`` widened by :data:`CONSTRAINT_REACH` and out to
    ``reach``, calendar arbitrage against ``previous`` on the overlap of the two quoted ranges.

    ``reach`` defaults to ``fit``'s own :func:`wing_reach`, but :func:`fit_svi` holds an arbitrage-free fit out to the
    reach of the unconstrained fit, which the surface also reads it to (``wing_reach(unconstrained)``): pass that to
    check a constrained fit. Its own reach can lie further out, where nothing held its density positive: on one 8 vol
    point noisy surface two arbitrage-free fits had g <= 0 at 30 and 53 points between the solver's reach (0.73 and
    0.81) and their own (0.75 and 1.21), and none within the solver's."""
    return _Constraints.build(quoted[0], quoted[1], T, previous, wing_reach(fit) if reach is None else reach).broken(
        fit
    )
