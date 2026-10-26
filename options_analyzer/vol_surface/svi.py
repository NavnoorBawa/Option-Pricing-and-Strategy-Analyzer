"""Gatheral's raw SVI smile at one maturity, and its least-squares calibration to implied vols.

:class:`SVISlice` is the smile ``w(y) = a + b (rho (y - m) + sqrt((y - m)^2 + s^2))`` in total implied variance at
forward log-moneyness ``y``, with analytic ``dw/dy`` and ``d2w/dy2``; :func:`fit_svi` fits it to one maturity's
implied vols, from Zeliade's quasi-explicit starting points (:func:`_svi_start_points`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

QUOTE_PRECISION = 1e-4  # implied-vol precision of a quote: differences below it are noise


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

    @property
    def min_variance(self) -> float:
        return self.a + self.b * self.s * np.sqrt(1.0 - self.rho * self.rho)


_RHO_FACE = 0.99  # |rho| of the starting points that sit on the edge of the admissible region


def _svi_start_points(y, iv, T, n_best: int = 3) -> list[np.ndarray]:
    """Quasi-explicit starting points (Zeliade): for fixed ``(m, s)`` SVI is linear in ``(a, c = b s, d = rho c)``.

    On a grid of ``(m, s)`` the weighted linear least-squares problem is solved subject to ``|d| < c``: the
    unconstrained solution where it satisfies that, otherwise the better of the two edges ``d = +-0.99 c``.
    Clipping an unconstrained solution instead (``c`` to about 0, or ``rho`` alone to 0.99) kept almost none of
    a steep, nearly linear skew's slope, and every start was a flat smile the optimiser could not leave.
    """
    w = iv * iv * T
    weight = 1.0 / (2.0 * iv * T)  # d(iv) = dw / (2 iv T): weighted variance residuals approximate vol residuals
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


def fit_svi(y, iv, T: float) -> SVISlice:
    """Least-squares raw-SVI fit to implied vols ``iv`` at forward log-moneyness ``y`` and maturity ``T``.

    The residuals are implied-vol errors. ``b >= 0``, ``|rho| < 1``, an ``s`` from
    a kink to a parabola and a vertex ``m`` within a few quoted ranges of the
    quotes hold by construction (``b = |beta|``, ``rho = 0.999 tanh(r)``,
    ``ln s`` and ``m`` mapped through ``tanh`` onto their ranges), with no
    parameter whose gradient is cut to zero; a negative minimum variance and a
    breach of Lee's wing bound ``b (1 + |rho|) <= 2`` are penalised. Several
    quasi-explicit starting points (:func:`_svi_start_points`) are refined and
    the best fit is kept. Noise-free SVI (hence SSVI) quotes are recovered to
    solver precision.
    """
    y = np.asarray(y, dtype=float)
    iv = np.asarray(iv, dtype=float)
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
        return np.concatenate([np.sqrt(w / T) - iv, [penalty * vega * max(-wmin, 0.0), penalty * max(lee, 0.0)]])

    def jacobian(x):
        _, beta, r, mu, sigma = x
        a, b, rho, m, s = unpack(x)
        z = y - m
        root = np.sqrt(z * z + s * s)
        w = a + b * (rho * z + root)
        scale = np.where(w > 0, 0.5 / np.sqrt(np.maximum(w, 1e-300) * T), 0.0)
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
        for a, b, rho, m, s in _svi_start_points(y, iv, T)
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
    return SVISlice(*map(float, unpack(best.x)))
