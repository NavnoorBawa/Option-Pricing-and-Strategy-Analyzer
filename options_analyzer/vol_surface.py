"""Implied and local volatility surfaces.

Surfaces are handled in *total implied variance* ``w(y, T) = sigma_BS^2 T``
as a function of forward log-moneyness ``y = ln(K / F_T)``, and local
volatility is obtained from Gatheral's form of Dupire's formula
(Gatheral, *The Volatility Surface*, 2006, eq. 1.10)::

    sigma_loc^2 = (dw/dT) / g(y, T),
    g = 1 - (y/w) w' + 1/4 (-1/4 - 1/w + y^2/w^2) w'^2 + 1/2 w''

with ``'`` denoting derivatives in ``y`` at fixed ``T`` and ``dw/dT`` taken at
fixed ``y``. Working in total variance avoids differentiating option prices
numerically (the classic source of noise in Dupire's formula), and the two
factors are exactly the static no-arbitrage conditions: ``dw/dT >= 0``
(no calendar arbitrage) and ``g >= 0`` (Durrleman's condition, no butterfly
arbitrage). Points violating either have no valid local volatility; they are
reported and left as NaN instead of being silently patched.

Two sources are supported:

* a parametric **SSVI** surface (Gatheral & Jacquier, 2014) with a
  power-law curvature function and a mean-reverting ATM variance term
  structure - free of static arbitrage by construction when
  ``eta (1 + |rho|) <= 2``;
* **market quotes** (strike, maturity and an implied vol, or an option price
  that is inverted with Black-Scholes-Merton). Each maturity is fitted with
  Gatheral's raw SVI smile, which gives ``w'`` and ``w''`` analytically.
  ``dw/dT`` at fixed ``y`` comes from secants between neighbouring maturities
  evaluated at the same ``y``: their Fritsch-Butland weighted harmonic mean
  inside, the one-sided three-point formula at the first maturity and the
  one-sided secant at the last. A secant that falls by more than the quotes'
  precision is calendar arbitrage; one that changes by less resolves nothing
  and is not used, and where no secant is left local vol is undefined.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import ndtr

from . import black_scholes as bs
from .implied_vol import ImpliedVolError, implied_volatility

__all__ = [
    "QuoteParseError",
    "QuoteWarning",
    "SSVIParams",
    "SVISlice",
    "VolSurface",
    "fit_market_surface",
    "fit_svi",
    "parse_quotes",
    "sample_quotes",
    "ssvi_surface",
    "surface_summary",
]


# ---------------------------------------------------------------------------
# Gatheral local variance
# ---------------------------------------------------------------------------


def durrleman_g(y, w, w_y, w_yy):
    """Durrleman's butterfly-arbitrage function (the local-variance denominator)."""
    return 1.0 - (y / w) * w_y + 0.25 * (-0.25 - 1.0 / w + y**2 / w**2) * w_y**2 + 0.5 * w_yy


def _local_vol(y, w, w_y, w_yy, w_t):
    g = durrleman_g(y, w, w_y, w_yy)
    calendar_bad = w_t < 0
    butterfly_bad = g <= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        local_var = w_t / g
    ok = ~calendar_bad & ~butterfly_bad & np.isfinite(local_var) & (local_var >= 0)
    return np.where(ok, np.sqrt(np.where(ok, local_var, 0.0)), np.nan), calendar_bad, butterfly_bad


# ---------------------------------------------------------------------------
# SVI slices
# ---------------------------------------------------------------------------


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


QUOTE_PRECISION = 1e-4  # implied-vol precision of a quote: differences below it are noise
MATURITY_TOLERANCE = 0.5 / 365  # maturities this close (half a day, in years) are the same expiry
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


# ---------------------------------------------------------------------------
# Surfaces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VolSurface:
    """Implied and local volatility on a (maturity x moneyness) grid; moneyness = K / S.

    Market surfaces also carry the cleaned quotes (with the fitted vol at each
    quote), the per-maturity SVI fits, their implied-vol RMSE and the quoted
    forward log-moneyness range of every maturity. Implied vol is NaN outside a
    maturity's quoted strike range and local vol is NaN wherever it is undefined.
    """

    moneyness: np.ndarray
    maturities: np.ndarray
    implied_vol: np.ndarray
    local_vol: np.ndarray
    calendar_violations: int
    butterfly_violations: int
    source: str
    notes: tuple[str, ...] = field(default_factory=tuple)
    carry: float = 0.0
    quotes: pd.DataFrame | None = None
    fit_rmse: np.ndarray | None = None
    svi: tuple[SVISlice, ...] = ()
    quoted_y: np.ndarray | None = None
    ssvi: SSVIParams | None = None

    @property
    def strikes_over_spot(self) -> np.ndarray:
        return self.moneyness

    @property
    def quoted_points(self) -> int:
        """Grid points where the surface has an implied vol (inside the quoted range)."""
        return int(np.isfinite(self.implied_vol).sum())

    def to_frames(self, spot: float) -> dict[str, pd.DataFrame]:
        """Tidy tables for export (strikes in currency units across, maturities in years down)."""
        index = pd.Index(np.round(self.maturities, 6), name="Maturity (years) \\ Strike")
        columns = pd.Index(np.round(self.moneyness * spot, 6), name="Strike")
        return {
            "ImpliedVol": pd.DataFrame(self.implied_vol, index=index, columns=columns),
            "LocalVol": pd.DataFrame(self.local_vol, index=index, columns=columns),
        }

    def implied_vol_at(self, moneyness: float, maturity: float) -> float:
        """Implied vol at ``K/S = moneyness`` and ``T = maturity``; NaN outside the surface's quoted region.

        Market surfaces interpolate total variance linearly in maturity at fixed
        forward log-moneyness between the two SVI slices around ``maturity``,
        and only where both slices were quoted at that strike. A maturity within
        half a day of a quoted one is read off that slice (quote files often
        round maturities, e.g. 30 days to 0.0822 years).
        """
        T, Ts = float(maturity), self.maturities
        if not (moneyness > 0 and Ts[0] - MATURITY_TOLERANCE <= T <= Ts[-1] + MATURITY_TOLERANCE):
            return np.nan
        if self.ssvi is not None:
            if not self.moneyness[0] <= moneyness <= self.moneyness[-1]:
                return np.nan
            y = np.log(moneyness) - self.carry * T
            return float(np.sqrt(self.ssvi.total_variance(y, T)[0] / T))
        if not self.svi or self.quoted_y is None:
            return np.nan
        nearest = int(np.argmin(np.abs(Ts - T)))
        snap = abs(Ts[nearest] - T) <= MATURITY_TOLERANCE
        if snap:
            T = float(Ts[nearest])
        y = np.log(moneyness) - self.carry * T
        j = nearest if snap else int(np.searchsorted(Ts, T)) - 1
        around = [j] if snap else [j, j + 1]
        if any(not self.quoted_y[k, 0] - 1e-9 <= y <= self.quoted_y[k, 1] + 1e-9 for k in around):
            return np.nan
        variances = [float(self.svi[k].total_variance(y)[0]) for k in around]
        w = variances[0] if snap else variances[0] + (variances[1] - variances[0]) * (T - Ts[j]) / (Ts[j + 1] - Ts[j])
        return float(np.sqrt(w / T)) if w > 0 else np.nan

    def at_moneyness(self, levels) -> tuple[np.ndarray, np.ndarray]:
        """``(implied, local)`` vol at K/S ``levels`` on every maturity of the surface (rows).

        Evaluated from the model itself (SSVI, or the SVI slices with the same
        ``dw/dT`` as the grid), not interpolated on the grid, so a coarse grid
        does not bias them. NaN outside the surface's K/S range (SSVI), outside
        a maturity's quoted strikes (market quotes), or where local vol is undefined.
        """
        levels = np.asarray(levels, dtype=float)
        if self.ssvi is not None:
            T = self.maturities[:, None]
            y = np.log(levels)[None, :] - self.carry * T
            w, w_y, w_yy, w_t = self.ssvi.total_variance(y, T)
            local = _local_vol(y, w, w_y, w_yy, w_t)[0]
            outside = (levels < self.moneyness[0] - 1e-12) | (levels > self.moneyness[-1] + 1e-12)
            return np.where(outside, np.nan, np.sqrt(w / T)), np.where(outside, np.nan, local)
        if self.svi and self.quoted_y is not None:
            values = _market_values(self.svi, self.maturities, self.quoted_y, self.carry, levels)
            return values.implied, values.local
        m = self.moneyness
        outside = (levels < m[0]) | (levels > m[-1])

        def grid(values):  # np.interp gives NaN next to a NaN, so it never bridges a gap
            return np.array([np.where(outside, np.nan, np.interp(levels, m, row)) for row in values])

        return grid(self.implied_vol), grid(self.local_vol)


# ---------------------------------------------------------------------------
# SSVI
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SSVIParams:
    atm_vol_short: float = 0.25  # ATM implied vol as T -> 0
    atm_vol_long: float = 0.20  # ATM implied vol as T -> infinity
    mean_reversion: float = 1.5  # speed at which ATM variance moves from short to long level
    rho: float = -0.6  # skew: correlation-like parameter in (-1, 1)
    eta: float = 1.0  # smile curvature level
    gamma: float = 0.5  # power-law decay of curvature with maturity, in (0, 0.5]

    def __post_init__(self):
        if not (self.atm_vol_short > 0 and self.atm_vol_long > 0):
            raise ValueError("ATM volatilities must be positive")
        if not self.mean_reversion > 0:
            raise ValueError("mean reversion must be positive")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must be strictly between -1 and 1")
        if not self.eta > 0:
            raise ValueError("eta must be positive")
        if not 0.0 < self.gamma <= 0.5:
            raise ValueError("gamma must be in (0, 0.5]")

    @property
    def max_arbitrage_free_eta(self) -> float:
        """Largest ``eta`` satisfying the sufficient condition ``eta (1 + |rho|) <= 2``."""
        return 2.0 / (1.0 + abs(self.rho))

    def theta(self, T):
        """ATM total variance: the time integral of a mean-reverting variance."""
        v0, vinf, k = self.atm_vol_short**2, self.atm_vol_long**2, self.mean_reversion
        return vinf * T + (v0 - vinf) * (1.0 - np.exp(-k * T)) / k

    def theta_prime(self, T):
        v0, vinf, k = self.atm_vol_short**2, self.atm_vol_long**2, self.mean_reversion
        return vinf + (v0 - vinf) * np.exp(-k * T)

    def phi(self, theta):
        return self.eta / (theta**self.gamma * (1.0 + theta) ** (1.0 - self.gamma))

    def phi_prime(self, theta):
        return self.phi(theta) * (-self.gamma / theta + (self.gamma - 1.0) / (1.0 + theta))

    def total_variance(self, y, T):
        """``(w, dw/dy, d2w/dy2, dw/dT)`` of the SSVI surface (all analytic)."""
        theta = self.theta(T)
        phi = self.phi(theta)
        rho = self.rho
        u = phi * y + rho
        root = np.sqrt(u * u + 1.0 - rho * rho)
        w = 0.5 * theta * (1.0 + rho * phi * y + root)
        w_y = 0.5 * theta * phi * (rho + u / root)
        w_yy = 0.5 * theta * phi * phi * (1.0 - rho * rho) / root**3
        dphi = self.phi_prime(theta)
        dw_dtheta = 0.5 * (1.0 + rho * phi * y + root) + 0.5 * theta * dphi * y * (rho + u / root)
        w_t = dw_dtheta * self.theta_prime(T)
        return w, w_y, w_yy, w_t


def ssvi_surface(
    params: SSVIParams,
    rate: float,
    dividend_yield: float,
    moneyness: np.ndarray,
    maturities: np.ndarray,
) -> VolSurface:
    """Evaluate an SSVI surface and its local volatility on a grid."""
    m = np.asarray(moneyness, dtype=float)[None, :]
    T = np.asarray(maturities, dtype=float)[:, None]
    y = np.log(m) - (rate - dividend_yield) * T
    w, w_y, w_yy, w_t = params.total_variance(y, T)
    implied = np.sqrt(w / T)
    local, cal, fly = _local_vol(y, w, w_y, w_yy, w_t)
    notes = []
    if params.eta > params.max_arbitrage_free_eta + 1e-12:
        notes.append(
            f"eta = {params.eta:.2f} exceeds 2/(1+|rho|) = {params.max_arbitrage_free_eta:.2f}; "
            "the surface may contain butterfly arbitrage."
        )
    return VolSurface(
        np.asarray(moneyness, float),
        np.asarray(maturities, float),
        implied,
        local,
        int(cal.sum()),
        int(fly.sum()),
        "SSVI",
        tuple(notes),
        carry=rate - dividend_yield,
        ssvi=params,
    )


# ---------------------------------------------------------------------------
# Market quotes: parsing
# ---------------------------------------------------------------------------


class QuoteParseError(ValueError):
    """The uploaded quotes cannot be turned into a surface."""


class QuoteWarning(str):
    """A parse message about a unit that had to be guessed: the reading may be wrong, so show it prominently."""


_UNITS = {
    "%": "percent",
    "pct": "percent",
    "percent": "percent",
    "d": "days",
    "day": "days",
    "days": "days",
    "y": "years",
    "yr": "years",
    "yrs": "years",
    "year": "years",
    "years": "years",
    **dict.fromkeys(("mo", "mos", "mon", "mth", "mths", "month", "months"), "months"),
    **dict.fromkeys(("w", "wk", "wks", "week", "weeks"), "weeks"),
    **dict.fromkeys(("h", "hr", "hrs", "hour", "hours"), "hours"),
    **dict.fromkeys(("min", "mins", "minute", "minutes"), "minutes"),
    "m": "m",  # months in tenor notation ("3M"), minutes elsewhere: refused as a unit of time to expiry
}
_UNIT_SUFFIXES = (
    ("percent", "percent"),
    ("pct", "percent"),
    ("years", "years"),
    ("yrs", "years"),
    ("days", "days"),
    ("months", "months"),
    ("weeks", "weeks"),
    ("hours", "hours"),
    ("minutes", "minutes"),
)
_BRACKETS = re.compile(r"[(\[{]([^)\]}]*)[)\]}]")
# Units of a time to expiry other than days and years: days per unit, and how the conversion is described.
_TIME_UNITS = {
    "months": (365.0 / 12.0, "months, divided by 12"),
    "weeks": (7.0, "weeks of 7 days, converted to years with ACT/365"),
    "hours": (1.0 / 24.0, "hours, converted to years with ACT/365"),
    "minutes": (1.0 / 1440.0, "minutes, converted to years with ACT/365"),
}


def _header(name) -> tuple[str, str | None]:
    """A column name reduced to lowercase letters and digits, plus the unit it states.

    The unit is percent, days, years, months, weeks, hours or minutes, or ``"m"``, which could be months or
    minutes. Bracketed text next to a name is removed but read for a unit (``"IV (% p.a.)"`` ->
    ``("iv", "percent")``, ``"Maturity (months)"`` -> ``("maturity", "months")``), as are a ``%`` sign and a
    trailing unit word (``"Maturity_years"`` -> ``("maturity", "years")``). A name wrapped entirely in brackets
    keeps its text (``"[STRIKE]"`` -> ``"strike"``).
    """
    text = str(name).strip().lower()
    unit = "percent" if "%" in text else None
    outside = _BRACKETS.sub(" ", text)
    if re.search(r"[a-z0-9]", outside):
        for inner in _BRACKETS.findall(text):
            for word in re.findall(r"[a-z]+", inner):
                unit = unit or _UNITS.get(word)
        text = outside
    base = re.sub(r"[^a-z0-9]", "", text)
    for suffix, suffix_unit in _UNIT_SUFFIXES:
        if base.endswith(suffix) and len(base) > len(suffix):
            base, unit = base[: -len(suffix)], unit or suffix_unit
            break
    if base.startswith("year"):  # "Years", "Years to expiry"
        unit = unit or "years"
    return base, unit


# Most specific first: "Implied Volatility" must win over a bare "Vol", which is often trading volume.
_ALIASES = {
    "strike": ("Strike", "Strike price", "K"),
    "iv": ("Implied volatility", "Implied vol", "Impl vol", "Imp vol", "Mid IV", "IV", "Volatility", "Vol", "Sigma"),
    "days": ("Days", "DTE", "Days to expiry", "Days to expiration", "Days to maturity", "Calendar days"),
    "maturity": (
        "Maturity",
        "Time to maturity",
        "Time to expiry",
        "Time to expiration",
        "Years to expiry",
        "Years to maturity",
        "Expiry",
        "Expiration",
        "Tenor",
        "TTM",
        "Years",
        "T",
    ),
    "type": ("Type", "Option type", "Call/Put", "Put/Call", "CP", "Right"),
    "price": ("Mid", "Mid price", "Price", "Option price", "Mark", "Premium"),
    "bid": ("Bid", "Bid price"),
    "ask": ("Ask", "Ask price", "Offer"),
    "last": ("Last", "Last price"),
}
_ALIAS_KEYS = {key: tuple(_header(alias)[0] for alias in names) for key, names in _ALIASES.items()}
_WEAK_IV = _header("Vol")[0]

MAX_ROWS = 200_000
MAX_PRICE_QUOTES = 10_000
MIN_QUOTED_VOL = 1e-4  # implied vols below 0.01% are placeholders (some data feeds write 1e-05 for none)


def _holds_times(frame: pd.DataFrame, column, unit: str | None) -> bool:
    """Whether a column has numbers that can be a time to expiry (not dates, not YYYYMMDD integers)."""
    days_per_unit = _TIME_UNITS[unit][0] if unit in _TIME_UNITS else 1.0
    try:
        return bool(_numeric(frame, column)[0].max() * days_per_unit <= MAX_DAYS)
    except QuoteParseError:
        return False


def _match_columns(frame: pd.DataFrame) -> dict[str, tuple[str, str | None]]:
    headers = {column: _header(column) for column in frame.columns}
    found: dict[str, tuple[str, str | None]] = {}
    claimed: set = set()
    for key, aliases in _ALIAS_KEYS.items():
        candidates = [c for alias in aliases for c, (base, _) in headers.items() if base == alias and c not in claimed]
        if not candidates:
            continue
        column = candidates[0]
        if key == "maturity":  # an expiry-date column often sits next to the time to expiry ("Expiration", "TTM")
            column = next((c for c in candidates if _holds_times(frame, c, headers[c][1])), column)
        found[key] = (column, headers[column][1])
        claimed.add(column)
    if "maturity" in found and found["maturity"][1] == "days":  # e.g. "Maturity (days)"
        days = found.pop("maturity")
        found.setdefault("days", days)
    return found


_THOUSANDS = r"[-+]?(?:[1-9]\d{0,2}(?:,\d{3})+|[1-9]\d?(?:,\d{2})+,\d{3})(?:\.\d*)?"  # 1,250,000.5 or 12,50,000.5
_DECIMAL_COMMA = r"[-+]?\d*,\d+"  # 25,3 or 0,0822


def _numeric(
    frame: pd.DataFrame, column, messages: list[str] | None = None, required: bool = True
) -> tuple[pd.Series, bool]:
    """A column as floats, with %, thousands separators, currency signs and spaces stripped.

    Commas are thousands separators when every number written with one is grouped that way (``1,250.50``), and
    decimal commas when every one has a single comma and no number in the column has a decimal point
    (``25,3``), which is then said in ``messages``; any other use of commas is refused. Also returns whether
    any value carried a ``%`` sign. A column with no numeric value at all raises :class:`QuoteParseError`
    naming it, or comes back as all NaN when it is not ``required``.
    """
    raw = frame[column]
    percent = False
    if pd.api.types.is_numeric_dtype(raw) and not pd.api.types.is_bool_dtype(raw):
        values = raw.astype(float)
    else:
        text = raw.astype("string").str.strip()
        percent = bool(text.str.endswith("%").fillna(False).any())
        text = text.str.replace(r"[\s$€£%]", "", regex=True)
        numbers = text.str.contains(r"\d", regex=True).fillna(False)
        commas = text[numbers & text.str.contains(",", regex=False).fillna(False)]
        points = bool((numbers & text.str.contains(".", regex=False).fillna(False)).any())
        if commas.str.fullmatch(_THOUSANDS).all():
            text = text.str.replace(",", "", regex=False)
        elif commas.str.fullmatch(_DECIMAL_COMMA).all() and not points:
            text = text.str.replace(",", ".", regex=False)
            if messages is not None:
                messages.append(f"Read the commas in '{column}' as decimal points (e.g. '{commas.iloc[0]}').")
        else:
            raise QuoteParseError(
                f"column '{column}' has commas that are neither thousands separators nor decimal commas "
                f"(e.g. '{commas.iloc[0]}'); write its numbers with a decimal point and no other separator"
            )
        values = pd.to_numeric(text, errors="coerce").astype(float)
    if required and not values.notna().any():
        sample = raw.dropna()
        example = f" (e.g. '{sample.iloc[0]}')" if len(sample) else ""
        raise QuoteParseError(f"column '{column}' has no numeric values{example}")
    return values, percent


MAX_DAYS = 36_500  # 100 years: larger "maturities" are neither years nor days (e.g. YYYYMMDD dates)
_TENOR = r"\d+(?:\.\d+)?\s*[dwmy]"  # "1M", "3m", "1W", "2Y", "30D"
_TIME_OF_DAY = r"\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:[ap]\.?m\.?)?"  # "16:00", "9:30 am": pandas dates them today


def _looks_like_dates(sample: pd.Series) -> bool:
    """Whether most of a column's text values are calendar dates, with or without a year.

    pandas gives a date without a year ("Nov 20", "11/20") the year 1, and a time alone the year 1 and 1 January
    ("5h", "5pm") or today's date ("16:00"), so those times are not counted.
    """
    dates = pd.to_datetime(sample, errors="coerce", format="mixed", utc=True)
    yearless = (dates.dt.year == 1) & ((dates.dt.month != 1) | (dates.dt.day != 1))
    dated = ((dates.dt.year >= 1900) | yearless) & ~sample.str.lower().str.fullmatch(_TIME_OF_DAY)
    return bool(len(sample)) and bool(dated.mean() > 0.5)


def _time_column(
    frame: pd.DataFrame, column, messages: list[str] | None = None, days_per_unit: float = 1.0
) -> pd.Series:
    """A maturity or days column as floats; dates, tenors and impossible values raise :class:`QuoteParseError`.

    ``days_per_unit`` is the column's unit in days when its header states one (7 for weeks), so values worth
    more than 100 years are refused in any unit.
    """
    try:
        values = _numeric(frame, column, messages)[0]
    except QuoteParseError:
        sample = frame[column].dropna().astype(str).str.strip().head(50)
        if len(sample) and sample.str.lower().str.fullmatch(_TENOR).mean() > 0.5:
            raise QuoteParseError(
                f"column '{column}' holds tenors (e.g. '{sample.iloc[0]}'), but the surface needs the time to expiry "
                "as a number: add a Days column (calendar days to expiry) or a Maturity column in years"
            ) from None
        if _looks_like_dates(sample):
            raise QuoteParseError(
                f"column '{column}' holds dates (e.g. '{sample.iloc[0]}'), but the surface needs the time to expiry: "
                "add a Days column (calendar days to expiry) or a Maturity column in years"
            ) from None
        raise
    largest = values.max()
    if largest * days_per_unit > MAX_DAYS:
        hint = " (8-digit values look like YYYYMMDD dates)" if 19_000_101 <= largest <= 29_991_231 else ""
        what = "neither years nor days" if days_per_unit == 1.0 else "more than 100 years"
        raise QuoteParseError(
            f"column '{column}' goes up to {largest:,.0f}, which is {what}{hint}; "
            "add a Days column (calendar days to expiry) or a Maturity column in years"
        )
    return values


MAX_USUAL_YEARS = 3.0  # listed options rarely run longer: unlabelled maturities beyond this may well be days
_NAME_THE_UNIT = "name the column 'Days' if they are days to expiry, or 'Maturity (years)' if they are years"


@dataclass(frozen=True)
class _GuessedYears:
    """A maturity column read as years although its header states no unit."""

    column: str
    described: str  # what its values look like: "go up to 9.65", "are whole numbers up to 4"
    suspect: bool  # whether the values alone are reason to doubt years


def _maturities(
    frame: pd.DataFrame, found: dict, used: list[str], messages: list[str]
) -> tuple[pd.Series, _GuessedYears | None]:
    """Time to expiry in years, from a days column if there is one, otherwise from a maturity column.

    A unit of months, weeks, hours or minutes stated in the header is converted; ``"m"`` alone (months or
    minutes) is refused. A maturity column whose header does not say years is converted from days when its
    values go above 30 (no listed option runs that many years). Up to 30, whole numbers from 5 up are as likely
    to be years (a 1, 2, 3, 5, 7, 10-year tenor grid) as days (a weekly chain at 7, 14, 21, 28), so the file is
    refused with a request to name the unit; anything smaller is read as years. Returns ``(years, guess)``:
    ``guess`` is a :class:`_GuessedYears` when years was only assumed (suspect for small whole numbers and
    values above :data:`MAX_USUAL_YEARS`), else None; :func:`parse_quotes` warns about the suspect ones.
    """
    column, unit = found["days"] if "days" in found else found["maturity"]
    if unit == "m":
        raise QuoteParseError(
            f"the unit 'm' in '{column}' could mean months or minutes; name the column 'Maturity (months)', "
            "'Days' or 'Maturity (years)'"
        )
    if "days" in found and "maturity" in found:
        messages.append(f"Ignored '{found['maturity'][0]}': the time to expiry comes from '{column}'.")
    if unit in _TIME_UNITS:
        days_per_unit, described = _TIME_UNITS[unit]
        used.append(f"maturity from '{column}' ({described})")
        return _time_column(frame, column, messages, days_per_unit) * (days_per_unit / 365.0), None
    if "days" in found:
        used.append(f"maturity from '{column}' (days, converted to years with ACT/365)")
        return _time_column(frame, column, messages) / 365.0, None
    values = _time_column(frame, column, messages)
    positive = values[values > 0]
    largest = values.max()
    whole = bool(len(positive)) and bool(np.all(positive == positive.round()))
    if largest > 30:
        messages.append(
            f"Maturities in '{column}' go up to {largest:g}, which only makes sense in days; converted days to "
            "years (ACT/365). Name the column 'Days' to make this explicit."
        )
        used.append(f"maturity from '{column}' (days)")
        return values / 365.0, None
    used.append(f"maturity from '{column}' (years)")
    if unit == "years":
        return values, None
    if whole and largest >= 5:
        raise QuoteParseError(
            f"the maturities in '{column}' are whole numbers from {positive.min():g} to {largest:g}, which "
            f"could be years or days; {_NAME_THE_UNIT}"
        )
    if whole:
        return values, _GuessedYears(column, f"are whole numbers up to {largest:g}", True)
    return values, _GuessedYears(column, f"go up to {largest:g}", largest > MAX_USUAL_YEARS)


SPREAD_CHECK_VOL = 0.01  # vol error allowed on each quote before prices that run the wrong way with strike count
FLY_CHECK_VOL = 0.005  # vol error allowed on each quote before a butterfly with a negative cost counts
# Price error allowed on a quoted implied vol, whose price and tick are unknown, as a fraction of the forward.
UNKNOWN_PRICE_ERROR = 1e-4


def _black_prices(strike, iv, T: float, forward: float, vol_error: float, price_error=0.0):
    """Strikes in ascending order with their undiscounted Black call and put prices at ``forward``, and the slack.

    The slack is how far a price moves when its vol is ``vol_error`` off (vega times the error), plus each
    quote's undiscounted ``price_error`` (its rounding or half its bid/ask spread) and 1e-12 of the forward.
    """
    order = np.argsort(strike, kind="stable")
    strike, iv = np.asarray(strike, dtype=float)[order], np.asarray(iv, dtype=float)[order]
    price_error = np.broadcast_to(np.asarray(price_error, dtype=float), order.shape)[order]
    sd = iv * np.sqrt(T)
    d1 = np.log(forward / strike) / sd + 0.5 * sd
    # Both prices directly: a call from put-call parity is rounding noise far out of the money.
    put = strike * ndtr(sd - d1) - forward * ndtr(-d1)
    call = forward * ndtr(d1) - strike * ndtr(d1 - sd)
    vega = forward * np.sqrt(T) * np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)
    return strike, call, put, vol_error * vega + price_error + 1e-12 * forward


def _spread_arbitrage(strike, iv, T: float, forward: float, price_error=0.0) -> bool:
    """Whether quoted vols price some put below a put at a lower strike, or some call above a call at a lower strike.

    Undiscounted Black prices at ``forward``; each quote's vol may be off by :data:`SPREAD_CHECK_VOL`, and its
    price by its undiscounted ``price_error``, in whichever direction removes the arbitrage. No arbitrage-free
    market prices a put below a lower-strike put or a call above a lower-strike call, so this is model-free.
    """
    _, call, put, slack = _black_prices(strike, iv, T, forward, SPREAD_CHECK_VOL, price_error)
    put_floor = np.maximum.accumulate(put - slack)[:-1]  # a put at a higher strike is worth at least this
    call_cap = np.minimum.accumulate(call + slack)[:-1]  # a call at a higher strike is worth at most this
    return bool(np.any(put[1:] + slack[1:] < put_floor) or np.any(call[1:] - slack[1:] > call_cap))


def _butterfly_arbitrage(
    strike, iv, T: float, forward: float, price_error=0.0
) -> tuple[int, tuple[float, float, float] | None]:
    """Butterflies of three neighbouring quoted strikes that cost less than nothing: ``(count, worst strikes)``.

    A long butterfly ``lam C(K1) - C(K2) + (1 - lam) C(K3)`` with ``K2 = lam K1 + (1 - lam) K3`` never pays
    less than zero, so in an arbitrage-free market it costs at least zero (call prices are convex in strike;
    put butterflies cost the same by put-call parity). Undiscounted Black prices at ``forward``; each quote's
    vol may be off by :data:`FLY_CHECK_VOL`, and its price by its undiscounted ``price_error``, in whichever
    direction removes the arbitrage, so quote noise and price rounding are not reported (with half the bid/ask
    spread as the price error, a reported butterfly costs less than nothing bought at the ask and sold at the
    bid). Model-free: it checks the quotes, not a fitted smile. Repeated strikes are averaged first.
    """
    unique, which = np.unique(np.asarray(strike, dtype=float), return_inverse=True)
    if len(unique) < 3:
        return 0, None
    count = np.bincount(which)
    iv = np.bincount(which, np.asarray(iv, dtype=float)) / count
    price_error = np.broadcast_to(np.asarray(price_error, dtype=float), which.shape)
    K, call, _, slack = _black_prices(unique, iv, T, forward, FLY_CHECK_VOL, np.bincount(which, price_error) / count)
    lam = (K[2:] - K[1:-1]) / (K[2:] - K[:-2])
    cost = lam * (call[:-2] + slack[:-2]) + (1.0 - lam) * (call[2:] + slack[2:]) - (call[1:-1] - slack[1:-1])
    bad = cost < 0
    if not bad.any():
        return 0, None
    worst = int(np.argmin(np.where(bad, cost, np.inf)))
    return int(bad.sum()), (float(K[worst]), float(K[worst + 1]), float(K[worst + 2]))


def _strike_text(strike: float) -> str:
    """A strike for a message: thousands separators and no trailing zeros (``2,452.5``, ``100``)."""
    return f"{strike:,.2f}".rstrip("0").rstrip(".")


def _option_kind(frame: pd.DataFrame, column) -> pd.Series:
    first = frame[column].astype("string").str.strip().str.lower().str[:1]
    return first.map({"c": "call", "p": "put"})


def _expiries(maturity) -> np.ndarray:
    """Expiry labels for times to expiry in years: each expiry spans at most :data:`MATURITY_TOLERANCE`.

    A label starts at the shortest time not yet labelled and takes every time up to half a day longer, so one
    expiry quoted at different times of a session (fractional days to expiry) gets one label, while listed
    expiries, at least a day apart, keep their own; times spread over more than half a day are never chained
    into one expiry.
    """
    maturity = np.asarray(maturity, dtype=float)
    unique = np.unique(maturity)
    labels = np.zeros(len(unique), dtype=int)
    start = unique[0] if len(unique) else 0.0
    for i in range(1, len(unique)):
        if unique[i] - start > MATURITY_TOLERANCE:
            start = unique[i]
            labels[i] = labels[i - 1] + 1
        else:
            labels[i] = labels[i - 1]
    return labels[np.searchsorted(unique, maturity)]


def _merge_close_maturities(data: pd.DataFrame, messages: list[str]) -> pd.DataFrame:
    """Set the times to expiry of each expiry (:func:`_expiries`) to their median, keeping total variance."""
    maturity = data["maturity"].to_numpy(float)
    labels = _expiries(maturity)
    n_times, n_expiries = len(np.unique(maturity)), int(labels.max()) + 1
    if n_expiries == n_times:
        return data
    snapped = pd.Series(maturity).groupby(labels).transform("median").to_numpy()
    messages.append(
        f"Read {n_times} times to expiry as {n_expiries} expiries: times within half a day of each other were "
        "set to their median, keeping each quote's total implied variance."
    )
    return data.assign(maturity=snapped, iv=data["iv"].to_numpy(float) * np.sqrt(maturity / snapped))


def _out_of_the_money(
    data: pd.DataFrame, spot: float, rate: float, dividend_yield: float, messages: list[str]
) -> pd.DataFrame:
    """Where an expiry and strike are quoted for both a call and a put, keep the out-of-the-money one.

    In-the-money quotes are the less reliable side (wide spreads, stale trades, early-exercise premium, and
    the placeholder vols some data feeds write). Rows without a call/put type are kept.
    """
    forward = spot * np.exp((rate - dividend_yield) * data["maturity"])
    otm = (data["type"] == "call") == (data["strike"] >= forward)
    expiry = _expiries(data["maturity"])
    both = data.assign(expiry=expiry).groupby(["expiry", "strike"])["type"].transform("nunique") > 1
    if (both & ~otm & data["type"].notna()).any():
        messages.append(
            f"Used the out-of-the-money option at {int((both & otm).sum())} strike(s) quoted for both calls and puts."
        )
        data = data[~both | otm | data["type"].isna()]
    return data


def _prices_to_vols(
    data: pd.DataFrame, spot: float, rate: float, dividend_yield: float, messages: list[str]
) -> pd.DataFrame:
    """Invert option prices into Black-Scholes-Merton implied vols, keeping the out-of-the-money side."""
    data = _out_of_the_money(data, spot, rate, dividend_yield, messages)
    if len(data) > MAX_PRICE_QUOTES:
        raise QuoteParseError(
            f"the file has {len(data):,} option prices to invert; at most {MAX_PRICE_QUOTES:,} are supported, "
            "so filter the chain (for example to fewer expiries) first"
        )
    vols, reasons = [], []
    for price, strike, T, kind in data[["price", "strike", "maturity", "type"]].itertuples(index=False):
        try:
            vols.append(implied_volatility(price, spot, strike, T, rate, dividend_yield, kind))
        except ImpliedVolError as exc:
            vols.append(np.nan)
            reasons.append(str(exc))
    data = data.assign(iv=vols)
    if reasons:
        messages.append(
            f"Dropped {len(reasons)} price quote(s) outside the no-arbitrage bounds at spot {spot:,.2f}, "
            f"rate {rate:.2%} and dividend yield {dividend_yield:.2%} (e.g. {reasons[0]})."
        )
    no_time_value = int((data["iv"] == 0).sum())
    if no_time_value:
        messages.append(f"Dropped {no_time_value} price quote(s) with no time value (zero implied vol).")
    data = data[data["iv"] > 0]
    messages.append(
        f"Inverted {len(data)} option price(s) into implied vols with Black-Scholes-Merton (European exercise)."
    )
    return data


_PRICE_STEPS = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)


def _price_step(prices: pd.Series) -> float:
    """The coarsest usual price increment that every positive price is a whole multiple of, or 0 if none is.

    A midpoint of a bid and an ask in 0.05 ticks lies on a 0.025 grid, and the price the quotes stand for can be
    up to 0.025 from it, so a price may be off by one step.
    """
    values = prices[prices > 0].to_numpy(float)
    if not len(values):
        return 0.0
    steps = [step for step in _PRICE_STEPS if np.all(np.abs(values / step - np.round(values / step)) < 1e-6)]
    return max(steps, default=0.0)


def _option_prices(
    frame: pd.DataFrame, found: dict, used: list[str], messages: list[str]
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Each row's option price and how far it may be off, and the rows dropped for a zero bid or an ask below the bid.

    The quoted price (Price, Mid, Mark) where it is positive, else the bid/ask midpoint, else the last trade: an
    empty or partly blank price column next to valid bids and asks no longer blocks them. A bid/ask midpoint may
    be off by half the spread, and any other price by the step of its column's grid (:func:`_price_step`).
    """
    price = pd.Series(np.nan, index=frame.index)
    error = pd.Series(np.nan, index=frame.index)
    unusable = pd.Series(False, index=frame.index)
    if "price" in found:
        column = found["price"][0]
        price = _numeric(frame, column, messages, required=False)[0].where(lambda values: values > 0)
        if price.notna().any():
            used.append(f"option price from '{column}'")
            error = error.where(price.isna(), _price_step(price))
    if "bid" in found and "ask" in found and price.isna().any():
        bid = _numeric(frame, found["bid"][0], messages, required=False)[0]
        ask = _numeric(frame, found["ask"][0], messages, required=False)[0]
        need = price.isna() & bid.notna() & ask.notna()
        unusable = need & ((bid <= 0) | (ask < bid))
        if unusable.any():
            messages.append(f"Dropped {int(unusable.sum())} quote(s) with a zero bid or an ask below the bid.")
        filled = need & ~unusable
        if filled.any():
            price = price.where(~filled, (bid + ask) / 2)
            error = error.where(~filled, (ask - bid) / 2)
            where = f" where '{found['price'][0]}' is blank" if "price" in found else ""
            used.append(f"option price as the bid/ask midpoint of '{found['bid'][0]}' and '{found['ask'][0]}'{where}")
    if "last" in found and (price.isna() & ~unusable).any():
        last = _numeric(frame, found["last"][0], messages, required=False)[0]
        filled = price.isna() & ~unusable & (last > 0)
        if filled.any():
            price = price.where(~filled, last)
            error = error.where(~filled, _price_step(last))
            where = " where no other price is given" if "price" in found or "bid" in found else ""
            used.append(f"option price from '{found['last'][0]}' (last trade){where}")
    if price.isna().all() and not unusable.all():
        columns = ", ".join(repr(found[key][0]) for key in ("price", "bid", "ask", "last") if key in found)
        raise QuoteParseError(f"no numeric option price in {columns}")
    return price, error, unusable


def parse_quotes(
    frame: pd.DataFrame, spot: float | None = None, rate: float = 0.0, dividend_yield: float = 0.0
) -> tuple[pd.DataFrame, list[str]]:
    """Validate and clean a quotes table.

    Accepts (case, spacing and punctuation ignored; common aliases allowed) a
    ``Strike`` column, a ``Days`` column or a ``Maturity`` column in years (a days
    column wins when both are present), and either an ``ImpliedVol`` column
    (decimal or percent) or option prices (``Price``/``Mid``, with ``Bid`` and ``Ask``
    filling the rows where it is blank) with a call/put ``Type`` column. Prices are
    inverted with Black-Scholes-Merton at ``spot``, ``rate`` and ``dividend_yield``;
    where an expiry and strike are quoted for both a call and a put, the
    out-of-the-money one is kept, for implied vols too when a ``Type`` column and the
    spot are given. Units in brackets are read (``IV (%)``, ``Maturity (days)``,
    ``Maturity (months)``, ``Maturity (weeks)``, ``DTE (hours)``; a bare ``m`` is
    refused as months or minutes), values may carry ``%`` signs, thousands
    separators or decimal commas, a maturity column that can only hold days (see
    :func:`_maturities`) is converted, and one that could hold years or days is
    refused; values read as years without a stated unit get a warning when they
    could well be days. Dates, tenors (``3M``) and values above 100 years are
    refused. Times to expiry within half a day of each other are one expiry
    (:func:`_expiries`), and implied vols below :data:`MIN_QUOTED_VOL` are dropped as
    placeholders.

    Returns ``(clean, messages)`` where ``clean`` has columns ``strike``,
    ``maturity`` (years), ``iv`` (decimal) and ``price_error`` (how far the option
    price behind each quote may be off: half the spread for a Bid/Ask midpoint, else
    the step of its price grid, NaN for an implied vol), and ``messages`` describes
    every interpretation and adjustment; a :class:`QuoteWarning` among them flags a
    unit that was guessed. Raises :class:`QuoteParseError` if the data is unusable.
    """
    messages: list[str] = []
    if len(frame) > MAX_ROWS:
        raise QuoteParseError(f"the file has {len(frame):,} rows; at most {MAX_ROWS:,} are supported")
    found = _match_columns(frame)
    has_prices = "price" in found or ("bid" in found and "ask" in found) or "last" in found
    if has_prices and "iv" in found and _header(found["iv"][0])[0] == _WEAK_IV:
        messages.append(
            f"Read '{found['iv'][0]}' as trading volume because the file has option prices; "
            "name the column 'IV' if it holds implied vols."
        )
        del found["iv"]
    if "strike" not in found or ("maturity" not in found and "days" not in found):
        raise QuoteParseError(
            "the file needs a Strike column and a Maturity column (in years) or a Days column; "
            f"found columns: {', '.join(map(str, frame.columns))}"
        )
    if "iv" not in found and not has_prices:
        raise QuoteParseError(
            "the file needs an ImpliedVol column, or option prices (Price, or Bid and Ask) with a call/put Type "
            f"column; found columns: {', '.join(map(str, frame.columns))}"
        )

    frame = frame.reset_index(drop=True)
    data = pd.DataFrame({"strike": _numeric(frame, found["strike"][0], messages)[0]})
    used = [f"strike from '{found['strike'][0]}'"]
    reported = pd.Series(False, index=data.index)  # rows already dropped with a specific message
    data["maturity"], guess = _maturities(frame, found, used, messages)

    iv_percent = None
    if "iv" in found:
        column, unit = found["iv"]
        data["iv"], percent_signs = _numeric(frame, column, messages)
        iv_percent = "header" if unit == "percent" else "values" if percent_signs else None
        used.append(f"implied vol from '{column}'")
        if "type" in found and spot is not None:  # calls and puts stacked: keep the out-of-the-money side
            data["type"] = _option_kind(frame, found["type"][0])
            used.append(f"call/put from '{found['type'][0]}'")
    else:
        if "type" not in found:
            raise QuoteParseError(
                "option prices need a Type column saying call or put for each row; "
                f"found columns: {', '.join(map(str, frame.columns))}"
            )
        if spot is None:
            raise QuoteParseError("option prices need the spot price, rate and dividend yield to be inverted")
        data["type"] = _option_kind(frame, found["type"][0])
        data["price"], data["price_error"], unusable = _option_prices(frame, found, used, messages)
        reported |= unusable
        used.append(f"call/put from '{found['type'][0]}'")
        bad_type = data["type"].isna() & frame[found["type"][0]].notna()
        if bad_type.any():
            messages.append(f"Dropped {int(bad_type.sum())} row(s) whose type is neither call nor put.")
        reported |= bad_type
    messages.insert(0, "Read " + ", ".join(used) + ".")

    value = "iv" if "iv" in data else "price"
    # An implied vol needs no type: a row whose type is blank or unknown is kept as it is.
    checked = data.drop(columns="type") if value == "iv" and "type" in data else data
    valid = checked.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    valid &= (data["strike"] > 0) & (data["maturity"] > 0) & (data[value] > 0)
    unexplained = int((~valid & ~reported).sum())
    if unexplained:
        messages.append(f"Dropped {unexplained} row(s) with missing, non-numeric or non-positive values.")
    data = data[valid]
    if data.empty:
        raise QuoteParseError("no valid rows remain after cleaning")

    if value == "price":
        data = _prices_to_vols(data, spot, rate, dividend_yield, messages)
        if data.empty:
            raise QuoteParseError(
                "no option price could be inverted into an implied vol; check the spot, rate and units"
            )
    else:
        if "type" in data:
            data = _out_of_the_money(data, spot, rate, dividend_yield, messages)
        # MIN_QUOTED_VOL is a decimal vol: 100 times that in a file of percentages (recognised as below).
        in_percent = iv_percent is not None or data.loc[data["iv"] >= MIN_QUOTED_VOL, "iv"].median() > 3.0
        placeholder = data["iv"] < MIN_QUOTED_VOL * (100.0 if in_percent else 1.0)
        if placeholder.any():
            messages.append(
                f"Dropped {int(placeholder.sum())} quote(s) with an implied vol below {MIN_QUOTED_VOL:.2%}, a "
                "placeholder some data feeds write where there is no vol."
            )
            data = data[~placeholder]
            if data.empty:
                raise QuoteParseError("no valid rows remain after cleaning")
        median = float(data["iv"].median())
        if iv_percent is not None:
            data["iv"] = data["iv"] / 100.0
            source = "the column header" if iv_percent == "header" else "the % signs"
            messages.append(f"Implied vols read as percentages (from {source}); divided by 100.")
        elif median > 3.0:
            data["iv"] = data["iv"] / 100.0
            messages.append(f"Implied vols look like percentages (median {median:.4g}); divided by 100.")
        else:
            messages.append(f"Implied vols read as decimals (median {median:.4g} = {median:.2%}).")
        if iv_percent is None and 1.0 <= median <= 5.0:
            messages.append(
                QuoteWarning(
                    f"A median implied vol of {median:.4g} is ambiguous (it could be {median:.4g}% or {median:.0%}); "
                    "name the column 'IV (%)' if the values are percentages."
                )
            )
    if (data["iv"] > 5.0).any():
        n_bad = int((data["iv"] > 5.0).sum())
        data = data[data["iv"] <= 5.0]
        messages.append(f"Dropped {n_bad} quote(s) with implied vol above 500%.")
    if data.empty:
        raise QuoteParseError("no valid rows remain after cleaning")

    data = _merge_close_maturities(data, messages)
    data = data.assign(maturity=data["maturity"].round(10))
    if "price_error" not in data:  # implied vols: the price behind each one, and so its rounding, is unknown
        data = data.assign(price_error=np.nan)
    duplicated = data.duplicated(["maturity", "strike"], keep=False)
    if duplicated.any():
        messages.append(f"Averaged {int(duplicated.sum())} duplicate (maturity, strike) quotes.")
        data = data.groupby(["maturity", "strike"], as_index=False)[["iv", "price_error"]].mean()
    columns = ["strike", "maturity", "iv", "price_error"]
    data = data.sort_values(["maturity", "strike"]).reset_index(drop=True)[columns]
    if guess is not None and guess.suspect:  # years was assumed for values that could well be days
        warning = f"Maturities in '{guess.column}' {guess.described}, read as years; {_NAME_THE_UNIT}."
        messages.append(QuoteWarning(warning))
    return data, messages


# ---------------------------------------------------------------------------
# Market quotes: fitting
# ---------------------------------------------------------------------------

MIN_STRIKES_PER_SLICE = 5
MAX_POINTS_PER_SLICE = 200  # denser slices are averaged into this many bins in y before the SVI fit
EXTRAPOLATION = 0.25  # a neighbour may be evaluated this fraction of its quoted y-range beyond its quotes
POOR_FIT_RMSE = 0.01
MIN_GRID_POINTS_PER_SLICE = 16  # the K/S grid is refined until every maturity has about this many points in its quotes
MAX_GRID_POINTS = 301
FAR_FROM_SPOT = 0.2  # |ln(K/S)| of the nearest quoted strike beyond which the spot looks wrong (about 22%)


def _binned(y: np.ndarray, iv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(y.min(), y.max(), MAX_POINTS_PER_SLICE + 1)
    which = np.clip(np.searchsorted(edges, y, side="right") - 1, 0, MAX_POINTS_PER_SLICE - 1)
    counts = np.bincount(which, minlength=MAX_POINTS_PER_SLICE)
    keep = counts > 0
    return (
        np.bincount(which, y, MAX_POINTS_PER_SLICE)[keep] / counts[keep],
        np.bincount(which, iv, MAX_POINTS_PER_SLICE)[keep] / counts[keep],
    )


def _moneyness_grid(ranges: np.ndarray, n_base: int) -> np.ndarray:
    """K/S grid over the union of the quoted ranges (rows of ``ranges``).

    ``n_base`` points span the union, and the spacing is refined inside narrower
    ranges so that each holds about :data:`MIN_GRID_POINTS_PER_SLICE` points (a
    short expiry quoted at 0.9-1.1 next to a LEAPS quoted at 0.3-2.5 is not
    reduced to four columns). At most :data:`MAX_GRID_POINTS` points.
    """
    lo, hi = ranges[:, 0].min(), ranges[:, 1].max()
    ref = np.linspace(lo, hi, 4001)
    mid = 0.5 * (ref[1:] + ref[:-1])
    width = np.maximum(ranges[:, 1] - ranges[:, 0], 1e-12)
    covers = (mid[None, :] >= ranges[:, :1]) & (mid[None, :] <= ranges[:, 1:])
    needed = np.where(covers, (MIN_GRID_POINTS_PER_SLICE - 1) / width[:, None], 0.0).max(axis=0)
    density = np.maximum((n_base - 1) / (hi - lo), needed)  # grid points per unit of K/S
    cumulative = np.concatenate([[0.0], np.cumsum(density * np.diff(ref))])
    n = min(int(np.ceil(cumulative[-1] - 1e-6)), MAX_GRID_POINTS - 1) + 1
    grid = np.interp(np.linspace(0.0, cumulative[-1], n), cumulative, ref)
    grid[0], grid[-1] = lo, hi
    return grid


def _time_derivative(w, cover, tol, Ts):
    """``dw/dT`` at fixed ``y`` from secants between neighbouring maturities, and the calendar-arbitrage mask.

    ``w[k]``, ``cover[k]`` and ``tol[k]`` (``k`` from -1 to 2) hold, for every row ``i``, slice ``i + k``
    evaluated at row ``i``'s ``y``, whether that is close enough to slice ``i + k``'s quotes, and the
    total-variance precision of its quotes (NaN / False where slice ``i + k`` does not exist).

    A secant is only used where its change in total variance is larger than the two slices' quote precision:
    one that falls by more is calendar arbitrage, and one that changes by less (two expiries a day apart a year
    out, say) does not resolve the slope or even its sign. Inside: the Fritsch-Butland weighted harmonic mean of
    the two secants, or the one that is resolved. First maturity: the one-sided three-point formula (second
    order, capped at three times the secant), or the one-sided secant where that is not positive or the third
    maturity is not quoted near this ``y``; at short maturities the secant's truncation error dominates. Last
    maturity: the one-sided secant, because at long maturities quote noise in total variance (which grows with
    ``T``) dominates and the three-point formula amplifies it. ``w_t`` is 0 where no secant is resolved.
    Returns ``(w_t, calendar_bad, available)``, each ``(n_T, n_y)``.
    """
    h = np.diff(Ts)
    h_l = np.r_[np.nan, h][:, None]
    h_r = np.r_[h, np.nan][:, None]
    diff_l, diff_r = w[0] - w[-1], w[1] - w[0]
    precision_l, precision_r = tol[-1] + tol[0], tol[0] + tol[1]
    negative = (cover[-1] & (diff_l < -precision_l)) | (cover[1] & (diff_r < -precision_r))
    with np.errstate(divide="ignore", invalid="ignore"):
        d_l = np.where(diff_l > precision_l, diff_l, 0.0) / h_l  # 0 unless the secant rises measurably
        d_r = np.where(diff_r > precision_r, diff_r, 0.0) / h_r
        harmonic = 3 * (h_l + h_r) / ((2 * h_r + h_l) / d_l + (h_r + 2 * h_l) / d_r)
    w_t = np.where((d_l > 0) & (d_r > 0), harmonic, np.maximum(d_l, d_r))
    available = cover[-1] & cover[1]
    w_t[0], available[0] = d_r[0], cover[1][0]
    w_t[-1], available[-1] = d_l[-1], cover[-1][-1]
    if len(Ts) >= 3:
        near = d_r[0]
        rise = w[2][0] - w[1][0]  # secant T1-T2 at row 0's y
        far = np.where(rise > tol[1][0] + tol[2][0], rise, 0.0) / h[1]
        three_point = ((2 * h[0] + h[1]) * near - h[0] * far) / (h[0] + h[1])
        use = cover[2][0] & (near > 0) & (far > 0) & (three_point > 0)
        w_t[0] = np.where(use, np.minimum(three_point, 3 * near), near)
    return w_t, negative, available


@dataclass(frozen=True)
class _MarketValues:
    implied: np.ndarray
    local: np.ndarray
    calendar_bad: np.ndarray
    butterfly_bad: np.ndarray
    defined: np.ndarray  # inside the quotes with positive total variance
    available: np.ndarray  # dw/dT has its neighbours close to their quotes
    unresolved: np.ndarray  # no secant rises by more than the quote precision, and none falls by more
    zero_variance: bool


def _market_values(fits, Ts, quoted_y, carry, moneyness) -> _MarketValues:
    """Implied and local vol of the SVI slices at K/S ``moneyness`` (1-D) on every quoted maturity (rows)."""
    n_T = len(Ts)
    y = np.log(np.asarray(moneyness, dtype=float))[None, :] - carry * Ts[:, None]
    lo, hi = quoted_y[:, 0], quoted_y[:, 1]
    allow = EXTRAPOLATION * (hi - lo)
    w, cover, tol = {}, {}, {}
    for offset in range(-1, 3):
        w[offset] = np.full(y.shape, np.nan)
        cover[offset] = np.zeros(y.shape, dtype=bool)
        T_k = np.full((n_T, 1), np.nan)
        for i in range(max(0, -offset), min(n_T, n_T - offset)):
            k = i + offset
            w[offset][i] = fits[k].total_variance(y[i])[0]
            cover[offset][i] = (y[i] >= lo[k] - allow[k]) & (y[i] <= hi[k] + allow[k])
            T_k[i] = Ts[k]
        tol[offset] = 2.0 * np.sqrt(np.maximum(w[offset], 0.0) * T_k) * QUOTE_PRECISION

    own = w[0]
    inside = (y >= lo[:, None] - 1e-9) & (y <= hi[:, None] + 1e-9)
    defined = inside & (own > 0)
    implied = np.where(defined, np.sqrt(np.maximum(own, 0.0) / Ts[:, None]), np.nan)
    w_y = np.array([svi.total_variance(y[i])[1] for i, svi in enumerate(fits)])
    w_yy = np.array([svi.total_variance(y[i])[2] for i, svi in enumerate(fits)])
    w_t, calendar_bad, available = _time_derivative(w, cover, tol, Ts)
    g = durrleman_g(y, np.where(defined, own, 1.0), w_y, w_yy)
    calendar_bad &= defined
    butterfly_bad = defined & (g <= 0)
    unresolved = defined & available & ~calendar_bad & ~(w_t > 0)
    ok = defined & available & ~calendar_bad & ~butterfly_bad & ~unresolved
    with np.errstate(divide="ignore", invalid="ignore"):
        local = np.where(ok, np.sqrt(np.where(ok, w_t / g, 0.0)), np.nan)
    return _MarketValues(
        implied,
        local,
        calendar_bad,
        butterfly_bad,
        defined,
        available,
        unresolved,
        bool((inside & (own <= 0)).any()),
    )


def fit_market_surface(
    quotes: pd.DataFrame,
    spot: float,
    rate: float,
    dividend_yield: float,
    n_moneyness: int = 41,
) -> VolSurface:
    """Fit a surface to cleaned quotes (see :func:`parse_quotes`) and compute local vol.

    Every maturity with enough strikes is fitted with raw SVI (:func:`fit_svi`).
    The result is evaluated at the quoted maturities (no extrapolation in time)
    on a K/S grid spanning the union of the quoted strike ranges, with
    ``n_moneyness`` points across it and more inside narrow ranges
    (:func:`_moneyness_grid`). Implied vol is NaN outside each maturity's own
    quoted range; local vol is also NaN where ``dw/dT`` would need a neighbouring
    maturity far outside its quotes, where the surface has calendar or butterfly
    arbitrage, or where it is otherwise undefined. The quotes themselves are checked
    for static arbitrage, allowing each price its ``price_error`` (see
    :func:`parse_quotes`; :data:`UNKNOWN_PRICE_ERROR` of the forward where it is NaN
    or the column is missing).
    """
    notes: list[str] = []
    carry = rate - dividend_yield
    Ts, fits, quoted, rmse, used, binned, arbitraged, flies = [], [], [], [], [], [], [], []
    for T, grp in quotes.groupby("maturity"):
        T = float(T)
        if grp["strike"].nunique() < MIN_STRIKES_PER_SLICE:
            notes.append(f"Skipped maturity {T:.4g}y: fewer than {MIN_STRIKES_PER_SLICE} strikes.")
            continue
        y = np.log(grp["strike"].to_numpy(float) / spot) - carry * T
        iv = grp["iv"].to_numpy(float)
        y_fit, iv_fit = (y, iv) if len(y) <= MAX_POINTS_PER_SLICE else _binned(y, iv)
        if len(y_fit) < len(y):
            binned.append(T)
        svi = fit_svi(y_fit, iv_fit, T)
        fitted = np.sqrt(np.maximum(svi.total_variance(y)[0], 0.0) / T)
        Ts.append(T)
        fits.append(svi)
        quoted.append((y.min(), y.max()))
        rmse.append(float(np.sqrt(np.mean((fitted - iv) ** 2))))
        forward = spot * np.exp(carry * T)
        # Each quote's price error, undiscounted like the Black prices it widens: its rounding or half its bid/ask
        # spread (see parse_quotes), or UNKNOWN_PRICE_ERROR of the forward for an implied vol without a price.
        known = grp["price_error"].to_numpy(float) if "price_error" in grp else np.full(len(grp), np.nan)
        price_error = np.where(np.isfinite(known), known * np.exp(rate * T), UNKNOWN_PRICE_ERROR * forward)
        arbitraged.append(_spread_arbitrage(grp["strike"].to_numpy(float), iv, T, forward, price_error))
        flies.append(_butterfly_arbitrage(grp["strike"].to_numpy(float), iv, T, forward, price_error))
        used.append(grp.assign(moneyness=grp["strike"] / spot, fitted_iv=fitted))
    if len(fits) < 2:
        n_maturities = quotes["maturity"].nunique()
        raise QuoteParseError(
            f"local volatility needs at least two maturities with {MIN_STRIKES_PER_SLICE}+ strikes each; the quotes "
            f"have {n_maturities} maturit{'y' if n_maturities == 1 else 'ies'}, {len(fits)} of them with "
            f"{MIN_STRIKES_PER_SLICE}+ strikes"
        )
    if binned:
        notes.append(
            f"Averaged dense quotes into at most {MAX_POINTS_PER_SLICE} points per maturity before fitting "
            f"({len(binned)} maturit{'y' if len(binned) == 1 else 'ies'})."
        )
    Ts = np.array(Ts)
    quoted_y = np.array(quoted)
    rmse = np.array(rmse)
    strikes = pd.concat(used)["strike"].to_numpy(float)
    log_moneyness = np.log(strikes / spot)
    nearest = int(np.argmin(np.abs(log_moneyness)))
    if abs(log_moneyness[nearest]) > FAR_FROM_SPOT or not 0.5 <= np.exp(np.median(log_moneyness)) <= 2.0:
        gap = strikes[nearest] / spot - 1.0
        notes.append(
            f"Quoted strikes run from {strikes.min():,.2f} to {strikes.max():,.2f}, far from the spot {spot:,.2f} "
            f"used to read them (the nearest is {abs(gap):.0%} {'below' if gap < 0 else 'above'} it); set the "
            "sidebar spot to the underlying price the quotes were taken at."
        )
    poor = rmse > POOR_FIT_RMSE
    crossed = poor & np.array(arbitraged)
    for which, cause in (
        (
            crossed,
            ", where no arbitrage-free smile can follow them: at those vols a put costs less than one at a lower "
            "strike, or a call more than one at a lower strike. Check those quotes and the maturity unit (days to "
            "expiry read as years give smiles this steep).",
        ),
        (
            poor & ~crossed,
            ": the quotes there are noisy, or their smile has a shape a single SVI curve cannot follow (an SVI "
            "smile has one trough and no humps).",
        ),
    ):
        if which.any():
            notes.append(
                f"The SVI smile is more than {POOR_FIT_RMSE * 100:.0f} vol point (RMSE) from the quotes at "
                f"maturities {', '.join(f'{t:.3g}y' for t in Ts[which])}{cause} The Smiles chart shows how far "
                "the fit is from each quote."
            )
    # Static arbitrage in the quotes themselves: the arbitrage-free SVI fit smooths it out of the surface, so the
    # surface's arbitrage counts cannot show it. Maturities already named in the note above are left out.
    fly = np.array([count > 0 for count, _ in flies]) & ~crossed
    spread = np.array(arbitraged) & ~crossed
    if fly.any() or spread.any():
        parts = []
        if fly.any():
            first = int(np.argmax(fly))
            k1, k2, k3 = (_strike_text(k) for k in flies[first][1])
            parts.append(
                f"at maturities {', '.join(f'{t:.3g}y' for t in Ts[fly])} a butterfly of neighbouring strikes costs "
                f"less than nothing (a negative risk-neutral density) even if each quote is {FLY_CHECK_VOL * 100:g} "
                f"vol point off, e.g. {k1}/{k2}/{k3} at {Ts[first]:.3g}y"
            )
        if spread.any():
            parts.append(
                f"at maturities {', '.join(f'{t:.3g}y' for t in Ts[spread])} a put costs less than one at a lower "
                "strike, or a call more than one at a lower strike, even if each quote is "
                f"{SPREAD_CHECK_VOL * 100:g} vol point off"
            )
        notes.append(
            f"The quotes have static arbitrage that the fitted surface smooths out: {'; '.join(parts)}. The checks "
            "also allow each price to be off by half its bid/ask spread, or by one step of the file's price grid "
            f"({UNKNOWN_PRICE_ERROR:.2%} of the forward for a quoted implied vol). The arbitrage counts above "
            "describe the fitted surface, not the quotes, so check these quotes."
        )

    moneyness = _moneyness_grid(np.exp(quoted_y + carry * Ts[:, None]), n_moneyness)
    values = _market_values(fits, Ts, quoted_y, carry, moneyness)
    uncovered = int((values.defined & ~values.available).sum())
    if uncovered:
        notes.append(
            f"Local vol is left blank at {uncovered} grid point(s) where dw/dT would need a neighbouring maturity "
            "far outside its quoted strikes."
        )
    unresolved = int(values.unresolved.sum())
    if unresolved:
        notes.append(
            f"Local vol is left blank at {unresolved} grid point(s) where total variance changes by less than the "
            "quotes' precision between neighbouring maturities, so dw/dT cannot be measured there (this is not "
            "calendar arbitrage)."
        )
    if values.zero_variance:
        notes.append("The fitted surface has zero total variance at some quoted points; check the quotes.")

    clean_quotes = pd.concat(used).reset_index(drop=True)[["strike", "maturity", "iv", "moneyness", "fitted_iv"]]
    return VolSurface(
        moneyness,
        Ts,
        values.implied,
        values.local,
        int(values.calendar_bad.sum()),
        int(values.butterfly_bad.sum()),
        "Market quotes",
        tuple(notes),
        carry=carry,
        quotes=clean_quotes,
        fit_rmse=rmse,
        svi=tuple(fits),
        quoted_y=quoted_y,
    )


def sample_quotes(
    spot: float = 100.0, rate: float = 0.04, dividend_yield: float = 0.0, seed: int = 0, prices: bool = False
) -> pd.DataFrame:
    """A realistic example quotes table (SSVI plus small noise) for the CSV template.

    With ``prices=True`` the table holds out-of-the-money call and put prices
    (Black-Scholes-Merton at the noisy vols) and a Type column instead of implied vols.
    """
    rng = np.random.default_rng(seed)
    params = SSVIParams()
    rows = []
    for days in (30, 60, 91, 182, 273, 365, 547, 730):
        T = days / 365.0
        for m in np.arange(0.70, 1.301, 0.05):
            y = np.log(m) - (rate - dividend_yield) * T
            w = params.total_variance(y, T)[0]
            iv = np.sqrt(w / T) + rng.normal(0.0, 0.002)
            strike = round(spot * m, 2)
            if prices:
                kind = "call" if y >= 0 else "put"
                value = bs.price(spot, strike, T, rate, iv, dividend_yield, kind)
                rows.append({"Strike": strike, "Days": days, "Type": kind, "Price": round(float(value), 4)})
            else:
                rows.append({"Strike": strike, "Maturity": round(T, 4), "ImpliedVol": round(float(iv), 4)})
    return pd.DataFrame(rows)


def surface_summary(surface: VolSurface) -> pd.DataFrame:
    """ATM level, 90/110 skew and smile curvature per maturity (vol points), plus the fit RMSE for quotes.

    Evaluated exactly at K/S = 0.9, 1 and 1.1 (:meth:`VolSurface.at_moneyness`),
    not interpolated on the grid; NaN where the surface is undefined at the
    strikes a value needs (outside a maturity's quotes, or no local vol there).
    """
    implied, local = surface.at_moneyness([0.9, 1.0, 1.1])
    rows = []
    for i, T in enumerate(surface.maturities):
        lo, atm, hi = implied[i]
        row = {
            "Maturity (years)": T,
            "ATM implied vol": atm,
            "ATM local vol": local[i, 1],
            "Skew 90-110 (vol pts)": (lo - hi) * 100,
            "Butterfly (vol pts)": (0.5 * (lo + hi) - atm) * 100,
        }
        if surface.fit_rmse is not None:
            row["Fit RMSE (vol pts)"] = surface.fit_rmse[i] * 100
        rows.append(row)
    return pd.DataFrame(rows)
