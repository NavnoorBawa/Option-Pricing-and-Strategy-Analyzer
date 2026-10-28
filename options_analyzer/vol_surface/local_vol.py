"""Local volatility from total implied variance: Gatheral's form of Dupire's formula.

``sigma_loc^2 = (dw/dT) / g(y, T)`` with Durrleman's butterfly function ``g`` (:func:`durrleman_g`); points with
calendar (``dw/dT < 0``) or butterfly (``g <= 0``) arbitrage have no local vol. For market quotes ``dw/dT`` at fixed
forward log-moneyness comes from secants between neighbouring SVI slices (:func:`_time_derivative`), and
:func:`_market_values` evaluates implied and local vol of the slices on a K/S grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .svi import QUOTE_PRECISION

EXTRAPOLATION = 0.25  # a neighbour may be evaluated this fraction of its quoted y-range beyond its quotes


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


def _market_values(fits, Ts, quoted_y, log_fs, moneyness) -> _MarketValues:
    """Implied and local vol of the SVI slices at K/S ``moneyness`` (1-D) on every quoted maturity (rows), where the
    forward of maturity ``i`` is ``S e^{log_fs[i]}``."""
    n_T = len(Ts)
    y = np.log(np.asarray(moneyness, dtype=float))[None, :] - np.asarray(log_fs, dtype=float)[:, None]
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
