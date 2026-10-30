"""Local volatility from total implied variance: Gatheral's form of Dupire's formula.

``sigma_loc^2 = (dw/dT) / g(y, T)`` with Durrleman's butterfly function ``g`` (:func:`durrleman_g`); points with
calendar (``dw/dT < 0``) or butterfly (``g <= 0``) arbitrage have no local vol. For market quotes ``dw/dT`` at fixed
forward log-moneyness comes from the SVI slices (:func:`_time_derivative`): the part that a smile keeping its shape in
``y / sqrt(theta)``, ``theta`` the ATM total variance, would have, read off the slice itself, plus the rest from
polynomials in ``T`` through neighbouring slices (:func:`_scaled_slope`); secants between neighbouring slices
(:func:`_secant_time_derivative`) decide calendar arbitrage and stand in where that cannot be used.
:func:`_market_values` evaluates implied and local vol of the slices on a K/S grid, and an estimate of the truncation
error of local vol (:func:`_truncation_error`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .svi import QUOTE_PRECISION, durrleman_g

EXTRAPOLATION = 0.25  # a neighbour may be evaluated this fraction of its quoted y-range beyond its quotes
SMILE_SCALING = 0.5  # the smile is taken to keep its shape in y / theta^0.5 (_time_derivative)
# The neighbours read at each row's y: one maturity back and two on for dw/dT (_stencil), two back and three on for
# its truncation error (_higher_stencils).
_NEAR_OFFSETS, _OFFSETS = range(-1, 3), range(-2, 4)


def _local_vol(y, w, w_y, w_yy, w_t):
    g = durrleman_g(y, w, w_y, w_yy)
    calendar_bad = w_t < 0
    butterfly_bad = g <= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        local_var = w_t / g
    ok = ~calendar_bad & ~butterfly_bad & np.isfinite(local_var) & (local_var >= 0)
    return np.where(ok, np.sqrt(np.where(ok, local_var, 0.0)), np.nan), calendar_bad, butterfly_bad


def _secant_time_derivative(w, cover, tol, Ts):
    """``dw/dT`` at fixed ``y`` from secants between neighbouring maturities, and the calendar-arbitrage mask.

    ``w[k]``, ``cover[k]`` and ``tol[k]`` (``k`` from -1 to 2 at least) hold, for every row ``i``, slice ``i + k``
    evaluated at row ``i``'s ``y``, whether that is close enough to slice ``i + k``'s quotes, and the
    total-variance precision of its quotes (NaN / False where slice ``i + k`` does not exist).

    A secant is only used where its change in total variance is larger than the two slices' quote precision:
    one that falls by more is calendar arbitrage, and one that changes by less (two expiries a day apart a year
    out, say) does not resolve the slope or even its sign. Inside: the Fritsch-Butland weighted harmonic mean of
    the two secants, or the one that is resolved. First maturity: the one-sided three-point formula (second
    order, capped at three times the secant), or the one-sided secant where that is not positive or the third
    maturity is not quoted near this ``y``. Last maturity: the one-sided secant. ``w_t`` is 0 where no secant is
    resolved. :func:`_time_derivative` replaces these values by more accurate ones where it can.
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


def _slope_weights(nodes, at: int) -> list[float]:
    """Weights ``c_j`` with ``p'(nodes[at]) = sum_j c_j p(nodes[j])`` for every polynomial ``p`` of degree below
    ``len(nodes)``: the derivatives of the Lagrange basis polynomials at ``nodes[at]`` (plain floats: two to four
    nodes, where NumPy's overhead is most of the time)."""
    x = [float(node) for node in nodes]
    weights = []
    for j, node in enumerate(x):
        if j == at:
            weights.append(sum(1.0 / (node - other) for m, other in enumerate(x) if m != at))
            continue
        weight = 1.0 / (node - x[at])
        for m, other in enumerate(x):
            if m not in (j, at):
                weight *= (x[at] - other) / (node - other)
        weights.append(weight)
    return weights


def _stencil(i: int, n: int) -> tuple[int, ...]:
    """The maturities (offsets from row ``i`` of ``n >= 2``) whose slices give row ``i``'s ``dw/dT``: three, one-sided,
    at the first maturity (two where there are only two), the row and its two neighbours inside, and the last two at
    the last maturity (see :func:`_time_derivative` for why)."""
    if i == 0:
        return (0, 1, 2) if n >= 3 else (0, 1)
    return (-1, 0) if i == n - 1 else (-1, 0, 1)


def _higher_stencils(i: int, n: int) -> list[tuple[int, ...]]:
    """:func:`_stencil` with one more maturity, one order higher: the next one out at the first and the last
    maturity, either neighbour's neighbour inside; those that exist."""
    near = _stencil(i, n)
    if i == 0:
        candidates = [(*near, near[-1] + 1)]
    elif i == n - 1:
        candidates = [(near[0] - 1, *near)]
    else:
        candidates = [(near[0] - 1, *near), (*near, near[-1] + 1)]
    return [offsets for offsets in candidates if i + offsets[0] >= 0 and i + offsets[-1] < n]


@dataclass(frozen=True, eq=False)  # holds arrays: compared by identity
class _Neighbours:
    """Everything :func:`_scaled_slope` reads: the slices and maturities; ``y`` (``(n_T, n_y)``); ``w``, ``cover``
    and ``tol`` by offset as in :func:`_secant_time_derivative`, and ``w_y``, each row's own ``dw/dy``; each slice's
    ATM total variance ``theta`` (at ``y = 0``), its precision, and whether ``y = 0`` is near the slice's quotes
    (``atm_covered``)."""

    fits: tuple
    Ts: np.ndarray
    y: np.ndarray
    w: dict
    cover: dict
    tol: dict
    w_y: np.ndarray
    theta: np.ndarray
    theta_tol: np.ndarray
    atm_covered: np.ndarray


def _scaled_slope(nb: _Neighbours, i: int, offsets: tuple[int, ...], *, limited: bool) -> np.ndarray:
    """``dw/dT`` at row ``i``'s ``y`` from the slices at ``offsets`` (:func:`_time_derivative`): NaN where they
    cannot give it.

    With ``theta_k`` each slice's ATM total variance and ``r_k = theta_k / theta_i``, a slice that kept its shape
    in ``y / sqrt(theta)`` would be ``r_k w_i(y / sqrt(r_k))`` at maturity ``k``, whose slope at ``T_i`` is
    ``theta'_i / theta_i (w_i - y w_i' / 2)``. The rest, ``w_k(y) - r_k w_i(y / sqrt(r_k))`` (0 at ``T_i``), is
    differentiated with the polynomial through it in ``T``, and ``theta'_i`` is the slope of the polynomial through
    ``theta`` in ``T``. With ``limited`` (the reported value) ``theta'_i`` keeps a monotone interpolant's limits,
    at most three times the smaller adjacent secant of ``theta`` and, where it is not positive (an end formula can
    be), that secant; and the result is NaN where it is not positive or more than three times the smaller adjacent
    secant of ``w`` at ``y`` (Hyman's limit, which keeps a cubic through the slices monotone). Either way it is NaN
    unless every slice of the stencil is quoted near ``y`` and near the forward (``y = 0``: the smile scales about
    it, and quotes far from it make ``theta`` an extrapolation), and both ``w`` at ``y`` and ``theta`` rise from
    each of its maturities to the next by more than the quotes' precision.
    """
    nodes = [i + k for k in offsets]
    at = offsets.index(0)
    theta, Ts = nb.theta[nodes], nb.Ts[nodes]
    nan = np.full(nb.y.shape[1], np.nan)
    rising = np.diff(theta) > nb.theta_tol[nodes][1:] + nb.theta_tol[nodes][:-1]
    if not (nb.atm_covered[nodes].all() and (theta > 0).all() and rising.all()):
        return nan
    log_slope = float(np.dot(_slope_weights(np.log(Ts), at), np.log(theta)))  # d ln(theta) / d ln(T)
    theta_t = log_slope * nb.theta[i] / nb.Ts[i]
    if limited:
        secants = np.diff(theta) / np.diff(Ts)
        adjacent = secants[max(at - 1, 0) : at + 1]
        if theta_t <= 0:  # only at an end: inside, the slope is a weighted mean of the two secants
            theta_t = float(adjacent[0])
        theta_t = min(theta_t, 3.0 * float(adjacent.min()))
    y = nb.y[i]
    value = theta_t / nb.theta[i] * (nb.w[0][i] - SMILE_SCALING * y * nb.w_y[i])
    for k, weight in zip(offsets, _slope_weights(Ts, at), strict=True):
        if k != 0:
            ratio = nb.theta[i + k] / nb.theta[i]
            value = value + weight * (nb.w[k][i] - ratio * nb.fits[i].total_variance(y / ratio**SMILE_SCALING)[0])
    rises = [nb.w[b][i] - nb.w[a][i] for a, b in zip(offsets[:-1], offsets[1:], strict=True)]
    usable = np.logical_and.reduce(
        [nb.cover[k][i] for k in offsets if k != 0]
        + [rise > nb.tol[a][i] + nb.tol[b][i] for (a, b), rise in zip(zip(offsets, offsets[1:]), rises, strict=True)]
    )
    if limited:
        slopes = [rise / (Ts[j + 1] - Ts[j]) for j, rise in enumerate(rises)]
        adjacent_slopes = slopes[max(at - 1, 0) : at + 1]
        usable &= (value > 0) & (value <= 3.0 * np.minimum.reduce(adjacent_slopes))
    return np.where(usable, value, nan)


def _time_derivative(nb: _Neighbours):
    """``dw/dT`` at fixed ``y`` for every quoted maturity (rows), and the calendar-arbitrage mask.

    Total variance at fixed ``y`` depends on ``T`` in two ways: through the ATM level ``theta(T)``, the integral of
    the forward ATM variance and a smooth function of ``T``, and through the smile, whose width in ``y`` grows
    roughly like ``sqrt(theta)`` (SSVI with ``gamma = 1/2``; the ATM skew of equity indices decays like ``T^-0.4``
    to ``T^-0.5``). Secants in ``T`` miss the second: near the money it adds a term in ``sqrt(theta)``, about
    ``sqrt(T)``, whose slope near ``T = 0`` no polynomial in ``T`` follows. So the part a smile that scales exactly
    would have is read off the slice itself, and only the rest (none for SSVI with ``gamma = 1/2`` while ``theta``
    is small, nor for a flat surface) is differentiated between slices (:func:`_scaled_slope`): one-sided through
    the first three maturities at the first, centred through the neighbours inside, and from the last two at the
    last (:func:`_stencil`). Chosen by measurement with noise-free SSVI quotes on 25 strikes from 70% to 130% of
    the spot (largest local-vol error in vol points, secants in ``T`` against this):

    * listed expiries 30, 60, 91, 182, 273, 365, 547 and 730 days: 2.12 against 0.27 (0.07 at 30 days);
    * weekly front 7, 14, 21, 28, 60, 91, 182 and 365 days: 3.37 against 1.15 (0.01 at 7 days; the 1.15 is at the
      year, after a half-year gap);
    * every two weeks from 14 days to a year: 2.70 against 0.07; quarterly to two years: 1.55 against 0.31;
    * 57 random arbitrage-free surfaces (ATM vols 10-60%, mean reversion 0.3-4, gamma 0.2-0.5) on 17 strikes,
      mean of the largest errors on the listed, weekly-front and quarterly expiries: 0.85, 2.85 and 0.75 against
      0.54, 2.42 and 0.55.

    A parabola in ``sqrt(T)`` at fixed ``y`` was as accurate on those grids (0.24 on the listed expiries) but 7.7
    vol points off at 14 days on an equity skew whose ATM vol rises from 16% to 47%, where this is 1.5 (secants:
    1.1): a power of ``T`` cannot follow both such a level and the smile's scaling. Polynomials in ``sqrt(theta)``
    for the rest were 15-20% more accurate inside and no noisier, at the first maturity 30% noisier; ``T`` is used
    throughout for simplicity.

    The ends are chosen for noise. With the 0.2 vol-point noise of the sample quotes (40 seeds) the RMS local-vol
    error is 0.94 vol point at 30 days, 0.57 inside and 0.85 at two years (secants: 1.28, 0.56 and 0.89); on 60
    random surfaces with that noise the median, 90th and 99th percentile errors are within a tenth of the secants'
    (first maturity 0.95%, 3.5% and 6.5% against 1.11%, 3.6% and 6.6%). The three-point formula at the last
    maturity would cut its clean error on the listed expiries from 0.27 to 0.09 vol point, but it raised the
    sample's RMS there to 1.07 and the random surfaces' 99th percentile to 45% (31% with secants): total variance
    grows with ``T``, and so does its noise.

    Calendar arbitrage, the 'unresolved' case and ``dw/dT > 0`` where the slices rise measurably are decided by the
    secant scheme (:func:`_secant_time_derivative`), as before: the scaled slope replaces its value only where that
    is positive, and :func:`_scaled_slope` is NaN wherever a secant of its stencil is calendar arbitrage or
    unresolved, or the slope is beyond the monotone limits. Returns ``(w_t, calendar_bad, available)``, each
    ``(n_T, n_y)``.
    """
    w_t, negative, available = _secant_time_derivative(nb.w, nb.cover, nb.tol, nb.Ts)
    n = len(nb.Ts)
    for i in range(n):
        scaled = _scaled_slope(nb, i, _stencil(i, n), limited=True)
        w_t[i] = np.where(np.isfinite(scaled) & (w_t[i] > 0), scaled, w_t[i])
    return w_t, negative, available


def _truncation_error(nb: _Neighbours, local: np.ndarray, g: np.ndarray) -> np.ndarray:
    """An estimate of the truncation error of ``local`` (the local vol of :func:`_time_derivative`) at every point:
    the largest change to it when ``dw/dT`` comes from one maturity more (:func:`_higher_stencils`, one order
    higher, without the monotone limits); NaN where no such stencil is usable or ``local`` is undefined. A slope
    that is not positive gives a local vol of 0, an error as large as the local vol itself."""
    error = np.full(local.shape, np.nan)
    n = len(nb.Ts)
    for i in range(n):
        for offsets in _higher_stencils(i, n):
            higher = _scaled_slope(nb, i, offsets, limited=False)
            change = np.abs(np.sqrt(np.maximum(higher, 0.0) / g[i]) - local[i])  # NaN where undefined: fmax skips it
            error[i] = np.fmax(error[i], change)
    return error


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
    local_error: np.ndarray | None = None  # estimated truncation error of local (_truncation_error), if asked for


def _market_values(fits, Ts, quoted_y, log_fs, moneyness, *, truncation_error: bool = False) -> _MarketValues:
    """Implied and local vol of the SVI slices at K/S ``moneyness`` (1-D) on every quoted maturity (rows), where the
    forward of maturity ``i`` is ``S e^{log_fs[i]}``; with ``truncation_error``, also the estimated truncation
    error of each local vol (:func:`_truncation_error`)."""
    n_T = len(Ts)
    y = np.log(np.asarray(moneyness, dtype=float))[None, :] - np.asarray(log_fs, dtype=float)[:, None]
    lo, hi = quoted_y[:, 0], quoted_y[:, 1]
    allow = EXTRAPOLATION * (hi - lo)
    w, cover, tol = {}, {}, {}
    for offset in _OFFSETS if truncation_error else _NEAR_OFFSETS:
        w[offset] = np.full(y.shape, np.nan)
        cover[offset] = np.zeros(y.shape, dtype=bool)
        T_k = np.full((n_T, 1), np.nan)
        for i in range(max(0, -offset), min(n_T, n_T - offset)):
            k = i + offset
            w[offset][i] = fits[k].total_variance(y[i])[0]
            cover[offset][i] = (y[i] >= lo[k] - allow[k]) & (y[i] <= hi[k] + allow[k])
            T_k[i] = Ts[k]
        tol[offset] = 2.0 * np.sqrt(np.maximum(w[offset], 0.0) * T_k) * QUOTE_PRECISION
    own, w_y, w_yy = np.array([fit.total_variance(row) for fit, row in zip(fits, y, strict=True)]).transpose(1, 0, 2)
    theta = np.array([float(fit.total_variance(0.0)[0]) for fit in fits])
    neighbours = _Neighbours(
        tuple(fits),
        np.asarray(Ts, dtype=float),
        y,
        w,
        cover,
        tol,
        w_y,
        theta,
        2.0 * np.sqrt(np.maximum(theta, 0.0) * Ts) * QUOTE_PRECISION,
        (lo - allow <= 0.0) & (hi + allow >= 0.0),
    )

    inside = (y >= lo[:, None] - 1e-9) & (y <= hi[:, None] + 1e-9)
    defined = inside & (own > 0)
    implied = np.where(defined, np.sqrt(np.maximum(own, 0.0) / Ts[:, None]), np.nan)
    w_t, calendar_bad, available = _time_derivative(neighbours)
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
        _truncation_error(neighbours, local, np.where(ok, g, np.nan)) if truncation_error else None,
    )
