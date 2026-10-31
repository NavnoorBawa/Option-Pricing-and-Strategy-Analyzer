"""Local volatility from total implied variance: Gatheral's form of Dupire's formula.

``sigma_loc^2 = (dw/dT) / g(y, T)`` with Durrleman's butterfly function ``g`` (:func:`durrleman_g`); points with
calendar (``dw/dT < 0``) or butterfly (``g <= 0``) arbitrage have no local vol. For market quotes ``dw/dT`` at fixed
forward log-moneyness comes from the SVI slices (:func:`_time_derivative`): the part that a smile keeping its shape in
``y / sqrt(theta)``, ``theta`` the ATM total variance, would have, read off the slice itself, plus the rest from
polynomials in ``T`` through neighbouring slices (:func:`_scaled_slope`); secants between neighbouring slices
(:func:`_secant_time_derivative`) decide calendar arbitrage and stand in where that cannot be used.
:func:`_market_values` evaluates implied and local vol of the slices on a K/S grid, and, if asked, an estimate of the
truncation error of each local vol: how much it changes when ``dw/dT`` comes from one slice more.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from .svi import QUOTE_PRECISION, durrleman_g

EXTRAPOLATION = 0.25  # a neighbour may be evaluated this fraction of its quoted y-range beyond its quotes
SMILE_SCALING = 0.5  # the smile is taken to keep its shape in y / theta^0.5 (_time_derivative)
#: Hyman's limit (Hyman 1983, SIAM J. Sci. Stat. Comput. 4(4)): a slope at a maturity of at most this many times the
#: smaller resolved secant next to it keeps a cubic through the slices monotone in T; a scaled slope beyond it is not
#: used (_time_derivative).
MONOTONE_LIMIT = 3.0
# The neighbours read at each row's y: one maturity back and two on for dw/dT (_stencils), two back and three on
# for its truncation error as well (_extended).
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


def _stencils(i: int, n: int) -> list[tuple[int, ...]]:
    """The maturities (offsets from row ``i`` of ``n >= 2``) whose slices may give row ``i``'s ``dw/dT``, in order
    of preference (:func:`_time_derivative` says why): three, one-sided, at the first maturity, then the first two;
    the row and its two neighbours inside, then the row and the one before, then the row and the one after; the last
    two at the last maturity."""
    if i == 0:
        return [(0, 1, 2), (0, 1)] if n >= 3 else [(0, 1)]
    return [(-1, 0)] if i == n - 1 else [(-1, 0, 1), (-1, 0), (0, 1)]


def _extended(offsets: tuple[int, ...], i: int, n: int) -> list[tuple[tuple[int, ...], tuple[int, ...] | None]]:
    """The stencils one order higher than ``offsets`` at row ``i`` of ``n``, as ``(offsets, theta_offsets)`` for
    :func:`_scaled_slope`: one maturity more on the far side of a one-sided stencil, on either side of a centred
    one. The slice before and the row, ``(-1, 0)``, take ``theta'`` from three slices where they can
    (:func:`_end_theta_slope`), so one slice more is also the last three with ``theta'`` from four. Those that
    exist."""
    candidates: list[tuple[tuple[int, ...], tuple[int, ...] | None]] = []
    if offsets[-1] > 0:
        candidates.append(((*offsets, offsets[-1] + 1), None))
    if offsets[0] < 0:
        candidates.append(((offsets[0] - 1, *offsets), None))
    if offsets == (-1, 0):
        candidates.append(((-2, -1, 0), (-3, -2, -1, 0)))
    return [(s, t) for s, t in candidates if i + (t or s)[0] >= 0 and i + s[-1] < n]


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


def _end_theta_slope(nb: _Neighbours, i: int, log_secant: float, secant: float) -> float:
    """``theta'`` at maturity ``i`` from it and the one before (the last maturity, or a row whose next slice cannot
    be used), given the slopes the two slices give: ``log_secant`` with ``theta`` a power of ``T`` between them,
    ``secant`` with it linear in ``T`` (the mean forward variance over the gap).

    Where the forward variance ``theta'`` settles monotonically towards its long-run level, as it does at the long end,
    the two bracket it: rising, the secant (a mean over the gap) is below it and the power law, whose elasticity
    ``T theta' / theta`` falls back towards 1, above it; falling, the other way round. Within that bracket the
    parabola through ``ln theta`` against ``ln T`` at ``i - 2``, ``i - 1`` and ``i`` picks the slope, second order
    where the term structure is smooth on the spacing of the expiries; outside it, that parabola is not trusted (an
    ATM vol falling from 80% to 20% within months, sampled a year apart, puts it 29% below the true ``theta'``) and
    the nearer end is used. Without a usable slice ``i - 2`` (quoted near the forward, ``theta`` rising from it to
    ``i - 1`` by more than the quotes' precision) it is ``log_secant``.
    """
    if i < 2 or not (nb.atm_covered[i - 2] and nb.theta[i - 2] > 0):
        return log_secant
    if not nb.theta[i - 1] - nb.theta[i - 2] > nb.theta_tol[i - 2] + nb.theta_tol[i - 1]:
        return log_secant
    nodes = [i - 2, i - 1, i]
    log_slope = float(np.dot(_slope_weights(np.log(nb.Ts[nodes]), 2), np.log(nb.theta[nodes])))
    parabola = log_slope * nb.theta[i] / nb.Ts[i]
    return min(max(parabola, min(log_secant, secant)), max(log_secant, secant))


def _scaled_slope(
    nb: _Neighbours,
    i: int,
    offsets: tuple[int, ...],
    limit: np.ndarray | None = None,
    theta_offsets: tuple[int, ...] | None = None,
) -> np.ndarray:
    """``dw/dT`` at row ``i``'s ``y`` from the slices at ``offsets`` (:func:`_time_derivative`): NaN where they
    cannot give it.

    With ``theta_k`` each slice's ATM total variance and ``r_k = theta_k / theta_i``, a slice that kept its shape
    in ``y / sqrt(theta)`` would be ``r_k w_i(y / sqrt(r_k))`` at maturity ``k``, whose slope at ``T_i`` is
    ``theta'_i / theta_i (w_i - y w_i' / 2)``. The rest, ``w_k(y) - r_k w_i(y / sqrt(r_k))`` (0 at ``T_i``), is
    differentiated with the polynomial through it in ``T``, and ``theta'_i`` with the polynomial through
    ``ln theta`` in ``ln T`` (``theta`` is close to proportional to ``T``) at ``theta_offsets`` (``offsets`` if
    None). With a ``limit`` (the reported value) ``theta'_i`` keeps a monotone interpolant's limits, at most
    :data:`MONOTONE_LIMIT` times the smaller adjacent secant of ``theta`` and, where it is not positive (an end
    formula can be), that secant; from the slice before and the row it comes from three slices where it can
    (:func:`_end_theta_slope`); and the result is NaN where it is not positive or above ``limit``. Either way it is
    NaN unless every slice of both stencils is quoted near ``y`` and near the forward (``y = 0``: the smile scales
    about it, and quotes far from it make ``theta`` an extrapolation), and both ``w`` at ``y`` and ``theta`` rise
    from each of its maturities to the next by more than the quotes' precision.
    """
    nodes = [i + k for k in offsets]
    theta_nodes = nodes if theta_offsets is None else [i + k for k in theta_offsets]
    every = sorted({*nodes, *theta_nodes})
    at = offsets.index(0)
    theta, Ts = nb.theta[nodes], nb.Ts[nodes]
    nan = np.full(nb.y.shape[1], np.nan)
    rising = np.diff(nb.theta[every]) > nb.theta_tol[every][1:] + nb.theta_tol[every][:-1]
    if not (nb.atm_covered[every].all() and (nb.theta[every] > 0).all() and rising.all()):
        return nan
    weights = _slope_weights(np.log(nb.Ts[theta_nodes]), theta_nodes.index(i))
    log_slope = float(np.dot(weights, np.log(nb.theta[theta_nodes])))  # d ln(theta) / d ln(T)
    theta_t = log_slope * nb.theta[i] / nb.Ts[i]
    if limit is not None:
        secants = np.diff(theta) / np.diff(Ts)
        adjacent = secants[max(at - 1, 0) : at + 1]
        if offsets == (-1, 0):
            theta_t = _end_theta_slope(nb, i, theta_t, float(adjacent[0]))
        if theta_t <= 0:  # only at an end: inside, the slope is a weighted mean of the two secants
            theta_t = float(adjacent[0])
        theta_t = min(theta_t, MONOTONE_LIMIT * float(adjacent.min()))
    y = nb.y[i]
    value = theta_t / nb.theta[i] * (nb.w[0][i] - SMILE_SCALING * y * nb.w_y[i])
    for k, weight in zip(offsets, _slope_weights(Ts, at), strict=True):
        if k != 0:
            ratio = nb.theta[i + k] / nb.theta[i]
            value = value + weight * (nb.w[k][i] - ratio * nb.fits[i].total_variance(y / ratio**SMILE_SCALING)[0])
    usable = np.logical_and.reduce(
        [nb.cover[k][i] for k in offsets if k != 0]
        + [nb.w[b][i] - nb.w[a][i] > nb.tol[a][i] + nb.tol[b][i] for a, b in pairwise(offsets)]
    )
    if limit is not None:
        usable &= (value > 0) & (value <= limit)
    return np.where(usable, value, nan)


def _monotone_limit(nb: _Neighbours, i: int) -> np.ndarray:
    """:data:`MONOTONE_LIMIT` times the smaller of row ``i``'s two secants of ``w`` in ``T`` at each ``y``, of those
    that rise by more than the quotes' precision to a neighbour quoted near ``y`` (infinite where neither does)."""
    limit = np.full(nb.y.shape[1], np.inf)
    for k in (-1, 1):
        j = i + k
        if 0 <= j < len(nb.Ts):
            rise = k * (nb.w[k][i] - nb.w[0][i])
            resolved = nb.cover[k][i] & (rise > nb.tol[k][i] + nb.tol[0][i])
            limit = np.where(resolved, np.minimum(limit, rise / abs(nb.Ts[j] - nb.Ts[i])), limit)
    return MONOTONE_LIMIT * limit


def _time_derivative(nb: _Neighbours, *, checks: bool = False):
    """``dw/dT`` at fixed ``y`` for every quoted maturity (rows), and the calendar-arbitrage mask.

    Total variance at fixed ``y`` depends on ``T`` in two ways: through the ATM level ``theta(T)``, the integral of
    the forward ATM variance and a smooth function of ``T``, and through the smile, whose width in ``y`` grows
    roughly like ``sqrt(theta)`` (SSVI with ``gamma = 1/2``; the ATM skew of equity indices decays like ``T^-0.4``
    to ``T^-0.5``). Secants in ``T`` miss the second: near the money it adds a term in ``sqrt(theta)``, about
    ``sqrt(T)``, whose slope near ``T = 0`` no polynomial in ``T`` follows. So the part a smile that scales exactly
    would have is read off the slice itself, and only the rest (none for SSVI with ``gamma = 1/2`` while ``theta``
    is small, nor for a flat surface) is differentiated between slices (:func:`_scaled_slope`): one-sided through
    the first three maturities at the first, centred through the neighbours inside, and from the last two at the
    last (:func:`_stencils`). Chosen by measurement with noise-free SSVI quotes on 25 strikes from 70% to 130% of
    the spot (largest local-vol error in vol points; secants in ``T`` against this):

    * listed expiries 30, 60, 91, 182, 273, 365, 547 and 730 days: 2.12 against 0.095 (the median at 30 days 0.45
      against 0.038; at the last expiry 0.61 against 0.056);
    * weekly front 7, 14, 21, 28, 60, 91, 182 and 365 days: 3.37 against 0.45 (0.045 at 7 days; the 0.45 is at the
      year, after a half-year gap);
    * every two weeks from 14 to 364 days: 2.70 against 0.069; quarterly to two years: 1.56 against 0.030;
    * random arbitrage-free surfaces (ATM vols 10-60%, mean reversion 0.3-4, gamma 0.2-0.5; 32 to 34 per grid, on 17
      strikes from 80% to 120%), mean of the largest errors on the listed, weekly-front and quarterly expiries: 0.95,
      2.38 and 1.04 against 0.42, 0.87 and 0.26.

    The alternatives, on the listed and the weekly-front expiries and the random surfaces on the listed ones:
    ``theta'`` from the parabola through ``theta`` in ``T`` instead of ``ln theta`` in ``ln T``, 0.27, 1.15 and 0.68;
    the rest differentiated in ``sqrt(theta)`` instead of ``T``, 0.13, 0.50 and 0.43; ``w`` interpolated in ``T`` at
    fixed standardised moneyness ``y / sqrt(theta)`` and converted back with the chain rule, 0.43, 1.51 and 0.77; a
    cubic in ``T`` at fixed ``y`` through one more neighbour (Hyman-limited), 1.46, 2.10 and 0.52; a parabola in
    ``sqrt(T)`` at fixed ``y``, 0.58, 1.18 and 1.16, and 7.9 vol points at 14 days on an equity skew whose ATM vol
    rises from 16% to 47% (``mean_reversion`` 4), where this is 0.63 and secants 2.15.

    The ends are chosen for noise. With the 0.2 vol-point noise of the sample quotes (40 seeds) the RMS local-vol
    error is 0.93 vol point at 30 days, 0.58 inside and 0.87 at two years (secants: 1.28, 0.56 and 0.89); on that
    equity skew with the same noise (30 seeds), 0.27 to 1.13 by expiry (secants: 0.42 to 2.50). The three-slice
    one-sided formula at the last maturity would cut its clean error on the weekly-front expiries from 0.45 to 0.13
    vol point, but it raised the RMS there to 1.22 on the sample quotes and to 1.77 on the skew, and the clean error
    on an inverted term structure (ATM vol from 80% to 20%, expiries at 30, 91, 365 and 730 days) from 0.74 to 3.9
    vol points: total variance grows with ``T``, and so does its noise.

    Calendar arbitrage, the 'unresolved' case and ``dw/dT > 0`` where the slices rise measurably are decided by the
    secant scheme (:func:`_secant_time_derivative`), as before: a scaled slope replaces its value only where that is
    positive, and :func:`_scaled_slope` is NaN wherever a secant of its stencil is calendar arbitrage or unresolved,
    a slice of it is not quoted near ``y`` or the forward, or the slope is beyond Hyman's monotone limit
    (:func:`_monotone_limit`). Where a row's three-slice stencil is NaN for those reasons the two-slice ones are
    tried (the slice before first, as at the last maturity), and only then are the secants left: with SSVI quotes
    rounded to 0.01 vol point at 30, 60, 91, 182, 365, 366 and 730 days, whose 365-366 day secant is within the
    quotes' precision, the 365 and 366-day rows are 0.38 vol point off with the two-slice scaled slopes and 1.6 with
    the secants. (Local vol still needs both neighbours of an inside maturity quoted near ``y``, and the next one
    of the first: ``available``, as before.)

    Returns ``(w_t, calendar_bad, available, higher)``, the first three ``(n_T, n_y)``. With ``checks``, ``higher``
    holds up to two ``(n_T, n_y)`` arrays of ``dw/dT`` from the stencils one order higher than the one used at each
    point (:func:`_extended`, without the monotone limits; at a point the secants decided, those of the row's first
    stencil), NaN where there is none; else it is empty.
    """
    w_t, negative, available = _secant_time_derivative(nb.w, nb.cover, nb.tol, nb.Ts)
    n = len(nb.Ts)
    higher = [np.full(w_t.shape, np.nan), np.full(w_t.shape, np.nan)] if checks else []
    for i in range(n):
        stencils = _stencils(i, n)
        # Where a secant rises measurably (one that falls beyond the precision is flagged anyway) and local vol can
        # be defined: elsewhere dw/dT is not used.
        open_ = (w_t[i] > 0) & available[i]
        limit = _monotone_limit(nb, i)
        for offsets in stencils:
            if not open_.any():  # each stencil costs about 0.1 ms: try the next only where the last one failed
                break
            scaled = _scaled_slope(nb, i, offsets, limit)
            use = open_ & np.isfinite(scaled)
            w_t[i] = np.where(use, scaled, w_t[i])
            if use.any():
                _higher_order(higher, nb, i, offsets, use)
            open_ &= ~use
        if open_.any():
            _higher_order(higher, nb, i, stencils[0], open_)  # where the secants decided
    return w_t, negative, available, higher


def _higher_order(higher: list, nb: _Neighbours, i: int, offsets: tuple[int, ...], where: np.ndarray) -> None:
    """Write ``dw/dT`` from the stencils one order higher than ``offsets`` (:func:`_extended`) into the arrays
    ``higher`` (none if empty) at row ``i``'s points ``where``, in turn (:func:`_time_derivative`)."""
    for check, (extended, theta_offsets) in zip(higher, _extended(offsets, i, len(nb.Ts)), strict=False):
        check[i] = np.where(where, _scaled_slope(nb, i, extended, theta_offsets=theta_offsets), check[i])


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
    local_error: np.ndarray | None = None  # estimated truncation error of local, if asked for (_market_values)


def _market_values(fits, Ts, quoted_y, log_fs, moneyness, *, truncation_error: bool = False) -> _MarketValues:
    """Implied and local vol of the SVI slices at K/S ``moneyness`` (1-D) on every quoted maturity (rows), where the
    forward of maturity ``i`` is ``S e^{log_fs[i]}``.

    With ``truncation_error``, also an estimate of the truncation error of each local vol, as a vol: the largest
    change to it when ``dw/dT`` comes from a stencil one maturity larger, one order higher (:func:`_time_derivative`),
    NaN where local vol is undefined or no such stencil can be used. A higher-order slope that is not positive gives
    a local vol of 0, an error as large as the local vol itself.
    """
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
    w_t, calendar_bad, available, higher = _time_derivative(neighbours, checks=truncation_error)
    g = durrleman_g(y, np.where(defined, own, 1.0), w_y, w_yy)
    calendar_bad &= defined
    butterfly_bad = defined & (g <= 0)
    unresolved = defined & available & ~calendar_bad & ~(w_t > 0)
    ok = defined & available & ~calendar_bad & ~butterfly_bad & ~unresolved
    with np.errstate(divide="ignore", invalid="ignore"):
        local = np.where(ok, np.sqrt(np.where(ok, w_t / g, 0.0)), np.nan)
    local_error = None
    if truncation_error:
        local_error = np.full(local.shape, np.nan)
        g_ok = np.where(ok, g, np.nan)  # NaN where local vol is undefined, so is its error
        for slope in higher:
            local_error = np.fmax(local_error, np.abs(np.sqrt(np.maximum(slope, 0.0) / g_ok) - local))
    return _MarketValues(
        implied,
        local,
        calendar_bad,
        butterfly_bad,
        defined,
        available,
        unresolved,
        bool((inside & (own <= 0)).any()),
        local_error,
    )
