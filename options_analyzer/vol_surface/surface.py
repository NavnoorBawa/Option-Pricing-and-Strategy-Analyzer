"""The volatility surface: :class:`VolSurface`, built from SSVI (:func:`ssvi_surface`) or fitted to market quotes
(:func:`fit_market_surface`), and its per-maturity summary (:func:`surface_summary`).

:func:`fit_market_surface` runs in four stages: :func:`_prepare` (each maturity's quotes in forward log-moneyness,
dense ones binned, thin ones skipped), :func:`_fit_slices` (a raw SVI smile per maturity, and the model-free
static-arbitrage checks of its quotes), :func:`_evaluate_grid` (implied and local vol on the K/S grid) and
:func:`_diagnostics` (the notes on the quotes, the fits and the grid). The first two work through the maturities
together, one at a time: each maturity is fitted before the next one is read.

:func:`ssvi_surface` lives here rather than in :mod:`.ssvi` because it builds a :class:`VolSurface`, which in turn
holds the :class:`~.ssvi.SSVIParams` it came from.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd

from . import svi as svi_module
from .local_vol import _local_vol, _market_values, _MarketValues
from .quotes import (
    FLY_CHECK_VOL,
    MATURITY_TOLERANCE,
    MIN_PRICE_ERROR,
    QUOTED_VOL_TICK,
    SPREAD_CHECK_VOL,
    UNKNOWN_PRICE_ERROR,
    QuoteParseError,
    _butterfly_arbitrage,
    _spread_arbitrage,
    _strike_text,
)
from .ssvi import SSVIParams
from .svi import WING_CAP, WING_SDS, SVISlice, fit_svi_pair, spread_weights, static_arbitrage, wing_reach


@dataclass(frozen=True)
class VolSurface:
    """Implied and local volatility on a (maturity x moneyness) grid; moneyness = K / S.

    Market surfaces also carry the cleaned quotes (with the fitted vol at each
    quote), the per-maturity SVI fits, their implied-vol RMSE and the quoted
    forward log-moneyness range of every maturity, and the forward of every maturity
    (``log_forward_ratio``, ``ln(F / S)``, with the ``spot`` it is relative to and the
    ``discount_factor``; :attr:`log_forward` is ``ln F``): from put-call parity when the
    quotes gave it, otherwise ``carry T``. Between maturities ``ln(F / S)`` is interpolated
    linearly in ``T``.
    ``forwards`` is the table of :func:`.forwards.implied_forwards` for the fitted
    maturities when the quotes were option prices. ``calibration`` says how the SVI
    slices were fitted (one of :data:`CALIBRATIONS`), and ``fit_rmse_unconstrained`` is
    each maturity's RMSE without the arbitrage constraints (equal to ``fit_rmse`` where
    the unconstrained fit had no arbitrage). ``arbitrage_cost`` is, for an arbitrage-free
    calibration, how much the constraints raised each maturity's spread-weighted RMSE (the
    error the fit minimises; 0 where they did not bind, NaN where no arbitrage-free smile was
    found and the maturity keeps its unconstrained fit, with a note). ``wing_reach`` is how far beyond its quotes
    each maturity's smile is read when extrapolating (:func:`.svi.wing_reach`). ``local_vol_error`` is, for every
    maturity, the estimated truncation error of its local vols from the spacing of the expiries, as a vol (0.01 is
    one vol point; see :func:`fit_market_surface`), and None for SSVI, whose ``dw/dT`` is analytic. Implied vol is
    NaN outside a maturity's quoted strike range and local vol is NaN wherever it is undefined.
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
    log_forward_ratio: np.ndarray | None = None
    spot: float | None = None
    discount_factor: np.ndarray | None = None
    forwards: pd.DataFrame | None = None
    fit_rmse_unconstrained: np.ndarray | None = None
    calibration: str | None = None
    wing_reach: np.ndarray | None = None
    arbitrage_cost: np.ndarray | None = None
    local_vol_error: np.ndarray | None = None

    @property
    def strikes_over_spot(self) -> np.ndarray:
        return self.moneyness

    @property
    def log_forward(self) -> np.ndarray | None:
        """``ln F`` at every maturity of a market surface (None for SSVI, whose forward is ``S e^{carry T}``)."""
        if self.log_forward_ratio is None or self.spot is None:
            return None
        return np.log(self.spot) + self.log_forward_ratio

    def _log_fs(self) -> np.ndarray:
        """``ln(F / S)`` at every maturity: the forwards of a market surface, else ``carry T``."""
        return self.carry * self.maturities if self.log_forward_ratio is None else self.log_forward_ratio

    def _reach(self) -> np.ndarray:
        """How far beyond its quotes (beyond the K/S grid for SSVI) each maturity's smile is read, in ``|ln K/F|``."""
        if self.wing_reach is not None:
            return self.wing_reach
        if self.ssvi is not None:
            return np.minimum(WING_CAP, WING_SDS * np.sqrt(self.ssvi.theta(self.maturities)))
        return np.zeros(len(self.maturities))

    def reach_at(self, maturity: float) -> float:
        """How far from the forward, in ``|ln K/F|``, :meth:`implied_vol_at` with ``extrapolate=True`` reads the smile
        at ``maturity``: :meth:`_reach` there, the smaller of the two maturities around it (that of a snapped one)."""
        Ts, reach = self.maturities, self._reach()
        if self.ssvi is not None:
            return float(min(WING_CAP, WING_SDS * np.sqrt(self.ssvi.theta(maturity))))
        j = int(np.clip(np.searchsorted(Ts, maturity), 0, len(Ts) - 1))
        nearest = int(np.argmin(np.abs(Ts - maturity)))
        if abs(Ts[nearest] - maturity) <= MATURITY_TOLERANCE:
            return float(reach[nearest])
        return float(min(reach[max(j - 1, 0)], reach[j]))

    def _log_fs_at(self, maturity: float) -> float:
        """``ln(F / S)`` at ``maturity``, linear in ``T`` between the surface's maturities."""
        if self.log_forward_ratio is None:
            return self.carry * maturity
        return float(np.interp(maturity, self.maturities, self.log_forward_ratio))

    def quoted_mask(self, levels) -> np.ndarray:
        """Where :meth:`at_moneyness` reads each maturity (rows) at K/S ``levels`` inside its quotes (inside the K/S
        grid for SSVI): the points that ``extrapolate=True`` adds are the others."""
        return np.isfinite(self.at_moneyness(levels)[0])

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

    def implied_vol_at(self, moneyness: float, maturity: float, *, extrapolate: bool = False) -> float:
        """Implied vol at ``K/S = moneyness`` and ``T = maturity``; NaN outside the surface's quoted region.

        Market surfaces interpolate total variance linearly in maturity at fixed
        forward log-moneyness between the two SVI slices around ``maturity``,
        and only where both slices were quoted at that strike. A maturity within
        half a day of a quoted one is read off that slice (quote files often
        round maturities, e.g. 30 days to 0.0822 years).

        With ``extrapolate=True`` the smiles are also read beyond their quotes (beyond the K/S grid for SSVI), out
        to a forward log-moneyness ``|ln K/F|`` of four at-the-money standard deviations, at most
        :data:`.svi.WING_CAP` (roadmap item SF-8; :func:`.svi.wing_reach`): an SVI wing grows at most linearly, with
        a total-variance slope within Lee's bound of 2, and an arbitrage-free fit keeps a positive density out to
        there. :meth:`is_extrapolated` says whether a point is read that way.
        """
        T, Ts = float(maturity), self.maturities
        if not (moneyness > 0 and Ts[0] - MATURITY_TOLERANCE <= T <= Ts[-1] + MATURITY_TOLERANCE):
            return np.nan
        if self.ssvi is not None:
            y = np.log(moneyness) - self.carry * T
            inside = self.moneyness[0] <= moneyness <= self.moneyness[-1]
            if not (inside or (extrapolate and abs(y) <= self.reach_at(T))):
                return np.nan
            return float(np.sqrt(self.ssvi.total_variance(y, T)[0] / T))
        if not self.svi or self.quoted_y is None:
            return np.nan
        nearest = int(np.argmin(np.abs(Ts - T)))
        snap = abs(Ts[nearest] - T) <= MATURITY_TOLERANCE
        if snap:
            T = float(Ts[nearest])
        y = np.log(moneyness) - self._log_fs_at(T)
        j = nearest if snap else int(np.searchsorted(Ts, T)) - 1
        around = [j] if snap else [j, j + 1]
        quoted = all(self.quoted_y[k, 0] - 1e-9 <= y <= self.quoted_y[k, 1] + 1e-9 for k in around)
        if not (quoted or (extrapolate and abs(y) <= self.reach_at(T))):
            return np.nan
        variances = [float(self.svi[k].total_variance(y)[0]) for k in around]
        w = variances[0] if snap else variances[0] + (variances[1] - variances[0]) * (T - Ts[j]) / (Ts[j + 1] - Ts[j])
        return float(np.sqrt(w / T)) if w > 0 else np.nan

    def is_extrapolated(self, moneyness: float, maturity: float) -> bool:
        """Whether :meth:`implied_vol_at` with ``extrapolate=True`` reads ``(moneyness, maturity)`` beyond the quotes
        (beyond the K/S grid for SSVI) rather than inside them; False where it gives no vol at all."""
        if not np.isfinite(self.implied_vol_at(moneyness, maturity, extrapolate=True)):
            return False
        return not np.isfinite(self.implied_vol_at(moneyness, maturity))

    def at_moneyness(self, levels, *, extrapolate: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """``(implied, local)`` vol at K/S ``levels`` on every maturity of the surface (rows).

        Evaluated from the model itself (SSVI, or the SVI slices with the same
        ``dw/dT`` as the grid), not interpolated on the grid, so a coarse grid
        does not bias them. NaN outside the surface's K/S range (SSVI), outside
        a maturity's quoted strikes (market quotes), or where local vol is undefined.
        With ``extrapolate=True`` implied vol is also given beyond them, out to each maturity's reach (see
        :meth:`implied_vol_at`); local vol is not, and :meth:`quoted_mask` tells the extrapolated points apart.
        """
        levels = np.asarray(levels, dtype=float)
        if self.ssvi is not None:
            T = self.maturities[:, None]
            y = np.log(levels)[None, :] - self.carry * T
            w, w_y, w_yy, w_t = self.ssvi.total_variance(y, T)
            local = _local_vol(y, w, w_y, w_yy, w_t)[0]
            outside = (levels < self.moneyness[0] - 1e-12) | (levels > self.moneyness[-1] + 1e-12)
            reach = (np.abs(y) <= self._reach()[:, None]) if extrapolate else ~outside
            return np.where(reach | ~outside, np.sqrt(w / T), np.nan), np.where(outside, np.nan, local)
        if self.svi and self.quoted_y is not None:
            values = _market_values(self.svi, self.maturities, self.quoted_y, self._log_fs(), levels)
            if not extrapolate:
                return values.implied, values.local
            y = np.log(levels)[None, :] - self._log_fs()[:, None]
            w = np.array([fit.total_variance(row)[0] for fit, row in zip(self.svi, y, strict=True)])
            wings = (np.abs(y) <= self._reach()[:, None]) & (w > 0) & ~np.isfinite(values.implied)
            implied = np.where(wings, np.sqrt(np.maximum(w, 0.0) / self.maturities[:, None]), values.implied)
            return implied, values.local
        m = self.moneyness
        outside = (levels < m[0]) | (levels > m[-1])

        def grid(values):  # np.interp gives NaN next to a NaN, so it never bridges a gap
            return np.array([np.where(outside, np.nan, np.interp(levels, m, row)) for row in values])

        return grid(self.implied_vol), grid(self.local_vol)


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
# Market quotes
# ---------------------------------------------------------------------------


MIN_STRIKES_PER_SLICE = 5
MAX_POINTS_PER_SLICE = 200  # denser slices are averaged into this many bins in y before the SVI fit
POOR_FIT_RMSE = 0.01
#: The arbitrage-free calibration is mentioned in a note only where it costs a maturity more than this in
#: spread-weighted RMSE (vol) and more than a quarter of its unconstrained spread-weighted RMSE: tick-rounded quotes
#: leave many unconstrained smiles a sliver of negative density just beyond their quotes, which the constraint
#: removes for a few thousandths of a vol point.
MATERIAL_ARBITRAGE_COST = 2e-4
#: How :func:`fit_market_surface` fits the SVI slices: free of static arbitrage (the default), or without that
#: constraint, to see what it costs.
CALIBRATIONS = ("arbitrage_free", "unconstrained")
MIN_GRID_POINTS_PER_SLICE = 16  # the K/S grid is refined until every maturity has about this many points in its quotes
MAX_GRID_POINTS = 301
FAR_FROM_SPOT = 0.2  # |ln(K/S)| of the nearest quoted strike beyond which the spot looks wrong (about 22%)


def _binned(
    y: np.ndarray, iv: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Quotes averaged into :data:`MAX_POINTS_PER_SLICE` bins of equal width in ``y``: ``(y, iv, weights)``.

    Without weights each bin's quotes are averaged and every bin weighs the same. With weights each bin's ``iv`` is
    their mean weighted by ``weight^2`` (a least-squares weight is the root of an inverse variance) and the bin
    weighs the root mean square of its quotes' weights, so a bin of tight quotes still counts more than one of wide.
    """
    edges = np.linspace(y.min(), y.max(), MAX_POINTS_PER_SLICE + 1)
    which = np.clip(np.searchsorted(edges, y, side="right") - 1, 0, MAX_POINTS_PER_SLICE - 1)
    counts = np.bincount(which, minlength=MAX_POINTS_PER_SLICE)
    keep = counts > 0
    y_bins = np.bincount(which, y, MAX_POINTS_PER_SLICE)[keep] / counts[keep]
    if weights is None:
        return y_bins, np.bincount(which, iv, MAX_POINTS_PER_SLICE)[keep] / counts[keep], None
    square = np.bincount(which, weights**2, MAX_POINTS_PER_SLICE)[keep]
    iv_bins = np.bincount(which, weights**2 * iv, MAX_POINTS_PER_SLICE)[keep] / square
    return y_bins, iv_bins, np.sqrt(square / counts[keep])


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


def fit_market_surface(
    quotes: pd.DataFrame,
    spot: float,
    rate: float,
    dividend_yield: float,
    n_moneyness: int = 41,
    *,
    forwards: pd.DataFrame | None = None,
    calibration: str = "arbitrage_free",
) -> VolSurface:
    """Fit a surface to cleaned quotes (see :func:`parse_quotes`) and compute local vol.

    Every maturity with enough strikes is fitted with raw SVI (:func:`fit_svi`) in forward
    log-moneyness ``y = ln(K / F_T)``: with the forward of each quote's expiry where the
    quotes carry ``forward`` and ``discount_factor`` columns (put-call parity, see
    :func:`parse_quotes`), otherwise ``F_T = spot e^{(rate - dividend_yield) T}`` and the
    discount factor ``e^{-rate T}``. ``forwards``, the table :func:`parse_quotes` returns with
    ``return_forwards=True``, is kept on the surface (its rows at the fitted maturities) for display.

    Each quote weighs on its slice's fit as one over the half-spread of its implied vol (its ``price_error`` over its
    vega; :func:`.svi.spread_weights`), equal weights for quoted implied vols. With ``calibration="arbitrage_free"``
    (the default) each slice is held free of butterfly arbitrage, and of calendar arbitrage against the slice fitted
    before it on the overlap of their quotes (:func:`.svi.fit_svi`); ``"unconstrained"`` drops that constraint.
    The result is evaluated at the quoted maturities (no extrapolation in time)
    on a K/S grid spanning the union of the quoted strike ranges, with
    ``n_moneyness`` points across it and more inside narrow ranges
    (:func:`_moneyness_grid`). Implied vol is NaN outside each maturity's own
    quoted range; local vol is also NaN where ``dw/dT`` would need a neighbouring
    maturity far outside its quotes, where the surface has calendar or butterfly
    arbitrage, or where it is otherwise undefined. The quotes themselves are checked
    for static arbitrage, allowing each price its ``price_error`` (see
    :func:`parse_quotes`; where it is NaN or the column is missing,
    :data:`UNKNOWN_PRICE_ERROR` of the forward or :data:`MIN_PRICE_ERROR` capped by the
    strike spacing, whichever is more); arbitrage in quoted vols that half a :data:`QUOTED_VOL_TICK` of error
    in each price would remove is reported as possible rather than certain.

    ``local_vol_error`` estimates the truncation error of local vol from the spacing of the expiries: at each
    maturity, the largest change to its local vols over the grid when ``dw/dT`` comes from one neighbouring slice
    more, one order higher (:func:`.local_vol._time_derivative`), as a vol (0.01 is one vol point). On noise-free SSVI
    quotes on the listed (30 days to two years), weekly-front, two-weekly and quarterly expiries it is 0.57 to 2.0
    times the actual error at every maturity where that exceeds 0.01 vol point; over 1,205 such maturities of 60
    random SSVI surfaces, each on the listed, weekly-front, quarterly and 30, 91, 365 and 730-day expiries, it is a
    median 1.08 times the error, at least the error at 60% of them and twice it at 90%. It is an estimate, not a
    bound: where the expiries are too far apart to resolve the term structure it falls short several-fold (on the
    30, 91, 365 and 730-day expiries with an ATM vol falling from 80% to 20%, 0.59 against 2.5 vol points at a year;
    at one maturity in a hundred of the random surfaces, a sixth of the error or less), and with noisy quotes it
    moves with the noise rather than measuring it. NaN at a maturity where no larger stencil can be used at any of
    its grid points: with fewer than three maturities, at all but the last of three, or where a slice of it is not
    quoted near the forward.
    """
    if calibration not in CALIBRATIONS:
        raise ValueError(f"calibration must be one of {', '.join(map(repr, CALIBRATIONS))}, got {calibration!r}")
    carry = rate - dividend_yield
    notes: list[str] = []
    # _prepare yields one maturity at a time, which _fit_slices fits before the next one is read: on quotes that
    # parse_quotes did not clean, the first maturity the fit fails on decides the error and the warnings.
    slices = _fit_slices(_prepare(quotes, spot, rate, carry, notes), spot, calibration == "arbitrage_free")
    if len(slices) < 2:
        n_maturities = quotes["maturity"].nunique()
        raise QuoteParseError(
            f"local volatility needs at least two maturities with {MIN_STRIKES_PER_SLICE}+ strikes each; the quotes "
            f"have {n_maturities} maturit{'y' if n_maturities == 1 else 'ies'}, {len(slices)} of them with "
            f"{MIN_STRIKES_PER_SLICE}+ strikes"
        )
    Ts = np.array([piece.T for piece in slices])
    quoted_y = np.array([piece.quoted for piece in slices])
    # ln(F/S) of each maturity: carry T exactly where it came from the rate and dividend yield (as before parity
    # forwards existed), ln(F/S) of the parity forward elsewhere.
    log_fs = np.array([carry * piece.T if piece.from_rates else np.log(piece.forward / spot) for piece in slices])
    moneyness, values = _evaluate_grid(slices, Ts, quoted_y, log_fs, n_moneyness)
    assert values.local_error is not None  # _evaluate_grid asks for it
    notes += _diagnostics(slices, Ts, values, quotes, spot)
    clean_quotes = pd.concat([piece.quotes for piece in slices]).reset_index(drop=True)
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
        quotes=clean_quotes[["strike", "maturity", "iv", "moneyness", "fitted_iv", "vol_half_spread"]],
        fit_rmse=np.array([piece.rmse for piece in slices]),
        fit_rmse_unconstrained=np.array([piece.rmse_unconstrained for piece in slices]),
        calibration=calibration,
        arbitrage_cost=(
            np.array([piece.arbitrage_cost for piece in slices]) if calibration == "arbitrage_free" else None
        ),
        wing_reach=np.array([piece.reach for piece in slices]),
        local_vol_error=np.fmax.reduce(values.local_error, axis=1),  # NaN, without a warning, where none is estimated
        svi=tuple(piece.svi for piece in slices),
        quoted_y=quoted_y,
        log_forward_ratio=log_fs,
        spot=float(spot),
        discount_factor=np.array([piece.discount_factor for piece in slices]),
        forwards=_fitted_forwards(forwards, Ts),
    )


def _fitted_forwards(table: pd.DataFrame | None, Ts: np.ndarray) -> pd.DataFrame | None:
    """The rows of a table of forwards (see :func:`parse_quotes`) at the fitted maturities ``Ts``, or None."""
    if table is None or table.empty:
        return None
    near = np.abs(table["maturity"].to_numpy(float)[:, None] - Ts[None, :]).min(axis=1) <= 1e-9
    return table[near].reset_index(drop=True)


# The stages of fit_market_surface, in the order it runs them: _prepare (each maturity's quotes), _fit_slices (an SVI
# smile per maturity and the model-free checks of its quotes), _evaluate_grid (implied and local vol on the K/S grid)
# and _diagnostics (the notes on the quotes, the fits and the grid, after those on skipped maturities).


@dataclass(frozen=True, eq=False)  # holds arrays and a table: compared by identity
class _Prepared:
    """One maturity's quotes, ready for the SVI fit (stage 1 of :func:`fit_market_surface`)."""

    T: float
    forward: float  # F_T, and the discount factor of the expiry
    discount_factor: float
    from_rates: bool  # whether they came from the rate and dividend yield (no forward columns in the quotes)
    quotes: pd.DataFrame  # its rows of the cleaned quotes
    y: np.ndarray  # forward log-moneyness of each quote
    iv: np.ndarray  # implied vol of each quote
    vol_half_spread: np.ndarray  # how far each quote's vol may be off: its price error over its vega (NaN: unknown)
    weights: np.ndarray | None  # each quote's weight on the fit (see svi.spread_weights); None: equal
    y_fit: np.ndarray  # what the SVI fit is given: y, iv and weights, or their bins (MAX_POINTS_PER_SLICE)
    iv_fit: np.ndarray
    weights_fit: np.ndarray | None  # see svi.spread_weights; None: equal


@dataclass(frozen=True, eq=False)  # holds a table: compared by identity
class _Slice:
    """One maturity's SVI fit and the checks of its quotes (stage 2 of :func:`fit_market_surface`)."""

    T: float
    forward: float  # F_T, and the discount factor of the expiry
    discount_factor: float
    from_rates: bool  # whether they came from the rate and dividend yield (no forward columns in the quotes)
    svi: SVISlice
    quoted: tuple[float, float]  # the range of forward log-moneyness of its quotes
    rmse: float  # implied-vol RMSE of the fit at the quotes
    rmse_unconstrained: float  # the same without the arbitrage constraints (see svi.fit_svi_pair)
    arbitrage_cost: float  # what the constraints added to the spread-weighted RMSE; NaN where they could not be met
    weighted_rmse_unconstrained: float  # the spread-weighted RMSE without the constraints, which that is set against
    held: tuple[bool, bool]  # (butterfly, calendar): arbitrage the unconstrained fit had and the constraints removed
    left: tuple[bool, bool]  # (butterfly, calendar): arbitrage left where no arbitrage-free smile was found
    reach: float  # how far beyond its quotes the smile is read (svi.wing_reach of the unconstrained fit)
    binned: bool  # whether dense quotes were averaged into MAX_POINTS_PER_SLICE bins before the fit
    spread_arbitrage: bool  # see _spread_arbitrage
    butterflies: tuple[int, tuple[float, float, float] | None]  # see _butterfly_arbitrage
    certain: bool  # whether arbitrage found in quoted vols is beyond what rounding their prices could cause
    quotes: pd.DataFrame  # its rows of the cleaned quotes, with their K/S (moneyness) and fitted vol (fitted_iv)
    budget_spent: bool = False  # whether the surface's solver iterations (svi.SURFACE_BUDGET) had run out by its end


def _prepare(quotes: pd.DataFrame, spot: float, rate: float, carry: float, notes: list[str]) -> Iterator[_Prepared]:
    """Stage 1 of :func:`fit_market_surface`: each maturity's forward and discount factor (:func:`_expiry_carry`),
    forward log-moneyness and vols, and those the SVI fit is given (dense quotes averaged into
    :data:`MAX_POINTS_PER_SLICE` bins, :func:`_binned`), yielded in order of maturity as they are read; a maturity
    with fewer than :data:`MIN_STRIKES_PER_SLICE` strikes is skipped with a note, appended to ``notes``."""
    for maturity, grp in quotes.groupby("maturity"):
        T = float(cast(float, maturity))
        if grp["strike"].nunique() < MIN_STRIKES_PER_SLICE:
            notes.append(f"Skipped maturity {T:.4g}y: fewer than {MIN_STRIKES_PER_SLICE} strikes.")
            continue
        forward, discount_factor = _expiry_carry(grp, spot, rate, carry, T)
        from_rates = forward is None
        if forward is None:
            y = np.log(grp["strike"].to_numpy(float) / spot) - carry * T
            forward = spot * np.exp(carry * T)
        else:
            y = np.log(grp["strike"].to_numpy(float) / forward)
        iv = grp["iv"].to_numpy(float)
        half_spread, weight_basis = _vol_half_spread(grp, y, iv, T, float(forward), discount_factor)
        weights = spread_weights(weight_basis)
        y_fit, iv_fit, weights_fit = (y, iv, weights) if len(y) <= MAX_POINTS_PER_SLICE else _binned(y, iv, weights)
        yield _Prepared(
            T, float(forward), discount_factor, from_rates, grp, y, iv, half_spread, weights, y_fit, iv_fit, weights_fit
        )


def _vol_half_spread(
    grp: pd.DataFrame, y: np.ndarray, iv: np.ndarray, T: float, forward: float, discount_factor: float
) -> tuple[np.ndarray, np.ndarray]:
    """How far each quote's implied vol may be off, and what its weight on the fit is figured from.

    The first is its ``price_error`` (half its bid/ask spread, or its price grid's step, see :func:`parse_quotes`)
    over its Black vega ``DF F sqrt(T) phi(d1)``; NaN where the price error is unknown (quoted implied vols) or zero
    (a price on no grid at all, with no bid or ask), so the summary's Inside bid/ask is blank rather than measured
    against a spread the quotes do not have. The second is the same, except that a zero price error counts as one
    unit of price: such quotes are weighted by their vega (:func:`.svi.spread_weights` normalises the scale away).
    One unit of price stood in for the spread in both before: exact prices with a vol point of noise read 100 %
    inside a "bid/ask" that was 2,482 vol points wide for an underlying priced at 0.5.
    """
    if "price_error" not in grp:
        unknown = np.full(len(y), np.nan)
        return unknown, unknown
    error = grp["price_error"].to_numpy(float)
    sd = iv * np.sqrt(T)
    d1 = -y / sd + 0.5 * sd
    vega = discount_factor * forward * np.sqrt(T) * np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)

    def in_vol(price_error: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return np.where(vega > 0, price_error / vega, np.inf)

    known = error > 0
    basis = in_vol(np.where(known, error, np.where(np.isfinite(error), 1.0, np.nan)))
    return np.where(known, basis, np.nan), basis


def _expiry_carry(grp: pd.DataFrame, spot: float, rate: float, carry: float, T: float) -> tuple[float | None, float]:
    """The forward and discount factor of one maturity's quotes: their ``forward`` and ``discount_factor`` columns
    (one value per expiry) where present and valid, else ``(None, e^{-rate T})`` (the forward is then
    ``spot e^{carry T}``, computed in log-moneyness exactly as before parity forwards existed)."""
    if "forward" in grp and "discount_factor" in grp:
        forward = float(np.median(grp["forward"].to_numpy(float)))
        discount_factor = float(np.median(grp["discount_factor"].to_numpy(float)))
        if forward > 0 and discount_factor > 0 and np.isfinite(forward) and np.isfinite(discount_factor):
            return forward, discount_factor
    return None, float(np.exp(-rate * T))


def _fit_slices(prepared: Iterable[_Prepared], spot: float, arbitrage_free: bool = True) -> list[_Slice]:
    """Stage 2 of :func:`fit_market_surface`: the raw-SVI fit of each maturity (:func:`fit_svi`), weighted by the
    quotes' spreads and, with ``arbitrage_free``, held free of butterfly arbitrage and of calendar arbitrage against
    the maturity fitted before it; its RMSE at the quotes, with and without those constraints; and the model-free
    static-arbitrage checks of the quotes (:func:`_quote_arbitrage`).

    The constraints' cost is the rise in spread-weighted RMSE, the error the fit minimises, clipped at 0: the
    unweighted RMSE can fall under the constraints (by 0.0004 vol point at one expiry of the synthetic SPX chain at a
    spot of 6,000 on 30 October at 10:00, noise seed 2), which read as a negative cost.

    The maturities share :data:`.svi.SURFACE_BUDGET` solver iterations, in order of maturity: once they are spent,
    a maturity whose unconstrained fit has arbitrage keeps it, with a note (:func:`_arbitrage_left`)."""
    slices: list[_Slice] = []
    budget = svi_module._Budget(svi_module.SURFACE_BUDGET)
    for piece in prepared:
        T, y, iv = piece.T, piece.y, piece.iv
        previous = (slices[-1].svi, slices[-1].quoted) if slices else None
        svi, free = fit_svi_pair(
            piece.y_fit,
            piece.iv_fit,
            T,
            weights=piece.weights_fit,
            previous=previous,
            arbitrage_free=arbitrage_free,
            budget=budget,
        )
        fitted = np.sqrt(np.maximum(svi.total_variance(y)[0], 0.0) / T)
        rmse = float(np.sqrt(np.mean((fitted - iv) ** 2)))
        unconstrained = np.sqrt(np.maximum(free.total_variance(y)[0], 0.0) / T)
        rmse_free = rmse if free is svi else float(np.sqrt(np.mean((unconstrained - iv) ** 2)))
        # The checks fit_svi_pair made, on the range of what it was given (bins of dense quotes): where it returned
        # the unconstrained fit of an arbitrage-free calibration, either that had no arbitrage or no start found a
        # smile without it.
        fit_range = (float(piece.y_fit.min()), float(piece.y_fit.max()))
        reach = wing_reach(free)
        held = (False, False) if free is svi else static_arbitrage(free, fit_range, T, previous, reach=reach)
        left = (False, False)
        if arbitrage_free and free is svi:
            left = static_arbitrage(svi, fit_range, T, previous, reach=reach)
        weights = np.ones_like(y) if piece.weights is None else piece.weights
        weighted_free = _weighted_rmse(unconstrained, iv, weights)
        cost = math.nan if any(left) else max(_weighted_rmse(fitted, iv, weights) - weighted_free, 0.0)
        spread, butterflies, certain = _quote_arbitrage(piece.quotes, iv, T, piece.forward, piece.discount_factor)
        with_fit = piece.quotes.assign(
            moneyness=piece.quotes["strike"] / spot, fitted_iv=fitted, vol_half_spread=piece.vol_half_spread
        )
        binned = len(piece.y_fit) < len(y)
        slices.append(
            _Slice(
                T,
                piece.forward,
                piece.discount_factor,
                piece.from_rates,
                svi,
                (y.min(), y.max()),
                rmse,
                rmse_free,
                cost,
                weighted_free,
                held,
                left,
                wing_reach(free),
                binned,
                spread,
                butterflies,
                certain,
                with_fit,
                budget_spent=budget.left <= 0,
            )
        )
    return slices


def _weighted_rmse(fitted: np.ndarray, iv: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.mean((weights * (fitted - iv)) ** 2)))


def _quote_arbitrage(
    grp: pd.DataFrame, iv: np.ndarray, T: float, forward: float, discount_factor: float
) -> tuple[bool, tuple[int, tuple[float, float, float] | None], bool]:
    """Vertical-spread and butterfly arbitrage in one maturity's quotes (:func:`_spread_arbitrage`,
    :func:`_butterfly_arbitrage`), and whether it is beyond what rounding the prices behind quoted vols to
    :data:`QUOTED_VOL_TICK` ticks could cause."""
    # Each quote's price error, undiscounted like the Black prices it widens: its rounding or half its bid/ask
    # spread (see parse_quotes), or for an implied vol without a price UNKNOWN_PRICE_ERROR of the forward or
    # MIN_PRICE_ERROR, whichever is more.
    known = grp["price_error"].to_numpy(float) if "price_error" in grp else np.full(len(grp), np.nan)
    unknown = ~np.isfinite(known)
    growth = 1.0 / discount_factor
    strike = grp["strike"].to_numpy(float)
    # Strikes on a sub-cent grid (FX, penny stocks) are priced in sub-cent ticks: scale both floors with the
    # strike spacing, or a half-cent floor hides gross arbitrage on an underlying priced near 1.
    gap = float(np.median(np.diff(np.unique(strike)))) if len(np.unique(strike)) > 1 else forward
    floor = min(MIN_PRICE_ERROR * growth, 0.01 * gap)
    price_error = np.where(unknown, max(UNKNOWN_PRICE_ERROR * forward, floor), known * growth)
    spread = _spread_arbitrage(strike, iv, T, forward, price_error)
    butterflies = _butterfly_arbitrage(strike, iv, T, forward, price_error)
    if not (unknown.any() and (spread or butterflies[0])):
        return spread, butterflies, True
    # Is it beyond the rounding of QUOTED_VOL_TICK?
    tick = min(0.5 * QUOTED_VOL_TICK * growth, 0.05 * gap)
    tick_error = np.where(unknown, np.maximum(price_error, tick), price_error)
    certain = (
        _spread_arbitrage(strike, iv, T, forward, tick_error)
        or _butterfly_arbitrage(strike, iv, T, forward, tick_error)[0] > 0
    )
    return spread, butterflies, certain


def _evaluate_grid(
    slices: list[_Slice], Ts: np.ndarray, quoted_y: np.ndarray, log_fs: np.ndarray, n_moneyness: int
) -> tuple[np.ndarray, _MarketValues]:
    """Stage 3 of :func:`fit_market_surface`: the K/S grid (:func:`_moneyness_grid`), and implied and local vol on it
    with the truncation error of local vol (:func:`_market_values`); ``log_fs`` is ``ln(F / S)`` at each maturity."""
    moneyness = _moneyness_grid(np.exp(quoted_y + log_fs[:, None]), n_moneyness)
    fits = [piece.svi for piece in slices]
    return moneyness, _market_values(fits, Ts, quoted_y, log_fs, moneyness, truncation_error=True)


def _diagnostics(
    slices: list[_Slice], Ts: np.ndarray, values: _MarketValues, quotes: pd.DataFrame, spot: float
) -> list[str]:
    """Stage 4 of :func:`fit_market_surface`: the notes on the quotes, the fits and the grid, in this order: dense
    quotes binned, strikes far from the spot (:func:`_far_from_spot`), poor fits (:func:`_poor_fits`), static
    arbitrage in the quotes (:func:`_static_arbitrage`) and grid points without a local vol
    (:func:`_blank_local_vol`)."""
    notes = []
    binned = sum(piece.binned for piece in slices)
    if binned:
        notes.append(
            f"Averaged dense quotes into at most {MAX_POINTS_PER_SLICE} points per maturity before fitting "
            f"({binned} maturit{'y' if binned == 1 else 'ies'})."
        )
    strikes = pd.concat([piece.quotes for piece in slices])["strike"].to_numpy(float)
    notes += _far_from_spot(strikes, spot)
    rmse = np.array([piece.rmse for piece in slices])
    spread = np.array([piece.spread_arbitrage for piece in slices])
    crossed = (rmse > POOR_FIT_RMSE) & spread
    notes += _poor_fits(Ts, rmse, crossed)
    notes += _held_arbitrage(slices, Ts)
    notes += _arbitrage_left(slices, Ts)
    notes += _static_arbitrage(slices, Ts, crossed, quotes)
    return notes + _blank_local_vol(values)


def _years(Ts: np.ndarray) -> str:
    """Maturities for a note: ``0.0822y, 0.249y``."""
    return ", ".join(f"{t:.3g}y" for t in Ts)


def _far_from_spot(strikes: np.ndarray, spot: float) -> list[str]:
    """A note when the quoted strikes are far from ``spot``, which then looks like the wrong underlying price."""
    log_moneyness = np.log(strikes / spot)
    nearest = int(np.argmin(np.abs(log_moneyness)))
    if abs(log_moneyness[nearest]) > FAR_FROM_SPOT or not 0.5 <= np.exp(np.median(log_moneyness)) <= 2.0:
        gap = strikes[nearest] / spot - 1.0
        return [
            f"Quoted strikes run from {strikes.min():,.2f} to {strikes.max():,.2f}, far from the spot {spot:,.2f} "
            f"used to read them (the nearest is {abs(gap):.0%} {'below' if gap < 0 else 'above'} it); set the "
            "sidebar spot to the underlying price the quotes were taken at."
        ]
    return []


def _poor_fits(Ts: np.ndarray, rmse: np.ndarray, crossed: np.ndarray) -> list[str]:
    """Notes on the maturities fitted more than :data:`POOR_FIT_RMSE` from their quotes: ``crossed`` ones, whose
    quotes have vertical-spread arbitrage, then the others."""
    poor = rmse > POOR_FIT_RMSE
    notes = []
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
                f"maturities {_years(Ts[which])}{cause} The Smiles chart shows how far the fit is from each quote."
            )
    return notes


def _held_arbitrage(slices: list[_Slice], Ts: np.ndarray) -> list[str]:
    """A note on the maturities whose unconstrained SVI fit had static arbitrage that the arbitrage-free calibration
    removed at a material cost (:data:`MATERIAL_ARBITRAGE_COST`), and what it cost: the quotes there imply the
    arbitrage, so they deserve a look."""
    cost = np.array([piece.arbitrage_cost for piece in slices])
    free = np.array([piece.weighted_rmse_unconstrained for piece in slices])
    material = (cost > MATERIAL_ARBITRAGE_COST) & (cost > 0.25 * free)  # NaN (no arbitrage-free smile): not here
    calendar = np.array([piece.held[1] for piece in slices]) & material
    butterfly = np.array([piece.held[0] for piece in slices]) & material & ~calendar
    if not (calendar.any() or butterfly.any()):
        return []
    parts = []
    if calendar.any():
        parts.append(
            f"at maturities {_years(Ts[calendar])} the quotes' total variance falls below that of the maturity before "
            "(calendar arbitrage)"
        )
    if butterfly.any():
        parts.append(
            f"at maturities {_years(Ts[butterfly])} the best smile has a negative density (butterfly arbitrage)"
        )
    return [
        f"The SVI fit was held free of static arbitrage: {'; '.join(parts)}. That moved the fit by up to "
        f"{cost[calendar | butterfly].max() * 100:.2f} vol point of spread-weighted RMSE (Arbitrage cost in the "
        "table below); check those quotes, or fit them unconstrained to see the arbitrage."
    ]


def _arbitrage_left(slices: list[_Slice], Ts: np.ndarray) -> list[str]:
    """A note on the maturities where the arbitrage-free calibration found no smile free of static arbitrage, so they
    keep their unconstrained fit, arbitrage and all (:func:`.svi.fit_svi`), and on those where that was because the
    surface's solver iterations (:data:`.svi.SURFACE_BUDGET`) had run out."""
    butterfly = np.array([piece.left[0] for piece in slices])
    calendar = np.array([piece.left[1] for piece in slices])
    if not (butterfly.any() or calendar.any()):
        return []
    parts = []
    if calendar.any():
        parts.append(f"at {_years(Ts[calendar])} its total variance falls below that of the maturity before (calendar)")
    if butterfly.any():
        parts.append(f"at {_years(Ts[butterfly])} it has a negative density (butterfly)")
    one = int((calendar | butterfly).sum()) == 1
    note = (
        f"No SVI smile free of static arbitrage was found at maturit{'y' if one else 'ies'} "
        f"{_years(Ts[calendar | butterfly])}, so {'it keeps its' if one else 'they keep their'} unconstrained fit and "
        f"its arbitrage: {'; '.join(parts)}. {'Its' if one else 'Their'} Arbitrage cost is blank; check those quotes."
    )
    starved = (calendar | butterfly) & np.array([piece.budget_spent for piece in slices])
    if starved.any():
        note += (
            f" At {_years(Ts[starved])} the search had run out of the {svi_module.SURFACE_BUDGET:,} solver iterations "
            "allowed for the whole surface, a cap that keeps a very noisy chain to a few seconds."
        )
    return [note]


def _static_arbitrage(slices: list[_Slice], Ts: np.ndarray, crossed: np.ndarray, quotes: pd.DataFrame) -> list[str]:
    """A note on static arbitrage in the quotes themselves, which the arbitrage-free SVI fit smooths out of the
    surface, so the surface's arbitrage counts cannot show it. Maturities already named by the ``crossed`` poor-fit
    note are left out."""
    fly = np.array([count > 0 for count, _ in (piece.butterflies for piece in slices)]) & ~crossed
    spread = np.array([piece.spread_arbitrage for piece in slices]) & ~crossed
    if not (fly.any() or spread.any()):
        return []
    parts = []
    if fly.any():
        first = int(np.argmax(fly))
        worst = slices[first].butterflies[1]
        assert worst is not None  # a butterfly was found at that maturity
        k1, k2, k3 = (_strike_text(k) for k in worst)
        parts.append(
            f"at maturities {_years(Ts[fly])} a butterfly of neighbouring strikes costs less than nothing (a negative "
            f"risk-neutral density) even if each quote is {FLY_CHECK_VOL * 100:g} vol point off, e.g. {k1}/{k2}/{k3} "
            f"at {Ts[first]:.3g}y"
        )
    if spread.any():
        parts.append(
            f"at maturities {_years(Ts[spread])} a put costs less than one at a lower strike, or a call more than one "
            f"at a lower strike, even if each quote is {SPREAD_CHECK_VOL * 100:g} vol point off"
        )
    # Arbitrage in quoted vols that the rounding of the prices behind them could cause may not be in the market.
    within_rounding = (fly | spread) & ~np.array([piece.certain for piece in slices])
    unsure = bool(within_rounding[fly | spread].all())
    possible = ""
    if within_rounding.any():
        where = "The" if unsure else f"At maturities {_years(Ts[within_rounding])} the"
        possible = (
            f" {where} arbitrage is small enough that rounding the prices behind the quoted vols to "
            f"{QUOTED_VOL_TICK:g} ticks could cause it, so it may not be in the market."
        )
    return [
        f"The quotes {'may have' if unsure else 'have'} static arbitrage that the fitted surface smooths out: "
        f"{'; '.join(parts)}. The checks also allow {', and '.join(_price_allowances(quotes))}.{possible} The "
        "arbitrage counts above describe the fitted surface, not the quotes, so check these quotes."
    ]


def _price_allowances(quotes: pd.DataFrame) -> list[str]:
    """What the static-arbitrage checks allow each price: its bid/ask or grid where the quotes give one, a fraction
    of the forward for the unknown price behind a quoted implied vol."""
    errors = quotes["price_error"].to_numpy(float) if "price_error" in quotes else np.full(len(quotes), np.nan)
    allowed = []
    if np.isfinite(errors).any():
        allowed.append(
            "each price to be off by half its bid/ask spread (as far as the farther of its bid and ask, for a "
            "price away from their midpoint) or by one step of the file's price grid"
        )
    if not np.isfinite(errors).all():
        allowed.append(
            f"the unknown price behind each quoted implied vol to be off by {UNKNOWN_PRICE_ERROR:.2%} of the "
            f"forward or {MIN_PRICE_ERROR:g} (at most 1% of the strike spacing), whichever is more"
        )
    return allowed


def _blank_local_vol(values: _MarketValues) -> list[str]:
    """Notes on the grid points inside the quotes where local vol is left blank although there is no arbitrage: no
    neighbouring maturity close enough to its quotes for ``dw/dT``, or no measurable change in total variance; and
    a note when the fitted total variance is zero at some quoted point."""
    notes = []
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
    return notes


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
            row.update(_fit_to_spreads(surface, i))
        rows.append(row)
    return pd.DataFrame(rows)


def _fit_to_spreads(surface: VolSurface, i: int) -> dict[str, float]:
    """How the fit of maturity ``i`` sits against its quotes' spreads: the share of quotes whose fitted vol is within
    their half-spread in vol (inside the bid/ask, in percent), the median of ``|fit - quote| / half-spread``, and,
    for an arbitrage-free fit, how much spread-weighted RMSE the constraints cost (vol points; NaN where they could
    not be met). The first two are NaN where no quote of the maturity has a bid/ask spread (or a price grid's step)."""
    row = {"Inside bid/ask (%)": np.nan, "Median |error| / half-spread": np.nan}
    if surface.quotes is not None and "vol_half_spread" in surface.quotes:
        quotes = surface.quotes[np.isclose(surface.quotes["maturity"], surface.maturities[i], rtol=0.0, atol=1e-9)]
        half = quotes["vol_half_spread"].to_numpy(float)
        known = np.isfinite(half) & (half > 0)
        if known.any():
            ratio = np.abs(quotes["fitted_iv"].to_numpy(float) - quotes["iv"].to_numpy(float))[known] / half[known]
            row = {
                "Inside bid/ask (%)": 100.0 * float(np.mean(ratio <= 1.0)),
                "Median |error| / half-spread": float(np.median(ratio)),
            }
    if surface.calibration == "arbitrage_free" and surface.arbitrage_cost is not None:
        row["Arbitrage cost (vol pts)"] = surface.arbitrage_cost[i] * 100
    return row
