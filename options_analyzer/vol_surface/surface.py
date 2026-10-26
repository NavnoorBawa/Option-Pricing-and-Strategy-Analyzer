"""The volatility surface: :class:`VolSurface`, built from SSVI (:func:`ssvi_surface`) or fitted to market quotes
(:func:`fit_market_surface`), and its per-maturity summary (:func:`surface_summary`).

:func:`fit_market_surface` runs in four stages: :func:`_prepare` (each maturity's quotes in forward log-moneyness,
dense ones binned, thin ones skipped), :func:`_fit_slices` (a raw SVI smile per maturity, and the model-free
static-arbitrage checks of its quotes), :func:`_evaluate_grid` (implied and local vol on the K/S grid) and
:func:`_diagnostics` (the notes on the quotes, the fits and the grid).

:func:`ssvi_surface` lives here rather than in :mod:`.ssvi` because it builds a :class:`VolSurface`, which in turn
holds the :class:`~.ssvi.SSVIParams` it came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd

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
from .svi import SVISlice, fit_svi


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
    :func:`parse_quotes`; where it is NaN or the column is missing,
    :data:`UNKNOWN_PRICE_ERROR` of the forward or :data:`MIN_PRICE_ERROR` capped by the
    strike spacing, whichever is more); arbitrage in quoted vols that half a :data:`QUOTED_VOL_TICK` of error
    in each price would remove is reported as possible rather than certain.
    """
    carry = rate - dividend_yield
    prepared, notes = _prepare(quotes, spot, carry)
    slices = _fit_slices(prepared, spot, rate, carry)
    if len(slices) < 2:
        n_maturities = quotes["maturity"].nunique()
        raise QuoteParseError(
            f"local volatility needs at least two maturities with {MIN_STRIKES_PER_SLICE}+ strikes each; the quotes "
            f"have {n_maturities} maturit{'y' if n_maturities == 1 else 'ies'}, {len(slices)} of them with "
            f"{MIN_STRIKES_PER_SLICE}+ strikes"
        )
    Ts = np.array([piece.T for piece in slices])
    quoted_y = np.array([piece.quoted for piece in slices])
    moneyness, values = _evaluate_grid(slices, Ts, quoted_y, carry, n_moneyness)
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
        quotes=clean_quotes[["strike", "maturity", "iv", "moneyness", "fitted_iv"]],
        fit_rmse=np.array([piece.rmse for piece in slices]),
        svi=tuple(piece.svi for piece in slices),
        quoted_y=quoted_y,
    )


# The stages of fit_market_surface, in the order it runs them: _prepare (each maturity's quotes), _fit_slices (an SVI
# smile per maturity and the model-free checks of its quotes), _evaluate_grid (implied and local vol on the K/S grid)
# and _diagnostics (the notes on the quotes, the fits and the grid, after those on skipped maturities).


@dataclass(frozen=True, eq=False)  # holds arrays and a table: compared by identity
class _Prepared:
    """One maturity's quotes, ready for the SVI fit (stage 1 of :func:`fit_market_surface`)."""

    T: float
    quotes: pd.DataFrame  # its rows of the cleaned quotes
    y: np.ndarray  # forward log-moneyness of each quote
    iv: np.ndarray  # implied vol of each quote
    y_fit: np.ndarray  # what the SVI fit is given: y and iv, or their averages in MAX_POINTS_PER_SLICE bins
    iv_fit: np.ndarray


@dataclass(frozen=True, eq=False)  # holds a table: compared by identity
class _Slice:
    """One maturity's SVI fit and the checks of its quotes (stage 2 of :func:`fit_market_surface`)."""

    T: float
    svi: SVISlice
    quoted: tuple[float, float]  # the range of forward log-moneyness of its quotes
    rmse: float  # implied-vol RMSE of the fit at the quotes
    binned: bool  # whether dense quotes were averaged into MAX_POINTS_PER_SLICE bins before the fit
    spread_arbitrage: bool  # see _spread_arbitrage
    butterflies: tuple[int, tuple[float, float, float] | None]  # see _butterfly_arbitrage
    certain: bool  # whether arbitrage found in quoted vols is beyond what rounding their prices could cause
    quotes: pd.DataFrame  # its rows of the cleaned quotes, with their K/S (moneyness) and fitted vol (fitted_iv)


def _prepare(quotes: pd.DataFrame, spot: float, carry: float) -> tuple[list[_Prepared], list[str]]:
    """Stage 1 of :func:`fit_market_surface`: each maturity's forward log-moneyness and vols, and those the SVI fit is
    given (dense quotes averaged into :data:`MAX_POINTS_PER_SLICE` bins, :func:`_binned`); a maturity with fewer than
    :data:`MIN_STRIKES_PER_SLICE` strikes is skipped with a note. Returns ``(prepared, notes)``."""
    prepared, notes = [], []
    for maturity, grp in quotes.groupby("maturity"):
        T = float(cast(float, maturity))
        if grp["strike"].nunique() < MIN_STRIKES_PER_SLICE:
            notes.append(f"Skipped maturity {T:.4g}y: fewer than {MIN_STRIKES_PER_SLICE} strikes.")
            continue
        y = np.log(grp["strike"].to_numpy(float) / spot) - carry * T
        iv = grp["iv"].to_numpy(float)
        y_fit, iv_fit = (y, iv) if len(y) <= MAX_POINTS_PER_SLICE else _binned(y, iv)
        prepared.append(_Prepared(T, grp, y, iv, y_fit, iv_fit))
    return prepared, notes


def _fit_slices(prepared: list[_Prepared], spot: float, rate: float, carry: float) -> list[_Slice]:
    """Stage 2 of :func:`fit_market_surface`: the raw-SVI fit of each maturity (:func:`fit_svi`), its RMSE at the
    quotes, and the model-free static-arbitrage checks of the quotes (:func:`_quote_arbitrage`)."""
    slices = []
    for piece in prepared:
        T, y, iv = piece.T, piece.y, piece.iv
        svi = fit_svi(piece.y_fit, piece.iv_fit, T)
        fitted = np.sqrt(np.maximum(svi.total_variance(y)[0], 0.0) / T)
        rmse = float(np.sqrt(np.mean((fitted - iv) ** 2)))
        spread, butterflies, certain = _quote_arbitrage(piece.quotes, iv, T, spot * np.exp(carry * T), rate)
        with_fit = piece.quotes.assign(moneyness=piece.quotes["strike"] / spot, fitted_iv=fitted)
        binned = len(piece.y_fit) < len(y)
        slices.append(_Slice(T, svi, (y.min(), y.max()), rmse, binned, spread, butterflies, certain, with_fit))
    return slices


def _quote_arbitrage(
    grp: pd.DataFrame, iv: np.ndarray, T: float, forward: float, rate: float
) -> tuple[bool, tuple[int, tuple[float, float, float] | None], bool]:
    """Vertical-spread and butterfly arbitrage in one maturity's quotes (:func:`_spread_arbitrage`,
    :func:`_butterfly_arbitrage`), and whether it is beyond what rounding the prices behind quoted vols to
    :data:`QUOTED_VOL_TICK` ticks could cause."""
    # Each quote's price error, undiscounted like the Black prices it widens: its rounding or half its bid/ask
    # spread (see parse_quotes), or for an implied vol without a price UNKNOWN_PRICE_ERROR of the forward or
    # MIN_PRICE_ERROR, whichever is more.
    known = grp["price_error"].to_numpy(float) if "price_error" in grp else np.full(len(grp), np.nan)
    unknown = ~np.isfinite(known)
    growth = np.exp(rate * T)
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
    slices: list[_Slice], Ts: np.ndarray, quoted_y: np.ndarray, carry: float, n_moneyness: int
) -> tuple[np.ndarray, _MarketValues]:
    """Stage 3 of :func:`fit_market_surface`: the K/S grid (:func:`_moneyness_grid`), and implied and local vol on it
    (:func:`_market_values`)."""
    moneyness = _moneyness_grid(np.exp(quoted_y + carry * Ts[:, None]), n_moneyness)
    return moneyness, _market_values([piece.svi for piece in slices], Ts, quoted_y, carry, moneyness)


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
        rows.append(row)
    return pd.DataFrame(rows)
