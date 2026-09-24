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
* **market quotes** (strike, maturity, implied vol), smoothed slice by
  slice with a cubic smoothing spline in ``y`` (penalty chosen by
  generalised cross-validation) and interpolated in ``T`` with a
  monotone (PCHIP) interpolant of total variance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, make_smoothing_spline

__all__ = [
    "QuoteParseError",
    "SSVIParams",
    "VolSurface",
    "fit_market_surface",
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


@dataclass(frozen=True)
class VolSurface:
    """Implied and local volatility on a (maturity x moneyness) grid; moneyness = K / S."""

    moneyness: np.ndarray
    maturities: np.ndarray
    implied_vol: np.ndarray
    local_vol: np.ndarray
    calendar_violations: int
    butterfly_violations: int
    source: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def strikes_over_spot(self) -> np.ndarray:
        return self.moneyness

    def to_frames(self, spot: float) -> dict[str, pd.DataFrame]:
        """Tidy tables for export (strikes in currency units)."""
        index = pd.Index(np.round(self.maturities, 6), name="Maturity (years)")
        columns = pd.Index(np.round(self.moneyness * spot, 6), name="Strike")
        return {
            "ImpliedVol": pd.DataFrame(self.implied_vol, index=index, columns=columns),
            "LocalVol": pd.DataFrame(self.local_vol, index=index, columns=columns),
        }


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
    )


# ---------------------------------------------------------------------------
# Market quotes
# ---------------------------------------------------------------------------


class QuoteParseError(ValueError):
    """The uploaded quotes cannot be turned into a surface."""


_ALIASES = {
    "strike": ("strike", "k", "strike_price", "strikeprice"),
    "maturity": ("maturity", "t", "expiry_years", "years", "tenor", "time_to_maturity", "ttm"),
    "days": ("days", "dte", "days_to_expiry", "daystoexpiry", "days_to_maturity"),
    "iv": ("impliedvol", "implied_vol", "implied_volatility", "iv", "vol", "volatility", "impliedvolatility"),
}


def _normalise(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum() or ch == "_")


def parse_quotes(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate and clean a quotes table.

    Accepts columns (case-insensitive, common aliases allowed) ``Strike``,
    ``Maturity`` in years (or ``Days`` to expiry) and ``ImpliedVol`` as a
    decimal or in percent. Returns ``(clean, messages)`` where ``clean`` has
    columns ``strike``, ``maturity``, ``iv`` and ``messages`` describes every
    adjustment made. Raises :class:`QuoteParseError` if the data is unusable.
    """
    messages: list[str] = []
    lookup = {_normalise(c): c for c in frame.columns}
    found = {}
    for key, names in _ALIASES.items():
        for alias in names:
            if _normalise(alias) in lookup:
                found[key] = lookup[_normalise(alias)]
                break
    if "strike" not in found or "iv" not in found or ("maturity" not in found and "days" not in found):
        raise QuoteParseError(
            "the file needs a Strike column, a Maturity (years) or Days column and an ImpliedVol column; "
            f"found columns: {', '.join(map(str, frame.columns))}"
        )

    data = pd.DataFrame(
        {
            "strike": pd.to_numeric(frame[found["strike"]], errors="coerce"),
            "iv": pd.to_numeric(frame[found["iv"]], errors="coerce"),
        }
    )
    if "maturity" in found:
        data["maturity"] = pd.to_numeric(frame[found["maturity"]], errors="coerce")
    else:
        data["maturity"] = pd.to_numeric(frame[found["days"]], errors="coerce") / 365.0
        messages.append("Converted days to expiry into years (ACT/365).")

    n0 = len(data)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data[(data["strike"] > 0) & (data["maturity"] > 0) & (data["iv"] > 0)]
    if len(data) < n0:
        messages.append(f"Dropped {n0 - len(data)} row(s) with missing, non-numeric or non-positive values.")
    if data.empty:
        raise QuoteParseError("no valid rows remain after cleaning")

    if data["iv"].median() > 3.0:
        data["iv"] = data["iv"] / 100.0
        messages.append("Implied vols look like percentages; divided by 100.")
    if (data["iv"] > 5.0).any():
        n_bad = int((data["iv"] > 5.0).sum())
        data = data[data["iv"] <= 5.0]
        messages.append(f"Dropped {n_bad} quote(s) with implied vol above 500%.")

    data["maturity"] = data["maturity"].round(10)
    duplicated = data.duplicated(["maturity", "strike"], keep=False)
    if duplicated.any():
        messages.append(f"Averaged {int(duplicated.sum())} duplicate (maturity, strike) quotes.")
        data = data.groupby(["maturity", "strike"], as_index=False)["iv"].mean()
    return data.sort_values(["maturity", "strike"]).reset_index(drop=True)[["strike", "maturity", "iv"]], messages


MIN_STRIKES_PER_SLICE = 5


def fit_market_surface(
    quotes: pd.DataFrame,
    spot: float,
    rate: float,
    dividend_yield: float,
    n_moneyness: int = 41,
    smooth: bool = True,
) -> VolSurface:
    """Fit a surface to cleaned quotes (see :func:`parse_quotes`) and compute local vol.

    The result is evaluated at the quoted maturities only (no extrapolation in
    time) and on the moneyness range covered by every slice (no extrapolation
    in strike).
    """
    notes: list[str] = []
    slices = []
    for T, grp in quotes.groupby("maturity"):
        T = float(T)
        if grp["strike"].nunique() < MIN_STRIKES_PER_SLICE:
            notes.append(f"Skipped maturity {T:.4g}y: fewer than {MIN_STRIKES_PER_SLICE} strikes.")
            continue
        y = np.log(grp["strike"].to_numpy() / (spot * np.exp((rate - dividend_yield) * T)))
        w = grp["iv"].to_numpy() ** 2 * T
        order = np.argsort(y)
        y, w = y[order], w[order]
        spline = make_smoothing_spline(y, w, lam=None if smooth else 1e-12)
        slices.append((T, spline, y.min(), y.max()))
    if len(slices) < 2:
        raise QuoteParseError(
            f"local volatility needs at least two maturities with {MIN_STRIKES_PER_SLICE}+ strikes each"
        )

    Ts = np.array([s[0] for s in slices])
    carry = rate - dividend_yield
    y_lo = max(s[2] for s in slices)
    y_hi = min(s[3] for s in slices)
    m_lo = max(np.exp(y_lo + carry * T) for T in Ts)
    m_hi = min(np.exp(y_hi + carry * T) for T in Ts)
    if not m_hi > m_lo * 1.02:
        raise QuoteParseError("the maturities do not share a common strike range; add overlapping strikes")
    moneyness = np.linspace(m_lo, m_hi, n_moneyness)

    implied = np.empty((len(Ts), n_moneyness))
    w_y = np.empty_like(implied)
    w_yy = np.empty_like(implied)
    w_t = np.empty_like(implied)
    y_grid = np.log(moneyness)[None, :] - carry * Ts[:, None]
    for i, (T, spline, _, _) in enumerate(slices):
        y = y_grid[i]
        w = spline(y)
        implied[i] = np.sqrt(np.maximum(w, 1e-12) / T)
        w_y[i] = spline.derivative(1)(y)
        w_yy[i] = spline.derivative(2)(y)
        # dw/dT at fixed y: evaluate every slice at this y and differentiate a PCHIP in T.
        across = np.array([s[1](np.clip(y, s[2], s[3])) for s in slices])
        w_t[i] = PchipInterpolator(Ts, across, axis=0).derivative()(T)
    w_grid = implied**2 * Ts[:, None]
    if (w_grid <= 1e-10).any():
        notes.append("The smoothed surface has near-zero total variance at some points; check the quotes.")
    local, cal, fly = _local_vol(y_grid, w_grid, w_y, w_yy, w_t)
    return VolSurface(moneyness, Ts, implied, local, int(cal.sum()), int(fly.sum()), "Market quotes", tuple(notes))


def sample_quotes(spot: float = 100.0, rate: float = 0.04, dividend_yield: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """A realistic example quotes table (SSVI plus small noise) for the CSV template."""
    rng = np.random.default_rng(seed)
    params = SSVIParams()
    rows = []
    for days in (30, 60, 91, 182, 273, 365, 547, 730):
        T = days / 365.0
        for m in np.arange(0.70, 1.301, 0.05):
            y = np.log(m) - (rate - dividend_yield) * T
            w = params.total_variance(y, T)[0]
            iv = np.sqrt(w / T) + rng.normal(0.0, 0.002)
            rows.append({"Strike": round(spot * m, 2), "Maturity": round(T, 4), "ImpliedVol": round(float(iv), 4)})
    return pd.DataFrame(rows)


def surface_summary(surface: VolSurface) -> pd.DataFrame:
    """ATM level, 90/110 skew and smile curvature per maturity (vol points)."""
    rows = []
    m = surface.moneyness
    for T, ivs, lvs in zip(surface.maturities, surface.implied_vol, surface.local_vol, strict=True):
        atm = float(np.interp(1.0, m, ivs)) if m[0] <= 1.0 <= m[-1] else np.nan
        lo = float(np.interp(0.9, m, ivs)) if m[0] <= 0.9 else np.nan
        hi = float(np.interp(1.1, m, ivs)) if m[-1] >= 1.1 else np.nan
        finite = np.isfinite(lvs)
        atm_local = (
            float(np.interp(1.0, m[finite], lvs[finite])) if finite.sum() > 1 and m[0] <= 1.0 <= m[-1] else np.nan
        )
        rows.append(
            {
                "Maturity (years)": T,
                "ATM implied vol": atm,
                "ATM local vol": atm_local,
                "Skew 90-110 (vol pts)": (lo - hi) * 100 if np.isfinite(lo) and np.isfinite(hi) else np.nan,
                "Butterfly (vol pts)": (0.5 * (lo + hi) - atm) * 100 if np.isfinite(lo + hi + atm) else np.nan,
            }
        )
    return pd.DataFrame(rows)
