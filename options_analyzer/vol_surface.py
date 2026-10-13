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
  inside, the one-sided secant at the first and last maturity. A secant that
  is negative beyond quote precision is calendar arbitrage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import black_scholes as bs
from .implied_vol import ImpliedVolError, implied_volatility

__all__ = [
    "QuoteParseError",
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


def _svi_start_points(y, iv, T, n_best: int = 3) -> list[np.ndarray]:
    """Quasi-explicit starting points: for fixed ``(m, s)`` SVI is linear in ``(a, b s, rho b s)``."""
    w = iv * iv * T
    weight = 1.0 / (2.0 * iv * T)  # d(iv) = dw / (2 iv T): weighted variance residuals approximate vol residuals
    lo, hi = y.min(), y.max()
    width = max(hi - lo, 1e-4)
    m_grid = np.linspace(lo - 0.5 * width, hi + 0.5 * width, 25)
    s_grid = width * np.geomspace(0.003, 2.0, 24)
    m_all, s_all = (g.ravel() for g in np.meshgrid(m_grid, s_grid, indexing="ij"))
    z = (y[None, :] - m_all[:, None]) / s_all[:, None]
    design = np.stack([np.ones_like(z), z, np.sqrt(z * z + 1.0)], axis=-1) * weight[None, :, None]
    normal = np.einsum("gni,gnj->gij", design, design) + 1e-14 * np.eye(3)
    rhs = np.einsum("gni,n->gi", design, w * weight)
    a, d, c = np.linalg.solve(normal, rhs[..., None])[..., 0].T
    c = np.maximum(c, 1e-10)
    rho = np.clip(d / c, -0.99, 0.99)
    b = np.minimum(c / s_all, 2.0 / (1.0 + np.abs(rho)))
    a = np.maximum(a, -b * s_all * np.sqrt(1.0 - rho * rho))  # keep the minimum variance non-negative
    params = np.column_stack([a, b, rho, m_all, s_all])
    root = np.sqrt((y[None, :] - m_all[:, None]) ** 2 + s_all[:, None] ** 2)
    fitted = a[:, None] + b[:, None] * (rho[:, None] * (y[None, :] - m_all[:, None]) + root)
    cost = np.sum(((fitted - w[None, :]) * weight[None, :]) ** 2, axis=1)
    return [params[i] for i in np.argsort(cost)[:n_best]]


QUOTE_PRECISION = 1e-4  # implied-vol precision of a quote: differences below it are noise
_RHO_MAX = 0.999
_LOG_S = (-25.0, 5.0)


def fit_svi(y, iv, T: float) -> SVISlice:
    """Least-squares raw-SVI fit to implied vols ``iv`` at forward log-moneyness ``y`` and maturity ``T``.

    The residuals are implied-vol errors. ``b >= 0``, ``|rho| < 1``, ``s > 0``
    and a vertex ``m`` within a few quoted ranges of the quotes hold by
    construction (``b = beta^2``, ``rho = 0.999 tanh(r)``, ``s = e^sigma``,
    ``m = centre + span tanh(mu)``); a negative minimum variance and a breach of
    Lee's wing bound ``b (1 + |rho|) <= 2`` are penalised. Several quasi-explicit
    starting points are refined and the best fit is kept. Noise-free SVI (hence
    SSVI) quotes are recovered to solver precision.
    """
    y = np.asarray(y, dtype=float)
    iv = np.asarray(iv, dtype=float)
    lo, hi = y.min(), y.max()
    centre, span = 0.5 * (lo + hi), 3.5 * max(hi - lo, 1e-4)
    penalty = 10.0 * np.sqrt(len(y))
    vega = 1.0 / (2.0 * np.sqrt(np.mean(iv * iv * T) * T))  # variance -> vol units for the penalties

    def unpack(x):
        a, beta, r, mu, log_s = x
        return a, beta * beta, _RHO_MAX * np.tanh(r), centre + span * np.tanh(mu), np.exp(np.clip(log_s, *_LOG_S))

    def residuals(x):
        a, b, rho, m, s = unpack(x)
        z = y - m
        root = np.sqrt(z * z + s * s)
        w = np.maximum(a + b * (rho * z + root), 1e-300)
        wmin = a + b * s * np.sqrt(1.0 - rho * rho)
        lee = b * (1.0 + abs(rho)) - 2.0
        return np.concatenate([np.sqrt(w / T) - iv, [penalty * vega * max(-wmin, 0.0), penalty * max(lee, 0.0)]])

    def jacobian(x):
        _, beta, r, mu, log_s = x
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
            2.0 * beta,
            _RHO_MAX * (1.0 - np.tanh(r) ** 2),
            span * (1.0 - np.tanh(mu) ** 2),
            s if _LOG_S[0] < log_s < _LOG_S[1] else 0.0,
        ]
        return jac * np.array(chain)[None, :]

    def solve(x0, budget):
        return least_squares(
            residuals, x0, jac=jacobian, method="lm", x_scale="jac", ftol=1e-10, xtol=1e-12, gtol=1e-12, max_nfev=budget
        )

    starts = [
        np.array(
            [
                a,
                np.sqrt(b),
                np.arctanh(np.clip(rho / _RHO_MAX, -0.999, 0.999)),
                np.arctanh(np.clip((m - centre) / span, -0.999, 0.999)),
                np.log(s),
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
        and only where both slices were quoted at that strike.
        """
        T, Ts = float(maturity), self.maturities
        if not (moneyness > 0 and Ts[0] - 1e-9 <= T <= Ts[-1] + 1e-9):
            return np.nan
        y = np.log(moneyness) - self.carry * T
        if self.ssvi is not None:
            if not self.moneyness[0] <= moneyness <= self.moneyness[-1]:
                return np.nan
            return float(np.sqrt(self.ssvi.total_variance(y, T)[0] / T))
        if not self.svi or self.quoted_y is None:
            return np.nan
        same = np.flatnonzero(np.abs(Ts - T) <= 1e-9)  # quoted maturities are rounded to 10 decimals
        j = int(same[0]) if same.size else int(np.searchsorted(Ts, T)) - 1
        around = [j] if same.size else [j, j + 1]
        if any(not self.quoted_y[k, 0] - 1e-9 <= y <= self.quoted_y[k, 1] + 1e-9 for k in around):
            return np.nan
        variances = [float(self.svi[k].total_variance(y)[0]) for k in around]
        w = (
            variances[0]
            if same.size
            else variances[0] + (variances[1] - variances[0]) * (T - Ts[j]) / (Ts[j + 1] - Ts[j])
        )
        return float(np.sqrt(w / T)) if w > 0 else np.nan


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
}
_UNIT_SUFFIXES = (("percent", "percent"), ("pct", "percent"), ("years", "years"), ("yrs", "years"), ("days", "days"))
_BRACKETS = re.compile(r"[(\[{]([^)\]}]*)[)\]}]")


def _header(name) -> tuple[str, str | None]:
    """A column name reduced to lowercase letters and digits, plus the unit it states (percent, days or years).

    Bracketed units are removed but read (``"IV (%)"`` -> ``("iv", "percent")``), as
    are a ``%`` sign and a trailing unit word (``"Maturity_years"`` -> ``("maturity", "years")``).
    """
    text = str(name).strip().lower()
    unit = None
    for inner in _BRACKETS.findall(text):
        unit = _UNITS.get(re.sub(r"[^a-z%]", "", inner), unit)
    text = _BRACKETS.sub(" ", text)
    if "%" in text:
        unit = "percent"
    base = re.sub(r"[^a-z0-9]", "", text)
    for suffix, suffix_unit in _UNIT_SUFFIXES:
        if base.endswith(suffix) and len(base) > len(suffix):
            base, unit = base[: -len(suffix)], unit or suffix_unit
            break
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


def _match_columns(frame: pd.DataFrame) -> dict[str, tuple[str, str | None]]:
    headers = {column: _header(column) for column in frame.columns}
    found: dict[str, tuple[str, str | None]] = {}
    claimed: set = set()
    for key, aliases in _ALIAS_KEYS.items():
        for alias in aliases:
            column = next((c for c, (base, _) in headers.items() if base == alias and c not in claimed), None)
            if column is not None:
                found[key] = (column, headers[column][1])
                claimed.add(column)
                break
    if "maturity" in found and found["maturity"][1] == "days":  # e.g. "Maturity (days)"
        days = found.pop("maturity")
        found.setdefault("days", days)
    return found


def _numeric(frame: pd.DataFrame, column) -> tuple[pd.Series, bool]:
    """A column as floats, with %, thousands separators, currency signs and spaces stripped.

    Also returns whether any value carried a ``%`` sign. A column with no
    numeric value at all raises :class:`QuoteParseError` naming it.
    """
    raw = frame[column]
    percent = False
    if pd.api.types.is_numeric_dtype(raw) and not pd.api.types.is_bool_dtype(raw):
        values = raw.astype(float)
    else:
        text = raw.astype("string").str.strip()
        percent = bool(text.str.endswith("%").fillna(False).any())
        values = pd.to_numeric(text.str.replace(r"[,\s$€£%]", "", regex=True), errors="coerce").astype(float)
    if not values.notna().any():
        sample = raw.dropna()
        example = f" (e.g. '{sample.iloc[0]}')" if len(sample) else ""
        raise QuoteParseError(f"column '{column}' has no numeric values{example}")
    return values, percent


def _option_kind(frame: pd.DataFrame, column) -> pd.Series:
    first = frame[column].astype("string").str.strip().str.lower().str[:1]
    return first.map({"c": "call", "p": "put"})


def _prices_to_vols(
    data: pd.DataFrame, spot: float, rate: float, dividend_yield: float, messages: list[str]
) -> pd.DataFrame:
    """Invert option prices into Black-Scholes-Merton implied vols, keeping the out-of-the-money side."""
    forward = spot * np.exp((rate - dividend_yield) * data["maturity"])
    otm = (data["type"] == "call") == (data["strike"] >= forward)
    both = data.groupby(["maturity", "strike"])["type"].transform("nunique") > 1
    if (both & ~otm).any():
        messages.append(
            f"Used the out-of-the-money option at {int((both & otm).sum())} strike(s) quoted for both calls and puts."
        )
        data = data[~both | otm]
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


def parse_quotes(
    frame: pd.DataFrame, spot: float | None = None, rate: float = 0.0, dividend_yield: float = 0.0
) -> tuple[pd.DataFrame, list[str]]:
    """Validate and clean a quotes table.

    Accepts (case, spacing and punctuation ignored; common aliases allowed) a
    ``Strike`` column, a ``Maturity`` column in years or a ``Days`` column, and
    either an ``ImpliedVol`` column (decimal or percent) or option prices
    (``Price``/``Mid``, or ``Bid`` and ``Ask``) with a call/put ``Type`` column.
    Prices are inverted with Black-Scholes-Merton at ``spot``, ``rate`` and
    ``dividend_yield``. Units in brackets are read (``IV (%)``, ``Maturity (days)``),
    values may carry ``%`` signs and thousands separators, and a maturity column
    with values above 30 is taken to be in days.

    Returns ``(clean, messages)`` where ``clean`` has columns ``strike``,
    ``maturity`` (years), ``iv`` (decimal) and ``messages`` describes every
    interpretation and adjustment. Raises :class:`QuoteParseError` if the data is unusable.
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
    data = pd.DataFrame({"strike": _numeric(frame, found["strike"][0])[0]})
    used = [f"strike from '{found['strike'][0]}'"]
    reported = pd.Series(False, index=data.index)  # rows already dropped with a specific message
    if "maturity" in found:
        column = found["maturity"][0]
        data["maturity"] = _numeric(frame, column)[0]
        largest = data["maturity"].max()
        if largest > 30:
            data["maturity"] = data["maturity"] / 365.0
            messages.append(
                f"Maturities in '{column}' go up to {largest:g}, which only makes sense in days; converted days "
                "to years (ACT/365). Name the column 'Days' to make this explicit."
            )
            used.append(f"maturity from '{column}' (days)")
        else:
            used.append(f"maturity from '{column}' (years)")
    else:
        column = found["days"][0]
        data["maturity"] = _numeric(frame, column)[0] / 365.0
        used.append(f"maturity from '{column}' (days, converted to years with ACT/365)")

    iv_percent = None
    if "iv" in found:
        column, unit = found["iv"]
        data["iv"], percent_signs = _numeric(frame, column)
        iv_percent = "header" if unit == "percent" else "values" if percent_signs else None
        used.append(f"implied vol from '{column}'")
    else:
        if "type" not in found:
            raise QuoteParseError(
                "option prices need a Type column saying call or put for each row; "
                f"found columns: {', '.join(map(str, frame.columns))}"
            )
        if spot is None:
            raise QuoteParseError("option prices need the spot price, rate and dividend yield to be inverted")
        data["type"] = _option_kind(frame, found["type"][0])
        if "price" in found:
            data["price"] = _numeric(frame, found["price"][0])[0]
            used.append(f"option price from '{found['price'][0]}'")
        elif "bid" in found and "ask" in found:
            bid = _numeric(frame, found["bid"][0])[0]
            ask = _numeric(frame, found["ask"][0])[0]
            unusable = (bid <= 0) | (ask < bid)
            if unusable.any():
                messages.append(f"Dropped {int(unusable.sum())} quote(s) with a zero bid or an ask below the bid.")
            reported |= unusable
            data["price"] = ((bid + ask) / 2).where(~unusable)
            used.append(f"option price as the bid/ask midpoint of '{found['bid'][0]}' and '{found['ask'][0]}'")
        else:
            data["price"] = _numeric(frame, found["last"][0])[0]
            used.append(f"option price from '{found['last'][0]}' (last trade)")
        used.append(f"call/put from '{found['type'][0]}'")
        bad_type = data["type"].isna() & frame[found["type"][0]].notna()
        if bad_type.any():
            messages.append(f"Dropped {int(bad_type.sum())} row(s) whose type is neither call nor put.")
        reported |= bad_type
    messages.insert(0, "Read " + ", ".join(used) + ".")

    value = "iv" if "iv" in data else "price"
    valid = data.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
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
                f"A median implied vol of {median:.4g} is ambiguous (it could be {median:.4g}% or {median:.0%}); "
                "name the column 'IV (%)' if the values are percentages."
            )
    if (data["iv"] > 5.0).any():
        n_bad = int((data["iv"] > 5.0).sum())
        data = data[data["iv"] <= 5.0]
        messages.append(f"Dropped {n_bad} quote(s) with implied vol above 500%.")
    if data.empty:
        raise QuoteParseError("no valid rows remain after cleaning")

    data = data.assign(maturity=data["maturity"].round(10))
    duplicated = data.duplicated(["maturity", "strike"], keep=False)
    if duplicated.any():
        messages.append(f"Averaged {int(duplicated.sum())} duplicate (maturity, strike) quotes.")
        data = data.groupby(["maturity", "strike"], as_index=False)["iv"].mean()
    return data.sort_values(["maturity", "strike"]).reset_index(drop=True)[["strike", "maturity", "iv"]], messages


# ---------------------------------------------------------------------------
# Market quotes: fitting
# ---------------------------------------------------------------------------

MIN_STRIKES_PER_SLICE = 5
MAX_POINTS_PER_SLICE = 200  # denser slices are averaged into this many bins in y before the SVI fit
EXTRAPOLATION = 0.25  # a neighbour may be evaluated this fraction of its quoted y-range beyond its quotes
POOR_FIT_RMSE = 0.01


def _binned(y: np.ndarray, iv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(y.min(), y.max(), MAX_POINTS_PER_SLICE + 1)
    which = np.clip(np.searchsorted(edges, y, side="right") - 1, 0, MAX_POINTS_PER_SLICE - 1)
    counts = np.bincount(which, minlength=MAX_POINTS_PER_SLICE)
    keep = counts > 0
    return (
        np.bincount(which, y, MAX_POINTS_PER_SLICE)[keep] / counts[keep],
        np.bincount(which, iv, MAX_POINTS_PER_SLICE)[keep] / counts[keep],
    )


def _time_derivative(w, cover, Ts, tol):
    """``dw/dT`` at fixed ``y`` from secants between neighbouring maturities, and the calendar-arbitrage mask.

    ``w[k]`` and ``cover[k]`` hold, for every row ``i`` and grid point, slice ``k``
    evaluated at row ``i``'s ``y`` (axis 1) and whether that is close enough to slice
    ``k``'s quotes. Returns ``(w_t, calendar_bad, available)`` on the ``(n_T, n_y)`` grid.
    """
    n = len(Ts)
    h = np.diff(Ts)
    shape = w.shape[1:]
    w_t = np.full(shape, np.nan)
    bad = np.zeros(shape, dtype=bool)
    available = np.zeros(shape, dtype=bool)
    for i in range(n):
        sides = []  # (secant, h, covered, negative)
        for j, step in ((i - 1, h[i - 1] if i > 0 else None), (i + 1, h[i] if i < n - 1 else None)):
            if step is None:
                continue
            lo, hi = sorted((i, j))
            diff = w[hi, i] - w[lo, i]
            covered = cover[j, i]
            negative = covered & (diff < -(tol[lo, i] + tol[hi, i]))
            sides.append((np.maximum(diff, 0.0) / step, step, covered, negative))
        if len(sides) == 1:
            d, _, covered, negative = sides[0]
        else:
            (dl, hl, cl, nl), (dr, hr, cr, nr) = sides
            covered, negative = cl & cr, nl | nr
            with np.errstate(divide="ignore", invalid="ignore"):
                harmonic = (3 * hl + 3 * hr) / ((2 * hr + hl) / dl + (hr + 2 * hl) / dr)  # Fritsch-Butland
            d = np.where((dl > 0) & (dr > 0), harmonic, (hr * dl + hl * dr) / (hl + hr))
        w_t[i] = d
        available[i] = covered
        bad[i] = negative | (covered & ~(d > 0))
    return w_t, bad, available


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
    on a K/S grid spanning the union of the quoted strike ranges. Implied vol is
    NaN outside each maturity's own quoted range; local vol is also NaN where
    ``dw/dT`` would need a neighbouring maturity far outside its quotes, where
    the surface has calendar or butterfly arbitrage, or where it is otherwise undefined.
    """
    notes: list[str] = []
    carry = rate - dividend_yield
    Ts, fits, quoted, rmse, used, binned = [], [], [], [], [], []
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
        used.append(grp.assign(moneyness=grp["strike"] / spot, fitted_iv=fitted))
    if len(fits) < 2:
        raise QuoteParseError(
            f"local volatility needs at least two maturities with {MIN_STRIKES_PER_SLICE}+ strikes each"
        )
    if binned:
        notes.append(
            f"Averaged dense quotes into at most {MAX_POINTS_PER_SLICE} points per maturity before fitting "
            f"({len(binned)} maturit{'y' if len(binned) == 1 else 'ies'})."
        )
    Ts = np.array(Ts)
    quoted_y = np.array(quoted)
    rmse = np.array(rmse)
    strikes = pd.concat(used)["strike"]
    if not strikes.min() <= spot <= strikes.max() or not 0.5 <= strikes.median() / spot <= 2.0:
        notes.append(
            f"Quoted strikes run from {strikes.min():,.2f} to {strikes.max():,.2f}, far from the spot {spot:,.2f} "
            "used to read them; set the sidebar spot to the underlying price the quotes were taken at."
        )
    poor = Ts[rmse > POOR_FIT_RMSE]
    if poor.size:
        notes.append(
            f"The SVI fit misses the quotes by more than {POOR_FIT_RMSE * 100:.0f} vol point (RMSE) at maturities "
            f"{', '.join(f'{t:.3g}y' for t in poor)}; check those quotes."
        )

    m_lo = np.exp(quoted_y[:, 0] + carry * Ts).min()
    m_hi = np.exp(quoted_y[:, 1] + carry * Ts).max()
    moneyness = np.linspace(m_lo, m_hi, n_moneyness)
    y_grid = np.log(moneyness)[None, :] - carry * Ts[:, None]
    n_T = len(Ts)

    # every slice evaluated at every row's y: axis 0 = slice, axis 1 = row
    w_all = np.array([svi.total_variance(y_grid)[0] for svi in fits])
    lo, hi = quoted_y[:, 0][:, None, None], quoted_y[:, 1][:, None, None]
    inside_all = (y_grid[None] >= lo - 1e-9) & (y_grid[None] <= hi + 1e-9)
    allow = EXTRAPOLATION * (hi - lo)
    cover_all = (y_grid[None] >= lo - allow) & (y_grid[None] <= hi + allow)
    tol_all = 2.0 * np.sqrt(np.maximum(w_all, 0.0) * Ts[:, None, None]) * QUOTE_PRECISION

    rows = np.arange(n_T)
    w = w_all[rows, rows]
    inside = inside_all[rows, rows] & (w > 0)
    implied = np.where(inside, np.sqrt(np.maximum(w, 0.0) / Ts[:, None]), np.nan)
    w_y = np.array([svi.total_variance(y_grid[i])[1] for i, svi in enumerate(fits)])
    w_yy = np.array([svi.total_variance(y_grid[i])[2] for i, svi in enumerate(fits)])
    w_t, calendar_bad, available = _time_derivative(w_all, cover_all, Ts, tol_all)
    g = durrleman_g(y_grid, np.where(inside, w, 1.0), w_y, w_yy)
    calendar_bad &= inside
    butterfly_bad = inside & (g <= 0)
    ok = inside & available & ~calendar_bad & ~butterfly_bad
    with np.errstate(divide="ignore", invalid="ignore"):
        local = np.where(ok, np.sqrt(np.where(ok, w_t / g, 0.0)), np.nan)
    uncovered = int((inside & ~available).sum())
    if uncovered:
        notes.append(
            f"Local vol is left blank at {uncovered} grid point(s) where dw/dT would need a neighbouring maturity "
            "far outside its quoted strikes."
        )
    if (inside_all[rows, rows] & (w <= 0)).any():
        notes.append("The fitted surface has zero total variance at some quoted points; check the quotes.")

    clean_quotes = pd.concat(used).reset_index(drop=True)[["strike", "maturity", "iv", "moneyness", "fitted_iv"]]
    return VolSurface(
        moneyness,
        Ts,
        implied,
        local,
        int(calendar_bad.sum()),
        int(butterfly_bad.sum()),
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

    Values are NaN where the surface is undefined at the strikes they need
    (interpolation never bridges a gap or clamps to the nearest defined point).
    """
    rows = []
    m = surface.moneyness

    def at(level, values):
        return float(np.interp(level, m, values)) if m[0] <= level <= m[-1] else np.nan

    for i, (T, ivs, lvs) in enumerate(zip(surface.maturities, surface.implied_vol, surface.local_vol, strict=True)):
        atm, lo, hi = at(1.0, ivs), at(0.9, ivs), at(1.1, ivs)
        row = {
            "Maturity (years)": T,
            "ATM implied vol": atm,
            "ATM local vol": at(1.0, lvs),
            "Skew 90-110 (vol pts)": (lo - hi) * 100,
            "Butterfly (vol pts)": (0.5 * (lo + hi) - atm) * 100,
        }
        if surface.fit_rmse is not None:
            row["Fit RMSE (vol pts)"] = surface.fit_rmse[i] * 100
        rows.append(row)
    return pd.DataFrame(rows)
