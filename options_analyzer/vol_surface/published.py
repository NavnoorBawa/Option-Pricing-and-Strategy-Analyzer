"""The published surface: what every tab reads from a fitted market surface and the chain it was fitted to.

Roadmap item SF-11 (core), UI spec 2.0, 2.7 item 6 and 3.4. A :class:`PublishedSurface` holds, for each listed expiry,
its forward, discount factor, settlement and maturity, and answers:

* the vol at a strike (:meth:`PublishedSurface.vol`), from the fitted surface, flagged where it is extrapolated
  beyond the expiry's quotes;
* the risk-neutral distribution of the underlying at the expiry: the CDF P(S_T <= K), its density and its inverse
  (:meth:`~PublishedSurface.cdf`, :meth:`~PublishedSurface.pdf`, :meth:`~PublishedSurface.quantile`), and the 68% and
  95% risk-neutral ranges (:meth:`~PublishedSurface.rn_range`);
* the chain's quote of a contract with its implied vols (:meth:`~PublishedSurface.quote`), and an expiry's listed
  strikes (:meth:`~PublishedSurface.strikes`);
* the listed strike whose own delta, at its own surface vol, is nearest a target (roadmap item RK-9a,
  :meth:`~PublishedSurface.delta_strike`).

The CDF is Breeden and Litzenberger's (1978): with C(K) = DF · Black(F, K, w(y)), the call value on the smile,
P(S_T <= K) = 1 + (dC/dK) / DF. Differentiating the smile analytically (y = ln K/F, w the total implied variance and
w' = dw/dy, both from the SVI slice):

    P(S_T <= K) = N(−d₂) + φ(d₂) · w' / (2√w),   d₂ = −y/√w − √w/2,

the flat-vol N(−d₂) plus the smile's slope term, which is what moves a 16Δ put's probability from 16.85% to 11.91%
under skew (UI spec 2.7 item 6). Its derivative is the density, Gatheral's g(y) · φ(d₂) / (K√w) with Durrleman's
g(y) >= 0 (no butterfly arbitrage) exactly where the density is non-negative. No finite difference is taken, so the
CDF carries no step-size error: it is exact for the fitted smile.

The class holds no Streamlit code, so the library and its tests use it without the app (``ui/surface_state.py``
builds and caches the app's one).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.special import ndtr

from ..lets_be_rational import implied_black_volatility
from .quotes import MATURITY_TOLERANCE
from .surface import VolSurface

__all__ = ["ONE_SD", "TWO_SD", "PublishedSurface", "Quote", "SurfaceVol"]

#: Probability inside one and two standard deviations of a normal: the "68%" and "95%" ranges, 68.27% and 95.45%.
ONE_SD = math.erf(1.0 / math.sqrt(2.0))
TWO_SD = math.erf(2.0 / math.sqrt(2.0))
_SQRT_2PI = math.sqrt(2.0 * math.pi)
#: The grid :meth:`PublishedSurface.quantile` brackets its roots on: ATM standard deviations of y either side of the
#: forward, widened until the CDF at its ends is within _TAIL of 0 and 1, and its points.
_GRID_SDS, _GRID_POINTS, _TAIL = 12.0, 801, 1e-12
#: Steps of :meth:`PublishedSurface.quantile`'s safeguarded Newton solver: 6 to 8 reach 1e-15 from the grid's
#: bracket; bisection alone would need about 40.
_SOLVER_STEPS = 50


class SurfaceVol(NamedTuple):
    """A vol from the published surface and whether it is extrapolated beyond the expiry's quotes (NaN beyond the
    surface's reach, :meth:`VolSurface.reach_at`)."""

    vol: float
    extrapolated: bool


class Quote(NamedTuple):
    """One contract's market in the chain ($ per share) and the Black-76 implied vols of its bid, ask and mid at
    the expiry's forward and discount factor. A zero bid (an offer only) has no bid or mid vol: NaN."""

    bid: float
    ask: float
    mid: float
    iv_bid: float
    iv_ask: float
    iv_mid: float


class _Expiry(NamedTuple):
    forward: float
    discount_factor: float
    maturity: float
    settlement: str
    lower: int  # the surface slices it is read from (-1: none)
    upper: int
    weight: float


def _date(value) -> date:
    """A calendar date from a date, datetime, pandas Timestamp or ISO text."""
    if isinstance(value, str):
        return date.fromisoformat(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.date()
    return value


@dataclass(frozen=True, eq=False)  # holds tables: compared by identity
class PublishedSurface:
    """The fitted surface of a chain, per listed expiry (see the module docstring). Build it with :meth:`build`.

    Attributes
    ----------
    surface : VolSurface
        The fitted market surface (SVI slices, forwards and discount factors).
    expiries : pandas.DataFrame
        One row per listed expiry in date order: ``expiry`` (date), ``root``, ``settlement`` (AM or PM),
        ``settles_at``, ``maturity`` (years to the settlement, minutes / 525,600), ``forward``, ``discount_factor``
        and the surface slices it is read from (``lower``, ``upper``, ``weight``: total variance
        ``(1 − weight) · w_lower + weight · w_upper`` at fixed y; one slice when the expiry was fitted itself).
    quotes : pandas.DataFrame
        The chain in the canonical schema with ``mid``, ``iv_bid``, ``iv_ask`` and ``iv_mid`` added.
    spot : float
        The underlying's level the surface is quoted against.
    as_of : datetime
        The valuation time.
    source : str
        Where the chain comes from, as labelled everywhere it shows (``Synthetic — not market data``).
    """

    surface: VolSurface
    expiries: pd.DataFrame
    quotes: pd.DataFrame
    spot: float
    as_of: datetime
    source: str
    _cache: dict = field(default_factory=dict, repr=False)

    # --- building ---------------------------------------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        surface: VolSurface,
        expiries: pd.DataFrame,
        chain: pd.DataFrame,
        *,
        spot: float,
        as_of: datetime,
        source: str,
    ) -> PublishedSurface:
        """Publish a fitted market surface with the chain it was fitted to.

        Parameters
        ----------
        surface : VolSurface
            A market surface (:func:`fit_market_surface`): it has SVI slices and per-maturity forwards.
        expiries : pandas.DataFrame
            One row per listed expiry: ``expiry``, ``root``, ``settlement``, ``settles_at``, ``maturity`` (years),
            ``forward`` and ``discount_factor`` (as the fit read them from put-call parity).
        chain : pandas.DataFrame
            The chain in the canonical schema (``market_data.CHAIN_COLUMNS``).
        spot : float
            The underlying's level.
        as_of : datetime
            The valuation time.
        source : str
            The chain's label (``synthetic.SYNTHETIC_LABEL`` for the synthetic chain).

        Returns
        -------
        PublishedSurface

        Raises
        ------
        ValueError
            If the surface has no SVI slices (an SSVI surface is not a market fit), or an expiry date is listed
            twice (two roots on one date: the expiry would be ambiguous).
        """
        if not surface.svi or surface.log_forward_ratio is None:
            raise ValueError("only a market surface (SVI slices with their forwards) can be published")
        table = expiries[["expiry", "root", "settlement", "settles_at", "maturity", "forward", "discount_factor"]]
        table = table.assign(expiry=[_date(d) for d in table["expiry"]]).sort_values("expiry", kind="stable")
        if table["expiry"].duplicated().any():
            raise ValueError("an expiry date is listed under two roots; publish one root per date")
        Ts = surface.maturities
        lower, upper, weight = [], [], []
        for T in table["maturity"].to_numpy(float):
            nearest = int(np.argmin(np.abs(Ts - T)))
            if abs(Ts[nearest] - T) <= MATURITY_TOLERANCE:  # fitted itself (as VolSurface.implied_vol_at snaps)
                lower.append(nearest), upper.append(nearest), weight.append(0.0)
            elif Ts[0] < T < Ts[-1]:  # between two fitted expiries: total variance linear in T at fixed y
                j = int(np.searchsorted(Ts, T)) - 1
                lower.append(j), upper.append(j + 1), weight.append(float((T - Ts[j]) / (Ts[j + 1] - Ts[j])))
            else:  # outside the fitted maturities: no smile
                lower.append(-1), upper.append(-1), weight.append(np.nan)
        table = table.assign(lower=lower, upper=upper, weight=weight).reset_index(drop=True)

        quotes = chain.copy()
        quotes["expiry"] = [_date(d) for d in quotes["expiry"]]
        rows = table.set_index("expiry")
        listed = quotes["expiry"].isin(rows.index)
        quotes = quotes[listed].reset_index(drop=True)
        forward = quotes["expiry"].map(rows["forward"]).to_numpy(float)
        discount = quotes["expiry"].map(rows["discount_factor"]).to_numpy(float)
        maturity = quotes["expiry"].map(rows["maturity"]).to_numpy(float)
        theta = np.where(quotes["type"].to_numpy() == "call", 1.0, -1.0)
        strikes = quotes["strike"].to_numpy(float)
        bid, ask = quotes["bid"].to_numpy(float), quotes["ask"].to_numpy(float)
        mid = 0.5 * (bid + ask)

        def implied(price: np.ndarray, quoted: np.ndarray) -> np.ndarray:
            """Black-76 vols of undiscounted prices; NaN where unquoted, at or below intrinsic, or at the maximum."""
            with np.errstate(all="ignore"):
                vol = np.asarray(implied_black_volatility(price / discount, forward, strikes, maturity, theta))
            return np.where(quoted & np.isfinite(vol) & (vol > 0), vol, np.nan)

        quotes["mid"] = mid
        quotes["iv_bid"] = implied(bid, bid > 0)
        quotes["iv_ask"] = implied(ask, ask > 0)
        quotes["iv_mid"] = implied(mid, bid > 0)  # a one-sided quote's mid is half the offer: no market vol
        return cls(surface, table, quotes, float(spot), as_of, source)

    # --- per expiry -------------------------------------------------------------------------------------------------

    def _row(self, expiry) -> _Expiry:
        """The expiry's row as plain numbers (a dictionary look-up: a pandas filter per call cost 0.2 ms)."""
        if "rows" not in self._cache:
            self._cache["rows"] = {
                row.expiry: _Expiry(
                    float(row.forward),
                    float(row.discount_factor),
                    float(row.maturity),
                    str(row.settlement),
                    int(row.lower),
                    int(row.upper),
                    float(row.weight),
                )
                for row in self.expiries.itertuples(index=False)
            }
        found = self._cache["rows"].get(_date(expiry))
        if found is None:
            raise KeyError(f"{_date(expiry)} is not a listed expiry")
        return found

    def forward(self, expiry) -> float:
        """The expiry's forward F (put-call parity, as fitted)."""
        return self._row(expiry).forward

    def discount_factor(self, expiry) -> float:
        """The expiry's discount factor DF to its settlement."""
        return self._row(expiry).discount_factor

    def maturity(self, expiry) -> float:
        """Years from the valuation time to the expiry's settlement (minutes / 525,600)."""
        return self._row(expiry).maturity

    def settlement(self, expiry) -> str:
        """``AM`` (at the open's settlement quotation) or ``PM`` (at the close)."""
        return self._row(expiry).settlement

    def strikes(self, expiry) -> np.ndarray:
        """The expiry's listed strikes, ascending."""
        key = ("strikes", _date(expiry))
        if key not in self._cache:
            self._row(expiry)
            mine = self.quotes["expiry"] == _date(expiry)
            self._cache[key] = np.unique(self.quotes.loc[mine, "strike"].to_numpy(float))
        return self._cache[key]

    # --- vols -------------------------------------------------------------------------------------------------------

    def vol(self, strike: float, expiry) -> SurfaceVol:
        """The surface's vol at ``strike`` on ``expiry`` (``VolSurface.implied_vol_at`` with extrapolation), and
        whether it is read beyond the expiry's quotes. NaN beyond the surface's reach."""
        T = self.maturity(expiry)
        moneyness = float(strike) / self.spot
        sigma = self.surface.implied_vol_at(moneyness, T, extrapolate=True)
        return SurfaceVol(float(sigma), bool(np.isfinite(sigma) and self.surface.is_extrapolated(moneyness, T)))

    def _variance(self, expiry, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Total implied variance ``w`` and its y-derivatives ``w'`` and ``w''`` at forward log-moneyness ``y``, from
        the expiry's SVI slice (or the two around it, linear in T). NaN for an expiry outside the fitted ones."""
        row = self._row(expiry)
        lo, hi, a = row.lower, row.upper, row.weight
        if lo < 0:
            nan = np.full(np.shape(y), np.nan)
            return nan, nan, nan
        w, w_y, w_yy = self.surface.svi[lo].total_variance(y)
        if hi != lo:
            w2, w2_y, w2_yy = self.surface.svi[hi].total_variance(y)
            w, w_y, w_yy = w + a * (w2 - w), w_y + a * (w2_y - w_y), w_yy + a * (w2_yy - w_yy)
        return w, w_y, w_yy

    # --- the risk-neutral distribution ------------------------------------------------------------------------------

    def _distribution(self, expiry, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(CDF, dCDF/dy)`` at forward log-moneyness ``y`` (see the module docstring)."""
        w, w_y, w_yy = self._variance(expiry, y)
        with np.errstate(all="ignore"):
            root = np.sqrt(w)
            d2 = -y / root - 0.5 * root
            phi = np.exp(-0.5 * d2 * d2) / _SQRT_2PI
            slope = phi * w_y / (2.0 * root)
            # Above the forward, one minus the survival N(d₂) − slope: rounding 1 − S is monotone in S, while the sum
            # N(−d₂) + slope near 1 wobbled by one unit in the last place (1.1e−16) between neighbouring strikes.
            cdf = np.where(y > 0, 1.0 - (ndtr(d2) - slope), ndtr(-d2) + slope)
            g = (1.0 - y * w_y / (2.0 * w)) ** 2 - 0.25 * w_y * w_y * (1.0 / w + 0.25) + 0.5 * w_yy
            return cdf, phi * g / root

    def cdf(self, strike, expiry) -> float | np.ndarray:
        """The risk-neutral probability P(S_T <= K) at the expiry, from the smile (Breeden-Litzenberger).

        Parameters
        ----------
        strike : float or array_like
            Strikes K; 0 for K <= 0.
        expiry : date
            A listed expiry.

        Returns
        -------
        float or numpy.ndarray
            The CDF, a float for a scalar strike.

        Raises
        ------
        KeyError
            If ``expiry`` is not listed.
        """
        K = np.asarray(strike, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.log(np.where(K > 0, K, np.nan) / self.forward(expiry))
        cdf, _ = self._distribution(expiry, y)
        cdf = np.where(K > 0, cdf, 0.0)
        return float(cdf) if cdf.ndim == 0 else cdf

    def pdf(self, strike, expiry) -> float | np.ndarray:
        """The risk-neutral density of S_T at K (per unit of K): dP(S_T <= K)/dK = g(y) φ(d₂) / (K√w)."""
        K = np.asarray(strike, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.log(np.where(K > 0, K, np.nan) / self.forward(expiry))
            _, density_y = self._distribution(expiry, y)
            density = np.where(K > 0, density_y / K, 0.0)
        return float(density) if density.ndim == 0 else density

    def _grid(self, expiry) -> tuple[np.ndarray, np.ndarray]:
        """The CDF tabulated in y for :meth:`quantile`'s brackets: from ±12 ATM standard deviations, each end pushed
        out (doubled, at most four times) until the CDF there is below :data:`_TAIL` from 0 or 1. SVI's wings are
        linear in y, so the left tail is fat: at a two-year expiry 12 standard deviations leave 1e−5 below."""
        key = ("grid", _date(expiry))
        if key not in self._cache:
            sd = math.sqrt(float(self._variance(expiry, np.zeros(1))[0][0]))
            lo, hi = -_GRID_SDS * sd, _GRID_SDS * sd
            for _ in range(4):
                ends = self._distribution(expiry, np.array([lo, hi]))[0]
                if ends[0] <= _TAIL and ends[1] >= 1.0 - _TAIL:
                    break
                lo, hi = lo * (2.0 if ends[0] > _TAIL else 1.0), hi * (2.0 if ends[1] < 1.0 - _TAIL else 1.0)
            y = np.linspace(lo, hi, _GRID_POINTS)
            cdf = self._distribution(expiry, y)[0]
            self._cache[key] = (y, np.maximum.accumulate(cdf))  # searchsorted needs it non-decreasing
        return self._cache[key]

    def quantile(self, p, expiry) -> float | np.ndarray:
        """The strike K with P(S_T <= K) = p at the expiry: the inverse of :meth:`cdf`.

        The CDF tabulated on ±12 ATM standard deviations of y brackets each root between two grid points; Newton
        steps in y with the density then converge, each step that would leave the bracket replaced by bisection
        (in the far tails the CDF changes by orders of magnitude between grid points, where plain Newton from an
        interpolated guess overshot to a strike whose CDF underflowed to 0). At most :data:`_SOLVER_STEPS` steps;
        the CDF at the result is within 1e-14 of ``p`` on the default chain.

        Parameters
        ----------
        p : float or array_like
            Probabilities in (0, 1).
        expiry : date
            A listed expiry.

        Returns
        -------
        float or numpy.ndarray
            Strikes; NaN for ``p`` outside (0, 1) or within 1e−12 of 0 or 1 (beyond the tabulated range).
        """
        probability = np.asarray(p, dtype=float)
        y_grid, cdf_grid = self._grid(expiry)
        inside = (probability > cdf_grid[0]) & (probability < cdf_grid[-1])
        target = np.where(inside, probability, 0.5)
        j = np.clip(np.searchsorted(cdf_grid, target), 1, len(y_grid) - 1)
        lo, hi = y_grid[j - 1], y_grid[j]
        y = 0.5 * (lo + hi)
        for _ in range(_SOLVER_STEPS):
            cdf, density_y = self._distribution(expiry, y)
            error = cdf - target
            if np.all(np.abs(error) <= 1e-15):
                break
            lo, hi = np.where(error < 0, y, lo), np.where(error > 0, y, hi)
            with np.errstate(divide="ignore", invalid="ignore"):
                step = y - error / density_y
            y = np.where((step > lo) & (step < hi), step, 0.5 * (lo + hi))
        K = np.where(inside, self.forward(expiry) * np.exp(y), np.nan)
        return float(K) if K.ndim == 0 else K

    def rn_range(self, expiry, level: float = ONE_SD) -> tuple[float, float]:
        """The central risk-neutral range of S_T holding probability ``level``: the quantiles at ``(1 − level)/2``
        and ``(1 + level)/2``. :data:`ONE_SD` (68.27%) and :data:`TWO_SD` (95.45%) are the chart's two ranges."""
        lo, hi = self.quantile([(1.0 - level) / 2.0, (1.0 + level) / 2.0], expiry)
        return float(lo), float(hi)

    # --- quotes and deltas ------------------------------------------------------------------------------------------

    def _quote_index(self) -> dict:
        if "quotes" not in self._cache:
            keys = zip(self.quotes["expiry"], self.quotes["strike"].astype(float), self.quotes["type"], strict=True)
            self._cache["quotes"] = {key: i for i, key in enumerate(keys)}
        return self._cache["quotes"]

    def quote(self, expiry, strike: float, kind: str) -> Quote:
        """The chain's quote of the ``kind`` (``"call"`` or ``"put"``) at ``strike`` on ``expiry``, with its vols.

        Raises
        ------
        KeyError
            If the contract is not listed.
        """
        i = self._quote_index().get((_date(expiry), float(strike), kind))
        if i is None:
            raise KeyError(f"no {kind} at {strike:g} on {_date(expiry)} in the chain")
        row = self.quotes.iloc[i]
        return Quote(*(float(row[c]) for c in ("bid", "ask", "mid", "iv_bid", "iv_ask", "iv_mid")))

    def delta(self, strike, expiry, kind: str) -> float | np.ndarray:
        """Spot delta e^{−qT}N(d₁) (a call) or −e^{−qT}N(−d₁) (a put) at each strike's own surface vol, with
        e^{−qT} = F · DF / S; NaN where the surface has no vol (beyond its reach)."""
        if kind not in ("call", "put"):
            raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
        K = np.atleast_1d(np.asarray(strike, dtype=float))
        T, F, DF = self.maturity(expiry), self.forward(expiry), self.discount_factor(expiry)
        sigma = np.array([self.vol(k, expiry).vol for k in K])
        with np.errstate(all="ignore"):
            root = sigma * math.sqrt(T)
            d1 = np.log(F / K) / root + 0.5 * root
        carry = F * DF / self.spot
        delta = carry * ndtr(d1) if kind == "call" else -carry * ndtr(-d1)
        return float(delta[0]) if np.ndim(strike) == 0 else delta

    def delta_strike(self, expiry, target: float, kind: str) -> float:
        """The listed strike whose own delta, at its own surface vol, is nearest ``target`` (roadmap item RK-9a).

        Parameters
        ----------
        expiry : date
            A listed expiry.
        target : float
            The delta wanted, spot convention: positive for a call (0.16), negative for a put (−0.16).
        kind : {"call", "put"}
            The option.

        Returns
        -------
        float
            The listed strike; among equally near ones, the lowest.

        Raises
        ------
        ValueError
            If the target's sign does not match ``kind`` or no listed strike has a vol.
        """
        if (kind == "call" and not 0 < target < 1) or (kind == "put" and not -1 < target < 0):
            raise ValueError(f"a {kind} delta lies in {'(0, 1)' if kind == 'call' else '(-1, 0)'}, got {target!r}")
        key = ("deltas", _date(expiry), kind)
        if key not in self._cache:
            strikes = self.strikes(expiry)
            self._cache[key] = (strikes, np.asarray(self.delta(strikes, expiry, kind)))
        strikes, deltas = self._cache[key]
        distance = np.abs(deltas - target)
        if not np.isfinite(distance).any():
            raise ValueError(f"no listed strike on {_date(expiry)} has a surface vol")
        return float(strikes[int(np.nanargmin(distance))])
