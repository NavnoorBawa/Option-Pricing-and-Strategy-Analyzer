"""The SSVI surface (Gatheral & Jacquier, 2014) and a sample quotes table drawn from it.

:class:`SSVIParams` holds a power-law curvature function and a mean-reverting ATM variance term structure, with the
analytic derivatives of total variance that local volatility needs; it is free of static arbitrage when
``eta (1 + |rho|) <= 2``. :func:`sample_quotes` is the example table of the quote upload (SSVI plus small noise).
:func:`~options_analyzer.vol_surface.ssvi_surface` evaluates the surface on a grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import black_scholes as bs


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
