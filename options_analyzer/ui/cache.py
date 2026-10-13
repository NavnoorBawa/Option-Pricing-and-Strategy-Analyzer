"""Cached wrappers around the heavier computations.

Every function is deterministic in its arguments (Monte Carlo uses explicit
seeds), so ``st.cache_data`` can safely memoise results across reruns.
``max_entries`` bounds memory on shared deployments.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from .. import binomial, implied_vol, monte_carlo, risk, vol_surface
from ..strategies import Market, Strategy


@st.cache_data(show_spinner=False, max_entries=64)
def tree_results(S, K, T, r, sigma, q, steps, american, method) -> dict:
    return {
        kind: binomial.price_with_greeks(S, K, T, r, sigma, q, kind, steps, american, method)
        for kind in ("call", "put")
    }


@st.cache_data(show_spinner=False, max_entries=32)
def tree_convergence(S, K, T, r, sigma, q, max_steps, american) -> tuple[np.ndarray, dict]:
    steps = np.unique(np.geomspace(10, min(max(max_steps, 20), 1000), 40).astype(int))

    def safe_price(n, method):
        try:
            return binomial.price(S, K, T, r, sigma, q, "call", int(n), american, method)
        except ValueError:  # CRR is not arbitrage-free at this step count
            return np.nan

    series = {label: [safe_price(n, method) for n in steps] for method, label in binomial.METHODS.items()}
    return steps, series


@st.cache_data(show_spinner=False, max_entries=32)
def mc_results(S, K, T, r, sigma, q, n_paths, seed, antithetic, control_variate) -> dict:
    return monte_carlo.price_european(S, K, T, r, sigma, q, ("call", "put"), n_paths, seed, antithetic, control_variate)


@st.cache_data(show_spinner=False, max_entries=16)
def mc_paths(S, T, r, sigma, q, seed) -> tuple[np.ndarray, np.ndarray]:
    steps = int(np.clip(round(T * 252), 20, 500))
    return monte_carlo.simulate_paths(S, T, r, sigma, q, n_paths=200, n_steps=steps, seed=seed)


@st.cache_data(show_spinner=False, max_entries=16)
def mc_terminal(S, T, r, sigma, q, n_paths, seed, antithetic) -> np.ndarray:
    return monte_carlo.terminal_prices(S, T, r, sigma, q, min(n_paths, 200_000), seed, antithetic)


@st.cache_data(show_spinner=False, max_entries=12)
def value_at_risk(strategy: Strategy, market: Market, config: risk.VaRConfig, multiplier: float) -> risk.VaRResult:
    return risk.value_at_risk(strategy, market, config, multiplier)


@st.cache_data(show_spinner=False, max_entries=16)
def ssvi_surface(params: vol_surface.SSVIParams, rate, dividend_yield, m_range, t_range) -> vol_surface.VolSurface:
    moneyness = np.linspace(m_range[0], m_range[1], 41)
    maturities = np.linspace(t_range[0], t_range[1], 30)
    return vol_surface.ssvi_surface(params, rate, dividend_yield, moneyness, maturities)


@st.cache_data(show_spinner=False, max_entries=8)
def market_surface(raw, spot, rate, dividend_yield) -> tuple[vol_surface.VolSurface | None, list[str], str | None]:
    """Clean an uploaded quotes table (inverting any prices) and fit it.

    Returns ``(surface, messages, error)`` so the parse messages survive a fit that fails.
    """
    quotes, messages = vol_surface.parse_quotes(raw, spot, rate, dividend_yield)
    try:
        return vol_surface.fit_market_surface(quotes, spot, rate, dividend_yield), messages, None
    except vol_surface.QuoteParseError as exc:
        return None, messages, str(exc)


@st.cache_data(show_spinner=False, max_entries=64)
def american_value(S, K, T, r, sigma, q, option_type) -> float:
    return binomial.price(S, K, T, r, sigma, q, option_type, 501, american=True, method="lr")


@st.cache_data(show_spinner=False, max_entries=64)
def american_iv(price, S, K, T, r, q, option_type) -> float:
    return implied_vol.implied_volatility_american(price, S, K, T, r, q, option_type)


@st.cache_data(show_spinner=False, max_entries=16)
def american_curve(S, K, T, r, q, option_type, max_vol) -> tuple[np.ndarray, np.ndarray]:
    vols = np.linspace(0.005, max_vol, 41)
    return vols, np.array([american_value(S, K, T, r, v, q, option_type) for v in vols])
