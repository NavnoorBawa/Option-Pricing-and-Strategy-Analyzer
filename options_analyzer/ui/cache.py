"""Cached wrappers around the heavier computations.

Every function is deterministic in its arguments (Monte Carlo uses explicit
seeds), so ``st.cache_data`` can safely memoise results across reruns.
``max_entries`` bounds memory on shared deployments.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from .. import american, binomial, implied_vol, monte_carlo, risk, vol_surface
from ..strategies import Market, Strategy

#: Tree used for American quotes when the sidebar model is neither a binomial tree nor the American engine.
DEFAULT_TREE = (501, "lr")
KINDS = ("call", "put")


@st.cache_data(show_spinner=False, max_entries=64)
def alo_results(S, K, T, r, sigma, q, scheme) -> tuple[dict, dict]:
    """The American engine on the call and the put: ``(details, greeks)``, each keyed by option type, with
    ``american.price_details`` and the finite-difference ``american.greeks``."""
    details = {kind: american.price_details(S, K, T, r, sigma, q, option_type=kind, scheme=scheme) for kind in KINDS}
    greeks = {kind: american.greeks(S, K, T, r, sigma, q, option_type=kind, scheme=scheme) for kind in KINDS}
    return details, greeks


@st.cache_data(show_spinner=False, max_entries=32)
def alo_boundary(K, T, r, sigma, q, option_type, scheme) -> tuple[np.ndarray, np.ndarray]:
    return american.exercise_boundary(K, T, r, sigma, q, option_type=option_type, scheme=scheme)


@st.cache_data(show_spinner=False, max_entries=64)
def alo_value(S, K, T, r, sigma, q, option_type, scheme) -> float:
    return float(american.price(S, K, T, r, sigma, q, option_type=option_type, scheme=scheme))


@st.cache_data(show_spinner=False, max_entries=64)
def alo_iv(price, S, K, T, r, q, option_type, scheme) -> float:
    return implied_vol.implied_volatility_alo(price, S, K, T, r, q, option_type, scheme=scheme)


@st.cache_data(show_spinner=False, max_entries=16)
def alo_curve(S, K, T, r, q, option_type, max_vol, scheme) -> tuple[np.ndarray, np.ndarray]:
    vols = np.linspace(0.005, max_vol, 101)

    def value(vol: float) -> float:
        try:
            return float(american.price(S, K, T, r, float(vol), q, option_type=option_type, scheme=scheme))
        except ValueError:  # the fallback tree has no arbitrage-free lattice here: leave a gap in the curve
            return np.nan

    return vols, np.array([value(v) for v in vols])


@st.cache_data(show_spinner=False, max_entries=64)
def tree_results(S, K, T, r, sigma, q, steps, american, method) -> dict:
    return {
        kind: binomial.price_with_greeks(S, K, T, r, sigma, q, kind, steps, american, method)
        for kind in ("call", "put")
    }


@st.cache_data(show_spinner=False, max_entries=64)
def tree_prices(S, K, T, r, sigma, q, steps, american, method) -> dict:
    return {kind: binomial.price(S, K, T, r, sigma, q, kind, steps, american, method) for kind in ("call", "put")}


@st.cache_data(show_spinner=False, max_entries=32)
def tree_convergence(S, K, T, r, sigma, q, max_steps, american) -> tuple[dict, dict]:
    """Call and put values on both lattices for step counts up to ``max_steps``, and converged references.

    Returns ``(series, references)``: ``series[kind][lattice name] = (steps used, values)`` and
    ``references[kind]`` (closed form for European exercise, an extrapolated Leisen-Reimer tree for American).
    """
    max_steps = max(int(max_steps), 10)
    grid = np.unique(np.r_[np.geomspace(10, max_steps, 40).astype(int), max_steps])

    def safe_price(kind, n, method):
        try:
            return binomial.price(S, K, T, r, sigma, q, kind, int(n), american, method)
        except ValueError:  # CRR is not arbitrage-free at this step count
            return np.nan

    series = {}
    for kind in ("call", "put"):
        series[kind] = {}
        for method, name in binomial.METHODS.items():
            # Leisen-Reimer rounds an even step count up to the next odd one: plot it where it was evaluated.
            used = grid | 1 if method == "lr" else grid
            series[kind][name] = (used, np.array([safe_price(kind, n, method) for n in grid]))
    references = {
        kind: binomial.extrapolated_price(S, K, T, r, sigma, q, kind, american=american) for kind in ("call", "put")
    }
    return series, references


@st.cache_data(show_spinner=False, max_entries=32)
def mc_results(S, K, T, r, sigma, q, n_paths, seed, antithetic, control_variate) -> dict:
    return monte_carlo.price_european(S, K, T, r, sigma, q, ("call", "put"), n_paths, seed, antithetic, control_variate)


@st.cache_data(show_spinner=False, max_entries=16)
def mc_paths(S, T, r, sigma, q, n_paths, seed, antithetic, n_show) -> tuple[np.ndarray, np.ndarray]:
    steps = int(np.clip(round(T * 252), 20, 500))
    return monte_carlo.scenario_paths(S, T, r, sigma, q, n_paths, seed, antithetic, n_show, steps)


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
    except ValueError as exc:  # QuoteParseError, or a numerical failure of the fit (numpy's LinAlgError)
        return None, messages, str(exc)


@st.cache_data(show_spinner=False, max_entries=64)
def american_value(S, K, T, r, sigma, q, option_type, steps=DEFAULT_TREE[0], method=DEFAULT_TREE[1]) -> float:
    return binomial.price(S, K, T, r, sigma, q, option_type, steps, american=True, method=method)


@st.cache_data(show_spinner=False, max_entries=64)
def american_iv(price, S, K, T, r, q, option_type, steps=DEFAULT_TREE[0], method=DEFAULT_TREE[1]) -> float:
    return implied_vol.implied_volatility_american(price, S, K, T, r, q, option_type, steps=steps, method=method)


@st.cache_data(show_spinner=False, max_entries=16)
def american_curve(
    S, K, T, r, q, option_type, max_vol, steps=DEFAULT_TREE[0], method=DEFAULT_TREE[1]
) -> tuple[np.ndarray, np.ndarray]:
    lowest = implied_vol.lowest_tree_volatility(T, r, q, steps, method)
    vols = np.linspace(max(0.005, lowest), max(max_vol, 2 * lowest), 41)

    def value(vol: float) -> float:
        try:
            return american_value(S, K, T, r, vol, q, option_type, steps, method)
        except ValueError:  # a short tree cannot represent this volatility: leave a gap in the curve
            return np.nan

    return vols, np.array([value(v) for v in vols])
