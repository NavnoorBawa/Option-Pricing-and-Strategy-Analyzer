"""Cached wrappers around the heavier computations.

Every function is deterministic in its arguments (Monte Carlo uses explicit
seeds), so ``st.cache_data`` can safely memoise results across reruns.
``max_entries`` bounds memory on shared deployments.
"""

from __future__ import annotations

import time

import numpy as np
import streamlit as st

from .. import american, binomial, implied_vol, monte_carlo, risk, vol_surface
from ..strategies import Market, Strategy

#: Tree used for American quotes when the sidebar model is neither a binomial tree nor the American engine.
DEFAULT_TREE = (501, "lr")
KINDS = ("call", "put")


@st.cache_data(show_spinner=False, max_entries=128)
def alo_option(S, K, T, r, sigma, q, option_type, scheme) -> tuple[american.AmericanPrice, american.AmericanGreeks]:
    """The American engine on one option: ``american.price_details`` and ``american.greeks``."""
    return (
        american.price_details(S, K, T, r, sigma, q, option_type=option_type, scheme=scheme),
        american.greeks(S, K, T, r, sigma, q, option_type=option_type, scheme=scheme),
    )


def alo_results(S, K, T, r, sigma, q, scheme) -> tuple[dict, dict]:
    """The American engine on the call and the put: ``(details, greeks)``, each keyed by option type (see
    :func:`alo_option`)."""
    results = {kind: alo_option(S, K, T, r, sigma, q, kind, scheme) for kind in KINDS}
    return {kind: res[0] for kind, res in results.items()}, {kind: res[1] for kind, res in results.items()}


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
def alo_curve(S, K, T, r, q, option_type, low, high, scheme) -> tuple[np.ndarray, np.ndarray]:
    """The engine's value at 41 vols from ``low`` to ``high``, as the tree's curve (:func:`american_curve`): a
    smooth curve needs no more, and where the engine falls back to the tree each point costs two trees (about
    0.1 s)."""
    vols = np.linspace(low, high, 41)

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
def tree_convergence(S, K, T, r, sigma, q, max_steps, american) -> dict:
    """Call and put values on both lattices for step counts up to ``max_steps``: ``series[kind][lattice name] =
    (steps used, values)``. The Pricing tab measures them against its engine table's reference (the closed form, or
    the American engine's high scheme), so no reference is computed here: the extrapolated Leisen-Reimer trees it
    once returned were thrown away, and took 0.19 s of the 0.39 s a new contract cost (S = K = 100, one year,
    American, 501 steps).
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
    return series


@st.cache_data(show_spinner=False, max_entries=32)
def mc_results(S, K, T, r, sigma, q, n_paths, seed, antithetic, control_variate) -> dict:
    return monte_carlo.price_european(S, K, T, r, sigma, q, ("call", "put"), n_paths, seed, antithetic, control_variate)


#: The smallest sample of the Monte Carlo convergence view (Engine diagnostics), and the factor between samples: a
#: quarter of the paths each step down doubles the interval, so the 1/sqrt(N) guide's slope reads at a glance.
MC_CONVERGENCE_FROM, MC_CONVERGENCE_STEP = 1_000, 4


def mc_convergence_paths(n_paths: int) -> list[int]:
    """The samples of the Monte Carlo convergence view: ``n_paths``, then a quarter of it each step down to
    :data:`MC_CONVERGENCE_FROM`, each even (antithetic pairs), in ascending order.

    >>> mc_convergence_paths(100_000)
    [1562, 6250, 25000, 100000]
    """
    paths, n = [], int(n_paths)
    while n >= MC_CONVERGENCE_FROM:
        paths.append(n - n % 2)
        n //= MC_CONVERGENCE_STEP
    return sorted(paths)


@st.cache_data(show_spinner=False, max_entries=16)
def mc_convergence(S, K, T, r, sigma, q, n_paths, seed, antithetic, control_variate) -> tuple[list[int], dict]:
    """The 95% interval's width (2 x 1.96 standard errors, $) of the call and the put at each of
    :func:`mc_convergence_paths`, on the same seed and settings: ``(paths, {kind: widths})``. The samples below
    ``n_paths`` add a third of its time (1/4 + 1/16 + ...)."""
    paths = mc_convergence_paths(n_paths)
    widths: dict[str, list[float]] = {"call": [], "put": []}
    for n in paths:
        results = monte_carlo.price_european(S, K, T, r, sigma, q, KINDS, n, seed, antithetic, control_variate)
        for kind, width in widths.items():
            width.append(2 * 1.96 * float(results[kind].std_error))
    return paths, widths


@st.cache_data(show_spinner=False, max_entries=24)
def value_at_risk(
    strategy: Strategy, market: Market, config: risk.VaRConfig, multiplier: float, calendar_years: float | None = None
) -> risk.VaRResult:
    """``risk.value_at_risk``. The Risk tab makes two runs per view, with implied vol moving and with it frozen (one
    seed), so the cache holds 24 entries: the last 12 views, as many as when a view made one run. Each entry keeps its
    P&L sample, 0.8 MB at the default 100,000 paths and 4 MB at the most, 500,000."""
    return risk.value_at_risk(strategy, market, config, multiplier, calendar_years)


@st.cache_data(show_spinner=False, max_entries=16)
def ssvi_surface(params: vol_surface.SSVIParams, rate, dividend_yield, m_range, t_range) -> vol_surface.VolSurface:
    moneyness = np.linspace(m_range[0], m_range[1], 41)
    maturities = np.linspace(t_range[0], t_range[1], 30)
    return vol_surface.ssvi_surface(params, rate, dividend_yield, moneyness, maturities)


@st.cache_data(show_spinner=False, max_entries=8)
def market_surface(
    raw, spot, rate, dividend_yield, forward="parity", calibration="arbitrage_free", exercise="european"
) -> tuple[vol_surface.VolSurface | None, list[str], str | None]:
    """Clean an uploaded quotes table (inverting any prices at each expiry's ``forward``: ``"parity"`` or
    ``"rates"``, which American ``exercise`` always uses; see ``vol_surface.parse_quotes``) and fit it
    (``calibration``: ``"arbitrage_free"`` or ``"unconstrained"``, see ``vol_surface.fit_market_surface``).

    Returns ``(surface, messages, error)`` so the parse messages survive a fit that fails.
    """
    quotes, messages, forwards = vol_surface.parse_quotes(
        raw, spot, rate, dividend_yield, forward=forward, exercise=exercise, return_forwards=True
    )
    try:
        surface = vol_surface.fit_market_surface(
            quotes, spot, rate, dividend_yield, forwards=forwards, calibration=calibration
        )
        return surface, messages, None
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
    S, K, T, r, q, option_type, low, high, steps=DEFAULT_TREE[0], method=DEFAULT_TREE[1]
) -> tuple[np.ndarray, np.ndarray]:
    """The tree's value at 41 vols from ``low`` to ``high``, starting no lower than the lowest vol the tree can
    represent (``implied_vol.lowest_tree_volatility``)."""
    lowest = implied_vol.lowest_tree_volatility(T, r, q, steps, method)
    start = max(low, lowest)
    vols = np.linspace(start, max(high, start + 0.01), 41)

    def value(vol: float) -> float:
        try:
            return american_value(S, K, T, r, vol, q, option_type, steps, method)
        except ValueError:  # a short tree cannot represent this volatility: leave a gap in the curve
            return np.nan

    return vols, np.array([value(v) for v in vols])


# --- The Pricing tab's Engine diagnostics: each engine on the call and the put, with its time ------------------------
# The time is measured once, when the result is computed (later runs read the cache), on the server: a guide to the
# engines' relative cost, not a benchmark (benchmarks/ holds those).


@st.cache_data(show_spinner=False, max_entries=64)
def tree_timed(S, K, T, r, sigma, q, steps, american, method) -> tuple[dict, float]:
    """``binomial.price`` of the call and the put on one tree, and the milliseconds the two took."""
    start = time.perf_counter()
    values = {kind: binomial.price(S, K, T, r, sigma, q, kind, steps, american, method) for kind in KINDS}
    return values, (time.perf_counter() - start) * 1e3


@st.cache_data(show_spinner=False, max_entries=32)
def mc_timed(S, K, T, r, sigma, q, n_paths, seed, antithetic, control_variate) -> tuple[dict, float]:
    """``monte_carlo.price_european`` of the call and the put (one sample), and the milliseconds it took."""
    start = time.perf_counter()
    results = monte_carlo.price_european(S, K, T, r, sigma, q, KINDS, n_paths, seed, antithetic, control_variate)
    return results, (time.perf_counter() - start) * 1e3


@st.cache_data(show_spinner=False, max_entries=64)
def alo_timed(S, K, T, r, sigma, q, scheme) -> tuple[dict, float]:
    """``american.price`` of the call and the put with one scheme, and the milliseconds the two took."""
    start = time.perf_counter()
    values = {kind: float(american.price(S, K, T, r, sigma, q, option_type=kind, scheme=scheme)) for kind in KINDS}
    return values, (time.perf_counter() - start) * 1e3


@st.cache_data(show_spinner=False, max_entries=64)
def alo_values(S, K, T, r, sigma, q, option_type, scheme) -> np.ndarray:
    """``american.price`` over arrays (spots, strikes, vols or expiries, broadcast together): the engine solves one
    boundary per distinct (K, T, r, σ, q), so 121 spots on one boundary take about 4 ms."""
    return np.asarray(american.price(S, K, T, r, sigma, q, option_type=option_type, scheme=scheme), dtype=float)
