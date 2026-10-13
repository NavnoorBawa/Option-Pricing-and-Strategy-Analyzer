"""Monte Carlo valuation of European options under geometric Brownian motion.

A European payoff depends only on the terminal price, and GBM has an exact
terminal distribution, so the estimator samples ``S_T`` directly instead of
stepping through time: this is unbiased, needs ``O(n_paths)`` memory and is
orders of magnitude faster than simulating full paths. Full paths are only
generated (in small numbers) for plotting: :func:`scenario_paths` fills in the
path to expiry of the scenarios behind the price, :func:`simulate_paths`
simulates independent paths.

Variance reduction
------------------
* **Antithetic variates** - every normal draw ``Z`` is paired with ``-Z`` and
  the pair average is treated as one independent sample.
* **Control variate** - the discounted terminal price, whose expectation
  ``S e^{-qT}`` is known exactly, is used with the regression-optimal
  coefficient.

Calls and puts are priced from the *same* draws (common random numbers) and
a seeded :class:`numpy.random.Generator` makes results reproducible, so the
displayed price does not jitter between Streamlit reruns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import black_scholes as bs

__all__ = ["MonteCarloResult", "price_european", "scenario_paths", "simulate_paths", "terminal_prices"]

_Z_95 = 1.959963984540054


@dataclass(frozen=True)
class MonteCarloResult:
    price: float
    #: Standard error of ``price``. NaN when the sample cannot estimate its own error: every simulated price
    #: finished out of the money (all payoffs zero), or every one finished in the money with the control
    #: variate on (the payoff is then linear in the control and the residuals are exactly constant).
    std_error: float
    n_paths: int
    antithetic: bool
    control_variate: bool
    #: Fraction of simulated terminal prices that finished in the money.
    itm_fraction: float = float("nan")

    @property
    def ci95(self) -> tuple[float, float]:
        half = _Z_95 * self.std_error
        return self.price - half, self.price + half


def _normals(n: int, antithetic: bool, rng: np.random.Generator) -> np.ndarray:
    if antithetic:
        half = rng.standard_normal((n + 1) // 2)
        return np.concatenate([half, -half])
    return rng.standard_normal(n)


def terminal_prices(
    S: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    n_paths: int = 100_000,
    seed: int | None = 42,
    antithetic: bool = True,
) -> np.ndarray:
    """Exact samples of ``S_T`` under the risk-neutral measure.

    With ``antithetic=True`` the first half of the array holds the draws for
    ``Z`` and the second half those for ``-Z``; an odd ``n_paths`` is then
    rounded up to the next even number.
    """
    rng = np.random.default_rng(seed)
    z = _normals(int(n_paths), antithetic, rng)
    tau = max(T, 0.0)
    return S * np.exp((r - q - 0.5 * sigma**2) * tau + sigma * np.sqrt(tau) * z)


def price_european(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_types: tuple[str, ...] = ("call", "put"),
    n_paths: int = 100_000,
    seed: int | None = 42,
    antithetic: bool = True,
    control_variate: bool = True,
) -> dict[str, MonteCarloResult]:
    """Price European options by Monte Carlo from one shared set of draws.

    Returns a dict ``{option_type: MonteCarloResult}``. The standard error uses
    the unbiased (``ddof=1``) sample deviation of the independent samples
    (pair averages when antithetic sampling is on). It is NaN when the sample
    cannot estimate it (see :class:`MonteCarloResult`).
    """
    n_paths = int(n_paths)
    if n_paths < 100:
        raise ValueError("n_paths must be at least 100")
    if S <= 0 or K <= 0:
        raise ValueError("spot and strike must be positive")
    if sigma < 0:
        raise ValueError("volatility must be non-negative")
    if antithetic and n_paths % 2:
        n_paths += 1

    if T <= 0 or sigma == 0:
        forward = S * np.exp((r - q) * max(T, 0.0))
        return {
            kind: MonteCarloResult(
                bs.price(S, K, T, r, sigma, q, kind),
                0.0,
                n_paths,
                antithetic,
                control_variate,
                float(bs.option_sign(kind) * (forward - K) > 0),
            )
            for kind in option_types
        }

    s_t = terminal_prices(S, T, r, sigma, q, n_paths, seed, antithetic)
    discount = np.exp(-r * T)
    control = discount * s_t
    control_mean = S * np.exp(-q * T)

    results = {}
    for kind in option_types:
        phi = bs.option_sign(kind)
        y = discount * np.maximum(phi * (s_t - K), 0.0)
        n_itm = int(np.count_nonzero(y > 0.0))
        x = control
        if antithetic:
            half = n_paths // 2
            y = 0.5 * (y[:half] + y[half:])
            x = 0.5 * (x[:half] + x[half:])
        if control_variate:
            x_centred = x - x.mean()
            var_x = float(np.dot(x_centred, x_centred))
            beta = float(np.dot(x_centred, y - y.mean()) / var_x) if var_x > 0 else 0.0
            y = y - beta * (x - control_mean)
        estimate = float(y.mean())
        degenerate = n_itm == 0 or (control_variate and n_itm == n_paths)
        std_error = float("nan") if degenerate else float(y.std(ddof=1) / np.sqrt(y.size))
        results[kind] = MonteCarloResult(estimate, std_error, n_paths, antithetic, control_variate, n_itm / n_paths)
    return results


def scenario_paths(
    S: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    n_paths: int = 100_000,
    seed: int | None = 42,
    antithetic: bool = True,
    n_show: int = 60,
    n_steps: int = 252,
) -> tuple[np.ndarray, np.ndarray]:
    """The first ``n_show`` scenarios behind :func:`price_european`, drawn as paths from today to expiry.

    The estimator samples only the terminal price, so each path is a Brownian bridge from ``S`` today to
    that scenario's ``S_T``: given its end point, a GBM path is a Brownian bridge in log space, so these
    are exact sample paths of the scenarios used for the price. Returns ``(times, paths)`` with ``paths``
    of shape ``(n_show, n_steps + 1)``; ``paths[:, -1]`` equals ``terminal_prices(...)[:n_show]``.
    """
    n_steps = max(int(n_steps), 1)
    tau = max(T, 0.0)
    times = np.linspace(0.0, tau, n_steps + 1)
    s_t = terminal_prices(S, T, r, sigma, q, max(int(n_paths), int(n_show)), seed, antithetic)[: int(n_show)]
    if tau == 0.0 or sigma == 0.0:
        return times, np.tile(S * np.exp((r - q) * times), (s_t.size, 1))
    drift = r - q - 0.5 * sigma**2
    w_end = (np.log(s_t / S) - drift * tau) / sigma
    # Independent stream for the interior points, so the scenarios themselves are those of the pricing draws.
    rng = np.random.default_rng(None if seed is None else [int(seed), 1])
    steps = np.sqrt(tau / n_steps) * rng.standard_normal((s_t.size, n_steps))
    free = np.concatenate([np.zeros((s_t.size, 1)), np.cumsum(steps, axis=1)], axis=1)
    bridge = free + (times / tau) * (w_end - free[:, -1])[:, None]
    paths = S * np.exp(drift * times + sigma * bridge)
    paths[:, -1] = s_t
    return times, paths


def simulate_paths(
    S: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    n_paths: int = 100,
    n_steps: int = 252,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate full GBM paths (exact log-normal increments) for visualisation.

    Returns ``(times, paths)`` with ``paths`` of shape ``(n_paths, n_steps + 1)``.
    """
    n_steps = max(int(n_steps), 1)
    rng = np.random.default_rng(seed)
    dt = max(T, 0.0) / n_steps
    increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal((int(n_paths), n_steps))
    log_paths = np.concatenate([np.zeros((int(n_paths), 1)), np.cumsum(increments, axis=1)], axis=1)
    times = np.linspace(0.0, max(T, 0.0), n_steps + 1)
    return times, S * np.exp(log_paths)
