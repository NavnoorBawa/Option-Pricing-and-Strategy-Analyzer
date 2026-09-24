"""Sidebar: market inputs and pricing-model configuration."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from ..binomial import METHODS
from ..strategies import Market

MODELS = {
    "bsm": "Black-Scholes-Merton",
    "binomial": "Binomial tree",
    "monte_carlo": "Monte Carlo",
}

PATH_CHOICES = (10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000)


@dataclass(frozen=True)
class ModelConfig:
    kind: str = "bsm"
    tree_method: str = "lr"
    steps: int = 501
    american: bool = False
    n_paths: int = 100_000
    seed: int = 42
    antithetic: bool = True
    control_variate: bool = True

    @property
    def label(self) -> str:
        if self.kind == "binomial":
            style = "American" if self.american else "European"
            return f"{METHODS[self.tree_method]} tree · {self.steps} steps · {style}"
        if self.kind == "monte_carlo":
            tricks = [n for n, on in (("antithetic", self.antithetic), ("control variate", self.control_variate)) if on]
            return f"Monte Carlo · {self.n_paths:,} paths" + (f" · {' + '.join(tricks)}" if tricks else "")
        return "Black-Scholes-Merton (closed form)"


@dataclass(frozen=True)
class Inputs:
    spot: float
    strike: float
    days: int
    volatility: float  # decimal
    rate: float  # decimal
    dividend_yield: float  # decimal
    model: ModelConfig

    @property
    def expiry(self) -> float:
        """Time to expiry in years (ACT/365)."""
        return self.days / 365.0

    @property
    def market(self) -> Market:
        return Market(self.spot, self.rate, self.volatility, self.expiry, self.dividend_yield)


def sidebar() -> Inputs:
    sb = st.sidebar
    sb.header("Market")
    spot = sb.number_input(
        "Spot price",
        min_value=0.01,
        max_value=1e7,
        value=100.0,
        step=1.0,
        format="%.2f",
        key="spot",
        bind="query-params",
        help="Current price of the underlying asset.",
    )
    strike = sb.number_input(
        "Strike price",
        min_value=0.01,
        max_value=1e7,
        value=100.0,
        step=1.0,
        format="%.2f",
        key="strike",
        bind="query-params",
    )
    days = sb.number_input(
        "Days to expiry",
        min_value=1,
        max_value=3650,
        value=365,
        step=1,
        key="days",
        bind="query-params",
        help="Calendar days; time to expiry T = days / 365.",
    )
    vol_pct = sb.number_input(
        "Volatility (%)",
        min_value=0.1,
        max_value=500.0,
        value=20.0,
        step=0.5,
        format="%.2f",
        key="vol",
        bind="query-params",
        help="Annualised implied volatility σ.",
    )
    rate_pct = sb.number_input(
        "Risk-free rate (%)",
        min_value=-5.0,
        max_value=50.0,
        value=5.0,
        step=0.05,
        format="%.2f",
        key="rate",
        bind="query-params",
        help="Continuously compounded; negative rates are allowed.",
    )
    div_pct = sb.number_input(
        "Dividend yield (%)",
        min_value=0.0,
        max_value=50.0,
        value=0.0,
        step=0.1,
        format="%.2f",
        key="div",
        bind="query-params",
        help="Continuous dividend yield q (Merton's extension).",
    )
    sb.caption(f"T = {days / 365:.4f} years")

    sb.header("Pricing model")
    kind = sb.radio(
        "Model",
        list(MODELS),
        format_func=MODELS.get,
        key="model",
        bind="query-params",
        label_visibility="collapsed",
    )
    config = ModelConfig(kind=kind)
    if kind == "binomial":
        method = sb.radio(
            "Lattice",
            list(METHODS),
            format_func=METHODS.get,
            key="tree",
            horizontal=True,
            help="Leisen-Reimer converges smoothly and much faster; CRR is the classic tree.",
        )
        steps = sb.slider("Steps", 10, 2000, 501, step=1, key="steps")
        style = sb.segmented_control(
            "Exercise", ["European", "American"], default="European", key="style", required=True
        )
        config = ModelConfig(kind=kind, tree_method=method, steps=int(steps), american=style == "American")
    elif kind == "monte_carlo":
        n_paths = sb.select_slider("Paths", PATH_CHOICES, value=100_000, format_func=lambda n: f"{n:,}", key="paths")
        seed = sb.number_input(
            "Random seed",
            min_value=0,
            max_value=2**32 - 1,
            value=42,
            step=1,
            key="seed",
            help="Fixed seed = reproducible prices that do not jitter between reruns.",
        )
        antithetic = sb.toggle("Antithetic variates", value=True, key="antithetic")
        control = sb.toggle("Control variate (discounted spot)", value=True, key="control_variate")
        config = ModelConfig(
            kind=kind, n_paths=int(n_paths), seed=int(seed), antithetic=antithetic, control_variate=control
        )

    sb.divider()
    sb.markdown(
        "Built by **[Navnoor Bawa](https://www.linkedin.com/in/navnoorbawa/)** · "
        "[Source](https://github.com/NavnoorBawa/Option-Pricing-and-Strategy-Analyzer)"
    )
    sb.caption("Educational tool — not investment advice.")

    return Inputs(
        spot=float(spot),
        strike=float(strike),
        days=int(days),
        volatility=float(vol_pct) / 100.0,
        rate=float(rate_pct) / 100.0,
        dividend_yield=float(div_pct) / 100.0,
        model=config,
    )
