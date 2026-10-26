"""Sidebar: market inputs, with the pricing-model section from :mod:`.model_config`."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from ..strategies import Market
from .model_config import ModelConfig, model_sidebar  # code that imports ModelConfig from here keeps working


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

    config = model_sidebar()

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
