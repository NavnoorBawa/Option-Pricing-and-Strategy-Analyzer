"""Pricing-model configuration: the sidebar's model choice and the settings of each model.

:class:`ModelConfig` is what the price cards and the model diagnostics run on, and :func:`model_sidebar` draws its
widgets. The market inputs (spot, strike, expiry, volatility, rates) are in :mod:`.inputs`, whose ``sidebar()``
calls :func:`model_sidebar`.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from ..binomial import METHODS

MODELS = {
    "bsm": "Black-Scholes-Merton",
    "binomial": "Binomial tree",
    "monte_carlo": "Monte Carlo",
}

PATH_CHOICES = (10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000)


@dataclass(frozen=True)
class ModelConfig:
    """The pricing model of the price cards and the model diagnostics.

    ``kind`` is a key of :data:`MODELS`. A binomial tree uses ``tree_method`` (a key of ``binomial.METHODS``),
    ``steps`` and ``american``; Monte Carlo uses ``n_paths``, ``seed`` and the two variance reductions.

    Examples
    --------
    >>> ModelConfig(kind="binomial", steps=2000, american=True).label
    'Leisen-Reimer tree · 2000 steps · American'
    """

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


def model_sidebar() -> ModelConfig:
    """Draw the sidebar's "Pricing model" section and return the chosen model.

    The model radio and the settings of the chosen model (lattice, steps and exercise for a tree; paths, seed and
    variance reductions for Monte Carlo) are bound to the URL's query parameters, so a shared link reopens the
    same model.

    Returns
    -------
    ModelConfig
        The selected model with its settings; the defaults of :class:`ModelConfig` for the settings the model
        does not use.

    Examples
    --------
    Outside a running app the widgets return their defaults:

    >>> model_sidebar().label
    'Black-Scholes-Merton (closed form)'
    """
    sb = st.sidebar
    sb.header("Pricing model")
    kind = sb.radio(
        "Model",
        list(MODELS),
        format_func=MODELS.get,
        key="model",
        bind="query-params",
        label_visibility="collapsed",
    )
    default = ModelConfig()
    config = ModelConfig(kind=kind)
    if kind == "binomial":
        method = sb.radio(
            "Lattice",
            list(METHODS),
            index=list(METHODS).index(default.tree_method),
            format_func=METHODS.get,
            key="tree",
            bind="query-params",
            horizontal=True,
            help="Leisen-Reimer converges smoothly and much faster; CRR is the classic tree.",
        )
        steps = sb.slider("Steps", 10, 2000, default.steps, step=1, key="steps", bind="query-params")
        style = sb.segmented_control(
            "Exercise", ["European", "American"], default="European", key="style", required=True, bind="query-params"
        )
        config = ModelConfig(kind=kind, tree_method=method, steps=int(steps), american=style == "American")
    elif kind == "monte_carlo":
        n_paths = sb.select_slider(
            "Paths",
            PATH_CHOICES,
            value=default.n_paths,
            format_func=lambda n: f"{n:,}",
            key="paths",
            bind="query-params",
        )
        seed = sb.number_input(
            "Random seed",
            min_value=0,
            max_value=2**32 - 1,
            value=default.seed,
            step=1,
            key="seed",
            bind="query-params",
            help="Fixed seed = reproducible prices that do not jitter between reruns.",
        )
        antithetic = sb.toggle("Antithetic variates", value=default.antithetic, key="antithetic", bind="query-params")
        control = sb.toggle(
            "Control variate (discounted spot)",
            value=default.control_variate,
            key="control_variate",
            bind="query-params",
        )
        config = ModelConfig(
            kind=kind, n_paths=int(n_paths), seed=int(seed), antithetic=antithetic, control_variate=control
        )
    return config
