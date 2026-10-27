"""Pricing-model configuration: the sidebar's model choice and the settings of each model.

:class:`ModelConfig` is what the price cards and the model diagnostics run on, and :func:`model_sidebar` draws its
widgets. The market inputs (spot, strike, expiry, volatility, rates) are in :mod:`.inputs`, whose ``sidebar()``
calls :func:`model_sidebar`.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from ..american import SCHEMES
from ..binomial import METHODS
from .format import md

#: The models of the sidebar, by kind, with their labels. A link names a model by its label (``?model=Monte+Carlo``),
#: or by one of :data:`MODEL_ALIASES`.
MODELS = {
    "bsm": "Black-Scholes-Merton",
    "binomial": "Binomial tree",
    "monte_carlo": "Monte Carlo",
    "alo": "American (ALO)",
}
#: Other names a link may give a model: the short names of the header's badge. A link's model is matched without case
#: or punctuation (``?model=alo``, ``?model=monte-carlo``), as its tab is.
MODEL_ALIASES = {"ALO": MODELS["alo"], "BSM": MODELS["bsm"], "MC": MODELS["monte_carlo"], "Tree": MODELS["binomial"]}
#: The schemes of the American engine (``american.SCHEMES``), with their sidebar labels, which a link gives too.
ALO_SCHEMES = {name: name.capitalize() for name in SCHEMES}

PATH_CHOICES = (10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000)


@dataclass(frozen=True)
class ModelConfig:
    """The pricing model of the price cards and the model diagnostics.

    ``kind`` is a key of :data:`MODELS`. A binomial tree uses ``tree_method`` (a key of ``binomial.METHODS``),
    ``steps`` and ``american``; Monte Carlo uses ``n_paths``, ``seed`` and the two variance reductions; the American
    engine (``american.price``) uses ``scheme`` (a key of ``american.SCHEMES``) and always values American exercise.

    Examples
    --------
    >>> ModelConfig(kind="binomial", steps=2000, american=True).label
    'Leisen-Reimer tree · 2000 steps · American'
    >>> ModelConfig(kind="alo").label
    'Andersen-Lake-Offengenden engine · accurate scheme · American'
    """

    kind: str = "bsm"
    tree_method: str = "lr"
    steps: int = 501
    american: bool = False
    n_paths: int = 100_000
    seed: int = 42
    antithetic: bool = True
    control_variate: bool = True
    scheme: str = "accurate"

    @property
    def is_american(self) -> bool:
        """Whether the price cards value American exercise: the American engine, or a tree with American exercise."""
        return self.kind == "alo" or (self.kind == "binomial" and self.american)

    @property
    def label(self) -> str:
        if self.kind == "binomial":
            style = "American" if self.american else "European"
            return f"{METHODS[self.tree_method]} tree · {self.steps} steps · {style}"
        if self.kind == "monte_carlo":
            tricks = [n for n, on in (("antithetic", self.antithetic), ("control variate", self.control_variate)) if on]
            return f"Monte Carlo · {self.n_paths:,} paths" + (f" · {' + '.join(tricks)}" if tricks else "")
        if self.kind == "alo":
            return f"Andersen-Lake-Offengenden engine · {self.scheme} scheme · American"
        return "Black-Scholes-Merton (closed form)"


def model_sidebar() -> ModelConfig:
    """Draw the sidebar's "Pricing model" section and return the chosen model.

    The model radio and the settings of the chosen model (lattice, steps and exercise for a tree; paths, seed and
    variance reductions for Monte Carlo; the scheme of the American engine) are bound to the URL's query
    parameters, so a shared link reopens the same model.

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
    elif kind == "alo":
        scheme = sb.segmented_control(
            "Scheme",
            list(ALO_SCHEMES),
            default=default.scheme,
            format_func=ALO_SCHEMES.get,
            key="scheme",
            required=True,
            bind="query-params",
            help=md(
                "How finely the engine solves the exercise boundary: fast (8 nodes), accurate (15, the default) or "
                "high (33). Largest error on the 4,495-put validation grid (strike $100) against QuantLib's engine: "
                "fast $0.0007, accurate $0.000005, high $0.00000005."
            ),
        )
        sb.caption(
            "American exercise: the engine solves for the early-exercise boundary by spectral collocation "
            "(Andersen, Lake and Offengenden 2016)."
        )
        config = ModelConfig(kind=kind, scheme=scheme)
    return config
