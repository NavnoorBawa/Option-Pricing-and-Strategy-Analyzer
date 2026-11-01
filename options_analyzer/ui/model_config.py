"""Pricing-model configuration: the sidebar's engine choice and the settings of each engine.

The sidebar offers two engines (UI spec 2.3 group 6, Appendix B decision 8): European exercise by the closed form
(Black-Scholes-Merton) and American exercise by the Andersen-Lake-Offengenden engine. The binomial trees and Monte
Carlo are diagnostics of those numbers, not global engines: they live in the Pricing tab's Engine diagnostics
(:mod:`.tab_pricing`), which builds their :class:`ModelConfig` from its own settings. Under the tree a ±10-strike chain
took 0.7 to 2.0 s and a 121-point profile 3.9 to 11.7 s, and Monte Carlo yields only delta (UI spec 2.0).

:class:`ModelConfig` is what the Pricing and IV tabs run on, and :func:`model_sidebar` draws its widgets. The market
inputs (spot, strike, expiry, volatility, rates) are in :mod:`.inputs`, whose ``sidebar()`` calls
:func:`model_sidebar`.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from ..american import SCHEMES
from ..binomial import METHODS
from .format import md

#: Every engine of the app, by kind, with its name. The sidebar offers :data:`SIDEBAR_MODELS`; the other two run in
#: the Pricing tab's Engine diagnostics, and a link that names them (version 2's ``?model=Binomial+tree``) opens
#: those diagnostics instead (``app._read_link``).
MODELS = {
    "bsm": "European (BSM)",
    "binomial": "Binomial tree",
    "monte_carlo": "Monte Carlo",
    "alo": "American (ALO)",
}
#: The engines of the sidebar's control, in its order.
SIDEBAR_MODELS = ("bsm", "alo")
#: Other names a link may give an engine: the short names of the header's badge, and version 2's label of the closed
#: form. A link's engine is matched without case or punctuation (``?model=alo``, ``?model=monte-carlo``), as its tab
#: is.
MODEL_ALIASES = {
    "ALO": MODELS["alo"],
    "BSM": MODELS["bsm"],
    "Black-Scholes-Merton": MODELS["bsm"],
    "European": MODELS["bsm"],
    "American": MODELS["alo"],
    "MC": MODELS["monte_carlo"],
    "Tree": MODELS["binomial"],
    "Binomial": MODELS["binomial"],
}
#: The schemes of the American engine (``american.SCHEMES``), with their sidebar labels, which a link gives too.
ALO_SCHEMES = {name: name.capitalize() for name in SCHEMES}
#: Why the index offers no American engine: the sidebar's caption in place of the engine control.
EUROPEAN_ONLY = (
    "European (BSM), the closed form: SPX options are European-exercise (Cboe). The American engine prices the "
    "Textbook market; trees and Monte Carlo are in the Pricing tab's Engine diagnostics."
)

PATH_CHOICES = (10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000)


@dataclass(frozen=True)
class ModelConfig:
    """An engine and its settings: the sidebar's (closed form or American engine) for the Pricing and IV tabs, or a
    diagnostics engine (a binomial tree, Monte Carlo) of the Pricing tab's Engine diagnostics.

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
        """Whether the engine values American exercise: the American engine, or a tree with American exercise."""
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


def model_sidebar(*, european_only: bool = False) -> ModelConfig:
    """Draw the sidebar's "Pricing model" section and return the chosen engine.

    One segmented control, ``European (BSM)`` or ``American (ALO)``, and the American engine's scheme while it is
    chosen, all bound to the URL's query parameters, so a shared link reopens the same engine.

    Parameters
    ----------
    european_only : bool
        The contract is European-exercise (the synthetic index: SPX options are). A caption names the engine and
        says why there is no other (:data:`EUROPEAN_ONLY`) in place of the control. Disabled, the control greyed
        both options (2.4:1 contrast, neither marked as chosen), and its reason was help on a collapsed label, which
        draws no help icon, so nothing on screen said why.

    Returns
    -------
    ModelConfig
        The selected engine with its settings; the defaults of :class:`ModelConfig` for the settings it does not use.

    Examples
    --------
    Outside a running app the widgets return their defaults:

    >>> model_sidebar().label
    'Black-Scholes-Merton (closed form)'
    """
    sb = st.sidebar
    sb.header("Pricing model")
    state = st.session_state
    if state.get("model") not in (None, *SIDEBAR_MODELS) or (european_only and state.get("model") == "alo"):
        state["model"] = "bsm"  # the American engine, from the textbook market, or an engine of an older session
    if european_only:
        sb.caption(EUROPEAN_ONLY)
        return ModelConfig(kind="bsm")
    kind = sb.segmented_control(
        "Engine",
        SIDEBAR_MODELS,
        default="bsm",
        required=True,
        format_func=MODELS.get,
        key="model",
        bind="query-params",
        label_visibility="collapsed",
        help="European (BSM): the closed form, exact for European exercise. American (ALO): early exercise by the "
        "Andersen-Lake-Offengenden engine. Trees and Monte Carlo are in the Pricing tab's Engine diagnostics.",
    )
    if kind != "alo":
        return ModelConfig(kind="bsm")
    default = ModelConfig()
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
    return ModelConfig(kind="alo", scheme=scheme)
