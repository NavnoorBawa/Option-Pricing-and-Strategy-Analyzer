"""The pricing-model section of the sidebar (ui/model_config.py) and its place in the sidebar (ui/inputs.py)."""

from __future__ import annotations

import pytest

from options_analyzer.ui import inputs, model_config
from options_analyzer.ui.model_config import ModelConfig


def _ok(at) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def _sidebar(at) -> list[str]:
    """The sidebar's headers and widget labels, in order."""
    out = []

    def walk(node):
        for child in getattr(node, "children", {}).values():
            if getattr(child, "type", "") == "header":
                out.append(f"# {child.value}")
            elif hasattr(child, "label"):
                out.append(child.label)
            walk(child)

    walk(at.sidebar)
    return out


def test_model_config_is_still_importable_from_inputs():
    # Existing code imports ModelConfig from inputs; it is the same class.
    assert inputs.ModelConfig is ModelConfig


@pytest.mark.parametrize(
    ("settings", "label"),
    [
        ({}, "Black-Scholes-Merton (closed form)"),
        ({"kind": "binomial"}, "Leisen-Reimer tree · 501 steps · European"),
        (
            {"kind": "binomial", "tree_method": "crr", "steps": 1001, "american": True},
            "Cox-Ross-Rubinstein tree · 1001 steps · American",
        ),
        ({"kind": "monte_carlo"}, "Monte Carlo · 100,000 paths · antithetic + control variate"),
        (
            {"kind": "monte_carlo", "n_paths": 1_000_000, "antithetic": False},
            "Monte Carlo · 1,000,000 paths · control variate",
        ),
        ({"kind": "monte_carlo", "antithetic": False, "control_variate": False}, "Monte Carlo · 100,000 paths"),
    ],
)
def test_model_labels(settings, label):
    assert ModelConfig(**settings).label == label


def test_sidebar_takes_its_model_from_model_sidebar(monkeypatch):
    # inputs.sidebar() draws the market inputs and leaves the model section to model_config.model_sidebar().
    chosen = ModelConfig(kind="binomial", steps=2000, american=True)
    monkeypatch.setattr(inputs, "model_sidebar", lambda: chosen)
    assert inputs.sidebar().model is chosen  # outside a running app the market widgets return their defaults


MARKET = ["# Market", "Spot price", "Strike price", "Days to expiry", "Volatility (%)", "Risk-free rate (%)",
          "Dividend yield (%)", "# Pricing model", "Model"]  # fmt: skip


@pytest.mark.parametrize(
    ("query", "widgets"),
    [
        ({}, []),
        ({"model": "Binomial tree"}, ["Lattice", "Steps", "Exercise"]),
        (
            {"model": "Monte Carlo"},
            ["Paths", "Random seed", "Antithetic variates", "Control variate (discounted spot)"],
        ),
    ],
)
def test_the_model_section_follows_the_market_inputs(new_app, query, widgets):
    at = new_app(**query).run()
    _ok(at)
    assert _sidebar(at) == [*MARKET, *widgets]


@pytest.mark.parametrize(
    ("query", "label"),
    [
        (
            {"model": "Binomial tree", "tree": "Cox-Ross-Rubinstein", "steps": 1001, "style": "American"},
            "Cox-Ross-Rubinstein tree · 1001 steps · American",
        ),
        (
            {"model": "Monte Carlo", "paths": "250,000", "seed": 7, "antithetic": "false"},
            "Monte Carlo · 250,000 paths · control variate",
        ),
    ],
)
def test_a_shared_link_reopens_the_same_model(new_app, query, label):
    # Every model widget is bound to the URL (shown values, as a shared link carries them). The header's model
    # badge names the model in full in its tooltip.
    at = new_app(**query).run()
    _ok(at)
    [badge] = [m for m in at.markdown if m.value.startswith(":gray-badge[")]
    assert badge.proto.help == f"Pricing model of the price cards: {label}."


def test_every_model_has_a_sidebar_label():
    assert model_config.MODELS == {
        "bsm": "Black-Scholes-Merton",
        "binomial": "Binomial tree",
        "monte_carlo": "Monte Carlo",
        "alo": "American (ALO)",
    }
