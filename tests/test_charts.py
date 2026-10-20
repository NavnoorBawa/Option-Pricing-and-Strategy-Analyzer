"""The Plotly house style: per-figure styling, colour tokens per mode, the desk template and charts.show."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pytest

from options_analyzer.ui import charts, theme
from options_analyzer.ui.format import MINUS

DARK, LIGHT = theme.SEMANTIC["dark"], theme.SEMANTIC["light"]
X = np.linspace(80.0, 120.0, 41)


def _builders(palette):
    """One figure from every builder, keyed by name."""
    curves = [(X - 100, "At expiry", charts.CALL, "solid"), (X - 99, "Today", charts.PUT, "dash")]
    steps = np.arange(10, 201, 10)
    grid = np.outer(np.linspace(-1, 1, 5), np.linspace(1, 2, 4)) * 100
    samples = np.linspace(-3.0, 3.0, 1001)  # no outcome beyond the plotted range, so no note above the plot
    var_line = [(-2.0, "VaR", palette["loss"])]
    m = np.linspace(0.8, 1.2, 9)
    return {
        "sensitivity": charts.sensitivity_chart(X, X, X[::-1], "Spot", "Value", 100.0, 105.0, palette=palette),
        "price_heatmap": charts.price_heatmap(X[:5], [0.1, 0.2, 0.3], np.ones((3, 5)), "Call", palette=palette),
        "convergence": charts.convergence_chart({"CRR": (steps, steps * 0.0 + 10)}, 10.0, "BSM", "Call", palette),
        "boundary": charts.exercise_boundary_chart({"Put boundary": (X, X)}, 100.0, palette=palette),
        "paths": charts.paths_chart(X, [X, X + 1], 100.0, X, palette=palette),
        "histogram": charts.histogram_chart(samples, "P&L", var_line, palette=palette),
        "payoff": charts.payoff_chart(X, curves, 100.0, [100.0], [101.0], (90.0, 110.0), palette=palette),
        "pnl_heatmap": charts.pnl_heatmap(list("abcd"), list("vwxyz"), grid, "Date", "Spot", palette=palette),
        "lines": charts.lines_chart(X, {"a": X, "b": X + 1}, "x", "y", vline=(100.0, "spot"), palette=palette),
        "bar": charts.bar_chart(["Crash", "Rally"], [-1234.5, 880.0], "P&L ($)", palette=palette),
        "surface": charts.surface_3d(m, [0.5, 1.0], np.full((2, 9), 0.2), "Implied volatility", "Viridis", palette),
        "smile": charts.smile_chart(m, [("30d", m * 0 + 0.2, m * 0 + 0.21)], "K/S", palette=palette),
    }  # fmt: skip


def _template_value(fig: go.Figure, *path):
    node = fig.layout.template.layout
    for key in path:
        node = node[key]
    return node


@pytest.mark.parametrize("palette", [DARK, LIGHT], ids=["dark", "light"])
def test_every_builder_has_a_height_a_tight_top_margin_and_a_crosshair(palette):
    for name, fig in _builders(palette).items():
        assert fig.layout.height, name
        # (A histogram's note about outcomes beyond the plotted range takes more room above the plot.)
        assert fig.layout.margin.t <= 28, name
        if name in ("price_heatmap", "pnl_heatmap", "bar", "surface"):
            continue  # cell grids, bars and the 3-D scene: a crosshair would only cover the cell values
        assert _template_value(fig, "xaxis", "showspikes") is True, name
        assert _template_value(fig, "xaxis", "spikemode") == "across", name
        assert fig.layout.xaxis.showspikes is not False, name


def test_no_colour_literals_in_charts():
    source = (Path(charts.__file__)).read_text(encoding="utf-8")
    assert not re.findall(r"#[0-9A-Fa-f]{6}\b", source)  # colours come from theme.py
    assert not re.findall(r"rgba?\(\s*\d", source)  # translucent() builds rgba() from a token


def test_the_desk_template_holds_only_keys_the_streamlit_theme_leaves_alone():
    layout = pio.templates["desk"].layout
    assert layout.hovermode == "x unified"
    assert (layout.xaxis.spikemode, layout.xaxis.spikesnap, layout.xaxis.ticks) == ("across", "cursor", "outside")
    assert (layout.legend.orientation, layout.legend.xanchor, layout.legend.yanchor) == ("h", "left", "bottom")
    # The theme overwrites these in a template, so style() sets them on each figure instead.
    assert layout.xaxis.gridcolor is None and layout.yaxis.gridcolor is None and layout.font.family is None
    assert charts.TEMPLATE == "streamlit+desk"


@pytest.mark.parametrize("palette", [DARK, LIGHT], ids=["dark", "light"])
def test_style_sets_grid_and_axis_colours_explicitly(palette):
    fig = charts.style(go.Figure(go.Scatter(x=[1, 2], y=[3, 4])), palette, 300)
    assert fig.layout.height == 300
    assert (fig.layout.margin.l, fig.layout.margin.r, fig.layout.margin.t, fig.layout.margin.b) == (8, 8, 28, 8)
    assert fig.layout.yaxis.gridcolor == fig.layout.yaxis.linecolor == palette["rule"]
    assert fig.layout.xaxis.linecolor == palette["rule"] and fig.layout.xaxis.spikecolor == palette["muted"]
    assert fig.layout.xaxis.zeroline is False and fig.layout.yaxis.zeroline is False
    assert fig.layout.hoverlabel.namelength == -1
    assert charts.style(go.Figure()).layout.height == charts.HEIGHT
    # Tick labels and axis titles too: the light theme's own faded grey was 3.62:1 at 10.5 px.
    for axis in (fig.layout.xaxis, fig.layout.yaxis):
        assert axis.tickfont.color == axis.title.font.color == palette["muted"]


@pytest.mark.parametrize("palette", [DARK, LIGHT], ids=["dark", "light"])
def test_every_builder_writes_its_axis_text_in_the_muted_colour(palette):
    # Also labels without a colour of their own (the Implied Volatility tab's "market price" and "lower bound" were
    # 3.62:1 in light mode) and the 3-D surface's axes and colour bar.
    for name, fig in _builders(palette).items():
        assert fig.layout.font.color == palette["muted"], name
        if name == "surface":
            scene = fig.layout.scene
            axes = (scene.xaxis, scene.yaxis, scene.zaxis)
            colorbar = fig.data[0].colorbar
            assert colorbar.tickfont.color == colorbar.title.font.color == palette["muted"]
        else:
            axes = (fig.layout.xaxis, fig.layout.yaxis)
        for axis in axes:
            assert axis.tickfont.color == axis.title.font.color == palette["muted"], name


def test_legends_under_the_plot_sit_at_the_bottom_of_the_figure():
    # A legend placed 20% of the plot height under the plot ran into the x-axis title once the charts were 360 px
    # high with outside ticks ('Days from today' was drawn through 'Spot +0.5σ (+10.5%)'). Placed in the figure's
    # coordinates, Plotly keeps room for it under the tick labels and the axis title.
    figures = _builders(DARK)
    for name in ("histogram", "payoff", "lines", "smile"):
        legend = figures[name].layout.legend
        assert (legend.orientation, legend.x, legend.xanchor) == ("h", 0.0, "left"), name
        assert (legend.yref, legend.y, legend.yanchor) == ("container", 0.0, "bottom"), name


def test_plotly_config():
    config = charts.PLOTLY_CONFIG
    assert config["displaylogo"] is False
    assert config["displayModeBar"] == "hover"
    assert {"lasso2d", "select2d", "autoScale2d"} <= set(config["modeBarButtonsToRemove"])
    assert config["toImageButtonOptions"]["format"] == "svg"


def test_show_draws_with_the_desk_config(monkeypatch):
    calls = []

    class Container:
        def plotly_chart(self, fig, **kwargs):
            calls.append((self, fig, kwargs))
            return "element"

    default = Container()
    monkeypatch.setattr(charts, "st", default)
    fig = go.Figure()
    assert charts.show(fig, key="payoff_chart") == "element"
    column = Container()
    charts.show(fig, key="heatmap_call", container=column, width=420)
    (where1, fig1, kw1), (where2, _, kw2) = calls
    assert where1 is default and fig1 is fig and where2 is column
    assert kw1["key"] == "payoff_chart" and kw1["theme"] == "streamlit"
    assert kw1["config"]["displaylogo"] is False
    assert kw1["config"]["toImageButtonOptions"] == {"format": "svg", "filename": "options-analyzer-payoff_chart"}
    assert kw2["config"]["toImageButtonOptions"]["filename"] == "options-analyzer-heatmap_call" and kw2["width"] == 420
    assert charts.PLOTLY_CONFIG["toImageButtonOptions"]["filename"] == "options-analyzer-chart"  # left unchanged


def _colours(fig: go.Figure) -> set[str]:
    """Every colour a figure sets explicitly (traces, shapes and annotations)."""
    found = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if "color" in key and isinstance(value, str):
                    found.add(value)
                elif "color" in key and isinstance(value, list | tuple):
                    found.update(v for v in value if isinstance(v, str))
                else:
                    walk(value)
        elif isinstance(node, list | tuple):
            for item in node:
                walk(item)

    spec = fig.to_plotly_json()
    walk(spec["data"])
    walk({k: v for k, v in spec["layout"].items() if k != "template"})
    return found


@pytest.mark.parametrize(("palette", "other"), [(LIGHT, DARK), (DARK, LIGHT)], ids=["light", "dark"])
def test_builders_use_only_the_tokens_of_their_palette(palette, other):
    allowed = set(palette.values()) | set(theme.categorical(palette))
    allowed |= {charts.translucent(palette[r], theme.FILL_OPACITY) for r in ("profit", "loss")}
    allowed |= {charts.CALL, charts.PUT}  # the payoff curves' colours come from the caller (the tab)
    for name, fig in _builders(palette).items():
        if name == "surface":
            continue
        assert _colours(fig) <= allowed, (name, _colours(fig) - allowed)
    payoff = _builders(palette)["payoff"]
    fills = [t.fillcolor for t in payoff.data if t.fill == "tozeroy"]
    assert fills == [charts.translucent(palette["profit"], 0.18), charts.translucent(palette["loss"], 0.18)]
    assert not {other["profit"], other["loss"]} & _colours(payoff)


def test_identity_colours():
    # Call first and put second without a colour: the theme's first two categorical colours (the call and put
    # colours of whichever mode the browser shows).
    sensitivity = charts.sensitivity_chart(X, X, X, "Spot", "Value", 100.0)
    assert [t.name for t in sensitivity.data] == ["Call", "Put"]
    assert all(t.line.color is None for t in sensitivity.data)
    # A put boundary drawn alone keeps the put colour, so these are explicit.
    put = charts.exercise_boundary_chart({"Put boundary": (X, X)}, 100.0, palette=LIGHT)
    assert put.data[0].line.color == LIGHT["put"]
    both = charts.exercise_boundary_chart({"Call boundary": (X, X), "Put boundary": (X, X)}, 100.0, palette=DARK)
    assert [t.line.color for t in both.data] == [DARK["call"], DARK["put"]]
    # Implied and local vol of one maturity share a colour, the categorical colour of the palette's mode.
    smile = charts.smile_chart(X, [("30d", X, X), ("60d", X, X)], "K/S", palette=LIGHT)
    assert [t.line.color for t in smile.data] == [theme.CATEGORICAL["light"][i] for i in (0, 0, 1, 1)]


def test_reference_lines_are_thin_and_muted():
    payoff = charts.payoff_chart(
        X, [(X - 100, "At expiry", "#000001", "solid")], 100.0, [95.0, 105.0], [], palette=DARK
    )
    lines = [s for s in payoff.layout.shapes if s.type == "line"]
    assert lines and all(s.line.color == DARK["muted"] and s.line.width == 1 for s in lines)
    spot = next(a for a in payoff.layout.annotations if a.text == "spot")
    assert spot.font.color == DARK["muted"] and spot.font.size == 11


def test_pnl_heatmap_takes_the_theme_diverging_ramp_centred_on_zero():
    fig = charts.pnl_heatmap(["a", "b"], ["x", "y"], [[-50.0, 10.0], [20.0, 30.0]], "spot", "vol")
    heat = fig.data[0]
    assert heat.autocolorscale is True and heat.colorscale is None and heat.zmid == 0
    assert (heat.zmin, heat.zmax) == (-50.0, 50.0)  # symmetric, so an all-positive grid still uses the ramp
    assert heat.textfont.color is None  # Plotly picks a legible label colour for each cell
    assert "%{z:+$,.2f}" in heat.hovertemplate  # P&L with its sign


def test_price_heatmap_follows_the_theme_unless_given_a_scale():
    values = np.ones((2, 2))
    themed = charts.price_heatmap([1.0, 2.0], [0.1, 0.2], values, "Call").data[0]
    assert themed.autocolorscale is True and themed.colorscale is None
    named = charts.price_heatmap([1.0, 2.0], [0.1, 0.2], values, "Call", "Blues").data[0]
    assert named.autocolorscale is None and named.colorscale is not None


def test_convergence_chart_has_a_log_step_axis_and_the_legend_beside_its_title():
    fig = charts.convergence_chart({"CRR": ([10, 100], [1.0, 2.0])}, 1.5, "BSM", "Call")
    assert fig.layout.xaxis.type == "log"
    assert fig.layout.title.text == "Call" and fig.layout.title.xanchor == "left"
    assert fig.layout.legend.xanchor == "right" and fig.layout.legend.x == 1.0


def test_pnl_labels_carry_their_sign():
    bars = charts.bar_chart(["Crash", "Flat", "Rally"], [-1234.5, 0.0, 45.2], "P&L ($)", palette=DARK)
    assert list(bars.data[0].text) == [f"{MINUS}$1,234", "$0.00", "+$45.20"]
    assert list(bars.data[0].marker.color) == [DARK["loss"], DARK["profit"], DARK["profit"]]
    assert "%{x:+$,.2f}" in bars.data[0].hovertemplate
    payoff = charts.payoff_chart(X, [(X - 100, "At expiry", "#000001", "solid")], 100.0, [], [])
    assert next(t for t in payoff.data if t.name == "At expiry").hovertemplate == "%{y:+$,.2f}"


def test_scenario_styles_order_falls_and_rises():
    put, call, muted = DARK["put"], DARK["call"], DARK["muted"]
    assert charts.scenario_styles(5, DARK) == [
        (put, "solid"),
        (put, "dash"),
        (muted, "solid"),
        (call, "dash"),
        (call, "solid"),
    ]
    assert charts.scenario_styles(4, DARK) == [(put, "solid"), (put, "dash"), (call, "dash"), (call, "solid")]
    assert charts.scenario_styles(1, DARK) == [(muted, "solid")]
    seven = charts.scenario_styles(7, LIGHT)
    assert [d for _, d in seven] == ["solid", "dash", "dot", "solid", "dot", "dash", "solid"]
    assert seven[3][0] == LIGHT["muted"]


def test_lines_chart_styles():
    styles = charts.scenario_styles(3, DARK)
    fig = charts.lines_chart(X, {"down": X, "flat": X, "up": X}, "x", "y", styles=styles)
    assert [(t.line.color, t.line.dash) for t in fig.data] == styles
    plain = charts.lines_chart(X, {"a": X, "b": X}, "x", "y")
    assert all(t.line.color is None for t in plain.data)  # the theme's categorical colours, in order


def test_colour_names_resolve_in_the_current_mode(monkeypatch):
    assert (
        DARK["call"],
        DARK["put"],
        DARK["loss"],
        DARK["profit"],
    ) == (charts.CALL, charts.PUT, charts.LOSS, charts.PROFIT)
    assert (DARK["muted"], DARK["accent"]) == (charts.NEUTRAL, charts.ACCENT)
    monkeypatch.setattr(theme, "mode", lambda: "light")
    assert (LIGHT["call"], LIGHT["muted"]) == (charts.CALL, charts.NEUTRAL)
    with pytest.raises(AttributeError):
        charts.SERIES  # noqa: B018


def test_translucent():
    assert charts.translucent("#26A69A", 0.18) == "rgba(38,166,154,0.18)"
    assert charts.translucent("#000000", 1.0) == "rgba(0,0,0,1)"
