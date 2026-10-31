"""Plotly builders of the Risk tab (UI spec 3.3): the scenario heatmap, the ladder, the stress bars and the explain
waterfall. Pure functions in the house style of :mod:`.charts` (its :func:`~.charts.apply_layout` and palette), so
they are tested without the app.

The heatmap follows UI spec 2.7 item 7: the theme's diverging ramp centred on zero with symmetric limits, every
cell's value printed by a scatter overlay of square, near-transparent markers whose text colour is chosen per cell
for contrast with its fill (Plotly's ``heatmap.textfont.color`` takes one colour, a scatter takes one per point), and
the same overlay is the click target, since a click on a heatmap cell returns no point.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import plotly.graph_objects as go

from . import theme as desk_theme
from .charts import HEIGHT, HEIGHT_SECONDARY, LABEL_SIZE, LINE_WIDTH, REFERENCE_WIDTH, _hline, _vline, apply_layout
from .format import signed

#: Smallest contrast of a cell's printed value with the cell's fill (WCAG 2.2 SC 1.4.3, normal text).
CELL_CONTRAST = 4.5
#: Size (px) of the printed cell values: the desk's smallest chart text (improvement guide item 16).
CELL_TEXT_SIZE = 11


def _rgb(color: str) -> np.ndarray:
    value = color.lstrip("#")
    return np.array([int(value[i : i + 2], 16) for i in (0, 2, 4)], dtype=float)


def _hex(rgb: np.ndarray) -> str:
    return "#" + "".join(f"{int(c + 0.5):02X}" for c in np.clip(rgb, 0, 255))


def luminance(color: str) -> float:
    """Relative luminance of a ``#RRGGBB`` colour (WCAG 2.2)."""
    channel = _rgb(color) / 255.0
    linear = np.where(channel <= 0.04045, channel / 12.92, ((channel + 0.055) / 1.055) ** 2.4)
    return float(linear @ np.array([0.2126, 0.7152, 0.0722]))


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio of two ``#RRGGBB`` colours, from 1 to 21."""
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def diverging_scale(palette: dict[str, str] | None = None) -> list[list]:
    """The theme's diverging ramp (``chartDivergingColors``) of the palette's mode as a Plotly colour scale: its ten
    colours at equal steps from 0 (loss) to 1 (profit), as the Streamlit theme spreads them."""
    ramp = desk_theme.DIVERGING[_mode(palette)]
    return [[i / (len(ramp) - 1), color] for i, color in enumerate(ramp)]


def _mode(palette: dict[str, str] | None) -> str:
    palette = palette or desk_theme.palette()
    return next((kind for kind in desk_theme.MODES if desk_theme.SEMANTIC[kind] == palette), desk_theme.mode())


def cell_fill(value: float, bound: float, palette: dict[str, str] | None = None) -> str:
    """The fill Plotly draws for ``value`` on :func:`diverging_scale` with limits ``±bound``: linear interpolation in
    RGB between the two neighbouring ramp colours."""
    ramp = desk_theme.DIVERGING[_mode(palette)]
    t = 0.5 if bound <= 0 else float(np.clip((value + bound) / (2 * bound), 0.0, 1.0))
    position = t * (len(ramp) - 1)
    i = min(int(np.floor(position)), len(ramp) - 2)
    w = position - i
    return _hex((1 - w) * _rgb(ramp[i]) + w * _rgb(ramp[i + 1]))


def cell_text_color(fill: str, palette: dict[str, str] | None = None) -> str:
    """The text colour of a cell: the theme's text colour, or its background colour where that contrasts more with
    the fill (a saturated red or teal end of the light ramp)."""
    colors = palette or desk_theme.palette()
    return max((colors["text"], colors["surface"]), key=lambda c: contrast(c, fill))


def scenario_heatmap(
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    z,
    x_title: str,
    y_title: str,
    *,
    text: Callable[[float, float], str],
    hover: str = "P&L",
    customdata=None,
    worst: bool = True,
    current: tuple[int, int] | None = None,
    height: int | None = None,
    palette: dict[str, str] | None = None,
) -> go.Figure:
    """A signed grid as a diverging heatmap with every value printed (UI spec 2.7 item 7).

    Parameters
    ----------
    x_labels, y_labels : sequence of str
        Column and row labels (category axes, in order; the first row at the bottom).
    z : array_like
        Values, shape ``(len(y_labels), len(x_labels))``.
    x_title, y_title : str
        Axis titles, with units.
    text : callable
        ``text(value, bound)``: a cell's printed value, ``bound`` the largest absolute value of the grid.
    hover : str, default "P&L"
        Name of the value in the hover label, which gives it to the cent.
    customdata : array_like, optional
        Per-cell data for a selection callback, shape ``(rows, columns, k)``; the overlay's points run row by row.
    worst : bool, default True
        Outline the lowest cell (the worst loss) in the loss colour.
    current : (row, column), optional
        Outline this cell in amber: the scenario the what-if strip holds.
    height : int, optional
        Figure height (px); default the primary chart's.
    palette : dict, optional
        Semantic colours; default the current mode's.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    colors = palette or desk_theme.palette()
    z = np.asarray(z, dtype=float)
    rows, columns = z.shape
    bound = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 0.0
    scale = bound or 1.0
    fig = go.Figure(
        go.Heatmap(
            x=list(x_labels),
            y=list(y_labels),
            z=z,
            zmin=-scale,
            zmax=scale,
            zmid=0.0,
            colorscale=diverging_scale(colors),
            showscale=False,
            hoverinfo="skip",
            xgap=1,
            ygap=1,
        )
    )
    xs = [x_labels[j] for i in range(rows) for j in range(columns)]
    ys = [y_labels[i] for i in range(rows) for j in range(columns)]
    values = z.ravel()
    fills = [cell_fill(v, scale, colors) for v in values]
    overlay = dict(
        x=xs,
        y=ys,
        mode="markers+text",
        text=[text(float(v), bound) for v in values],
        textfont=dict(color=[cell_text_color(f, colors) for f in fills], size=CELL_TEXT_SIZE),
        # Near-transparent squares: the click target of each cell (a click on the heatmap itself selects nothing).
        marker=dict(symbol="square", size=30, color=fills, opacity=0.01),
        showlegend=False,
        hovertemplate=f"{x_title} %{{x}}<br>{y_title} %{{y}}<br>{hover} %{{customdata[0]:+$,.2f}}<extra></extra>",
        name="cells",
    )
    data = np.column_stack([values])
    if customdata is not None:
        data = np.column_stack([values, np.asarray(customdata, dtype=float).reshape(rows * columns, -1)])
    fig.add_trace(go.Scatter(**overlay, customdata=data))
    outlines = []
    if worst and bound > 0 and np.nanmin(z) < 0:
        i, j = np.unravel_index(int(np.nanargmin(z)), z.shape)
        outlines.append((int(i), int(j), colors["loss"], "dot"))
    if current is not None:
        outlines.append((current[0], current[1], desk_theme.AMBER[_mode(colors)], "solid"))
    for i, j, color, dash in outlines:
        fig.add_shape(
            type="rect",
            x0=j - 0.5,
            x1=j + 0.5,
            y0=i - 0.5,
            y1=i + 0.5,
            xref="x",
            yref="y",
            line=dict(color=color, width=2, dash=dash),
            fillcolor="rgba(0,0,0,0)",
        )
    fig = apply_layout(fig, None, x_title, y_title, height=height or HEIGHT, palette=colors)
    fig.update_layout(hovermode="closest", dragmode=False, clickmode="event+select")
    fig.update_xaxes(type="category", showspikes=False, showgrid=False, tickfont=dict(size=LABEL_SIZE))
    fig.update_yaxes(type="category", showgrid=False, tickfont=dict(size=LABEL_SIZE))
    return fig


def ladder_chart(
    spots,
    values,
    y_title: str,
    *,
    spot: float,
    labels: Sequence[str] | None = None,
    palette: dict[str, str] | None = None,
) -> go.Figure:
    """One metric of the spot ladder against spot: the ladder's points joined by a line, today's spot dashed.

    Parameters
    ----------
    spots, values : array_like
        The ladder's spot levels and the metric at each.
    y_title : str
        The metric with its unit (``P&L ($)``).
    spot : float
        Spot at the ladder's centre (the what-if spot).
    labels : sequence of str, optional
        Each point's move, for the hover (``−2%``).
    palette : dict, optional
        Semantic colours.
    """
    colors = palette or desk_theme.palette()
    fig = go.Figure(
        go.Scatter(
            x=list(spots),
            y=list(values),
            mode="lines+markers",
            line=dict(color=colors["accent"], width=LINE_WIDTH),
            marker=dict(size=6, color=colors["accent"]),
            customdata=list(labels) if labels is not None else None,
            hovertemplate=("%{customdata} · " if labels is not None else "")
            + "spot %{x:,.2f}<br>%{y:+,.0f}<extra></extra>",
            name=y_title,
            showlegend=False,
        )
    )
    _hline(fig, 0.0, colors)
    _vline(fig, spot, "spot", colors, dash="dash")
    fig = apply_layout(fig, None, "Spot", y_title, height=HEIGHT_SECONDARY, palette=colors)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(tickformat=",.0f")
    fig.update_yaxes(tickformat=",.0f")
    return fig


def stress_bars(names: Sequence[str], values, *, palette: dict[str, str] | None = None) -> go.Figure:
    """Horizontal P&L bars, one per stress scenario in the order given (worst first, at the top), each labelled with
    its signed amount.

    Labels inside a bar are in the background colour and labels outside in the text colour (UI spec 2.6): white on
    the dark theme's loss red was 3.49:1, the background colour on it is 5.51:1.
    """
    colors = palette or desk_theme.palette()
    values = [float(v) for v in values]
    # A label goes inside its bar only when the bar spans at least a third of the axis: about 85 px of a phone's
    # 250 px plot, room for "−$4,096" at 12 px. Plotly's "auto" kept a label inside a short bar and let it run out
    # of the bar in the background colour, where it vanished (Vol crush, −$47, on the default call spread).
    span = (max(values, default=0.0) - min(values, default=0.0)) or 1.0
    fig = go.Figure(
        go.Bar(
            x=values,
            y=list(names),
            orientation="h",
            marker_color=[colors["profit"] if v >= 0 else colors["loss"] for v in values],
            text=[signed(v, 0 if abs(v) >= 100 else 2, "$") for v in values],
            textposition=["inside" if abs(v) >= span / 3 else "outside" for v in values],
            insidetextfont=dict(color=colors["surface"], size=LABEL_SIZE + 1),
            outsidetextfont=dict(color=colors["text"], size=LABEL_SIZE + 1),
            cliponaxis=False,
            hovertemplate="%{y}<br>%{x:+$,.2f}<extra></extra>",
            name="P&L",
        )
    )
    fig.add_vline(x=0.0, line=dict(color=colors["muted"], width=REFERENCE_WIDTH))
    fig = apply_layout(fig, None, "P&L ($)", "", height=max(HEIGHT_SECONDARY, 24 * len(values) + 70), palette=colors)
    # Plotly shrinks a label to fit inside its bar (to 4 px on a short bar): at least 11 px, moved outside if need be.
    fig.update_layout(
        hovermode="closest", yaxis=dict(autorange="reversed"), bargap=0.25, uniformtext=dict(minsize=11, mode="show")
    )
    fig.update_xaxes(showspikes=False, tickformat="$,.0f")
    return fig


def waterfall(names: Sequence[str], values, total: float, *, palette: dict[str, str] | None = None) -> go.Figure:
    """An explain waterfall: one bar per term from zero, then the full-revaluation P&L as the total bar.

    Rises are in the profit colour and falls in the loss colour, each labelled with its signed amount; the total is
    in the accent colour, so it is never read as a term.
    """
    colors = palette or desk_theme.palette()
    values = [float(v) for v in values]
    labels = [signed(v, 0 if abs(v) >= 100 else 2, "$") for v in [*values, total]]
    fig = go.Figure(
        go.Waterfall(
            x=[*names, "Total"],
            y=[*values, float(total)],
            measure=["relative"] * len(values) + ["total"],
            text=labels,
            textposition="outside",
            textfont=dict(color=colors["text"], size=LABEL_SIZE),
            increasing=dict(marker=dict(color=colors["profit"])),
            decreasing=dict(marker=dict(color=colors["loss"])),
            totals=dict(marker=dict(color=colors["accent"])),
            connector=dict(line=dict(color=colors["rule"], width=1)),
            cliponaxis=False,
            hovertemplate="%{x}<br>%{y:+$,.2f}<extra></extra>",
            name="Explain",
        )
    )
    fig = apply_layout(fig, None, "", "P&L ($)", height=HEIGHT_SECONDARY, palette=colors)
    fig.update_layout(hovermode="closest", showlegend=False)
    fig.update_xaxes(showspikes=False, type="category")
    fig.update_yaxes(tickformat="$,.0f")
    return fig
