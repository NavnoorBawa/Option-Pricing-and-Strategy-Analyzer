"""Plotly builders of the Risk tab (UI spec 3.3): the scenario heatmap, the ladder, the stress bars and the explain
waterfall. Pure functions in the house style of :mod:`.charts` (its :func:`~.charts.apply_layout` and palette), so
they are tested without the app.

The heatmap follows UI spec 2.7 item 7: the theme's diverging ramp centred on zero with symmetric limits, every
cell's value printed by a scatter overlay of square, near-transparent markers whose text colour is chosen per cell
for contrast with its fill (Plotly's ``heatmap.textfont.color`` takes one colour, a scatter takes one per point), and
the same overlay is the click target, since a click on a heatmap cell returns no point.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import plotly.graph_objects as go

from . import theme as desk_theme
from .charts import (
    HEIGHT,
    HEIGHT_SECONDARY,
    LABEL_SIZE,
    LINE_WIDTH,
    MARGIN,
    REFERENCE_WIDTH,
    _hline,
    _vline,
    apply_layout,
)
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


def worst_cell(z, centre: tuple[int, int] | None = None) -> tuple[int, int, int] | None:
    """``(row, column, ties)`` of a grid's lowest cell: of the cells equal to it as the grid shows them (to the
    dollar, or to the cent when every cell is below $10), the one nearest ``centre`` (default the middle cell),
    column first, so the smallest spot move that reaches the loss; ``ties`` counts them. ``None`` without a finite
    cell.

    A plain argmin outlined the first in row order: on the default call spread at its maximum loss, the −10% ·
    −10 vp corner of two cells at −$4,096, and on Spot × Date T+0 · −10% when −$4,096 runs from T+15 to expiry.

    >>> worst_cell([[-4096.2, -4095.8, -4000.0], [-3611.0, -3000.0, -2000.0]])
    (0, 1, 2)
    >>> worst_cell([[-4096.0, -4096.0, 5.0], [-4096.0, 0.0, 9.0]], centre=(1, 1))
    (0, 1, 3)
    """
    values = np.asarray(z, dtype=float)
    if not np.isfinite(values).any():
        return None
    decimals = 0 if np.nanmax(np.abs(values)) >= 10 else 2
    shown = np.round(values, decimals)
    ties = np.argwhere(shown == np.nanmin(shown))
    ci, cj = centre if centre is not None else (values.shape[0] // 2, values.shape[1] // 2)
    i, j = min(ties.tolist(), key=lambda cell: (abs(cell[1] - cj), abs(cell[0] - ci), cell[0], cell[1]))
    return int(i), int(j), len(ties)


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
    centre: tuple[int, int] | None = None,
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
        Outline the lowest cell (the worst loss) in the loss colour: of equal ones, the nearest ``centre``
        (:func:`worst_cell`).
    centre : (row, column), optional
        The no-move cell, from which the outlined worst cell is the nearest of equal ones; default the middle cell.
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
        i, j, _ = worst_cell(z, centre)
        outlines.append((i, j, colors["loss"], "dot"))
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


#: Most ticks :func:`signed_ticks` puts on the ladder's axis (:data:`~.charts.HEIGHT_SECONDARY`, about 230 px of
#: plot: 19 px a tick at 12, for 11 px tick labels).
MAX_TICKS = 12


def signed_ticks(values) -> list[float] | None:
    """Tick values of an axis that draws ``values`` and zero: the largest round step (1, 2 or 5 times a power of ten)
    that leaves three to :data:`MAX_TICKS` ticks in the range and a tick on each side of zero that the data reach by
    a tenth of the range or more; ``None`` (Plotly's own ticks) for a flat range.

    Plotly's automatic step on the ladder's chart was 5,000 for P&L from −$4,096 to +$6,070, which left the loss
    side without a tick: no loss could be read off the chart.

    >>> signed_ticks([-4096.0, 6070.0])
    [-4000.0, -2000.0, 0.0, 2000.0, 4000.0, 6000.0]
    >>> signed_ticks([-0.5, 12.0]), signed_ticks([0.0, 0.0])
    ([0.0, 5.0, 10.0], None)
    >>> signed_ticks([-1500.0, 10000.0])
    [-1000.0, 0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0]
    """
    finite = [float(v) for v in values if math.isfinite(float(v))]
    lo, hi = min([*finite, 0.0]), max([*finite, 0.0])
    span = hi - lo
    if span <= 0:
        return None
    unit = 10.0 ** math.floor(math.log10(span))
    fallback = None
    for step in (m * unit * 10.0**e for e in (0, -1, -2) for m in (5, 2, 1)):  # largest first
        ticks = [round(step * n, 10) + 0.0 for n in range(math.ceil(lo / step), math.floor(hi / step) + 1)]
        if len(ticks) < 3:
            continue
        if len(ticks) > MAX_TICKS:
            break
        fallback = fallback or ticks
        if (lo > -0.1 * span or ticks[0] < 0) and (hi < 0.1 * span or ticks[-1] > 0):
            return ticks
    return fallback


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
    ticks = signed_ticks(values)
    if ticks is None:
        fig.update_yaxes(tickformat=",.0f")
    else:  # the step's decimals: 0.05 on the textbook contract's Θ column
        decimals = next(d for d in range(7) if all(abs(t - round(t, d)) < 1e-9 for t in ticks))
        fig.update_yaxes(tickmode="array", tickvals=ticks, tickformat=f",.{decimals}f")
    return fig


#: Width (px) of a character of the stress chart's amounts at 12 px, generous for the app's sans fonts (Source Sans
#: draws "−$4,096" 44 px wide at 12 px: 6.3 px a character), and the gap between the plot and the amounts.
AMOUNT_CHAR_WIDTH, AMOUNT_GAP = 7.0, 8
#: Height (px) of a stress bar's row with its name above the bar, the gap between bars, and the offset of the
#: name's baseline above the bar's centre. Measured in Chromium: the plot gets 0.92 of the row (33 px of 36), so a
#: bar is 13 px thick at a gap of 0.6, and a name's 14 px box ends 1.5 px below its anchor; 8 px above the centre it
#: clears its bar by 3 px and the bar above by 3 px. Rows of 34 px left 0.2 px to the bar above, and a name centred
#: 14 px above, the first try, sat on its own bar on a phone.
STRESS_ROW, STRESS_NAME_SHIFT, STRESS_BAR_GAP = 36, 8, 0.6


def stress_bars(names: Sequence[str], values, *, palette: dict[str, str] | None = None) -> go.Figure:
    """Horizontal P&L bars, one per stress scenario in the order given (worst first, at the top), with each signed
    amount in a column right of the plot, right-aligned like a table's figures, in the text colour, and each name
    above its bar at the plot's left edge.

    The names sat left of the plot before, as the axis's tick labels: on a 390 px phone they took 150 px and the
    amounts 60, which left the bars about 70 px. Above the bars they take no width, and the bars the whole plot.

    The amounts sat on the bars before (inside a long bar, outside a short one), sized for a plot about 250 px wide:
    on a 390 px phone the scenario names take half the chart and the plot is about 140 px, so an amount inside a
    bar lost its last digit ("−$3,35") and one outside a negative bar ran over its scenario's name. In a margin of
    their own the amounts never meet a bar or a name at any width, and need no second colour (spec 2.6: text in the
    text colour on the background).
    """
    colors = palette or desk_theme.palette()
    values = [float(v) for v in values]
    labels = [signed(v, 0 if abs(v) >= 100 else 2, "$") for v in values]
    column = int(np.ceil(AMOUNT_CHAR_WIDTH * max((len(t) for t in labels), default=0))) + AMOUNT_GAP
    fig = go.Figure(
        go.Bar(
            x=values,
            y=list(names),
            orientation="h",
            marker_color=[colors["profit"] if v >= 0 else colors["loss"] for v in values],
            customdata=labels,
            hovertemplate="%{y}<br>%{customdata}<extra></extra>",
            name="P&L",
        )
    )
    for name, label in zip(names, labels, strict=True):
        fig.add_annotation(
            x=1.0,
            xref="paper",
            xanchor="right",
            xshift=column,
            y=name,
            yref="y",
            text=label,
            showarrow=False,
            font=dict(color=colors["text"], size=LABEL_SIZE + 1),
        )
        fig.add_annotation(
            x=0.0,
            xref="paper",
            xanchor="left",
            y=name,
            yref="y",
            yanchor="bottom",
            yshift=STRESS_NAME_SHIFT,
            text=name,
            showarrow=False,
            font=dict(color=colors["text"], size=LABEL_SIZE),
        )
    fig.add_vline(x=0.0, line=dict(color=colors["muted"], width=REFERENCE_WIDTH))
    height = max(HEIGHT_SECONDARY, STRESS_ROW * len(values) + 70)
    fig = apply_layout(fig, None, "P&L ($)", "", height=height, palette=colors)
    fig.update_layout(
        hovermode="closest",
        yaxis=dict(autorange="reversed", showticklabels=False),
        bargap=STRESS_BAR_GAP,
        margin=dict(r=column + MARGIN["r"], t=MARGIN["t"] + STRESS_NAME_SHIFT),
    )
    fig.update_xaxes(showspikes=False, tickformat="$,.0f")
    ticks = signed_ticks(values)  # on a phone Plotly's own step, 5,000, drew no tick on the loss side
    if ticks is not None and all(t == round(t) for t in ticks):
        fig.update_xaxes(tickmode="array", tickvals=ticks)
    return fig


def waterfall(names: Sequence[str], values, total: float, *, palette: dict[str, str] | None = None) -> go.Figure:
    """An explain waterfall: one bar per term from zero, then the full-revaluation P&L as the total bar.

    Rises are in the profit colour and falls in the loss colour, each labelled with its signed amount; the total is
    in the accent colour, so it is never read as a term.
    """
    colors = palette or desk_theme.palette()
    values = [float(v) for v in values]
    # One precision for every label, the explain table's (a money column: cents below $1,000, whole dollars from
    # $1,000): per label, "+$77.03" sat beside the table's 77 once the total passed $1,000. A term that is zero at
    # that precision (Δ with spot unchanged) has no label: a row of "$0.00" read as noise.
    amounts = [*values, float(total)]
    decimals = 2 if round(max((abs(v) for v in amounts), default=0.0), 2) < 1000 else 0
    labels = [signed(v, decimals, "$") if round(abs(v), decimals) > 0 else "" for v in amounts]
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
