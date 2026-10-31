"""The Strategy tab's figures (UI spec 3.2 item 4, 2.7): the P&L chart with its Greek strip in one figure, and the
date × spot P&L table (UI-9).

Pure functions, data in and figure out, in the desk house style of :mod:`.charts`: time encodes as dash (the expiry
curve solid in the text colour, T+0 solid in the accent, dated curves muted, dashed then dotted), each curve labelled
at its right end; P&L areas only under the expiry curve; risk-neutral ranges as muted bands named with their basis in
the legend; the straddle-implied move as two accent ticks on the x-axis; reference lines for spot (dashed), the
forward (solid accent) and the strikes (dotted, labelled while at most six are in view).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import charts
from . import theme as desk_theme
from .format import level, signed

#: Heights (px): the P&L row (a primary chart) and the Greek strip under it, in one figure (spec 2.7 item 1).
PNL_HEIGHT, STRIP_HEIGHT = 360, 140
#: Strike labels show while at most this many strikes are in view; strikes closer than this share of the range share
#: one label (spec 2.7 item 3).
MAX_STRIKE_LABELS, MERGE_SHARE = 6, 0.03
#: Opacity of the 68% and 95% risk-neutral bands (spec 2.7 item 6).
RN_OPACITY = (0.06, 0.03)
#: Label size of reference lines and curve ends (px).
SMALL = 10


@dataclass(frozen=True)
class Curve:
    """One P&L curve: ``role`` is ``"expiry"``, ``"t0"`` or ``"dated"`` (its style), ``end`` its right-end label."""

    values: np.ndarray
    name: str
    role: str
    end: str


@dataclass(frozen=True)
class Band:
    """A risk-neutral range: its bounds and its legend entry (``RN 68% to 18‑Dec: 5,491–6,162 (smile)``)."""

    low: float
    high: float
    name: str


def _strike_labels(strikes: Sequence[float], x_range: tuple[float, float], text) -> list[tuple[float, str]]:
    """``(x, label)`` of the strike labels: none while more than :data:`MAX_STRIKE_LABELS` strikes are in view; strikes
    closer than :data:`MERGE_SHARE` of the range share one label at their mean (``K 5,300/5,400``)."""
    lo, hi = x_range
    shown = sorted(k for k in strikes if lo <= k <= hi)
    if not shown or len(shown) > MAX_STRIKE_LABELS:
        return []
    groups: list[list[float]] = [[shown[0]]]
    for k in shown[1:]:
        if k - groups[-1][-1] < MERGE_SHARE * (hi - lo):
            groups[-1].append(k)
        else:
            groups.append([k])
    return [(float(np.mean(g)), "K " + "/".join(text(k) for k in g)) for g in groups]


def payoff_figure(
    spots: np.ndarray,
    curves: Sequence[Curve],
    strip: tuple[np.ndarray, str] | None,
    *,
    spot: float,
    forward: float | None,
    strikes: Sequence[float],
    breakevens: Sequence[tuple[float, str]],
    bands: Sequence[Band] = (),
    straddle: tuple[float, float, str] | None = None,
    scenario: tuple[float, float, str] | None = None,
    note: str = "",
    x_range: tuple[float, float] | None = None,
    strike_text=lambda k: f"{k:,.0f}",
    palette: dict[str, str] | None = None,
) -> go.Figure:
    """The P&L chart: P&L ($) against spot in the top row and, under it on the same x-axis, the Greek strip.

    Parameters
    ----------
    spots : numpy.ndarray
        The x grid (spot).
    curves : sequence of Curve
        The expiry curve first (shaded profit and loss areas), then T+0 and the dated curves.
    strip : (values, axis title), optional
        The Greek strip's series; None draws the P&L row alone.
    spot, forward : float
        Today's spot (dashed muted line, ``S 5,800``) and the front expiry's forward (solid accent line, ``F``);
        ``forward`` None draws no forward line.
    strikes : sequence of float
        Dotted muted lines, labelled per :func:`_strike_labels`.
    breakevens : sequence of (x, label)
        Accent diamonds on y = 0 with their labels (``BE 5,378 (−7.3%)``).
    bands : sequence of Band
        Risk-neutral ranges, widest last, drawn at :data:`RN_OPACITY`.
    straddle : (low, high, label), optional
        The straddle-implied move: two accent ticks on the x-axis and a label at the upper one.
    scenario : (x, y, label), optional
        The what-if spot on the T+0 curve, marked.
    note : str
        An 11 px muted annotation inside the plot at the top left (spec 2.7 item 8).
    x_range : (low, high), optional
        The x-axis range (spec 2.7 item 11's Range control); default: the grid.
    strike_text : callable
        Writes a strike for its label.
    palette : dict, optional
        The colours (default: the current mode's).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    colors = palette or desk_theme.palette()
    rows = 2 if strip is not None else 1
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[PNL_HEIGHT, STRIP_HEIGHT] if rows == 2 else None,
    )
    lo, hi = x_range or (float(spots[0]), float(spots[-1]))
    styles = {"expiry": (colors["text"], "solid"), "t0": (colors["accent"], "solid")}
    dated = iter(("dash", "dot", "dashdot"))

    # Risk-neutral ranges: muted bands behind everything, named in the legend with their basis.
    for band, opacity in zip(bands, RN_OPACITY, strict=False):
        fig.add_vrect(x0=band.low, x1=band.high, fillcolor=colors["muted"], opacity=opacity, line_width=0, row=1, col=1)
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=band.name,
                marker=dict(symbol="square", size=12, color=colors["muted"], opacity=0.25 + 3 * opacity),
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
    first = np.asarray(curves[0].values, dtype=float)
    for part, role in ((np.where(first > 0, first, 0.0), "profit"), (np.where(first < 0, first, 0.0), "loss")):
        fig.add_trace(
            go.Scatter(
                x=spots,
                y=part,
                fill="tozeroy",
                mode="none",
                fillcolor=charts.translucent(colors[role], desk_theme.FILL_OPACITY),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
    inside = (spots >= lo) & (spots <= hi)
    for curve in curves:
        color, dash = styles.get(curve.role) or (colors["muted"], next(dated, "dot"))
        fig.add_trace(
            go.Scatter(
                x=spots,
                y=curve.values,
                name=curve.name,
                line=dict(color=color, width=charts.LINE_WIDTH, dash=dash),
                hovertemplate="%{y:+$,.0f}",
            ),
            row=1,
            col=1,
        )
        if inside.any():  # the right-end label, at the last point in view
            j = int(np.flatnonzero(inside)[-1])
            fig.add_annotation(
                x=spots[j], y=float(curve.values[j]), text=curve.end, xanchor="right", yanchor="bottom",
                showarrow=False, font=dict(size=SMALL, color=color), row=1, col=1,
            )  # fmt: skip
    fig.add_hline(y=0.0, line=dict(color=colors["rule"], width=charts.REFERENCE_WIDTH), row=1, col=1)
    # Reference lines across both rows (the strip shares the x-axis): spot, forward, strikes.
    for k in strikes:
        if lo <= k <= hi:
            fig.add_vline(x=k, line=dict(color=colors["muted"], width=charts.REFERENCE_WIDTH, dash="dot"))
    for x, label in _strike_labels(strikes, (lo, hi), strike_text):
        fig.add_annotation(
            x=x, y=1.0, xref="x", yref="paper", text=label, showarrow=False, yanchor="bottom",
            font=dict(size=SMALL, color=colors["muted"]),
        )  # fmt: skip
    fig.add_vline(x=spot, line=dict(color=colors["muted"], width=charts.REFERENCE_WIDTH, dash="dash"))
    fig.add_annotation(
        x=spot, y=1.0, xref="x", yref="paper", text=f"S {level(spot).split('.')[0]}", showarrow=False,
        yanchor="bottom", yshift=12, font=dict(size=SMALL, color=colors["muted"]),
    )  # fmt: skip
    if forward is not None:
        fig.add_vline(x=forward, line=dict(color=colors["accent"], width=charts.REFERENCE_WIDTH))
        fig.add_annotation(
            x=forward, y=1.0, xref="x", yref="paper", text="F", showarrow=False, yanchor="bottom", yshift=12,
            xanchor="left", font=dict(size=SMALL, color=colors["accent"]),
        )  # fmt: skip
    if breakevens:
        fig.add_trace(
            go.Scatter(
                x=[x for x, _ in breakevens],
                y=[0.0] * len(breakevens),
                mode="markers+text",
                name="Breakeven",
                text=[label for _, label in breakevens],
                textposition="bottom center",
                textfont=dict(size=SMALL, color=colors["accent"]),
                marker=dict(size=9, color=colors["accent"], symbol="diamond"),
                hovertemplate="%{text}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if straddle is not None:
        s_lo, s_hi, s_label = straddle
        for x in (s_lo, s_hi):
            fig.add_shape(
                type="line", x0=x, x1=x, xref="x", y0=0.0, y1=0.035, yref="y domain",
                line=dict(color=colors["accent"], width=2), row=1, col=1,
            )  # fmt: skip
        fig.add_annotation(
            x=s_hi, y=0.035, xref="x", yref="y domain", text=s_label, showarrow=False, xanchor="left",
            yanchor="bottom", font=dict(size=SMALL, color=colors["accent"]), row=1, col=1,
        )  # fmt: skip
    if scenario is not None:
        x, y, label = scenario
        fig.add_trace(
            go.Scatter(
                x=[x], y=[y], mode="markers", name=label, showlegend=False,
                marker=dict(size=9, color=colors["accent"], symbol="circle-open", line=dict(width=2)),
                hovertemplate=f"{label}<extra></extra>",
            ),
            row=1,
            col=1,
        )  # fmt: skip
    if note:
        fig.add_annotation(
            x=0.0, y=1.0, xref="x domain", yref="y domain", text=note, showarrow=False, xanchor="left",
            yanchor="top", align="left", font=dict(size=11, color=colors["muted"]), row=1, col=1,
        )  # fmt: skip
    if strip is not None:
        values, title = strip
        fig.add_trace(
            go.Scatter(
                x=spots,
                y=values,
                name=title,
                showlegend=False,
                line=dict(color=colors["accent"], width=1.5),
                hovertemplate="%{y:+,.0f}",
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=0.0, line=dict(color=colors["rule"], width=charts.REFERENCE_WIDTH), row=2, col=1)
        fig.update_yaxes(title_text=title, row=2, col=1, nticks=3)
    height = PNL_HEIGHT + (STRIP_HEIGHT if strip is not None else 0)
    charts.style(fig, colors, height)
    fig.update_yaxes(title_text="P&L ($)", row=1, col=1, tickformat="$,.0f")
    fig.update_xaxes(range=[lo, hi], showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot")
    fig.update_xaxes(title_text="Spot", row=rows, col=1, tickformat=",.0f")
    fig.update_layout(margin=dict(t=40), legend=dict(font=dict(size=11)))
    return fig


def pnl_table_figure(
    columns: Sequence[str],
    rows: Sequence[str],
    values: np.ndarray,
    *,
    spot_row: int | None,
    unit: str,
    palette: dict[str, str] | None = None,
) -> go.Figure:
    """The date × spot P&L table (UI-9) as a heatmap of the theme's diverging ramp, centred on zero, with every cell's
    value printed (:func:`.charts.pnl_heatmap`) and the row of today's spot ruled above and below.

    Parameters
    ----------
    columns : sequence of str
        Dates, ``Fri<br>06‑Nov``.
    rows : sequence of str
        Spot levels with their change, ``5,800 (0.0%)``, lowest first (drawn at the bottom).
    values : numpy.ndarray
        Rows × columns, in ``unit``.
    spot_row : int, optional
        The row of today's spot, ruled.
    unit : str
        ``$``, ``% of max loss`` or ``% of premium``.
    """
    colors = palette or desk_theme.palette()
    fig = charts.pnl_heatmap(list(columns), list(rows), values, "Date", "Spot", colorbar=False, palette=colors)
    trace = fig.data[0]
    bound = float(np.nanmax(np.abs(values))) or 1.0
    if unit != "$":
        trace.text = [[signed(float(v), 0) + "%" if np.isfinite(v) else "–" for v in row] for row in values]
        trace.hovertemplate = "Date %{x}<br>Spot %{y}<br>P&L %{z:+,.1f}" + f" {unit}<extra></extra>"
    else:
        trace.text = [[charts.pnl_cell_text(float(v), bound) for v in row] for row in values]
    if spot_row is not None:
        for edge in (spot_row - 0.5, spot_row + 0.5):
            fig.add_hline(y=edge, line=dict(color=colors["text"], width=1))
    fig.update_layout(height=max(260, 28 * len(rows) + 90))
    fig.update_xaxes(side="top", title_text=None)
    return fig
