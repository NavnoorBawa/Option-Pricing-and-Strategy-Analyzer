"""The Strategy tab's figures (UI spec 3.2 item 4, 2.7): the P&L chart with its Greek strip in one figure, and the
date × spot P&L table (UI-9).

Pure functions, data in and figure out, in the desk house style of :mod:`.charts`: time encodes as dash (the expiry
curve solid in the text colour, T+0 solid in the accent, dated curves muted, dashed then dotted), each curve labelled
in the right margin at the height of its right end; P&L areas only under the expiry curve; risk-neutral ranges as
muted bands named with their basis in the legend; the straddle-implied move as two accent ticks on the x-axis;
reference lines for spot (dashed), the forward (solid accent) and the strikes (dotted, labelled while at most six are
in view).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import charts
from . import theme as desk_theme
from .format import signed

#: Heights (px): the P&L row (a primary chart) and the Greek strip under it, in one figure (spec 2.7 item 1).
PNL_HEIGHT, STRIP_HEIGHT = 360, 140
#: Strike labels show while at most this many strikes are in view; strikes closer than this share of the range share
#: one label (spec 2.7 item 3, which says 3%). The server does not know the plot's width, so the share is set for the
#: narrowest: a phone's plot is about 230 px wide (390 px screen), where ``K 5,300`` (about 45 px in 10 px Plex) and
#: ``K 5,400`` 100 points apart in a 2,200-point range sat 10 px apart and overprinted as ``K 5K 5,400``. At 15% they
#: share ``K 5,300/5,400`` (about 75 px), and the next label is at least 35 px clear of it on a phone.
MAX_STRIKE_LABELS, MERGE_SHARE = 6, 0.15
#: A breakeven's label (two lines, ``BE 5,378`` over ``−7.3%``, about 55 px) is drawn outward from its diamond, unless
#: the breakeven lies within this share of the range from the plot's edge: then inward. An annotation is not clipped
#: at the plot's edge as the scatter text was (cut to ``78 (−7.3%)`` on a phone): the default condor's lower
#: breakeven, 21% of the range from the left edge, is 48 px in on a phone's 230 px plot, so its label reaches about
#: 12 px into the axis margin, between the $0 and $2,000 tick labels. Inward, it would cover the expiry curve's steep
#: rise from the breakeven to the short strike.
BREAKEVEN_EDGE = 0.12
#: Opacity of the 68% and 95% risk-neutral bands (spec 2.7 item 6).
RN_OPACITY = (0.06, 0.03)
#: Label size of reference lines and curve ends (px).
SMALL = 10
#: Room above the highest curve, as a share of the curves' span: two 14 px lines of labels (strikes; S and F) and a
#: margin, 35 px of about 270 px of plot, so h / (1.06 + h) = 35 / 270 gives h = 0.158. The note (expiry, days and
#: basis) was a third line inside the plot (0.24 of headroom), over S, F and the payoff on a phone; it is the spot
#: axis's title now.
HEADROOM = 0.16
#: Least distance between two curves' right-end labels, as a share of the curves' span, which the axis shows over
#: 1.06 + HEADROOM = 1.22 spans: 26 px (a two-line 10 px label) on a phone, whose P&L plot is 238 px tall under the
#: four-line legend (26 × 1.22 / 238 = 0.133).
LABEL_GAP = 0.135
#: Gap (px) between the plot's right edge and the curves' labels in the margin.
END_GAP = 4
#: Width (px) given to a right-aligned label per character, and its padding: IBM Plex Sans at 10 px averages about
#: 5.6 px a character in these labels. Plotly sizes an annotation's box from the text as first drawn, before the
#: web font has loaded, so a box sized by Plotly was narrower than the label in Plex and cut its right end at the
#: plot's edge; a box of a set width with the text right-aligned in it ends at its anchor whatever the font.
CHAR_PX, LABEL_PAD = 6.4, 4


def _label_width(text: str) -> int:
    """The box width (px) of a right-aligned label: generous for its characters at :data:`SMALL` px."""
    return round(CHAR_PX * len(text)) + LABEL_PAD


def _spread(ys: Sequence[float], gap: float, floor: float = -np.inf, ceiling: float = np.inf) -> list[float]:
    """Label heights near ``ys`` with at least ``gap`` between neighbours, none below ``floor`` and, where they fit,
    none above ``ceiling``.

    Labels closer than ``gap`` form a cluster laid out ``gap`` apart around the mean of their heights; clusters that
    then overlap merge, until none do. A pass from the bottom up keeps every label at or above ``floor`` and ``gap``
    above the one below it, so converging curves at the plot's bottom keep their labels inside it; then labels above
    ``ceiling`` move down, as far as the floor allows, so curves that converge at the top keep theirs inside too.

    >>> _spread([0.0, 0.1, 5.0], 1.0)
    [-0.45, 0.55, 5.0]
    >>> _spread([0.0, 0.1, 0.2], 1.0, floor=0.0)
    [0.0, 1.0, 2.0]
    >>> _spread([5.0, 5.0, 5.0], 1.0, ceiling=5.0)
    [3.0, 4.0, 5.0]
    """
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    clusters: list[list[int]] = []

    def bottom(cluster: list[int]) -> float:
        return float(np.mean([ys[k] for k in cluster])) - gap * (len(cluster) - 1) / 2

    for i in order:
        clusters.append([i])
        while len(clusters) > 1 and bottom(clusters[-1]) - (bottom(clusters[-2]) + gap * (len(clusters[-2]) - 1)) < gap:
            last = clusters.pop()
            clusters[-1] += last
    placed = [0.0] * len(ys)
    previous = -np.inf
    for cluster in clusters:
        first = bottom(cluster)
        for n, k in enumerate(cluster):
            placed[k] = max(first + n * gap, floor, previous + gap)
            previous = placed[k]
    over = max(placed, default=-np.inf) - ceiling
    if over > 0:
        down = min(over, min(placed) - floor)
        placed = [v - down for v in placed]
    return [round(v, 12) for v in placed]


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


def _vline(x: float, line: dict) -> dict:
    """A vertical reference line through both rows (the P&L chart and its Greek strip): one shape on the shared x-axis
    from the bottom of the figure's plot area to its top. ``add_vline`` on a figure of subplots works out the axes of
    every subplot for each line: with it a what-if step's server run took about 0.25 s, with these shapes 0.22 s."""
    return dict(type="line", x0=x, x1=x, xref="x", y0=0, y1=1, yref="paper", line=line)


def _hline(row: int, line: dict) -> dict:
    """The zero line across row ``row``."""
    axis = "" if row == 1 else str(row)
    return dict(type="line", x0=0, x1=1, xref=f"x{axis} domain", y0=0.0, y1=0.0, yref=f"y{axis}", line=line)


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
        Accent diamonds on y = 0 with ``label`` in their hover (``BE 5,378 (−7.3%)``), each labelled in two lines on
        the surface colour (``BE 5,378`` over ``−7.3%``), outward but inside the plot.
    bands : sequence of Band
        Risk-neutral ranges, widest last, drawn at :data:`RN_OPACITY`.
    straddle : (low, high, label), optional
        The straddle-implied move: two accent ticks on the x-axis, named in the legend (on a phone a label beside the
        ticks overprinted the curves' end labels).
    scenario : (x, y, label), optional
        The what-if spot on the T+0 curve, marked.
    note : str
        The chart's context (expiry, days, basis), written after the spot axis's title: inside the plot at the top
        left (spec 2.7 item 8) it lay over S, F and the payoff on a phone.
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
    # The shapes and labels go in with one layout update at the end: add_shape and add_annotation revalidate the
    # growing tuple at each call: the default condor's figure (12 shapes, 11 labels) took 79 ms one at a time and 47
    # ms at once (the fastest of 15 runs; the same figure, its JSON compared, on three positions).
    shapes: list[dict] = []
    annotations: list[dict] = []

    # Risk-neutral ranges: muted bands behind everything, named in the legend with their basis.
    for band, opacity in zip(bands, RN_OPACITY, strict=False):
        shapes.append(dict(
            type="rect", x0=band.low, x1=band.high, xref="x", y0=0, y1=1, yref="y domain", fillcolor=colors["muted"],
            opacity=opacity, line=dict(width=0), layer="below",
        ))  # fmt: skip
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
    ends = []
    for curve in curves:
        color, dash = styles.get(curve.role) or (colors["muted"], next(dated, "dot"))
        fig.add_trace(
            go.Scatter(
                x=spots,
                y=curve.values,
                name=curve.name,
                # The curves are named at their right ends; the legend keeps T+0, whose name carries the shifts.
                showlegend=curve.role == "t0",
                line=dict(color=color, width=charts.LINE_WIDTH, dash=dash),
                hovertemplate="%{y:+$,.0f}",
            ),
            row=1,
            col=1,
        )
        if inside.any():  # the label of the curve's right end, at the last point in view
            j = int(np.flatnonzero(inside)[-1])
            ends.append((float(curve.values[j]), curve.end.replace(" · ", "<br>"), color))
    # The labels sit in the right margin, left-aligned, at their curves' heights: inside the plot their boxes hid the
    # last 90 px of the curves at 1,440 px (and the expiry's drop at the wing on a phone), and Plotly cut "01‑Dec" at
    # the plot's edge. Two lines (``T+13`` over its date) keep the margin to the width of a date.
    column = max((_label_width(line) for _, text, _ in ends for line in text.split("<br>")), default=0) + END_GAP
    if ends:
        shown = np.concatenate([np.asarray(c.values, dtype=float)[inside] for c in curves] + [np.zeros(1)])
        span = float(np.nanmax(shown) - np.nanmin(shown)) or 1.0
        floor = float(np.nanmin(shown)) - 0.04 * span  # the axis' lower end (see the range below), less a margin
        ceiling = float(np.nanmax(shown)) + 0.5 * span * LABEL_GAP  # half a label above the highest curve
        heights = _spread([e[0] for e in ends], span * LABEL_GAP, floor, ceiling)
        for (_y, text, color), placed in zip(ends, heights, strict=True):
            annotations.append(dict(
                x=1.0, y=placed, xref="paper", yref="y", text=text, xanchor="left", xshift=END_GAP, yanchor="middle",
                showarrow=False, align="left", font=dict(size=SMALL, color=color),
            ))  # fmt: skip
    shapes.append(_hline(1, dict(color=colors["rule"], width=charts.REFERENCE_WIDTH)))
    # Reference lines across both rows (the strip shares the x-axis): spot, forward, strikes.
    for k in strikes:
        if lo <= k <= hi:
            shapes.append(_vline(k, dict(color=colors["muted"], width=charts.REFERENCE_WIDTH, dash="dot")))
    for x, label in _strike_labels(strikes, (lo, hi), strike_text):
        annotations.append(dict(
            x=x, y=1.0, xref="x", yref="y domain", text=label, showarrow=False, yanchor="top",
            font=dict(size=SMALL, color=colors["muted"]), bgcolor=colors["surface"],
        ))  # fmt: skip
    shapes.append(_vline(spot, dict(color=colors["muted"], width=charts.REFERENCE_WIDTH, dash="dash")))
    annotations.append(dict(
        x=spot, y=1.0, xref="x", yref="y domain", text=f"S {strike_text(spot)}", showarrow=False, xanchor="right",
        align="right", width=_label_width(f"S {strike_text(spot)}"), yanchor="top", yshift=-14, xshift=-2,
        font=dict(size=SMALL, color=colors["muted"]),
    ))  # fmt: skip
    if forward is not None:
        shapes.append(_vline(forward, dict(color=colors["accent"], width=charts.REFERENCE_WIDTH)))
        annotations.append(dict(
            x=forward, y=1.0, xref="x", yref="y domain", text=" F", showarrow=False, yanchor="top", yshift=-14,
            xanchor="left", font=dict(size=SMALL, color=colors["accent"]),
        ))  # fmt: skip
    if breakevens:
        fig.add_trace(
            go.Scatter(
                x=[x for x, _ in breakevens],
                y=[0.0] * len(breakevens),
                mode="markers",
                name="Breakeven",
                showlegend=False,
                text=[label for _, label in breakevens],
                marker=dict(size=9, color=colors["accent"], symbol="diamond"),
                hovertemplate="%{text}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        for x, label in breakevens:
            # Outward and above the zero line, where the curves of a short-premium structure are not; inward near the
            # plot's edge. On the surface colour, so that a strike line through it does not cross the text.
            left = x < spot
            if (x - lo) < BREAKEVEN_EDGE * (hi - lo) or (hi - x) < BREAKEVEN_EDGE * (hi - lo):
                left = (hi - x) < (x - lo)
            level, _, change = label.partition(" (")
            lines = (level, change.rstrip(")"))
            annotations.append(dict(
                x=x, y=0.0, xref="x", yref="y", text="<br>".join(lines), showarrow=False,
                xanchor="right" if left else "left", align="right" if left else "left", yanchor="bottom",
                xshift=-5 if left else 5, yshift=8, width=max(_label_width(t) for t in lines),
                font=dict(size=SMALL, color=colors["accent"]), bgcolor=colors["surface"],
            ))  # fmt: skip
    if straddle is not None:
        s_lo, s_hi, s_label = straddle
        for x in (s_lo, s_hi):
            shapes.append(dict(
                type="line", x0=x, x1=x, xref="x", y0=0.0, y1=0.035, yref="y domain",
                line=dict(color=colors["accent"], width=2),
            ))  # fmt: skip
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers", name=s_label, hoverinfo="skip",
                marker=dict(symbol="line-ns-open", size=10, color=colors["accent"], line=dict(width=2)),
            ),
            row=1,
            col=1,
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
        shapes.append(_hline(2, dict(color=colors["rule"], width=charts.REFERENCE_WIDTH)))
        fig.update_yaxes(title_text=title, row=2, col=1, nticks=3, tickformat=",.0f")
    fig.update_layout(shapes=shapes, annotations=annotations)
    height = PNL_HEIGHT + (STRIP_HEIGHT if strip is not None else 0)
    charts.style(fig, colors, height)
    # Headroom above the curves for the two lines of labels at the plot's top (strikes; S and F).
    shown = np.concatenate([np.asarray(c.values, dtype=float)[inside] for c in curves] + [np.zeros(1)])
    low, high = float(np.nanmin(shown)), float(np.nanmax(shown))
    pad = (high - low) or max(abs(high), 1.0)
    fig.update_yaxes(
        title_text="P&L ($)", row=1, col=1, tickformat="$,.0f", range=[low - 0.06 * pad, high + HEADROOM * pad]
    )
    fig.update_xaxes(range=[lo, hi], showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot")
    fig.update_xaxes(title_text="Spot" + (f" · {note}" if note else ""), row=rows, col=1, tickformat=",.0f")
    fig.update_layout(margin=dict(t=40, r=charts.MARGIN["r"] + column), legend=dict(font=dict(size=11)))
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
