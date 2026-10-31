"""Figure builders of the Pricing and IV tabs (UI spec 3.1 items 4 and 7, 3.5 item 4), in the desk house style of
:mod:`.charts`: pure functions, data in and figure out, drawn with :func:`.charts.show`.

* :func:`profile_chart`: a value or Greek of the call and the put against spot, vol or time, the engine's curves solid
  and, under the American engine, the closed form dotted as ``BSM ref``; the one-day decay dashed beside theta.
* :func:`error_chart`: a tree's absolute error against its step count, log-log, with a reference slope.
* :func:`iv_price_chart`: the model value against vol, with the quote's bid, mid and ask and their implied vols.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

from . import charts
from . import theme as desk_theme
from .format import MINUS, number, premium, vol


@dataclass(frozen=True)
class Line:
    """One curve of :func:`profile_chart`: its name in the legend, its values, the option it belongs to (``"call"``
    or ``"put"``, which sets its colour) and how it is drawn: ``"engine"`` (solid, 2 px), ``"ref"`` (dotted, 1 px:
    the closed form under the American engine) or ``"decay"`` (dashed, 2 px: the one-day decay beside theta)."""

    name: str
    y: Sequence[float]
    option: str
    role: str = "engine"


_DASH = {"engine": "solid", "ref": "dot", "decay": "dash"}
_WIDTH = {"engine": charts.LINE_WIDTH, "ref": charts.REFERENCE_WIDTH + 0.5, "decay": charts.LINE_WIDTH}


def _text(value: str) -> str:
    return value.replace("-", MINUS)


def profile_chart(
    x: Sequence[float],
    lines: Sequence[Line],
    *,
    x_title: str,
    y_title: str,
    current: tuple[float, str],
    references: Sequence[tuple[float, str, str]] = (),
    palette: dict[str, str] | None = None,
) -> go.Figure:
    """The Pricing tab's profile: each of ``lines`` against ``x``, a marker on every engine curve at the current x,
    and labelled reference lines.

    Parameters
    ----------
    x : sequence of float
        The x values, ascending (spot, vol in %, or days to expiry).
    lines : sequence of Line
        The curves, drawn in order.
    x_title, y_title : str
        Axis titles, with their units.
    current : (float, str)
        The current x and its label (``S 5,800.00``), drawn as a dashed muted line with the label at its top.
    references : sequence of (float, str, str)
        Further vertical lines: ``(x, label, kind)`` with kind ``"strike"`` (dotted, muted) or ``"forward"`` (solid
        1 px, accent). Lines outside the x range are left out.
    palette : dict, optional
        The mode's colours (default: the current mode's).

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    >>> fig = profile_chart([90.0, 100.0, 110.0], [Line("Call", [1.0, 4.0, 11.0], "call")], x_title="Spot",
    ...                     y_title="Value ($)", current=(100.0, "S 100.00"))
    >>> [trace.name for trace in fig.data]
    ['Call', 'Call now']
    """
    colors = palette or desk_theme.palette()
    xs = np.asarray(x, dtype=float)
    fig = go.Figure()
    here, label = current
    for line in lines:
        color = colors["call" if line.option == "call" else "put"]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=np.asarray(line.y, dtype=float),
                name=line.name,
                mode="lines",
                line=dict(color=color, width=_WIDTH[line.role], dash=_DASH[line.role]),
            )
        )
    # The current x on each engine curve, interpolated (the grid need not hold it): a marker the eye can find at once.
    for line in lines:
        if line.role != "engine":
            continue
        y = np.asarray(line.y, dtype=float)
        finite = np.isfinite(y)
        if not finite.any() or not xs[0] <= here <= xs[-1]:
            continue
        fig.add_trace(
            go.Scatter(
                x=[here],
                y=[float(np.interp(here, xs[finite], y[finite]))],
                name=f"{line.name} now",
                mode="markers",
                marker=dict(color=colors["call" if line.option == "call" else "put"], size=8),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    lo, hi = float(xs[0]), float(xs[-1])
    charts._vline(fig, here, _text(label), colors, dash="dash", position="top left")
    for value, text, kind in references:
        if not lo <= value <= hi or abs(value - here) <= 1e-9 * max(1.0, abs(here)):
            continue
        if kind == "forward":
            charts._vline(fig, value, _text(text), colors, dash="solid", position="top right", color=colors["accent"])
        else:
            charts._vline(fig, value, _text(text), colors, dash="dot", position="top right")
    fig = charts.apply_layout(fig, None, x_title, y_title, palette=colors)
    # Plotly groups thousands only from five digits: without this the spot axis reads 5400 beside "S 5,800.00".
    fig.update_xaxes(range=[lo, hi], separatethousands=True)
    fig.update_yaxes(separatethousands=True)
    return fig


def error_chart(
    series: dict[str, tuple[Sequence[float], Sequence[float], str, str]],
    *,
    reference: str,
    palette: dict[str, str] | None = None,
) -> go.Figure:
    """Absolute error of tree values against their step count, both axes logarithmic, with a guide of slope −1.

    Parameters
    ----------
    series : dict
        ``name -> (steps, absolute errors, option, dash)``: option ``"call"`` or ``"put"`` sets the colour, and dash
        tells the lattices apart (``"solid"`` for Leisen-Reimer, ``"dot"`` for Cox-Ross-Rubinstein). Zero errors
        (a tree that hits the reference to the last digit) are left out, as a log axis cannot show them.
    reference : str
        What the errors are measured against, for the y-axis title (``BSM``, ``ALO high``).
    palette : dict, optional
        The mode's colours.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    colors = palette or desk_theme.palette()
    fig = go.Figure()
    floor = []
    for name, (steps, errors, option, dash) in series.items():
        n = np.asarray(steps, dtype=float)
        e = np.abs(np.asarray(errors, dtype=float))
        keep = np.isfinite(e) & (e > 0)
        floor.append((n[keep], e[keep]))
        fig.add_trace(
            go.Scatter(
                x=n[keep],
                y=e[keep],
                name=name,
                mode="lines",
                line=dict(color=colors["call" if option == "call" else "put"], width=charts.LINE_WIDTH, dash=dash),
            )
        )
    # A guide of slope -1 (first order: the error halves as the steps double), through the largest first error.
    starts = [(n[0], e[0]) for n, e in floor if len(n)]
    ends = [n[-1] for n, e in floor if len(n)]
    if starts:
        n0, e0 = max(starts, key=lambda p: p[1])
        n1 = max(ends)
        fig.add_trace(
            go.Scatter(
                x=[n0, n1],
                y=[e0, e0 * n0 / n1],
                name="slope −1 (1/n)",
                mode="lines",
                line=dict(color=colors["muted"], width=charts.REFERENCE_WIDTH, dash="dash"),
            )
        )
    fig = charts.apply_layout(fig, None, "Steps", f"|Tree − {reference}| ($)", palette=colors)
    ticks = [10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000]
    fig.update_xaxes(type="log", tickmode="array", tickvals=ticks, ticktext=[number(t, 0) for t in ticks])
    fig.update_yaxes(type="log", exponentformat="power")
    return fig


def iv_price_chart(
    vols: Sequence[float],
    values: Sequence[float],
    *,
    quotes: Sequence[tuple[str, float, float]],
    floor: float | None,
    name: str,
    palette: dict[str, str] | None = None,
) -> go.Figure:
    """The IV tab's chart: the model value against vol, a horizontal line at each quoted price, a marker where it
    meets the curve (its implied vol), and the prices below the no-arbitrage floor shaded.

    The lines are named in the legend with their price and implied vol (``Bid 97.20 · 15.34%``), not labelled on the
    plot: a bid and an ask 0.17 vol points apart would print their labels on top of each other (the SPX default).

    Parameters
    ----------
    vols : sequence of float
        Vols in percent, ascending: the x range of the chart.
    values : sequence of float
        The model value at each vol ($ per share).
    quotes : sequence of (str, float, float)
        ``(label, price, implied vol in percent)`` for the bid, mid and ask (or the single price); a NaN implied vol
        (a price outside the bounds) draws the price line alone.
    floor : float or None
        The no-arbitrage lower bound ($). The band below it is shaded muted and labelled while it reaches the data
        (a price below it, or the curve near zero vol); None or 0 draws none.
    name : str
        The curve's name (``Model value (BSM)``).
    palette : dict, optional
        The mode's colours.

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    >>> fig = iv_price_chart([10.0, 20.0, 30.0], [4.0, 8.0, 12.0], quotes=[("Price", 8.0, 20.0)], floor=None,
    ...                      name="Model value (BSM)")
    >>> [trace.name for trace in fig.data]
    ['Model value (BSM)', 'Price 8.00 · 20.00%', 'IV price']
    """
    colors = palette or desk_theme.palette()
    xs = np.asarray(vols, dtype=float)
    ys = np.asarray(values, dtype=float)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=xs, y=ys, name=name, mode="lines", line=dict(color=colors["call"], width=charts.LINE_WIDTH))
    )
    finite = ys[np.isfinite(ys)].tolist()
    prices = [p for _, p, _ in quotes if np.isfinite(p)]
    data = [*finite, *prices]
    bottom = min(data) if data else 0.0
    if floor and floor > bottom:
        top = max(data)
        fig.add_hrect(
            y0=bottom - 0.05 * (top - bottom + 1e-9),
            y1=floor,
            fillcolor=charts.translucent(colors["muted"], 0.12),
            line_width=0,
            annotation_text="below the no-arbitrage floor",
            annotation_position="bottom right",
            annotation_font=dict(size=charts.LABEL_SIZE, color=colors["muted"]),
        )
    # Each quote reads off the curve as on paper: across from its price to the curve, then down to its vol.
    pad = 0.04 * ((max(data) - bottom) if data else 1.0) or 1.0
    y_low = bottom - pad
    dashes = {"Bid": "dot", "Mid": "dash", "Ask": "dashdot", "Price": "dash"}
    for label, price, iv in quotes:
        if not np.isfinite(price):
            continue
        shown = f"{label} {premium(price)} · " + (vol(iv / 100.0) if np.isfinite(iv) else "no IV")
        inside = np.isfinite(iv) and xs[0] <= iv <= xs[-1]
        x, y = ([xs[0], iv, iv], [price, price, y_low]) if inside else ([xs[0], xs[-1]], [price, price])
        fig.add_trace(
            go.Scatter(
                x=x, y=y, name=_text(shown), mode="lines", hoverinfo="skip",
                line=dict(color=colors["muted"], width=charts.REFERENCE_WIDTH + 0.5, dash=dashes.get(label, "dash")),
            )
        )  # fmt: skip
        if inside:
            fig.add_trace(
                go.Scatter(
                    x=[iv], y=[price], mode="markers", name=f"IV {label.lower()}", showlegend=False,
                    marker=dict(color=colors["accent"], size=8, symbol="diamond"),
                    hovertemplate=f"{label}: %{{x:.2f}}%<extra></extra>",
                )
            )  # fmt: skip
    fig = charts.apply_layout(fig, None, "Vol (%)", "Option value ($ per share)", palette=colors)
    fig.update_xaxes(range=[float(xs[0]), float(xs[-1])])
    if data:
        fig.update_yaxes(range=[y_low, max(data) + pad], separatethousands=True)
    return fig
