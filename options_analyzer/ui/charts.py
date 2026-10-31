"""Plotly figure builders and the desk house style.

Pure functions (data in, figure out) so they are easy to test. The app draws every chart with :func:`show`, which
passes the desk Plotly config and the chart's description for screen readers, and keeps ``theme="streamlit"``, so
fonts, backgrounds and the categorical colours follow the light or dark app theme.

Colours come from :mod:`.theme`. Each builder takes a ``palette`` (default: the current mode's semantic colours)
for the colours that carry a meaning: profit and loss, call and put, reference lines. Series that are only told
apart are left to the theme's categorical colours, in trace order. The Streamlit theme overrides a template's grid
and axis colours, so :func:`style` sets them on each figure; the ``desk`` template holds the rest of the house
style (unified hover, a crosshair, outside ticks, a legend at the top left), which the theme leaves alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# As desk_theme, so that show() can take st.plotly_chart's theme argument under its own name.
from . import theme as desk_theme
from .format import signed

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

#: Heights (px) of a primary chart and of a smaller, secondary one.
HEIGHT, HEIGHT_SECONDARY = 360, 260
#: Plot margins (px): the axes grow them for their tick labels and titles (``automargin``).
MARGIN = dict(l=8, r=8, t=28, b=8)
#: Width (px) of data lines and of reference lines, and the size of reference-line labels.
LINE_WIDTH, REFERENCE_WIDTH, LABEL_SIZE = 2, 1, 11

pio.templates["desk"] = go.layout.Template(
    layout=dict(
        hovermode="x unified",
        separators=".,",
        xaxis=dict(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikedash="dot",
            spikethickness=1,
            ticks="outside",
            ticklen=4,
            showline=True,
        ),
        yaxis=dict(ticks="outside", ticklen=4, showline=True),
        legend=dict(orientation="h", x=0.0, xanchor="left", y=1.0, yanchor="bottom"),
    )
)
#: The template of every figure: Streamlit's (which the app theme fills in) with the desk keys on top.
TEMPLATE = "streamlit+desk" if "streamlit" in pio.templates else "desk"

#: Plotly config for :func:`show`: no Plotly logo, no selection tools (the charts have no selection events) or
#: autoscale (double-click resets the zoom), and downloads as SVG, which stays sharp at any size. The toolbar keeps
#: Plotly's default, ``displayModeBar: "hover"`` (shown while the pointer is over the chart), which the config no
#: longer repeats: on a touch screen Plotly turns "hover" into "always" (plotly.js plot_api), so there the desk
#: stylesheet hides it instead (``@media (hover: none)``, through the ``chart_<key>`` container of :func:`show`).
PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "svg", "filename": "options-analyzer-chart"},
}

#: Colour names of earlier versions (``charts.CALL`` and so on), resolved in the current mode by :func:`__getattr__`.
_ALIASES = {"CALL": "call", "PUT": "put", "PROFIT": "profit", "LOSS": "loss", "NEUTRAL": "muted", "ACCENT": "accent"}


def __getattr__(name: str) -> str:
    """``charts.CALL``, ``charts.LOSS`` and the other colour names: the token of the current light/dark mode."""
    if name in _ALIASES:
        return desk_theme.palette()[_ALIASES[name]]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def translucent(color: str, alpha: float) -> str:
    """``rgba(...)`` of a ``#RRGGBB`` colour at opacity ``alpha``."""
    value = color.lstrip("#")
    red, green, blue = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha:g})"


def show(
    fig: go.Figure,
    *,
    key: str,
    alt: str,
    container: DeltaGenerator | None = None,
    theme: Literal["streamlit"] | None = "streamlit",
    **kwargs: Any,
) -> Any:
    """Draw ``fig`` with the desk Plotly config and an accessible name: the one way the app draws a chart.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure, usually from one of the builders here.
    key : str
        The chart's element key. It also names the SVG download: ``options-analyzer-<key>.svg``.
    alt : str
        A short, specific description of the chart for screen readers, such as "Call and put delta against spot
        price" (Streamlit's ``alt``: the chart's accessible name). It must not be blank.
    container : DeltaGenerator, optional
        Where to draw the chart (a column or a horizontal container, say); default: the current position.
    theme : {"streamlit", None}, default "streamlit"
        ``"streamlit"`` lets the app theme fill in the fonts, backgrounds and categorical colours of the light or
        dark mode; ``None`` draws the figure exactly as built.
    **kwargs
        Further keyword arguments of ``st.plotly_chart`` (``width`` in px, ``on_select``, ...).

    Returns
    -------
    DeltaGenerator or PlotlyState
        What ``st.plotly_chart`` returns.

    Raises
    ------
    ValueError
        If ``alt`` is blank.

    Examples
    --------
    Outside a running app Streamlit draws nothing:

    >>> fig = sensitivity_chart([90.0, 100.0, 110.0], [3.0, 10.0, 18.0], [12.0, 5.0, 2.0], "Spot", "Value", 100.0)
    >>> _ = show(fig, key="sensitivity_chart", alt="Call and put value against spot price")
    """
    if not alt.strip():
        raise ValueError(f"chart {key!r} needs a description (alt) for screen readers")
    config = PLOTLY_CONFIG | {"toImageButtonOptions": {"format": "svg", "filename": f"options-analyzer-{key}"}}
    # Each chart sits in its own keyed container, the hook of the desk stylesheet's touch-screen rule (UI spec 2.7
    # item 10). A chart of fixed width gets a container of that width, so that a row of them keeps its layout.
    width = kwargs.get("width")
    box = (container or st).container(key=f"chart_{key}", width=width if isinstance(width, int) else "stretch")
    return box.plotly_chart(fig, key=key, config=config, theme=theme, alt=alt, **kwargs)


def style(fig: go.Figure, palette: dict[str, str] | None = None, height: int | None = None) -> go.Figure:
    """Apply the desk house style: template, height, margins, full series names in the hover label, and the grid,
    axis-line, crosshair and text colours of ``palette``. The zero lines are off: charts that need one draw it.

    The axis text (tick labels and titles), and any label without a colour of its own, is in the muted colour: 6.36:1
    on the dark background and 5.73:1 on the light one, where the theme's own faded grey was 3.62:1 at 10.5 px.
    (Legends and chart titles keep the theme's text colours.)
    """
    colors = palette or desk_theme.palette()
    fig.update_layout(
        template=TEMPLATE,
        height=height or fig.layout.height or HEIGHT,
        margin=MARGIN,
        hoverlabel=dict(namelength=-1),
        font=dict(color=colors["muted"]),
    )
    text = dict(tickfont=dict(color=colors["muted"]), title_font=dict(color=colors["muted"]))
    fig.update_xaxes(
        linecolor=colors["rule"], gridcolor=colors["rule"], spikecolor=colors["muted"], zeroline=False, **text
    )
    fig.update_yaxes(linecolor=colors["rule"], gridcolor=colors["rule"], zeroline=False, **text)
    return fig


def apply_layout(
    fig: go.Figure,
    title: str | None,
    x_title: str,
    y_title: str,
    height: int = HEIGHT,
    palette: dict[str, str] | None = None,
) -> go.Figure:
    """:func:`style` plus a title (top left; the legend then moves to the top right) and the axis titles."""
    style(fig, palette, height)
    fig.update_layout(xaxis_title_text=x_title, yaxis_title_text=y_title)  # keeps the title font of style()
    if title:
        fig.update_layout(title=dict(text=title, x=0.0, xanchor="left"), legend=dict(x=1.0, xanchor="right"))
    return fig


def _legend_below(fig: go.Figure) -> go.Figure:
    """Horizontal legend at the bottom of the figure, under the x-axis title, where the Plotly toolbar cannot cover
    it.

    The legend is placed in the figure's own coordinates (``yref="container"``), so Plotly keeps room for it under
    the tick labels and the axis title at any chart height and for any number of legend rows. Placed a fraction of
    the plot height under the plot, it ran into the axis title on the shorter charts.
    """
    fig.update_layout(legend=dict(orientation="h", x=0.0, xanchor="left", yref="container", y=0.0, yanchor="bottom"))
    return fig


def _vline(
    fig: go.Figure,
    x: float,
    label: str,
    colors: dict[str, str],
    dash: str = "dot",
    position: str = "top",
    color: str | None = None,
) -> None:
    """A thin reference line with a small label, in the muted colour unless ``color`` says otherwise."""
    color = color or colors["muted"]
    fig.add_vline(
        x=x,
        line=dict(color=color, dash=dash, width=REFERENCE_WIDTH),
        annotation_text=label,
        annotation_position=position,
        annotation_font=dict(size=LABEL_SIZE, color=color),
    )


def _hline(fig: go.Figure, y: float, colors: dict[str, str], label: str | None = None, **kwargs) -> None:
    """A thin horizontal reference line (dashed when it has a label) in the muted colour."""
    line = dict(color=colors["muted"], width=REFERENCE_WIDTH, dash="dash" if label else "solid")
    if label:
        kwargs |= dict(annotation_text=label, annotation_font=dict(size=LABEL_SIZE, color=colors["muted"]))
    fig.add_hline(y=y, line=line, **kwargs)


def scenario_styles(n: int, palette: dict[str, str] | None = None) -> list[tuple[str, str]]:
    """``(colour, dash)`` for ``n`` ordered scenarios, from the largest fall to the largest rise (spot -1σ to
    +1σ): falls in the put orange, rises in the call blue and the middle ("unchanged") in the muted grey, with the
    dash marking the size of the move (solid for the largest, then dashed, dotted, dash-dotted). Use it instead of
    the categorical colours where the order of the series means something."""
    colors = palette or desk_theme.palette()
    dashes = ("solid", "dash", "dot", "dashdot")
    half = (n - 1) / 2
    styles = []
    for i in range(n):
        offset = i - half  # negative for falls, positive for rises
        if abs(offset) < 0.25:
            styles.append((colors["muted"], "solid"))
            continue
        rank = int(np.ceil(half - abs(offset) - 1e-9))  # 0 for the largest move
        styles.append((colors["put" if offset < 0 else "call"], dashes[min(rank, len(dashes) - 1)]))
    return styles


def pair_styles(n: int, palette: dict[str, str] | None = None) -> list[tuple[str, str]]:
    """``(colour, dash)`` for ``n`` pairs of series, such as theta / 365 and the 1-day decay at each date: pair ``i``
    in the ``i``-th categorical colour of the palette's mode, its first series solid and its second dashed (as the
    smile chart pairs implied and local volatility)."""
    series = desk_theme.categorical(palette or desk_theme.palette())
    return [(series[i % len(series)], dash) for i in range(n) for dash in ("solid", "dash")]


def sensitivity_chart(
    x,
    call_y,
    put_y,
    x_title: str,
    y_title: str,
    current_x: float,
    strike_x: float | None = None,
    palette: dict[str, str] | None = None,
    dashed: tuple[str, str, Sequence[float], Sequence[float]] | None = None,
):
    """Call and put values or Greeks against one input. The call is the first trace and the put the second, so
    they take the theme's first two categorical colours (the call and put colours of the mode).

    ``dashed = (solid name, dashed name, call_y, put_y)`` adds a second quantity in the same units as dashed lines
    after the two, each in its option's colour (the categorical colour of the palette's mode, as the smile chart's
    implied and local volatility share one), and names every line by its quantity and option: ``Theta / 365 ·
    Call``."""
    colors = palette or desk_theme.palette()
    fig = go.Figure()
    if dashed is None:
        fig.add_trace(go.Scatter(x=x, y=call_y, name="Call", line=dict(width=LINE_WIDTH)))
        fig.add_trace(go.Scatter(x=x, y=put_y, name="Put", line=dict(width=LINE_WIDTH)))
    else:
        solid, second, call_dashed, put_dashed = dashed
        call_color, put_color = desk_theme.categorical(colors)[:2]
        for style, name, call, put in (("solid", solid, call_y, put_y), ("dash", second, call_dashed, put_dashed)):
            for option, y, color in (("Call", call, call_color), ("Put", put, put_color)):
                line = dict(color=color, width=LINE_WIDTH, dash=style)
                fig.add_trace(go.Scatter(x=x, y=y, name=f"{name} · {option}", line=line))
    # Labels sit inside the top of the plot, each on the far side of its line from the other line, so they
    # cannot overlap each other or the tick labels; a strike equal to the current value gets one shared label.
    if strike_x is None:
        _vline(fig, current_x, "current", colors, position="top right")
    elif abs(strike_x - current_x) <= 1e-9 * max(1.0, abs(current_x)):
        _vline(fig, current_x, "current = strike", colors, position="top right")
    else:
        current_side, strike_side = ("top left", "top right") if current_x < strike_x else ("top right", "top left")
        _vline(fig, current_x, "current", colors, position=current_side)
        _vline(fig, strike_x, "strike", colors, dash="dash", position=strike_side)
    return apply_layout(fig, None, x_title, y_title, palette=colors)


def _decimals_for_step(values, minimum: int, maximum: int = 8, scale: float = 1.0) -> int:
    """Decimals (at least ``minimum``) that resolve the grid step of ``values * scale`` to a tenth of a step
    and keep every label distinct."""
    scaled = np.asarray(values, dtype=float) * scale
    step = np.min(np.abs(np.diff(scaled))) if scaled.size > 1 else 0.0
    decimals = max(minimum, int(np.ceil(1.0 - np.log10(step)))) if step > 0 else minimum
    while decimals < maximum and len({f"{v:.{decimals}f}" for v in scaled}) < len(set(scaled.tolist())):
        decimals += 1
    return min(decimals, maximum)


def price_heatmap(spots, vols, values, title: str, colorscale=None, palette: dict[str, str] | None = None):
    """Option value on a (volatility x spot) grid with the values written in each cell.

    Without a ``colorscale`` the cells take the theme's sequential ramp (``chartSequentialColors``) of the current
    mode, faint for low values; a Plotly colour scale (a name such as ``"Blues"`` or a list) replaces it.

    Label and cell precision follow the grid: low-priced underlyings (penny stocks, FX rates) and very low
    volatilities get more decimals, so neighbouring rows and columns never share a label (Plotly would merge
    them) and the cell values stay readable. There is no colour scale: it would repeat the cell values and take
    the width the cells need (Plotly shrinks the cell text to fit its cell).
    """
    values = np.asarray(values, dtype=float)
    spot_dp = _decimals_for_step(spots, 2)
    vol_dp = _decimals_for_step(vols, 1, scale=100.0)
    largest = float(np.nanmax(np.abs(values))) if values.size else 0.0
    value_dp = int(np.clip(2 - np.floor(np.log10(largest)), 2, 8)) if largest > 0 else 2
    scale = dict(autocolorscale=True) if colorscale is None else dict(colorscale=colorscale)
    fig = go.Figure(
        go.Heatmap(
            x=[f"{s:,.{spot_dp}f}" for s in spots],
            y=[f"{v * 100:.{vol_dp}f}%" for v in vols],
            z=values,
            text=_cell_text(values, lambda v: f"{v:,.{value_dp}f}"),
            texttemplate="%{text}",
            showscale=False,
            hovertemplate=f"Spot %{{x}}<br>Vol %{{y}}<br>Value %{{z:,.{value_dp + 2}f}}<extra></extra>",
            **scale,
        )
    )
    fig = apply_layout(fig, title, "Spot price", "Volatility", height=460, palette=palette)
    fig.update_layout(hovermode="closest")
    # Plotly would otherwise parse labels such as "20.0%" as numbers and drop the formatting.
    fig.update_xaxes(type="category", showspikes=False)
    fig.update_yaxes(type="category")
    return fig


def convergence_chart(
    series: dict,
    reference: float | None,
    reference_label: str,
    title: str | None = None,
    palette: dict[str, str] | None = None,
):
    """Tree value against the number of steps (log scale, so the first steps are not squeezed together);
    ``series`` maps a name to ``(steps, values)``."""
    colors = palette or desk_theme.palette()
    fig = go.Figure()
    for name, (steps, values) in series.items():
        fig.add_trace(go.Scatter(x=steps, y=values, name=name, mode="lines", line=dict(width=LINE_WIDTH)))
    if reference is not None:
        _hline(fig, reference, colors, f"{reference_label} {reference:,.4f}", annotation_position="bottom right")
    fig = apply_layout(fig, title, "Number of steps", "Option value ($)", palette=colors)
    # Labelled round step counts: Plotly's default log ticks print bare minor digits ("2 3 4 5") in small type.
    steps = [s for s_values, _ in series.values() for s in np.atleast_1d(s_values)]
    lo, hi = (min(steps), max(steps)) if steps else (1, 10)
    ticks = [t for t in (10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000) if lo <= t <= hi] or [lo, hi]
    fig.update_xaxes(type="log", tickmode="array", tickvals=ticks, ticktext=[f"{t:,.0f}" for t in ticks])
    return fig


def exercise_boundary_chart(boundaries: dict, strike: float, palette: dict[str, str] | None = None):
    """Critical spot price over time; ``boundaries`` maps names starting with "Call" or "Put" to ``(days, spots)``.
    The colours follow the option type (a put boundary drawn alone is still the put colour)."""
    colors = palette or desk_theme.palette()
    fig = go.Figure()
    for name, (days, spots) in boundaries.items():
        color = colors["call"] if name.lower().startswith("call") else colors["put"]
        fig.add_trace(go.Scatter(x=days, y=spots, name=name, mode="lines", line=dict(color=color, width=LINE_WIDTH)))
    _hline(fig, strike, colors, "strike")
    return apply_layout(fig, None, "Days from today", "Critical spot price ($)", palette=colors)


def paths_chart(
    times_days,
    paths,
    strike: float,
    mean_path=None,
    mean_label: str = "Risk-neutral mean (forward)",
    palette: dict[str, str] | None = None,
):
    colors = palette or desk_theme.palette()
    fig = go.Figure()
    for path in paths:
        fig.add_trace(
            go.Scatter(
                x=times_days,
                y=path,
                mode="lines",
                line=dict(width=1),
                opacity=0.35,
                showlegend=False,
                hoverinfo="skip",
            )
        )
    if mean_path is not None:
        fig.add_trace(
            go.Scatter(x=times_days, y=mean_path, name=mean_label, line=dict(color=colors["accent"], width=2.5))
        )
    _hline(fig, strike, colors, "strike")
    fig = apply_layout(fig, None, "Days from today", "Underlying price", palette=colors)
    fig.update_layout(hovermode="closest")
    return fig


def _cell_text(values, label) -> list[list[str]]:
    """``label`` applied to every cell of a 2-D grid. A list comprehension rather than ``np.vectorize``, whose object
    ufunc can report a floating-point flag that something else raised in the same thread as a RuntimeWarning."""
    return [[label(float(v)) for v in row] for row in np.asarray(values, dtype=float)]


def _plain(text: str) -> str:
    """Normalise a negative zero such as ``-0`` or ``-0.00`` (it reads as a small loss)."""
    stripped = text.lstrip("-")
    return stripped if stripped.strip("0.,") == "" else text


#: Height (px) above the plot area kept for the marker-line labels; the histogram's tail note sits above it.
NOTE_SHIFT = 22


def histogram_chart(
    values,
    x_title: str,
    lines: Sequence[tuple[float, str, str]] = (),
    bins: int = 120,
    pdf=None,
    value_text=lambda v: f"{v:,.2f}",
    palette: dict[str, str] | None = None,
):
    """Histogram from pre-binned counts (keeps the payload small for large samples).

    The axis spans the 0.05 % to 99.95 % quantiles and every marker line. Outcomes beyond it are left out of the
    bars rather than piled into the end bins, and a note above the chart says how many there are and how far they go.
    Densities are relative to the whole sample, so the bars integrate to the share of outcomes shown.
    """
    colors = palette or desk_theme.palette()
    values = np.asarray(values, dtype=float)
    lo, hi = np.percentile(values, [0.05, 99.95])
    marks = [float(x) for x, _, _ in lines if np.isfinite(x)]
    if marks:
        lo, hi = min(lo, *marks), max(hi, *marks)
    if hi <= lo:
        lo, hi = values.min() - 1.0, values.max() + 1.0
    pad = 0.02 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    counts, edges = np.histogram(values[(values >= lo) & (values <= hi)], bins=bins, range=(lo, hi))
    centres = 0.5 * (edges[:-1] + edges[1:])
    density = counts / (values.size * np.diff(edges))
    below, above = int(np.sum(values < lo)), int(np.sum(values > hi))
    outside = []
    if below:
        outside.append(f"{below:,} outcomes below the plotted range (lowest {value_text(values.min())})")
    if above:
        outside.append(f"{above:,} outcomes above the plotted range (highest {value_text(values.max())})")
    fig = go.Figure(
        go.Bar(
            x=centres,
            y=density,
            marker_color=colors["muted"],
            opacity=0.75,
            name="Simulated",
            hovertemplate="%{x:,.2f}<br>density %{y:.4g}<extra></extra>",
        )
    )
    if pdf is not None:
        fig.add_trace(
            go.Scatter(x=pdf[0], y=pdf[1], name="Log-normal density", line=dict(color=colors["accent"], width=2))
        )
    # With several lines the left-most label goes to the left of its line and the others to the right, so that close
    # lines (ES and VaR) do not print their labels on top of each other.
    for rank, (x, label, color) in enumerate(sorted(lines, key=lambda line: line[0])):
        position = "top" if len(lines) == 1 else ("top left" if rank == 0 else "top right")
        _vline(fig, x, label, colors, dash="dash", position=position, color=color)
    fig = apply_layout(fig, None, x_title, "Density", palette=colors)
    fig.update_layout(bargap=0.02, hovermode="closest")
    if outside:
        # A small note above the marker labels, where it cannot cover the bars or the lines: one short line per
        # side, so that it fits a phone-width or half-width chart.
        fig.add_annotation(
            text="<br>".join(outside),
            name="outside",
            x=0.0,
            xref="paper",
            xanchor="left",
            y=1.0,
            yref="paper",
            yanchor="bottom",
            yshift=NOTE_SHIFT,
            align="left",
            showarrow=False,
            font=dict(size=LABEL_SIZE, color=colors["muted"]),
        )
        fig.update_layout(margin=dict(t=NOTE_SHIFT + 16 * len(outside) + 6))
    return _legend_below(fig)


def payoff_chart(
    spots,
    curves: Sequence[tuple[np.ndarray, str, str, str]],
    spot: float,
    strikes,
    breakevens,
    band=None,
    breakeven_decimals: int = 2,
    palette: dict[str, str] | None = None,
):
    """P&L profile. ``curves`` holds ``(values, name, colour, dash)``; the first is shaded, in the profit colour
    above zero and the loss colour below it.

    The ±1σ band is named in the legend rather than annotated, so its label cannot collide with the spot label.
    The breakeven hover shows ``breakeven_decimals`` decimals, like the breakevens written under the cards.
    """
    colors = palette or desk_theme.palette()
    fig = go.Figure()
    if band is not None:
        fig.add_vrect(x0=band[0], x1=band[1], fillcolor=colors["muted"], opacity=0.10, line_width=0)
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="±1σ range at expiry",
                marker=dict(symbol="square", size=14, color=colors["muted"], opacity=0.35),
                hoverinfo="skip",
            )
        )
    first = np.asarray(curves[0][0], dtype=float)
    for part, role in ((np.where(first > 0, first, 0.0), "profit"), (np.where(first < 0, first, 0.0), "loss")):
        fig.add_trace(
            go.Scatter(
                x=spots,
                y=part,
                fill="tozeroy",
                mode="none",
                # A translucent fill colour: Plotly does not apply a trace's opacity to a fill without lines.
                fillcolor=translucent(colors[role], desk_theme.FILL_OPACITY),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    for values, name, color, dash in curves:
        fig.add_trace(
            go.Scatter(
                x=spots,
                y=values,
                name=name,
                line=dict(color=color, width=LINE_WIDTH, dash=dash),
                hovertemplate="%{y:+$,.2f}",
            )
        )
    _hline(fig, 0.0, colors)
    _vline(fig, spot, "spot", colors, dash="dash")
    for k in strikes:
        fig.add_vline(x=k, line=dict(color=colors["muted"], width=REFERENCE_WIDTH, dash="dot"))
    if breakevens:
        fig.add_trace(
            go.Scatter(
                x=breakevens,
                y=[0.0] * len(breakevens),
                mode="markers",
                name="Breakeven",
                marker=dict(size=10, color=colors["accent"], symbol="diamond"),
                hovertemplate=f"Breakeven %{{x:$,.{breakeven_decimals}f}}<extra></extra>",
            )
        )
    fig = apply_layout(fig, None, "Underlying price", "Profit / loss ($)", height=440, palette=colors)
    return _legend_below(fig)


def pnl_cell_text(value: float, bound: float) -> str:
    """A heatmap cell value, short enough to stay legible in a narrow cell.

    ``bound`` is the largest absolute value in the grid: cents below $10, then one decimal, whole dollars below
    $10,000, and thousands (``−21.6k``, ``+216k``) or millions (``−2.16M``) above, always with an explicit sign
    (U+2212 for minus) except at zero.
    """
    if bound < 1e4:
        return signed(value, 0 if bound >= 100 else (1 if bound >= 10 else 2))
    scale, suffix, decimals = (1e6, "M", 2) if bound >= 1e6 else (1e3, "k", 1 if bound < 1e5 else 0)
    text = signed(value / scale, decimals)
    return "0" if text.strip("0.,") == "" else text + suffix


def pnl_heatmap(
    x_labels,
    y_labels,
    z,
    x_title: str,
    y_title: str,
    colorbar: bool = True,
    palette: dict[str, str] | None = None,
):
    """P&L grid with the value in each cell (see :func:`pnl_cell_text`; the hover shows it to the cent).

    The cells take the theme's diverging ramp (``chartDivergingColors``, loss red to profit teal through the
    background colour) of the current mode, centred on zero. ``colorbar=False`` leaves out the colour scale,
    which repeats the cell values, to give the cells its width.
    """
    z = np.asarray(z, dtype=float)
    bound = float(np.nanmax(np.abs(z))) or 1.0
    fig = go.Figure(
        go.Heatmap(
            x=x_labels,
            y=y_labels,
            z=z,
            # Symmetric limits keep the diverging ramp whatever the signs: Plotly picks the sequential ramp when
            # every value has the same sign.
            zmin=-bound,
            zmax=bound,
            zmid=0.0,
            autocolorscale=True,
            text=_cell_text(z, lambda v: pnl_cell_text(v, bound)),
            texttemplate="%{text}",
            colorbar=dict(title="P&L ($)"),
            showscale=colorbar,
            hovertemplate=f"{x_title} %{{x}}<br>{y_title} %{{y}}<br>P&L %{{z:+$,.2f}}<extra></extra>",
        )
    )
    fig = apply_layout(fig, None, x_title, y_title, height=440, palette=palette)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(type="category", showspikes=False)
    fig.update_yaxes(type="category")
    return fig


def lines_chart(
    x,
    series: dict,
    x_title: str,
    y_title: str,
    hline: float | None = 0.0,
    reverse_x: bool = False,
    vline: tuple[float, str] | None = None,
    styles: Sequence[tuple[str, str]] | None = None,
    palette: dict[str, str] | None = None,
):
    """One line per series. ``styles`` gives each series a ``(colour, dash)``, such as :func:`scenario_styles`
    for ordered scenarios; without it the series take the theme's categorical colours in order."""
    colors = palette or desk_theme.palette()
    fig = go.Figure()
    for i, (name, y) in enumerate(series.items()):
        line: dict[str, object] = dict(width=LINE_WIDTH)
        if styles is not None:
            line |= dict(color=styles[i][0], dash=styles[i][1])
        fig.add_trace(go.Scatter(x=x, y=y, name=name, line=line))
    if hline is not None:
        _hline(fig, hline, colors)
    if vline is not None:
        _vline(fig, vline[0], vline[1], colors, dash="dash")
    fig = apply_layout(fig, None, x_title, y_title, palette=colors)
    if reverse_x:
        fig.update_xaxes(autorange="reversed")
    return _legend_below(fig)


def bar_chart(names, values, x_title: str, palette: dict[str, str] | None = None):
    """Horizontal P&L bars in the profit or loss colour, each labelled with its signed amount."""
    colors = palette or desk_theme.palette()
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=[colors["profit"] if v >= 0 else colors["loss"] for v in values],
            text=[signed(v, 0 if abs(v) >= 100 else 2, "$") for v in values],
            textposition="auto",
            hovertemplate="%{y}<br>%{x:+$,.2f}<extra></extra>",
        )
    )
    fig = apply_layout(fig, None, x_title, "", height=max(HEIGHT_SECONDARY, 42 * len(names) + 80), palette=colors)
    fig.update_layout(hovermode="closest", yaxis=dict(autorange="reversed"))
    fig.update_xaxes(showspikes=False)
    return fig


def surface_3d(moneyness, maturities, z, title: str, colorscale: str, palette: dict[str, str] | None = None):
    # A cube-shaped scene (Plotly's automatic aspect stretches some surfaces out of view) and a colour bar under
    # the plot. Plotly sizes a 3-D scene by the plot's height, not its width, so the camera distance fixes the
    # scene at about 270 px across: every axis, tick label and title stays inside a chart at least that wide
    # (a phone column is 328 px or more; the Surface tab wraps the two charts before they get
    # narrower). A margin does not help: the scene is clipped at its own domain, so a margin only moves the cut.
    # The axis and colour-bar text is in the muted colour, as in style().
    text = dict(color=(palette or desk_theme.palette())["muted"])
    fig = go.Figure(
        go.Surface(
            x=moneyness,
            y=maturities,
            z=z,
            colorscale=colorscale,
            colorbar=dict(
                title=dict(text="Volatility", side="top", font=text),
                tickfont=text,
                tickformat=".0%",
                orientation="h",
                thickness=12,
                len=0.8,
                x=0.5,
                xanchor="center",
                y=-0.02,
                yanchor="top",
            ),
            hovertemplate="K/S %{x:.3f}<br>T %{y:.3f} years<br>vol %{z:.2%}<extra></extra>",
            connectgaps=False,
        )
    )
    # No desk template: its hover mode, crosshair and ticks are for 2-D axes; the 3-D scene has its own.
    fig.update_layout(
        title=dict(text=title, x=0.0),
        height=560,
        margin=dict(l=0, r=0, t=MARGIN["t"], b=70),
        font=text,
        scene=dict(
            xaxis=dict(title=dict(text="Moneyness K/S", font=text), tickfont=text),
            yaxis=dict(title=dict(text="Maturity (years)", font=text), tickfont=text),
            zaxis=dict(title=dict(text="Volatility", font=text), tickfont=text, tickformat=".0%"),
            aspectmode="cube",
            camera=dict(eye=dict(x=2.47, y=-2.47, z=1.43), center=dict(x=0.0, y=0.0, z=-0.12)),
        ),
    )
    return fig


def smile_chart(
    x,
    slices: Sequence[tuple[str, np.ndarray, np.ndarray]],
    x_title: str,
    spot_line: float | None = 1.0,
    quotes: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray] | None] | None = None,
    palette: dict[str, str] | None = None,
    wings: Sequence[np.ndarray | None] | None = None,
):
    """Implied (solid) and local (dashed) volatility per slice, sharing a colour, and implied vol extrapolated beyond
    the quotes (dotted).

    ``quotes`` (aligned with ``slices``) adds the market quotes as markers:
    ``(moneyness, quoted implied vol, fitted implied vol)`` or ``None`` for a slice without quotes.
    ``wings`` (aligned with ``slices``) adds each implied-vol smile's extrapolated part as a dotted line (dashes are
    local vol's): implied vols on ``x``, NaN where the smile is not extrapolated; ``None`` or all NaN for none.
    """
    colors = palette or desk_theme.palette()
    series = desk_theme.categorical(colors)
    fig = go.Figure()
    for i, (label, implied, local) in enumerate(slices):
        color = series[i % len(series)]
        fig.add_trace(go.Scatter(x=x, y=implied, name=f"Implied · {label}", line=dict(color=color, width=LINE_WIDTH)))
        fig.add_trace(
            go.Scatter(x=x, y=local, name=f"Local · {label}", line=dict(color=color, width=LINE_WIDTH, dash="dash"))
        )
        if wings is not None and (wing := wings[i]) is not None and np.isfinite(wing).any():
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=wing,
                    name=f"Extrapolated · {label}",
                    line=dict(color=color, width=LINE_WIDTH, dash="dot"),
                    # The point that joins a wing to its smile is the smile's last quoted one, not extrapolated.
                    customdata=np.where(np.isfinite(np.asarray(implied, dtype=float)), "quoted", "extrapolated"),
                    hovertemplate="%{y:.2%} · %{customdata}",
                )
            )
        if quotes is not None and (quote := quotes[i]) is not None:
            qx, qiv, fitted = quote
            fig.add_trace(
                go.Scatter(
                    x=qx,
                    y=qiv,
                    mode="markers",
                    name=f"Quotes · {label}",
                    marker=dict(color=color, size=8, symbol="circle-open", line=dict(width=2)),
                    customdata=np.column_stack([(np.asarray(qiv) - np.asarray(fitted)) * 100]),
                    hovertemplate="quote %{y:.2%} · fit error %{customdata[0]:+.2f} vol pts",
                )
            )
    if spot_line is not None:
        # Inside the top of the plot beside the line, above the smile's trough: at the bottom the label was drawn
        # over the x-axis tick label "1" at every width.
        _vline(fig, spot_line, "ATM", colors, position="top right")
    fig = apply_layout(fig, None, x_title, "Volatility", height=480, palette=colors)
    fig.update_yaxes(tickformat=".0%")
    return _legend_below(fig)
