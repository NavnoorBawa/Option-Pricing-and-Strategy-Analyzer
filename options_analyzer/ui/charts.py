"""Plotly figure builders.

Pure functions (data in, figure out) so they are easy to test. Figures are
rendered with ``st.plotly_chart(..., theme="streamlit")``, which adapts them to
the light or dark app theme; only semantic colours are set explicitly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go

from .format import money

CALL = "#3B82F6"
PUT = "#F97316"
PROFIT = "#22C55E"
LOSS = "#EF4444"
NEUTRAL = "#94A3B8"
ACCENT = "#A855F7"
SERIES = ("#3B82F6", "#F97316", "#22C55E", "#A855F7", "#EAB308", "#14B8A6", "#EC4899", "#64748B")

# Solid colours so white cell labels stay readable in both the light and dark themes.
DIVERGING = [[0.0, "#B91C1C"], [0.5, "#475569"], [1.0, "#15803D"]]


def apply_layout(fig: go.Figure, title: str | None, x_title: str, y_title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left") if title else None,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def _vline(
    fig: go.Figure, x: float, label: str, color: str = NEUTRAL, dash: str = "dot", position: str = "top"
) -> None:
    fig.add_vline(x=x, line=dict(color=color, dash=dash, width=1), annotation_text=label, annotation_position=position)


def sensitivity_chart(x, call_y, put_y, x_title: str, y_title: str, current_x: float, strike_x: float | None = None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=call_y, name="Call", line=dict(color=CALL, width=2.5)))
    fig.add_trace(go.Scatter(x=x, y=put_y, name="Put", line=dict(color=PUT, width=2.5)))
    # Labels sit inside the top of the plot, each on the far side of its line from the other line, so they
    # cannot overlap each other or the tick labels; a strike equal to the current value gets one shared label.
    if strike_x is None:
        _vline(fig, current_x, "current", position="top right")
    elif abs(strike_x - current_x) <= 1e-9 * max(1.0, abs(current_x)):
        _vline(fig, current_x, "current = strike", position="top right")
    else:
        current_side, strike_side = ("top left", "top right") if current_x < strike_x else ("top right", "top left")
        _vline(fig, current_x, "current", position=current_side)
        _vline(fig, strike_x, "strike", dash="dash", position=strike_side)
    return apply_layout(fig, None, x_title, y_title)


def _decimals_for_step(values, minimum: int, maximum: int = 8, scale: float = 1.0) -> int:
    """Decimals (at least ``minimum``) that resolve the grid step of ``values * scale`` to a tenth of a step
    and keep every label distinct."""
    scaled = np.asarray(values, dtype=float) * scale
    step = np.min(np.abs(np.diff(scaled))) if scaled.size > 1 else 0.0
    decimals = max(minimum, int(np.ceil(1.0 - np.log10(step)))) if step > 0 else minimum
    while decimals < maximum and len({f"{v:.{decimals}f}" for v in scaled}) < len(set(scaled.tolist())):
        decimals += 1
    return min(decimals, maximum)


def price_heatmap(spots, vols, values, title: str, colorscale: str = "Viridis"):
    """Option value on a (volatility x spot) grid with the values written in each cell.

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
    fig = go.Figure(
        go.Heatmap(
            x=[f"{s:,.{spot_dp}f}" for s in spots],
            y=[f"{v * 100:.{vol_dp}f}%" for v in vols],
            z=values,
            text=np.vectorize(lambda v: f"{v:,.{value_dp}f}")(values),
            texttemplate="%{text}",
            colorscale=colorscale,
            showscale=False,
            hovertemplate=f"Spot %{{x}}<br>Vol %{{y}}<br>Value %{{z:,.{value_dp + 2}f}}<extra></extra>",
        )
    )
    fig = apply_layout(fig, title, "Spot price", "Volatility", height=460)
    fig.update_layout(hovermode="closest")
    # Plotly would otherwise parse labels such as "20.0%" as numbers and drop the formatting.
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    return fig


def convergence_chart(series: dict, reference: float | None, reference_label: str, title: str | None = None):
    """Tree value against the number of steps; ``series`` maps a name to ``(steps, values)``."""
    fig = go.Figure()
    for i, (name, (steps, values)) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=steps, y=values, name=name, mode="lines", line=dict(color=SERIES[i], width=2)))
    if reference is not None:
        fig.add_hline(
            y=reference,
            line=dict(color=NEUTRAL, dash="dash"),
            annotation_text=f"{reference_label} {reference:,.4f}",
            annotation_position="bottom right",
        )
    return apply_layout(fig, title, "Number of steps", "Option value ($)")


def exercise_boundary_chart(boundaries: dict, strike: float):
    """Critical spot price over time; ``boundaries`` maps names starting with "Call" or "Put" to ``(days, spots)``."""
    fig = go.Figure()
    for name, (days, spots) in boundaries.items():
        color = CALL if name.lower().startswith("call") else PUT
        fig.add_trace(go.Scatter(x=days, y=spots, name=name, mode="lines", line=dict(color=color, width=2.5)))
    fig.add_hline(y=strike, line=dict(color=NEUTRAL, dash="dash"), annotation_text="strike")
    return apply_layout(fig, None, "Days from today", "Critical spot price ($)")


def paths_chart(times_days, paths, strike: float, mean_path=None, mean_label: str = "Risk-neutral mean (forward)"):
    fig = go.Figure()
    for i, path in enumerate(paths):
        fig.add_trace(
            go.Scatter(
                x=times_days,
                y=path,
                mode="lines",
                line=dict(width=1, color=SERIES[i % len(SERIES)]),
                opacity=0.35,
                showlegend=False,
                hoverinfo="skip",
            )
        )
    if mean_path is not None:
        fig.add_trace(go.Scatter(x=times_days, y=mean_path, name=mean_label, line=dict(color=ACCENT, width=3)))
    fig.add_hline(y=strike, line=dict(color=NEUTRAL, dash="dash"), annotation_text="strike")
    fig = apply_layout(fig, None, "Days from today", "Underlying price")
    fig.update_layout(hovermode="closest")
    return fig


def _legend_below(fig: go.Figure) -> go.Figure:
    """Horizontal legend under the plot, where the Plotly toolbar cannot cover it."""
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0.0))
    return fig


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
):
    """Histogram from pre-binned counts (keeps the payload small for large samples).

    The axis spans the 0.05 % to 99.95 % quantiles and every marker line. Outcomes beyond it are left out of the
    bars rather than piled into the end bins, and a note above the chart says how many there are and how far they go.
    Densities are relative to the whole sample, so the bars integrate to the share of outcomes shown.
    """
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
            marker_color=NEUTRAL,
            opacity=0.75,
            name="Simulated",
            hovertemplate="%{x:,.2f}<br>density %{y:.4g}<extra></extra>",
        )
    )
    if pdf is not None:
        fig.add_trace(go.Scatter(x=pdf[0], y=pdf[1], name="Log-normal density", line=dict(color=ACCENT, width=2)))
    # With several lines the left-most label goes to the left of its line and the others to the right, so that close
    # lines (ES and VaR) do not print their labels on top of each other.
    for rank, (x, label, color) in enumerate(sorted(lines, key=lambda line: line[0])):
        position = "top" if len(lines) == 1 else ("top left" if rank == 0 else "top right")
        _vline(fig, x, label, color=color, dash="dash", position=position)
    fig = apply_layout(fig, None, x_title, "Density")
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
            font=dict(size=11, color=NEUTRAL),
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
):
    """P&L profile. ``curves`` holds ``(values, name, colour, dash)``; the first is shaded.

    The ±1σ band is named in the legend rather than annotated, so its label cannot collide with the spot label.
    The breakeven hover shows ``breakeven_decimals`` decimals, like the breakevens written under the cards.
    """
    fig = go.Figure()
    if band is not None:
        fig.add_vrect(x0=band[0], x1=band[1], fillcolor=NEUTRAL, opacity=0.10, line_width=0)
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="±1σ range at expiry",
                marker=dict(symbol="square", size=14, color=NEUTRAL, opacity=0.35),
                hoverinfo="skip",
            )
        )
    first = np.asarray(curves[0][0], dtype=float)
    fig.add_trace(
        go.Scatter(
            x=spots,
            y=np.where(first > 0, first, 0.0),
            fill="tozeroy",
            mode="none",
            fillcolor="rgba(34,197,94,0.18)",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=spots,
            y=np.where(first < 0, first, 0.0),
            fill="tozeroy",
            mode="none",
            fillcolor="rgba(239,68,68,0.18)",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    for values, name, color, dash in curves:
        fig.add_trace(
            go.Scatter(
                x=spots, y=values, name=name, line=dict(color=color, width=2.5, dash=dash), hovertemplate="%{y:$,.2f}"
            )
        )
    fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1))
    _vline(fig, spot, "spot", color=ACCENT, dash="dash")
    for k in strikes:
        fig.add_vline(x=k, line=dict(color=NEUTRAL, width=1, dash="dot"))
    if breakevens:
        fig.add_trace(
            go.Scatter(
                x=breakevens,
                y=[0.0] * len(breakevens),
                mode="markers",
                name="Breakeven",
                marker=dict(size=10, color=ACCENT, symbol="diamond"),
                hovertemplate=f"Breakeven %{{x:$,.{breakeven_decimals}f}}<extra></extra>",
            )
        )
    fig = apply_layout(fig, None, "Underlying price", "Profit / loss ($)", height=500)
    return _legend_below(fig)


def pnl_cell_text(value: float, bound: float) -> str:
    """A heatmap cell value, short enough to stay legible in a narrow cell.

    ``bound`` is the largest absolute value in the grid: cents below $10, then one decimal, whole dollars below
    $10,000, and thousands (``-21.6k``, ``216k``) or millions (``-2.16M``) above.
    """
    if bound < 1e4:
        return _plain(f"{value:,.{0 if bound >= 100 else (1 if bound >= 10 else 2)}f}")
    scale, suffix, decimals = (1e6, "M", 2) if bound >= 1e6 else (1e3, "k", 1 if bound < 1e5 else 0)
    text = _plain(f"{value / scale:,.{decimals}f}")
    return "0" if text.strip("0.,") == "" else text + suffix


def pnl_heatmap(x_labels, y_labels, z, x_title: str, y_title: str, colorbar: bool = True):
    """P&L grid with the value in each cell (see :func:`pnl_cell_text`; the hover shows it to the cent).

    ``colorbar=False`` leaves out the colour scale, which repeats the cell values, to give the cells its width.
    """
    z = np.asarray(z, dtype=float)
    bound = float(np.nanmax(np.abs(z))) or 1.0
    fig = go.Figure(
        go.Heatmap(
            x=x_labels,
            y=y_labels,
            z=z,
            zmin=-bound,
            zmax=bound,
            colorscale=DIVERGING,
            text=np.vectorize(lambda v: pnl_cell_text(v, bound))(z),
            texttemplate="%{text}",
            textfont=dict(color="white"),
            colorbar=dict(title="P&L ($)"),
            showscale=colorbar,
            hovertemplate=f"{x_title} %{{x}}<br>{y_title} %{{y}}<br>P&L %{{z:$,.2f}}<extra></extra>",
        )
    )
    fig = apply_layout(fig, None, x_title, y_title, height=440)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(type="category")
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
):
    fig = go.Figure()
    for i, (name, y) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=x, y=y, name=name, line=dict(color=SERIES[i % len(SERIES)], width=2.5)))
    if hline is not None:
        fig.add_hline(y=hline, line=dict(color=NEUTRAL, width=1))
    if vline is not None:
        _vline(fig, vline[0], vline[1], color=ACCENT, dash="dash")
    fig = apply_layout(fig, None, x_title, y_title)
    if reverse_x:
        fig.update_xaxes(autorange="reversed")
    return _legend_below(fig)


def bar_chart(names, values, x_title: str):
    colors = [PROFIT if v >= 0 else LOSS for v in values]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[money(v, 0 if abs(v) >= 100 else 2) for v in values],
            textposition="auto",
            hovertemplate="%{y}<br>%{x:$,.2f}<extra></extra>",
        )
    )
    fig = apply_layout(fig, None, x_title, "", height=max(320, 42 * len(names) + 80))
    fig.update_layout(hovermode="closest", yaxis=dict(autorange="reversed"))
    return fig


def surface_3d(moneyness, maturities, z, title: str, colorscale: str):
    # A cube-shaped scene, a camera far enough out and a colour bar under the plot keep every axis
    # inside a half-width column (Plotly's automatic aspect otherwise stretches some surfaces out of view).
    fig = go.Figure(
        go.Surface(
            x=moneyness,
            y=maturities,
            z=z,
            colorscale=colorscale,
            colorbar=dict(
                title=dict(text="Volatility", side="top"),
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
    fig.update_layout(
        title=dict(text=title, x=0.0),
        height=560,
        margin=dict(l=0, r=0, t=40, b=70),
        scene=dict(
            xaxis_title="Moneyness K/S",
            yaxis_title="Maturity (years)",
            zaxis=dict(title="Volatility", tickformat=".0%"),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.9, y=-1.9, z=1.1), center=dict(x=0.0, y=0.0, z=-0.12)),
        ),
    )
    return fig


def smile_chart(
    x,
    slices: Sequence[tuple[str, np.ndarray, np.ndarray]],
    x_title: str,
    spot_line: float | None = 1.0,
    quotes: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray] | None] | None = None,
):
    """Implied (solid) and local (dashed) volatility per slice, sharing a colour.

    ``quotes`` (aligned with ``slices``) adds the market quotes as markers:
    ``(moneyness, quoted implied vol, fitted implied vol)`` or ``None`` for a slice without quotes.
    """
    fig = go.Figure()
    for i, (label, implied, local) in enumerate(slices):
        color = SERIES[i % len(SERIES)]
        fig.add_trace(go.Scatter(x=x, y=implied, name=f"Implied · {label}", line=dict(color=color, width=2.5)))
        fig.add_trace(go.Scatter(x=x, y=local, name=f"Local · {label}", line=dict(color=color, width=2, dash="dash")))
        if quotes is not None and quotes[i] is not None:
            qx, qiv, fitted = quotes[i]
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
        _vline(fig, spot_line, "ATM", position="bottom")
    fig = apply_layout(fig, None, x_title, "Volatility", height=480)
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0.0))
    return fig
