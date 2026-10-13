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
    _vline(fig, current_x, "current")
    if strike_x is not None:
        _vline(fig, strike_x, "strike", dash="dash", position="bottom")
    return apply_layout(fig, None, x_title, y_title)


def price_heatmap(spots, vols, values, title: str, colorscale: str = "Viridis"):
    """Option value on a (volatility x spot) grid with the values written in each cell."""
    fig = go.Figure(
        go.Heatmap(
            x=[f"{s:,.2f}" for s in spots],
            y=[f"{v:.1%}" for v in vols],
            z=values,
            text=np.vectorize(lambda v: f"{v:,.2f}")(values),
            texttemplate="%{text}",
            colorscale=colorscale,
            colorbar=dict(title="Value"),
            hovertemplate="Spot %{x}<br>Vol %{y}<br>Value %{z:,.4f}<extra></extra>",
        )
    )
    fig = apply_layout(fig, title, "Spot price", "Volatility", height=460)
    fig.update_layout(hovermode="closest")
    # Plotly would otherwise parse labels such as "20.0%" as numbers and drop the formatting.
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    return fig


def convergence_chart(steps, series: dict, reference: float | None, reference_label: str):
    fig = go.Figure()
    for i, (name, values) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=steps, y=values, name=name, mode="lines", line=dict(color=SERIES[i], width=2)))
    if reference is not None:
        fig.add_hline(y=reference, line=dict(color=NEUTRAL, dash="dash"), annotation_text=reference_label)
    return apply_layout(fig, None, "Number of steps", "Option value")


def exercise_boundary_chart(boundaries: dict, strike: float):
    fig = go.Figure()
    for i, (name, (days, spots)) in enumerate(boundaries.items()):
        fig.add_trace(
            go.Scatter(x=days, y=spots, name=name, mode="lines", line=dict(color=(PUT, CALL)[i % 2], width=2.5))
        )
    fig.add_hline(y=strike, line=dict(color=NEUTRAL, dash="dash"), annotation_text="strike")
    return apply_layout(fig, None, "Days from today", "Critical spot price")


def paths_chart(times_days, paths, strike: float, mean_path=None):
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
        fig.add_trace(go.Scatter(x=times_days, y=mean_path, name="Mean path", line=dict(color=ACCENT, width=3)))
    fig.add_hline(y=strike, line=dict(color=NEUTRAL, dash="dash"), annotation_text="strike")
    fig = apply_layout(fig, None, "Days from today", "Underlying price")
    fig.update_layout(hovermode="closest")
    return fig


def histogram_chart(values, x_title: str, lines: Sequence[tuple[float, str, str]] = (), bins: int = 120, pdf=None):
    """Histogram from pre-binned counts (keeps the payload small for large samples)."""
    values = np.asarray(values, dtype=float)
    lo, hi = np.percentile(values, [0.05, 99.95])
    if hi <= lo:
        lo, hi = values.min() - 1.0, values.max() + 1.0
    counts, edges = np.histogram(np.clip(values, lo, hi), bins=bins, range=(lo, hi))
    centres = 0.5 * (edges[:-1] + edges[1:])
    density = counts / (counts.sum() * np.diff(edges))
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
    for x, label, color in lines:
        _vline(fig, x, label, color=color, dash="dash")
    fig = apply_layout(fig, None, x_title, "Density")
    fig.update_layout(bargap=0.02, hovermode="closest")
    return fig


def payoff_chart(
    spots, curves: Sequence[tuple[np.ndarray, str, str, str]], spot: float, strikes, breakevens, band=None
):
    """P&L profile. ``curves`` holds ``(values, name, colour, dash)``; the first is shaded."""
    fig = go.Figure()
    if band is not None:
        fig.add_vrect(
            x0=band[0],
            x1=band[1],
            fillcolor=NEUTRAL,
            opacity=0.10,
            line_width=0,
            annotation_text="±1σ at expiry",
            annotation_position="top left",
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
                hovertemplate="Breakeven %{x:$,.2f}<extra></extra>",
            )
        )
    return apply_layout(fig, None, "Underlying price", "Profit / loss ($)", height=480)


def pnl_heatmap(x_labels, y_labels, z, x_title: str, y_title: str):
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
            text=np.vectorize(lambda v: f"{v:,.0f}")(z),
            texttemplate="%{text}",
            textfont=dict(color="white"),
            colorbar=dict(title="P&L ($)"),
            hovertemplate=f"{x_title} %{{x}}<br>{y_title} %{{y}}<br>P&L %{{z:$,.2f}}<extra></extra>",
        )
    )
    fig = apply_layout(fig, None, x_title, y_title, height=440)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    return fig


def lines_chart(x, series: dict, x_title: str, y_title: str, hline: float | None = 0.0, reverse_x: bool = False):
    fig = go.Figure()
    for i, (name, y) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=x, y=y, name=name, line=dict(color=SERIES[i % len(SERIES)], width=2.5)))
    if hline is not None:
        fig.add_hline(y=hline, line=dict(color=NEUTRAL, width=1))
    fig = apply_layout(fig, None, x_title, y_title)
    if reverse_x:
        fig.update_xaxes(autorange="reversed")
    return fig


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
