"""Figure builders of the Surface tab (UI spec 3.4): the smile against its quotes, the term structure and the 3-D
implied surface.

Pure functions, data in and figure out, in the house style of :mod:`.charts` (its template, :func:`.charts.style`
and the semantic colours of :mod:`.theme`): calls in the call colour, puts in the put colour, the forward a solid
accent line, extrapolated parts dashed and muted, and every curve named in its legend.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import charts
from . import theme as desk_theme

#: The smile's x axes (``sf_x``), by label: what each plots, and its title.
X_AXES = {"K/F (%)": "K/F (%)", "k = ln K/F": "k = ln(K/F)", "Call Δ": "Call delta", "Strike": "Strike"}
#: The Call Δ axis runs from deep in the money (puts' side) on the left to the wing on the right, in 5-delta steps.
DELTA_RANGE = (0.90, 0.10)
#: Height (px) of the smile's residual panel, under the primary chart's 360.
RESIDUAL_HEIGHT = 100


@dataclass(frozen=True)
class SmileData:
    """One expiry's smile and quotes, in strikes (the builder maps them onto the chosen axis).

    ``fit_strikes`` and ``fit_vols`` trace the fitted smile; ``fit_inside`` says which points lie within the quoted
    range (solid) and which are extrapolated (dashed). ``quotes`` hold the used quotes (``strike``, ``call``: bool,
    ``iv_bid``, ``iv_ask``, ``iv_mid``, ``fit_iv``), ``excluded`` the others shown (``strike``, ``iv``, ``call``,
    ``status``). ``call_delta`` maps a strike to its call delta at its fitted vol (the ``Call Δ`` axis).
    """

    forward: float
    fit_strikes: np.ndarray
    fit_vols: np.ndarray
    fit_inside: np.ndarray
    quoted_range: tuple[float, float] | None
    note: str
    quotes: dict | None = None
    excluded: dict | None = None
    violations: tuple[tuple[float, str], ...] = ()


def x_values(strikes, forward: float, axis: str, call_delta=None) -> np.ndarray:
    """Strikes on the smile's x axis: K/F in percent, ln(K/F), the call delta (``call_delta(strikes)``) or the
    strike itself.

    >>> x_values(np.array([5800.0]), 5800.0, "K/F (%)").tolist(), x_values(np.array([5800.0]), 5800.0, "k = ln K/F")
    ([100.0], array([0.]))
    """
    strikes = np.asarray(strikes, dtype=float)
    if axis == "K/F (%)":
        return 100.0 * strikes / forward
    if axis == "k = ln K/F":
        return np.log(strikes / forward)
    if axis == "Call Δ":
        assert call_delta is not None
        return np.asarray(call_delta(strikes), dtype=float)
    return strikes


def smile_fit(data: SmileData, axis: str = "K/F (%)", *, call_delta=None, palette: dict[str, str] | None = None):
    """The smile of one expiry over its quotes, with the residual panel under it (UI spec 3.4).

    Row 1: each used quote as a vertical bid-to-ask bar in implied vol with its mid as a tick (calls in the call
    colour, puts in the put colour), other quotes as hollow muted markers, the fit as a solid line within the quotes
    and a dashed muted one beyond them, a solid accent line at the forward and dotted lines at the quoted range.
    Row 2: each used quote's residual, (σ_mid − σ_fit) ÷ its half-spread in vol, with the ±1 band (inside the bid/ask).
    The note (expiry, time, RMSE, share in band, quotes used) sits inside the plot at the top left.
    """
    colors = palette or desk_theme.palette()
    has_quotes = data.quotes is not None and len(data.quotes["strike"])
    fig = (
        make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.05)
        if has_quotes
        else go.Figure()
    )
    where = {"row": 1, "col": 1} if has_quotes else {}

    def x_of(strikes):
        return x_values(strikes, data.forward, axis, call_delta)

    # The fit: solid inside the quotes, dashed and muted beyond (each segment joins the next at the range's edge).
    inside = np.asarray(data.fit_inside, dtype=bool)
    x_fit, y_fit = x_of(data.fit_strikes), 100.0 * np.asarray(data.fit_vols, dtype=float)
    beyond = ~inside
    edge = np.zeros_like(beyond)  # the last point within the quotes next to each wing, so that the wing joins
    edge[1:] |= beyond[:-1]
    edge[:-1] |= beyond[1:]
    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=np.where(beyond | (edge & inside), y_fit, np.nan) if inside.any() else y_fit,
            name="Fit, extrapolated" if inside.any() else "Fit",
            mode="lines",
            line=dict(color=colors["muted"] if inside.any() else colors["text"], width=charts.LINE_WIDTH, dash="dash"),
            hovertemplate="fit %{y:.2f}% (extrapolated)<extra></extra>",
        ),
        **where,
    )
    if inside.any():
        fig.add_trace(
            go.Scatter(
                x=x_fit,
                y=np.where(inside, y_fit, np.nan),
                name="Fit",
                mode="lines",
                line=dict(color=colors["text"], width=charts.LINE_WIDTH),
                hovertemplate="fit %{y:.2f}%<extra></extra>",
            ),
            **where,
        )
    if data.excluded is not None and len(data.excluded["strike"]):
        ex = data.excluded
        fig.add_trace(
            go.Scatter(
                x=x_of(ex["strike"]),
                y=100.0 * np.asarray(ex["iv"], dtype=float),
                name="Excluded",
                mode="markers",
                marker=dict(color=colors["muted"], size=7, symbol="circle-open", line=dict(width=1.5)),
                customdata=np.column_stack([np.asarray(ex["strike"], dtype=float), np.asarray(ex["status"])]),
                hovertemplate="K %{customdata[0]:,.0f} · %{y:.2f}% · %{customdata[1]}<extra></extra>",
            ),
            **where,
        )
    if has_quotes:
        q = data.quotes
        assert q is not None
        for side, color in (("call", colors["call"]), ("put", colors["put"])):
            mine = np.asarray(q["call"], dtype=bool) == (side == "call")
            if not mine.any():
                continue
            bid, ask, mid = (100.0 * np.asarray(q[c], dtype=float)[mine] for c in ("iv_bid", "iv_ask", "iv_mid"))
            strikes = np.asarray(q["strike"], dtype=float)[mine]
            fig.add_trace(
                go.Scatter(
                    x=x_of(strikes),
                    y=mid,
                    name=f"{side.capitalize()}s, bid–ask",
                    mode="markers",
                    marker=dict(color=color, size=9, symbol="line-ew-open", line=dict(width=2, color=color)),
                    error_y=dict(
                        type="data", symmetric=False, array=ask - mid, arrayminus=mid - bid, width=0, thickness=1.5
                    ),
                    customdata=np.column_stack([strikes, bid, ask]),
                    hovertemplate="K %{customdata[0]:,.0f} · bid %{customdata[1]:.2f}% · mid %{y:.2f}% · "
                    "ask %{customdata[2]:.2f}%<extra></extra>",
                ),
                row=1,
                col=1,
            )
            half = 0.5 * (ask - bid)
            fit = 100.0 * np.asarray(q["fit_iv"], dtype=float)[mine]
            with np.errstate(divide="ignore", invalid="ignore"):
                residual = np.where(half > 0, (mid - fit) / half, np.nan)
            fig.add_trace(
                go.Bar(
                    x=x_of(strikes),
                    y=residual,
                    name=f"{side.capitalize()}s, residual",
                    marker_color=color,
                    showlegend=False,
                    customdata=strikes,
                    hovertemplate="K %{customdata:,.0f} · (mid − fit) / half-spread %{y:+.2f}<extra></extra>",
                ),
                row=2,
                col=1,
            )
        for level in (-1.0, 1.0):
            fig.add_hline(y=level, line=dict(color=colors["muted"], width=1, dash="dot"), row=2, col=1)
        fig.add_hline(y=0.0, line=dict(color=colors["rule"], width=1), row=2, col=1)
    # The forward and the quoted range.
    forward_x = float(x_of(np.array([data.forward]))[0])
    fig.add_vline(
        x=forward_x,
        line=dict(color=colors["accent"], width=charts.REFERENCE_WIDTH),
        annotation_text="F",
        annotation_position="top right",
        annotation_font=dict(size=charts.LABEL_SIZE, color=colors["accent"]),
        **where,
    )
    if data.quoted_range is not None:
        for edge in data.quoted_range:
            fig.add_vline(
                x=float(x_of(np.array([edge]))[0]),
                line=dict(color=colors["muted"], width=charts.REFERENCE_WIDTH, dash="dot"),
                **where,
            )
    for strike, kind in data.violations:
        fig.add_annotation(
            x=float(x_of(np.array([strike]))[0]),
            y=0.0,
            yref="y domain",
            text=f"▲ {kind}",
            showarrow=False,
            yanchor="bottom",
            font=dict(size=charts.LABEL_SIZE, color=colors["text"]),
        )
    fig.add_annotation(
        x=0.0,
        y=1.0,
        xref="x domain",
        yref="y domain",
        xanchor="left",
        yanchor="top",
        text=data.note,
        showarrow=False,
        font=dict(size=charts.LABEL_SIZE, color=colors["muted"]),
    )
    height = charts.HEIGHT + (RESIDUAL_HEIGHT if has_quotes else 0)
    charts.style(fig, colors, height)
    fig.update_layout(
        hovermode="closest", legend=dict(orientation="h", x=1.0, xanchor="right", y=1.0, yanchor="bottom")
    )
    fig.update_xaxes(showspikes=False)
    fig.update_yaxes(title_text="IV (%)", ticksuffix="", **({"row": 1, "col": 1} if has_quotes else {}))
    x_title = X_AXES[axis]
    if has_quotes:
        fig.update_yaxes(title_text="Resid.", row=2, col=1, zeroline=False)
        fig.update_xaxes(title_text=x_title, row=2, col=1)
    else:
        fig.update_xaxes(title_text=x_title)
    if axis == "Call Δ":
        fig.update_xaxes(range=list(DELTA_RANGE), dtick=0.05, tickformat=".2f")
    elif axis == "Strike":
        fig.update_xaxes(tickformat=",.0f")
    return fig


def term_chart(
    constant: Sequence[tuple[float, float]],
    listed: Sequence[dict],
    palette: dict[str, str] | None = None,
):
    """The term structure: ATMF vol at constant maturities (a line) with the listed expiries as markers, those whose
    variance window holds a holiday hollow; under it, the 25-delta risk reversal and butterfly by expiry (vp).

    ``constant`` is ``(days, vol)``; each of ``listed`` holds ``days``, ``atmf``, ``rr25``, ``bf25``, ``name`` and
    ``holiday`` (bool).
    """
    colors = palette or desk_theme.palette()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.06)
    days = np.array([d for d, _ in constant], dtype=float)
    vols = np.array([v for _, v in constant], dtype=float)
    fig.add_trace(
        go.Scatter(
            x=days,
            y=100 * vols,
            name="ATMF, constant maturity",
            mode="lines+markers",
            line=dict(color=colors["text"], width=charts.LINE_WIDTH),
            marker=dict(size=5),
            hovertemplate="%{x:.0f}d · ATMF %{y:.2f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )
    x = np.array([row["days"] for row in listed], dtype=float)
    holiday = np.array([row["holiday"] for row in listed], dtype=bool)
    names = np.array([row["name"] for row in listed], dtype=object)
    for mask, label, symbol in ((~holiday, "Listed expiry", "circle"), (holiday, "Holiday in window", "circle-open")):
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=x[mask],
                    y=100 * np.array([row["atmf"] for row in listed], dtype=float)[mask],
                    name=label,
                    mode="markers",
                    marker=dict(color=colors["accent"], size=8, symbol=symbol, line=dict(width=2)),
                    customdata=names[mask],
                    hovertemplate="%{customdata} · ATMF %{y:.2f}%<extra></extra>",
                ),
                row=1,
                col=1,
            )
    for key, name, color, dash in (
        ("rr25", "25Δ RR", colors["put"], "solid"),
        ("bf25", "25Δ BF", colors["call"], "dash"),
    ):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=100 * np.array([row[key] for row in listed], dtype=float),
                name=name,
                mode="lines+markers",
                line=dict(color=color, width=charts.LINE_WIDTH, dash=dash),
                marker=dict(size=5),
                customdata=names,
                hovertemplate=f"%{{customdata}} · {name} %{{y:+.2f}} vp<extra></extra>",
            ),
            row=2,
            col=1,
        )
    fig.add_hline(y=0.0, line=dict(color=colors["rule"], width=1), row=2, col=1)
    charts.style(fig, colors, charts.HEIGHT + 60)
    fig.update_layout(hovermode="closest", legend=dict(orientation="h", x=0.0, xanchor="left", y=1.0, yanchor="bottom"))
    fig.update_xaxes(type="log", showspikes=False)
    fig.update_xaxes(title_text="Days to expiry (log scale)", tickvals=[7, 14, 30, 60, 91, 182, 365, 730], row=2, col=1)
    fig.update_yaxes(title_text="ATMF (%)", row=1, col=1)
    fig.update_yaxes(title_text="vp", row=2, col=1)
    return fig


def implied_3d(
    moneyness_pct: np.ndarray, days: np.ndarray, vols: np.ndarray, palette: dict[str, str] | None = None
) -> go.Figure:
    """The implied surface in 3-D over K/F (%) and days to expiry, on the fixed sequential scale of the theme."""
    colors = palette or desk_theme.palette()
    ramp = desk_theme.SEQUENTIAL["light" if colors == desk_theme.SEMANTIC["light"] else "dark"]
    scale = [[i / (len(ramp) - 1), c] for i, c in enumerate(ramp)]
    text = dict(color=colors["muted"])
    fig = go.Figure(
        go.Surface(
            x=moneyness_pct,
            y=days,
            z=100 * vols,
            colorscale=scale,
            colorbar=dict(
                title=dict(text="IV (%)", side="top", font=text),
                tickfont=text,
                orientation="h",
                thickness=12,
                len=0.6,
                x=0.5,
                xanchor="center",
                y=-0.02,
                yanchor="top",
            ),
            hovertemplate="K/F %{x:.1f}%<br>%{y:.0f} days<br>IV %{z:.2f}%<extra></extra>",
            connectgaps=False,
        )
    )
    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=charts.MARGIN["t"], b=60),
        font=text,
        scene=dict(
            xaxis=dict(title=dict(text="K/F (%)", font=text), tickfont=text),
            yaxis=dict(title=dict(text="Days to expiry", font=text), tickfont=text),
            zaxis=dict(title=dict(text="IV (%)", font=text), tickfont=text),
            aspectmode="cube",
            camera=dict(eye=dict(x=2.0, y=-2.0, z=1.2), center=dict(x=0.0, y=0.0, z=-0.12)),
        ),
    )
    return fig


def durrleman_chart(y: np.ndarray, g: np.ndarray, inside: np.ndarray, palette: dict[str, str] | None = None):
    """Durrleman's g(k) of one smile: no butterfly arbitrage where it is positive (dashed beyond the quotes)."""
    colors = palette or desk_theme.palette()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y, y=np.where(inside, g, np.nan), name="g(k)", line=dict(color=colors["text"])))
    fig.add_trace(
        go.Scatter(
            x=y,
            y=np.where(inside, np.nan, g),
            name="g(k), extrapolated",
            line=dict(color=colors["muted"], dash="dash"),
        )
    )
    fig.add_hline(y=0.0, line=dict(color=colors["loss"], width=1, dash="dot"))
    charts.style(fig, colors, charts.HEIGHT_SECONDARY)
    fig.update_layout(xaxis_title_text="k = ln(K/F)", yaxis_title_text="g(k)")
    return fig
