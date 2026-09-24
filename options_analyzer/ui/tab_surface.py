"""Volatility Surface tab: SSVI or market quotes -> implied and local volatility."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from .. import vol_surface as vs
from . import cache, charts
from .inputs import Inputs

SSVI = "Parametric (SSVI)"
QUOTES = "Market quotes (CSV)"
COLORSCALES = ("Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo")


def _ssvi_controls(inputs: Inputs) -> vs.VolSurface | None:
    base = inputs.volatility * 100
    lo, hi = 0.5, 500.0
    c = st.columns(3)
    short = c[0].number_input(
        "Short-term ATM vol (%)",
        lo,
        hi,
        float(np.clip(round(base * 1.2, 2), lo, hi)),
        0.5,
        key="ssvi_short",
        help="ATM implied vol as maturity → 0.",
    )
    long = c[1].number_input(
        "Long-term ATM vol (%)",
        lo,
        hi,
        float(np.clip(round(base, 2), lo, hi)),
        0.5,
        key="ssvi_long",
        help="ATM implied vol as maturity → ∞.",
    )
    kappa = c[2].number_input(
        "Term-structure speed κ",
        0.05,
        20.0,
        1.5,
        0.1,
        key="ssvi_kappa",
        help="How fast ATM variance moves from the short- to the long-term level.",
    )
    c = st.columns(3)
    rho = c[0].slider(
        "Skew ρ", -0.95, 0.95, -0.6, 0.05, key="ssvi_rho", help="Negative = downside skew (typical for equities)."
    )
    eta_max = 2.0 / (1.0 + abs(rho))
    eta = c[1].slider(
        "Smile curvature η",
        0.05,
        4.0,
        min(1.0, round(eta_max, 2)),
        0.05,
        key="ssvi_eta",
        help=f"Arbitrage-free for η ≤ 2/(1+|ρ|) = {eta_max:.2f}.",
    )
    gamma = c[2].slider(
        "Curvature decay γ",
        0.05,
        0.5,
        0.5,
        0.05,
        key="ssvi_gamma",
        help="How quickly the smile flattens with maturity (power-law SSVI).",
    )
    c = st.columns(2)
    m_range = c[0].slider("Moneyness range K/S", 0.3, 2.5, (0.7, 1.3), 0.05, key="ssvi_m")
    t_range = c[1].slider("Maturity range (years)", 0.02, 5.0, (0.08, 2.0), 0.01, key="ssvi_t")
    if m_range[1] - m_range[0] < 0.1 or t_range[1] - t_range[0] < 0.05:
        st.warning("Widen the moneyness or maturity range to see a surface.")
        return None
    params = vs.SSVIParams(short / 100, long / 100, kappa, rho, eta, gamma)
    return cache.ssvi_surface(params, inputs.rate, inputs.dividend_yield, m_range, t_range)


def _quote_controls(inputs: Inputs) -> tuple[vs.VolSurface | None, pd.DataFrame | None]:
    c1, c2 = st.columns([2, 1])
    upload = c1.file_uploader(
        "Option quotes (CSV)",
        type=["csv"],
        key="quotes_file",
        help="Columns: Strike, Maturity (years) or Days, ImpliedVol (decimal or %). "
        "At least two maturities with five or more strikes each.",
    )
    sample = vs.sample_quotes(inputs.spot, inputs.rate, inputs.dividend_yield)
    c2.download_button(
        "Download a sample file",
        sample.to_csv(index=False).encode(),
        "sample_option_quotes.csv",
        "text/csv",
        on_click="ignore",
        icon=":material/download:",
        key="sample_quotes",
    )
    smooth = c2.toggle(
        "Smooth quotes (GCV spline)",
        value=True,
        key="quotes_smooth",
        help="Cubic smoothing spline per maturity with the penalty chosen by generalised "
        "cross-validation. Turn off to interpolate the quotes exactly.",
    )
    if upload is None:
        st.info(
            "Upload a quotes file, or download the sample to see the expected format. "
            f"Quotes are interpreted with the sidebar spot ({inputs.spot:,.2f}), rate and dividend yield."
        )
        return None, None
    try:
        raw = pd.read_csv(io.BytesIO(upload.getvalue()))
        quotes, messages = vs.parse_quotes(raw)
        for message in messages:
            st.caption(f"ℹ️ {message}")
        surface = cache.market_surface(quotes, inputs.spot, inputs.rate, inputs.dividend_yield, smooth)
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty.")
        return None, None
    except UnicodeDecodeError:
        st.error("The uploaded file is not a UTF-8 text CSV.")
        return None, None
    except (vs.QuoteParseError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not build a surface from this file: {exc}")
        return None, None
    return surface, quotes


def _downloads(surface: vs.VolSurface, summary: pd.DataFrame, spot: float, quotes: pd.DataFrame | None) -> None:
    frames = surface.to_frames(spot)
    c = st.columns(3)
    c[0].download_button(
        "Implied vol grid (CSV)",
        frames["ImpliedVol"].to_csv().encode(),
        "implied_vol_surface.csv",
        "text/csv",
        on_click="ignore",
        icon=":material/download:",
        key="dl_iv",
    )
    c[1].download_button(
        "Local vol grid (CSV)",
        frames["LocalVol"].to_csv().encode(),
        "local_vol_surface.csv",
        "text/csv",
        on_click="ignore",
        icon=":material/download:",
        key="dl_lv",
    )
    buffer = io.BytesIO()
    try:
        with pd.ExcelWriter(buffer) as writer:
            frames["ImpliedVol"].to_excel(writer, sheet_name="ImpliedVol")
            frames["LocalVol"].to_excel(writer, sheet_name="LocalVol")
            summary.to_excel(writer, sheet_name="Summary", index=False)
            if quotes is not None:
                quotes.to_excel(writer, sheet_name="Quotes", index=False)
    except ImportError:
        c[2].caption("Install `xlsxwriter` or `openpyxl` for Excel export.")
        return
    c[2].download_button(
        "Everything (Excel)",
        buffer.getvalue(),
        "volatility_surface.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        on_click="ignore",
        icon=":material/download:",
        key="dl_xlsx",
    )


@st.fragment
def render(inputs: Inputs) -> None:
    source = st.segmented_control("Surface source", [SSVI, QUOTES], default=SSVI, required=True, key="surface_source")
    quotes = None
    if source == SSVI:
        surface = _ssvi_controls(inputs)
    else:
        surface, quotes = _quote_controls(inputs)
    if surface is None:
        return

    total = surface.local_vol.size
    valid = int(np.isfinite(surface.local_vol).sum())
    c = st.columns(4)
    c[0].metric(
        "Grid", f"{len(surface.maturities)} × {len(surface.moneyness)}", border=True, help="Maturities × strikes."
    )
    c[1].metric(
        "Calendar-arbitrage points",
        surface.calendar_violations,
        border=True,
        help="Total implied variance decreasing in maturity at fixed forward moneyness (∂w/∂T < 0).",
    )
    c[2].metric(
        "Butterfly-arbitrage points",
        surface.butterfly_violations,
        border=True,
        help="Durrleman's condition g(y) ≤ 0: negative risk-neutral density.",
    )
    c[3].metric(
        "Local vol defined",
        f"{valid / total:.0%}",
        border=True,
        help="Share of grid points with a valid local volatility.",
    )
    for note in surface.notes:
        st.warning(note)
    if valid < total:
        st.caption("Local volatility is left blank where the surface has static arbitrage — it is undefined there.")

    colorscale = st.selectbox("Colour scale", COLORSCALES, key="surface_colors")
    left, right = st.columns(2)
    left.plotly_chart(
        charts.surface_3d(surface.moneyness, surface.maturities, surface.implied_vol, "Implied volatility", colorscale),
        key="surf_iv",
    )
    right.plotly_chart(
        charts.surface_3d(
            surface.moneyness, surface.maturities, surface.local_vol, "Local volatility (Dupire / Gatheral)", colorscale
        ),
        key="surf_lv",
    )

    st.subheader("Smiles")
    labels = [f"{t:.2f}y" for t in surface.maturities]
    n = len(labels)
    default = sorted({labels[0], labels[n // 2], labels[-1]}, key=labels.index)
    picked = st.multiselect("Maturities", labels, default=default, max_selections=6, key="smile_pick")
    slices = [(lab, surface.implied_vol[labels.index(lab)], surface.local_vol[labels.index(lab)]) for lab in picked]
    if slices:
        st.plotly_chart(charts.smile_chart(surface.moneyness, slices, "Moneyness K/S"), key="smiles")

    st.subheader("Term structure and skew")
    summary = vs.surface_summary(surface)
    series = {"ATM implied vol": summary["ATM implied vol"], "ATM local vol": summary["ATM local vol"]}
    fig = charts.lines_chart(summary["Maturity (years)"], series, "Maturity (years)", "Volatility", hline=None)
    fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig, key="term_structure")
    shown = summary.assign(**{c: summary[c] * 100 for c in ("ATM implied vol", "ATM local vol")})
    st.dataframe(
        shown,
        hide_index=True,
        column_config={
            "Maturity (years)": st.column_config.NumberColumn(format="%.3f"),
            "ATM implied vol": st.column_config.NumberColumn(format="%.2f%%"),
            "ATM local vol": st.column_config.NumberColumn(format="%.2f%%"),
            "Skew 90-110 (vol pts)": st.column_config.NumberColumn(format="%.2f"),
            "Butterfly (vol pts)": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption(
        "ATM means K = S. Skew is the 90% minus 110% moneyness implied vol; butterfly is their average "
        "minus ATM. Near the money, local vol skew is roughly twice the implied skew (Gatheral's rule of thumb)."
    )

    st.subheader("Download")
    _downloads(surface, summary, inputs.spot, quotes)
