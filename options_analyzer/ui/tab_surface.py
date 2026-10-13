"""Volatility Surface tab: SSVI or market quotes -> implied and local volatility."""

from __future__ import annotations

import functools
import importlib.util
import io
import zlib

import numpy as np
import pandas as pd
import streamlit as st

from .. import vol_surface as vs
from . import cache, charts
from .inputs import Inputs

SSVI = "Parametric (SSVI)"
QUOTES = "Market quotes (CSV)"
COLORSCALES = ("Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo")
VOL_LIMITS = (0.5, 500.0)


def _follow_sidebar_vol(inputs: Inputs) -> None:
    """Re-seed the SSVI ATM vols from the sidebar volatility whenever it changes (and on first render)."""
    base = inputs.volatility * 100
    lo, hi = VOL_LIMITS
    seeds = {
        "ssvi_short": float(np.clip(round(base * 1.2, 2), lo, hi)),
        "ssvi_long": float(np.clip(round(base, 2), lo, hi)),
    }
    changed = st.session_state.get("ssvi_vol_seed") != base
    st.session_state["ssvi_vol_seed"] = base
    for key, value in seeds.items():
        if changed or key not in st.session_state:
            st.session_state[key] = value


def _ssvi_controls(inputs: Inputs) -> vs.VolSurface | None:
    _follow_sidebar_vol(inputs)
    lo, hi = VOL_LIMITS
    c = st.columns(3)
    short = c[0].number_input(
        "Short-term ATM vol (%)",
        min_value=lo,
        max_value=hi,
        step=0.5,
        key="ssvi_short",
        help="ATM implied vol as maturity → 0. Starts at 1.2 × the sidebar volatility and follows it.",
    )
    long = c[1].number_input(
        "Long-term ATM vol (%)",
        min_value=lo,
        max_value=hi,
        step=0.5,
        key="ssvi_long",
        help="ATM implied vol as maturity → ∞. Starts at the sidebar volatility and follows it.",
    )
    kappa = c[2].number_input(
        "Term-structure speed κ",
        0.05,
        20.0,
        1.5,
        0.1,
        key="ssvi_kappa",
        help="How fast ATM variance moves from the short- to the long-term level (per year).",
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


def _quote_controls(inputs: Inputs) -> vs.VolSurface | None:
    c1, c2 = st.columns([2, 1])
    upload = c1.file_uploader(
        "Option quotes (CSV)",
        type=["csv"],
        key="quotes_file",
        help="Columns: Strike; Days (to expiry) or Maturity (years); and either ImpliedVol (decimal or %) or "
        "option prices (Price, or Bid and Ask) with a Type column saying call or put. At least two maturities "
        "with five or more strikes each.",
    )
    for label, prices, key in (
        ("Sample with implied vols", False, "sample_quotes"),
        ("Sample with prices", True, "sample_prices"),
    ):
        sample = vs.sample_quotes(inputs.spot, inputs.rate, inputs.dividend_yield, prices=prices)
        c2.download_button(
            label,
            sample.to_csv(index=False).encode(),
            f"sample_option_{'prices' if prices else 'quotes'}.csv",
            "text/csv",
            on_click="ignore",
            icon=":material/download:",
            key=key,
            width="stretch",
        )
    context = (
        f"Quotes are read with the sidebar spot ({inputs.spot:,.2f}), rate ({inputs.rate:.2%}) and dividend yield "
        f"({inputs.dividend_yield:.2%}); prices are inverted with Black-Scholes-Merton (European exercise)."
    )
    if upload is None:
        st.info("Upload a quotes file, or download a sample to see the expected format. " + context)
        return None
    st.caption(context)
    try:
        raw = pd.read_csv(io.BytesIO(upload.getvalue()))
        with st.spinner("Fitting an SVI smile to every maturity…"):
            surface, messages, error = cache.market_surface(raw, inputs.spot, inputs.rate, inputs.dividend_yield)
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty.")
        return None
    except UnicodeDecodeError:
        st.error("The uploaded file is not a UTF-8 text CSV.")
        return None
    except (vs.QuoteParseError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not build a surface from this file: {exc}")
        return None
    for message in messages:
        if isinstance(message, vs.QuoteWarning):  # a unit was guessed: the surface may be read wrongly
            st.warning(message)
        else:
            st.caption(f"ℹ️ {message}")
    if error:
        st.error(f"Could not build a surface from this file: {error}")
    return surface


def _maturity_labels(maturities: np.ndarray) -> list[str]:
    """Unique labels: days below one year, years above, with as many decimals as needed to tell them apart."""
    for decimals in range(5):
        labels = [f"{t * 365:.{decimals}f}d" if t < 1.0 else f"{t:.{decimals + 2}f}y" for t in maturities]
        if len(set(labels)) == len(labels):
            return labels
    return [f"{t:.8f}y" for t in maturities]


def _contract_readout(surface: vs.VolSurface, inputs: Inputs) -> None:
    moneyness = inputs.strike / inputs.spot
    iv = surface.implied_vol_at(moneyness, inputs.expiry)
    where = f"strike {inputs.strike:,.2f} (K/S = {moneyness:.3f}) and expiry {inputs.days} days"
    if np.isfinite(iv):
        st.info(
            f"At the sidebar {where}, the surface's implied vol is **{iv:.2%}** "
            f"(sidebar volatility {inputs.volatility:.2%}).",
            icon=":material/my_location:",
        )
    else:
        region = "quoted region" if surface.quotes is not None else "grid"
        st.caption(
            f"The sidebar {where} lies outside the surface's {region}, so the surface gives no implied vol for it."
        )


def _workbook(surface: vs.VolSurface, summary: pd.DataFrame, spot: float) -> bytes:
    """The Excel export: both grids, the summary and, for market quotes, every cleaned quote with its fit."""
    frames = surface.to_frames(spot)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        frames["ImpliedVol"].to_excel(writer, sheet_name="ImpliedVol")
        frames["LocalVol"].to_excel(writer, sheet_name="LocalVol")
        summary.to_excel(writer, sheet_name="Summary", index=False)
        if surface.quotes is not None:
            surface.quotes.rename(
                columns={
                    "strike": "Strike",
                    "maturity": "Maturity (years)",
                    "iv": "Quoted implied vol",
                    "moneyness": "Moneyness K/S",
                    "fitted_iv": "SVI implied vol",
                }
            ).to_excel(writer, sheet_name="Quotes", index=False)
    return buffer.getvalue()


def _downloads(surface: vs.VolSurface, summary: pd.DataFrame, spot: float) -> None:
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
    if not any(importlib.util.find_spec(engine) for engine in ("xlsxwriter", "openpyxl")):
        c[2].caption("Install `xlsxwriter` or `openpyxl` for Excel export.")
        return
    # Built only when clicked: a workbook holding a large quote file takes seconds to write, and building it
    # on every rerun slowed every interaction on this tab.
    c[2].download_button(
        "Everything (Excel)",
        functools.partial(_workbook, surface, summary, spot),
        "volatility_surface.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        on_click="ignore",
        icon=":material/download:",
        key="dl_xlsx",
    )


def _smiles(surface: vs.VolSurface) -> None:
    st.subheader("Smiles")
    maturities = surface.maturities
    labels = _maturity_labels(maturities)
    n = len(labels)
    # The key follows the maturity grid, so the default slices apply again whenever the grid changes.
    grid_id = zlib.crc32(np.round(maturities, 8).tobytes())
    picked = st.multiselect(
        "Maturities",
        list(range(n)),
        default=sorted({0, n // 2, n - 1}),
        format_func=labels.__getitem__,
        max_selections=6,
        key=f"smile_pick_{grid_id}",
    )
    if not picked:
        return
    slices = [(labels[i], surface.implied_vol[i], surface.local_vol[i]) for i in picked]
    quotes = None
    if surface.quotes is not None:
        quotes = []
        for i in picked:
            rows = surface.quotes[np.isclose(surface.quotes["maturity"], maturities[i], rtol=0.0, atol=1e-9)]
            quotes.append((rows["moneyness"].to_numpy(), rows["iv"].to_numpy(), rows["fitted_iv"].to_numpy()))
    st.plotly_chart(charts.smile_chart(surface.moneyness, slices, "Moneyness K/S", quotes=quotes), key="smiles")
    if quotes is not None:
        st.caption("Circles are the quotes; hover one to see how far the SVI fit misses it.")


@st.fragment
def render(inputs: Inputs) -> None:
    source = st.segmented_control("Surface source", [SSVI, QUOTES], default=SSVI, required=True, key="surface_source")
    surface = _ssvi_controls(inputs) if source == SSVI else _quote_controls(inputs)
    if surface is None:
        return

    quoted = surface.quoted_points
    valid = int(np.isfinite(surface.local_vol).sum())
    c = st.columns(4)
    c[0].metric(
        "Grid", f"{len(surface.maturities)} × {len(surface.moneyness)}", border=True, help="Maturities × strikes."
    )
    c[1].metric(
        "Calendar-arbitrage points",
        surface.calendar_violations,
        border=True,
        help="Grid points where total implied variance falls as maturity increases at fixed forward moneyness "
        "(∂w/∂T < 0; for market quotes, by more than the quotes' precision).",
    )
    c[2].metric(
        "Butterfly-arbitrage points",
        surface.butterfly_violations,
        border=True,
        help="Durrleman's condition g(y) ≤ 0: negative risk-neutral density.",
    )
    c[3].metric(
        "Local vol defined",
        f"{valid / quoted:.0%}" if quoted else "–",
        border=True,
        help="Share of the grid points that have an implied vol (inside the quoted strikes) where local "
        "volatility is also defined.",
    )
    for note in surface.notes:
        st.warning(note)
    if valid < quoted:
        st.caption(
            "Local volatility is left blank where it is undefined: where the surface has static arbitrage, and for "
            "quotes where ∂w/∂T would need a neighbouring maturity far outside its quoted strikes or where total "
            "variance changes too little between maturities to measure it."
        )
    if quoted < surface.implied_vol.size:
        st.caption("Implied volatility is left blank outside each maturity's quoted strikes (no extrapolation).")
    _contract_readout(surface, inputs)

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

    _smiles(surface)

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
            "Fit RMSE (vol pts)": st.column_config.NumberColumn(
                format="%.3f", help="Root-mean-square gap between the SVI smile and the quotes at this maturity."
            ),
        },
    )
    st.caption(
        "ATM means K = S. Skew is the 90% minus 110% moneyness implied vol; butterfly is their average "
        "minus ATM. Blank cells need strikes outside the quoted range or a local vol that is undefined. Near the "
        "money, local vol skew is roughly twice the implied skew (Gatheral's rule of thumb)."
    )

    st.subheader("Download")
    _downloads(surface, summary, inputs.spot)
