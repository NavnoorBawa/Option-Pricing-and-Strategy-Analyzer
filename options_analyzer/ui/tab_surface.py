"""Surface tab: SSVI or market quotes -> implied and local volatility surfaces."""

from __future__ import annotations

import functools
import importlib.util
import math
import zlib

import numpy as np
import pandas as pd
import streamlit as st

from .. import vol_surface as vs
from . import cache, charts, exports
from . import market as mk
from .components import card_row, current_key
from .format import md_code, signed
from .inputs import Inputs, days_text

SSVI = "Parametric (SSVI)"
QUOTES = "Market quotes (CSV)"
CHAIN = "Synthetic chain"
#: Where each expiry's forward comes from when option prices are inverted (``vol_surface.parse_quotes``).
FORWARD_CHOICES = {"Put-call parity": "parity", "Sidebar r, q": "rates"}
#: How the SVI slices are fitted (``vol_surface.fit_market_surface``).
CALIBRATION_CHOICES = {"Arbitrage-free": "arbitrage_free", "Unconstrained": "unconstrained"}
COLORSCALES = ("Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo")
VOL_LIMITS = (0.5, 500.0)
SURFACE_CHART_WIDTH = 380  # px: two fit side by side in the 804 px main column at 1,280 px with the sidebar open


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
        persist_state="session",
        help="ATM implied vol as maturity → 0. Starts at 1.2 × the sidebar volatility and follows it.",
    )
    long = c[1].number_input(
        "Long-term ATM vol (%)",
        min_value=lo,
        max_value=hi,
        step=0.5,
        key="ssvi_long",
        persist_state="session",
        help="ATM implied vol as maturity → ∞. Starts at the sidebar volatility and follows it.",
    )
    kappa = c[2].number_input(
        "Term-structure speed κ",
        0.05,
        20.0,
        1.5,
        0.1,
        key="ssvi_kappa",
        persist_state="session",
        help="How fast ATM variance moves from the short- to the long-term level (per year).",
    )
    c = st.columns(3)
    rho = c[0].slider(
        "Skew ρ",
        -0.95,
        0.95,
        -0.6,
        0.05,
        key="ssvi_rho",
        persist_state="session",
        help="Negative = downside skew (typical for equities).",
    )
    eta_max = 2.0 / (1.0 + abs(rho))
    eta = c[1].slider(
        "Smile curvature η",
        0.05,
        4.0,
        min(1.0, round(eta_max, 2)),
        0.05,
        key="ssvi_eta",
        persist_state="session",
        help=f"Arbitrage-free for η ≤ 2/(1+|ρ|) = {eta_max:.2f}.",
    )
    gamma = c[2].slider(
        "Curvature decay γ",
        0.05,
        0.5,
        0.5,
        0.05,
        key="ssvi_gamma",
        persist_state="session",
        help="How quickly the smile flattens with maturity (power-law SSVI).",
    )
    c = st.columns(2)
    m_range = c[0].slider("Moneyness range K/S", 0.3, 2.5, (0.7, 1.3), 0.05, key="ssvi_m", persist_state="session")
    t_range = c[1].slider("Maturity range (years)", 0.02, 5.0, (0.08, 2.0), 0.01, key="ssvi_t", persist_state="session")
    if m_range[1] - m_range[0] < 0.1 or t_range[1] - t_range[0] < 0.05:
        st.warning("Widen the moneyness or maturity range to see a surface.")
        return None
    params = vs.SSVIParams(short / 100, long / 100, kappa, rho, eta, gamma)
    return cache.ssvi_surface(params, inputs.rate, inputs.dividend_yield, m_range, t_range)


def _keep_upload() -> None:
    """Keep the uploaded quotes file (or its removal) in the session: an uploader that is not drawn in a run, as
    while another tab is open, forgets its file, and st.file_uploader has no persist_state."""
    st.session_state["quotes_upload"] = st.session_state["quotes_file"]


def _remove_upload() -> None:
    st.session_state["quotes_upload"] = None


def _size(n_bytes: int) -> str:
    """A file size as st.file_uploader shows it (outside Windows): one decimal in the first decimal unit that
    keeps the number at most 500, as ``400.0B``, ``2.0KB`` or ``1.4MB``."""
    size, units = float(n_bytes), ["B", "KB", "MB", "GB"]
    while size > 500 and len(units) > 1:
        size /= 1000
        units.pop(0)
    return f"{size:.1f}{units[0]}"


def _kept_file(container, kept) -> None:
    """The quotes file kept from an earlier visit to the tab, under the uploader (which no longer shows it), with a
    button that removes it as the uploader's own × would."""
    row = container.container(horizontal=True, vertical_alignment="center", gap="small", key="quotes_kept")
    row.caption(
        f":material/description: {md_code(kept.name)} · {_size(kept.size)} · uploaded earlier in this "
        "session and used below; upload another file to replace it."
    )
    row.button("Remove file", key="quotes_remove", icon=":material/close:", type="tertiary", on_click=_remove_upload)


def _quote_controls(inputs: Inputs) -> vs.VolSurface | None:
    c1, c2 = st.columns([2, 1])
    upload = c1.file_uploader(
        "Option quotes (CSV)",
        type=["csv"],
        key="quotes_file",
        on_change=_keep_upload,
        help="Columns: Strike; Days (to expiry) or Maturity (years), or another unit in brackets such as "
        "'Maturity (months)', 'Maturity (weeks)' or 'DTE (hours)'; and either ImpliedVol (decimal or %) or option "
        "prices (Price or Mid, with Bid and Ask used where it is blank). Prices need a Type column saying call or "
        "put; with implied vols a Type column is optional and keeps the out-of-the-money side where a strike has "
        "both. At least two maturities with five or more strikes each.",
    )
    sample_provenance = exports.provenance(
        inputs,
        tag=exports.SYNTHETIC,
        source=exports.SYNTHETIC_SOURCE,
        model="Black-76 prices on a parametric SSVI surface",
        dynamics="none (a snapshot)",
        forwards="S·e^((r−q)T) from the sidebar r and q",
        fit="none (the sample's own SSVI surface)",
        seed=0,  # vol_surface.sample_quotes's noise
    )
    for label, prices, key in (
        ("Sample with implied vols", False, "sample_quotes"),
        ("Sample with prices", True, "sample_prices"),
    ):
        c2.download_button(
            label,
            exports.builder(_sample_csv, inputs.spot, inputs.rate, inputs.dividend_yield, prices, sample_provenance),
            sample_provenance.file_name(f"sample_option_{'prices' if prices else 'quotes'}", "csv"),
            "text/csv",
            on_click="ignore",
            icon=":material/download:",
            key=key,
            width="stretch",
        )
    forward_choice = c2.segmented_control(
        "Forwards",
        list(FORWARD_CHOICES),
        default="Put-call parity",
        required=True,
        key="surface_forward",
        persist_state="session",
        help="Where each expiry's forward and discount factor come from when option prices are inverted: put-call "
        "parity on the expiry's calls and puts (C − P = DF·(F − K), fitted near the money), or the sidebar rate and "
        "dividend yield. Parity needs a call and a put quoted at 6 or more common strikes of an expiry; the others "
        "use the sidebar.",
    )
    forward = FORWARD_CHOICES[forward_choice]
    american = inputs.model.is_american
    if american:
        # The control stays, so that its choice is kept for European quotes; it has no effect on American ones.
        c2.caption(
            "The sidebar's pricing model is American, so put-call parity is not used: it holds only for European "
            "options, and the early-exercise premium would read as a wrong forward. Every expiry's forward and "
            "discount factor come from the sidebar rate and dividend yield."
        )
    calibration_choice = c2.segmented_control(
        "Calibration",
        list(CALIBRATION_CHOICES),
        default="Arbitrage-free",
        required=True,
        key="surface_calibration",
        persist_state="session",
        help="Arbitrage-free: each SVI smile is held free of butterfly arbitrage (a negative density) and of calendar "
        "arbitrage against the expiry before it, where both are quoted; the summary's Arbitrage cost column shows "
        "what that costs in RMSE. Unconstrained: the best smile through each expiry's quotes, arbitrage and all. "
        "Either way each quote counts as one over its bid/ask half-spread in vol.",
    )
    calibration = CALIBRATION_CHOICES[calibration_choice]
    source = (
        "each expiry's forward and discount factor from put-call parity where calls and puts are quoted at common "
        "strikes, else from the sidebar"
        if forward == "parity" and not american
        else "the sidebar"
    )
    exercise = (
        "prices are read as American (the sidebar's pricing model) but still inverted with Black-76 as European "
        "options, which overstates their vols by the early-exercise premium"
        if american
        else "prices are inverted with Black-76 (European exercise)"
    )
    context = (
        f"Quotes are read with the sidebar spot ({inputs.spot:,.2f}) and {source} rate ({inputs.rate:.2%}) and "
        f"dividend yield ({inputs.dividend_yield:.2%}); {exercise}."
    )
    kept = st.session_state.get("quotes_upload")
    if upload is None and kept is not None:
        upload = kept
        _kept_file(c1, kept)
    if upload is None:
        st.info("Upload a quotes file, or download a sample to see the expected format. " + context)
        return None
    st.caption(context)
    try:
        raw = exports.read_csv(upload.getvalue())  # an export of this app starts with its provenance lines
        with st.spinner("Fitting an SVI smile to every maturity…"):
            surface, messages, error = cache.market_surface(
                raw,
                inputs.spot,
                inputs.rate,
                inputs.dividend_yield,
                forward,
                calibration,
                "american" if american else "european",
            )
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
            st.caption(f":material/info: {message}")
    if error:
        st.error(f"Could not build a surface from this file: {error}")
    return surface


def _chain_surface(inputs: Inputs) -> vs.VolSurface:
    """The index preset's surface: the one fitted to the synthetic chain at the valuation time, which the sidebar's
    contract reads its vol, forward and carry from (:mod:`.market`)."""
    context = inputs.context
    assert context is not None  # offered under the index preset only
    quotes = context.quotes
    expiries = quotes["Maturity (years)"].nunique()
    st.caption(
        f"{mk.SYNTHETIC_NOTE} The {mk.INDEX_ROOT} chain at {context.as_of:%d %b %Y %H:%M} ET: {len(quotes):,} bid/ask "
        f"quotes over {expiries} expiries, read like an upload: each expiry's forward and discount factor from "
        "put-call parity, prices inverted with Black-76, and an arbitrage-free SVI smile fitted to each expiry with "
        "spread weights. The sidebar's contract takes its vol, forward and carry from this surface."
    )
    with st.expander(f"How the chain was read ({len(context.messages)} notes)", icon=":material/info:"):
        for message in context.messages:
            st.caption(message)
    provenance = _provenance(inputs, CHAIN)
    st.download_button(
        "Download the chain (CSV)",
        exports.builder(exports.csv, quotes, provenance),
        provenance.file_name(f"{mk.INDEX_ROOT.lower()}_chain", "csv"),
        "text/csv",
        on_click="ignore",
        icon=":material/download:",
        key="chain_csv",
        help="The quotes this surface is fitted to, in the layout the Market quotes source reads.",
    )
    return context.surface


FORWARD_SOURCE_LABELS = {"parity": "parity", "parity_forward": "parity (DF: rate)", "rates": "sidebar"}
#: The implied rates a parity fit may give (vol_surface.forwards.RATE_LIMITS), as the help text shows them.
RATE_RANGE = " to ".join(f"{limit:.0%}".replace("-", "\u2212") for limit in vs.forwards.RATE_LIMITS)


def _forwards(surface: vs.VolSurface, inputs: Inputs) -> None:
    """The forward and discount factor of every fitted expiry, and a warning where the quotes resolve a forward
    that the sidebar rate and dividend yield miss (``vol_surface.carry_gaps``). Not for the index preset's chain:
    there every tab prices each expiry at its own parity forward, and the chain was read with the Treasury curve,
    so the gap is the box spread, not a sidebar error."""
    table = surface.forwards
    if table is None or table.empty:
        return
    chain = surface is getattr(inputs.context, "surface", None)
    gaps = vs.carry_gaps(table)
    if gaps.any() and not chain:
        worst = float(np.max(np.abs(table["gap_bp"].to_numpy(float)[gaps])))
        st.warning(
            f"At {int(gaps.sum())} of {len(table)} expiries the forward that put-call parity reads from the quotes "
            f"is more than {vs.forwards.MIN_CARRY_GAP_BP:g} bp, and more than half the at-the-money spread, from the "
            f"sidebar's (rate {inputs.rate:.2%}, dividend yield {inputs.dividend_yield:.2%}); up to {worst:,.1f} bp. "
            "The surface uses the parity forwards; the other tabs still price with the sidebar rate and yield.",
            icon=":material/compare_arrows:",
        )
    shown = pd.DataFrame(
        {
            "Days": table["maturity"].to_numpy(float) * 365,
            "Forward": table["forward"],
            "Discount factor": table["discount_factor"],
            "Implied rate (%)": table["rate"] * 100,
            "Implied yield (%)": table["dividend_yield"] * 100,
            "Gap to sidebar (bp)": table["gap_bp"],
            "Pairs": table["pairs"],
            "Residual": table["residual"],
            "Source": table["source"].map(FORWARD_SOURCE_LABELS),
        }
    )
    if chain:
        shown = shown.drop(columns="Gap to sidebar (bp)")
    with st.expander("Forwards and discounting", icon=":material/compare_arrows:"):
        st.dataframe(
            # The gap is signed: shown with the minus sign U+2212 and no sign at zero (a format string gives "-" and
            # "+0.0"); a Styler's text is shown only in a column whose config has no format, and sorts by the number.
            shown.style.format(
                functools.partial(signed, decimals=1), subset=[c for c in ["Gap to sidebar (bp)"] if c in shown]
            ),
            hide_index=True,
            column_config={
                "Days": st.column_config.NumberColumn(format="%.2f", help="Calendar days to expiry."),
                "Forward": st.column_config.NumberColumn(format="%.2f"),
                "Discount factor": st.column_config.NumberColumn(format="%.6f"),
                "Implied rate (%)": st.column_config.NumberColumn(
                    format="%.2f", help="−ln(DF) / T: the rate the options are discounted at (the box rate)."
                ),
                "Implied yield (%)": st.column_config.NumberColumn(
                    format="%.2f", help="Implied rate − ln(F / S) / T: the dividend yield that gives this forward."
                ),
                "Gap to sidebar (bp)": st.column_config.NumberColumn(
                    help="ln(F / sidebar forward), in basis points of the forward."
                ),
                "Pairs": st.column_config.NumberColumn(
                    format="%d", help="Call/put pairs near the money used for the fit, after outliers are removed."
                ),
                "Residual": st.column_config.NumberColumn(
                    format="%.4f", help="Root-mean-square gap of C − P from the fitted line DF·(F − K), in price units."
                ),
                "Source": st.column_config.TextColumn(
                    help="parity: forward and discount factor fitted to the pairs; parity (DF: rate): forward from "
                    "parity, discount factor from the sidebar rate plus the spread over it measured at the other "
                    "expiries, because this expiry's own fit could not measure it (its strikes span too little, or "
                    f"it implied a rate outside {RATE_RANGE}); sidebar: too few pairs, the Sidebar r, q choice, or "
                    "American quotes."
                ),
            },
        )
        st.caption(
            "Black-76 with this forward F and discount factor DF is Black-Scholes-Merton at the implied rate and "
            "yield, so each quote is inverted at its own expiry's carry."
        )


def _local_vol_share(valid: int, quoted: int) -> str:
    """``valid`` of ``quoted`` points as a percentage that reads 100% or 0% only when it is exactly that.

    A partial share is rounded down to one decimal and kept within 0.1-99.9%: 649 of 652 points used to show
    as "100%" next to a note that local vol is blank at three of them.
    """
    if not quoted:
        return "–"
    if 0 < valid < quoted:
        return f"{min(max(math.floor(valid / quoted * 1000) / 10, 0.1), 99.9):.1f}%"
    return f"{valid / quoted:.0%}"


def _maturity_labels(maturities: np.ndarray) -> list[str]:
    """Unique labels: days below one year, years above, with as many decimals as needed to tell them apart."""
    for decimals in range(5):
        labels = [f"{t * 365:.{decimals}f}d" if t < 1.0 else f"{t:.{decimals + 2}f}y" for t in maturities]
        if len(set(labels)) == len(labels):
            return labels
    return [f"{t:.8f}y" for t in maturities]


def _contract_readout(surface: vs.VolSurface, inputs: Inputs) -> None:
    moneyness = inputs.strike / inputs.spot
    iv = surface.implied_vol_at(moneyness, inputs.expiry, extrapolate=True)
    where = f"strike {inputs.strike:,.2f} (K/S = {moneyness:.3f}) and expiry {days_text(inputs.days)}"
    region = "quoted strikes" if surface.quotes is not None else "grid"
    if np.isfinite(iv):
        extrapolated = surface.is_extrapolated(moneyness, inputs.expiry)
        how = f", extrapolated beyond the {region}" if extrapolated else ""
        st.info(
            f"At the sidebar {where}, the surface's implied vol is **{iv:.2%}**{how} "
            f"(sidebar volatility {inputs.volatility:.2%}).",
            icon=":material/my_location:",
        )
        return
    Ts = surface.maturities
    if not Ts[0] - vs.MATURITY_TOLERANCE <= inputs.expiry <= Ts[-1] + vs.MATURITY_TOLERANCE:
        st.caption(
            f"The sidebar {where} lies outside the surface's maturities ({_maturity_labels(Ts[[0]])[0]} to "
            f"{_maturity_labels(Ts[[-1]])[0]}), so the surface gives no implied vol for it."
        )
        return
    reach = surface.reach_at(inputs.expiry)
    st.caption(
        f"The sidebar {where} lies outside the {region}, and beyond them the surface extrapolates only out to "
        f"|ln K/F| = {reach:.2f} from that expiry's forward F (four at-the-money standard deviations, at most "
        f"{vs.svi.WING_CAP:g}), so it gives no implied vol for it."
    )


def _sample_csv(spot: float, rate: float, dividend_yield: float, prices: bool, provenance: exports.Provenance) -> bytes:
    """A sample quotes file with its provenance lines, built when its button is clicked (not on every rerun)."""
    return exports.csv(vs.sample_quotes(spot, rate, dividend_yield, prices=prices), provenance)


def _provenance(inputs: Inputs, source: str) -> exports.Provenance:
    """Where the tab's exports come from (UI spec 2.13): the synthetic chain, the user's quotes file with the
    forwards and calibration chosen for it, or the parametric SSVI surface."""
    snapshot = "none (a snapshot)"
    if source == CHAIN:
        return exports.provenance(inputs, model="Black-76 inversion of the chain's quotes", dynamics=snapshot)
    if source == SSVI:
        return exports.provenance(
            inputs,
            tag=exports.PARAMETRIC,
            source=f"{exports.PARAMETRIC} — a parametric SSVI surface, not market data",
            model="SSVI (Gatheral-Jacquier), parameters typed",
            dynamics=snapshot,
            forwards="S·e^((r−q)T) from the sidebar r and q",
            fit="none (parametric)",
        )
    upload = st.session_state.get("quotes_file") or st.session_state.get("quotes_upload")
    name = getattr(upload, "name", "quotes.csv")
    forward = st.session_state.get("surface_forward", "Put-call parity")
    calibration = st.session_state.get("surface_calibration", "Arbitrage-free")
    return exports.provenance(
        inputs,
        tag=exports.UPLOAD,
        source=f"{exports.UPLOAD} · {' '.join(str(name).split())} — the user's file",
        model="Black-76 inversion of the file's quotes",
        dynamics=snapshot,
        forwards="parity" if FORWARD_CHOICES.get(forward) == "parity" else "the sidebar r and q",
        fit=f"SVI {calibration.lower()}, spread-weighted",
    )


def _workbook(surface: vs.VolSurface, summary: pd.DataFrame, spot: float, provenance: exports.Provenance) -> bytes:
    """The Excel export: its provenance, both grids, the summary and, for market quotes, every cleaned quote with
    its fit."""
    frames = surface.to_frames(spot)
    sheets = {
        "ImpliedVol": frames["ImpliedVol"].reset_index(),
        "LocalVol": frames["LocalVol"].reset_index(),
        "Summary": summary,
    }
    if surface.quotes is not None:
        sheets["Quotes"] = surface.quotes.rename(
            columns={
                "strike": "Strike",
                "maturity": "Maturity (years)",
                "iv": "Quoted implied vol",
                "moneyness": "Moneyness K/S",
                "fitted_iv": "SVI implied vol",
                "vol_half_spread": "Half bid/ask spread in vol",
            }
        )
    return exports.xlsx(sheets, provenance)


def _downloads(surface: vs.VolSurface, summary: pd.DataFrame, spot: float, provenance: exports.Provenance) -> None:
    frames = surface.to_frames(spot)
    c = st.columns(3)
    c[0].download_button(
        "Implied vol grid (CSV)",
        exports.builder(exports.csv, frames["ImpliedVol"], provenance, index=True),
        provenance.file_name("implied_vol_surface", "csv"),
        "text/csv",
        on_click="ignore",
        icon=":material/download:",
        key="dl_iv",
    )
    c[1].download_button(
        "Local vol grid (CSV)",
        exports.builder(exports.csv, frames["LocalVol"], provenance, index=True),
        provenance.file_name("local_vol_surface", "csv"),
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
        exports.builder(_workbook, surface, summary, spot, provenance),
        provenance.file_name("volatility_surface", "xlsx"),
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
        key=current_key("smile_pick_", f"smile_pick_{grid_id}"),
        persist_state="session",
    )
    if not picked:
        return
    slices = [(labels[i], surface.implied_vol[i], surface.local_vol[i]) for i in picked]
    # Each smile read on beyond its quotes (SVI wings within Lee's bound, out to its reach), where the grid has room.
    extended = surface.implied_at_moneyness(surface.moneyness, extrapolate=True)
    wings = [_wing(surface.implied_vol[i], extended[i]) for i in picked]
    quotes = None
    if surface.quotes is not None:
        quotes = []
        for i in picked:
            rows = surface.quotes[np.isclose(surface.quotes["maturity"], maturities[i], rtol=0.0, atol=1e-9)]
            quotes.append((rows["moneyness"].to_numpy(), rows["iv"].to_numpy(), rows["fitted_iv"].to_numpy()))
    charts.show(
        charts.smile_chart(surface.moneyness, slices, "Moneyness K/S", quotes=quotes, wings=wings),
        key="smiles",
        alt="Implied and local volatility against moneyness for the selected maturities"
        + (", with the market quotes" if quotes is not None else ""),
    )
    region = "quoted strikes" if quotes is not None else "grid"
    extrapolation = (
        f" Dotted lines carry an implied-vol smile on beyond its {region}: extrapolated, out to four at-the-money "
        f"standard deviations from the forward (|ln K/F| at most {vs.svi.WING_CAP:g}), as far as this chart reaches."
        if any(np.isfinite(wing).any() for wing in wings)
        else ""
    )
    if quotes is not None:
        st.caption("Circles are the quotes; hover one to see how far the SVI fit misses it." + extrapolation)
    elif extrapolation:
        st.caption(extrapolation.strip())


def _wing(inside: np.ndarray, extended: np.ndarray) -> np.ndarray:
    """The extrapolated part of one smile on the grid: ``extended`` where ``inside`` (the smile within its quotes) is
    blank, and the last point within the quotes next to each wing, so that the wing joins the smile; NaN elsewhere."""
    beyond = np.isfinite(extended) & ~np.isfinite(inside)
    next_to = np.zeros_like(beyond)
    next_to[1:] |= beyond[:-1]
    next_to[:-1] |= beyond[1:]
    return np.where(beyond | (next_to & np.isfinite(inside)), extended, np.nan)


@st.fragment
def render(inputs: Inputs) -> None:
    sources = [CHAIN, QUOTES, SSVI] if inputs.context is not None else [SSVI, QUOTES]
    source = st.segmented_control(
        "Surface source", sources, default=sources[0], required=True, key="surface_source", persist_state="session"
    )
    if source == CHAIN:
        surface = _chain_surface(inputs)
    else:
        surface = _ssvi_controls(inputs) if source == SSVI else _quote_controls(inputs)
    if surface is None:
        return

    quoted = surface.quoted_points
    valid = int(np.isfinite(surface.local_vol).sum())
    from_quotes = surface.quotes is not None
    fitted = " of the fitted surface" if from_quotes else ""
    in_quotes = " Arbitrage in the quotes themselves is listed in the notes below." if from_quotes else ""
    # Short labels: "Calendar-arbitrage points" was cut to "Calendar-arbitrage p…" in a row of four at 1,280 px.
    card_row(
        "surface_cards",
        ("surface_cards_a", "surface_cards_b"),
        [
            ("Grid", f"{len(surface.maturities)} × {len(surface.moneyness)}", {"help": "Maturities × strikes."}),
            (
                "Calendar arbitrage",
                surface.calendar_violations,
                {
                    "help": f"Number of grid points{fitted} where total implied variance falls as maturity increases "
                    "at fixed forward moneyness (∂w/∂T < 0; for market quotes, by more than the quotes' "
                    f"precision).{in_quotes}"
                },
            ),
            (
                "Butterfly arbitrage",
                surface.butterfly_violations,
                {
                    "help": f"Number of grid points{fitted} that fail Durrleman's condition: g(y) < 0, a negative "
                    f"risk-neutral density.{in_quotes}"
                },
            ),
            (
                "Local vol defined",
                _local_vol_share(valid, quoted),
                {
                    "help": "Share of the grid points that have an implied vol (inside the quoted strikes) where "
                    f"local volatility is also defined: {valid:,} of {quoted:,}."
                },
            ),
        ],
    )
    _forwards(surface, inputs)
    for note in surface.notes:
        st.warning(note)
    if valid < quoted:
        st.caption(
            "Local volatility is left blank where it is undefined: where the surface has static arbitrage, and for "
            "quotes where ∂w/∂T would need a neighbouring maturity far outside its quoted strikes or where total "
            "variance changes too little between maturities to measure it."
        )
    if quoted < surface.implied_vol.size:
        st.caption(
            "The grid and the 3-D surfaces leave implied volatility blank outside each maturity's quoted strikes; the "
            "contract readout and the Smiles chart read on beyond them, marked as extrapolated."
        )
    _contract_readout(surface, inputs)

    colorscale = st.selectbox("Colour scale", COLORSCALES, key="surface_colors", persist_state="session")
    # Side by side when both fit (1,280 px with the sidebar open), otherwise one per row; a phone shrinks them to
    # its column. Two half-width columns were 212-274 px wide at 900-1,024 px with the sidebar open, which cut off
    # the 3-D scene's axes (Plotly sizes the scene by the chart's height, so a wider chart only adds margin).
    # Centred, so a wide screen does not leave the right half of the row empty.
    row = st.container(horizontal=True, horizontal_alignment="center", key="surfaces")
    charts.show(
        charts.surface_3d(surface.moneyness, surface.maturities, surface.implied_vol, "Implied volatility", colorscale),
        key="surf_iv",
        alt="3-D implied volatility surface over moneyness and maturity",
        container=row,
        width=SURFACE_CHART_WIDTH,
    )
    charts.show(
        charts.surface_3d(
            surface.moneyness, surface.maturities, surface.local_vol, "Local volatility (Dupire / Gatheral)", colorscale
        ),
        key="surf_lv",
        alt="3-D local volatility surface over moneyness and maturity",
        container=row,
        width=SURFACE_CHART_WIDTH,
    )

    _smiles(surface)

    st.subheader("Term structure and skew")
    summary = vs.surface_summary(surface)
    series = {"ATM implied vol": summary["ATM implied vol"], "ATM local vol": summary["ATM local vol"]}
    fig = charts.lines_chart(summary["Maturity (years)"], series, "Maturity (years)", "Volatility", hline=None)
    fig.update_yaxes(tickformat=".1%")
    charts.show(fig, key="term_structure", alt="At-the-money implied and local volatility against maturity")
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
            "Inside bid/ask (%)": st.column_config.NumberColumn(
                format="%.0f",
                help="Share of the quotes whose fitted vol lies within their bid/ask spread in vol (or within one "
                "step of the file's price grid). Blank for quoted implied vols and for prices with neither.",
            ),
            "Median |error| / half-spread": st.column_config.NumberColumn(
                format="%.2f",
                help="Median gap between fit and quote, in units of the quote's half-spread in vol. Blank where the "
                "quotes have no spread.",
            ),
            "Arbitrage cost (vol pts)": st.column_config.NumberColumn(
                format="%.3f",
                help="How much holding this maturity's smile free of static arbitrage raised the fit's error: the "
                "RMSE with each quote weighted as in the fit (by one over its half-spread in vol, relative to the "
                "median quote). 0 where the smile had no arbitrage; blank where no arbitrage-free smile was found "
                "(a note above says so).",
            ),
        },
    )
    st.caption(
        "ATM means K = S. Skew is the 90% minus 110% moneyness implied vol; butterfly is their average "
        "minus ATM. Blank cells need strikes outside the quoted range or a local vol that is undefined. Near the "
        "money, local vol skew is roughly twice the implied skew (Gatheral's rule of thumb)."
    )

    st.subheader("Download")
    _downloads(surface, summary, inputs.spot, _provenance(inputs, source))
