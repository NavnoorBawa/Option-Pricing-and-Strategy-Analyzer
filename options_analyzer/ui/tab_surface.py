"""Surface tab: the chain read quote by quote, its arbitrage-checked SVI surface, and the desk's view of both (UI spec
3.4; roadmap items SF-1, SF-6 and SF-11).

Top to bottom: the data panel (the source, the counts that reconcile, quoted = used + the exclusions by reason, and the
Filters popover with the fit settings), six tiles (ATMF 30D with the VIX-style model-free vol, the 25-delta risk
reversal and butterfly at 30 days, the 30-to-365-day term slope, the fit against the spreads, the arbitrage counts),
the primary view of the focus expiry (Smile, Quotes, Term, Grid or an opt-in 3D), the expiry table whose row click
sets the focus expiry, the Arbitrage and forwards expander, the Pricing contract's readout, the exports and a footer.

Sources: under the synthetic index, the chain's published surface (:mod:`.market`, :mod:`.surface_state`), which every
tab prices from; under the textbook market, a parametric SSVI surface. Either way an uploaded file can be read here:
a chain in one of the three layouts of ``market_data.read_chain``, screened quote by quote, or a table of strikes,
times and vols or prices (``vol_surface.parse_quotes``). An upload is fitted on this tab only; the other tabs keep
the synthetic chain (or the typed inputs).

The focus expiry is the sidebar's (key ``expiry``) under the synthetic chain, so that the expiry table's row click
(:func:`on_expiry_select`) and the ◂ ▸ buttons move the Pricing contract too; for an upload or SSVI it is this tab's
own (``sf_focus``). Numbers are formatted by :mod:`.format` and the column registry (:func:`.components.table`).
"""

from __future__ import annotations

import functools
import importlib.util
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import ndtr

from .. import clock
from .. import vol_surface as vs
from ..market_data import read_chain
from ..vol_surface import diagnostics as dg
from . import cache, charts, charts_surface, exports, surface_state, theme, whatif
from . import market as mk
from .components import card_row, table
from .components import col as column
from .format import (
    MISSING,
    delta_label,
    level,
    md_code,
    moneyness,
    number,
    pct,
    strike,
    time_left,
    timestamp,
    vol,
    vol_change,
)
from .format import date as short_date
from .inputs import Inputs

CHAIN, QUOTES, SSVI = "Synthetic chain", "Upload", "Parametric (SSVI)"
#: The primary views (``sf_view``).
VIEWS = ("Smile", "Quotes", "Term", "Grid", "3D")
#: The smile's x axes (``sf_x``).
X_AXES = tuple(charts_surface.X_AXES)
#: Session key of the focus expiry of an upload or the SSVI surface (a row of the expiry table); the synthetic chain's
#: is the sidebar's ``expiry``.
FOCUS_KEY = "sf_focus"
#: The six tiles in two groups of three: 3 + 3 at 1,440 px with the sidebar open (six across would need about
#: 1,060 px of the 1,037 there; three pairs went 4 + 2), then one group per line.
TILE_GROUPS = ("surface_tiles_a", "surface_tiles_b")
#: The tiles' constant maturities, in calendar days.
TILE_DAYS, TERM_DAYS = 30, 365
VOL_LIMITS = (0.5, 500.0)
#: The chain layouts an upload is read in (``market_data.read_chain``), one line each for the empty state.
LAYOUTS = (
    "the app's long layout: symbol or root, expiry, type, strike, bid, ask (settlement and vendor_iv optional);",
    "a yfinance export of option_chain() calls and puts, stacked, with contractSymbol, bid, ask;",
    "Cboe's delayed-quotes download, calls left and puts right of the strike;",
)
#: Or the older table of quotes: one row per strike and expiry with an implied vol or prices.
QUOTES_LAYOUT = (
    "or a table with Strike, Days (or Maturity in years) and ImpliedVol, or prices (Price, or Bid and Ask) with a "
    "call/put Type."
)
#: The footer: what the fit is, in one line.
FOOTER = (
    "SVI per expiry (Gatheral 2004), Durrleman g ≥ 0, calendar w(k, T) non-decreasing; wing cap |k| ≤ "
    f"{number(vs.svi.WING_CAP, 1)} (SF-8). Method: Methodology › Surface."
)


@dataclass(frozen=True, eq=False)  # holds tables: compared by identity
class Shown:
    """What the tab shows: the source, its surface and, for a chain, the fit quote by quote; the expiry rows; and
    the reference carry the carry gap is measured against."""

    source: str
    surface: vs.VolSurface
    fit: dg.ChainFit | None
    rows: pd.DataFrame  # one per fitted expiry: dg.slice_table's columns plus name, minutes, target
    rate: float
    dividend_yield: float
    as_of: datetime | None
    panel: str  # the data panel's line
    notes: tuple[str, ...]
    provenance: exports.Provenance
    published: bool  # whether this is the surface the other tabs price from
    file_name: str | None = None
    vendor: bool = False  # whether the quotes carry vendor implied vols


# --------------------------------------------------------------------------------------------------------------------
# Callbacks (plain functions, unit-tested: AppTest cannot fire on_select)
# --------------------------------------------------------------------------------------------------------------------


def on_expiry_select(key: str, targets: list, focus_key: str) -> None:
    """The expiry table's row click: set the focus expiry (the sidebar's ``expiry`` under the synthetic chain, else
    ``sf_focus``) to the clicked row's target. A click that clears the selection leaves the focus as it is."""
    selection = st.session_state[key]
    rows = selection["selection"]["rows"] if isinstance(selection, dict) else selection.selection.rows
    if rows:
        st.session_state[focus_key] = targets[int(rows[0])]


def on_expiry_step(targets: list, focus_key: str, current: int, step: int) -> None:
    """The ◂ ▸ buttons: the previous or next expiry (the keyboard route to the row click), clamped at the ends."""
    st.session_state[focus_key] = targets[min(max(current + step, 0), len(targets) - 1)]


def _use_synthetic() -> None:
    st.session_state["surface_source"] = CHAIN


def _keep_upload() -> None:
    """Keep the uploaded file (or its removal) in the session: an uploader that is not drawn in a run, as while
    another tab is open, forgets its file, and st.file_uploader has no persist_state."""
    st.session_state["quotes_upload"] = st.session_state["quotes_file"]


def _remove_upload() -> None:
    st.session_state["quotes_upload"] = None


def _reset_fit() -> None:
    """Every fit setting back to its default (the FIT chip goes)."""
    st.session_state.update(surface_state.WIDGET_DEFAULTS)


def _size(n_bytes: int) -> str:
    """A file size as st.file_uploader shows it (outside Windows): one decimal in the first decimal unit that
    keeps the number at most 500, as ``400.0B``, ``2.0KB`` or ``1.4MB``."""
    size, units = float(n_bytes), ["B", "KB", "MB", "GB"]
    while size > 500 and len(units) > 1:
        size /= 1000
        units.pop(0)
    return number(size, 1) + units[0]


# --------------------------------------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------------------------------------


def _counts_text(counts: dict[str, int], *, dropped_reading: int = 0) -> str:
    """``5,012 quoted → 2,357 used (149 zero bid · 2,506 ITM side · 0 wide)``: every reason with quotes, and wide.

    For an uploaded chain, ``dropped_reading`` counts the contracts the file reader dropped before the screen
    (``read_chain``: unreadable, expired, crossed or zero-bid rows), so the file's rows still reconcile."""
    reasons = [f"{number(counts[r], 0)} {r}" for r in dg.REASONS if counts[r] or r == "wide"]
    if dropped_reading:
        reasons.insert(0, f"{number(dropped_reading, 0)} dropped reading")
    quoted = counts["quoted"] + dropped_reading
    return f"{number(quoted, 0)} quoted → {number(counts['used'], 0)} used ({' · '.join(reasons)})"


def _method_text(surface: vs.VolSurface, settings: dg.FitSettings) -> str:
    calibration = "SVI arb‑free" if surface.calibration != "unconstrained" else "SVI unconstrained"
    forwards = surface.forwards
    parity = forwards is not None and len(forwards) and (forwards["source"] != "rates").any()
    return f"forward: {'parity' if parity else 'rate and yield'} · {calibration} · mid"


def _chain_rows(fit: dg.ChainFit, rate: float, dividend_yield: float, *, iso: bool) -> pd.DataFrame:
    """The expiry rows of a chain fit, with their names (``20-Nov-26 AM``), minutes to settlement and focus targets
    (the ISO date for the sidebar, else the row number)."""
    rows = dg.expiry_table(fit, rate=rate, dividend_yield=dividend_yield)
    rows["name"] = [f"{short_date(d)} {s}" for d, s in zip(rows["expiry"], rows["settlement"], strict=True)]
    rows["minutes"] = rows["maturity"] * clock.MINUTES_PER_YEAR
    rows["target"] = [d.isoformat() for d in rows["expiry"]] if iso else list(range(len(rows)))
    return rows


def _surface_rows(surface: vs.VolSurface, rate: float, dividend_yield: float) -> pd.DataFrame:
    """The expiry rows of a surface without dated expiries (a table of quotes, or SSVI): named by days to expiry."""
    rows = dg.slice_table(surface, rate=rate, dividend_yield=dividend_yield)
    rows["minutes"] = rows["maturity"] * clock.MINUTES_PER_YEAR
    rows["name"] = day_labels(rows["maturity"].to_numpy(float) * 365.0)
    rows["expiry"], rows["settlement"] = None, None
    rows["target"] = list(range(len(rows)))
    return rows


def day_labels(days: np.ndarray) -> list[str]:
    """Unique labels of times to expiry in calendar days, with as many decimals as tell them apart (never years).

    >>> day_labels(np.array([7.0, 30.0, 730.0])), day_labels(np.array([1.0, 1.4, 1.42]))
    (['7d', '30d', '730d'], ['1.00d', '1.40d', '1.42d'])
    """
    for decimals in range(5):
        labels = [number(d, decimals) + "d" for d in days]
        if len(set(labels)) == len(labels):
            return labels
    return [number(d, 6) + "d" for d in days]


def _chain_source(inputs: Inputs) -> Shown:
    """The synthetic chain at the valuation time, as fitted and published (:func:`.market.index_market`)."""
    context = inputs.context
    assert context is not None and context.fit is not None  # offered under the index preset only
    fit = context.fit
    rate, dividend_yield = mk.INDEX_CARRY
    panel = (
        f"{mk.INDEX_ROOT} chain · {mk.SYNTHETIC_NOTE.rstrip('.')} · {timestamp(context.as_of)} · "
        f"{_counts_text(fit.counts())} · {_method_text(fit.surface, fit.settings)}"
    )
    return Shown(
        CHAIN,
        fit.surface,
        fit,
        _chain_rows(fit, rate, dividend_yield, iso=True),
        rate,
        dividend_yield,
        context.as_of,
        panel,
        fit.messages,
        _provenance(inputs, CHAIN),
        True,
    )


def _ssvi_controls(inputs: Inputs, container) -> vs.VolSurface | None:
    """The SSVI parameters, in a popover of the data panel; their ATM vols follow the sidebar's vol."""
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
    with container.popover("SSVI parameters", icon=":material/tune:", key="ssvi_params"):
        c = st.columns(2)
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
        kappa = st.number_input(
            "Term-structure speed κ (per year)",
            0.05,
            20.0,
            1.5,
            0.1,
            key="ssvi_kappa",
            persist_state="session",
            help="How fast ATM variance moves from the short- to the long-term level.",
        )
        rho = st.slider(
            "Skew ρ", -0.95, 0.95, -0.6, 0.05, key="ssvi_rho", persist_state="session", help="Negative: put skew."
        )
        eta_max = 2.0 / (1.0 + abs(rho))
        eta = st.slider(
            "Smile curvature η",
            0.05,
            4.0,
            min(1.0, round(eta_max, 2)),
            0.05,
            key="ssvi_eta",
            persist_state="session",
            help=f"Arbitrage-free for η ≤ 2/(1+|ρ|) = {number(eta_max, 2)}.",
        )
        gamma = st.slider(
            "Curvature decay γ",
            0.05,
            0.5,
            0.5,
            0.05,
            key="ssvi_gamma",
            persist_state="session",
            help="How quickly the smile flattens with maturity (power-law SSVI).",
        )
        m_range = st.slider("Strike range K/S", 0.3, 2.5, (0.7, 1.3), 0.05, key="ssvi_m", persist_state="session")
    if m_range[1] - m_range[0] < 0.1:
        st.warning("Widen the strike range to see a surface.")
        return None
    params = vs.SSVIParams(short / 100, long / 100, kappa, rho, eta, gamma)
    maturities = np.array(dg.CONSTANT_MATURITIES, dtype=float) / 365.0
    return vs.ssvi_surface(
        params, inputs.rate, inputs.dividend_yield, np.linspace(m_range[0], m_range[1], 41), maturities
    )


def _ssvi_source(inputs: Inputs, container) -> Shown | None:
    surface = _ssvi_controls(inputs, container)
    if surface is None:
        return None
    rows = _surface_rows(surface, inputs.rate, inputs.dividend_yield)
    panel = (
        "Parametric SSVI surface (Gatheral and Jacquier), parameters typed, not market data · forward "
        f"S·e^((r−q)T) at the sidebar's r {pct(inputs.rate)} and q {pct(inputs.dividend_yield)} · "
        f"{len(surface.maturities)} constant maturities"
    )
    return Shown(
        SSVI, surface, None, rows, inputs.rate, inputs.dividend_yield, None, panel, (), _provenance(inputs, SSVI), False
    )


@st.cache_resource(show_spinner=False, max_entries=4)
def _fit_upload(
    data: bytes,
    as_of: datetime,
    spot: float,
    rate: float,
    dividend_yield: float,
    settings: dg.FitSettings,
    exercise: str,
) -> tuple[dg.ChainFit | None, float, tuple[str, ...], str | None]:
    """An uploaded chain read (``read_chain``) and fitted (:func:`.diagnostics.fit_chain`), with the spot the fit
    used (parity's where the quotes give it, else ``spot``), the reader's messages and the error, if any.

    Raises
    ------
    ValueError
        If the file is in none of the chain layouts (the caller then reads it as a table of quotes).
    """
    chain, messages = read_chain(data, as_of=as_of)
    implied = dg.implied_spot(chain, as_of=as_of) if exercise == "european" else math.nan
    used_spot = implied if math.isfinite(implied) else spot
    try:
        fit = dg.fit_chain(
            chain,
            as_of=as_of,
            spot=used_spot,
            rate=rate,
            dividend_yield=dividend_yield,
            settings=settings,
            exercise=exercise,
        )
    except ValueError as exc:  # QuoteParseError: too few expiries left, or a numerical failure of the fit
        return None, used_spot, tuple(messages), str(exc)
    return fit, used_spot, tuple(messages), None


def _upload_source(inputs: Inputs, upload) -> Shown | None:
    """An uploaded file: a chain in one of the three layouts, else a table of quotes."""
    settings = surface_state.settings()
    index = inputs.context is not None
    as_of = inputs.context.as_of if index else mk.last_close()
    rate, dividend_yield = mk.INDEX_CARRY if index else (inputs.rate, inputs.dividend_yield)
    american = inputs.model.is_american
    name = " ".join(str(upload.name).split())
    data = upload.getvalue()
    try:
        with st.spinner("Reading the chain and fitting a smile to every expiry…"):
            fit, spot, messages, error = _fit_upload(
                data, as_of, inputs.spot, rate, dividend_yield, settings, "american" if american else "european"
            )
    except ValueError:  # not a chain layout: a table of strikes, times and vols or prices
        return _quotes_table_source(inputs, upload, settings, as_of, rate, dividend_yield)
    except UnicodeDecodeError:
        st.error("The uploaded file is not a text CSV.")
        return None
    if fit is None:
        st.error(f"Could not build a surface from {md_code(name)}: {error}")
        for message in messages:
            st.caption(message)
        return None
    spot_text = f"S {level(spot)} from parity (front F·DF)" if spot != inputs.spot else f"S {level(spot)} (sidebar)"
    panel = (
        f"Upload · {md_code(name)} · the user's file, not published: the other tabs price from "
        f"{'the synthetic chain' if index else 'the typed inputs'} · valued at {timestamp(as_of)}, quote time unknown "
        f"· {spot_text} · {_counts_text(fit.counts(), dropped_reading=_dropped_reading(messages, fit))} · "
        f"{_method_text(fit.surface, settings)}"
    )
    vendor = bool(fit.quotes["vendor_iv"].notna().any()) if "vendor_iv" in fit.quotes else False
    _file_notes(fit.surface)
    return Shown(
        QUOTES,
        fit.surface,
        fit,
        _chain_rows(fit, rate, dividend_yield, iso=False),
        rate,
        dividend_yield,
        as_of,
        panel,
        (*messages, *fit.messages),
        _provenance(inputs, QUOTES, name=name, as_of=as_of),
        False,
        name,
        vendor,
    )


def _dropped_reading(messages, fit: dg.ChainFit) -> int:
    """Contracts of the file that ``read_chain`` dropped: its first message counts the contracts read."""
    found = re.match(r"Read ([\d,]+) contract", messages[0]) if messages else None
    return max(int(found.group(1).replace(",", "")) - len(fit.quotes), 0) if found else 0


def _quotes_table_source(
    inputs: Inputs, upload, settings: dg.FitSettings, as_of: datetime, rate: float, dividend_yield: float
) -> Shown | None:
    """A table of quotes (``vol_surface.parse_quotes``): the forward and calibration settings apply; the quote
    screen does not (a row has no contract to screen)."""
    name = " ".join(str(upload.name).split())
    american = inputs.model.is_american
    try:
        raw = exports.read_csv(upload.getvalue())  # an export of this app starts with its provenance lines
        with st.spinner("Fitting an SVI smile to every maturity…"):
            surface, messages, error = cache.market_surface(
                raw,
                inputs.spot,
                inputs.rate,
                inputs.dividend_yield,
                settings.forward,
                settings.calibration,
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
    if error or surface is None:
        st.error(f"Could not build a surface from this file: {error}")
        for message in messages:
            if not isinstance(message, vs.QuoteWarning):
                st.caption(message)
        return None
    _file_notes(surface)
    used = len(surface.quotes) if surface.quotes is not None else 0
    quoted = len(raw)
    panel = (
        f"Upload · {md_code(name)} · the user's table of quotes, not published · {number(quoted, 0)} rows → "
        f"{number(used, 0)} used ({number(quoted - used, 0)} dropped in reading, see Arbitrage and forwards) · "
        f"S {level(inputs.spot)}, r {pct(inputs.rate)}, q {pct(inputs.dividend_yield)} from the sidebar · "
        f"{_method_text(surface, settings)}"
    )
    rows = _surface_rows(surface, inputs.rate, inputs.dividend_yield)
    return Shown(
        QUOTES,
        surface,
        None,
        rows,
        inputs.rate,
        inputs.dividend_yield,
        None,
        panel,
        tuple(messages),
        _provenance(inputs, QUOTES, name=name),
        False,
        name,
    )


def _file_notes(surface: vs.VolSurface) -> None:
    """The fit's notes on an uploaded file, as warnings under the data panel: they are about the user's quotes (a
    spot far from the strikes, a poor fit, arbitrage held out), unlike the synthetic chain's, which the Arbitrage and
    forwards expander lists."""
    for note in surface.notes:
        st.warning(note)


def _provenance(inputs: Inputs, source: str, *, name: str = "quotes.csv", as_of: datetime | None = None):
    """Where the tab's exports come from (UI spec 2.13): the synthetic chain, the user's file with the forwards and
    calibration chosen for it, or the parametric SSVI surface."""
    snapshot = "none (a snapshot)"
    settings = surface_state.settings()
    fit = f"SVI {settings.calibration.replace('_', '-')}, spread-weighted" + (
        f"; {'; '.join(settings.changes())}" if settings.changes() else ""
    )
    if source == CHAIN:
        return exports.provenance(inputs, model="Black-76 inversion of the chain's quotes", dynamics=snapshot, fit=fit)
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
    prov = exports.provenance(
        inputs,
        tag=exports.UPLOAD,
        source=f"{exports.UPLOAD} · {name} — the user's file",
        model="Black-76 inversion of the file's quotes",
        dynamics=snapshot,
        forwards="parity" if settings.forward == "parity" else "the given r and q",
        fit=fit,
    )
    return prov if as_of is None else exports.Provenance(**{**prov.__dict__, "as_of": as_of})


# --------------------------------------------------------------------------------------------------------------------
# The data panel
# --------------------------------------------------------------------------------------------------------------------


def _filters(inputs: Inputs, container) -> None:
    """The Filters popover: the quote screen, the forwards and the calibration (``surface_state.KEYS``), each kept for
    the session; a non-default value shows the FIT chip on every tab."""
    defaults = surface_state.WIDGET_DEFAULTS
    with container.popover("Filters", icon=":material/filter_alt:", key="surface_data_edit"):
        st.caption("Quote screen (defaults after LSEG's volatility-surface service). Applies to chains, not tables.")
        st.toggle(
            "Out-of-the-money side only",
            value=defaults["sf_otm"],
            key="sf_otm",
            persist_state="session",
            help="Fit puts below the forward and calls at and above it. Off: the in-the-money side too, where it has "
            "more time value than half its spread; a strike quoted both ways averages the two vols.",
        )
        st.selectbox(
            "Max bid/ask spread",
            surface_state.SPREAD_CHOICES,
            index=surface_state.SPREAD_CHOICES.index(defaults["sf_spread"]),
            format_func=lambda m: "No limit" if math.isinf(m) else f"{number(m, 0)}× the expiry's median",
            key="sf_spread",
            persist_state="session",
            help="A quote whose spread ($) is wider than this multiple of its expiry's median is left out as wide.",
        )
        c = st.columns(2)
        c[0].number_input(
            "Min strikes per expiry",
            min_value=5,
            max_value=50,
            value=defaults["sf_strikes"],
            step=1,
            key="sf_strikes",
            persist_state="session",
            help="An expiry with fewer strikes left is left out whole (thin expiry).",
        )
        c[1].number_input(
            "Min days to expiry",
            min_value=0.0,
            max_value=30.0,
            value=defaults["sf_days"],
            step=0.5,
            key="sf_days",
            persist_state="session",
            help="Expiries closer than this are left out (short expiry). LSEG's default is 2 days; here 0, because "
            "every listed expiry prices the sidebar's contract.",
        )
        st.toggle(
            "Weeklies",
            value=defaults["sf_weeklies"],
            key="sf_weeklies",
            persist_state="session",
            help="Fit the expiries that are not standard monthlies (third Fridays). Off, the sidebar lists only "
            "the monthlies.",
        )
        st.segmented_control(
            "Forwards",
            list(surface_state.FORWARD_CHOICES),
            default=defaults["sf_fwd"],
            required=True,
            key="sf_fwd",
            persist_state="session",
            help="Each expiry's forward and discount factor: put-call parity on its calls and puts "
            "(C − P = DF·(F − K), fitted near the money), or the fallback rate and yield (the sidebar's; for the "
            "synthetic chain the Treasury rate and its 1.30% yield). American quotes always take the rate and yield.",
        )
        st.segmented_control(
            "Calibration",
            list(surface_state.CALIBRATION_CHOICES),
            default=defaults["sf_cal"],
            required=True,
            key="sf_cal",
            persist_state="session",
            help="Arbitrage-free: each SVI smile held free of butterfly arbitrage, and of calendar arbitrage against "
            "the expiry before it. Unconstrained: the best smile through each expiry's quotes, arbitrage and all. "
            "Either way each quote weighs one over its half-spread in vol.",
        )
        if not surface_state.settings().is_default:
            st.button("Reset to defaults", key="sf_reset", icon=":material/restart_alt:", on_click=_reset_fit)
        # The samples show the layout of a table of quotes, at the sidebar's spot.
        st.caption("Sample files (synthetic) in the layout of a table of quotes:")
        prov = exports.provenance(
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
            st.download_button(
                label,
                exports.builder(_sample_csv, inputs.spot, inputs.rate, inputs.dividend_yield, prices, prov),
                prov.file_name(f"sample_option_{'prices' if prices else 'quotes'}", "csv"),
                "text/csv",
                on_click="ignore",
                icon=":material/download:",
                key=key,
                width="stretch",
            )


def _sample_csv(spot: float, rate: float, dividend_yield: float, prices: bool, provenance: exports.Provenance) -> bytes:
    """A sample quotes file with its provenance lines, built when its button is clicked (not on every rerun)."""
    return exports.csv(vs.sample_quotes(spot, rate, dividend_yield, prices=prices), provenance)


def _uploader(container, *, compact: bool):
    """The one file uploader, and the file kept from an earlier run (the uploader forgets it when not drawn)."""
    upload = container.file_uploader(
        "Chain or quotes file (CSV)",
        type=["csv"],
        key="quotes_file",
        on_change=_keep_upload,
        label_visibility="collapsed" if compact else "visible",
        help="A chain in the app's long layout, a yfinance export or Cboe's download (read_chain), or a table of "
        "strikes, times to expiry and vols or prices.",
    )
    kept = st.session_state.get("quotes_upload")
    if upload is None and kept is not None:
        row = container.container(horizontal=True, vertical_alignment="center", gap="small", key="quotes_kept")
        row.caption(f":material/description: {md_code(kept.name)} · {_size(kept.size)} · kept for this session.")
        row.button(
            "Remove file", key="quotes_remove", icon=":material/close:", type="tertiary", on_click=_remove_upload
        )
        return kept
    return upload


def _data_panel(inputs: Inputs) -> Shown | None:
    """The data panel (``surface_data``): the source, its one-line state and the Filters; returns what to show."""
    panel = st.container(border=True, key="surface_data")
    row = panel.container(horizontal=True, vertical_alignment="center", gap="small", key="surface_data_row")
    sources = [CHAIN, QUOTES] if inputs.context is not None else [SSVI, QUOTES]
    source = row.segmented_control(
        "Surface source",
        sources,
        default=sources[0],
        required=True,
        key="surface_source",
        persist_state="session",
        label_visibility="collapsed",
        help="The synthetic chain is the surface every tab prices from. An upload is fitted here only.",
    )
    _filters(inputs, row)
    shown: Shown | None
    if source == CHAIN:
        shown = _chain_source(inputs)
    elif source == SSVI:
        shown = _ssvi_source(inputs, row)
    else:
        kept = st.session_state.get("quotes_upload")
        upload = _uploader(panel, compact=kept is not None)
        if upload is None:
            panel.markdown(":small[Upload a chain: " + " ".join(LAYOUTS) + " " + QUOTES_LAYOUT + "]")
            if inputs.context is not None:
                panel.button("Use synthetic", key="surface_use_synthetic", on_click=_use_synthetic)
            return None
        shown = _upload_source(inputs, upload)
    if shown is not None:
        note = whatif.note(whatif.current())
        panel.markdown(f":small[{shown.panel}]" + (f"  \n:small[:gray[{note}]]" if note else ""))
    return shown


# --------------------------------------------------------------------------------------------------------------------
# Tiles
# --------------------------------------------------------------------------------------------------------------------


def _tiles(shown: Shown | None) -> None:
    """Six tiles (UI spec 3.4): ATMF 30D (VIX-style below it), 25Δ RR and BF at 30 days, the term slope, the fit
    against the spreads and the arbitrage counts; ``–`` with the reason where a value cannot be measured."""
    blank = {"help": "No surface: upload a file or pick another source."}
    if shown is None:
        labels = ("ATMF 30D", "25Δ RR 30D", "25Δ BF 30D", "TERM 30→365D", "FIT / ½‑SPR", "ARB")
        card_row("surface_tiles", TILE_GROUPS, [(lbl, MISSING, blank) for lbl in labels])
        return
    surface, rate = shown.surface, shown.rate
    T30, T365 = TILE_DAYS / 365.0, TERM_DAYS / 365.0
    atmf30, atmf365 = dg.atmf_vol(surface, T30), dg.atmf_vol(surface, T365)
    rr, bf = dg.risk_reversal(surface, T30, rate=rate)
    outside = f"– where {TILE_DAYS} or {TERM_DAYS} days lies outside the fitted expiries (no extrapolation in time)."
    atmf_options: dict = {
        "help": f"At-the-money-forward vol at a constant {TILE_DAYS} days: total variance at K = F interpolated "
        f"linearly in time between the expiries around it ({vol(atmf30, 4)}). {outside}"
    }
    if shown.fit is not None:
        vix, pair = dg.vix_style_vol(shown.fit, TILE_DAYS)
        if math.isfinite(vix):
            names = [shown.fit.expiries.at[i, "expiry"] for i in pair or ()]
            atmf_options |= {
                "delta": f"VIX‑style {vol(vix)}",  # UI-26: the model-free 30-day vol
                "delta_color": "off",
                "delta_arrow": "off",
                "help": atmf_options["help"]
                + f" Below it, the model-free {TILE_DAYS}-day vol by the VIX method (Cboe white paper): every "
                f"out-of-the-money mid of {' and '.join(short_date(d) for d in names)}, interpolated in total "
                f"variance; {vol_change(vix - atmf30)} above ATMF, the skew's and wings' share of the variance.",
            }
    cal, bfly = surface.calendar_violations, surface.butterfly_violations
    term = atmf365 - atmf30
    tiles = [
        ("ATMF 30D", vol(atmf30), atmf_options),
        (
            "25Δ RR 30D",
            vol_change(rr),
            {
                "help": f"σ(25Δ call) − σ(25Δ put) at {TILE_DAYS} days, spot delta e^(−qT)N(d₁). Negative: put "
                f"skew (puts richer), the FX market's sign convention. {outside}"
            },
        ),
        (
            "25Δ BF 30D",
            vol_change(bf),
            {"help": f"½(σ25Δc + σ25Δp) − σATMF at {TILE_DAYS} days: the smile's curvature. {outside}"},
        ),
        (
            "TERM 30→365D",
            vol_change(term),
            {
                "help": f"ATMF {TERM_DAYS}D ({vol(atmf365)}) − ATMF {TILE_DAYS}D: positive is an upward-sloping term "
                f"structure. {outside}"
            },
        ),
        _fit_tile(shown),
        (
            "ARB",
            f"{number(cal, 0)} / {number(bfly, 0)}",
            {
                "help": "Calendar / butterfly violations: grid points of the fitted surface where total variance falls "
                "with maturity at fixed K/F, or Durrleman's g < 0 (a negative density). "
                + ("0 / 0: passes." if cal == bfly == 0 else "See Arbitrage and forwards.")
            },
        ),
    ]
    card_row("surface_tiles", TILE_GROUPS, tiles)


def _fit_tile(shown: Shown) -> tuple[str, str, dict]:
    """FIT: the median RMSE across expiries against the median half-spread in vol, and the share of quotes in band."""
    rows = shown.rows
    rmse = float(np.nanmedian(rows["rmse"])) if rows["rmse"].notna().any() else math.nan
    if shown.fit is not None:
        used = shown.fit.quotes[shown.fit.quotes["status"] == dg.USED]
        half = 0.5 * (used["iv_ask"] - used["iv_bid"]).to_numpy(float)
        in_band = float(np.mean(np.abs(used["fit_iv"] - used["iv_mid"]).to_numpy(float) <= half))
    elif shown.surface.quotes is not None and "vol_half_spread" in shown.surface.quotes:
        quotes = shown.surface.quotes
        half = quotes["vol_half_spread"].to_numpy(float)
        known = np.isfinite(half) & (half > 0)
        in_band = (
            float(np.mean(np.abs(quotes["fitted_iv"] - quotes["iv"]).to_numpy(float)[known] <= half[known]))
            if known.any()
            else math.nan
        )
    else:
        return ("FIT / ½‑SPR", MISSING, {"help": "A parametric surface has no quotes to fit."})
    median_half = float(np.nanmedian(half)) if np.isfinite(half).any() else math.nan
    options = {
        "help": "Median across expiries of the fit's RMSE in vol / the median half bid/ask spread in vol of "
        "the quotes used: a fit well inside the spreads is as good as the quotes allow. Then the share of used "
        "quotes whose fitted vol lies within their bid/ask.",
    }
    if math.isfinite(in_band):
        options |= {"delta": f"{number(in_band * 100, 0)}% in band", "delta_color": "off", "delta_arrow": "off"}
    value = (
        f"{number(rmse * 100, 2)} vp"
        if not math.isfinite(median_half)
        else (f"{number(rmse * 100, 2)} / {number(median_half * 100, 2)} vp")
    )
    return ("FIT / ½‑SPR", value, options)


# --------------------------------------------------------------------------------------------------------------------
# The primary view
# --------------------------------------------------------------------------------------------------------------------


def _focus(shown: Shown) -> tuple[int, str]:
    """The focus expiry's row and the session key that holds it."""
    rows = shown.rows
    if shown.source == CHAIN:
        current = st.session_state.get("expiry")
        targets = list(rows["target"])
        return (targets.index(current) if current in targets else 0), "expiry"
    default = int(np.argmin(np.abs(rows["maturity"].to_numpy(float) * 365.0 - TILE_DAYS)))
    current = st.session_state.get(FOCUS_KEY)
    if not isinstance(current, (int, np.integer)) or not 0 <= current < len(rows):
        st.session_state[FOCUS_KEY] = default
        current = default
    return int(current), FOCUS_KEY


def _view_controls(shown: Shown, focus: int, focus_key: str) -> tuple[str, str]:
    """The view switch, the focus expiry's ◂ name ▸ stepper and, for the smile, its x axis."""
    rows = shown.rows
    targets = list(rows["target"])
    bar = st.container(horizontal=True, vertical_alignment="center", gap="small", key="surface_view_bar")
    view = bar.segmented_control(
        "View",
        VIEWS,
        default=VIEWS[0],
        required=True,
        key="sf_view",
        persist_state="session",
        label_visibility="collapsed",
    )
    stepper = bar.container(horizontal=True, vertical_alignment="center", gap="xsmall", width="content")
    stepper.button(
        "",
        icon=":material/chevron_left:",
        key="sf_prev",
        on_click=on_expiry_step,
        args=(targets, focus_key, focus, -1),
        disabled=focus == 0,
        help="Previous expiry",
    )
    row = rows.iloc[focus]
    dated = row["expiry"] is not None
    stepper.markdown(f"**{row['name']}**" + (f" · {time_left(row['minutes'])}" if dated else " to expiry"))
    stepper.button(
        "",
        icon=":material/chevron_right:",
        key="sf_next",
        on_click=on_expiry_step,
        args=(targets, focus_key, focus, +1),
        disabled=focus == len(rows) - 1,
        help="Next expiry",
    )
    axis = X_AXES[0]
    if view == "Smile":
        axis = bar.segmented_control(
            "x axis",
            X_AXES,
            default=X_AXES[0],
            required=True,
            key="sf_x",
            persist_state="session",
            label_visibility="collapsed",
            help="The smile's x axis: K/F in percent, log-moneyness ln(K/F), the call delta at the fitted vol "
            "(0.90 on the put side to 0.10), or the strike.",
        )
    return view, axis


def _smile_data(shown: Shown, focus: int) -> charts_surface.SmileData:
    surface, row = shown.surface, shown.rows.iloc[focus]
    j, T = int(row["slice"]), float(row["maturity"])
    forward = float(row["forward"])
    reach = surface.reach_at(T)
    if surface.quoted_y is not None:
        lo, hi = (float(v) for v in surface.quoted_y[j])
    else:
        log_fs = dg.log_forward_ratio(surface, T)
        lo, hi = float(np.log(surface.moneyness[0]) - log_fs), float(np.log(surface.moneyness[-1]) - log_fs)
    width = hi - lo
    left, right = max(lo - 0.25 * width, -reach), min(hi + 0.25 * width, reach)
    y = np.linspace(min(left, lo), max(right, hi), 241)
    w = dg.smile_variance(surface, T, y)[0]
    fit_vols = np.sqrt(np.maximum(w, 0.0) / T)
    inside = (y >= lo - 1e-9) & (y <= hi + 1e-9)
    quotes = excluded = None
    used = None
    violations: list[tuple[float, str]] = []
    if shown.fit is not None:
        label = int(row["label"])
        mine = shown.fit.quotes[shown.fit.quotes["label"] == label]
        used = mine[mine["status"] == dg.USED]
        quotes = {
            "strike": used["strike"].to_numpy(float),
            "call": (used["type"] == "call").to_numpy(),
            "iv_bid": used["iv_bid"].to_numpy(float),
            "iv_ask": used["iv_ask"].to_numpy(float),
            "iv_mid": used["iv_mid"].to_numpy(float),
            "fit_iv": used["fit_iv"].to_numpy(float),
        }
        # Excluded quotes, except the in-the-money side (the out-of-the-money quote of the same strike is drawn):
        # at its mid vol, or at its ask vol for an offer only.
        other = mine[(mine["status"] != dg.USED) & (mine["status"] != "ITM side")]
        iv = other["iv_mid"].to_numpy(float)
        iv = np.where(np.isfinite(iv), iv, other["iv_ask"].to_numpy(float))
        window = (other["strike"] >= forward * math.exp(y[0])) & (other["strike"] <= forward * math.exp(y[-1]))
        keep = np.isfinite(iv) & window.to_numpy()
        excluded = {
            "strike": other["strike"].to_numpy(float)[keep],
            "iv": iv[keep],
            "call": (other["type"] == "call").to_numpy()[keep],
            "status": other["status"].to_numpy(object)[keep],
        }
    elif surface.quotes is not None:
        mine = surface.quotes[np.isclose(surface.quotes["maturity"], T, rtol=0.0, atol=1e-9)]
        half = mine["vol_half_spread"].to_numpy(float)
        half = np.where(np.isfinite(half), half, 0.0)
        iv = mine["iv"].to_numpy(float)
        quotes = {
            "strike": mine["strike"].to_numpy(float),
            "call": mine["strike"].to_numpy(float) >= forward,
            "iv_bid": iv - half,
            "iv_ask": iv + half,
            "iv_mid": iv,
            "fit_iv": mine["fitted_iv"].to_numpy(float),
        }
    for k_over_s, kind in dg.slice_violations(surface, j):
        violations.append((k_over_s * (surface.spot or 1.0), kind))
    n_used = len(quotes["strike"]) if quotes is not None else 0
    in_band = row["in_band"]
    parts = [str(row["name"]), time_left(row["minutes"])]
    if math.isfinite(row["rmse"]):
        parts.append(f"RMSE {number(row['rmse'] * 100, 2)} vp")
    if math.isfinite(in_band):
        parts.append(f"{number(in_band * 100, 0)}% in band")
    if quotes is not None:
        parts.append(f"{number(n_used, 0)} used")
    return charts_surface.SmileData(
        forward=forward,
        fit_strikes=forward * np.exp(y),
        fit_vols=fit_vols,
        fit_inside=inside,
        quoted_range=(forward * math.exp(lo), forward * math.exp(hi)),
        note=" · ".join(parts),
        quotes=quotes,
        excluded=excluded,
        violations=tuple(violations[:6]),
    )


def _call_delta(shown: Shown, focus: int):
    """Strikes to their call delta (spot convention) at the focus expiry's fitted vol, for the Call Δ axis."""
    surface, row = shown.surface, shown.rows.iloc[focus]
    T, forward = float(row["maturity"]), float(row["forward"])
    carry = forward * float(row["discount_factor"]) / (surface.spot if surface.spot is not None else forward)

    def delta(strikes):
        y = np.log(np.asarray(strikes, dtype=float) / forward)
        w = dg.smile_variance(surface, T, y)[0]
        with np.errstate(all="ignore"):
            root = np.sqrt(w)
            return carry * ndtr(-y / root + 0.5 * root)

    return delta


def _smile(shown: Shown, focus: int, axis: str) -> None:
    data = _smile_data(shown, focus)
    fig = charts_surface.smile_fit(data, axis, call_delta=_call_delta(shown, focus))
    alt = f"Implied vol smile of {shown.rows.iloc[focus]['name']} against {charts_surface.X_AXES[axis]}"
    charts.show(fig, key="chart_smile", alt=alt + (", with each quote's bid and ask" if data.quotes else ""))


def _quotes_view(shown: Shown, focus: int) -> None:
    """The focus expiry's quotes, each with its status (UI spec 3.4: Mid vs fit, never Edge, against the fit)."""
    row = shown.rows.iloc[focus]
    if shown.fit is None:
        if shown.surface.quotes is None:
            st.info("A parametric surface has no quotes. Upload a chain to see each quote against the fit.")
            return
        mine = shown.surface.quotes[np.isclose(shown.surface.quotes["maturity"], row["maturity"], rtol=0, atol=1e-9)]
        frame = pd.DataFrame(
            {
                "Strike": mine["strike"],
                "IV mid": mine["iv"] * 100,
                "Fit": mine["fitted_iv"] * 100,
                "Mid vs fit": (mine["iv"] - mine["fitted_iv"]) * 100,
                "Status": dg.USED,
            }
        )
        kinds = {"Strike": "strike", "IV mid": "vol", "Fit": "vol", "Mid vs fit": "vol_change", "Status": "text"}
        helps = {k: QUOTE_HELP[k] for k in frame.columns}
        table(frame, kinds, helps, key="tbl_surface_quotes")
        return
    every = shown.fit.quotes[shown.fit.quotes["label"] == int(row["label"])]
    show_all = st.toggle(
        "Every quote",
        value=False,
        key="sf_all_quotes",
        persist_state="session",
        help="Off: the quotes the fit used. On: every quote of the expiry, with why the screen left it out.",
    )
    mine = every if show_all else every[every["status"] == dg.USED]
    frame = pd.DataFrame(
        {
            "Strike": mine["strike"].to_numpy(float),
            "Side": mine["type"].map({"call": "C", "put": "P"}).to_numpy(object),
            "Bid": mine["bid"].to_numpy(float),
            "Ask": mine["ask"].to_numpy(float),
            "IV bid": mine["iv_bid"].to_numpy(float) * 100,
            "IV mid": mine["iv_mid"].to_numpy(float) * 100,
            "IV ask": mine["iv_ask"].to_numpy(float) * 100,
            "Fit": mine["fit_iv"].to_numpy(float) * 100,
            "Mid vs fit": (mine["iv_mid"] - mine["fit_iv"]).to_numpy(float) * 100,
            "Status": mine["status"].to_numpy(object),
        }
    )
    kinds = {
        "Strike": "strike",
        "Side": "text",
        "Bid": "premium",
        "Ask": "premium",
        "IV bid": "vol",
        "IV mid": "vol",
        "IV ask": "vol",
        "Fit": "vol",
        "Mid vs fit": "vol_change",
        "Status": "text",
    }
    if shown.vendor:
        frame["Vendor IV"] = mine["vendor_iv"].to_numpy(float) * 100
        kinds["Vendor IV"] = "vol"
    helps = {k: QUOTE_HELP[k] for k in frame.columns}
    table(frame, kinds, helps, key="tbl_surface_quotes")
    st.caption(
        f"{number(len(mine), 0)} of {number(len(every), 0)} quotes of {row['name']} · the chain's "
        f"{number(len(shown.fit.quotes), 0)} quotes inverted in {number(shown.fit.inversion_ms, 0)} ms (batch LBR)."
    )


#: The Quotes view's column help (UI spec 2.5's edge convention: against the fit the column is Mid vs fit).
QUOTE_HELP = {
    "Strike": "Listed strike.",
    "Side": "C call, P put.",
    "Bid": "Best bid, $ per share (index points).",
    "Ask": "Best offer, $ per share (index points).",
    "IV bid": "Black-76 vol of the bid at the expiry's forward and discount factor; – for a zero bid.",
    "IV mid": "Black-76 vol of the mid; – for a zero bid (an offer only has no mid).",
    "IV ask": "Black-76 vol of the ask.",
    "Fit": "The fitted SVI smile's vol at this strike; – beyond the smile's reach.",
    "Mid vs fit": "σ_mid − σ_fit: positive where the market is rich against the fit. A residual, not an edge: the fit "
    "is made from these mids.",
    "Status": "used, or why the screen left it out: " + ", ".join(dg.REASONS) + ".",
    "Vendor IV": "The data vendor's own implied vol, a diagnostic only: vendors use their own rates and models.",
}


def _term(shown: Shown) -> None:
    surface = shown.surface
    constant = [(d, dg.atmf_vol(surface, d / 365.0)) for d in dg.CONSTANT_MATURITIES]
    listed = []
    previous: date | None = shown.as_of.date() if shown.as_of is not None else None
    for _, row in shown.rows.iterrows():
        holiday = False
        if row["expiry"] is not None and previous is not None:
            day = previous + timedelta(days=1)
            while day <= row["expiry"]:
                if day.weekday() < 5 and not clock.is_trading_day(day):
                    holiday = True
                day += timedelta(days=1)
            previous = row["expiry"]
        listed.append(
            {
                "days": row["maturity"] * 365.0,
                "atmf": row["atmf"],
                "rr25": row["rr25"],
                "bf25": row["bf25"],
                "name": row["name"],
                "holiday": holiday,
            }
        )
    fig = charts_surface.term_chart(constant, listed)
    charts.show(fig, key="chart_term", alt="ATMF vol term structure, with the 25-delta risk reversal and butterfly")
    forwards = dg.forward_vols(surface)
    strip = pd.DataFrame({f"{a}→{b}d": [v * 100] for a, b, v in forwards})
    table(
        strip,
        dict.fromkeys(strip.columns, "vol"),
        dict.fromkeys(strip.columns, FORWARD_VOL_HELP),
        key="tbl_surface_forward_vols",
    )
    if any(row["holiday"] for row in listed):
        st.caption("Hollow markers: an exchange holiday falls between the expiry and the one before it.")


FORWARD_VOL_HELP = (
    "ATMF forward vol between the two constant maturities: √((w₂ − w₁)/(T₂ − T₁)); – where either end is not fitted."
)


def _grid(shown: Shown) -> None:
    grid = dg.delta_grid(shown.surface, rate=shown.rate)
    grid = grid[grid.notna().any(axis=1)]  # constant maturities beyond the fitted expiries: no row
    if grid.empty:
        st.info("No constant maturity lies within the fitted expiries.")
        return
    labels = [delta_label(d, atm=False) if abs(d) != 0.5 else "50Δ" for d in grid.columns]
    frame = pd.DataFrame(grid.to_numpy() * 100, columns=labels)
    frame.insert(0, "IV (%)", [f"{d}d" for d in grid.index])
    values = frame[labels].to_numpy(float)
    finite = values[np.isfinite(values)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    ramp = theme.SEQUENTIAL[theme.mode()]

    def shade(value):
        if not np.isfinite(value):
            return ""
        i = round((value - lo) / max(hi - lo, 1e-12) * (len(ramp) - 1))
        dark = (i >= len(ramp) // 2) == (theme.mode() == "light")
        text = theme.SEMANTIC[theme.mode()]["surface" if dark else "text"]
        return f"background-color: {ramp[i]}; color: {text}"

    styled = frame.style.map(shade, subset=labels).format(precision=2, na_rep=MISSING, subset=labels)
    config = {
        "IV (%)": column(
            "text", "IV (%)", help="Implied vol in %, at a constant maturity in calendar days.", pinned=True
        )
    }
    for label, d in zip(labels, grid.columns, strict=True):
        side = "call" if d > 0 else "put"
        config[label] = column(
            "vol", label, help=f"Implied vol (%) of the {side} at spot delta {number(abs(d), 2)}, at its own vol."
        )
        config[label]["label"] = label  # the unit is in the first column's header: 17 columns fit without it
    st.dataframe(
        styled,
        column_config=config,
        hide_index=True,
        row_height=28,
        height="content",
        placeholder=MISSING,
        key="tbl_surface_grid",
    )
    st.caption("Implied vol by spot delta at constant maturities; – beyond the fitted expiries or the smile's reach.")


def _three_d(shown: Shown) -> None:
    surface = shown.surface
    days = np.array([d for d in dg.CONSTANT_MATURITIES if math.isfinite(dg.atmf_vol(surface, d / 365.0))], float)
    if not days.size:
        st.info("No constant maturity lies within the fitted expiries.")
        return
    m = np.linspace(80.0, 120.0, 41)
    vols = np.full((days.size, m.size), np.nan)
    for i, d in enumerate(days):
        T = d / 365.0
        y = np.log(m / 100.0)
        reach = surface.reach_at(T)
        w = dg.smile_variance(surface, T, y)[0]
        vols[i] = np.where((np.abs(y) <= reach) & (w > 0), np.sqrt(np.maximum(w, 0.0) / T), np.nan)
    charts.show(
        charts_surface.implied_3d(m, days, vols),
        key="chart_surface_3d",
        alt="Implied vol surface in 3-D over K/F and days to expiry",
    )


# --------------------------------------------------------------------------------------------------------------------
# The expiry table, the diagnostics, the readout and the exports
# --------------------------------------------------------------------------------------------------------------------

EXPIRY_HELP = {
    "Expiry": "Expiry date and settlement (AM: the opening print, SPX monthlies; PM: the close). ◂ marks the focus "
    "expiry; click a row to make it the focus (the sidebar's expiry, for the synthetic chain).",
    "DTE": "Time to the settlement.",
    "F": "Forward, from put-call parity on the expiry's quotes (or the fallback carry: see the expander).",
    "Carry gap": "ln(F/S) − (r − q)T against the fallback carry, in bp of the forward.",
    "ATMF": "Vol at K = F.",
    "25Δ RR": "σ(25Δ call) − σ(25Δ put), spot delta. Negative: put skew.",
    "25Δ BF": "½(σ25Δc + σ25Δp) − σATMF.",
    "RMSE": "Root-mean-square gap between the fit and the quotes' mid vols.",
    "In band": "Share of used quotes whose fitted vol lies within their bid/ask in vol.",
    "Used": "Quotes the fit used.",
    "Conf.": "Share of the call-delta interval 0.10 to 0.90 that the used strikes span (after ORATS's confidence).",
    "Flags": "carry: the fallback carry misses the parity forward by more than 5 bp and half the ATM spread; thin: "
    "fewer than twice the minimum strikes; cal, bfly: arbitrage on the slice's grid; extrap: a 25-delta strike "
    "beyond the quotes.",
}


def _expiry_table(shown: Shown, focus: int, focus_key: str) -> None:
    rows = shown.rows
    names = [f"{name} ◂" if i == focus else str(name) for i, name in enumerate(rows["name"])]
    frame = pd.DataFrame(
        {
            "Expiry": names,
            "DTE": [time_left(m) for m in rows["minutes"]],
            "F": rows["forward"].to_numpy(float),
            "Carry gap": rows["carry_gap_bp"].to_numpy(float),
            "ATMF": rows["atmf"].to_numpy(float) * 100,
            "25Δ RR": rows["rr25"].to_numpy(float) * 100,
            "25Δ BF": rows["bf25"].to_numpy(float) * 100,
            "RMSE": rows["rmse"].to_numpy(float) * 100,
            "In band": rows["in_band"].to_numpy(float) * 100,
            "Used": rows["used"].to_numpy(float),
            "Conf.": rows["confidence"].to_numpy(float) * 100,
            "Flags": rows["flags"].to_numpy(object),
        }
    )
    kinds = {
        "Expiry": "text",
        "DTE": "text",
        "F": "level",
        "Carry gap": "bp",
        "ATMF": "vol",
        "25Δ RR": "vol_change",
        "25Δ BF": "vol_change",
        "RMSE": "vol_change",
        "In band": "share",
        "Used": "count",
        "Conf.": "share",
        "Flags": "text",
    }
    targets = list(rows["target"])
    key = f"tbl_surface_expiries_{targets[focus]}"  # remounts, with no stale selection, when the focus moves
    table(
        frame,
        kinds,
        EXPIRY_HELP,
        key=key,
        selection_mode="single-row",
        selection_default={"selection": {"rows": [focus]}},
        on_select=functools.partial(on_expiry_select, key, targets, focus_key),
        max_rows=16,
    )


def _diagnostics(shown: Shown, focus: int) -> None:
    """The Arbitrage and forwards expander: the forwards, the arbitrage notes, Durrleman's g of the focus expiry,
    local-vol coverage, the 90–110 and Mixon skews, and the notes on reading the quotes."""
    surface = shown.surface
    expander = st.expander("Arbitrage and forwards", key="surface_diag", on_change="rerun", icon=":material/rule:")
    if not expander.open:
        return
    with expander:
        forwards = surface.forwards
        if forwards is not None and len(forwards):
            spot = surface.spot or 1.0
            frame = pd.DataFrame(
                {
                    "T (days)": forwards["maturity"].to_numpy(float) * 365.0,
                    "F": forwards["forward"].to_numpy(float),
                    "DF": forwards["discount_factor"].to_numpy(float),
                    "r": forwards["rate"].to_numpy(float) * 100,
                    "q": forwards["dividend_yield"].to_numpy(float) * 100,
                    "Pairs": forwards["pairs"].to_numpy(float),
                    "Residual": forwards["residual"].to_numpy(float),
                    "SE of F": forwards["forward_se"].to_numpy(float),
                    "Gap": 1e4
                    * (
                        np.log(forwards["forward"].to_numpy(float) / spot)
                        - (shown.rate - shown.dividend_yield) * forwards["maturity"].to_numpy(float)
                    ),
                    "Source": forwards["source"].map(FORWARD_SOURCE_LABELS).to_numpy(object),
                }
            )
            table(
                frame,
                {
                    "T (days)": "days",
                    "F": "level",
                    "DF": "premium_diag",
                    "r": "rate",
                    "q": "rate",
                    "Pairs": "count",
                    "Residual": "premium",
                    "SE of F": "premium",
                    "Gap": "bp",
                    "Source": "text",
                },
                {
                    "T (days)": "Calendar days to the settlement.",
                    "F": "Forward.",
                    "DF": "Discount factor to the settlement.",
                    "r": "Implied rate −ln(DF)/T, the box rate the options discount at.",
                    "q": "Implied yield r − ln(F/S)/T.",
                    "Pairs": "Call/put pairs near the money in the parity fit, after outliers.",
                    "Residual": "RMS gap of C − P from the fitted line DF·(F − K), $ per share.",
                    "SE of F": "Standard error of the forward, index points.",
                    "Gap": "ln(F/S) − (r − q)T against the fallback carry, bp of the forward.",
                    "Source": "parity; parity (DF: rate) where the slope could not be measured; rate and yield.",
                },
                labels={"r": "Implied r", "q": "Implied q", "Gap": "Gap"},
                key="tbl_surface_forwards",
            )
        notes = list(surface.notes)
        if surface.calendar_violations == surface.butterfly_violations == 0 and not notes:
            st.caption("No calendar or butterfly violation on the fitted grid, and no note on the quotes.")
        for note in notes:
            st.markdown(f":small[{note}]")
        row = shown.rows.iloc[focus]
        T = float(row["maturity"])
        reach = surface.reach_at(T)
        y = np.linspace(-reach, reach, 241)
        if surface.quoted_y is not None:
            lo, hi = surface.quoted_y[int(row["slice"])]
        else:
            lo, hi = -reach, reach
        g = dg.durrleman(surface, T, y)
        charts.show(
            charts_surface.durrleman_chart(y, g, (y >= lo) & (y <= hi)),
            key="chart_durrleman",
            alt=f"Durrleman's g of the {row['name']} smile against log-moneyness",
        )
        valid = int(np.isfinite(surface.local_vol).sum())
        quoted = surface.quoted_points
        st.caption(
            f"Local vol defined at {_share(valid, quoted)} of the grid points inside the quotes "
            f"({number(valid, 0)} of {number(quoted, 0)}); blank where the surface has arbitrage or ∂w/∂T is not "
            "resolved."
        )
        skew = pd.DataFrame(
            {
                "Expiry": shown.rows["name"].to_numpy(object),
                "Skew 90–110": shown.rows["skew_90_110"].to_numpy(float) * 100,
                "Mixon skew": shown.rows["mixon"].to_numpy(float),
            }
        )
        table(
            skew,
            {"Expiry": "text", "Skew 90–110": "vol_change", "Mixon skew": "delta"},
            {
                "Expiry": "Expiry.",
                "Skew 90–110": "σ(K/F 90%) − σ(K/F 110%).",
                "Mixon skew": "(σ25Δp − σ25Δc) / σATMF: Mixon's (2011) normalised skew, comparable across expiries.",
            },
            key="tbl_surface_skew",
        )
        for message in shown.notes:
            st.markdown(f":small[{message}]")


FORWARD_SOURCE_LABELS = {"parity": "parity", "parity_forward": "parity (DF: rate)", "rates": "rate and yield"}


def _share(valid: int, quoted: int) -> str:
    """``valid`` of ``quoted`` points as a percentage that reads 100% or 0% only when it is exactly that (649 of 652
    points used to show as "100%" next to a note that local vol is blank at three of them)."""
    if not quoted:
        return MISSING
    if 0 < valid < quoted:
        return number(min(max(math.floor(valid / quoted * 1000) / 10, 0.1), 99.9), 1) + "%"
    return number(valid / quoted * 100, 0) + "%"


def _contract_readout(shown: Shown, inputs: Inputs) -> None:
    """One line: the Pricing contract's vol on this surface, inside the quotes or extrapolated."""
    surface = shown.surface
    T = inputs.expiry
    k_over_s = inputs.strike / inputs.spot
    iv = surface.implied_vol_at(k_over_s, T, extrapolate=True)
    forward = inputs.spot * math.exp(dg.log_forward_ratio(surface, T)) if shown.published else inputs.forward
    when = (
        f"{short_date(inputs.expiry_date)} {inputs.settlement}"
        if inputs.expiry_date is not None
        else time_left(T * clock.MINUTES_PER_YEAR)
    )
    lead = f"Pricing contract K {strike(inputs.strike, 5.0 if inputs.context is not None else 0.01)} · {when}"
    if not math.isfinite(iv):
        st.caption(f"{lead} → no vol on this surface: beyond its expiries or its reach.")
        return
    where = (
        f"extrapolated, |k| {number(abs(math.log(inputs.strike / forward)), 2)}"
        if surface.is_extrapolated(k_over_s, T)
        else "inside quotes"
    )
    tail = "" if shown.published else f" · Pricing uses {'the synthetic chain' if inputs.context else 'the typed σ'}"
    st.caption(f"{lead} → σ {vol(iv)} (K/F {moneyness(inputs.strike, forward)}, {where}){tail}")


def _grid_frames(surface: vs.VolSurface) -> dict[str, pd.DataFrame]:
    """The implied and local vol grids for export, with maturities in days (strikes across)."""
    index = pd.Index(np.round(surface.maturities * 365.0, 4), name="Days to expiry \\ Strike")
    spot = surface.spot if surface.spot is not None else 1.0
    columns = pd.Index(np.round(surface.moneyness * spot, 6), name="Strike")
    return {
        "ImpliedVol": pd.DataFrame(surface.implied_vol, index=index, columns=columns),
        "LocalVol": pd.DataFrame(surface.local_vol, index=index, columns=columns),
    }


def _used_quotes(shown: Shown) -> pd.DataFrame:
    if shown.fit is not None:
        used = shown.fit.quotes[shown.fit.quotes["status"] == dg.USED]
        return pd.DataFrame(
            {
                "expiry": [pd.Timestamp(d).date().isoformat() for d in used["expiry"]],
                "settlement": used["settlement"],
                "type": used["type"],
                "strike": used["strike"],
                "bid": used["bid"],
                "ask": used["ask"],
                "days": used["maturity"] * 365.0,
                "forward": used["forward"],
                "discount_factor": used["discount_factor"],
                "iv_bid": used["iv_bid"],
                "iv_mid": used["iv_mid"],
                "iv_ask": used["iv_ask"],
                "fit_iv": used["fit_iv"],
            }
        )
    quotes = shown.surface.quotes
    if quotes is None:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "strike": quotes["strike"],
            "days": quotes["maturity"] * 365.0,
            "iv": quotes["iv"],
            "fit_iv": quotes["fitted_iv"],
        }
    )


def _svi_json(shown: Shown, provenance: exports.Provenance) -> bytes:
    surface = shown.surface
    slices = []
    for _, row in shown.rows.iterrows():
        fit = surface.svi[int(row["slice"])]
        slices.append(
            {
                "expiry": row["expiry"].isoformat() if row["expiry"] is not None else None,
                "settlement": row["settlement"],
                "days": float(row["maturity"] * 365.0),
                "forward": float(row["forward"]),
                "discount_factor": float(row["discount_factor"]),
                "svi": {"a": fit.a, "b": fit.b, "rho": fit.rho, "m": fit.m, "sigma": fit.s},
            }
        )
    note = "raw SVI w(k) = a + b(ρ(k − m) + √((k − m)² + σ²)), k = ln(K/F), w total implied variance"
    return exports.json({"parameterisation": note, "expiries": slices}, provenance)


def _workbook(shown: Shown, provenance: exports.Provenance) -> bytes:
    frames = _grid_frames(shown.surface)
    sheets = {
        "ImpliedVol": frames["ImpliedVol"].reset_index(),
        "LocalVol": frames["LocalVol"].reset_index(),
        "Expiries": shown.rows.drop(columns=["target"], errors="ignore"),
    }
    used = _used_quotes(shown)
    if len(used):
        sheets["Quotes"] = used
    return exports.xlsx(sheets, provenance)


def _downloads(shown: Shown) -> None:
    prov = shown.provenance
    frames = _grid_frames(shown.surface)
    row = st.container(horizontal=True, gap="small", key="surface_exports")
    row.download_button(
        "Implied vol grid (CSV)",
        exports.builder(exports.csv, frames["ImpliedVol"], prov, index=True),
        prov.file_name("implied_vol_surface", "csv"),
        "text/csv",
        on_click="ignore",
        icon=":material/download:",
        key="dl_iv",
    )
    row.download_button(
        "Local vol grid (CSV)",
        exports.builder(exports.csv, frames["LocalVol"], prov, index=True),
        prov.file_name("local_vol_surface", "csv"),
        "text/csv",
        on_click="ignore",
        icon=":material/download:",
        key="dl_lv",
    )
    if shown.fit is not None or shown.surface.quotes is not None:
        row.download_button(
            "Used quotes (CSV)",
            exports.builder(exports.csv, _used_quotes(shown), prov),
            prov.file_name("used_quotes", "csv"),
            "text/csv",
            on_click="ignore",
            icon=":material/download:",
            key="dl_quotes",
        )
    if shown.surface.svi:
        row.download_button(
            "SVI parameters (JSON)",
            exports.builder(_svi_json, shown, prov),
            prov.file_name("svi_parameters", "json"),
            "application/json",
            on_click="ignore",
            icon=":material/download:",
            key="dl_svi",
        )
    if shown.source == CHAIN:
        context_quotes = shown.fit.quotes if shown.fit is not None else None
        if context_quotes is not None:
            chain = context_quotes[["symbol", "root", "expiry", "settlement", "type", "strike", "bid", "ask"]]
            row.download_button(
                "The chain (CSV)",
                exports.builder(exports.csv, chain, prov),
                prov.file_name(f"{mk.INDEX_ROOT.lower()}_chain", "csv"),
                "text/csv",
                on_click="ignore",
                icon=":material/download:",
                key="chain_csv",
                help="The synthetic chain this surface is fitted to, in the app's long layout (an upload reads it).",
            )
    if any(importlib.util.find_spec(engine) for engine in ("xlsxwriter", "openpyxl")):
        # Built only when clicked: a workbook holding a large quote file takes seconds to write.
        row.download_button(
            "Everything (Excel)",
            exports.builder(_workbook, shown, prov),
            prov.file_name("volatility_surface", "xlsx"),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            on_click="ignore",
            icon=":material/download:",
            key="dl_xlsx",
        )


def render(inputs: Inputs) -> None:
    """The Surface tab (see the module docstring). Not a fragment: the expiry table and the ◂ ▸ buttons move the
    sidebar's focus expiry, which the whole page shows."""
    shown = _data_panel(inputs)
    _tiles(shown)
    if shown is None:
        return
    if shown.rows.empty:
        st.info("No expiry was fitted.")
        return
    focus, focus_key = _focus(shown)
    view, axis = _view_controls(shown, focus, focus_key)
    if view == "Smile":
        _smile(shown, focus, axis)
    elif view == "Quotes":
        _quotes_view(shown, focus)
    elif view == "Term":
        _term(shown)
    elif view == "Grid":
        _grid(shown)
    else:
        _three_d(shown)
    _expiry_table(shown, focus, focus_key)
    _diagnostics(shown, focus)
    _contract_readout(shown, inputs)
    _downloads(shown)
    st.caption(FOOTER)
