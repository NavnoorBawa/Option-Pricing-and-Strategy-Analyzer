"""The desk header strip: what is on screen, and the app's actions.

One row above the price cards, ``st.container(horizontal=True, key="desk_header")``, holds:

* the identity block: the "Options Analytics" brand and the contract ticker in IBM Plex Mono
  (:func:`ticker`, ``S 100.00 · K 100.00 · 26-Oct-27 (365d) · σ 20.00% · r 5.00% · q 0.00% · F 105.13``);
* one action group on the right: the pricing-model badge (:func:`model_badge`), a Share popover with the link to
  this exact view (:func:`share_link`) and a Methodology button (or Alt+Shift+M), with which the app opens the
  Methodology in a drawer on the right, over the open tab (:func:`.tab_methodology.drawer`).

Under the synthetic index the ticker is market-level (:func:`index_ticker`: valuation time and index level), a
SYNTHETIC badge (and an OVERRIDE one while the vol is typed) leads the action group, and the contract is the strip's
last row (:func:`contract_line`).

How the row wraps is set in ``static/desk.css``. The ticker wraps between its items, never inside one, in lines
of balanced length. Where the ticker's two lines fit beside the brand, they start one under the other (a hanging
indent), and the action group stays beside them while there is room; otherwise the group moves, whole, under the
ticker on the right. :func:`render` gives the stylesheet the width that this two-line layout takes, which depends
on the ticker's text. On a phone, where a column beside the brand would need three lines, the ticker's second
line starts under the brand instead, and the Share and Methodology buttons show their icons only, so that the
group stays on one line down to a 360 px screen. :func:`logo` draws the monogram of ``static/logo.svg``.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from urllib.parse import urlencode, urlsplit, urlunsplit

import numpy as np
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit.errors import StreamlitAPIException

from ..binomial import METHODS
from ..market_data.synthetic import SYNTHETIC_LABEL
from . import market as mk
from .format import MINUS, expiry, level, pct
from .format import date as short_date
from .inputs import Inputs
from .model_config import ModelConfig

#: The monogram (Streamlit shows it at the top of the sidebar, and in the top-left corner while the sidebar is
#: closed).
LOGO = Path(__file__).with_name("static") / "logo.svg"
#: The brand, which the stylesheet sets in small capitals.
BRAND = "Options Analytics"
#: What the Share popover says the link holds: the inputs bound to the URL (``bind="query-params"``), which
#: tests/test_app_header.py checks against the parameters the app writes.
SHARE_NOTE = (
    "The link holds the market (the preset, and its valuation time or typed inputs), the contract, the pricing model "
    "and its settings, the open tab, the Strategy tab's template, strike width, contracts, shares per contract and "
    "commission, and the what-if shifts. Edited legs, uploaded quotes and the other settings of a tab stay in this "
    "session."
)
#: The keyboard shortcut of the Methodology button (Streamlit shows Option for Alt on a Mac).
METHODOLOGY_SHORTCUT = "Alt+Shift+M"
#: Short names of the binomial lattices for the badge.
_LATTICE = {"crr": "CRR", "lr": "LR"}
#: The ticker's items are joined by a no-break space, a middle dot and a space: a line can end after the dot only.
_SEPARATOR = "\xa0· "
#: The identity block's text in static/desk.css, in rem of the theme's 14 px root, where the ticker's lines start
#: one under the other, with the ticker at its full 0.8125rem (it has that size on a header strip of 406 px and
#: more; on a narrower one it is smaller and needs less room): every character of the ticker in IBM Plex Mono is
#: 0.6 em wide, but σ comes from Plex Sans Regular (0.583 em), as Plex Mono has no Greek. The brand is Plex Sans
#: SemiBold capitals at 0.75rem, 10.106 em of glyphs and 0.08 em of tracking after each of its 17 characters
#: (8.6rem), in a box 9.35rem wide, which leaves a 0.75rem gap before the space that precedes the ticker.
#: tests/test_app_header.py checks these advances against the fonts and the box against the stylesheet.
_TICKER_REM, _MONO_EM, _SIGMA_EM = 0.8125, 0.6, 0.583
_BRAND_REM = 9.35
#: Room to spare on the identity block's width (half a ticker character), for browsers that round glyph widths.
_SPARE_REM = 0.25


def _rate(value: float) -> str:
    """A rate or yield in percent with two decimals, the minus sign U+2212 and no negative zero."""
    text = pct(value, 2)
    return text.replace("-", MINUS) if text.strip("-0.%") else text.lstrip("-")


def ticker(inputs: Inputs, *, today: dt.date | None = None) -> str:
    """The contract ticker: spot, strike, expiry date (days to expiry), volatility, rate, dividend yield and forward;
    under the index preset, the valuation time, the index level and the contract (:func:`index_ticker`).

    The items are separated by `` · ``, and no item can break: its spaces are no-break spaces (U+00A0, also before
    the separator) and the hyphens of its date non-breaking hyphens (U+2011). A narrow screen thus wraps the ticker
    between items only, with each separator at the end of a line.

    Parameters
    ----------
    inputs : Inputs
        The sidebar inputs.
    today : datetime.date, optional
        The valuation date that the days to expiry count from (default: the server's date, as everywhere in the
        app).

    Returns
    -------
    str
        Plain text (no Markdown formatting), for example
        ``S 100.00 · K 100.00 · 26-Oct-27 (365d) · σ 20.00% · r 5.00% · q 0.00% · F 105.13``.

    Examples
    --------
    >>> market = Inputs(5812.37, 5800.0, 30, 0.2, 0.05, 0.013, ModelConfig())
    >>> ticker(market, today=dt.date(2026, 10, 26)).replace("\\xa0", " ").replace("\\u2011", "-")
    'S 5,812.37 · K 5,800.00 · 25-Nov-26 (30d) · σ 20.00% · r 5.00% · q 1.30% · F 5,830.07'
    """
    if inputs.context is not None:
        return index_ticker(inputs)
    forward = inputs.spot * np.exp((inputs.rate - inputs.dividend_yield) * inputs.expiry)
    items = (
        f"S {level(inputs.spot)}",
        f"K {level(inputs.strike)}",
        expiry(inputs.days, today).replace("-", "\u2011"),
        f"σ {pct(inputs.volatility, 2)}",
        f"r {_rate(inputs.rate)}",
        f"q {_rate(inputs.dividend_yield)}",
        f"F {level(float(forward))}",
    )
    return _SEPARATOR.join(item.replace(" ", "\xa0") for item in items)


def index_ticker(inputs: Inputs) -> str:
    """The index preset's ticker, which is market-level: the valuation time and the index level. The contract
    (expiry, strike, vol, forward) is per expiry and has its own line above the price cards (:func:`contract_line`).

    Examples
    --------
    >>> market = mk.index_market(dt.datetime(2026, 10, 23, 16, 0))
    >>> inputs = Inputs(market.spot, 5815.0, 28.0, 0.15, 0.04, 0.01, ModelConfig(), mk.INDEX, market)
    >>> index_ticker(inputs).replace("\\xa0", " ").replace("\\u2011", "-")
    '23-Oct-26 16:00 ET · SPX 5,800.00'
    """
    context = inputs.context
    assert context is not None  # the index preset
    asof = context.as_of
    items = (
        f"{short_date(asof.date())} {asof:%H:%M} ET".replace("-", "\u2011"),
        f"{mk.INDEX_ROOT} {level(inputs.spot)}",
    )
    return _SEPARATOR.join(item.replace(" ", "\xa0") for item in items)


def contract_line(inputs: Inputs) -> str:
    """The index preset's contract, the header strip's last row: the expiry with its settlement and time to it, the
    strike, the vol (with an asterisk when it is typed over the surface's, beside the OVERRIDE badge) and the
    expiry's parity forward, in about 50 characters, one line on a 390 px phone (the sidebar gives the discount
    factor and the carry). Items are joined like the ticker's and do not break.

    Examples
    --------
    >>> market = mk.index_market(dt.datetime(2026, 10, 23, 16, 0))
    >>> row = market.expiry_row(market.default_expiry())
    >>> r, q = mk.carry(row["forward"], row["discount_factor"], market.spot, row["maturity"])
    >>> inputs = Inputs(market.spot, 5815.0, row["maturity"] * 365, 0.1561, r, q, ModelConfig(), mk.INDEX, market,
    ...                 row["expiry"], row["settlement"], row["maturity"], 0.1561)
    >>> contract_line(inputs).replace("\\xa0", " ").replace("\\u2011", "-")
    'SPX 20-Nov-26 AM 5,815 · 28d · σ 15.61% · F 5,813.91'
    """
    expiry_date = inputs.expiry_date
    assert expiry_date is not None and inputs.surface_vol is not None  # the index preset
    strike = f"{inputs.strike:,.0f}" if float(inputs.strike).is_integer() else level(inputs.strike)
    items = (
        f"{mk.INDEX_ROOT} {short_date(expiry_date)} {inputs.settlement} {strike}".replace("-", "\u2011"),
        mk.span(inputs.days),
        f"σ {pct(inputs.volatility, 2)}" + ("*" if inputs.vol_overridden else ""),
        f"F {level(inputs.forward)}",
    )
    return _SEPARATOR.join(item.replace(" ", "\xa0") for item in items)


def _ticker_rem(text: str) -> float:
    """Width in rem of ``text`` in the ticker's font at its full size."""
    return _TICKER_REM * sum(_SIGMA_EM if char == "σ" else _MONO_EM for char in text)


def _identity_rem(text: str) -> float:
    """Width in rem that the identity block needs to show the brand and, beside it, the ticker ``text``
    (:func:`ticker`) in two lines at most that start one under the other: the brand's box, the space after it and
    the narrowest of the ticker's layouts, each with its first ``k`` items on the first line and the rest (if any)
    on the second. A line that ends at a separator keeps its middle dot; the space after the dot does not count."""
    items = text.split(_SEPARATOR)

    def column(k: int) -> float:
        first = _SEPARATOR.join(items[:k]) + (_SEPARATOR.rstrip() if k < len(items) else "")
        return max(_ticker_rem(first), _ticker_rem(_SEPARATOR.join(items[k:])))

    return _BRAND_REM + _ticker_rem(" ") + min(map(column, range(1, len(items) + 1))) + _SPARE_REM


def model_badge(model: ModelConfig) -> str:
    """The pricing model in a few words, for the header's badge.

    Parameters
    ----------
    model : ModelConfig
        The model of the price cards.

    Returns
    -------
    str
        ``BSM · closed form``, ``LR tree · 2,000 steps · American``, ``MC · 1,000,000 paths · AV+CV`` or
        ``ALO · accurate · American``: a Monte Carlo simulation names the variance reductions in use, which its
        standard error includes: antithetic variates (AV) and a control variate (CV); the American engine of
        Andersen, Lake and Offengenden (ALO) names its scheme. The badge's tooltip spells the model out.

    Examples
    --------
    >>> model_badge(ModelConfig(kind="binomial", steps=2000, american=True))
    'LR tree · 2,000 steps · American'
    >>> model_badge(ModelConfig(kind="monte_carlo", n_paths=1_000_000))
    'MC · 1,000,000 paths · AV+CV'
    >>> model_badge(ModelConfig(kind="monte_carlo", antithetic=False, control_variate=False))
    'MC · 100,000 paths'
    >>> model_badge(ModelConfig(kind="alo", scheme="high"))
    'ALO · high · American'
    """
    if model.kind == "binomial":
        lattice = _LATTICE.get(model.tree_method, METHODS[model.tree_method])
        return f"{lattice} tree · {model.steps:,} steps · {'American' if model.american else 'European'}"
    if model.kind == "monte_carlo":
        reductions = "+".join(name for name, on in (("AV", model.antithetic), ("CV", model.control_variate)) if on)
        return f"MC · {model.n_paths:,} paths" + (f" · {reductions}" if reductions else "")
    if model.kind == "alo":
        return f"ALO · {model.scheme} · American"
    return "BSM · closed form"


def share_link(base: str | None, params: Mapping[str, Sequence[str]]) -> str:
    """The link to the current view: the app's address with the URL parameters of the inputs.

    Parameters
    ----------
    base : str or None
        The app's address without a query string, as ``st.context.url`` gives it, or None where it is unknown
        (outside a browser session).
    params : mapping of str to sequence of str
        The query parameters, each with its values (``st.query_params.get_all``).

    Returns
    -------
    str
        The address with the parameters as its query string. Without ``base``, the query string alone
        (``?spot=105.0``), or ``""`` when there are no parameters (the defaults).

    Examples
    --------
    >>> share_link("https://options.example.org", {"spot": ["105.0"], "model": ["Binomial tree"]})
    'https://options.example.org/?spot=105.0&model=Binomial+tree'
    >>> share_link(None, {"tab": ["Risk"]})
    '?tab=Risk'
    """
    # A colon is allowed in a query (RFC 3986), so the valuation time reads 2026-10-23T16:00, not 16%3A00.
    query = urlencode([(key, value) for key, values in params.items() for value in values], safe=":")
    if base is None:
        return f"?{query}" if query else ""
    parts = urlsplit(base)
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", query, ""))


def logo() -> None:
    """Show the monogram (:data:`LOGO`) with ``st.logo``: at the top of the sidebar, and in the top-left corner of
    the page while the sidebar is closed on a desktop.

    Streamlit reads every logo image with Pillow, one of its own dependencies. Where Pillow cannot be imported
    (a broken installation), Streamlit raises; the monogram is decoration, so the app then runs without it.

    Examples
    --------
    >>> from streamlit.testing.v1 import AppTest
    >>> def page():
    ...     from options_analyzer.ui import header
    ...     header.logo()
    >>> AppTest.from_function(page).run().exception
    ElementList()
    """
    try:
        st.logo(str(LOGO), size="medium")
    except StreamlitAPIException as exc:
        if not isinstance(exc.__cause__, ImportError):
            raise


def render(inputs: Inputs, *, on_methodology: Callable[[], None]) -> DeltaGenerator:
    """Draw the header strip and return its Share popover, for :func:`fill_share` at the end of the script run.

    Parameters
    ----------
    inputs : Inputs
        The inputs of the values on screen: the ticker shows them and the badge names their model.
    on_methodology : callable
        Called (with no arguments) when the Methodology button is clicked, before the next run: the app's
        callback asks that run to open the Methodology drawer (:func:`.tab_methodology.drawer`).

    Returns
    -------
    DeltaGenerator
        The Share popover, still empty.

    Examples
    --------
    >>> from streamlit.testing.v1 import AppTest
    >>> def page():
    ...     from options_analyzer.ui import header
    ...     from options_analyzer.ui.inputs import Inputs
    ...     from options_analyzer.ui.model_config import ModelConfig
    ...     market = Inputs(100.0, 100.0, 365, 0.2, 0.05, 0.0, ModelConfig(kind="binomial", steps=2000))
    ...     header.render(market, on_methodology=lambda: None)
    >>> at = AppTest.from_function(page).run()
    >>> at.markdown[0].value.split(" S")[0], at.markdown[1].value, at.button[0].label
    ('**Options Analytics**', ':gray-badge[LR tree · 2,000 steps · European]', 'Methodology')
    """
    text = ticker(inputs)
    # The width of the identity block's two-line layout, which static/desk.css reads to keep the action group
    # beside the ticker only while the ticker needs two lines at most there; from that width of the strip on, the
    # ticker's lines start one under the other (--desk-hang).
    width = f"{_identity_rem(text):.2f}rem"
    st.html(
        f"<style>.st-key-desk_header {{ --desk-identity-min: {width}; }} "
        f"@container desk-header (width >= {width}) {{ .st-key-desk_identity {{ --desk-hang: 1; }} }}</style>"
    )
    row = st.container(horizontal=True, key="desk_header", vertical_alignment="center", gap="small")
    # One paragraph: the brand (bold, which the stylesheet sets in small capitals) and the ticker, so that on a
    # 390 px phone the ticker can flow under the brand and both take two lines.
    row.container(key="desk_identity").markdown(f"**{BRAND}** {text}")
    actions = row.container(horizontal=True, key="desk_actions", vertical_alignment="center", width="content")
    if inputs.context is not None:
        actions.badge(
            "SYNTHETIC",
            color="gray",
            help=f"{SYNTHETIC_LABEL}: an SPX-like chain priced by the app's generator at the valuation time, with "
            "bid/ask spreads and rounding, then read like real quotes.",
        )
    if inputs.vol_overridden:
        assert inputs.surface_vol is not None
        gap = (inputs.volatility - inputs.surface_vol) * 100
        actions.badge(
            "OVERRIDE",
            color="yellow",  # amber: user state (UI spec 2.6); orange is the put colour
            help=f"σ {pct(inputs.volatility, 2)} typed over the surface's {pct(inputs.surface_vol, 2)} "
            f"({gap:+.2f} vp).".replace("-", MINUS),
        )
    model = f"Pricing model of the price cards: {inputs.model.label}."
    actions.badge(model_badge(inputs.model), color="gray", help=model)
    share = actions.popover("Share", icon=":material/link:", key="desk_share", help="Link to this view")
    actions.button(
        "Methodology",
        key="desk_methodology",
        icon=":material/menu_book:",
        on_click=on_methodology,
        help=f"Models, conventions, validation and references, beside this tab. Shortcut: {METHODOLOGY_SHORTCUT} "
        "(Option+Shift+M on a Mac).",
        shortcut=METHODOLOGY_SHORTCUT,
    )
    if inputs.context is not None:  # the index preset's ticker is market-level: the contract is the strip's last row
        row.container(key="contract_line").markdown(contract_line(inputs))
    return share


def fill_share(popover: DeltaGenerator, *, pinned: Mapping[str, str] | None = None) -> None:
    """Write the link to the current view into the Share popover of :func:`render`.

    Call it after every widget bound to the URL has run: each writes its value into ``st.query_params`` as it runs
    (a value that Streamlit corrects, such as one beyond a widget's range, included). In a browser the link is the
    page's address (``st.context.url``) with those parameters; where the address is unknown (a test), the popover
    shows the query string to add to it.

    Parameters
    ----------
    popover : DeltaGenerator
        The Share popover that :func:`render` returned in this script run.
    pinned : mapping of str to str, optional
        Parameters the link holds even where the URL leaves them out at their default: the valuation time, whose
        default (the last close) moves with the clock, so that the link reopens the same numbers on another day.

    Examples
    --------
    >>> from streamlit.testing.v1 import AppTest
    >>> def page():
    ...     import streamlit as st
    ...     from options_analyzer.ui import header
    ...     header.fill_share(st.popover("Share"))
    >>> at = AppTest.from_function(page)
    >>> at.query_params["spot"] = "105"
    >>> at.run().code[0].value
    '?spot=105'
    """
    params = {key: st.query_params.get_all(key) for key in st.query_params}
    for key, value in (pinned or {}).items():
        params.setdefault(key, [value])
    link = share_link(st.context.url, params)
    with popover:
        if link.startswith("?"):
            st.caption("Add this to the app's address to open this view:")
        if link:
            st.code(link, language=None, wrap_lines=True)
        else:
            st.caption("The app's address opens this view: every input is at its default.")
        st.caption(SHARE_NOTE)
