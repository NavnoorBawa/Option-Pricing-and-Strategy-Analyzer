"""The desk header strip: what is on screen, and the app's actions.

One row above the price cards, ``st.container(horizontal=True, key="desk_header")``, holds:

* the identity block: the "Options Analytics" brand and the contract ticker in IBM Plex Mono
  (:func:`ticker`, ``S 100.00 · K 100.00 · 26-Oct-27 (365d) · σ 20.00% · r 5.00% · q 0.00% · F 105.13``);
* one action group on the right: the pricing-model badge (:func:`model_badge`), a Share popover with the link to
  this exact view (:func:`share_link`) and a Methodology button.

How the row wraps is set in ``static/desk.css``. The ticker wraps between its items, never inside one, in lines
of balanced length. The action group stays beside it while the brand and the ticker fit in two lines there
(:func:`render` gives the stylesheet the width that takes, which depends on the ticker's text), and otherwise
moves, whole, under it on the right. On a phone the Share and Methodology buttons show their icons only, so that
the group stays on one line down to a 360 px screen. :func:`logo` draws the monogram of ``static/logo.svg``.
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
from .format import MINUS, expiry, level, pct
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
    "The link holds the market inputs, the pricing model and its settings, the open tab, and the Strategy tab's "
    "template, strike width, contracts, shares per contract and commission. Edited legs, uploaded quotes and the "
    "other settings of a tab stay in this session."
)
#: Short names of the binomial lattices for the badge.
_LATTICE = {"crr": "CRR", "lr": "LR"}
#: The ticker's items are joined by a no-break space, a middle dot and a space: a line can end after the dot only.
_SEPARATOR = "\xa0· "
#: The identity block's text in static/desk.css, in rem of the theme's 14 px root, where the action group sits
#: beside it (a header strip of 600 px and more, so the ticker is at its full 0.8125rem): every character of the
#: ticker in IBM Plex Mono is 0.6 em wide, but σ comes from Plex Sans Regular (0.583 em), as Plex Mono has no Greek.
#: The brand is Plex Sans SemiBold capitals at 0.75rem, 10.106 em of glyphs and 0.08 em of tracking after each of
#: its 17 characters, and has a 0.75rem margin. tests/test_app_header.py checks these advances against the fonts.
_TICKER_REM, _MONO_EM, _SIGMA_EM = 0.8125, 0.6, 0.583
_BRAND_REM = 0.75 * (10.106 + 0.08 * len(BRAND)) + 0.75
#: Room to spare on the identity block's width (half a ticker character), for browsers that round glyph widths.
_SPARE_REM = 0.25


def _rate(value: float) -> str:
    """A rate or yield in percent with two decimals, the minus sign U+2212 and no negative zero."""
    text = pct(value, 2)
    return text.replace("-", MINUS) if text.strip("-0.%") else text.lstrip("-")


def ticker(inputs: Inputs, *, today: dt.date | None = None) -> str:
    """The contract ticker: spot, strike, expiry date (days to expiry), volatility, rate, dividend yield and forward.

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


def _ticker_rem(text: str) -> float:
    """Width in rem of ``text`` in the ticker's font at its full size."""
    return _TICKER_REM * sum(_SIGMA_EM if char == "σ" else _MONO_EM for char in text)


def _identity_rem(text: str) -> float:
    """Width in rem that the identity block needs to show the brand and the ticker ``text`` (:func:`ticker`) in two
    lines at most: the narrowest of its two-line layouts, each with the first ``k`` items on the brand's line (after
    the space that follows the brand) and the rest on the second line. A line that ends at a separator keeps its
    middle dot; the space after the dot does not count."""
    items = text.split(_SEPARATOR)

    def lines(k: int) -> float:
        first = _BRAND_REM + (_ticker_rem(" " + _SEPARATOR.join(items[:k]) + _SEPARATOR.rstrip()) if k else 0.0)
        return max(first, _ticker_rem(_SEPARATOR.join(items[k:])))

    return min(map(lines, range(len(items)))) + _SPARE_REM


def model_badge(model: ModelConfig) -> str:
    """The pricing model in a few words, for the header's badge.

    Parameters
    ----------
    model : ModelConfig
        The model of the price cards.

    Returns
    -------
    str
        ``BSM · closed form``, ``LR tree · 2,000 steps · American`` or ``Monte Carlo · 1,000,000 paths``.

    Examples
    --------
    >>> model_badge(ModelConfig(kind="binomial", steps=2000, american=True))
    'LR tree · 2,000 steps · American'
    >>> model_badge(ModelConfig(kind="monte_carlo", n_paths=1_000_000))
    'Monte Carlo · 1,000,000 paths'
    """
    if model.kind == "binomial":
        lattice = _LATTICE.get(model.tree_method, METHODS[model.tree_method])
        return f"{lattice} tree · {model.steps:,} steps · {'American' if model.american else 'European'}"
    if model.kind == "monte_carlo":
        return f"Monte Carlo · {model.n_paths:,} paths"
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
    query = urlencode([(key, value) for key, values in params.items() for value in values])
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
        Called (with no arguments) when the Methodology button is clicked, before the next run.

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
    # beside the ticker only while the ticker needs two lines at most there.
    st.html(f"<style>.st-key-desk_header {{ --desk-identity-min: {_identity_rem(text):.2f}rem; }}</style>")
    row = st.container(horizontal=True, key="desk_header", vertical_alignment="center", gap="small")
    # One paragraph: the brand (bold, which the stylesheet sets in small capitals) and the ticker flow together,
    # so that a 390 px phone needs two lines for both.
    row.container(key="desk_identity").markdown(f"**{BRAND}** {text}")
    actions = row.container(horizontal=True, key="desk_actions", vertical_alignment="center", width="content")
    model = f"Pricing model of the price cards: {inputs.model.label}."
    actions.badge(model_badge(inputs.model), color="gray", help=model)
    share = actions.popover("Share", icon=":material/link:", key="desk_share", help="Link to this view")
    actions.button(
        "Methodology",
        key="desk_methodology",
        icon=":material/menu_book:",
        on_click=on_methodology,
        help="Models, conventions and limitations",
    )
    return share


def fill_share(popover: DeltaGenerator) -> None:
    """Write the link to the current view into the Share popover of :func:`render`.

    Call it after every widget bound to the URL has run: each writes its value into ``st.query_params`` as it runs
    (a value that Streamlit corrects, such as one beyond a widget's range, included). In a browser the link is the
    page's address (``st.context.url``) with those parameters; where the address is unknown (a test), the popover
    shows the query string to add to it.

    Parameters
    ----------
    popover : DeltaGenerator
        The Share popover that :func:`render` returned in this script run.

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
    link = share_link(st.context.url, {key: st.query_params.get_all(key) for key in st.query_params})
    with popover:
        if link.startswith("?"):
            st.caption("Add this to the app's address to open this view:")
        if link:
            st.code(link, language=None, wrap_lines=True)
        else:
            st.caption("The app's address opens this view: every input is at its default.")
        st.caption(SHARE_NOTE)
