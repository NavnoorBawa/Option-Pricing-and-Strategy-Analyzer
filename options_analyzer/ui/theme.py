"""Desk design tokens: semantic colours for each light/dark mode and the chart colour ramps.

The ramps mirror ``chartCategoricalColors``, ``chartSequentialColors`` and ``chartDivergingColors`` in
``.streamlit/config.toml``; ``tests/test_theme.py`` checks that the two agree and that the colours pass the
colour-vision checks (separation under simulated protanopia and deuteranopia, and contrast with the background).

Colour never carries meaning alone: profit and loss also have a sign, and every series has a name in its legend.
"""

from __future__ import annotations

import streamlit as st

#: The two colour modes. The configured ``theme.base`` is dark, so dark is the fallback.
MODES = ("dark", "light")
DEFAULT_MODE = "dark"


def _ramp(colors: str) -> tuple[str, ...]:
    return tuple(colors.split())


#: ``chartCategoricalColors`` of each mode. Exactly ten (see the config): slot 9 is the mode's muted grey.
CATEGORICAL = {
    "dark": _ramp("#3987E5 #D95926 #199E70 #C98500 #D55181 #008300 #9085E9 #E66767 #8B95A7 #5B6577"),
    "light": _ramp("#2A78D6 #EB6834 #00A672 #CB8100 #DA6F98 #008300 #6250D6 #E34948 #5B6577 #8892A4"),
}
#: ``chartSequentialColors``: one blue hue from faint to strong (dark mode starts near the background).
SEQUENTIAL = {
    "dark": _ramp("#13243A #1C385C #244E81 #2C64A8 #357BD0 #4C92E8 #6DA6EE #8DBAF3 #ADCEF7 #CDE2FB"),
    "light": _ramp("#E4EEFB #BCD4F5 #94BAED #6CA0E5 #4286DB #2770C9 #2061B1 #195299 #134482 #0D366B"),
}
#: ``chartDivergingColors``: loss red to profit teal through the background colour.
DIVERGING = {
    "dark": _ramp("#B23E3B #8F3B3A #6D3638 #4C2F34 #2C2730 #1D2B34 #203C41 #204F50 #1E625E #17766D"),
    "light": _ramp("#D32F2F #DF655C #E88F87 #EEB7B2 #EFDEDE #D8E4E5 #ACCAC7 #80B1AA #51988E #0B7F73"),
}

#: Semantic roles for each mode:
#:
#: * ``profit`` / ``loss``: P&L fills and bars (TradingView's open-source pair in dark mode).
#: * ``call`` / ``put``: option identity, the first two categorical colours.
#: * ``accent``: highlighted marks (breakevens, forward, a third curve), the violet categorical colour.
#: * ``muted``: reference lines (spot, strike, zero) and their labels; ``rule``: grid and axis lines.
#: * ``text`` and ``surface``: the theme's text and background colours.
SEMANTIC = {
    "dark": {
        "profit": "#26A69A",
        "loss": "#EF5350",
        "call": CATEGORICAL["dark"][0],
        "put": CATEGORICAL["dark"][1],
        "accent": CATEGORICAL["dark"][6],
        "muted": CATEGORICAL["dark"][8],
        "rule": "#2A3342",
        "text": "#E6EAF2",
        "surface": "#0B0F16",
    },
    "light": {
        "profit": "#0B7F73",
        "loss": "#D32F2F",
        "call": CATEGORICAL["light"][0],
        "put": CATEGORICAL["light"][1],
        "accent": CATEGORICAL["light"][6],
        "muted": CATEGORICAL["light"][8],
        "rule": "#D5DBE3",
        "text": "#0E1726",
        "surface": "#FCFCFD",
    },
}

#: Opacity of the profit and loss areas under a P&L curve.
FILL_OPACITY = 0.18

#: Text and icons on a fill of the primary colour, in each mode. Streamlit draws them in white, which is 5.13:1 on
#: the light theme's dark amber but only 2.16:1 on the dark theme's amber, so dark mode takes its background colour
#: instead (8.90:1).
ON_PRIMARY = {"dark": SEMANTIC["dark"]["surface"], "light": "#FFFFFF"}

#: Where Streamlit (1.65 and 1.66) writes white on the primary colour in the widgets the app uses: the chips of a
#: multiselect, and the step buttons of a number input while hovered or focused. (Primary buttons do the same; the
#: app has none. Sliders and date and time inputs choose black or white for the primary colour themselves.)
ON_PRIMARY_SELECTORS = (
    "[data-testid='stMultiSelect'] [data-tag]",
    "[data-testid='stNumberInput'] :is([data-testid='stNumberInputStepUp'], [data-testid='stNumberInputStepDown'])"
    ":is(:hover, :focus):enabled",
)

#: Text on the primary colour. ``light-dark()`` follows the ``color-scheme`` that Streamlit sets on the app for the
#: theme it draws, so the rule switches with the theme at once, without waiting for a rerun.
ON_PRIMARY_RULE = (
    f"{', '.join(ON_PRIMARY_SELECTORS)} {{ color: light-dark({ON_PRIMARY['light']}, {ON_PRIMARY['dark']}); }}"
)

#: The name of the quotes file that the Surface tab keeps (its row has the key ``quotes_kept``) is a code span in a
#: caption, so that Markdown shows it as typed. Streamlit draws inline code at 0.75 em, which in a caption is 9.2 px
#: against the 12.25 px of the words around it: the one piece of the user's data in the row was the hardest to read.
KEPT_FILE_RULE = ".st-key-quotes_kept code { font-size: inherit; }"

#: The desk stylesheet (see :func:`apply_stylesheet`).
STYLESHEET = f"{ON_PRIMARY_RULE} {KEPT_FILE_RULE}"


def mode() -> str:
    """``"light"`` or ``"dark"``: the theme the viewer's browser reported for this script run.

    Outside a running app (tests, scripts) and while the browser has not reported a theme, this is the configured
    base, dark. Caveat: ``st.context.theme.type`` is what the browser reported when the script run started. The
    first run of a session already has it (checked in Chromium on Streamlit 1.65 and 1.66), but switching the
    theme in the app menu restyles the page without a rerun, so colours set from this mode keep the old mode until
    the next interaction. Colours that only tell series apart are therefore left to the theme's categorical colours
    (by trace order), which the browser swaps at once, and this mode is used for colours with a meaning, such as
    profit and loss.
    """
    if not st.runtime.exists():
        return DEFAULT_MODE
    kind = st.context.theme.type
    return kind if kind in MODES else DEFAULT_MODE


def palette(kind: str | None = None) -> dict[str, str]:
    """The semantic colours of mode ``kind`` (default: the current :func:`mode`)."""
    return SEMANTIC[kind or mode()]


def categorical(colors: dict[str, str] | None = None) -> tuple[str, ...]:
    """The categorical chart colours of the mode whose semantic colours are ``colors`` (default, or for colours of
    neither mode: the current mode)."""
    for kind in MODES:
        if colors == SEMANTIC[kind]:
            return CATEGORICAL[kind]
    return CATEGORICAL[mode()]


def apply_stylesheet() -> None:
    """Add :data:`STYLESHEET` to the page. Call it on every run; Streamlit sends HTML that holds only a stylesheet
    to the page without giving it any room in the layout."""
    st.html(f"<style>{STYLESHEET}</style>")
