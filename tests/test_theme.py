"""The desk theme: config.toml, the semantic colour tokens, the colour-vision checks and the self-hosted fonts."""

from __future__ import annotations

import ast
import base64
import hashlib
import itertools
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from options_analyzer.ui import charts, theme

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: Streamlit 1.65 depends on the toml package
    import toml as tomllib

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "options_analyzer" / "ui"
CONFIG = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
THEME = CONFIG["theme"]
#: The static folders Streamlit serves at app/static/: next to main.py (``streamlit run main.py``, Streamlit
#: Community Cloud) and next to options_analyzer/ui/streamlit_app.py (the ``options-analyzer`` command).
STATIC_FOLDERS = (ROOT / "static", UI / "static")

#: SHA-256 of the fonts as published in IBM's npm packages @ibm/plex-sans 1.1.0 and @ibm/plex-mono 2.5.0
#: (fonts/complete/woff2, the complete character sets).
FONTS = {
    "IBMPlexSans-Regular.woff2": "ba711a3085ff9f27440b6b9c4550cfc47c97bf36591d5da958b975bb3add8c1a",
    "IBMPlexSans-Medium.woff2": "5660f8a658f8bb50dbc005232f885eadffd2bc1c235c4f6fbb63469d1f9cde6d",
    "IBMPlexSans-SemiBold.woff2": "f78048030eab62e860efa39a0df79e2e5581bf122eb95b9bc42c0b8a4988d205",
    "IBMPlexSans-Bold.woff2": "fa7130d854a660b39a7fc9e6e0f2dc23dba5f1346e2adea3e1fe37b6d884133d",
    "IBMPlexMono-Regular.woff2": "ba204497f16b6d334cee9d1e963a831b73e3a56e1d6300a8489d18df7214b350",
    "IBMPlexMono-Medium.woff2": "33faf307fa6031fb4062276d7320a6d632de890cbb347576fd80cfa01077bc25",
}

#: Code-point ranges of the IBM Plex Sans character map (all four weights share it; checked against the fonts by
#: test_the_cmap_snapshot_matches_the_fonts when fontTools and brotli are installed).
PLEX_SANS_CMAP = (
    (0, 0), (13, 13), (32, 126), (160, 383), (399, 399), (402, 402), (416, 417), (431, 432), (461, 476),
    (506, 511), (536, 539), (567, 567), (601, 601), (699, 700), (710, 711), (728, 733), (768, 772), (774, 780),
    (786, 786), (789, 789), (795, 795), (803, 803), (806, 808), (894, 894), (900, 906), (908, 908), (910, 929),
    (931, 974), (1024, 1119), (1122, 1123), (1130, 1131), (1138, 1141), (1168, 1218), (1231, 1241), (1244, 1257),
    (1262, 1273), (1316, 1317), (3647, 3647), (7808, 7813), (7838, 7838), (7840, 7929), (8192, 8205), (8208, 8213),
    (8216, 8218), (8220, 8222), (8224, 8226), (8230, 8230), (8232, 8233), (8239, 8240), (8242, 8243), (8249, 8250),
    (8260, 8260), (8304, 8304), (8308, 8313), (8320, 8329), (8353, 8353), (8356, 8356), (8358, 8358), (8360, 8366),
    (8369, 8370), (8372, 8373), (8376, 8378), (8381, 8381), (8383, 8383), (8467, 8467), (8470, 8470), (8482, 8482),
    (8486, 8486), (8494, 8494), (8528, 8529), (8531, 8542), (8592, 8601), (8617, 8618), (8624, 8627), (8630, 8631),
    (8634, 8635), (8644, 8644), (8646, 8646), (8706, 8706), (8710, 8710), (8719, 8719), (8721, 8722), (8725, 8725),
    (8729, 8730), (8734, 8734), (8747, 8747), (8758, 8758), (8776, 8776), (8800, 8800), (8804, 8805), (9674, 9674),
    (10003, 10003), (10060, 10060), (11022, 11025), (60640, 60640), (61388, 61388), (63191, 63192), (64257, 64258),
    (65279, 65279), (65533, 65533),
)  # fmt: skip


# --- Colour-vision checks (OKLab distance, Machado 2009 dichromacy simulation, WCAG contrast) ------------------------

#: Machado, Oliveira & Fernandes (2009) dichromacy simulation at severity 1.0, on linear sRGB.
MACHADO = {
    "protan": ((0.152286, 1.052583, -0.204868), (0.114503, 0.786281, 0.099216), (-0.003882, -0.048116, 1.051998)),
    "deutan": ((0.367322, 0.860646, -0.227968), (0.280085, 0.672501, 0.047413), (-0.011820, 0.042940, 0.968881)),
}
#: OKLCH lightness band of a categorical colour in each mode, and the chroma below which a hue reads as grey.
BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10


def _linear(color: str) -> tuple[float, ...]:
    channels = (int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)


def _oklab(rgb) -> tuple[float, float, float]:
    r, g, b = rgb
    cbrt = [
        math.copysign(abs(v) ** (1 / 3), v)
        for v in (
            0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b,
            0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b,
            0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b,
        )
    ]
    l_, m, s = cbrt
    return (
        0.2104542553 * l_ + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l_ - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l_ + 0.7827717662 * m - 0.8086757660 * s,
    )


def _simulate(color: str, kind: str) -> tuple[float, ...]:
    rgb = _linear(color)
    return tuple(min(1.0, max(0.0, sum(m * c for m, c in zip(row, rgb, strict=True)))) for row in MACHADO[kind])


def delta_e(a: str, b: str, kind: str | None = None) -> float:
    """OKLab distance x 100, under normal vision or a simulated dichromacy."""
    pa, pb = (_oklab(_simulate(c, kind) if kind else _linear(c)) for c in (a, b))
    return 100 * math.dist(pa, pb)


def cvd_delta_e(a: str, b: str) -> float:
    return min(delta_e(a, b, kind) for kind in MACHADO)


def luminance(color: str) -> float:
    r, g, b = _linear(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def lightness_chroma(color: str) -> tuple[float, float]:
    L, a, b = _oklab(_linear(color))
    return L, math.hypot(a, b)


def test_the_colour_checks_reproduce_the_reference_validator():
    # Reference figures from an independent JavaScript implementation of the same checks, for the same colours.
    assert cvd_delta_e("#26A69A", "#EF5350") == pytest.approx(11.6, abs=0.05)
    assert delta_e("#26A69A", "#EF5350") == pytest.approx(29.6, abs=0.05)
    assert cvd_delta_e("#0B7F73", "#D32F2F") == pytest.approx(12.1, abs=0.05)
    assert cvd_delta_e("#3987E5", "#D95926") == pytest.approx(26.8, abs=0.05)
    assert cvd_delta_e("#2A78D6", "#EB6834") == pytest.approx(24.7, abs=0.05)
    assert cvd_delta_e("#22C55E", "#EF4444") == pytest.approx(7.4, abs=0.05)  # the old pair failed the target
    assert contrast("#E8A33D", "#FFFFFF") == pytest.approx(2.16, abs=0.01)  # why there are no primary buttons


# --- config.toml -----------------------------------------------------------------------------------------------


def _mode(kind: str) -> dict:
    return THEME[kind]


def test_both_modes_have_exactly_ten_colours_in_every_chart_ramp():
    # With fewer than ten categorical colours Streamlit 1.64-1.66 leaves Plotly's colour placeholders unreplaced and
    # every chart falls back to Plotly's own colours; the diverging and sequential ramps have ten stops too.
    for kind in theme.MODES:
        for key in ("chartCategoricalColors", "chartDivergingColors", "chartSequentialColors"):
            colors = _mode(kind)[key]
            assert len(colors) == 10, (kind, key)
            assert all(re.fullmatch(r"#[0-9A-F]{6}", c) for c in colors), (kind, key)


def test_the_tokens_mirror_the_config():
    for kind in theme.MODES:
        cfg, tokens = _mode(kind), theme.SEMANTIC[kind]
        assert tuple(cfg["chartCategoricalColors"]) == theme.CATEGORICAL[kind]
        assert tuple(cfg["chartSequentialColors"]) == theme.SEQUENTIAL[kind]
        assert tuple(cfg["chartDivergingColors"]) == theme.DIVERGING[kind]
        assert (tokens["profit"], tokens["loss"]) == (cfg["greenColor"], cfg["redColor"])
        assert (tokens["text"], tokens["surface"]) == (cfg["textColor"], cfg["backgroundColor"])
        # Identity colours are the first categorical colours, so a trace without a colour gets the same one.
        assert (tokens["call"], tokens["put"]) == tuple(cfg["chartCategoricalColors"][:2])
        assert tokens["muted"] == cfg["chartCategoricalColors"][8]
        assert (cfg["blueColor"], cfg["orangeColor"]) == (tokens["call"], tokens["put"])
        # The diverging ramp runs from the loss side (red) to the profit side (teal): its poles are nearest the
        # loss and profit colours.
        poles = cfg["chartDivergingColors"][0], cfg["chartDivergingColors"][-1]
        assert delta_e(poles[0], tokens["loss"]) < delta_e(poles[0], tokens["profit"])
        assert delta_e(poles[1], tokens["profit"]) < delta_e(poles[1], tokens["loss"])


def test_desk_theme_settings():
    assert THEME["base"] == "dark"
    assert THEME["baseFontSize"] == 14
    assert THEME["headingFontSizes"] == ["1.5rem", "1.125rem", "1rem", "0.875rem", "0.8125rem", "0.75rem"]
    assert THEME["headingFontWeights"] == [600] * 6
    assert (THEME["metricValueFontSize"], THEME["metricValueFontWeight"]) == ("1.625rem", 500)
    assert THEME["baseRadius"] == "0.25rem" and THEME["showWidgetBorder"] and THEME["showSidebarBorder"]
    assert CONFIG["client"]["toolbarMode"] == "viewer"  # no Deploy button
    assert CONFIG["server"]["enableStaticServing"] is True
    expected = {
        "dark": ("#E8A33D", "#0B0F16", "#141A24", "#E6EAF2", "#232C3A", "#0E131B"),
        "light": ("#9E5F00", "#FCFCFD", "#F1F4F8", "#0E1726", "#D5DBE3", "#F5F7FA"),
    }
    for kind, (primary, background, secondary, text, border, sidebar) in expected.items():
        cfg = _mode(kind)
        assert (cfg["primaryColor"], cfg["backgroundColor"], cfg["secondaryBackgroundColor"]) == (
            primary,
            background,
            secondary,
        )
        assert (cfg["textColor"], cfg["borderColor"], cfg["sidebar"]["backgroundColor"]) == (text, border, sidebar)
        assert contrast(text, background) >= 15  # body text
        # The selected tab's label, radio buttons and sliders are drawn in the primary colour.
        assert min(contrast(primary, bg) for bg in (background, secondary, sidebar)) >= 4.5
    assert (_mode("dark")["greenColor"], _mode("dark")["redColor"]) == ("#26A69A", "#EF5350")
    assert (_mode("light")["greenColor"], _mode("light")["redColor"]) == ("#0B7F73", "#D32F2F")


def test_every_option_is_known_to_the_installed_streamlit():
    # Options that need a newer Streamlit only log "not a valid config option"; keep the file clean for the floor.
    from streamlit import config

    known = set(config._config_options_template)

    def names(section, prefix=""):
        for key, value in section.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                yield from names(value, name)
            else:
                yield name

    unknown = [n for n in names(CONFIG) if n not in known]
    assert not unknown
    assert "paddingTop" not in THEME and "paddingBottom" not in THEME  # Streamlit 1.66 and later only


def test_no_primary_buttons():
    # Streamlit writes white text on the primary colour, only 2.16:1 on the amber of the dark theme.
    pattern = re.compile(r"""type\s*=\s*["']primary["']""")
    offenders = [p.name for p in (ROOT / "options_analyzer").rglob("*.py") if pattern.search(p.read_text("utf-8"))]
    assert not offenders


#: Widgets that Streamlit (1.65, 1.66) fills with the primary colour and writes white on, with the test id the
#: stylesheet covers them by: multiselect chips, a number input's step buttons while hovered or focused, and the
#: camera input's button while pressed (not covered: the app has none). Primary buttons: test_no_primary_buttons.
WHITE_ON_PRIMARY = {"multiselect": "stMultiSelect", "number_input": "stNumberInput", "camera_input": None}


def test_text_on_the_primary_colour_is_legible():
    # Streamlit also writes white on the primary colour in multiselect chips and in a number input's step buttons
    # while they are hovered: the dark Surface tab showed its maturities ('29d', '1.07y') at 2.16:1.
    for kind in theme.MODES:
        assert contrast(theme.ON_PRIMARY[kind], _mode(kind)["primaryColor"]) >= 4.5, kind
    assert theme.ON_PRIMARY["dark"] == _mode("dark")["backgroundColor"]
    assert theme.ON_PRIMARY["light"] == "#FFFFFF"  # Streamlit's own colour, kept where it is legible
    rule = re.fullmatch(r"(.+) \{ color: light-dark\((#[0-9A-F]{6}), (#[0-9A-F]{6})\); \}", theme.ON_PRIMARY_RULE)
    assert theme.ON_PRIMARY_RULE in theme.STYLESHEET
    assert rule and rule[1] == ", ".join(theme.ON_PRIMARY_SELECTORS)
    assert (rule[2], rule[3]) == (theme.ON_PRIMARY["light"], theme.ON_PRIMARY["dark"])
    # Every widget the app uses that Streamlit fills with the primary colour and writes white on is covered.
    source = "\n".join(p.read_text("utf-8") for p in (ROOT / "options_analyzer").rglob("*.py"))
    for widget, testid in WHITE_ON_PRIMARY.items():
        if f".{widget}(" in source:
            assert testid and any(testid in selector for selector in theme.ON_PRIMARY_SELECTORS), widget


def test_apply_stylesheet_sends_style_only_html(monkeypatch):
    sent = []
    monkeypatch.setattr(theme.st, "html", sent.append)
    theme.apply_stylesheet()
    assert sent == [f"<style>{theme.STYLESHEET}</style>"]


# --- Palettes pass the colour-vision checks -----------------------------------------------------------------------


@pytest.mark.parametrize("kind", theme.MODES)
def test_categorical_colours_pass_the_colour_vision_checks(kind):
    colors, surface = theme.CATEGORICAL[kind], theme.SEMANTIC[kind]["surface"]
    for a, b in itertools.pairwise(colors):
        assert cvd_delta_e(a, b) >= 8, (a, b)  # neighbours stay apart under protanopia and deuteranopia
        assert delta_e(a, b) >= 15, (a, b)  # and for full colour vision
    for color in colors:
        assert contrast(color, surface) >= 3, color  # every line stands out from the background
    # The eight hues sit in the mode's lightness band with enough chroma to read as a hue; slots 9 and 10 are the
    # neutral greys (muted lines and labels), exempt from the chroma floor by design.
    lo, hi = BAND[kind]
    for color in colors[:8]:
        lightness, chroma = lightness_chroma(color)
        assert lo <= lightness <= hi and chroma >= CHROMA_FLOOR, color


@pytest.mark.parametrize("kind", theme.MODES)
@pytest.mark.parametrize(("first", "second", "cvd"), [("profit", "loss", 11.5), ("call", "put", 24.5)])
def test_token_pairs_pass_the_colour_vision_checks(kind, first, second, cvd):
    tokens = theme.SEMANTIC[kind]
    a, b = tokens[first], tokens[second]
    assert cvd_delta_e(a, b) >= cvd  # dark profit/loss 11.6, light 12.1; call/put 26.8 and 24.7
    assert delta_e(a, b) >= 15
    for color in (a, b):
        assert contrast(color, tokens["surface"]) >= 3


@pytest.mark.parametrize("kind", theme.MODES)
def test_reference_colours_are_legible(kind):
    tokens = theme.SEMANTIC[kind]
    assert contrast(tokens["muted"], tokens["surface"]) >= 4.5  # reference-line labels are 11 px text
    assert contrast(tokens["accent"], tokens["surface"]) >= 3
    assert contrast(tokens["rule"], tokens["surface"]) < 1.6  # grid lines stay in the background


@pytest.mark.parametrize("kind", theme.MODES)
def test_ramps_change_lightness_monotonically(kind):
    def lightness(colors):
        return [lightness_chroma(c)[0] for c in colors]

    sequential = lightness(theme.SEQUENTIAL[kind])
    steps = [b - a for a, b in itertools.pairwise(sequential)]
    assert all(s > 0 for s in steps) if kind == "dark" else all(s < 0 for s in steps)
    # Each arm of the diverging ramp moves away from the background colour towards its pole.
    diverging = lightness(theme.DIVERGING[kind])
    surface = lightness([theme.SEMANTIC[kind]["surface"]])[0]
    distance = [abs(v - surface) for v in diverging]
    assert all(a > b for a, b in itertools.pairwise(distance[:5]))
    assert all(a < b for a, b in itertools.pairwise(distance[5:]))


def test_dark_heatmap_cells_keep_light_labels_legible():
    # Plotly writes white labels on dark cells; the dark ramp's poles are capped so that every cell keeps 4.5:1.
    assert min(contrast("#FFFFFF", c) for c in theme.DIVERGING["dark"]) >= 4.5
    assert min(contrast(theme.SEMANTIC["dark"]["text"], c) for c in theme.DIVERGING["dark"]) >= 4.5


def test_scenario_colours_are_tokens():
    for kind in theme.MODES:
        tokens = theme.SEMANTIC[kind]
        colors = {c for c, _ in charts.scenario_styles(5, tokens)}
        assert colors == {tokens["put"], tokens["muted"], tokens["call"]}


# --- Current mode -----------------------------------------------------------------------------------------------


def test_mode_outside_a_running_app_is_the_configured_base():
    assert theme.mode() == "dark" == THEME["base"]
    assert theme.palette() is theme.SEMANTIC["dark"]


@pytest.mark.parametrize(
    ("reported", "expected"), [("light", "light"), ("dark", "dark"), (None, "dark"), ("x", "dark")]
)
def test_mode_follows_the_browser_theme(monkeypatch, reported, expected):
    monkeypatch.setattr(theme.st.runtime, "exists", lambda: True)
    monkeypatch.setattr(theme.st, "context", SimpleNamespace(theme=SimpleNamespace(type=reported)))
    assert theme.mode() == expected
    assert theme.palette() is theme.SEMANTIC[expected]
    assert theme.categorical() == theme.CATEGORICAL[expected]


def test_categorical_colours_of_a_palette():
    assert theme.categorical(theme.SEMANTIC["light"]) == theme.CATEGORICAL["light"]
    assert theme.categorical(theme.SEMANTIC["dark"]) == theme.CATEGORICAL["dark"]
    assert theme.categorical({"surface": "#123456"}) == theme.CATEGORICAL["dark"]  # neither mode: the current one


# --- Self-hosted fonts --------------------------------------------------------------------------------------------


def test_font_faces_are_served_from_both_static_folders():
    faces = {(f["family"], f["weight"]): f["url"] for f in THEME["fontFaces"]}
    assert set(faces) == {("IBM Plex Sans", w) for w in (400, 500, 600, 700)} | {
        ("IBM Plex Mono", w) for w in (400, 500)
    }
    for url in faces.values():
        prefix, name = url.rsplit("/", 1)
        assert prefix == "app/static/fonts" and name in FONTS
    assert THEME["font"].startswith("'IBM Plex Sans'") and THEME["headingFont"].startswith("'IBM Plex Sans'")
    # Plex Mono has no Greek letters, so it is only the code font, never the font of labels such as "Δ".
    assert THEME["codeFont"].startswith("'IBM Plex Mono'")


@pytest.mark.parametrize("folder", STATIC_FOLDERS, ids=["repository root", "package"])
def test_static_folders_hold_ibm_plex_and_its_licence(folder):
    fonts = folder / "fonts"
    assert sorted(p.name for p in fonts.iterdir()) == sorted([*FONTS, "LICENSE.txt"])
    for name, digest in FONTS.items():
        assert hashlib.sha256((fonts / name).read_bytes()).hexdigest() == digest, name
    licence = (fonts / "LICENSE.txt").read_text(encoding="utf-8")
    assert 'Copyright © 2017 IBM Corp. with Reserved Font Name "Plex"' in licence
    assert "SIL OPEN FONT LICENSE Version 1.1" in licence
    assert sum((fonts / name).stat().st_size for name in FONTS) <= 400_000


def test_the_entry_points_share_one_script_and_find_the_fonts_next_to_them():
    # Streamlit serves the static/ folder and reads .streamlit/config.toml next to the script it runs: main.py for
    # `streamlit run main.py` and Community Cloud, ui/streamlit_app.py for the installed options-analyzer command.
    # main.py only calls streamlit_app.main, so both load the same stylesheet and app.
    shim = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
    imports = [(n.module, [a.name for a in n.names]) for n in ast.walk(shim) if isinstance(n, ast.ImportFrom)]
    assert imports == [("options_analyzer.ui.streamlit_app", ["main"])]
    assert [getattr(n.func, "id", None) for n in ast.walk(shim) if isinstance(n, ast.Call)] == ["main"]
    for script in (ROOT / "main.py", UI / "streamlit_app.py"):
        assert (script.parent / "static" / "fonts" / "IBMPlexSans-Regular.woff2").is_file()
        assert (script.parent / ".streamlit" / "config.toml").read_bytes() == (
            ROOT / ".streamlit" / "config.toml"
        ).read_bytes()
    launcher = (UI / "launch.py").read_text(encoding="utf-8")
    assert 'with_name("streamlit_app.py")' in launcher


def test_every_run_loads_the_stylesheet_before_the_app(monkeypatch):
    from options_analyzer.ui import streamlit_app

    calls = []
    monkeypatch.setattr(theme, "apply_stylesheet", lambda: calls.append("stylesheet"))
    monkeypatch.setattr(streamlit_app, "run", lambda: calls.append("app"))
    streamlit_app.main()
    assert calls == ["stylesheet", "app"]


def test_the_page_carries_the_stylesheet():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "main.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert at.get("html")[0].proto.body == f"<style>{theme.STYLESHEET}</style>"


def test_both_static_folders_are_identical():
    root, package = (sorted(p.relative_to(f) for p in f.rglob("*") if p.is_file()) for f in STATIC_FOLDERS)
    fonts_only = [p for p in package if p.parts[0] == "fonts"]
    assert root == fonts_only  # the root copy holds the fonts only
    for path in root:
        assert (STATIC_FOLDERS[0] / path).read_bytes() == (STATIC_FOLDERS[1] / path).read_bytes(), path


#: Labels of the headline cards (Pricing and IV tabs), whose widths ui/components.py estimates. (The
#: other card rows are checked as whole rows in tests/test_app_components.py.)
CARD_LABELS = (
    "Call value", "Put value", "Forward price", "Call ITM probability", "Exercise premium", "Tree error vs BSM",
    "Std. error", "Implied volatility", "Model check", "Vega at IV", "Time value",
)  # fmt: skip


def test_the_card_width_model_uses_the_theme_font_sizes():
    # The cards' sizes come from a model of their text in the theme's font: with values modelled at Streamlit's
    # default 36 px in Source Sans, the Plex values on a phone grew to 36 px and were cut to '$10.45…'.
    from options_analyzer.ui import components

    base = THEME["baseFontSize"]
    value_px = base * float(THEME["metricValueFontSize"].removesuffix("rem"))
    small_px = base * 0.875  # Streamlit's small text: the card labels and deltas
    assert (components._VALUE_PX, components._LABEL_PX) == (value_px, small_px) == (22.75, 12.25)
    assert THEME["metricValueFontWeight"] == 500  # the value widths are those of Plex Sans Medium


def test_the_card_width_model_matches_plex_sans():
    pytest.importorskip("brotli")  # WOFF2 decompression
    ttlib = pytest.importorskip("fontTools.ttLib")
    from options_analyzer.ui import components

    def advances(weight: str) -> dict[str, float]:
        font = ttlib.TTFont(STATIC_FOLDERS[1] / "fonts" / f"IBMPlexSans-{weight}.woff2")
        cmap, metrics, units = font.getBestCmap(), font["hmtx"], font["head"].unitsPerEm
        return {chr(cp): metrics[name][0] / units for cp, name in cmap.items() if cp < 0x2300}

    medium, regular = advances("Medium"), advances("Regular")
    # A character of an amount or a percentage, with the suffixes of the shortened amounts ("$46.62M", "-$4.14T",
    # "$1.2e+15"), is never modelled narrower than it is, and the model is exact to 0.002 em. ("M" fell back to the
    # 0.6 em of a digit, 0.214 em short.)
    for char in "0123456789$+−,. -%MBTe":
        model = components._CHAR_EM.get(char, components._DIGIT_EM)
        assert medium[char] <= model <= medium[char] + 0.002, char
    # The words and signs a card shows instead of an amount are not modelled narrower than they are either.
    for value in ("Unlimited", "No profit", "No loss", "Too large", "–", ">99.9%", "<0.1%", "30 × 41"):
        model = components._value_em(value) - components._VALUE_SPARE_EM
        assert model >= sum(medium[c] for c in value) - 1e-9, value
    # Labels: at most _LABEL_EM per character on average.
    for label in CARD_LABELS:
        assert sum(regular[c] for c in label) <= components._LABEL_EM * len(label), label


def _covered(code_point: int) -> bool:
    return any(lo <= code_point <= hi for lo, hi in PLEX_SANS_CMAP)


def _is_emoji(text: str, i: int) -> bool:
    """Emoji are drawn with the system's emoji font: pictographs, and any character shown as an emoji (U+FE0F)."""
    char = text[i]
    return ord(char) >= 0x1F000 or char == "️" or text[i + 1 : i + 2] == "️"


def test_plex_sans_has_every_character_the_app_writes():
    # Google's subsets of Plex miss ∂ √ ≤ ∞ ≈ (and the latin subset all Greek), which then fell back to another
    # font; the complete Plex Sans covers every character in the package except emoji.
    missing = set()
    for path in (ROOT / "options_analyzer").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for i, char in enumerate(text):
            if ord(char) > 127 and not _is_emoji(text, i) and not _covered(ord(char)):
                missing.add(f"U+{ord(char):04X} {char} in {path.name}")
    assert not missing


def test_the_cmap_snapshot_matches_the_fonts():
    pytest.importorskip("brotli")  # WOFF2 decompression
    ttlib = pytest.importorskip("fontTools.ttLib")
    snapshot = {cp for lo, hi in PLEX_SANS_CMAP for cp in range(lo, hi + 1)}
    fonts = STATIC_FOLDERS[1] / "fonts"
    for name in FONTS:
        cmap = set(ttlib.TTFont(fonts / name).getBestCmap())
        if "Sans" in name:
            assert cmap == snapshot, name
        else:
            assert not {ord(c) for c in "ΔΓΘσνρ"} & cmap  # Plex Mono has no Greek


# --- In Chromium --------------------------------------------------------------------------------------------------
# The theme as a browser draws it: the running app (main.py on a free local port) and a few charts drawn by
# plotly.js. These tests need Playwright and a Chromium it can launch (CHROMIUM_EXECUTABLE, Playwright's own, or any
# revision Playwright has downloaded); without them they are skipped. The tests that drive the app are marked e2e.


def _launch_chromium(playwright):
    """CHROMIUM_EXECUTABLE if set; else Playwright's own Chromium, or else the newest one it has downloaded."""
    if executable := os.environ.get("CHROMIUM_EXECUTABLE"):
        return playwright.chromium.launch(executable_path=executable)
    root = Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or Path.home() / ".cache" / "ms-playwright")
    patterns = ("chromium-*/chrome-linux*/chrome", "chromium-*/chrome-win*/chrome.exe",
                "chromium-*/chrome-mac*/Chromium.app/Contents/MacOS/Chromium")  # fmt: skip
    for executable in [None, *(str(p) for pattern in patterns for p in sorted(root.glob(pattern), reverse=True))]:
        try:
            return playwright.chromium.launch(executable_path=executable)
        except Exception:  # not installed, or not this Playwright's revision: try the next one
            continue
    return None


@pytest.fixture(scope="module")
def chromium():
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as playwright:
        browser = _launch_chromium(playwright)
        if browser is None:
            pytest.skip("no Chromium that Playwright can launch")
        yield browser
        browser.close()


@pytest.fixture(scope="module")
def app_url(chromium):
    """URL of the app (``streamlit run main.py``) on a free local port, stopped after the module's tests."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    options = {"server.port": port, "server.address": "127.0.0.1", "server.headless": "true",
               "server.fileWatcherType": "none", "browser.gatherUsageStats": "false"}  # fmt: skip
    command = [sys.executable, "-m", "streamlit", "run", str(ROOT / "main.py")]
    command += [arg for key, value in options.items() for arg in (f"--{key}", str(value))]
    server = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        deadline = time.monotonic() + 120
        while True:
            try:
                direct.open(f"{url}/_stcore/health", timeout=2).close()
                break
            except OSError:
                if server.poll() is not None or time.monotonic() > deadline:
                    pytest.fail("the app did not start")
                time.sleep(0.25)
        yield url
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


#: Streamlit marks its app view with data-test-script-state ("running" while a script run is under way).
_RUNNING = "document.querySelector('[data-test-script-state]')?.getAttribute('data-test-script-state') === 'running'"
#: An element of each tab that only the tab's own run draws (the browser holds the open tab's panel only, and the
#: app leaves the closed tabs empty but for the Strategy tab's controls): once it is on the page, a click on the tab
#: has run it. (The script-state mark does not do: a fast run can come and go between two frames.)
TAB_READY = {
    "Pricing": ".st-key-sensitivity_chart",
    "Strategy": ".st-key-strategy_cards",
    "Risk": ".st-key-var_cards",
    "Surface": ".st-key-surface_cards",
    "IV": ".st-key-iv_cards",
    "Methodology": "[data-testid=stMarkdown]",
}


def _settle(page) -> None:
    """Wait until the script run is over, the fonts have loaded and the charts have drawn."""
    page.wait_for_function(f"() => !({_RUNNING})", timeout=120_000)
    page.evaluate("document.fonts.ready.then(() => true)")
    page.wait_for_timeout(1000)


def _open(chromium, url: str, scheme: str, width: int):
    page = chromium.new_page(viewport={"width": width, "height": 900}, color_scheme=scheme)
    page.goto(url)
    page.wait_for_selector("[data-testid=stMetricValue]", timeout=120_000)
    _settle(page)
    return page


def _tab(page, name: str) -> None:
    """Open a tab by clicking it, and wait until the run that this starts (only the open tab runs) has drawn it."""
    page.get_by_role("tab", name=name, exact=True).click()
    page.locator(f"[role=tabpanel] {TAB_READY[name]}").first.wait_for(state="visible", timeout=120_000)
    _settle(page)


def _hex(css: str) -> str:
    """``#RRGGBB`` of a computed ``rgb(r, g, b)`` colour."""
    red, green, blue = (int(v) for v in re.findall(r"[\d.]+", css)[:3])
    return f"#{red:02X}{green:02X}{blue:02X}"


#: Card values (``stMetricValue``) cut off with an ellipsis, and the font size of every visible card value.
_CARD_VALUES = """() => {
  const values = [...document.querySelectorAll('[data-testid=stMetricValue]')]
    .filter(e => e.getBoundingClientRect().width > 0);
  const cut = values.filter(e => [e, ...e.querySelectorAll('*')]
    .some(t => t.scrollWidth > t.clientWidth && getComputedStyle(t).overflow !== 'visible'));
  return {
    cut: cut.map(e => `${e.innerText} @ ${getComputedStyle(e).fontSize}`),
    sizes: values.map(e => parseFloat(getComputedStyle(e).fontSize)),
    plex: document.fonts.check('500 22.75px "IBM Plex Sans"'),
  };
}"""


#: The tab of every card row (the price cards sit above the tabs).
CARD_ROW_TABS = {
    "price_cards": "Pricing",
    "strategy_cards": "Strategy",
    "greek_cards": "Strategy",
    "expectation_cards": "Strategy",
    "var_cards": "Risk",
    "surface_cards": "Surface",
    "iv_cards": "IV",
}


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize("width", [1440, 390, 360])
def test_no_card_value_is_cut_off(chromium, app_url, width):
    # IBM Plex digits are 0.6 em wide where the cards' model assumed Source Sans' 0.5 em, and on a phone the values
    # grew to 36 px: at 390 and 360 px the price cards read '$10.45…' and '$105.1…', the IV cards '20.00…'.
    page = _open(chromium, app_url, "dark", width)
    try:
        for tab in dict.fromkeys(CARD_ROW_TABS.values()):
            _tab(page, tab)
            values = page.evaluate(_CARD_VALUES)
            assert values["plex"], "IBM Plex Sans did not load"
            assert not values["cut"], (tab, values["cut"])
            assert values["sizes"] and max(values["sizes"]) <= 22.75, (tab, values["sizes"])
    finally:
        page.close()


#: Every card of a card row: its box, the top and the left and right edges of its value's text, the right edge and
#: the number of lines of its label's text, the edges of its content box (inside the border and padding), in px,
#: and its value's font size.
_CARD_ROW = """key => [...document.querySelectorAll(`.st-key-${key} [data-testid=stMetric]`)].map(card => {
  const box = card.getBoundingClientRect(), outer = getComputedStyle(card);
  const inner = getComputedStyle(card.firstElementChild);
  const glyphs = element => {
    const range = document.createRange();
    range.selectNodeContents(element);
    const rects = [...range.getClientRects()].filter(r => r.width > 0);
    return {top: Math.min(...rects.map(r => r.top)), left: Math.min(...rects.map(r => r.left)),
            right: Math.max(...rects.map(r => r.right)), lines: new Set(rects.map(r => Math.round(r.top))).size};
  };
  const valueBox = card.querySelector('[data-testid=stMetricValue]'), value = glyphs(valueBox);
  const label = glyphs(card.querySelector('[data-testid=stMetricLabel] [data-testid=stMarkdownContainer]'));
  const left = parseFloat(outer.borderLeftWidth) + parseFloat(outer.paddingLeft) + parseFloat(inner.paddingLeft);
  const right = parseFloat(outer.borderRightWidth) + parseFloat(outer.paddingRight) + parseFloat(inner.paddingRight);
  return {label: card.querySelector('[data-testid=stMetricLabel]').innerText.trim(), x: box.left, y: box.top,
          width: box.width, height: box.height, valueTop: value.top - box.top, valueLeft: value.left,
          valueRight: value.right, labelRight: label.right, labelLines: label.lines, contentLeft: box.left + left,
          contentRight: box.right - right, font: parseFloat(getComputedStyle(valueBox).fontSize)};
})"""


def _lines(cards: list[dict]) -> list[list[dict]]:
    """The cards of a row by line (cards whose tops are within 2 px), top to bottom."""
    lines: list[list[dict]] = []
    for card in sorted(cards, key=lambda c: (c["y"], c["x"])):
        if lines and abs(lines[-1][0]["y"] - card["y"]) < 2:
            lines[-1].append(card)
        else:
            lines.append([card])
    return lines


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize(
    ("width", "query", "three_up"),
    # Phones (Android and iPhone widths) and a laptop with the sidebar open (the strategy and Greek rows 2 + 2, the
    # three-card rows 3 across), and on a phone a 100,000-lot position, whose amounts are in millions ("$46.62M").
    [
        pytest.param(360, "", False, id="360"),
        pytest.param(390, "", False, id="390"),
        pytest.param(1024, "", True, id="1024"),
        pytest.param(390, "?contracts=100000", False, id="390-100000-lots"),
    ],
)
def test_card_rows_line_up(chromium, app_url, width, query, three_up):
    # The cards of each line of a row have one width, one height and one value baseline; a card is as wide on a
    # line of its own (2 + 1) as above it; and no value or label runs out of its card's content box. Every value is
    # at the theme's full size, and on a line of three cards no label takes more than two lines. The rows of three
    # go 2 + 1 on phones, and the Strategy summary and Greek rows wrap together with their cards lined up.
    # (The subgrid and stretch rules of ui/components.py depend on Streamlit's DOM: the bordered stMetric box, the
    # padded box inside it and the element container around it.)
    page = _open(chromium, app_url + query, "light", width)
    try:
        rows = {}
        for key, tab in CARD_ROW_TABS.items():
            if not page.locator(f".st-key-{key} [data-testid=stMetric]").first.is_visible():
                _tab(page, tab)
            rows[key] = page.evaluate(_CARD_ROW, key)
        for key, cards in rows.items():
            assert len(cards) in (3, 4), key
            for line in _lines(cards):
                for measure in ("width", "height", "valueTop"):
                    values = [card[measure] for card in line]
                    assert max(values) - min(values) <= 1, (key, measure, [(c["label"], c[measure]) for c in line])
                if len(line) >= 3:
                    assert max(card["labelLines"] for card in line) <= 2, (key, [c["label"] for c in line])
            widths = [card["width"] for card in cards]
            assert max(widths) - min(widths) <= 1, (key, widths)
            for card in cards:
                assert card["font"] == pytest.approx(22.75, abs=0.01), (key, card["label"], card["font"])
                assert card["valueLeft"] >= card["contentLeft"] - 0.5, (key, card["label"])
                assert max(card["valueRight"], card["labelRight"]) <= card["contentRight"] + 0.5, (key, card)
        for key in ("expectation_cards", "var_cards"):
            assert [len(line) for line in _lines(rows[key])] == ([3] if three_up else [2, 1]), key
        for key in ("strategy_cards", "greek_cards"):
            assert [len(line) for line in _lines(rows[key])] == [2, 2], key

        def boxes(key: str) -> list[list[tuple[int, int]]]:
            return [[(round(c["x"]), round(c["width"])) for c in line] for line in _lines(rows[key])]

        assert boxes("strategy_cards") == boxes("greek_cards")
    finally:
        page.close()


#: A chart that each tab draws (the key given to charts.show), and the number of charts it draws, at the default URL.
TAB_CHART = {"Pricing": ("sensitivity_chart", 3), "Strategy": ("payoff_chart", 1), "Risk": ("var_hist", 6),
             "Surface": ("surf_iv", 4), "IV": ("iv_curve", 1), "Methodology": (None, 0)}  # fmt: skip


def _query(page) -> dict[str, list[str]]:
    return urllib.parse.parse_qs(urllib.parse.urlsplit(page.url).query)


@pytest.mark.e2e
@pytest.mark.slow
def test_a_click_on_a_tab_runs_that_tab_and_puts_it_in_the_link(chromium, app_url):
    # Only the open tab runs, so the page holds that tab's charts and no other's; the link names the tab (none for
    # the first) and keeps the Strategy tab's position while another tab is open: it reopens the same view.
    from options_analyzer.ui.app import TABS

    page = _open(chromium, app_url, "light", 1440)
    try:
        for tab in (*TABS[1:], TABS[0]):
            _tab(page, tab)
            key, count = TAB_CHART[tab]
            assert page.locator("[data-testid=stPlotlyChart]").count() == count, tab
            if key is not None:  # Plotly draws a chart some time after the run (on a busy machine, seconds)
                page.locator(f".st-key-{key} [data-testid=stPlotlyChart]").wait_for(state="visible", timeout=60_000)
            assert _query(page).get("tab") == (None if tab == TABS[0] else [tab])
            assert not page.locator("[data-testid=stException]").count(), tab
        _tab(page, "Strategy")
        contracts = page.locator("[data-testid=stMain] input[aria-label=Contracts]")
        contracts.fill("3")
        contracts.press("Enter")
        _settle(page)
        _tab(page, "Risk")
        assert _query(page) == {"contracts": ["3"], "tab": ["Risk"]}
        analysed = "Long 1 × 100 Call, Short 1 × 110 Call · 3 × 100 multiplier"
        assert analysed in page.get_by_text("Analysing").inner_text()
        link = _open(chromium, page.url, "light", 1440)
        try:
            link.locator(".st-key-var_hist [data-testid=stPlotlyChart]").wait_for(state="visible", timeout=60_000)
            assert link.locator("[role=tab][aria-selected=true]").inner_text().strip() == "Risk"
            assert analysed in link.get_by_text("Analysing").inner_text()
        finally:
            link.close()
    finally:
        page.close()


@pytest.mark.e2e
@pytest.mark.slow
def test_a_hand_typed_link_opens_the_tab_and_template_it_names(chromium, app_url):
    # Streamlit reads a bound parameter verbatim and drops any other value; the app rewrites such a link before any
    # widget reads it, in one step: the address bar then shows the app's own link, one history entry later.
    plain = _open(chromium, f"{app_url}/?tab=Risk", "light", 1440)
    entries = plain.evaluate("history.length")
    plain.close()
    page = _open(chromium, f"{app_url}/?tab=risk&strategy=Iron+Condor", "light", 1440)
    try:
        page.locator(".st-key-var_hist [data-testid=stPlotlyChart]").wait_for(state="visible", timeout=120_000)
        assert page.locator("[role=tab][aria-selected=true]").inner_text().strip() == "Risk"
        condor = "Iron Condor — Long 1 × 80 Put, Short 1 × 90 Put, Short 1 × 110 Call, Long 1 × 120 Call"
        assert condor in page.get_by_text("Analysing").inner_text()
        assert _query(page) == {"tab": ["Risk"], "strategy_category": ["Range-bound"],
                                "strategy_Range-bound": ["Iron Condor"]}  # fmt: skip
        assert page.evaluate("history.length") == entries + 1
        assert not page.locator("[data-testid=stAlertContentWarning]").count()
    finally:
        page.close()
    page = _open(chromium, f"{app_url}/?tab=Risk&tab=IV&strategy=Iron+Condr", "light", 1440)
    try:
        assert page.locator("[role=tab][aria-selected=true]").inner_text().strip() == "IV"
        assert _query(page) == {"tab": ["IV"]}
        note = page.locator("[data-testid=stAlertContentWarning]")
        assert "a strategy named Iron Condr, which is not one of the templates" in note.inner_text()
        assert note.locator("code").inner_text() == "Iron Condr" and not note.locator("a").count()
    finally:
        page.close()


def _net_premium(page) -> str:
    card = page.locator("[data-testid=stMetric]").filter(has_text="Net premium")
    return card.locator("[data-testid=stMetricValue]").inner_text()


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize("tab", ["Pricing", "Risk"])
def test_a_leg_edit_is_kept_when_another_tab_is_clicked_before_enter(chromium, app_url, tab):
    # The legs editor commits a cell when it loses the focus: a value typed into a cell and left by clicking another
    # tab arrives with the run that opens that tab, where the editor is not drawn. Clicking Pricing lost it.
    page = _open(chromium, f"{app_url}/?tab=Strategy", "light", 1440)
    try:
        assert _net_premium(page) == "$441.05"  # long the 100 call, short the 110 call, for 100 shares
        grid = page.locator("[data-testid=stMain] [data-testid=stDataFrame]").first
        box = grid.bounding_box()
        # The grid draws on a canvas: click the first row's Type cell (the header and each row are 35 px high),
        # step to Quantity and type into it, which opens the cell's editor (an input over the grid) a moment later.
        page.mouse.click(box["x"] + 40, box["y"] + 35 + 17)
        page.keyboard.press("ArrowRight")
        page.keyboard.type("3")
        page.wait_for_function(
            "() => document.activeElement?.matches('input.gdg-input') && document.activeElement.value === '3'"
        )
        _tab(page, tab)
        if tab == "Risk":
            assert "Long 3 × 100 Call, Short 1 × 110 Call" in page.get_by_text("Analysing").inner_text()
        _tab(page, "Strategy")
        assert _net_premium(page) == "$2,531"  # 3 x $10.4506 - $6.0401, for 100 shares
    finally:
        page.close()


#: Font of the kept quotes file's name (a code span) and of the words of its caption, in the row under the uploader.
_KEPT_FILE = """() => {
  const row = document.querySelector('.st-key-quotes_kept');
  const code = getComputedStyle(row.querySelector('code')), words = getComputedStyle(row.querySelector('p'));
  return {name: row.querySelector('code').innerText, size: parseFloat(code.fontSize), family: code.fontFamily,
          wordSize: parseFloat(words.fontSize)};
}"""


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize("scheme", ["dark", "light"])
def test_the_kept_quotes_file_is_named_at_the_size_of_its_caption(chromium, app_url, tmp_path, scheme):
    # Streamlit draws inline code at 0.75 em: in the caption that names the kept file, the name was 9.2 px.
    from options_analyzer import vol_surface

    quotes = tmp_path / "quotes.csv"
    quotes.write_text(vol_surface.sample_quotes().to_csv(index=False), encoding="utf-8")
    page = _open(chromium, f"{app_url}/?tab=Surface", scheme, 1440)
    try:
        page.get_by_text("Market quotes (CSV)", exact=True).click()
        _settle(page)
        page.locator("[data-testid=stFileUploader] input[type=file]").set_input_files(quotes)
        fitted = page.locator("[data-testid=stMetric]").filter(has_text="Calendar arbitrage")  # a card of the fit
        fitted.wait_for(state="visible", timeout=120_000)
        _settle(page)
        _tab(page, "IV")
        _tab(page, "Surface")
        page.locator(".st-key-quotes_kept code").wait_for(state="visible", timeout=60_000)
        kept = page.evaluate(_KEPT_FILE)
        assert kept["name"] == "quotes.csv"
        assert kept["wordSize"] == pytest.approx(12.25, abs=0.01)  # a caption: 0.875 of the 14 px base size
        assert kept["size"] == kept["wordSize"] and kept["family"].startswith('"IBM Plex Mono"'), kept
    finally:
        page.close()


#: Colour and background of every visible multiselect chip.
_CHIPS = """() => [...document.querySelectorAll('[data-testid=stMultiSelect] [data-tag]')]
  .filter(e => e.getBoundingClientRect().width > 0)
  .map(e => [e.innerText, getComputedStyle(e).color, getComputedStyle(e).backgroundColor])"""


@pytest.mark.e2e
@pytest.mark.parametrize("scheme", ["dark", "light"])
def test_text_on_the_primary_colour_is_legible_in_the_app(chromium, app_url, scheme):
    page = _open(chromium, app_url, scheme, 1440)
    try:
        _tab(page, "Surface")
        chips = page.evaluate(_CHIPS)
        assert chips  # the maturities of the smile chart
        primary = _mode(scheme)["primaryColor"]
        for text, color, background in chips:
            assert _hex(background) == primary, text
            assert contrast(_hex(color), _hex(background)) >= 4.5, (text, color, background)
        _tab(page, "IV")
        step = page.locator("[data-testid=stMain] [data-testid=stNumberInputStepUp]:visible").first
        step.hover()
        page.wait_for_timeout(400)
        color, background = step.evaluate("e => [getComputedStyle(e).color, getComputedStyle(e).backgroundColor]")
        assert _hex(background) == primary
        assert contrast(_hex(color), _hex(background)) >= 4.5, (color, background)
    finally:
        page.close()


def _chart_figures() -> dict:
    """Charts with a legend under the plot, as the app draws them, with long legend entries."""
    rng = np.random.default_rng(7)
    days = np.arange(0, 366)
    moves = ("Spot −1σ (−18.1%)", "Spot −0.5σ (−9.5%)", "Spot unchanged", "Spot +0.5σ (+10.5%)", "Spot +1σ (+22.1%)")
    series = {name: (i - 2) * 150 + days * 0.5 for i, name in enumerate(moves)}
    spots = np.linspace(60.0, 170.0, 221)
    curves = [(np.clip(spots - 100, -440, 560), "At expiry (365d)", theme.SEMANTIC["dark"]["call"], "solid"),
              (spots - 100, "Today", theme.SEMANTIC["dark"]["put"], "dash"),
              (spots - 99, "In 183 days", theme.SEMANTIC["dark"]["accent"], "dot")]  # fmt: skip
    m = np.linspace(0.8, 1.2, 21)
    slices = [(label, 0.2 + (m - 1) ** 2, 0.21 + (m - 1) ** 2) for label in ("29d", "1.07y", "2.00y")]
    quotes = [(m[::4], 0.2 + (m[::4] - 1) ** 2, 0.2 + (m[::4] - 1) ** 2)] * 3
    terminal = 100 * np.exp(rng.normal(0.03, 0.2, 20_000))
    grid = np.linspace(40, 260, 200)
    return {
        "lines": charts.lines_chart(days, series, "Days from today", "P&L ($)", styles=charts.scenario_styles(5)),
        "histogram": charts.histogram_chart(
            terminal, "Underlying price at expiry", [(100.0, "strike", charts.NEUTRAL)], pdf=(grid, grid * 0 + 0.01)
        ),
        "payoff": charts.payoff_chart(spots, curves, 100.0, [100.0, 110.0], [100.0], (84.0, 126.0)),
        "smile": charts.smile_chart(m, slices, "Moneyness K/S", quotes=quotes),
    }


#: Draw a figure with plotly.js and return the boxes of its legend, its x-axis title and tick labels, and itself.
_DRAW = """async (spec) => {
  const div = document.getElementById('chart');
  Plotly.purge(div);
  await Plotly.newPlot(div, spec.data, spec.layout, {displayModeBar: false});
  const box = e => { const r = e.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; };
  return {legend: box(div.querySelector('g.legend rect.bg')), figure: box(div),
          axis: [...div.querySelectorAll('g.g-xtitle text, g.xtick text')].map(box).filter(b => b[2] > b[0])};
}"""


def _with_streamlit_theme(spec: dict) -> dict:
    """What Streamlit's theme adds to a chart (1.65 and 1.66, at the 14 px base): axes that grow the margins for
    their labels, 10.5 px text and tick labels, and 12.25 px axis titles 17.5 px (x) and 21 px (y) from them."""
    layout = spec["layout"]
    layout["font"] = {"family": "'IBM Plex Sans', sans-serif", "size": 10.5, **layout.get("font", {})}
    for axis, standoff in (("xaxis", 17.5), ("yaxis", 21.0)):
        layout.setdefault(axis, {})["automargin"] = True
        layout[axis].setdefault("tickfont", {})["size"] = 10.5
        title = layout[axis].setdefault("title", {})
        title.setdefault("font", {})["size"] = 12.25
        title["standoff"] = standoff
    return spec


@pytest.mark.parametrize("width", [360, 390, 1440])
def test_legends_under_the_plot_clear_the_axis_title(chromium, width):
    # At 360 px the charts were 360 px high and the legend sat 20% of the plot height under the plot: 'Days from
    # today' was drawn through the time-decay legend, and the Monte Carlo histogram's legend touched its axis title.
    import plotly

    plotly_js = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    plex = base64.b64encode((STATIC_FOLDERS[1] / "fonts" / "IBMPlexSans-Regular.woff2").read_bytes()).decode()
    page = chromium.new_page(viewport={"width": width, "height": 900})
    try:
        page.set_content(
            "<!doctype html><style>@font-face { font-family: 'IBM Plex Sans'; "
            f"src: url(data:font/woff2;base64,{plex}) format('woff2'); }} body {{ margin: 0 14px; }}</style>"
            "<div id='chart'></div>"
        )
        page.add_script_tag(path=str(plotly_js))
        page.evaluate("document.fonts.load('12.25px \"IBM Plex Sans\"')")
        for name, fig in _chart_figures().items():
            boxes = page.evaluate(_DRAW, _with_streamlit_theme(json.loads(fig.to_json())))
            legend, figure = boxes["legend"], boxes["figure"]
            assert boxes["axis"], name
            assert legend[1] >= max(b[3] for b in boxes["axis"]), name  # wholly below the axis title and ticks
            assert legend[3] <= figure[3] + 0.5, name  # and inside the chart
    finally:
        page.close()
