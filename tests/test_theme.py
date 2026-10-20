"""The desk theme: config.toml, the semantic colour tokens, the colour-vision checks and the self-hosted fonts."""

from __future__ import annotations

import ast
import hashlib
import itertools
import math
import re
from pathlib import Path
from types import SimpleNamespace

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


def test_the_entry_points_share_one_app_and_find_the_fonts_next_to_them():
    # Streamlit serves the static/ folder and reads .streamlit/config.toml next to the script it runs: main.py for
    # `streamlit run main.py` and Community Cloud, ui/streamlit_app.py for the installed options-analyzer command.
    for script in (ROOT / "main.py", UI / "streamlit_app.py"):
        tree = ast.parse(script.read_text(encoding="utf-8"))
        imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        assert any(n.module == "options_analyzer.ui.app" and [a.name for a in n.names] == ["run"] for n in imports)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "run"]
        assert len(calls) == 1, script
        assert (script.parent / "static" / "fonts" / "IBMPlexSans-Regular.woff2").is_file()
        assert (script.parent / ".streamlit" / "config.toml").read_bytes() == (
            ROOT / ".streamlit" / "config.toml"
        ).read_bytes()
    launcher = (UI / "launch.py").read_text(encoding="utf-8")
    assert 'with_name("streamlit_app.py")' in launcher


def test_both_static_folders_are_identical():
    root, package = (sorted(p.relative_to(f) for p in f.rglob("*") if p.is_file()) for f in STATIC_FOLDERS)
    fonts_only = [p for p in package if p.parts[0] == "fonts"]
    assert root == fonts_only  # the root copy holds the fonts only
    for path in root:
        assert (STATIC_FOLDERS[0] / path).read_bytes() == (STATIC_FOLDERS[1] / path).read_bytes(), path


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
