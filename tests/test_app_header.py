"""The desk shell (ui/header.py and ui/app.py): the header strip, the page settings, the logo and the UI's text."""

from __future__ import annotations

import datetime as dt
import math
import re
import xml.etree.ElementTree as ET

import pytest
import streamlit as st
from streamlit.errors import StreamlitAPIException
from streamlit.proto.WidgetStates_pb2 import WidgetState
from streamlit.testing.v1 import AppTest

from options_analyzer.ui import header
from options_analyzer.ui.app import TABS
from options_analyzer.ui.inputs import Inputs
from options_analyzer.ui.model_config import ModelConfig

NBSP, NB_HYPHEN = "\xa0", "‑"


def _ok(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def _nodes(node):
    for child in getattr(node, "children", {}).values():
        yield child
        yield from _nodes(child)


def _key(node) -> str:
    """The user key of a keyed block ("" for other nodes)."""
    block_id = str(getattr(getattr(node, "proto", None), "id", "") or "")
    return block_id.rsplit("-", 1)[-1] if block_id.startswith("$$ID-") else ""


def _block(at: AppTest, key: str):
    return next(node for node in _nodes(at._tree) if _key(node) == key)


def _identity(at: AppTest) -> str:
    """The identity block's text with plain spaces and hyphens."""
    [paragraph] = _block(at, "desk_identity").markdown
    return paragraph.value.replace(NBSP, " ").replace(NB_HYPHEN, "-")


def _badge(at: AppTest) -> str:
    [badge] = _block(at, "desk_actions").markdown
    return re.fullmatch(r":gray-badge\[(.*)\]", badge.value).group(1)


_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _expiry(days: int) -> str:
    """The expiry date ``days`` calendar days from today, as ``26-Oct-27``."""
    day = dt.date.today() + dt.timedelta(days=days)
    return f"{day.day:02d}-{_MONTHS[day.month - 1]}-{day.year % 100:02d}"


def _inputs(spot=100.0, strike=100.0, days=365, vol=0.2, rate=0.05, q=0.0, model=None) -> Inputs:
    return Inputs(spot, strike, days, vol, rate, q, model or ModelConfig())


# --- The header strip in the app -----------------------------------------------------------------------------------


def test_the_header_strip_replaces_the_title(app):
    # One row: the brand and the ticker, then the badge, the Share popover and the Methodology button. No title,
    # no feature list.
    _ok(app)
    assert not app.title  # the 44 px H1, and the feature list under it, are gone
    assert not any("multi-leg strategies" in c.value for c in app.caption)
    row = _block(app, "desk_header")
    assert [_key(child) for child in row.children.values()] == ["desk_identity", "desk_actions"]
    expected = f"S 100.00 · K 100.00 · {_expiry(365)} (365d) · σ 20.00% · r 5.00% · q 0.00% · F 105.13"
    assert _identity(app) == f"**Options Analytics** {expected}"
    actions = _block(app, "desk_actions")
    assert [getattr(c, "type", type(c).__name__) for c in actions.children.values()] == [
        "markdown", "popover", "button"
    ]  # fmt: skip
    assert _badge(app) == "BSM · closed form"
    assert [b.label for b in actions.button] == ["Methodology"]
    first = next(iter(app.main.children.values()))
    assert _key(first) == "desk_header"  # the first thing on the page


def test_the_ticker_updates_after_a_spot_change(app):
    app.sidebar.number_input(key="spot").set_value(105.0).run()
    _ok(app)
    forward = 105 * math.exp(0.05)  # 110.3813
    assert _identity(app).endswith(
        f"S 105.00 · K 100.00 · {_expiry(365)} (365d) · σ 20.00% · r 5.00% · q 0.00% · F {forward:,.2f}"
    )
    assert "F 110.38" in _identity(app)
    app.sidebar.number_input(key="days").set_value(30).run()
    app.sidebar.number_input(key="div").set_value(2.5).run()
    app.sidebar.number_input(key="rate").set_value(-1.0).run()
    _ok(app)
    forward = 105 * math.exp((-0.01 - 0.025) * 30 / 365)
    assert _identity(app).endswith(
        f"S 105.00 · K 100.00 · {_expiry(30)} (30d) · σ 20.00% · r −1.00% · q 2.50% · F {forward:,.2f}"
    )


@pytest.mark.parametrize(
    ("query", "badge"),
    [
        ({"model": "Binomial tree", "steps": 2000, "style": "American"}, "LR tree · 2,000 steps · American"),
        ({"model": "Binomial tree", "tree": "Cox-Ross-Rubinstein"}, "CRR tree · 501 steps · European"),
        ({"model": "Monte Carlo", "paths": "1,000,000"}, "Monte Carlo · 1,000,000 paths"),
    ],
)
def test_the_badge_names_the_pricing_model(new_app, query, badge):
    at = new_app(**query).run()
    _ok(at)
    assert _badge(at) == badge


def test_the_badge_names_the_model_of_the_values_on_screen(app):
    # A 10-step CRR tree at 30% rates and 0.5% volatility is not arbitrage-free: the cards fall back to
    # Black-Scholes-Merton, and so does the badge.
    for key, value in (("model", "binomial"), ("tree", "crr")):
        app.sidebar.radio(key=key).set_value(value).run()
    app.sidebar.number_input(key="vol").set_value(0.5).run()
    app.sidebar.number_input(key="rate").set_value(30.0).run()
    app.sidebar.slider(key="steps").set_value(10).run()
    assert not app.exception
    assert any("Showing Black-Scholes-Merton values instead" in w.value for w in app.warning)
    assert _badge(app) == "BSM · closed form"


def test_the_methodology_button_opens_the_methodology_tab(app):
    app.button(key="desk_methodology").click().run()
    _ok(app)
    open_tabs = [t.label for t in app.tabs if list(_nodes(t))]
    assert open_tabs[-1] == "Methodology" and "Pricing" not in open_tabs
    assert app.query_params.get("tab") == "Methodology"
    assert any(m.value.startswith("### Conventions") for m in app.markdown)


def test_the_share_popover_holds_the_link_to_this_view(new_app):
    # The link is written last in the run, so it holds every input bound to the URL, including the Strategy tab's
    # menus that a short link (?strategy=...) expands to. In a test there is no browser address: the popover shows
    # the query string to add to it.
    at = new_app(spot=105, tab="Risk", strategy="Iron Condor").run()
    _ok(at)
    [code] = _block(at, "desk_share").code
    assert code.value == "?spot=105&tab=Risk&strategy_category=Range-bound&strategy_Range-bound=Iron+Condor"
    captions = [c.value for c in _block(at, "desk_share").caption]
    assert captions == ["Add this to the app's address to open this view:", header.SHARE_NOTE]


def test_the_share_popover_at_the_default_view(app):
    assert not _block(app, "desk_share").code
    assert [c.value for c in _block(app, "desk_share").caption] == [
        "The app's address opens this view: every input is at its default.",
        header.SHARE_NOTE,
    ]


def test_the_page_settings(new_app, monkeypatch):
    calls = []
    real = st.set_page_config
    monkeypatch.setattr(st, "set_page_config", lambda **kwargs: calls.append(kwargs) or real(**kwargs))
    _ok(new_app().run())
    [config] = calls
    assert config["page_title"] == "Options Analytics"
    assert config["page_icon"] == ":material/candlestick_chart:"
    assert config["layout"] == "wide"
    version = __import__("options_analyzer").__version__
    assert config["menu_items"]["About"] == (
        f"Options Analytics v{version} (Option-Pricing-and-Strategy-Analyzer) by Navnoor Bawa. "
        "Educational tool, not investment advice."
    )


# --- No emoji -------------------------------------------------------------------------------------------------------

#: Code points drawn as emoji: pictographs, the dingbats and miscellaneous symbols, and the emoji presentation
#: selector.
_EMOJI = re.compile("[\U0001f000-\U0001faff☀-➿⬀-⯿⌚⌛⏩-⏺️]")
#: Emoji shortcodes in Markdown (``:chart_with_upwards_trend:``); Material icons (``:material/...:``) and coloured
#: text and badges (``:gray-badge[...]``) are not emoji.
_SHORTCODE = re.compile(r"(?<![\w/:]):[a-z][a-z0-9_+-]*:(?![\w/\[])")


def _strings(message) -> list[str]:
    """Every string field of a protobuf message, its nested messages included."""
    found = []
    for field, value in message.ListFields():
        repeated = getattr(field, "is_repeated", None)  # protobuf 5.28 and later; earlier, the field's label
        if repeated is None:
            repeated = field.label == field.LABEL_REPEATED
        values = value if repeated else [value]
        for item in values:
            if field.type == field.TYPE_STRING:
                found.append(item)
            elif field.type == field.TYPE_MESSAGE:
                found += _strings(item)
    return found


def _open(at: AppTest, tab: str) -> AppTest:
    """Open ``tab`` as a click on it does (see tests/test_app.py)."""
    container = next(node for node in _nodes(at._tree) if getattr(node, "type", None) == "tab_container")
    states = at._tree.get_widget_states()
    states.widgets.append(WidgetState(id=container.proto.tab_container.id, string_value=tab))
    if tab == TABS[0]:
        at.query_params.pop("tab", None)
    else:
        at.query_params["tab"] = tab
    at._run(states)
    _ok(at)
    return at


def test_no_emoji_in_the_ui(new_app, monkeypatch):
    # Every string the app sends to the browser, on every tab with each pricing model, and the page settings.
    calls = []
    real = st.set_page_config
    monkeypatch.setattr(st, "set_page_config", lambda **kwargs: calls.append(kwargs) or real(**kwargs))
    seen = 0
    for query in ({}, {"model": "Binomial tree", "style": "American"}, {"model": "Monte Carlo"}):
        at = new_app(**query).run()
        _ok(at)
        for tab in TABS:
            _open(at, tab)
            for node in _nodes(at._tree):
                for text in _strings(node.proto) if hasattr(node, "proto") and node.proto is not None else ():
                    seen += 1
                    assert not _EMOJI.search(text), (tab, text[:120])
                    assert not _SHORTCODE.search(text), (tab, text[:120])
    assert seen > 500
    for config in calls:
        for text in [config["page_title"], config["page_icon"], *config["menu_items"].values()]:
            assert not _EMOJI.search(text) and not _SHORTCODE.search(text), text


# --- The parts of the strip ----------------------------------------------------------------------------------------


def test_the_ticker_wraps_only_between_its_items():
    text = header.ticker(_inputs(), today=dt.date(2026, 10, 26))
    items = text.split(f"{NBSP}· ")
    assert len(items) == 7
    for item in items:  # no space or hyphen to break at inside an item
        assert " " not in item and "-" not in item, item
    assert text.replace(NBSP, " ").replace(NB_HYPHEN, "-") == (
        "S 100.00 · K 100.00 · 26-Oct-27 (365d) · σ 20.00% · r 5.00% · q 0.00% · F 105.13"
    )


@pytest.mark.parametrize(
    ("inputs", "spot", "forward"),
    [
        # An index level: separators and two decimals.
        (_inputs(spot=45123.5, strike=45000), "45,123.50", "47,437.03"),
        # A penny underlying with a high dividend yield and a negative rate for ten years: the forward,
        # 0.01·e^(−0.55·10) = 4.0868e-05, keeps four significant digits instead of reading 0.0000.
        (_inputs(spot=0.01, strike=0.01, days=3650, rate=-0.05, q=0.5), "0.0100", "0.00004087"),
        (_inputs(spot=0.5, strike=0.5), "0.5000", "0.5256"),
    ],
)
def test_the_ticker_levels(inputs, spot, forward):
    items = header.ticker(inputs, today=dt.date(2026, 10, 26)).replace(NBSP, " ").split(" · ")
    assert items[0] == f"S {spot}" and items[-1] == f"F {forward}"
    true_forward = inputs.spot * math.exp((inputs.rate - inputs.dividend_yield) * inputs.days / 365)
    assert float(forward.replace(",", "")) == pytest.approx(true_forward, rel=5e-4)


def test_the_ticker_rates_use_the_minus_sign():
    items = header.ticker(_inputs(rate=-0.0125, q=0.0), today=dt.date(2026, 10, 26)).replace(NBSP, " ").split(" · ")
    assert items[4] == "r −1.25%" and items[5] == "q 0.00%"
    assert header.ticker(_inputs(rate=-1e-6), today=dt.date(2026, 10, 26)).count("r\xa00.00%") == 1  # no −0.00%


@pytest.mark.parametrize(
    ("model", "badge"),
    [
        (ModelConfig(), "BSM · closed form"),
        (ModelConfig(kind="binomial"), "LR tree · 501 steps · European"),
        (ModelConfig(kind="binomial", tree_method="crr", steps=20, american=True), "CRR tree · 20 steps · American"),
        (ModelConfig(kind="monte_carlo", n_paths=25_000, antithetic=False), "Monte Carlo · 25,000 paths"),
    ],
)
def test_model_badge(model, badge):
    assert header.model_badge(model) == badge


@pytest.mark.parametrize(
    ("base", "params", "link"),
    [
        ("http://localhost:8501", {}, "http://localhost:8501/"),
        ("http://localhost:8501/", {"tab": ["Risk"]}, "http://localhost:8501/?tab=Risk"),
        # An app served under a path keeps it, without an added slash.
        ("https://example.org/desk", {"spot": ["105"]}, "https://example.org/desk?spot=105"),
        # Spaces, ampersands and a repeated parameter are encoded as a browser would send them back.
        (
            "https://x.streamlit.app",
            {"strategy_Range-bound": ["Iron Condor"], "note": ["P&L"], "k": ["1", "2"]},
            "https://x.streamlit.app/?strategy_Range-bound=Iron+Condor&note=P%26L&k=1&k=2",
        ),
        (None, {}, ""),
        (None, {"spot": ["105"], "tab": ["IV"]}, "?spot=105&tab=IV"),
    ],
)
def test_share_link(base, params, link):
    assert header.share_link(base, params) == link


# --- The logo -------------------------------------------------------------------------------------------------------


def _luminance(color: str) -> float:
    def linear(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (linear(int(color[i : i + 2], 16) / 255) for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a: str, b: str) -> float:
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_the_logo_is_a_simple_theme_safe_monogram():
    svg = ET.parse(header.LOGO).getroot()
    ns = "{http://www.w3.org/2000/svg}"
    assert svg.tag == f"{ns}svg" and svg.get("viewBox") == "0 0 32 32"
    tags = {element.tag.removeprefix(ns) for element in svg.iter()}
    assert tags <= {"svg", "title", "rect", "g", "circle", "path"}  # no text (an <img> has no web fonts), no script
    assert not any(k.endswith("href") for element in svg.iter() for k in element.attrib)
    tile = svg.find(f"{ns}rect").get("fill")
    glyph = svg.find(f"{ns}g").get("stroke")
    # The tile stands out from both sidebar backgrounds (3:1 for graphics), and the letters from the tile (4.5:1).
    for sidebar in ("#0E131B", "#F5F7FA"):
        assert _contrast(tile, sidebar) >= 3, sidebar
    assert _contrast(glyph, tile) >= 4.5
    assert header.LOGO.stat().st_size < 1000


def test_the_app_shows_the_logo(new_app, monkeypatch):
    calls = []
    monkeypatch.setattr(st, "logo", lambda *args, **kwargs: calls.append((args, kwargs)))
    _ok(new_app().run())
    assert calls == [((str(header.LOGO),), {"size": "medium"})]


def test_the_app_runs_without_the_logo_where_pillow_is_broken(monkeypatch):
    # Streamlit reads a logo with Pillow; a broken Pillow must not stop the app. Any other error still raises.
    def broken(*args, **kwargs):
        raise StreamlitAPIException("The image passed to st.logo is invalid") from ImportError("cannot import _imaging")

    monkeypatch.setattr(st, "logo", broken)
    header.logo()

    def invalid(*args, **kwargs):
        raise StreamlitAPIException("The image passed to st.logo is invalid") from ValueError("not an image")

    monkeypatch.setattr(st, "logo", invalid)
    with pytest.raises(StreamlitAPIException):
        header.logo()
