"""The shared metric-card rows (ui/components.py): their sizes, their structure and their use on every tab."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from _apptest import ASOF
from streamlit.testing.v1 import AppTest

from options_analyzer import vol_surface
from options_analyzer.ui import components, theme

FONTS = Path(components.__file__).with_name("static") / "fonts"
#: Width (px) of the row on the narrowest phone the cards are checked at: 360 px less the page's side margins.
PHONE_ROW = 360 - 2 * 14


def _ok(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def _key(node) -> str:
    """The user key of a keyed container ("" for other blocks)."""
    block_id = str(getattr(getattr(node, "proto", None), "id", "") or "")
    return block_id.rsplit("-", 1)[-1] if block_id.startswith("$$ID-") else ""


def _apps(at: AppTest | list[AppTest]) -> list[AppTest]:
    return at if isinstance(at, list) else [at]


def _card_rows(at: AppTest | list[AppTest]) -> dict[str, list[list]]:
    """Every metric of the page (or of the pages, one per tab), by card row: ``{row key: [[metrics of group 1],
    ...]}``. Fails on a metric that is not in a group of a keyed row (a plain ``st.metric`` or one in
    ``st.columns``)."""
    found: dict[str, list[list]] = {}
    for page in _apps(at):
        rows: dict[str, list[list]] = {}

        def walk(node, ancestors, rows=rows):
            for child in getattr(node, "children", {}).values():
                if getattr(child, "type", None) == "metric":
                    group, row = ancestors[-1], ancestors[-2]
                    assert _key(group) and _key(row), f"{child.label!r} is not in a card row"
                    assert all(a.type != "column" for a in ancestors), f"{child.label!r} is in st.columns"
                    groups = rows.setdefault(_key(row), [])
                    if not groups or groups[-1][0] != _key(group):
                        groups.append([_key(group), []])
                    groups[-1][1].append(child)
                else:
                    walk(child, [*ancestors, child])

        walk(page._tree, [])
        found |= {row: [metrics for _, metrics in groups] for row, groups in rows.items()}
    return found


def _styles(at: AppTest | list[AppTest]) -> list[str]:
    return [h.proto.body for page in _apps(at) for h in page.get("html")]


#: The tabs with card rows (only the open tab runs; the Pricing tab owns its headline tiles, UI-1).
CARD_TABS = ("Pricing", "Strategy", "Risk", "Surface", "IV")


def _app_states(new_app) -> dict[str, list[AppTest]]:
    """The app at its default URL (every tab's default cards), with market quotes on the surface tab, with a
    100,000-lot position (amounts in millions: "$46.62M") and with a short straddle at the input limits (the widest
    strategy values): for each, one page per tab with cards."""

    def pages(**query) -> list[AppTest]:
        return [new_app(tab=tab, **query).run() for tab in CARD_TABS]

    quotes = new_app(tab="Surface").run()
    quotes.segmented_control(key="surface_source").set_value("Upload").run()
    csv = vol_surface.sample_quotes().to_csv(index=False).encode()
    quotes.file_uploader(key="quotes_file").upload("quotes.csv", csv, "text/csv").run()
    limits = pages(
        spot=9999999,
        strike=9999999,
        contracts=100000,
        multiplier=10000,
        vol=500,
        days=1,
        strategy="Short Straddle",
    )
    default = pages()
    with_quotes = [quotes if tab == "Surface" else page for tab, page in zip(CARD_TABS, default, strict=True)]
    # The app's own default market, the synthetic index, whose Strategy tiles hold the widest default values (the
    # package market "20.80 / 23.90"); the other states are on the textbook market the tests pin.
    index = [new_app(tab="Strategy", preset=None, asof=ASOF).run()]
    states = {
        "default": default,
        "index": index,
        "quotes": with_quotes,
        "large": pages(contracts=100000),
        "limits": limits,
    }
    for page in (page for state in states.values() for page in state):
        _ok(page)
    return states


# --- the stylesheet ------------------------------------------------------------------------------------------


FOUR = [("Call value", "$10.4506"), ("Put value", "$5.5735"), ("Forward price", "$105.1271"), ("ITM", "55.96%")]
THREE = [("VaR 99.0%", "$1,234"), ("Expected Shortfall 99.0%", "$1,456"), ("Delta-normal VaR", "$1,111")]
GAP, CHROME = 14, 28  # px between cards; padding and border of a card


def _width(cards, value_px: float = 22.75, label_lines: int | None = 1) -> int:
    """The row's card width in the model: its widest card, rounded up to whole px."""
    return int(-(-max(components._card_width(*card[:2], value_px, label_lines, *card[2:]) for card in cards) // 1))


def _layout(style: str, row: float) -> tuple[int, float] | None:
    """What a browser makes of a card-row stylesheet on a row ``row`` px wide: the cards on each line of a group and
    their value size in px, from the container queries whose condition holds (the last one wins, as in CSS), or
    None at full size (no query applies)."""
    layout = None
    for block in style.split("@container ")[1:]:
        narrower = re.match(r"[\w-]+ \(width < (\d+)px\)", block)  # (a one-line row's ``width >=`` query sets no grid)
        if narrower and row < int(narrower.group(1)):
            columns = int(re.search(r"grid-template-columns: repeat\((\d+), minmax\(0, 1fr\)\)", block).group(1))
            share, padding, em = re.search(r"calc\(\(([\d.]+)cqw - ([\d.]+)px\) / ([\d.]+)\)", block).groups()
            font = min(max(16.0, (float(share) * row / 100 - float(padding)) / float(em)), 22.75)
            layout = columns, font
    return layout


def test_four_cards_in_two_pairs_wrap_by_pairs():
    style = components._cards_style(FOUR, "row", ("row_a", "row_b"))
    card, compact = _width(FOUR), _width(FOUR, 16, None)
    pair = 2 * card + GAP
    assert f".st-key-row > :has(> .st-key-row_a, > .st-key-row_b) {{ flex: 1 1 {pair}px !important; }}" in style
    assert f".st-key-row_a > *, .st-key-row_b > * {{ flex: 1 1 {card}px !important; }}" in style
    # Narrower than one pair at full size: each pair on its own line, as a grid of two equal columns, with values
    # sized for half the row less half the gap and the padding; one card per line once two compact cards no longer
    # fit, with values sized for the whole row.
    assert f"@container row (width < {pair}px) {{" in style
    assert ".st-key-row_a, .st-key-row_b { grid-template-columns: repeat(2, minmax(0, 1fr)); }" in style
    assert "calc((50cqw - 35px) / 5.19)" in style
    assert f"@container row (width < {2 * compact + GAP}px) {{" in style
    assert "calc((100cqw - 28px) / 5.19)" in style
    assert _layout(style, pair) is None
    assert _layout(style, pair - 1)[0] == _layout(style, 2 * compact + GAP)[0] == 2
    assert _layout(style, 2 * compact + GAP - 1)[0] == 1


def test_three_cards_go_two_and_one_with_the_third_card_as_wide_as_the_others():
    # In one flex group the third card wrapped onto a line of its own and stretched across the whole row, next to
    # two half-width cards, with its value still sized for a third of the row. The compact group is a grid: three
    # columns while every card has room for its value at full size and its label on two lines, then two (the third
    # card takes one column, as wide as the others) while two compact cards fit, then one; the values are sized for
    # the column width.
    style = components._cards_style(THREE, "var", ("var_all",))
    card, three_up, compact = _width(THREE), _width(THREE, 22.75, 2), _width(THREE, 16, None)
    group = 3 * card + 2 * GAP
    assert f".st-key-var > :has(> .st-key-var_all) {{ flex: 1 1 {group}px !important; }}" in style
    assert f".st-key-var_all > * {{ flex: 1 1 {card}px !important; }}" in style
    assert f"@container var (width < {group}px) {{" in style
    assert ".st-key-var_all { display: grid !important; row-gap: 0 !important; }" in style
    assert "grid-column" not in style  # every card takes one column
    em = max(components._value_em(value) for _, value in THREE)
    assert f"calc((33.3333cqw - 37.3333px) / {em:g})" in style  # a third of the row, less 2/3 of a gap and padding
    assert f"@container var (width < {3 * three_up + 2 * GAP}px) {{" in style
    # The cards after the first line keep the gap above them (the grid has no row gap, see the next test).
    assert ".st-key-var_all > :nth-child(n + 3) { margin-top: 14px; }" in style
    assert f"@container var (width < {2 * compact + GAP}px) {{" in style
    assert ".st-key-var_all > :nth-child(n + 2) { margin-top: 14px; }" in style
    assert _layout(style, group) is None
    # At every row width of a phone or a narrow window, each card is at least its compact width, and the widest
    # value at its clamped size fits the card; on a line of three, the values are at full size.
    for row in range(compact, group):
        per_line, font = _layout(style, row)
        width = (row - (per_line - 1) * GAP) / per_line
        assert per_line == (3 if row >= 3 * three_up + 2 * GAP else 2 if row >= 2 * compact + GAP else 1), row
        assert width >= compact and font * em <= width - CHROME + 1e-6, row
        if per_line == 3:
            assert width >= three_up and font == 22.75, row


def test_a_label_on_two_lines_breaks_at_the_best_space():
    # The longest line of the best two-line break: "EXPECTED" | "SHORTFALL 99.0%" (not "EXPECTED SHORTFALL" |
    # "99.0%"), "EXPECTED P&L" | "AT EXPIRY"; and one word per line, the longest word.
    chrome = components._HELP_ICON_PX + components._CARD_CHROME_PX
    for label, lines, longest in [
        ("Expected Shortfall 99.0%", 1, "Expected Shortfall 99.0%"),
        ("Expected Shortfall 99.0%", 2, "Shortfall 99.0%"),
        ("Expected Shortfall 99.0%", None, "Shortfall"),
        ("Expected P&L at expiry", 2, "Expected P&L"),
        ("Gamma", 2, "Gamma"),
        ("Gamma", None, "Gamma"),
    ]:
        width = components._card_width(label, "1", 16, lines)
        assert width == pytest.approx(components._label_px(longest) + chrome), (label, lines)


def test_labels_are_modelled_in_spaced_capitals():
    # The labels are capitals at 0.72rem with 0.06 em after each character (static/desk.css): "GAMMA" is 5 x (0.707
    # + 0.06) em plus the M's extra width, wider than "Gamma" in lower case at the old 12.25 px.
    em = sum(components._LABEL_CHAR_EM[c] + components._LABEL_TRACKING_EM for c in "GAMMA")
    assert components._label_px("Gamma") == pytest.approx(components._LABEL_PX * em)
    assert components._label_px("gamma") == components._label_px("GAMMA")
    # A character outside the table counts as the widest in it, "%".
    assert components._label_px("Σ") == pytest.approx(components._LABEL_PX * (0.927 + 0.06))


def test_the_cards_of_a_line_share_their_label_value_and_delta_rows():
    # Labels wrap to different numbers of lines on a narrow row, and each value followed its own label, so the
    # values of a line sat at different heights (35 and 46 px below the card top for "VaR 99.0%" and "Expected
    # Shortfall 99.0%" on a 390 px phone). Each card, and the two boxes inside it, is now a subgrid on three rows of
    # the group's grid (label, value, delta), so the values share a row; the labels keep to the top of theirs.
    style = components._cards_style(THREE, "var", ("var_all",))
    compact = style.split("@container ")[1]
    subgrid = "{ display: grid; grid-row: span 3; grid-template-rows: subgrid; row-gap: 0; }"
    assert (
        ".st-key-var_all > *, .st-key-var_all [data-testid='stMetric'], "
        f".st-key-var_all [data-testid='stMetric'] > div {subgrid}" in compact
    )
    assert ".st-key-var [data-testid='stMetricLabel'] { align-self: start; }" in compact
    assert "white-space: normal;" in compact
    # At full size the labels stay on one line, so no rule is needed there.
    assert "subgrid" not in style.split("@container ")[0]


def test_the_cards_of_a_line_are_as_tall_as_the_tallest():
    # At full size a card with a delta line ("debit (paid)") was 23 px taller than the cards next to it. The groups
    # now stretch to their line, the cards to their group and the bordered box to its card (by the desk
    # stylesheet, for every card row), at every size (in a compact row the subgrid makes them equal too).
    style = components._cards_style(FOUR, "row", ("row_a", "row_b"))
    full_size = style.split("@container ")[0]
    assert ".st-key-row, .st-key-row_a, .st-key-row_b { align-items: stretch; }" in full_size
    desk = re.sub(r"\s+", " ", theme.STYLESHEET.read_text(encoding="utf-8"))
    assert "[data-testid='stMetric'] { height: 100%; }" in desk


def test_rows_sized_with_each_other_get_the_same_sizes():
    # The Strategy summary wrapped 2 x 2 at 1,024 px with the sidebar open, while the Greeks row under it stayed 4
    # across: each row was sized from its own text only.
    greeks = [("Delta (shares)", "18.72"), ("Gamma", "-0.1026"), ("Theta / day", "-$0.14"), ("Vega / vol pt", "-$2.05")]
    summary = [("Net premium", "$441.05", "debit (paid)"), ("Max profit", "$558.95"), ("Max loss", "$441.05")]
    summary.append(("Probability of profit", "47.4%"))
    alone = components._cards_style(greeks, "greek_cards", ("greek_cards_a", "greek_cards_b"))
    paired = components._cards_style(greeks, "greek_cards", ("greek_cards_a", "greek_cards_b"), size_with=summary)
    other = components._cards_style(
        summary, "strategy_cards", ("strategy_cards_a", "strategy_cards_b"), size_with=greeks
    )
    assert paired != alone
    assert paired == other.replace("strategy_cards", "greek_cards").replace("strategy-cards", "greek-cards")
    assert f"flex: 1 1 {_width(summary + greeks)}px" in paired


@pytest.mark.parametrize(
    ("cards", "groups"),
    [([], ("a",)), ([("A", "1", {})] * 4, ()), ([("A", "1", {})] * 3, ("a", "b")), ([("A", "1", {})] * 5, ("a", "b"))],
)
def test_card_row_needs_cards_that_divide_into_the_groups(cards, groups):
    with pytest.raises(ValueError, match="equal groups"):
        components.card_row("row", groups, cards)


def test_card_row_draws_bordered_metrics_in_their_groups():
    def script():
        from options_analyzer.ui.components import card_row

        card_row(
            "demo",
            ("demo_all",),
            [
                ("Net premium", "$2.50", {"delta": "debit (paid)", "delta_color": "off", "help": "Paid up front."}),
                ("Grid", 12, {}),
                ("Drift used", "5.00%", {"help": None}),
            ],
        )

    at = AppTest.from_function(script).run()
    _ok(at)
    assert {row: [[m.label for m in g] for g in groups] for row, groups in _card_rows(at).items()} == {
        "demo": [["Net premium", "Grid", "Drift used"]]
    }
    assert [(m.label, m.value) for m in at.metric] == [
        ("Net premium", "$2.50"),
        ("Grid", "12"),
        ("Drift used", "5.00%"),
    ]
    assert all(m.proto.show_border for m in at.metric)
    assert at.metric[0].delta == "debit (paid)" and at.metric[0].proto.help == "Paid up front."
    assert any(".st-key-demo { container: demo / inline-size; }" in s for s in _styles(at))


# --- every tab ------------------------------------------------------------------------------------------------


#: Every card row of the app with the labels of each group, at the default URL (with market quotes for the
#: surface cards).
ROWS = {
    "pricing_tiles": [["Call", "Put"], ["Straddle", "Vol"], ["Forward", "Time"]],
    "strategy_tiles": [["Net premium", "Package"], ["Max profit", "Max loss"], ["Breakevens", "P(profit)"]],
    "var_cards": [["VaR 99.0%", "Expected Shortfall 99.0%"], ["Spot-only VaR 99.0%", "Spot-only ES 99.0%"]],
    "surface_tiles": [["ATMF 30D", "25Δ RR 30D"], ["25Δ BF 30D", "TERM 30→365D"], ["FIT vs ½‑spr (vp)", "ARB"]],
    "iv_tiles": [["IV bid", "IV ask"], ["IV", "Surface"], ["Vega", "Bounds"]],
}


@pytest.fixture(scope="module")
def states():
    from _apptest import make_app

    return _app_states(make_app)


def test_every_metric_of_every_tab_is_in_a_card_row(states):
    # The Strategy, Risk and Surface tabs drew their metrics in st.columns, which stack one card per row on a phone.
    found = {}
    for at in states.values():
        for row, groups in _card_rows(at).items():
            assert all(m.proto.show_border for group in groups for m in group), row
            assert any(f".st-key-{row} {{ container:" in s for s in _styles(at)), f"{row} has no stylesheet"
            found[row] = [[m.label for m in group] for group in groups]
    assert found == ROWS


def test_the_rows_of_pairs_stay_two_across_on_a_phone(states):
    # On a 360 px phone each pair of a row of pairs fits the row side by side (the cards do not stack): the four-card
    # rows and the Strategy tab's six tiles in three pairs (the Pricing and IV tiles have their own test below). At
    # the input limits the Strategy tiles show the breakevens of a straddle struck at 9,999,999 ("9,215,931 /
    # 10,784,067"): those stack there.
    checked = set()
    for state, at in states.items():
        for row, groups in _card_rows(at).items():
            if any(len(g) != 2 for g in groups) or row in ("pricing_tiles", "iv_tiles"):
                continue
            if state == "limits" and row == "strategy_tiles":
                continue
            style = next(s for s in _styles(at) if f".st-key-{row} {{ container:" in s)
            layout = _layout(style, PHONE_ROW)
            if layout is None:
                # No query applies: every pair is at full size, its two cards side by side (one flex line of two).
                layout = (2, None)
            assert layout[0] == 2, (state, row)
            checked.add(row)
    assert checked == {"strategy_tiles", "var_cards", "surface_tiles"}


@pytest.mark.parametrize("row", ["pricing_tiles", "iv_tiles"])
def test_the_six_tile_rows_go_pair_by_pair_on_a_phone(states, row):
    # UI spec 2.9: at most six tiles a row. In three pairs they wrap 2 + 2 + 2 on 360 to 440 px phones, each pair side
    # by side at the theme's full value size (no container query applies), where two groups of three went 2 + 1 twice.
    style = next(s for s in _styles(states["default"]) if f".st-key-{row} {{ container:" in s)
    for phone in (360, 390, 440):
        assert _layout(style, phone - 2 * 14) in (None, (2, 22.75)), phone


#: The index's Strategy tiles' value size (px) on the phones where it is below the theme's 22.75 px.
INDEX_TILES = {360: 20.0, 375: 21.14, 390: 22.28}


@pytest.mark.parametrize("phone", [360, 375, 390, 412, 430, 440])
def test_every_card_value_of_a_phone_screen_has_one_size(states, phone):
    # Every row of every tab at the default URL shows its values at the theme's full size on a phone, so the type
    # does not change from row to row on one screen (no query applies: full size). On a 320 px phone the price
    # cards' values shrink to 21.4 px so that each pair stays side by side. The exception is the index's Strategy
    # tiles below a 412 px phone: two values of 13 characters ("20.80 / 23.90", the package market, and "5,378 /
    # 6,172", the breakevens) set the width of every tile, and keep their pairs side by side at 20.0, 21.1 and 22.3 px
    # on 360, 375 and 390 px phones (spec 2.11 does not gate tile positions on phones).
    row = phone - 2 * 14
    for state in ("default", "index"):
        styles = _styles(states[state])
        for key, style in ((k, s) for s in styles for k in ROWS if f".st-key-{k} {{ container:" in s):
            size = (_layout(style, row) or (0, 22.75))[1]
            if state == "index" and key == "strategy_tiles" and phone in INDEX_TILES:
                assert size == pytest.approx(INDEX_TILES[phone], abs=0.05)
                continue
            assert size == 22.75, (state, key, _layout(style, row))


def _advances(weight: str) -> dict[str, float]:
    ttlib = pytest.importorskip("fontTools.ttLib")
    pytest.importorskip("brotli")  # WOFF2 decompression
    font = ttlib.TTFont(FONTS / f"IBMPlexSans-{weight}.woff2")
    cmap, metrics, units = font.getBestCmap(), font["hmtx"], font["head"].unitsPerEm
    return {chr(cp): metrics[name][0] / units for cp, name in cmap.items()}


def test_the_card_width_model_covers_every_card_row_in_plex_sans(states):
    # Every card of a row is based on the row's widest card in the model. It must not be narrower than the widest
    # card's real text (Plex Sans advance widths): the value in Medium at 22.75 px, the label in Regular capitals at
    # 10.08 px with its letter-spacing (plus its help icon) and the delta in Regular at 12.25 px (plus its pill's
    # padding), on one line or on two (a line of three cards on a phone), and the value at 16 px with the label one
    # word per line (a pair on a phone). And no value is modelled narrower than it is, as the compact rows size their
    # values from the model: "$46.62M" was modelled 0.11 em too narrow ("M" is 0.814 em wide), so on a 390 px phone
    # it ran 2.6 px into its card's padding.
    medium, regular = _advances("Medium"), _advances("Regular")
    tracking = components._LABEL_TRACKING_EM * components._LABEL_PX

    def text(chars: str, table: dict[str, float], px: float) -> float:
        return sum(table[c] for c in chars) * px

    def label(line: str) -> float:
        """A label line in small capitals: each character followed by its letter-spacing."""
        return text(line.upper(), regular, components._LABEL_PX) + tracking * len(line)

    def delta(line: str) -> float:
        return text(line, regular, components._DELTA_PX)

    def real(metric, value_px: float, lines: int | None) -> float:
        def wrapped(chars: str, width) -> float:
            """Width of the longest line of ``chars`` at its best break onto ``lines`` lines."""
            words = chars.split()
            if not words:
                return 0.0
            if lines is None:
                breaks = [words]  # one word per line
            elif lines == 2 and len(words) > 1:
                breaks = [[" ".join(words[:i]), " ".join(words[i:])] for i in range(1, len(words))]
            else:
                breaks = [[chars]]
            return min(max(width(line) for line in parts) for parts in breaks)

        label_px = wrapped(metric.label, label) + components._HELP_ICON_PX
        value_width = text(metric.value, medium, value_px)
        # A delta sits in a pill, with its padding on either side.
        pill = wrapped(metric.proto.delta, delta) + components._DELTA_PADDING_PX if metric.proto.delta else 0.0
        return max(value_width, label_px, pill) + components._CARD_CHROME_PX

    for at in states.values():
        for row, groups in _card_rows(at).items():
            metrics = [m for group in groups for m in group]
            for value_px, lines in (
                (components._VALUE_PX, 1),
                (components._VALUE_PX, 2),
                (components._MIN_VALUE_PX, None),
            ):
                model = max(components._card_width(m.label, m.value, value_px, lines, m.proto.delta) for m in metrics)
                # (the label widths are modelled exactly: the font's widths are whole 1/1000 em)
                assert model >= max(real(m, value_px, lines) for m in metrics) - 1e-9, (row, value_px, lines)
            for m in metrics:
                modelled = components._value_em(m.value) - components._VALUE_SPARE_EM
                assert modelled >= text(m.value, medium, 1) - 1e-9, m.value  # (the widths are whole 1/1000 em)
