"""The shared metric-card rows (ui/components.py): their sizes, their structure and their use on every tab."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
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


#: The tabs with card rows (only the open tab runs; the price cards sit above the tabs, on every page).
CARD_TABS = ("Strategy", "Risk", "Surface", "IV")


def _app_states(new_app) -> dict[str, list[AppTest]]:
    """The app at its default URL (every tab's default cards), with market quotes on the surface tab, with a
    100,000-lot position (amounts in millions: "$46.62M") and with a short straddle at the input limits (the widest
    strategy values): for each, one page per tab with cards."""

    def pages(**query) -> list[AppTest]:
        return [new_app(tab=tab, **query).run() for tab in CARD_TABS]

    quotes = new_app(tab="Surface").run()
    quotes.segmented_control(key="surface_source").set_value("Market quotes (CSV)").run()
    csv = vol_surface.sample_quotes().to_csv(index=False).encode()
    quotes.file_uploader(key="quotes_file").upload("quotes.csv", csv, "text/csv").run()
    limits = pages(
        spot=9999999,
        strike=9999999,
        contracts=100000,
        multiplier=10000,
        vol=500,
        days=1,
        strategy_category="Volatility",
        strategy_Volatility="Short Straddle",
    )
    default = pages()
    with_quotes = [quotes if tab == "Surface" else page for tab, page in zip(CARD_TABS, default, strict=True)]
    states = {"default": default, "quotes": with_quotes, "large": pages(contracts=100000), "limits": limits}
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
        if row < int(re.match(r"[\w-]+ \(width < (\d+)px\)", block).group(1)):
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
    "price_cards": [["Call value", "Put value"], ["Forward price", "Call ITM probability"]],
    "strategy_cards": [["Net premium", "Max profit"], ["Max loss", "Probability of profit"]],
    "greek_cards": [["Delta (shares)", "Gamma"], ["Theta / day", "Vega / vol pt"]],
    "expectation_cards": [["Expected P&L at expiry", "P&L standard deviation", "Drift used"]],
    "var_cards": [["VaR 99.0%", "Expected Shortfall 99.0%", "Delta-normal VaR"]],
    "surface_cards": [["Grid", "Calendar arbitrage"], ["Butterfly arbitrage", "Local vol defined"]],
    "iv_cards": [["Implied volatility", "Model check"], ["Vega at IV", "Time value"]],
}


@pytest.fixture(scope="module")
def states():
    from _apptest import make_app

    return _app_states(make_app)


def test_every_metric_of_every_tab_is_in_a_card_row(states):
    # The Strategy, Risk and Surface tabs drew their metrics in st.columns, which stack one card per row on a phone.
    found = {}
    for state, at in states.items():
        for row, groups in _card_rows(at).items():
            assert all(m.proto.show_border for group in groups for m in group), row
            assert any(f".st-key-{row} {{ container:" in s for s in _styles(at)), f"{row} has no stylesheet"
            labels = [[m.label for m in group] for group in groups]
            if state == "limits" and row == "greek_cards":
                # One day to expiry: the decay to expiry takes theta's place (tab_strategy.decay_card).
                assert labels == [["Delta (shares)", "Gamma"], ["Decay to expiry", "Vega / vol pt"]]
                continue
            found[row] = labels
    assert found == ROWS


def test_the_four_card_rows_stay_two_by_two_on_a_phone(states):
    # On a 360 px phone each pair of a four-card row fits the row side by side (the cards do not stack). At the input
    # limits this holds for the strategy cards, whose amounts are shortened to at most nine characters; the price
    # and implied-volatility cards show every digit of a seven-digit price (a forward of "$10,001,368.9567") and
    # stack there.
    checked = set()
    for state, at in states.items():
        for row, groups in _card_rows(at).items():
            if len(groups) != 2 or (state == "limits" and row in ("price_cards", "iv_cards")):
                continue
            style = next(s for s in _styles(at) if f".st-key-{row} {{ container:" in s)
            assert _layout(style, PHONE_ROW)[0] == 2, (state, row)
            checked.add(row)
    assert checked == {"price_cards", "strategy_cards", "greek_cards", "surface_cards", "iv_cards"}


@pytest.mark.parametrize(
    ("phone", "expected"),
    # Cards per line of the three-card rows at the default URL on common phone widths (360 px Android, 375 and 390
    # px iPhone, 412, 430 and 440 px large phones). The expected-P&L cards go 3 across from a 409 px row (phones of
    # 437 px and up), where each card has room for its value at full size and "EXPECTED P&L" over "AT EXPIRY" in
    # the labels' small capitals; the VaR cards need a 460 px row ("EXPECTED" over "SHORTFALL 99.0%"), so they go
    # 2 + 1 on every phone.
    [
        (360, {"expectation_cards": 2, "var_cards": 2}),
        (375, {"expectation_cards": 2, "var_cards": 2}),
        (390, {"expectation_cards": 2, "var_cards": 2}),
        (412, {"expectation_cards": 2, "var_cards": 2}),
        (430, {"expectation_cards": 2, "var_cards": 2}),
        (440, {"expectation_cards": 3, "var_cards": 2}),
    ],
)
def test_the_three_card_rows_on_a_phone(states, phone, expected):
    # At 375 and 390 px the expected-P&L cards were 3 across with values shrunk to 19.6 and 20.9 px under labels on
    # three lines, next to rows of 22.75 px values on the same screen.
    row = phone - 2 * 14  # the page's side margins on a phone
    for key, per_line in expected.items():
        style = next(s for s in _styles(states["default"]) if f".st-key-{key} {{ container:" in s)
        assert _layout(style, row) == (per_line, 22.75), key


@pytest.mark.parametrize("phone", [360, 375, 390, 412, 430, 440])
def test_every_card_value_of_a_phone_screen_has_one_size(states, phone):
    # Every row of every tab at the default URL shows its values at the theme's full size on a phone, so the type
    # does not change from row to row on one screen (no query applies: full size). On a 320 px phone the price
    # cards' values shrink to 21.4 px so that each pair stays side by side.
    row = phone - 2 * 14
    for key, style in ((k, s) for s in _styles(states["default"]) for k in ROWS if f".st-key-{k} {{ container:" in s):
        assert (_layout(style, row) or (0, 22.75))[1] == 22.75, (key, _layout(style, row))


def test_the_strategy_summary_and_greek_rows_wrap_together(states):
    # The two rows sit one above the other on the Strategy tab, so each is sized with the other's cards: their
    # stylesheets are the same but for the keys, and they wrap at the same widths with their cards lined up.
    for state, at in states.items():
        summary, greeks = (
            next(s for s in _styles(at) if f".st-key-{key} {{ container:" in s)
            for key in ("strategy_cards", "greek_cards")
        )
        assert greeks == summary.replace("strategy_cards", "greek_cards").replace("strategy-cards", "greek-cards"), (
            state
        )


def _advances(weight: str) -> dict[str, float]:
    ttlib = pytest.importorskip("fontTools.ttLib")
    pytest.importorskip("brotli")  # WOFF2 decompression
    font = ttlib.TTFont(FONTS / f"IBMPlexSans-{weight}.woff2")
    cmap, metrics, units = font.getBestCmap(), font["hmtx"], font["head"].unitsPerEm
    return {chr(cp): metrics[name][0] / units for cp, name in cmap.items()}


def test_the_card_width_model_covers_every_card_row_in_plex_sans(states):
    # Every card of a row is based on the row's widest card in the model. It must not be narrower than the widest
    # card's real text (Plex Sans advance widths): the value in Medium at 22.75 px, the label in Regular capitals at
    # 10.08 px with its letter-spacing (plus its help icon) and the delta in Regular at 12.25 px, on one line or on
    # two (a line of three cards on a phone), and the value at 16 px with the label one word per line (a pair on a
    # phone). And no value is modelled narrower than it is, as the compact rows size their values from the model:
    # "$46.62M" was modelled 0.11 em too narrow ("M" is 0.814 em wide), so on a 390 px phone it ran 2.6 px into its
    # card's padding.
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
        return max(value_width, label_px, wrapped(metric.proto.delta, delta)) + components._CARD_CHROME_PX

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
