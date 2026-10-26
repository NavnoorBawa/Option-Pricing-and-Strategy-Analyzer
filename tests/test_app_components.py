"""The shared metric-card rows (ui/components.py): their sizes, their structure and their use on every tab."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from options_analyzer import vol_surface
from options_analyzer.ui import components

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


def _card_rows(at: AppTest) -> dict[str, list[list]]:
    """Every metric of the page, by card row: ``{row key: [[metrics of group 1], ...]}``. Fails on a metric that is
    not in a group of a keyed row (a plain ``st.metric`` or one in ``st.columns``)."""
    rows: dict[str, list[list]] = {}

    def walk(node, ancestors):
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

    walk(at._tree, [])
    return {row: [metrics for _, metrics in groups] for row, groups in rows.items()}


def _styles(at: AppTest) -> list[str]:
    return [h.proto.body for h in at.get("html")]


def _app_states(new_app) -> dict[str, AppTest]:
    """The app at its default URL (every tab's default cards), with market quotes on the surface tab, and with a
    short straddle at the input limits (the widest strategy values)."""
    default = new_app().run()
    quotes = new_app().run()
    quotes.segmented_control(key="surface_source").set_value("Market quotes (CSV)").run()
    csv = vol_surface.sample_quotes().to_csv(index=False).encode()
    quotes.file_uploader(key="quotes_file").upload("quotes.csv", csv, "text/csv").run()
    limits = new_app(
        spot=9999999,
        strike=9999999,
        contracts=100000,
        multiplier=10000,
        vol=500,
        days=1,
        strategy_category="Volatility",
        strategy_Volatility="Short Straddle",
    ).run()
    states = {"default": default, "quotes": quotes, "limits": limits}
    for at in states.values():
        _ok(at)
    return states


# --- the stylesheet ------------------------------------------------------------------------------------------


FOUR = [("Call value", "$10.4506"), ("Put value", "$5.5735"), ("Forward price", "$105.1271"), ("ITM", "55.96%")]
THREE = [("VaR 99.0%", "$1,234"), ("Expected Shortfall 99.0%", "$1,456"), ("Delta-normal VaR", "$1,111")]


def _card(cards) -> int:
    return int(-(-max(components._card_width(label, value) for label, value in cards) // 1))


def test_four_cards_in_two_pairs_wrap_by_pairs():
    style = components._cards_style(FOUR, "row", ("row_a", "row_b"))
    card = _card(FOUR)
    assert (
        f".st-key-row > :has(> .st-key-row_a, > .st-key-row_b) {{ flex: 1 1 {2 * card + 14}px !important; }}" in style
    )
    assert f".st-key-row_a > *, .st-key-row_b > * {{ flex: 1 1 {card}px !important; }}" in style
    assert f"@container row (width < {2 * card + 14}px) {{" in style
    # Narrower than one pair: each card of a pair gets half the row, less half the gap and its padding.
    assert "calc((50cqw - 35px) / 5.19)" in style


def test_three_cards_in_one_group_stay_three_across_and_shrink_together():
    # Three cards in st.columns(3) stacked into three full-width cards on a phone. In one group they keep a third
    # of the row each, with the values shrinking to fit it (down to 16 px), before the group wraps.
    style = components._cards_style(THREE, "var", ("var_all",))
    card = _card(THREE)
    group = 3 * card + 2 * 14
    assert f".st-key-var > :has(> .st-key-var_all) {{ flex: 1 1 {group}px !important; }}" in style
    assert f".st-key-var_all > * {{ flex: 1 1 {card}px !important; }}" in style
    assert f"@container var (width < {group}px) {{" in style
    em = max(components._value_em(value) for _, value in THREE)
    # A third of the row, less two thirds of a 14 px gap and the 28 px padding and border.
    assert f"calc((33.3333cqw - 37.3333px) / {em:g})" in style
    compact = int(re.search(r"\.st-key-var_all > \* \{ flex-basis: (\d+)px", style).group(1))
    assert f"{{ flex-basis: {3 * compact + 28}px !important; }}" in style
    # At every row width where the three compact cards fit side by side, the widest value at its clamped size fits
    # its third of the row.
    for row in range(3 * compact + 28, group):
        font = min(max(16.0, (row / 3 - 37.3333) / em), 22.75)
        assert font * em <= row / 3 - 37.3333 + 1e-6, row


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
    for at in states.values():
        for row, groups in _card_rows(at).items():
            found[row] = [[m.label for m in group] for group in groups]
            assert all(m.proto.show_border for group in groups for m in group), row
            assert any(f".st-key-{row} {{ container:" in s for s in _styles(at)), f"{row} has no stylesheet"
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
            pair = re.search(r"@container [\w-]+ \(width < \d+px\) \{ [^{]+ \{ flex-basis: (\d+)px", style)
            assert int(pair.group(1)) <= PHONE_ROW, (row, pair.group(1))
            checked.add(row)
    assert checked == {"price_cards", "strategy_cards", "greek_cards", "surface_cards", "iv_cards"}


def _advances(weight: str) -> dict[str, float]:
    ttlib = pytest.importorskip("fontTools.ttLib")
    pytest.importorskip("brotli")  # WOFF2 decompression
    font = ttlib.TTFont(FONTS / f"IBMPlexSans-{weight}.woff2")
    cmap, metrics, units = font.getBestCmap(), font["hmtx"], font["head"].unitsPerEm
    return {chr(cp): metrics[name][0] / units for cp, name in cmap.items()}


def test_the_card_width_model_covers_every_card_row_in_plex_sans(states):
    # Every card of a row is based on the row's widest card in the model. It must not be narrower than the widest
    # card's real text (Plex Sans advance widths): the value in Medium at 22.75 px and the label (plus its help
    # icon) and delta in Regular at 12.25 px, and on a phone the value at 16 px with the label one word per line.
    medium, regular = _advances("Medium"), _advances("Regular")

    def text(chars: str, table: dict[str, float], px: float) -> float:
        return sum(table[c] for c in chars) * px

    def real(metric, value_px: float, wrap: bool) -> float:
        def words(chars: str) -> float:
            parts = chars.split() if wrap else [chars]
            return max((text(p, regular, components._LABEL_PX) for p in parts), default=0.0)

        label = words(metric.label) + components._HELP_ICON_PX
        return max(text(metric.value, medium, value_px), label, words(metric.proto.delta)) + components._CARD_CHROME_PX

    for at in states.values():
        for row, groups in _card_rows(at).items():
            metrics = [m for group in groups for m in group]
            for value_px, wrap in ((components._VALUE_PX, False), (components._MIN_VALUE_PX, True)):
                model = max(components._card_width(m.label, m.value, value_px, wrap, m.proto.delta) for m in metrics)
                assert model >= max(real(m, value_px, wrap) for m in metrics), (row, value_px)
