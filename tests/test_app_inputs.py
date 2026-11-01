"""The sidebar's market (ui/inputs.py, ui/market.py): the synthetic index preset, the textbook preset, and the links
that open them."""

from __future__ import annotations

import math
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from _apptest import ASOF, TEXTBOOK
from hypothesis import given
from hypothesis import strategies as hs
from scipy.special import ndtr

from options_analyzer.market_data import synthetic
from options_analyzer.ui import header
from options_analyzer.ui import market as mk
from options_analyzer.ui.app import _link_asof
from options_analyzer.ui.inputs import SURFACE

NEW_YORK = ZoneInfo("America/New_York")
#: The pinned valuation time as a datetime (New York wall time).
AT = datetime.fromisoformat(ASOF)


def _ok(at) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def _cards(at) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


def _plain(text: str) -> str:
    return text.replace("\xa0", " ").replace("‑", "-")


def _badges(at) -> list[str]:
    return [m.value for m in at.markdown if m.value.startswith(":")]


# --------------------------------------------------------------------------------------------------- market.py


@pytest.mark.parametrize(
    ("now", "close"),
    [
        (datetime(2026, 10, 23, 16, 0), datetime(2026, 10, 23, 16, 0)),  # at the close: that close
        (datetime(2026, 10, 23, 15, 59), datetime(2026, 10, 22, 16, 0)),  # before it: the day before
        (datetime(2026, 10, 24, 12, 0), datetime(2026, 10, 23, 16, 0)),  # Saturday
        (datetime(2026, 10, 26, 9, 0), datetime(2026, 10, 23, 16, 0)),  # Monday before the open
        (datetime(2026, 11, 26, 12, 0), datetime(2026, 11, 25, 16, 0)),  # Thanksgiving
        (datetime(2026, 11, 27, 12, 0), datetime(2026, 11, 25, 16, 0)),  # the early close is at 13:00
        (datetime(2026, 11, 27, 14, 0), datetime(2026, 11, 27, 13, 0)),
    ],
)
def test_the_default_valuation_time_is_the_last_completed_cash_close(now, close):
    assert mk.last_close(now.replace(tzinfo=NEW_YORK)) == close
    # The same moment in another time zone gives the same close (New York wall time, no time zone).
    assert mk.last_close(now.replace(tzinfo=NEW_YORK).astimezone(ZoneInfo("Asia/Tokyo"))) == close


def test_a_valuation_time_is_kept_on_its_five_minute_step():
    assert mk.snap(datetime(2026, 10, 23, 16, 4, 59, 999)) == datetime(2026, 10, 23, 16, 0)
    assert mk.snap(datetime(2026, 10, 23, 20, 7, tzinfo=ZoneInfo("UTC"))) == datetime(2026, 10, 23, 16, 5)


@given(
    forward=hs.floats(1.0, 1e5),
    spot=hs.floats(1.0, 1e5),
    discount=hs.floats(0.5, 1.02),
    maturity=hs.floats(1 / 525_600, 5.0),
)
def test_the_carry_puts_the_forward_and_discount_factor_into_bsm(forward, spot, discount, maturity):
    rate, dividend_yield = mk.carry(forward, discount, spot, maturity)
    assert spot * math.exp((rate - dividend_yield) * maturity) == pytest.approx(forward, rel=1e-12)
    assert math.exp(-rate * maturity) == pytest.approx(discount, rel=1e-12)


@pytest.mark.parametrize(
    ("moment", "note"),
    [
        (datetime(2026, 10, 23, 16, 0), "cash close"),
        (datetime(2026, 11, 27, 13, 0), "early close"),
        (datetime(2026, 10, 23, 9, 0), "before the open"),
        (datetime(2026, 10, 23, 9, 30), "regular hours"),
        (datetime(2026, 10, 23, 17, 0), "after the close"),
        (datetime(2026, 10, 24, 12, 0), "weekend"),
        (datetime(2026, 12, 25, 12, 0), "holiday"),
    ],
)
def test_the_session_note(moment, note):
    assert mk.session_note(moment) == note


@pytest.mark.parametrize(
    ("days", "text"),
    # A countdown in whole units, as every tab writes it (format.time_left): 55d 17h 31m is 55d, never 56d.
    [(55.73, "55d"), (30.0, "30d"), (27.73, "27d 17h"), (7.0, "7d 0h"), (2.999, "2d 23h"), (0.22, "5h 16m")],
)
def test_time_to_expiry_in_short(days, text):
    assert mk.span(days) == text


def test_the_index_market_reads_the_chain_back():
    # The chain is priced by the generator and read back like an upload, so its forwards, discount factors and ATM
    # vols must come back close to the generator's: measured at the pinned time, forwards within 2.8e-5 relative
    # (0.3 bp), discount factors within 2.0e-4 (the parity slope's error with $0.05 ticks) and ATM vols within
    # 0.006 vol point of the generator's noise-free ones.
    market = mk.index_market(AT)
    expiries, generated = market.expiries, market.chain.expiries
    assert np.all(np.diff(expiries["maturity"].to_numpy(float)) > 0)
    box_rate = mk.INDEX_RATE + synthetic.BOX_SPREAD
    maturity = expiries["maturity"].to_numpy(float)
    true_forward = mk.INDEX_SPOT * np.exp((box_rate - mk.INDEX_YIELD) * maturity)
    assert np.max(np.abs(expiries["forward"] / true_forward - 1)) < 5e-5
    assert np.max(np.abs(expiries["discount_factor"] / np.exp(-box_rate * maturity) - 1)) < 4e-4
    assert np.max(np.abs(expiries["atm_vol"].to_numpy() - generated["atm_vol"].to_numpy())) < 0.0002
    # The listed expiry nearest 30 days: 20-Nov-26 AM at 27.7 days beats 27-Nov-26 PM at 34.9.
    assert market.default_expiry() == date(2026, 11, 20)
    assert market.atm_strike(date(2026, 11, 20)) == 5815.0  # F 5,813.91 on the 5-point grid
    assert mk.expiry_label(market.expiry_row(date(2026, 11, 20))) == "20‑Nov‑26 AM · 27d 17h"
    assert mk.index_market(AT) is market  # cached per valuation time


# --------------------------------------------------------------------------------------------- links (app.py)


@pytest.mark.parametrize(
    ("value", "moment"),
    [
        ("2026-10-23T16:00", datetime(2026, 10, 23, 16, 0)),
        ("2026-10-23T16:02:30", datetime(2026, 10, 23, 16, 0)),
        ("2026-10-23 10:07", datetime(2026, 10, 23, 10, 5)),
        ("2026-10-23T20:00Z", datetime(2026, 10, 23, 16, 0)),  # UTC
        ("2026-10-23", datetime(2026, 10, 23, 16, 0)),  # a date: its close
        ("2026-11-27", datetime(2026, 11, 27, 13, 0)),  # an early close
        ("2026-10-24", datetime(2026, 10, 23, 16, 0)),  # a Saturday: Friday's close
        ("1999-12-31T16:00", None),
        # A date alone outside the calendar (2000 to 2100), or whose trading day before falls outside it, is out of
        # range, not an error: these raised ValueError from the calendar and crashed the app.
        ("1999-06-01", None),
        ("2000-01-01", None),
        ("2098-01-02", None),
        ("2150-01-01", None),
        ("2000-01-03", datetime(2000, 1, 3, 16, 0)),  # the first trading day it accepts
        ("noon", None),
    ],
)
def test_a_link_gives_the_valuation_time_in_any_iso_form(value, moment):
    assert _link_asof(value) == moment


# ----------------------------------------------------------------------------------------------- the app itself


def test_the_app_opens_on_the_synthetic_index(index_app):
    at = index_app
    _ok(at)
    assert at.sidebar.selectbox(key="preset").value == mk.INDEX
    with pytest.raises(KeyError):  # under a chain the spot is observed, not typed
        at.sidebar.number_input(key="spot")
    identity = at.markdown[0].value
    assert _plain(identity) == "**Options Analytics** 23-Oct-26 16:00 ET · SPX 5,800.00"
    # The contract is the Pricing tab's, above its tiles, not the header's (UI spec 2.2).
    contract = [_plain(c.value) for c in at.main.caption if c.value.startswith("SPX")]
    assert contract == ["SPX 20-Nov-26 AM 5,815 · last trade Thu 19-Nov · settles at the SOQ"]
    assert not [m.value for m in at.markdown if m.value.startswith("SPX")]  # not in the header
    assert ":gray-badge[SYNTHETIC]" in _badges(at)
    assert at.sidebar.selectbox(key="expiry").value == "2026-11-20"
    assert at.sidebar.number_input(key="strike").value == 5815.0
    assert at.sidebar.segmented_control(key="vol_src").value == SURFACE
    assert not at.warning


def test_the_index_prices_are_black_76_on_the_parity_forward_and_the_surface_vol(index_app):
    # Black-Scholes-Merton at the implied carry is Black-76 on the expiry's forward F and discount factor DF:
    # C = DF·(F·N(d1) − K·N(d2)), with the vol of the fitted surface at the strike.
    market = mk.index_market(AT)
    row = market.expiry_row(date(2026, 11, 20))
    F, DF, T, K = float(row["forward"]), float(row["discount_factor"]), float(row["maturity"]), 5815.0
    vol = market.surface.implied_vol_at(K / market.spot, T, extrapolate=True)
    d1 = (math.log(F / K) + 0.5 * vol**2 * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    call = DF * (F * ndtr(d1) - K * ndtr(d2))
    put = DF * (K * ndtr(-d2) - F * ndtr(-d1))
    cards = _cards(index_app)  # the Pricing tab's tiles (UI-1)
    assert cards["Call"] == f"{call:,.2f}" and cards["Put"] == f"{put:,.2f}"
    assert cards["Forward"] == f"{F:,.2f}"
    assert cards["Vol"] == f"{vol:.2%}"
    assert any(f"DF {DF:.5f}" in c.value for c in index_app.sidebar.caption)


def test_the_textbook_preset_reproduces_version_2(new_app):
    at = new_app(preset="textbook").run()  # a hand-typed name: rewritten to the label
    _ok(at)
    assert at.sidebar.selectbox(key="preset").value == mk.TEXTBOOK
    assert at.query_params["preset"] == "Textbook"
    cards = _cards(at)
    assert (cards["Call"], cards["Put"]) == ("10.45", "5.57")


@pytest.mark.parametrize("query", [{"spot": "105"}, {"days": "30"}, {"rate": "4", "div": "1"}])
def test_a_version_2_link_opens_the_textbook_market(new_app, query):
    # A link with the typed market inputs of version 2 (and no preset) reproduces the numbers it was made with.
    at = new_app(preset=None, **query).run()
    _ok(at)
    assert at.sidebar.selectbox(key="preset").value == mk.TEXTBOOK
    assert not at.warning


def test_a_link_to_a_market_that_does_not_exist_opens_the_index_with_a_note(new_app):
    at = new_app(preset="crypto", asof=ASOF).run()
    _ok(at)
    assert at.sidebar.selectbox(key="preset").value == mk.INDEX
    assert [w.value for w in at.warning] == [
        "This link asks for a market named `crypto`, which does not exist, so the Synthetic index market is used."
    ]


@pytest.mark.parametrize(
    ("asof", "shown", "notes"),
    [
        ("2026-10-23", datetime(2026, 10, 23, 16, 0), []),
        ("2026-10-23T11:02", datetime(2026, 10, 23, 11, 0), []),
        (
            "yesterday",
            None,
            [
                "This link asks for a valuation time `yesterday`, which is not a date and time between 2000 and 2097 "
                "(2026-10-23T16:00), so the last close is used."
            ],
        ),
    ],
)
def test_a_link_sets_the_valuation_time(new_app, asof, shown, notes):
    at = new_app(preset=None, asof=asof).run()
    _ok(at)
    assert at.sidebar.datetime_input(key="asof").value == (shown or mk.last_close())
    assert [w.value for w in at.warning] == notes


@pytest.mark.parametrize("asof", ["1999-06-01", "2000-01-01", "2150-01-01"])
def test_a_link_with_a_date_outside_the_calendar_gets_a_note(new_app, asof):
    # These crashed the app with the calendar's ValueError (it covers 2000 to 2100).
    at = new_app(preset=None, asof=asof).run()
    _ok(at)
    assert at.sidebar.datetime_input(key="asof").value == mk.last_close()
    assert len(at.warning) == 1 and f"`{asof}`" in at.warning[0].value


@pytest.mark.parametrize("value", ["inf", "-inf", "nan", "1e400"])
@pytest.mark.parametrize(
    ("key", "query", "default"),
    [
        ("contracts", {"preset": None, "asof": ASOF}, 1),
        ("multiplier", {"preset": None, "asof": ASOF}, 100),
        ("days", {}, 365),
        ("wr", {"tab": "Risk"}, 0),
    ],
)
def test_a_link_with_a_number_that_is_not_finite_gets_a_note(new_app, key, query, default, value):
    # An integer input read int(float("inf")) from the URL and raised OverflowError on every run of every tab.
    at = new_app(**query, **{key: value}).run()
    _ok(at)
    assert at.number_input(key=key).value == default
    assert [w.value for w in at.warning if f"`{value}`" in w.value] != []
    assert key not in at.query_params


def test_a_link_with_a_number_out_of_range_gets_a_note(new_app):
    at = new_app(preset=None, asof=ASOF, strike="100000").run()
    _ok(at)
    assert at.sidebar.number_input(key="strike").value == 5815.0  # ATMF
    assert [w.value for w in at.warning] == [
        "This link gives the strike as `100000`, which is not a number from 5 to 23,200, so the default is used."
    ]


#: Every parameter bound to a single-choice control or a number, per tab, that a link may hold empty.
EMPTY_KEYS = {
    "Pricing": ("pr_metric", "pr_x", "pr_range", "pr_chain_win", "pr_chain_stride", "pr_greeks", "strike", "vol_src"),
    "Strategy": ("st_view", "st_range", "st_greek", "st_units", "strategy", "contracts", "multiplier", "commission"),
    "Risk": ("report", "lad_axis", "lad_roll", "lad_metric", "ws", "wv", "wvmode", "wd", "wr"),
}


@pytest.mark.parametrize("tab", list(EMPTY_KEYS))
def test_a_link_with_empty_values_means_no_parameters(new_app, tab):
    # An empty value of a required control read as None: ?pr_metric= raised KeyError, ?pr_range= AttributeError.
    at = new_app(preset=None, asof=ASOF, tab=tab, **dict.fromkeys(EMPTY_KEYS[tab], "")).run()
    _ok(at)
    assert not at.warning
    assert not set(EMPTY_KEYS[tab]) & set(at.query_params)


def test_an_empty_textbook_parameter_is_no_parameter(new_app):
    # A link with ?spot= and no preset is not a version-2 link: it opens the index, as a link without ?spot= does.
    at = new_app(preset=None, asof=ASOF, spot="", days=" ").run()
    _ok(at)
    assert at.sidebar.selectbox(key="preset").value == mk.INDEX
    assert not at.warning


@pytest.mark.parametrize("report", ["Ladder", "Scenarios", "VaR"])
def test_a_risk_report_link_with_empty_values_opens(new_app, report):
    keys = ("lad_roll", "lad_metric", "sc_volmode", "var_model", "sc_view", "lad_axis")
    at = new_app(preset=None, asof=ASOF, tab="Risk", report=report, **dict.fromkeys(keys, "")).run()
    _ok(at)


def test_a_strike_off_the_listed_grid_moves_to_the_nearest_listed_one(new_app):
    # 5,817.37 priced a contract SPX does not list, while the box showed 5817.
    at = new_app(preset=None, asof=ASOF, strike="5817.37").run()
    _ok(at)
    assert at.sidebar.number_input(key="strike").value == 5815.0
    assert [w.value for w in at.warning] == [
        "This link asks for the strike `5817.37`, which is not listed (every 5 points near the money), so the nearest "
        "listed one, 5,815, is used."
    ]
    at.sidebar.number_input(key="strike").set_value(5838.0).run()
    _ok(at)
    assert at.sidebar.number_input(key="strike").value == 5840.0
    assert [i.value for i in at.info] == [
        "5,838 is not a listed strike (every 5 points near the money), so the nearest listed one, 5,840, is used."
    ]


@pytest.mark.parametrize(
    ("expiry", "chosen", "note"),
    [
        ("2026-11-27", "2026-11-27", None),
        ("2026-11-25", "2026-11-27", "25-Nov-26 is not a listed expiry at this valuation time, so the nearest "
                                     "listed one, 27-Nov-26, is used."),
        ("soon", "2026-11-20", "The expiry 'soon' is not a date (2026-11-20), so the expiry nearest 30 days is used."),
    ],
)  # fmt: skip
def test_a_link_picks_a_listed_expiry(new_app, expiry, chosen, note):
    at = new_app(preset=None, asof=ASOF, expiry=expiry).run()
    _ok(at)
    assert at.sidebar.selectbox(key="expiry").value == chosen
    assert [i.value for i in at.info if "expiry" in i.value] == ([note] if note else [])
    # The link holds the listed expiry, or nothing for the default one.
    assert at.query_params.get("expiry") == (None if chosen == "2026-11-20" else chosen)
    label = mk.expiry_label(mk.index_market(AT).expiry_row(date.fromisoformat(chosen)))
    assert _plain(label).split(" · ")[0] in _plain(next(c.value for c in at.main.caption if c.value.startswith("SPX")))


def test_the_index_offers_no_american_engine(new_app):
    at = new_app(preset=None, asof=ASOF, model="alo").run()
    _ok(at)
    assert not [c for c in at.sidebar.segmented_control if c.key == "model"]  # a caption says why, in its place
    assert any(c.value.startswith("European (BSM), the closed form") for c in at.sidebar.caption)
    assert [w.value for w in at.warning] == [
        "This link asks for the American engine, but options on the Synthetic index are European-exercise (as SPX "
        "options are), so European (BSM) is used. The American engine prices the Textbook market."
    ]
    tree = new_app(preset=None, asof=ASOF, model="Binomial tree").run()
    assert not tree.exception
    assert any(c.value.startswith("European (BSM), the closed form") for c in tree.sidebar.caption)
    assert [w.value for w in tree.warning] == [
        "This link asks for a binomial tree, which now runs in the Pricing tab's Engine diagnostics (open below); "
        "the sidebar prices with European (BSM)."
    ]
    # The diagnostics' trees value European exercise, as SPX options are.
    engines = next(d.value for d in tree.dataframe if "Engine" in d.value)
    assert engines["Engine"].str.contains("European").all()


def test_switching_the_market_resets_the_contract(new_app):
    at = new_app(preset=None, asof=ASOF).run()
    at.sidebar.number_input(key="strike").set_value(5900.0).run()
    at.sidebar.selectbox(key="preset").set_value(mk.TEXTBOOK).run()
    _ok(at)
    assert at.sidebar.number_input(key="strike").value == 100.0
    assert (_cards(at)["Call"], _cards(at)["Put"]) == ("10.45", "5.57")
    at.sidebar.selectbox(key="preset").set_value(mk.INDEX).run()
    _ok(at)
    assert at.sidebar.number_input(key="strike").value == 5815.0
    assert at.sidebar.segmented_control(key="vol_src").value == SURFACE


def test_atmf_snaps_the_strike_to_the_forward(new_app):
    at = new_app(preset=None, asof=ASOF, strike="6000", expiry="2027-01-15").run()
    _ok(at)
    assert at.sidebar.number_input(key="strike").value == 6000.0
    at.sidebar.button(key="strike_atmf").click().run()
    forward = float(mk.index_market(AT).expiry_row(date(2027, 1, 15))["forward"])
    assert at.sidebar.number_input(key="strike").value == 5 * round(forward / 5) == 5840.0


def test_a_strike_beyond_the_surface_draws_no_override_badge(new_app):
    # Both vols were NaN there and math.isclose(nan, nan) is False, so the amber OVERRIDE badge showed, with a tooltip
    # that read "+nan vp", while the vol source was still Surface.
    at = new_app(preset=None, asof=ASOF, strike="9000").run()
    assert len(at.error) == 1 and "gives no vol at the strike 9,000" in at.error[0].value
    assert not any("OVERRIDE" in b for b in _badges(at))


def test_vol_overridden_reads_a_missing_surface_vol():
    model = header.ModelConfig()
    typed = header.Inputs(5800.0, 9000.0, 28.0, 0.30, 0.04, 0.01, model, mk.INDEX, surface_vol=math.nan)
    assert typed.vol_overridden  # a typed vol where the surface gives none: user state
    assert not header.Inputs(5800.0, 9000.0, 28.0, math.nan, 0.04, 0.01, model, surface_vol=math.nan).vol_overridden
    assert not header.Inputs(5800.0, 5815.0, 28.0, 0.15, 0.04, 0.01, model, surface_vol=0.15).vol_overridden
    assert not header.Inputs(100.0, 100.0, 365, 0.2, 0.05, 0.0, model).vol_overridden  # textbook: no surface


def test_an_override_vol_is_marked(new_app):
    at = new_app(preset=None, asof=ASOF).run()
    surface_card = _cards(at)["Call"]
    at.sidebar.segmented_control(key="vol_src").set_value("Override").run()
    _ok(at)
    vol = at.sidebar.number_input(key="vol")
    assert _cards(at)["Call"] == surface_card  # the override starts at the surface's vol, rounded
    assert not any("OVERRIDE" in b for b in _badges(at))
    vol.set_value(vol.value + 1.0).run()
    _ok(at)
    assert ":yellow-badge[OVERRIDE]" in _badges(at)
    vol_tile = next(m for m in at.metric if m.label == "Vol")
    assert vol_tile.value == f"{vol.value:.2f}%" and _plain(vol_tile.delta) == "override +1.00 vp"
    assert _cards(at)["Call"] != surface_card
    assert any(c.value.startswith("+1.00 vp against the surface's") for c in at.sidebar.caption)


def test_the_valuation_time_buttons(new_app):
    at = new_app(preset=None, asof=ASOF).run()
    at.sidebar.button(key="asof_close").click().run()
    _ok(at)
    assert at.sidebar.datetime_input(key="asof").value == mk.last_close()


def test_the_share_link_pins_the_valuation_time(new_app):
    at = new_app(preset=None, asof=ASOF, expiry="2026-11-27").run()
    _ok(at)
    codes = [c.value for c in at.code]  # the Share popover's link
    assert codes and codes[0].startswith("?") and "asof=2026-10-23T16:00" in codes[0]
    assert "expiry=2026-11-27" in codes[0]


def test_the_surface_tab_shows_the_published_surface(new_app):
    at = new_app(preset=None, asof=ASOF, tab="Surface").run()
    _ok(at)
    assert at.segmented_control(key="surface_source").value == "Synthetic chain"
    cards = _cards(at)
    assert cards["ARB"] == "0 / 0"  # calendar / butterfly violations
    assert any(mk.SYNTHETIC_NOTE.rstrip(".") in m.value for m in at.markdown)  # the data panel
    assert not any("sidebar's (rate" in w.value for w in at.warning)  # no carry-gap warning on the chain


def test_the_textbook_header_keeps_its_contract_ticker(new_app):
    at = new_app().run()
    _ok(at)
    assert _plain(at.markdown[0].value).startswith("**Options Analytics** S 100.00 · K 100.00 · ")
    assert not any("SYNTHETIC" in b for b in _badges(at))
    assert header.SHARE_NOTE and TEXTBOOK["preset"] == mk.TEXTBOOK
