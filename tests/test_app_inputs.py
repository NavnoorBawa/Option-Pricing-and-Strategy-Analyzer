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
from options_analyzer.ui.inputs import SURFACE, days_text

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
    ("days", "text"), [(27.73, "28d"), (7.0, "7d"), (3.26, "3d 6h"), (2.999, "3d 0h"), (0.22, "5.3h")]
)
def test_time_to_expiry_in_short(days, text):
    assert mk.span(days) == text


def test_days_in_captions():
    assert (days_text(365), days_text(27.73), days_text(0.22)) == ("365 days", "27.7 days", "5.3 hours")


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
    assert mk.expiry_label(market.expiry_row(date(2026, 11, 20))) == "20‑Nov‑26 AM · 28d"
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
    identity, contract = at.markdown[0].value, next(m.value for m in at.markdown if m.value.startswith("SPX"))
    assert _plain(identity) == "**Options Analytics** 23-Oct-26 16:00 ET · SPX 5,800.00"
    assert _plain(contract).startswith("SPX 20-Nov-26 AM 5,815 · 28d · σ ")
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
    cards = _cards(index_app)
    assert cards["Call value"] == f"${call:,.2f}" and cards["Put value"] == f"${put:,.2f}"
    assert cards["Forward price"] == f"${F:,.2f}"
    contract = next(m.value for m in index_app.markdown if m.value.startswith("SPX"))
    assert f"σ\xa0{vol:.2%}\xa0·" in contract
    assert any(f"DF {DF:.5f}" in c.value for c in index_app.sidebar.caption)


def test_the_textbook_preset_reproduces_version_2(new_app):
    at = new_app(preset="textbook").run()  # a hand-typed name: rewritten to the label
    _ok(at)
    assert at.sidebar.selectbox(key="preset").value == mk.TEXTBOOK
    assert at.query_params["preset"] == "Textbook"
    cards = _cards(at)
    assert (cards["Call value"], cards["Put value"]) == ("$10.4506", "$5.5735")


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
    assert _plain(label).split(" · ")[0] in _plain(next(m.value for m in at.markdown if m.value.startswith("SPX")))


def test_the_index_offers_no_american_engine(new_app):
    at = new_app(preset=None, asof=ASOF, model="alo").run()
    _ok(at)
    assert at.sidebar.radio(key="model").value == "bsm"
    assert "alo" not in at.sidebar.radio(key="model").options
    assert [w.value for w in at.warning] == [
        "This link asks for the American engine, but options on the Synthetic index are European-exercise (as SPX "
        "options are), so Black-Scholes-Merton is used. The American engine prices the Textbook market."
    ]
    tree = new_app(preset=None, asof=ASOF, model="Binomial tree").run()
    _ok(tree)
    with pytest.raises(KeyError):  # a tree values European exercise: no Exercise control
        tree.sidebar.segmented_control(key="style")
    assert "European exercise: SPX options are European-style." in [c.value for c in tree.sidebar.caption]


def test_switching_the_market_resets_the_contract(new_app):
    at = new_app(preset=None, asof=ASOF).run()
    at.sidebar.number_input(key="strike").set_value(5900.0).run()
    at.sidebar.selectbox(key="preset").set_value(mk.TEXTBOOK).run()
    _ok(at)
    assert at.sidebar.number_input(key="strike").value == 100.0
    assert (_cards(at)["Call value"], _cards(at)["Put value"]) == ("$10.4506", "$5.5735")
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


def test_an_override_vol_is_marked(new_app):
    at = new_app(preset=None, asof=ASOF).run()
    surface_card = _cards(at)["Call value"]
    at.sidebar.segmented_control(key="vol_src").set_value("Override").run()
    _ok(at)
    vol = at.sidebar.number_input(key="vol")
    assert _cards(at)["Call value"] == surface_card  # the override starts at the surface's vol, rounded
    assert not any("OVERRIDE" in b for b in _badges(at))
    vol.set_value(vol.value + 1.0).run()
    _ok(at)
    assert ":orange-badge[OVERRIDE]" in _badges(at)
    assert "%*" in next(m.value for m in at.markdown if m.value.startswith("SPX"))
    assert _cards(at)["Call value"] != surface_card
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
    assert cards["Calendar arbitrage"] == "0" and cards["Butterfly arbitrage"] == "0"
    assert any(c.value.startswith(mk.SYNTHETIC_NOTE) for c in at.caption)
    assert not any("sidebar's (rate" in w.value for w in at.warning)  # no carry-gap warning on the chain


def test_the_textbook_header_keeps_its_contract_ticker(new_app):
    at = new_app().run()
    _ok(at)
    assert _plain(at.markdown[0].value).startswith("**Options Analytics** S 100.00 · K 100.00 · ")
    assert not any("SYNTHETIC" in b for b in _badges(at))
    assert header.SHARE_NOTE and TEXTBOOK["preset"] == mk.TEXTBOOK
