"""The what-if strip (options_analyzer.ui.whatif, UI-4; UI spec 3.2 item 3): its pure functions against the risk
library and the clock, and the strip itself in a small AppTest page."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from options_analyzer import clock
from options_analyzer import strategies as strat
from options_analyzer.ui import whatif
from options_analyzer.ui.whatif import Shifts

ASOF = datetime(2026, 10, 23, 16, 0)  # Friday, the cash close the app's tests pin


# --- the pure functions -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shifts", "kwargs"),
    [
        (Shifts(vol=3.0), {"vol_shift": 0.03}),
        (Shifts(vol=-4.0), {"vol_shift": -0.04}),
        (Shifts(vol=10.0, mode="relative"), {"vol_multiplier": 1.10}),
        (Shifts(vol=25.0, mode="1M-scaled"), {"vol_multiplier": 1.25, "vol_tenor_decay": 0.25}),
    ],
)
@pytest.mark.parametrize("days", [7.0, 30.0, 55.7, 400.0])
def test_leg_vol_is_the_risk_librarys_vol_shock(shifts, kwargs, days):
    # strategies.position_value applies the same three modes (RK-1's vol_mode): a leg valued at leg_vol's vol must be
    # worth exactly what position_value gives with the matching shock arguments.
    from options_analyzer import black_scholes as bs

    market = strat.Market(5800.0, 0.0475, 0.1552, days / 365.0, 0.013)
    leg = strat.Leg("call", 1, 5900.0)
    shocked = float(strat.position_value(strat.Strategy("call", (leg,)), market, **kwargs))
    sigma = whatif.leg_vol(0.1552, market.expiry, shifts)
    by_hand = float(bs.price(5800.0, 5900.0, market.expiry, 0.0475, sigma, 0.013, "call"))
    assert by_hand == pytest.approx(shocked, rel=1e-13)


def test_leg_vol_worked_examples_of_the_labels():
    # UI spec 3.2: "+3.00 vp: ATMF 15.52% → 18.52%"; relative "+10%: 15.52% → 17.07%".
    assert whatif.leg_vol(0.1552, 0.15, Shifts(vol=3.0)) == pytest.approx(0.1852, abs=1e-15)
    assert whatif.leg_vol(0.1552, 0.15, Shifts(vol=10.0, mode="relative")) == pytest.approx(0.17072, abs=1e-15)
    # 1M-scaled: a 30-day leg takes the whole shock, a shorter one no more, a one-year leg (1/12.17)^0.25 = 0.536 of it.
    one_month = strat.VOL_SHOCK_REFERENCE_TENOR  # 30 days
    assert whatif.leg_vol(0.2, one_month, Shifts(vol=10.0, mode="1M-scaled")) == pytest.approx(0.22, abs=1e-15)
    assert whatif.leg_vol(0.2, 1 / 365, Shifts(vol=10.0, mode="1M-scaled")) == pytest.approx(0.22, abs=1e-15)
    assert whatif.leg_vol(0.2, 1.0, Shifts(vol=10.0, mode="1M-scaled")) == pytest.approx(
        0.2 * (1 + 0.1 * (365 / 30) ** -0.25), abs=1e-15
    )
    assert whatif.leg_vol(0.10, 0.5, Shifts(vol=-20.0)) == whatif.VOL_FLOOR  # never negative


def test_sessions_skip_weekends_and_holidays():
    dates = whatif.session_dates(date(2026, 10, 23), 10)
    assert len(dates) == 11 and dates[0] == date(2026, 10, 23) and dates[-1] == date(2026, 11, 6)
    assert all(clock.is_trading_day(d) for d in dates[1:])
    # Thanksgiving (26 November 2026) is not a session; the early close on the 27th is.
    assert date(2026, 11, 26) not in whatif.session_dates(date(2026, 11, 20), 8)
    assert whatif.sessions_to(date(2026, 10, 23), date(2026, 12, 18)) == 39
    assert whatif.sessions_to(date(2026, 10, 23), date(2026, 10, 23)) == 0
    # T+0 is the valuation date even on a weekend, and T+1 the next session.
    assert whatif.session_dates(date(2026, 10, 24), 1) == (date(2026, 10, 24), date(2026, 10, 26))


def test_the_scenario_time_counts_calendar_time():
    # Friday's close to Monday's close is three calendar days of decay (spec 2.5: theta per calendar day).
    monday = whatif.scenario_time(ASOF, 1)
    assert monday == datetime(2026, 10, 26, 16, 0)
    assert clock.year_fraction(ASOF, monday) == pytest.approx(3 / 365, abs=1e-15)
    assert whatif.scenario_time(ASOF, 5) == datetime(2026, 10, 30, 16, 0)


def test_apply_shifts_the_market():
    market = strat.Market(5800.0, 0.0475, 0.1552, 55.7 / 365, 0.013)
    shifted = whatif.apply(market, Shifts(spot=-2.0, vol=3.0, sessions=5, rate=25.0), as_of=ASOF)
    assert shifted.spot == pytest.approx(5684.0, abs=1e-9)  # spec 3.2: "Spot −2.0% → 5,684.00"
    assert shifted.rate == pytest.approx(0.05, abs=1e-15)
    assert shifted.dividend_yield == market.dividend_yield  # the rate shift holds q (spec 2.0)
    assert shifted.volatility == pytest.approx(0.1852, abs=1e-15)
    assert shifted.expiry == pytest.approx(market.expiry - 7 / 365, abs=1e-15)  # Fri 23 to Fri 30 October
    assert whatif.apply(market, Shifts(), as_of=ASOF) == market  # no shift, the same market
    # Past the expiry the time left stops at one minute rather than going negative.
    late = whatif.apply(market, Shifts(sessions=60), as_of=ASOF)
    assert late.expiry == pytest.approx(1 / 525_600)


def test_shifts_are_active_while_any_is_non_zero():
    assert not Shifts().active and not Shifts(mode="relative").active
    for shifts in (Shifts(spot=0.5), Shifts(vol=-1.0), Shifts(sessions=1), Shifts(rate=5.0)):
        assert shifts.active
    with pytest.raises(ValueError, match="mode"):
        Shifts(mode="sqrt")


def test_the_note_names_each_shift():
    assert whatif.note(Shifts()) == ""
    assert whatif.note(Shifts(spot=-2.0, vol=3.0, sessions=5)) == (
        "Scenario on Strategy, Risk: spot −2.0% · vol +3.00 vp · T+5"
    )
    assert whatif.note(Shifts(vol=10.0, mode="relative", rate=-25.0)) == (
        "Scenario on Strategy, Risk: vol +10.0% · rate −25 bp"
    )
    assert "1M-scaled" in whatif.note(Shifts(vol=10.0, mode="1M-scaled"))


# --- the strip, in a small page -------------------------------------------------------------------------------------


def _page():
    from datetime import datetime

    import streamlit as st

    from options_analyzer.ui import whatif

    shifts = whatif.strip(
        spot=5800.0,
        atm_vol=0.1552,
        tenor=55.7 / 365,
        rate=0.0475,
        as_of=datetime(2026, 10, 23, 16, 0),
        horizon=datetime(2026, 12, 18, 9, 30),
    )
    st.text(f"shifts={shifts.spot},{shifts.vol},{shifts.mode},{shifts.sessions},{shifts.rate}")


def _app(**query):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_page, default_timeout=60)
    for name, value in query.items():
        at.query_params[name] = str(value)
    at.run()
    assert not at.exception, at.exception
    return at


def _labels(at) -> dict[str, str]:
    widgets = [*at.number_input, *at.slider]
    return {w.key: w.label for w in widgets}


def _scenario(at) -> bool:
    return any("SCENARIO" in m.value for m in at.markdown)


def test_the_strip_at_rest():
    at = _app()
    labels = _labels(at)
    assert labels["ws"] == "Spot 0.0% → 5,800.00"
    assert labels["wv"] == "Vol 0.00 vp: ATMF 15.52% → 15.52%"
    assert labels["wr"] == "Rate 0 bp: 4.75% → 4.75%"
    assert labels["wd"] == "T+0 · Fri 23\u2011Oct\u201126 (55d left)"
    assert not _scenario(at)
    assert at.button(key="whatif_reset").disabled
    assert at.text[0].value == "shifts=0.0,0.0,absolute,0,0.0"
    assert at.caption[0].value == "Sticky strike: leg vols fixed by strike."


def test_the_strip_states_its_semantics_with_worked_examples():
    at = _app(ws=-2, wv=3, wd=5, wr=25)
    labels = _labels(at)
    assert labels["ws"] == "Spot −2.0% → 5,684.00"
    assert labels["wv"] == "Vol +3.00 vp: ATMF 15.52% → 18.52%"
    assert labels["wr"] == "Rate +25 bp: 4.75% → 5.00%"
    # UI spec 3.2 acceptance: the date label reads "T+5 · Fri 30‑Oct‑26", here with the time left to 18-Dec AM
    # (Fri 30 October 16:00 to Fri 18 December 09:30: 48 days 17.5 hours, whole days from 30 days on).
    assert at.slider(key="wd").value == 5
    assert labels["wd"] == "T+5 · Fri 30‑Oct‑26 (48d left)"
    assert _scenario(at) and not at.button(key="whatif_reset").disabled
    assert at.text[0].value == "shifts=-2.0,3.0,absolute,5,25.0"


def test_the_relative_mode_reads_in_percent():
    at = _app(wv=10, wvmode="relative")
    assert _labels(at)["wv"] == "Vol +10.0%: ATMF 15.52% → 17.07%"  # spec 3.2: "+10%: 15.52% → 17.07%"
    at = _app(wv=10, wvmode="1M-scaled")
    assert _labels(at)["wv"].startswith("Vol +10.0% 1M: ATMF 15.52% → ")


def test_a_step_relabels_and_every_shift_is_bound_to_the_url():
    at = _app()
    at.number_input(key="ws").set_value(-2.0).run()
    assert _labels(at)["ws"] == "Spot −2.0% → 5,684.00"
    assert _scenario(at)
    at.slider(key="wd").set_value(10).run()
    assert _labels(at)["wd"].startswith("T+10 · Fri 06\u2011Nov\u201126 (")
    # The browser writes a bound widget's value to the link (AppTest has no browser): each one names its parameter.
    for widget in (*at.number_input, *at.slider, at.button_group(key="wvmode")):
        assert widget.proto.query_param_key == widget.key


def test_reset_clears_the_four_shifts_and_keeps_the_mode():
    at = _app(ws=-2, wv=10, wvmode="relative", wd=5, wr=25)
    at.button(key="whatif_reset").click().run()
    assert at.text[0].value == "shifts=0.0,0.0,relative,0,0.0"
    assert not _scenario(at)
    assert at.button(key="whatif_reset").disabled


def test_current_reads_the_link_before_the_strip_is_drawn():
    # Pricing, IV and Surface note the scenario without drawing the strip: current() reads the link.
    def page():
        import streamlit as st

        from options_analyzer.ui import whatif

        st.text(whatif.note(whatif.current()))

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(page)
    for name, value in {"ws": -2, "wv": 3, "wd": 5}.items():
        at.query_params[name] = str(value)
    at.run()
    assert at.text[0].value == "Scenario on Strategy, Risk: spot −2.0% · vol +3.00 vp · T+5"


def test_out_of_range_links_read_as_zero():
    def page():
        import streamlit as st

        from options_analyzer.ui import whatif

        st.text(repr(whatif.current()))

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(page)
    at.query_params["ws"] = "999"
    at.query_params["wv"] = "abc"
    at.query_params["wvmode"] = "sqrt"
    at.run()
    assert at.text[0].value == repr(Shifts())
