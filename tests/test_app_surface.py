"""Volatility Surface tab, end to end with streamlit.testing (headless)."""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from options_analyzer import vol_surface
from options_analyzer.ui import tab_surface

APP = str(Path(__file__).resolve().parents[1] / "main.py")


@pytest.fixture
def app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def _ok(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def _picker(at: AppTest):
    return next(w for w in at.multiselect if w.label == "Maturities")


def _trace_names(at: AppTest) -> list[str]:
    return [t.get("name") or "" for c in at.get("plotly_chart") for t in json.loads(c.proto.spec)["data"]]


def _metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def _upload(at: AppTest, frame: pd.DataFrame) -> AppTest:
    at.segmented_control(key="surface_source").set_value("Market quotes (CSV)").run()
    at.file_uploader(key="quotes_file").upload("quotes.csv", frame.to_csv(index=False).encode(), "text/csv").run()
    return at


def _ssvi_quotes(days, moneyness, rate=0.05) -> pd.DataFrame:
    params = vol_surface.SSVIParams()
    rows = []
    for d in days:
        t = d / 365
        for m in moneyness:
            iv = np.sqrt(params.total_variance(np.log(m) - rate * t, t)[0] / t)
            rows.append({"Strike": round(100 * m, 4), "Days": d, "ImpliedVol": round(float(iv), 6)})
    return pd.DataFrame(rows)


def test_smile_picker_falls_back_to_default_slices_when_the_maturity_grid_changes(app):
    assert _picker(app).value == [0, 15, 29]
    app.slider(key="ssvi_t").set_value((0.5, 3.0)).run()
    _ok(app)
    # The keyed picker used to keep labels of the old grid, which no longer existed: empty selection, no chart.
    assert _picker(app).value == [0, 15, 29]
    assert sum(name.startswith("Implied · ") for name in _trace_names(app)) == 3


def test_smile_picker_labels_tell_close_expiries_apart(app):
    days = (1, 2, 3, 4, 7, 8, 9, 10, 30, 60)
    _upload(app, _ssvi_quotes(days, np.linspace(0.9, 1.1, 9)))
    _ok(app)
    picker = _picker(app)
    assert picker.options == [f"{d}d" for d in days]  # were '0.00y', '0.01y', '0.01y', '0.01y', ...
    picker.set_value([2, 3, 6]).run()
    _ok(app)
    names = _trace_names(app)
    assert {"Implied · 3d", "Implied · 4d", "Implied · 9d"} <= set(names)


def test_ssvi_atm_vols_follow_the_sidebar_volatility(app):
    assert app.number_input(key="ssvi_short").value == 24.0 and app.number_input(key="ssvi_long").value == 20.0
    app.sidebar.number_input(key="vol").set_value(40.0).run()
    _ok(app)
    assert app.number_input(key="ssvi_short").value == 48.0 and app.number_input(key="ssvi_long").value == 40.0
    app.number_input(key="ssvi_long").set_value(30.0).run()
    app.slider(key="ssvi_rho").set_value(-0.5).run()
    assert app.number_input(key="ssvi_long").value == 30.0  # an edit sticks until the sidebar volatility moves
    app.segmented_control(key="surface_source").set_value("Market quotes (CSV)").run()
    app.segmented_control(key="surface_source").set_value("Parametric (SSVI)").run()
    _ok(app)
    assert app.number_input(key="ssvi_short").value == 48.0  # re-seeded, not reset to the 0.5% minimum
    assert not app.warning


def test_option_prices_build_a_surface_with_quote_markers_and_fit_quality(app):
    _upload(app, vol_surface.sample_quotes(100, 0.05, 0.0, prices=True))
    _ok(app)
    captions = " ".join(c.value for c in app.caption)
    assert "Inverted 104 option price(s)" in captions
    assert "spot (100.00), rate (5.00%) and dividend yield (0.00%)" in captions
    assert _metric(app, "Calendar-arbitrage points") == "0" and _metric(app, "Local vol defined") == "100%"
    names = _trace_names(app)
    assert {"Quotes · 30d", "Quotes · 273d", "Quotes · 2.00y"} <= set(names)
    summary = next(d.value for d in app.dataframe if "Fit RMSE (vol pts)" in d.value.columns)
    assert summary["Fit RMSE (vol pts)"].between(0.03, 0.4).all()
    assert any("the surface's implied vol is" in i.value for i in app.info)


def test_local_vol_defined_is_a_share_of_the_quoted_points(app):
    narrow = _ssvi_quotes((30, 60), np.linspace(0.95, 1.05, 7))
    wide = _ssvi_quotes((91, 182, 365), np.linspace(0.6, 1.5, 19))
    frame = pd.concat([narrow, wide])
    _upload(app, frame)
    _ok(app)
    clean, _ = vol_surface.parse_quotes(frame)
    surface = vol_surface.fit_market_surface(clean, 100, 0.05, 0.0)
    expected = np.isfinite(surface.local_vol).sum() / surface.quoted_points
    assert _metric(app, "Local vol defined") == f"{expected:.0%}"
    assert expected > 0.6  # the old denominator (the whole grid) gave a misleadingly low share
    assert any("outside each maturity's quoted strikes" in c.value for c in app.caption)


def test_interior_calendar_arbitrage_is_reported(app):
    quotes = vol_surface.sample_quotes(100, 0.05, 0.0)
    maturities = np.sort(quotes["Maturity"].unique())
    quotes.loc[quotes["Maturity"] == maturities[4], "ImpliedVol"] *= 0.8
    _upload(app, quotes)
    _ok(app)
    assert int(_metric(app, "Calendar-arbitrage points")) > 0  # was 0, with a local vol of exactly 0%
    assert _metric(app, "Local vol defined") == "75%"  # two of the eight maturities are left undefined


def test_parse_messages_stay_visible_when_the_fit_fails(app):
    frame = pd.DataFrame(
        {
            "Strike": [90, 95, 100, 105, 110, 100],
            "Days": [30, 30, 30, 30, 30, 60],
            "ImpliedVol": ["25%", "23%", "22%", "22%", "23%", "n/a"],
        }
    )
    _upload(app, frame)
    assert not app.exception
    assert any("two maturities" in e.value for e in app.error)
    captions = " ".join(c.value for c in app.caption)
    assert "Dropped 1 row(s)" in captions and "% signs" in captions


def test_quotes_far_from_the_sidebar_spot_are_flagged(app):
    # SPX-level strikes read against the default spot of 100; the header is the one the uploader's help names.
    quotes = vol_surface.sample_quotes(5000, 0.05, 0.0).rename(columns={"Maturity": "Maturity (years)"})
    _upload(app, quotes)
    _ok(app)
    assert any("far from the spot 100.00" in w.value for w in app.warning)


def test_sidebar_contract_at_the_first_expiry_of_the_sample_file_gets_an_implied_vol(app):
    # The sample stores 30 days as 0.0822 years; the sidebar's 30 days used to land just before it and the tab
    # said the surface gave no implied vol there.
    app.sidebar.number_input(key="days").set_value(30).run()
    _upload(app, vol_surface.sample_quotes(100, 0.05, 0.0))
    _ok(app)
    readout = [i.value for i in app.info if "the surface's implied vol is" in i.value]
    assert readout and "expiry 30 days" in readout[0]
    assert not any("outside the surface's quoted region" in c.value for c in app.caption)


def test_an_expiration_date_column_next_to_days_to_expiry_builds_a_surface(app):
    quotes = vol_surface.sample_quotes(100, 0.05, 0.0)
    quotes["DTE"] = (quotes.pop("Maturity") * 365).round().astype(int)
    quotes["Expiration"] = (pd.Timestamp("2026-10-13") + pd.to_timedelta(quotes["DTE"], unit="D")).dt.strftime(
        "%Y-%m-%d"
    )
    _upload(app, quotes)
    _ok(app)
    assert _metric(app, "Grid") == "8 × 41"
    assert any("Ignored 'Expiration'" in c.value for c in app.caption)


def test_the_excel_workbook_is_only_built_when_downloaded(app, monkeypatch):
    # Writing every quote into the workbook on every rerun made each click on a large upload take seconds.
    built = []
    writer = pd.ExcelWriter

    def counting_writer(*args, **kwargs):
        built.append(1)
        return writer(*args, **kwargs)

    monkeypatch.setattr(pd, "ExcelWriter", counting_writer)
    _upload(app, vol_surface.sample_quotes(100, 0.05, 0.0))
    app.selectbox(key="surface_colors").set_value("Plasma").run()
    _ok(app)
    assert not built
    surface = vol_surface.fit_market_surface(vol_surface.parse_quotes(vol_surface.sample_quotes())[0], 100, 0.04, 0.0)
    workbook = tab_surface._workbook(surface, vol_surface.surface_summary(surface), 100.0)
    assert built
    with zipfile.ZipFile(io.BytesIO(workbook)) as archive:
        sheets = archive.read("xl/workbook.xml").decode()
    assert all(f'name="{name}"' in sheets for name in ("ImpliedVol", "LocalVol", "Summary", "Quotes"))


def _maturity_column(header: str, values, as_years: bool) -> pd.DataFrame:
    params = vol_surface.SSVIParams()
    rows = []
    for value in values:
        t = value if as_years else value / 365
        for m in np.linspace(0.8, 1.2, 9):
            iv = np.sqrt(params.total_variance(np.log(m) - 0.05 * t, t)[0] / t)
            rows.append({"Strike": round(100 * m, 4), header: value, "ImpliedVol": round(float(iv), 6)})
    return pd.DataFrame(rows)


def test_whole_year_tenors_without_a_unit_ask_for_one_instead_of_becoming_days(app):
    # 1-10 years under 'Maturity' were read as 1-10 days and a ten-day surface was drawn.
    _upload(app, _maturity_column("Maturity", (1, 2, 3, 5, 7, 10), as_years=True))
    assert not app.exception
    assert any("could be years or days" in e.value and "'Maturity (years)'" in e.value for e in app.error)
    assert not any(m.label == "Grid" for m in app.metric)  # no surface
    _upload(app, _maturity_column("Maturity (years)", (1, 2, 3, 5, 7, 10), as_years=True))
    _ok(app)
    assert _metric(app, "Grid").startswith("6 × ")
    readout = [i.value for i in app.info if "the surface's implied vol is" in i.value]
    assert readout  # the sidebar's one-year contract lies inside a 1-10 year surface


def test_a_guessed_maturity_unit_is_a_visible_warning(app):
    # Fractional days (6.65 to 27.65) were read as years with nothing but a caption saying "(years)".
    _upload(app, _maturity_column("Maturity", (6.65, 13.65, 20.65, 27.65), as_years=False))
    for _ in range(2):  # the second run reads the parse messages back from the cache
        _ok(app)
        warnings = [w.value for w in app.warning]
        assert any("go up to 27.65, read as years; name the column 'Days'" in w for w in warnings)
        assert not any("27.65, read as years" in c.value for c in app.caption)
        app.selectbox(key="surface_colors").set_value("Plasma").run()


def test_fractional_days_up_to_ten_get_a_units_warning_and_a_fit_note_that_names_the_unit(app):
    # 2.65 to 9.65 days under 'Maturity' were read as years with no units hint, and the only message was a
    # poor-fit note blaming noisy quotes or a shape SVI cannot follow.
    _upload(app, _maturity_column("Maturity", (2.65, 4.65, 9.65), as_years=False))
    _ok(app)
    warnings = [w.value for w in app.warning]
    assert any("go up to 9.65, read as years; name the column 'Days'" in w and "too steep" in w for w in warnings)
    fit_notes = [w for w in warnings if w.startswith("The SVI smile")]
    assert fit_notes and all("the maturity unit" in w and "noisy" not in w for w in fit_notes)
    _upload(app, _maturity_column("Days", (2.65, 4.65, 9.65), as_years=False))
    _ok(app)
    assert not app.warning, [w.value for w in app.warning]


def test_flat_total_variance_at_the_last_expiry_is_not_reported_as_calendar_arbitrage(app):
    params = vol_surface.SSVIParams()
    rows = []
    for days in (30, 60, 91, 182, 365, 366):
        t = min(days, 365) / 365  # the 366-day quotes repeat the 365-day total variance at each forward moneyness
        for m in np.linspace(0.8, 1.2, 9):
            y = np.log(m) - 0.05 * t
            iv = np.sqrt(params.total_variance(y, t)[0] / (days / 365))
            rows.append({"Strike": 100 * np.exp(y + 0.05 * days / 365), "Days": days, "IV": iv})
    _upload(app, pd.DataFrame(rows).round(4))
    _ok(app)
    assert _metric(app, "Calendar-arbitrage points") == "0"  # was 31
    assert any("changes by less than the quotes' precision" in w.value for w in app.warning)
