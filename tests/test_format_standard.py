"""The number standard of UI spec 2.5 (DK-4, UI-3): golden strings of every quantity, its rules over a wide sample,
the column registry and the table helper of spec 2.8.

The golden strings are the spec's own examples where the spec gives one (its 2.5 table), each computed from the
quantity's definition here rather than typed, so that a change of convention breaks the test that names it.
"""

from __future__ import annotations

import datetime as dt
import doctest
import importlib
import itertools
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from options_analyzer.ui import components
from options_analyzer.ui import format as fmt
from options_analyzer.ui.format import MINUS, MISSING

ROOT = Path(__file__).resolve().parents[1]
NB = fmt.NB_HYPHEN


def _sample(n: int = 3000, seed: int = 11) -> list[float]:
    """Values over 20 orders of magnitude of both signs, with the rounding edges at each precision."""
    rng = np.random.default_rng(seed)
    values = list(10.0 ** rng.uniform(-10, 10, n) * rng.choice([-1.0, 1.0], n))
    for decimals in range(5):
        step = 10.0**-decimals
        values += [s * k * step for s in (-1.0, 1.0) for k in (0.4999999, 0.5, 0.5000001, 1.0, 9.5, 99.5)]
    return [*values, 0.0, -0.0, 5e-324, -5e-324]


# --- golden strings, quantity by quantity (the rows of the spec's table) --------------------------------------------


def test_levels_strikes_and_premiums():
    assert fmt.level(5815.2249) == "5,815.22"  # spot, forward, index level
    assert fmt.strike(5815.0, 5.0) == "5,815"  # the chain's increment: whole
    assert fmt.strike(582.5, 2.5) == "582.5"
    assert fmt.strike(5812.37, 5.0) == "5,812.37"  # a typed strike off the grid keeps its decimals
    assert fmt.premium(98.570661) == "98.57"  # per share, 2 dp
    assert fmt.premium(98.570661, 6) == "98.570661"  # diagnostics
    assert fmt.signed_sci(-3.1e-7) == "−3.1e−7"
    assert fmt.signed_sci(4.44e-16, 3) == "+4.44e−16"
    assert fmt.premium(-22.4) == "−22.40"


def test_money_and_signed_losses():
    # Money unsigned with "cr"/"db" as words; P&L and every loss signed, losses negative, no parentheses.
    assert f"{fmt.money(2240.0, 0)} cr" == "$2,240 cr"
    assert fmt.signed(1234.0, 0, prefix="$") == "+$1,234"
    assert fmt.signed(-7760.0, 0, prefix="$") == "−$7,760"  # MAX LOSS
    assert fmt.signed(-7728.4, 0, prefix="$") == "−$7,728"  # WORST STRESS
    assert fmt.signed(-1520.55, 0, prefix="$") == "−$1,521"  # VaR, a loss: negative
    assert fmt.signed(-0.4, 0, prefix="$") == "$0"  # rounds to zero: no sign
    assert fmt.money_short(-19_525.0, signed=True) == "-$19,525"  # tiles: st.metric reads only an ASCII minus


def test_vols_rates_and_basis_points():
    assert fmt.vol(0.15464) == "15.46%"
    assert fmt.vol_change(0.0035) == "+0.35 vp"
    assert fmt.vol_change(-0.0007) == "−0.07 vp"  # mid vs fit
    assert fmt.vol_change(-0.0421) == "−4.21 vp"  # 25Δ RR: put skew
    assert fmt.pct(0.0475) == "4.75%"
    assert fmt.bp(0.0045) == "+45 bp"  # the box spread
    assert fmt.bp(0.000005) == "+0.05 bp"
    assert fmt.bp(-0.00031) == "−3.1 bp"
    assert fmt.bp(0.0) == "0 bp"


def test_deltas_and_position_delta_in_futures():
    assert fmt.greek(0.508, "delta", digits=3, unit=False) == "+0.508"
    assert [fmt.delta_label(d, side=False) for d in (0.16, 0.25)] == ["16Δ", "25Δ"]
    assert [fmt.delta_label(d) for d in (-0.25, 0.5, 0.25)] == ["25Δp", "ATM", "25Δc"]
    # Position delta, spec 2.5: −$19,525 · −0.07 ES at S 5,800, the 18-Dec ES future at F 5,830.63 (carry factor
    # F/S); ES $50 a point, MES $5 (CME). The spec's −0.67 MES is the same delta.
    spot, carry = 5800.0, 5830.63 / 5800.0
    delta = -19_525.0 / spot  # $Δ = Δ · S
    assert f"{fmt.signed(delta * spot, 0, prefix='$')} · {fmt.futures_equiv(delta, carry_factor=carry)}" == (
        "−$19,525 · −0.07 ES"
    )
    assert fmt.futures_equiv(delta, carry_factor=carry, contract="MES") == "−0.67 MES"
    assert fmt.futures_contracts(50.0 * carry, carry_factor=carry) == pytest.approx(1.0, rel=1e-15)
    assert fmt.futures_equiv(0.001) == "0.00 ES"


def test_gamma_pnl_one_percent_and_its_help():
    # Γ P&L 1% ($) = ½Γ(0.01S)² × units, the convexity P&L of a 1% move; its help gives ΓS²/100 × units, the change
    # in $Δ per 1%, 200 times larger (spec 2.5).
    spot, units = 5800.0, 100.0
    gamma = -68.0 / (0.5 * (0.01 * spot) ** 2 * units)
    pnl = fmt.dollar_gamma_1pct(gamma, spot, units)
    change = gamma * spot**2 / 100 * units
    assert fmt.signed(pnl, 0, prefix="$") == "−$68"
    assert fmt.signed(change, 0, prefix="$") == "−$13,600"
    assert change / pnl == pytest.approx(200.0, rel=1e-12)
    assert fmt.greek(0.000016137, "gamma", digits=3, unit=False) == "+0.0000161"  # per option: 3 significant digits


def test_vega_theta_rho():
    assert fmt.signed(fmt.dollar_vega(-306.4, 100.0), 0, prefix="$") == "−$306"
    assert fmt.signed(fmt.dollar_theta(41.2 * 365 / 100, 100.0), 0, prefix="$") == "+$41"
    assert fmt.signed(fmt.dollar_rho(31.0, 100.0), 0, prefix="$") == "+$31"


def test_probabilities_carry_their_basis():
    assert fmt.probability_text(0.735, "RN smile") == "73.5% RN smile"
    assert fmt.probability_text(0.1685, "flat vol") == "16.9% flat vol"
    assert fmt.probability_text(0.99962, "view") == ">99.9% view"
    assert fmt.probability_text(0.0003) == "<0.1%"
    assert fmt.probability_text(1.0, sure=True) == "100.0%" and fmt.probability_text(0.0, never=True) == "0.0%"
    with pytest.raises(ValueError, match="basis"):
        fmt.probability_text(0.5, "smile")


def test_moneyness_quantities_and_counts():
    assert fmt.moneyness(5670.0, 5815.22) == "97.5%"
    assert fmt.moneyness(5670.0, 5815.22, log=True) == "−0.025"
    assert fmt.quantity(1) == "+1" and fmt.quantity(-2) == "−2"
    assert fmt.number(0.07597, 6) == "0.075970"  # year fraction, diagnostics
    assert fmt.number(2363, 0) == "2,363"


def test_dates_times_and_sessions():
    assert fmt.date(dt.date(2026, 11, 20)) + " AM" == "20-Nov-26 AM"
    assert fmt.timestamp(dt.datetime(2026, 10, 23, 16, 0)) == f"23{NB}Oct{NB}26 16:00 ET"
    eastern_winter = dt.datetime(2026, 12, 18, 14, 30, tzinfo=dt.timezone.utc)
    assert fmt.timestamp(eastern_winter) == f"18{NB}Dec{NB}26 09:30 ET"  # EST, UTC−5
    assert fmt.session_label(10, dt.date(2026, 11, 6)) == f"T+10 · Fri 06{NB}Nov{NB}26"
    minutes = (27 + 17.9 / 24) * 1440
    assert [fmt.time_left(55.7 * 1440), fmt.time_left(minutes), fmt.time_left(390)] == ["55d", "27d 17h", "6h 30m"]
    assert fmt.time_left(135, expiry_day=True) == "0DTE 2h 15m"


# --- the edge kinds -------------------------------------------------------------------------------------------------

#: Every function of the standard, with a call that shows a value.
FUNCTIONS = {
    "level": fmt.level,
    "strike": lambda x: fmt.strike(x, 5.0),
    "premium": fmt.premium,
    "signed_sci": fmt.signed_sci,
    "vol": fmt.vol,
    "vol_change": fmt.vol_change,
    "bp": fmt.bp,
    "delta_label": fmt.delta_label,
    "futures_equiv": fmt.futures_equiv,
    "moneyness": lambda x: fmt.moneyness(x, 100.0),
    "time_left": fmt.time_left,
    "probability_text": fmt.probability_text,
    "signed": fmt.signed,
    "quantity": fmt.quantity,
}


@pytest.mark.parametrize("name", list(FUNCTIONS))
def test_missing_values_read_as_the_en_dash(name):
    for missing in (float("nan"), None):
        assert FUNCTIONS[name](missing) == MISSING, (name, missing)


def test_timestamp_of_nothing_is_the_en_dash():
    assert fmt.timestamp(None) == MISSING


@pytest.mark.parametrize("name", ["premium", "vol", "vol_change", "bp", "futures_equiv", "signed", "strike", "level"])
def test_one_minus_sign_and_no_negative_zero(name):
    function = FUNCTIONS[name]
    for value in _sample():
        text = function(value)
        assert "-" not in text, (name, value, text)  # U+2212 only
        lead = re.match(f"([{MINUS}+]?)[$]?([0-9.,]+)", text)
        assert lead, (name, value, text)
        sign, digits = lead.groups()
        if sign:  # never −0.00 or +0.00: a signed value shows a non-zero digit
            assert digits.strip("0.,"), (name, value, text)
        assert (sign == MINUS) == (value < 0 and bool(digits.strip("0.,"))), (name, value, text)


@pytest.mark.parametrize("name", ["vol_change", "bp", "futures_equiv", "signed"])
def test_signed_values_line_up_whatever_their_sign(name):
    function = FUNCTIONS[name]
    for value in _sample():
        assert len(function(value)) == len(function(-value)), (name, value)


def test_time_left_is_floored_and_monotone():
    minutes = np.concatenate([np.linspace(0.0, 3 * 1440, 4001), np.linspace(29 * 1440, 31 * 1440, 501)])
    order = {"expired": -1}

    def rank(text: str) -> float:
        if text in order:
            return order[text]
        days = re.search(r"(\d+)d", text)
        hours = re.search(r"(\d+)h", text)
        mins = re.search(r"(\d+)m", text)
        return sum(int(m.group(1)) * k for m, k in ((days, 1440), (hours, 60), (mins, 1)) if m)

    ranks = [rank(fmt.time_left(m)) for m in minutes]
    assert all(b >= a for a, b in itertools.pairwise(ranks))
    assert all(r <= m + 1e-6 for r, m in zip(ranks, minutes, strict=True))  # never more time than is left


def test_probabilities_never_round_to_certainty():
    for p in np.linspace(0.999, 1.0, 101):
        assert fmt.probability_text(p) in {">99.9%", "99.9%"}
    for p in np.linspace(0.0, 0.001, 101):
        assert fmt.probability_text(p) in {"<0.1%", "0.1%"}


def test_the_probability_function_moved_from_the_strategy_tab():
    from options_analyzer.ui import tab_strategy

    assert tab_strategy.probability_text is fmt.probability_text


# --- the column registry and the table helper (spec 2.8) ------------------------------------------------------------


def test_every_column_kind_has_a_format_and_a_unit_in_its_header():
    for kind, spec in components.COLUMN_KINDS.items():
        config = components.col(kind, "Value", help="What it is.", digits=2)
        assert config["help"] == "What it is."
        if spec.unit:
            assert config["label"] == f"Value ({spec.unit})", kind
        if spec.format is not None:
            printf = config["type_config"]["format"]
            assert re.fullmatch(r"%[+,]?(\.\d+)?[dfg]", printf), (kind, printf)
            assert "+" not in printf or kind == "quantity"  # printf's "+" also signs zero


def _sprintf(printf: str, value: float) -> str:
    """What Streamlit's NumberColumn shows for ``value`` under ``printf`` (its sprintf-js, 1.65 and 1.66): the sign
    from ``value >= 0`` (so −0.0 shows no sign), ``toFixed`` for f, ``Number(toPrecision)`` for g, ``parseInt`` for d,
    and the "," flag's thousands separators in the integer part. Golden values avoid exact binary ties, where
    JavaScript's toFixed and Python's rounding may differ."""
    flags, precision, kind = re.fullmatch(r"%([+,]*)(?:\.(\d+))?([dfg])", printf).groups()
    size = abs(value)
    if kind == "f":
        text = f"{size:.{int(precision)}f}"
    elif kind == "g":
        text = repr(float(f"{size:.{int(precision)}g}"))
        text = text[:-2] if text.endswith(".0") else text
    else:
        text = str(int(size))
    if "," in flags:
        whole, _, fraction = text.partition(".")
        text = f"{int(whole):,}" + (f".{fraction}" if fraction else "")
    sign = "-" if value < 0 else ("+" if "+" in flags else "")
    return sign + text


#: One golden cell per column kind: (kind, the column's values in display units, the header for label "X", the
#: text of the first cell). Signed losses read with the hyphen-minus inside st.dataframe (spec 2.8).
GOLDEN_CELLS = [
    ("level", [5815.2249], "X", "5,815.22"),
    ("strike", [5815.0, 5820.0], "X", "5,815"),
    ("strike", [582.5, 585.0], "X", "582.5"),
    ("premium", [98.570661], "X", "98.57"),
    ("premium_diag", [98.570661], "X", "98.570661"),
    ("money", [2240.0, 150.0], "X ($)", "2,240"),  # one precision from the largest value: whole from $1,000
    ("money", [224.0, 15.5], "X ($)", "224.00"),
    ("pnl", [-7760.0, 1234.4], "X ($)", "-7,760"),  # MAX LOSS: a loss is negative
    ("pnl", [-0.004, 12.0], "X ($)", "0.00"),  # rounds to zero: no sign
    ("dollar_delta", [-19525.3], "X ($)", "-19,525"),
    ("futures", [-0.0673], "X (ES)", "-0.07"),  # −$19,525 of delta at S 5,800 is −0.07 ES
    # A position's Greeks in whole dollars, spec 2.5's examples; a contract's vega per share in cents.
    ("gamma_pnl", [-68.2], "X ($)", "-68"),
    ("vega", [-306.4], "X ($/vp)", "-306"),
    ("theta", [41.2], "X ($/day)", "41"),
    ("rho", [31.0], "X ($/1%)", "31"),
    ("vega_share", [6.3712], "X ($/vp)", "6.37"),
    ("vol", [15.4612], "X (%)", "15.46"),
    ("vol_change", [-0.0701], "X (vp)", "-0.07"),
    ("rate", [4.75], "X (%)", "4.75"),
    ("bp", [45.2, 3.0], "X (bp)", "45"),
    ("bp", [4.5], "X (bp)", "4.5"),
    ("bp", [0.05], "X (bp)", "0.05"),
    ("delta", [0.50791], "X", "0.508"),
    ("gamma", [0.093612], "X", "0.0936"),
    ("greek", [1.23456e-4], "X", "0.000123"),
    ("probability", [73.49], "X (%)", "73.5"),
    ("probability", [99.96], "X (%)", "99.9"),  # never a certainty: spec 2.5's ">99.9%"
    ("probability", [0.03], "X (%)", "0.1"),  # "<0.1%"
    ("probability", [100.0], "X (%)", "100.0"),  # an exact certainty stays
    ("share", [99.04], "X (%)", "99.0"),
    ("share", [99.97], "X (%)", "99.9"),
    ("moneyness", [97.502], "X (%)", "97.5"),
    ("log_moneyness", [-0.02525], "X", "-0.025"),
    ("quantity", [-2.0], "X", "-2"),
    ("quantity", [1.0], "X", "+1"),
    ("count", [2363.0], "X", "2,363"),
    ("days", [55.7], "X", "56"),
    ("year_fraction", [0.07597], "X", "0.075970"),
]


def test_golden_cells_of_every_column_kind(monkeypatch):
    numeric = {k for k, spec in components.COLUMN_KINDS.items() if spec.format is not None}
    assert {kind for kind, *_ in GOLDEN_CELLS} == numeric  # every number kind has a golden cell
    for kind, values, header, cell in GOLDEN_CELLS:
        data, options = _drawn(monkeypatch, pd.DataFrame({"X": values}), {"X": kind})
        config = options["column_config"]["X"]
        assert config["label"] == header, kind
        assert _sprintf(config["type_config"]["format"], data["X"].iloc[0]) == cell, (kind, values)


def test_a_data_dependent_column_can_take_a_fixed_precision(monkeypatch):
    # Money is in cents below $1,000; a caller may fix whole dollars for a column of small amounts.
    data, options = _drawn(monkeypatch, pd.DataFrame({"G": [-68.2, 12.4]}), {"G": "pnl"}, digits={"G": 0})
    assert _sprintf(options["column_config"]["G"]["type_config"]["format"], data["G"].iloc[0]) == "-68"


def test_a_probability_short_of_certainty_says_so_in_its_help(monkeypatch):
    # Spec 2.5: ">99.9%" and "<0.1%" at the ends. A number column cannot write them, so the cell is held at 99.9 or
    # 0.1 and the help explains it; a column with nothing held keeps its help as written.
    frame = pd.DataFrame({"P": [99.96, 0.03, 50.0, 0.0, 100.0], "Q": [73.49, 26.51, 50.0, 0.0, 100.0]})
    data, options = _drawn(monkeypatch, frame, {"P": "probability", "Q": "probability"})
    assert list(data["P"]) == [99.9, 0.1, 50.0, 0.0, 100.0]
    assert "Above 99.9% shows as 99.9" in options["column_config"]["P"]["help"]
    assert options["column_config"]["Q"]["help"] == "About Q."


def test_the_sprintf_model_reads_streamlits_formats():
    assert _sprintf("%,.2f", -1234.5) == "-1,234.50" and _sprintf("%+d", 0.0) == "+0"
    assert _sprintf("%,.0f", -0.0) == "0"  # JavaScript's -0 >= 0 is true: no sign


def test_a_label_that_names_its_unit_keeps_it():
    assert components.col("vol_change", "Mid vs fit (vp)", help="h")["label"] == "Mid vs fit (vp)"
    assert components.col("dollar_delta", "$Δ", help="h")["label"] == "$Δ"
    assert components.col("vega", "Vega", help="h")["label"] == "Vega ($/vp)"


def test_a_column_needs_help_and_a_known_kind():
    with pytest.raises(ValueError, match="help"):
        components.col("vol", "IV", help=" ")
    with pytest.raises(ValueError, match="unknown column kind"):
        components.col("percentish", "IV", help="h")


def _drawn(monkeypatch, frame, kinds, **kwargs):
    calls = []
    monkeypatch.setattr(components.st, "dataframe", lambda data, **kw: calls.append((data, kw)))
    components.table(frame, kinds, {c: f"About {c}." for c in frame}, **kwargs)
    [(data, options)] = calls
    return data, options


def test_the_table_helper_follows_the_tables_standard(monkeypatch):
    frame = pd.DataFrame(
        {
            "Strike": [5800.0, 5805.0, 5810.0],
            "IV": [15.4612, -0.0001, np.nan],
            "P&L": [-1234.567, 12.0, -0.001],
            "Edge": [0.004, -0.004, 1.0],
        }
    )
    kinds = {"Strike": "strike", "IV": "vol", "P&L": "pnl", "Edge": "bp"}
    data, options = _drawn(monkeypatch, frame, kinds, key="tbl_demo")
    config = options["column_config"]
    assert options["row_height"] == 28 and options["hide_index"] and options["height"] == "content"
    assert options["placeholder"] == MISSING and options["key"] == "tbl_demo"
    assert config["Strike"]["pinned"] is True and all(config[c]["pinned"] is not True for c in ("IV", "P&L", "Edge"))
    assert all(config[c]["help"] for c in frame)
    formats = {c: config[c]["type_config"]["format"] for c in frame}
    # One precision per column: strikes whole, P&L whole dollars from $1,000, bp at 1 dp (largest 1.0 bp rounds to 1).
    assert formats == {"Strike": "%,.0f", "IV": "%.2f", "P&L": "%,.0f", "Edge": "%,.1f"}
    # Values sent at that precision, with no negative zero for printf to write as "-0.00".
    assert list(data["IV"].iloc[:2]) == [15.46, 0.0] and math.copysign(1.0, data["IV"].iloc[1]) == 1.0
    assert list(data["P&L"]) == [-1235.0, 12.0, 0.0] and math.copysign(1.0, data["P&L"].iloc[2]) == 1.0
    assert config["P&L"]["label"] == "P&L ($)" and config["IV"]["label"] == "IV (%)"


def test_a_long_table_scrolls_at_fifteen_rows(monkeypatch):
    frame = pd.DataFrame({"Strike": np.arange(5000.0, 5100.0, 5.0)})
    _, options = _drawn(monkeypatch, frame, {"Strike": "strike"})
    assert options["height"] == components.table_height(15)
    _, options = _drawn(monkeypatch, frame.head(15), {"Strike": "strike"})
    assert options["height"] == "content"


def test_the_table_helper_refuses_a_column_without_kind_or_help(monkeypatch):
    monkeypatch.setattr(components.st, "dataframe", lambda *a, **k: None)
    frame = pd.DataFrame({"Strike": [5800.0], "IV": [15.0]})
    with pytest.raises(ValueError, match="kind and a help"):
        components.table(frame, {"Strike": "strike"}, {"Strike": "s", "IV": "i"})
    with pytest.raises(ValueError, match="kind and a help"):
        components.table(frame, {"Strike": "strike", "IV": "vol"}, {"Strike": "s"})


def test_significant_digit_columns_round_to_three_digits(monkeypatch):
    frame = pd.DataFrame({"Vanna": [1.23456e-4, -9.87654e-7, 0.0]})
    data, options = _drawn(monkeypatch, frame, {"Vanna": "greek"})
    assert options["column_config"]["Vanna"]["type_config"]["format"] == "%.3g"
    np.testing.assert_array_equal(data["Vanna"], [1.23e-4, -9.88e-7, 0.0])


def test_every_kind_of_missing_value_reads_as_the_en_dash():
    # Spec 2.5: never "nan". A NumPy float32 NaN is not a Python float, pandas' NA has no truth value and NaT is a
    # datetime: each once showed "nan" or raised.
    for missing in (None, math.nan, np.float32("nan"), np.float64("nan"), pd.NA):
        for writer in (fmt.vol, fmt.premium, fmt.vol_change, fmt.bp, fmt.signed_sci, fmt.time_left):
            assert writer(missing) == MISSING, (writer.__name__, missing)
        assert fmt.strike(missing, 5.0) == MISSING
    assert fmt.timestamp(pd.NaT) == MISSING and fmt.timestamp(None) == MISSING
    assert fmt.time_left(math.inf) == MISSING
    assert fmt.vol(np.float32(0.1552)) == "15.52%"  # a float32 that is there still writes
    # A strike without a known increment keeps its own decimals.
    assert fmt.strike(5815.0, math.nan) == "5,815" and fmt.strike(582.5, None) == "582.5"


# --- the lint the tabs will pass once they adopt the standard --------------------------------------------------------

#: Hand formatting in a tab, any of:
#:
#: * an f-string or str.format spec with grouping, a precision, a width or a number type (``:.2f``, ``:,.0f``,
#:   ``:+.1%``, ``:.2g``, ``:,``, ``:5d``); a date's ``:%d %b`` is not one. No space may follow the colon, so that
#:   a dict literal (ruff writes ``{"a": 1.5}``) is not read as one;
#: * ``format(x, ",.2f")`` and the printf operator (``"%.2f" % x``);
#: * a printf column format typed in place (``format="%.4f"``);
#: * a hand-made minus sign (``.replace("-", "−")``).
HAND_FORMAT = re.compile(
    "|".join(
        (
            r"\{[^{}:]*:[+\-]?#?0?\d*[,_.][^{}]*\}",  # grouping or a precision
            r"\{[^{}:]*:[+\-]?#?0?\d*[bdeEfFgGnoxX%]\}",  # a width, a sign or a number type alone
            r"\bformat\([^()]*,\s*[\"'][^\"']*[\"']\)",  # the format() builtin
            r"[\"'][^\"']*%[+\- #0,]*\d*(?:\.\d+)?[dfeEgGi][^\"']*[\"']\s*%\s*[\w(]",  # "%.2f" % x
            r"format=\"%[^\"]*\"",
            r"\.replace\(\"-\", \"(−|\\\\u2212)\"\)",
        )
    )
)


@pytest.mark.xfail(strict=True, reason="tabs adopt the number standard (DK-4)")
def test_no_tab_formats_numbers_by_hand():
    offenders = {}
    for path in sorted((ROOT / "options_analyzer" / "ui").glob("tab_*.py")):
        hits = [n for n, line in enumerate(path.read_text("utf-8").splitlines(), 1) if HAND_FORMAT.search(line)]
        if hits:
            offenders[path.name] = hits
    assert not offenders, offenders


def test_the_lint_catches_hand_formatting():
    caught = (
        'f"{x:.2f}"',
        'f"{x:,.0f} cr"',
        'f"{p:.1%}"',
        'NumberColumn(format="%.4f")',
        's.replace("-", "−")',
        'f"{res.last_step:.2g}"',  # tab_pricing's PDE step, before the standard
        'f"{n:,} paths"',  # a count: format.number(n, 0)
        'f"{n:5d}"',
        'f"{x:+d}"',
        'f"{rows[i].iv:.4f}"',
        'format(x, ",.2f")',
        '"%.2f" % x',
        '"%d paths" % (n,)',
    )
    for line in caught:
        assert HAND_FORMAT.search(line), line
    passed = (
        "fmt.vol(x)",
        'col("vol", "IV", help="h")',
        'f"{name}"',
        'f"{as_of:%d %b %Y}"',
        'f"{t:%H:%M}"',
        'f"{name!r}"',
        '{"Strike": 5800.0, "IV": 15.5}',  # a dict literal
        "{k: 0.5 for k in keys}",
        'st.number_input("Strike", format=fmt.STRIKE)',
        '"Spot moved 5% today"',
    )
    for line in passed:
        assert not HAND_FORMAT.search(line), line


# --- the docstring examples -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [
        "options_analyzer.ui.format",
        "options_analyzer.ui.components",
        "options_analyzer.ui.whatif",
        "options_analyzer.ui.exports",
        "options_analyzer.vol_surface.published",
    ],
)
def test_the_examples_in_the_docstrings_run(module):
    result = doctest.testmod(importlib.import_module(module), optionflags=doctest.ELLIPSIS)
    assert result.failed == 0
