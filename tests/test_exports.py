"""Exports carry their provenance (options_analyzer.ui.exports; UI spec 2.13, UI-28). The app's own download buttons
are checked in tests/test_data_policy.py; these are the builders."""

from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime

import pandas as pd
import pytest

from options_analyzer import __version__
from options_analyzer.ui import exports
from options_analyzer.ui.exports import SYNTHETIC, SYNTHETIC_SOURCE, Provenance

ASOF = datetime(2026, 10, 23, 16, 0)
PROV = Provenance(SYNTHETIC, SYNTHETIC_SOURCE, ASOF, "BSM · European", seed=0)


def test_the_four_csv_lines_of_the_spec(monkeypatch):
    # UI spec 2.13's example, field for field (with the app's version and commit, which the spec writes as abc1234).
    monkeypatch.setattr(exports, "commit", lambda: "abc1234")
    assert PROV.lines() == [
        "source: SYNTHETIC — not market data",
        "as_of: 2026-10-23T16:00-04:00 (cash close)",
        "model: BSM · European; dynamics: sticky strike; forwards: parity; fit: SVI arb-free",
        f"app: v{__version__} (abc1234); seed: 0",
    ]
    monkeypatch.setattr(exports, "commit", lambda: None)  # an installed package has no git checkout
    assert PROV.lines()[-1] == f"app: v{__version__}; seed: 0"


def test_the_commit_is_a_short_hash_or_nothing():
    sha = exports.commit()
    assert sha is None or re.fullmatch(r"[0-9a-f]{7,}", sha)


def test_file_names_carry_the_source_and_the_time():
    assert PROV.file_name("spx_pricing_chain", "csv") == "spx_pricing_chain_SYNTHETIC_20261023T1600.csv"
    # A valuation time with a zone is written in New York time.
    utc = datetime.fromisoformat("2026-12-18T21:00+00:00")
    assert Provenance("UPLOAD", "UPLOAD · a.csv", utc, "x").file_name("iv", "xlsx") == "iv_UPLOAD_20261218T1600.xlsx"


def test_a_csv_reads_back_with_pandas_and_with_the_apps_reader():
    frame = pd.DataFrame({"Strike": [5800.0, 5805.0], "Note": ["#1 listed", "b"]})
    data = exports.csv(frame, PROV)
    assert data.decode().splitlines()[:5] == [*(f"# {line}" for line in PROV.lines()), "Strike,Note"]
    pd.testing.assert_frame_equal(exports.read_csv(data), frame)  # a "#" inside a cell stays
    assert list(pd.read_csv(io.BytesIO(data), comment="#")["Strike"]) == [5800.0, 5805.0]


def test_the_workbook_has_the_provenance_sheet_first():
    pytest.importorskip("xlsxwriter")
    data = exports.xlsx({"Quotes": pd.DataFrame({"Strike": [5800.0]})}, PROV)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        book = archive.read("xl/workbook.xml").decode()
        strings = archive.read("xl/sharedStrings.xml").decode()
    assert re.findall(r'<sheet name="([^"]+)"', book) == ["Provenance", "Quotes"]
    assert all(value in strings for value in ("source", "SYNTHETIC — not market data", "as_of", "seed"))


def test_json_leads_with_its_provenance():
    body = json.loads(exports.json({"slices": [1, 2]}, PROV))
    assert list(body) == ["provenance", "slices"]
    assert body["provenance"]["source"] == SYNTHETIC_SOURCE and body["provenance"]["seed"] == "0"


def test_the_textbook_market_has_no_valuation_time():
    textbook = Provenance("TEXTBOOK", "TEXTBOOK — typed inputs, not market data", None, "BSM · closed form")
    assert textbook.fields()["as_of"].startswith("none (textbook inputs")
    assert re.fullmatch(r"x_TEXTBOOK_\d{8}T\d{4}\.csv", textbook.file_name("x", "csv"))


def test_a_builder_builds_only_when_called():
    calls = []

    def build(frame, prov):
        calls.append(1)
        return exports.csv(frame, prov)

    builder = exports.builder(build, pd.DataFrame({"a": [1]}), PROV)
    assert calls == []
    assert builder().startswith(b"# source: SYNTHETIC") and calls == [1]
