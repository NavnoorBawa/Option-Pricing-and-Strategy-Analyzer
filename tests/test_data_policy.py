"""The data and licensing policy (docs/data_policy.md), enforced (DT-6).

* No module of the package, and not the entry point, names a Cboe or Yahoo quote host or imports yfinance; a module
  that a user opts into for local use would be listed in ``OPT_IN_MODULES`` and must stay off the default import path.
* The default code path reaches no other machine. Importing every default module in a fresh interpreter, and
  rendering every tab of the app with the proxies pointed at a dead port, make no host-name look-up, no connection
  and no datagram to any host but this one, and no connection to the dead proxy. Every attempt is recorded before the
  test suite's network guard blocks it, so a fetch that ignores the proxy settings (http.client, aiohttp,
  ``requests`` with ``trust_env=False``) is caught even when the code swallows the error, and so is a host built at
  run time, which no grep can see.
* Every file in tests/fixtures, and every file the package ships in options_analyzer/data, starts with a provenance
  header; no quote or price file is real third-party data, and package data is public domain with its source.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import http.client
import importlib.util
import io
import ipaddress
import itertools
import json
import os
import re
import socket
import subprocess
import sys
import textwrap
import time
import urllib.request
import zipfile
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "options_analyzer"
FIXTURES = ROOT / "tests" / "fixtures"
#: Reference data the package ships (the planned public-domain Treasury curve, roadmap item DT-3).
PACKAGE_DATA = PACKAGE / "data"
POLICY = ROOT / "docs" / "data_policy.md"
#: Hosts whose quotes the public app must never fetch (Cboe's delayed-quote JSON and Yahoo Finance).
FORBIDDEN_HOSTS = ("cdn.cboe.com", "finance.yahoo.com")
#: Modules a user may run on their own machine to fetch data; none exists yet (roadmap item DT-7).
OPT_IN_MODULES: frozenset[str] = frozenset()
PROVENANCES = ("fabricated", "synthetic", "published-facts", "public-domain")
DATA_SUFFIXES = {".csv", ".tsv", ".parquet", ".feather", ".xlsx", ".xls", ".json", ".dat"}
#: Words of a column name that mark quotes or prices (after splitting camelCase and punctuation, plural "s" removed).
PRICE_WORDS = frozenset({"ask", "bid", "close", "iv", "last", "mark", "mid", "price", "quote", "strike", "volatility"})


def _default_path_sources() -> list[Path]:
    """The entry point and every module of the package except the opt-in ones."""
    modules = [ROOT / "main.py", *sorted(PACKAGE.rglob("*.py"))]
    return [path for path in modules if _module_name(path) not in OPT_IN_MODULES]


def _module_name(path: Path) -> str:
    parts = path.relative_to(ROOT).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def test_no_cboe_or_yahoo_url_and_no_yfinance_import_on_the_default_code_path():
    """The grep test: every string and import of the package and the entry point, comments included."""
    offences = []
    for path in _default_path_sources():
        text = path.read_text(encoding="utf-8")
        for host in FORBIDDEN_HOSTS:
            if host in text.lower():
                offences.append(f"{path.relative_to(ROOT)} names {host}")
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif (
                isinstance(node, ast.Call)  # __import__("yfinance"), importlib.import_module("yfinance")
                and getattr(node.func, "id", getattr(node.func, "attr", "")) in ("__import__", "import_module")
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                names = [str(node.args[0].value)]
            else:
                continue
            where = path.relative_to(ROOT)
            offences += [f"{where} imports {name}" for name in names if name.split(".")[0] == "yfinance"]
    assert offences == []


# --- The network, at run time -------------------------------------------------------------------------------------


def _is_loopback(host: object) -> bool:
    """True for this machine: no host, ``localhost``, or a loopback or unspecified address."""
    if host is None:
        return True
    if isinstance(host, (bytes, bytearray)):
        host = host.decode("ascii", "replace")
    name = str(host).strip("[]").split("%", 1)[0].lower()
    if name in ("", "localhost"):
        return True
    try:
        address = ipaddress.ip_address(name)
    except ValueError:
        return False  # a host name: looking it up would ask DNS
    return address.is_loopback or address.is_unspecified


def _dead_proxy(monkeypatch: pytest.MonkeyPatch) -> int:
    """Point every proxy variable at a local port nothing listens on, and return the port."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead = probe.getsockname()[1]
    for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.setenv(name, f"http://127.0.0.1:{dead}")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    return dead


def _goes_outside(address: object, dead_port: int) -> bool:
    """An (host, port, ...) address of another machine, or of the dead proxy."""
    if not (isinstance(address, tuple) and address):
        return False
    return not _is_loopback(address[0]) or (len(address) > 1 and address[1] == dead_port)


def _record_network(monkeypatch: pytest.MonkeyPatch, *, dead_port: int) -> list[str]:
    """Record every look-up of another host and every connection or datagram to another host or to ``dead_port``.

    The recorders wrap the test suite's network guard (tests/conftest.py), so each attempt is listed before the guard
    blocks it, whatever the code then does with the error.
    """
    attempts: list[str] = []

    def lookup(function: Callable[..., Any], name: str) -> Callable[..., Any]:
        @functools.wraps(function)
        def recording(*args: Any, **kwargs: Any) -> Any:
            host = args[0] if args else kwargs.get("host")
            if not _is_loopback(host):
                attempts.append(f"{name}({host!r})")
            return function(*args, **kwargs)

        return recording

    def connection(method: Callable[..., Any], name: str) -> Callable[..., Any]:
        @functools.wraps(method)
        def recording(sock: socket.socket, *args: Any) -> Any:
            address = args[-1] if args else None  # connect(address), sendto(data[, flags], address)
            if sock.family in (socket.AF_INET, socket.AF_INET6) and _goes_outside(address, dead_port):
                attempts.append(f"{name}({address!r})")
            return method(sock, *args)

        return recording

    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
        monkeypatch.setattr(socket, name, lookup(getattr(socket, name), name))
    for name in ("connect", "connect_ex", "sendto"):
        monkeypatch.setattr(socket.socket, name, connection(getattr(socket.socket, name), name))
    return attempts


def test_the_recorder_sees_fetches_that_ignore_the_proxy_or_swallow_their_error(monkeypatch):
    """The checks below would pass with a Cboe fetch on the default path unless the recorder sees it: a client that
    ignores the proxy settings (http.client) inside a try/except, a host assembled at run time, and a client that
    honours them (urllib) and so goes to the dead proxy."""
    dead = _dead_proxy(monkeypatch)
    attempts = _record_network(monkeypatch, dead_port=dead)
    host = "cdn." + "cboe" + ".com"
    with contextlib.suppress(Exception):
        http.client.HTTPSConnection(host, timeout=1).request("GET", "/api/global/delayed_quotes/options/_SPX.json")
    assert attempts == [f"getaddrinfo({host!r})"]
    with contextlib.suppress(Exception):
        urllib.request.urlopen("https://finance." + "yahoo" + ".com/quote/SPY/options", timeout=1)
    assert len(attempts) == 2 and str(dead) in attempts[1]


def test_default_app_renders_every_tab_with_the_network_unreachable(monkeypatch, new_app):
    """DT-6: with the proxies pointed at a dead local port, the app renders every tab from its own data, quickly, and
    neither looks up nor connects to any other machine, directly or through the proxy."""
    dead = _dead_proxy(monkeypatch)
    attempts = _record_network(monkeypatch, dead_port=dead)
    at = new_app()
    start = time.perf_counter()
    at.run()
    elapsed = time.perf_counter() - start
    assert not at.exception, at.exception
    assert attempts == []
    assert len(at.tabs) >= 6
    assert elapsed < 60  # a hung connection would wait for the AppTest timeout (120 s)


def test_the_default_import_path_loads_no_opt_in_module_and_reaches_no_other_machine():
    """In a fresh interpreter with the proxies at a dead port, importing the entry point and every default module of
    the package loads no opt-in module and no yfinance, and an audit hook (which sees every socket call, in Python or
    in C) records and refuses any look-up of, or connection to, another machine."""
    code = textwrap.dedent(
        """
        import importlib, ipaddress, json, pkgutil, socket, sys

        attempts = []
        dead = int(sys.argv[1])
        opt_in = set(json.loads(sys.argv[2]))

        def loopback(host):
            if host is None:
                return True
            if isinstance(host, (bytes, bytearray)):
                host = host.decode("ascii", "replace")
            name = str(host).strip("[]").split("%", 1)[0].lower()
            if name in ("", "localhost"):
                return True
            try:
                address = ipaddress.ip_address(name)
            except ValueError:
                return False
            return address.is_loopback or address.is_unspecified

        def hook(event, args):
            if event in ("socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"):
                target, outside = args[0], not loopback(args[0])
            elif event in ("socket.connect", "socket.sendto"):
                address = args[1]
                if not (isinstance(address, tuple) and address):
                    return
                target = address
                outside = not loopback(address[0]) or (len(address) > 1 and address[1] == dead)
            else:
                return
            if outside:
                attempts.append(f"{event}({target!r})")
                raise OSError("the network is off limits on the default code path")

        sys.addaudithook(hook)
        import main
        import options_analyzer
        for module in pkgutil.walk_packages(options_analyzer.__path__, "options_analyzer."):
            if module.name not in opt_in:
                importlib.import_module(module.name)
        ours = ("options_analyzer", "yfinance")
        loaded = sorted(name for name in sys.modules if name.split(".")[0] in ours)
        print(json.dumps({"loaded": loaded, "attempts": attempts}))
        """
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead = probe.getsockname()[1]
    env = {name: value for name, value in os.environ.items() if not name.lower().endswith("_proxy")}
    env.update(dict.fromkeys(("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"), f"http://127.0.0.1:{dead}"))
    env["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        [sys.executable, "-c", code, str(dead), json.dumps(sorted(OPT_IN_MODULES))],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    report = json.loads(result.stdout.strip().splitlines()[-1])
    loaded = set(report["loaded"])
    assert report["attempts"] == []
    assert {"options_analyzer.market_data.adapters", "options_analyzer.ui.app", "options_analyzer.clock"} <= loaded
    assert not loaded & OPT_IN_MODULES
    assert not any(name.split(".")[0] == "yfinance" for name in loaded)


# --- Data files and their provenance ------------------------------------------------------------------------------


def _header(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[: next((i for i, line in enumerate(lines) if not line.startswith("#")), len(lines))]


def _price_columns(lines: list[str]) -> set[str]:
    """Words that mark quotes or prices in a table's header: the first line after the comments, and any of the next
    four lines that holds no digit (a header row after lines about the underlying, as in Cboe's download)."""
    rows = lines[:1] + [line for line in lines[1:5] if not any(character.isdigit() for character in line)]
    words = set()
    for row in rows:
        for cell in row.split(","):
            for word in re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+", cell):
                words.add(word.lower().removesuffix("s"))
    return words & PRICE_WORDS


def _provenance_problems(path: Path, *, package_data: bool = False) -> list[str]:
    """What is wrong with a data file's provenance header (an empty list when nothing is)."""
    header = _header(path)
    if not header:
        return ["no provenance header"]
    match = re.fullmatch(r"# Provenance: ([a-z-]+)", header[0].strip())
    if not match or match.group(1) not in PROVENANCES:
        return [f"the first line must be '# Provenance: <{' | '.join(PROVENANCES)}>'"]
    kind = match.group(1)
    fields: dict[str, list[str]] = {}
    for line in header[1:]:
        key, _, value = line.lstrip("# ").partition(":")
        fields.setdefault(key.strip().lower(), []).append(value.strip())
    problems = []
    if package_data and kind != "public-domain":
        problems.append(f"data the package ships must be public-domain, not {kind}")
    if kind in ("published-facts", "public-domain"):
        if not fields.get("source"):
            problems.append("name the source ('# Source: ...')")
        if not (fields.get("url") and all(url.startswith("https://") for url in fields["url"])):
            problems.append("give the source's https:// URL ('# URL: ...')")
        try:
            retrieved = date.fromisoformat(fields.get("retrieved", [""])[0])
        except ValueError:
            problems.append("give the retrieval date ('# Retrieved: YYYY-MM-DD')")
        else:
            if not date(2000, 1, 1) <= retrieved <= date.today():
                problems.append(f"the retrieval date {retrieved} is not a past date")
    # Quotes, prices and index histories in the repository are made up, never copies of third-party data.
    words = _price_columns(path.read_text(encoding="utf-8").splitlines()[len(header) :])
    if words and kind not in ("fabricated", "synthetic"):
        problems.append(
            f"it holds quotes or prices ({', '.join(sorted(words))}), which must be fabricated or synthetic"
        )
    return problems


def _data_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.is_file() and "__pycache__" not in path.parts)


@pytest.mark.parametrize("path", _data_files(FIXTURES), ids=lambda path: path.name)
def test_every_fixture_carries_a_provenance_header(path):
    assert _provenance_problems(path) == [], path.name


def test_package_data_is_public_domain_with_its_source():
    """Every file in options_analyzer/data (none yet: the bundled Treasury curve is roadmap item DT-3)."""
    problems = {path.name: _provenance_problems(path, package_data=True) for path in _data_files(PACKAGE_DATA)}
    assert {name: found for name, found in problems.items() if found} == {}


def _write(folder: Path, name: str, *lines: str) -> Path:
    path = folder / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_the_provenance_check_catches_what_it_should(tmp_path):
    source = ["# Source: US Treasury", "# URL: https://home.treasury.gov/", "# Retrieved: 2026-10-26"]
    yields = _write(tmp_path, "yields.csv", "# Provenance: public-domain", *source, "Date,1 Mo,2 Mo,10 Yr", "x,4,4,4")
    assert _provenance_problems(yields) == _provenance_problems(yields, package_data=True) == []
    holidays = _write(
        tmp_path, "holidays.csv", "# Provenance: published-facts", *source, "date,kind,name", "2026-11-27,early_close,x"
    )
    assert _provenance_problems(holidays) == []
    assert _provenance_problems(holidays, package_data=True) == [
        "data the package ships must be public-domain, not published-facts"
    ]
    # A real price history or chain labelled as facts, in any column-naming style.
    for columns in (
        "Date,Strike,Expiry,Mid Price,Bid_Price,Ask_Price,Adj Close",
        "contractSymbol,lastTradeDate,strike,lastPrice,impliedVolatility",
        "Date,Open,High,Low,Close,Volume",
        "DATE,SETTLE,IV",
    ):
        mislabelled = _write(tmp_path, "real.csv", "# Provenance: published-facts", *source, columns, "1,2,3,4,5,6,7")
        assert any("holds quotes or prices" in problem for problem in _provenance_problems(mislabelled)), columns
    cboe = _write(tmp_path, "cboe.csv", "# Provenance: public-domain", *source, '"S&P 500 INDEX",Last: 7915.00', "x")
    assert "holds quotes or prices (last)" in _provenance_problems(cboe)[0]
    no_header = _write(tmp_path, "bare.csv", "date,value", "2026-01-01,1")
    assert _provenance_problems(no_header) == ["no provenance header"]
    undated = _write(tmp_path, "undated.csv", "# Provenance: public-domain", *source[:2], "Date,1 Mo", "x,4")
    assert _provenance_problems(undated) == ["give the retrieval date ('# Retrieved: YYYY-MM-DD')"]


def test_data_files_live_only_in_the_fixtures_or_the_package_data():
    """Market data in any other folder would escape the provenance check (hidden folders and environments aside)."""
    skipped = {"node_modules", "site-packages", "build", "dist", "venv"}
    stray = []
    for path in ROOT.rglob("*"):
        parts = path.relative_to(ROOT).parts
        if any(part.startswith(".") or part in skipped for part in parts) or not path.is_file():
            continue
        if path.suffix.lower() in DATA_SUFFIXES and not ({FIXTURES, PACKAGE_DATA} & set(path.parents)):
            stray.append(str(path.relative_to(ROOT)))
    assert stray == []


def test_the_policy_names_every_source_and_the_public_demo_rule():
    text = POLICY.read_text(encoding="utf-8")
    for needle in (
        "## The public-demo rule",
        "Synthetic — not market data",
        "No server-side fetching or scraping",
        "www.cboe.com/delayed_quotes",
        "www.cboe.com/terms",
        "legal.yahoo.com",
        "2(d)(ix)",
        "fred.stlouisfed.org",
        "www.federalreserve.gov/releases/h15",
        "home.treasury.gov",
        "www.nyse.com/markets/hours-calendars",
        "# Provenance: fabricated | synthetic | published-facts | public-domain",
        "options_analyzer/data/",
        *FORBIDDEN_HOSTS,
    ):
        assert needle in text, needle


# --- Exports carry their provenance (UI spec 2.13, UI-28) -----------------------------------------------------------

#: A file name that carries its source and its time: ``spx_chain_SYNTHETIC_20261023T1600.csv``.
EXPORT_NAME = re.compile(r"[a-z0-9_]+_(SYNTHETIC|UPLOAD|TEXTBOOK|PARAMETRIC)_\d{8}T\d{4}\.(csv|xlsx|json)")


def test_every_download_button_builds_its_file_through_the_exports_module():
    """The static half: no ``st.download_button`` in the package hands out bytes it built itself, so that no export
    can leave without its provenance lines; and each passes a builder, so that no file is built until it is clicked
    (spec 2.13: ``data`` a callable)."""
    offences = []
    for path in sorted((PACKAGE / "ui").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "download_button":
                data = node.args[1] if len(node.args) > 1 else next(k.value for k in node.keywords if k.arg == "data")
                name = (
                    node.args[2] if len(node.args) > 2 else next(k.value for k in node.keywords if k.arg == "file_name")
                )
                built = ast.unparse(data)
                if not built.startswith("exports.builder(exports.") and not re.match(r"exports\.builder\(_\w+,", built):
                    offences.append(f"{path.name}:{node.lineno} data={built}")
                if ".file_name(" not in ast.unparse(name):
                    offences.append(f"{path.name}:{node.lineno} file_name={ast.unparse(name)}")
    assert offences == []


def _content(data: Any) -> bytes:
    data = data() if callable(data) else data
    return data.encode("utf-8") if isinstance(data, str) else bytes(data)


def _provenance_of(file_name: str, content: bytes) -> dict[str, str]:
    """The provenance fields an export carries: a CSV's leading comment lines, an XLSX's Provenance sheet (read from
    the file itself: pd.read_excel needs openpyxl, which the app does not depend on), a JSON's provenance object."""
    if file_name.endswith(".csv"):
        lines = content.decode("utf-8").splitlines()
        comments = [line[2:] for line in itertools.takewhile(lambda line: line.startswith("# "), lines)]
        assert len(comments) == 4, comments  # the spec's four lines
        fields = {}
        for line in comments:
            for part in line.split("; "):
                key, _, value = part.partition(": ")
                fields[key] = value
        return fields
    if file_name.endswith(".xlsx"):
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            book = archive.read("xl/workbook.xml").decode()
            strings = re.findall(r"<si>.*?<t[^>]*>([^<]*)</t>.*?</si>", archive.read("xl/sharedStrings.xml").decode())
            sheet = archive.read("xl/worksheets/sheet1.xml").decode()  # the first sheet
        assert re.findall(r'<sheet name="([^"]+)"', book)[0] == "Provenance"
        # Each cell's text by its reference: shared strings (t="s") by index, inline strings as they are.
        cells = {}
        for attributes, body in re.findall(r"<c ([^>]*?)(?:/>|>(.*?)</c>)", sheet):
            ref = re.search(r'r="([A-Z]+\d+)"', attributes).group(1)
            kind = re.search(r't="(\w+)"', attributes)
            text = re.search(r"<(?:v|t)[^>]*>([^<]*)</(?:v|t)>", body)
            value = text.group(1) if text else ""
            cells[ref] = strings[int(value)] if kind and kind.group(1) == "s" else value
        assert (cells["A1"], cells["B1"]) == ("Field", "Value")
        rows = {int(ref[1:]) for ref in cells if ref.startswith("A")} - {1}
        return {cells[f"A{row}"]: cells[f"B{row}"] for row in sorted(rows)}
    return json.loads(content)["provenance"]


def _exports(monkeypatch, at, *steps) -> list[tuple[str, bytes]]:
    """Every file the app offers for download in the runs of ``at`` (and of each step after it), built as a click
    would build it."""
    import streamlit as st
    from streamlit.elements.widgets import button

    offered = []
    original = button.ButtonMixin.download_button

    def recording(self, label, data, file_name=None, *args, **kwargs):
        offered.append((file_name, data))
        return original(self, label, data, file_name, *args, **kwargs)

    # Columns look the method up on the class; st.download_button is bound to the main container at import.
    monkeypatch.setattr(button.ButtonMixin, "download_button", recording)
    monkeypatch.setattr(st, "download_button", functools.partial(recording, st._main))
    at.run()
    assert not at.exception, at.exception
    for step in steps:
        step(at)
        assert not at.exception, at.exception
    return [(name, _content(data)) for name, data in dict(offered).items()]


@pytest.mark.parametrize(
    ("query", "tag"),
    [({"preset": None, "asof": "2026-10-23T16:00"}, "SYNTHETIC"), ({}, "TEXTBOOK")],
    ids=["index", "textbook"],
)
def test_no_export_lacks_its_source(monkeypatch, new_app, query, tag):
    """The run-time half (DT-6): every file the Surface and Strategy tabs offer, under each surface source, names its
    source on its first line (a CSV) or in its Provenance sheet or object, with the valuation time, the model, the
    app's version and the seed, and its file name carries the source and the time."""

    def source(name):
        return lambda at: at.button_group(key="surface_source").set_value(name).run()

    from options_analyzer.ui import tab_surface

    surface = _exports(
        monkeypatch, new_app(tab="Surface", **query), source(tab_surface.QUOTES), source(tab_surface.SSVI)
    )
    strategy = _exports(monkeypatch, new_app(tab="Strategy", **query))
    files = surface + strategy
    names = {name for name, _ in files}
    assert {n.split("_SYNTHETIC")[0].split("_TEXTBOOK")[0] for n in names if "PARAMETRIC" not in n} >= {
        "sample_option_quotes",
        "sample_option_prices",
        "bull_call_spread_pnl",
    }
    assert any("PARAMETRIC" in n for n in names) and any(n.endswith(".xlsx") for n in names) == bool(
        importlib.util.find_spec("xlsxwriter") or importlib.util.find_spec("openpyxl")
    )
    if tag == "SYNTHETIC":
        assert "spx_chain_SYNTHETIC_20261023T1600.csv" in names
    for name, content in files:
        assert EXPORT_NAME.fullmatch(name), name
        if name.endswith(".csv"):
            assert content.startswith(b"# source: "), name
        fields = _provenance_of(name, content)
        assert fields["source"].split(" ")[0] in {"SYNTHETIC", "TEXTBOOK", "PARAMETRIC"}, (name, fields)
        assert "not market data" in fields["source"], (name, fields)
        assert {"as_of", "model", "app", "seed"} <= set(fields), (name, fields)
        assert fields["app"].startswith("v"), (name, fields)
        if tag == "SYNTHETIC":
            assert fields["as_of"] == "2026-10-23T16:00-04:00 (cash close)", (name, fields)
        if "SYNTHETIC" in name:
            assert fields["seed"] == "0", (name, fields)


def test_an_export_reads_back_into_the_app(new_app):
    """A downloaded sample, provenance lines and all, is a valid upload (the uploader skips the leading comments)."""
    from options_analyzer import vol_surface
    from options_analyzer.ui import exports, tab_surface

    at = new_app(tab="Surface").run()
    at.button_group(key="surface_source").set_value(tab_surface.QUOTES).run()
    prov = exports.Provenance(exports.SYNTHETIC, exports.SYNTHETIC_SOURCE, None, "Black-76", seed=0)
    data = exports.csv(vol_surface.sample_quotes(100.0, 0.05, 0.0), prov)
    at.file_uploader(key="quotes_file").upload("sample.csv", data, "text/csv").run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    assert any(m.label == "Grid" for m in at.metric)
