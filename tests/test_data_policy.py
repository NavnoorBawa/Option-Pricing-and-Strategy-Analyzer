"""The data and licensing policy (docs/data_policy.md), enforced (DT-6).

* No module of the package, and not the entry point, names a Cboe or Yahoo quote host or imports yfinance; a module
  that a user opts into for local use would be listed in ``OPT_IN_MODULES`` and must stay off the default import path.
* The default app runs without reaching the network: with the proxy pointed at a dead port it renders every tab, and
  no connection goes to that port.
* Every file in tests/fixtures starts with a provenance header, and no quote or price file is real third-party data.
"""

from __future__ import annotations

import ast
import json
import os
import re
import socket
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "options_analyzer"
FIXTURES = ROOT / "tests" / "fixtures"
POLICY = ROOT / "docs" / "data_policy.md"
#: Hosts whose quotes the public app must never fetch (Cboe's delayed-quote JSON and Yahoo Finance).
FORBIDDEN_HOSTS = ("cdn.cboe.com", "finance.yahoo.com")
#: Modules a user may run on their own machine to fetch data; none exists yet (roadmap item DT-7).
OPT_IN_MODULES: frozenset[str] = frozenset()
PROVENANCES = ("fabricated", "synthetic", "published-facts", "public-domain")
DATA_SUFFIXES = {".csv", ".tsv", ".parquet", ".feather", ".xlsx", ".xls", ".json", ".dat"}


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


def test_the_default_import_path_loads_no_opt_in_module_and_no_yfinance():
    """Importing the entry point and the data package in a fresh interpreter loads none of the opt-in modules."""
    code = (
        "import json, sys\n"
        "import main, options_analyzer.market_data, options_analyzer.clock\n"
        "ours = ('options_analyzer', 'yfinance')\n"
        "print(json.dumps(sorted(name for name in sys.modules if name.split('.')[0] in ours)))"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120, check=True
    )
    loaded = set(json.loads(result.stdout.strip().splitlines()[-1]))
    assert {"options_analyzer.market_data.adapters", "options_analyzer.ui.app"} <= loaded
    assert not loaded & OPT_IN_MODULES
    assert not any(name.split(".")[0] == "yfinance" for name in loaded)


def test_default_app_renders_every_tab_with_the_network_unreachable(monkeypatch, new_app):
    """DT-6: with the proxies pointed at a dead local port, the app renders every tab from its own data, quickly, and
    opens no connection to that port (nothing on the default path tries to reach the internet)."""
    with socket.socket() as probe:  # a port nothing listens on
        probe.bind(("127.0.0.1", 0))
        dead = probe.getsockname()[1]
    proxy = f"http://127.0.0.1:{dead}"
    for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.setenv(name, proxy)
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    attempts = []
    connect = socket.socket.connect

    def recording_connect(sock, address):
        if isinstance(address, tuple) and len(address) > 1 and address[1] == dead:
            attempts.append(address)
        return connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", recording_connect)
    at = new_app()
    start = time.perf_counter()
    at.run()
    elapsed = time.perf_counter() - start
    assert not at.exception, at.exception
    assert attempts == []
    assert len(at.tabs) >= 6
    assert elapsed < 60  # a hung connection would wait for the AppTest timeout (120 s)


def _header(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[: next((i for i, line in enumerate(lines) if not line.startswith("#")), len(lines))]


def _fixture_files() -> list[Path]:
    return sorted(path for path in FIXTURES.rglob("*") if path.is_file() and "__pycache__" not in path.parts)


@pytest.mark.parametrize("path", _fixture_files(), ids=lambda path: path.name)
def test_every_fixture_carries_a_provenance_header(path):
    header = _header(path)
    assert header, f"{path.name} has no provenance header"
    match = re.fullmatch(r"# Provenance: ([a-z-]+)", header[0].strip())
    assert match and match.group(1) in PROVENANCES, f"{path.name}: the first line must be '# Provenance: <kind>'"
    fields = {}
    for line in header[1:]:
        key, _, value = line.lstrip("# ").partition(":")
        fields.setdefault(key.strip().lower(), []).append(value.strip())
    if match.group(1) in ("published-facts", "public-domain"):
        assert fields.get("source"), f"{path.name}: name the source"
        assert fields.get("url") and all(url.startswith("https://") for url in fields["url"]), path.name
        retrieved = date.fromisoformat(fields["retrieved"][0])
        assert date(2000, 1, 1) <= retrieved <= date.today(), path.name
    # Quotes, prices and index histories in the repository are made up, never copies of third-party data.
    data = path.read_text(encoding="utf-8").splitlines()[len(header) :]
    columns = {re.sub(r"[^a-z]", "", cell.lower()) for line in data[:5] for cell in line.split(",")}
    if columns & {"bid", "ask", "close", "lastprice", "lastsale"}:
        assert match.group(1) in ("fabricated", "synthetic"), f"{path.name} holds quotes or prices"


def test_data_files_live_only_in_the_fixtures():
    """Market data in any other folder would escape the provenance check (hidden folders and environments aside)."""
    skipped = {"node_modules", "site-packages", "build", "dist", "venv"}
    stray = []
    for path in ROOT.rglob("*"):
        parts = path.relative_to(ROOT).parts
        if any(part.startswith(".") or part in skipped for part in parts) or not path.is_file():
            continue
        if path.suffix.lower() in DATA_SUFFIXES and FIXTURES not in path.parents:
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
        *FORBIDDEN_HOSTS,
    ):
        assert needle in text, needle
