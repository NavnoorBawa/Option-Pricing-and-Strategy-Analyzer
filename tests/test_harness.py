"""The test harness itself: network guard, strict pytest settings, Hypothesis profiles, package data and pins."""

import asyncio
import fnmatch
import importlib.metadata
import math
import os
import socket
import urllib.request
from pathlib import Path

import pytest

from options_analyzer import black_scholes as bs

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10; pytest itself depends on tomli there
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
BLOCKED = "Network access is blocked in tests"
#: TEST-NET-1 (RFC 5737): documentation addresses that are never routed, so a missed block cannot reach anyone.
UNROUTABLE = "192.0.2.1"


# --- Network guard (this test is not marked, so the guard is active) -----------------------------------------------


def _socket_call(kind: int, method: str, port: int) -> None:
    with socket.socket(socket.AF_INET, kind) as sock:
        if method == "sendto":
            sock.sendto(b"x", (UNROUTABLE, port))
        else:
            getattr(sock, method)((UNROUTABLE, port))


@pytest.mark.parametrize(
    "reach_out",
    [
        pytest.param(lambda: socket.create_connection((UNROUTABLE, 80), timeout=1), id="create_connection"),
        pytest.param(lambda: _socket_call(socket.SOCK_STREAM, "connect", 80), id="connect"),
        pytest.param(lambda: _socket_call(socket.SOCK_STREAM, "connect_ex", 80), id="connect_ex"),
        pytest.param(lambda: _socket_call(socket.SOCK_DGRAM, "sendto", 53), id="udp_sendto"),
        pytest.param(lambda: socket.getaddrinfo("pypi.org", 443), id="getaddrinfo"),
        pytest.param(lambda: socket.gethostbyname("example.com"), id="gethostbyname"),
        pytest.param(lambda: socket.gethostbyname_ex("example.com"), id="gethostbyname_ex"),
        pytest.param(lambda: urllib.request.urlopen("https://pypi.org/simple/", timeout=1), id="urlopen"),
    ],
)
def test_unmarked_tests_cannot_reach_other_machines(reach_out):
    with pytest.raises(RuntimeError, match=BLOCKED):
        reach_out()


def test_loopback_and_local_sockets_stay_available():
    # A local server (as browser tests start) and asyncio's internal socket pair must keep working.
    with socket.create_server(("127.0.0.1", 0)) as server, socket.create_connection(server.getsockname(), 5) as client:
        peer, _ = server.accept()
        with peer:
            client.sendall(b"ping")
            assert peer.recv(4) == b"ping"
    assert socket.getaddrinfo("localhost", 80)
    assert socket.getaddrinfo(None, 80)
    left, right = socket.socketpair()
    with left, right:
        left.sendall(b"ok")
        assert right.recv(2) == b"ok"
    assert asyncio.run(asyncio.sleep(0, result="done")) == "done"


def test_a_separate_session_fails_unmarked_network_tests_and_lets_marked_ones_through(pytester, monkeypatch):
    # The repository's own pytest settings and conftest, in a fresh interpreter (no guard inherited from this run).
    pytester.makepyprojecttoml((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pytester.makeconftest((ROOT / "tests" / "conftest.py").read_text(encoding="utf-8"))
    pytester.makepyfile(
        test_inner=f"""
        import socket

        import pytest
        from hypothesis import settings


        def test_reaches_out():
            socket.create_connection(("{UNROUTABLE}", 80), timeout=1)


        @pytest.mark.network
        def test_opted_in():
            # Nothing is wrapped: the guard is off for tests marked network (no traffic is sent here).
            assert "connect" not in vars(socket.socket)
            assert not hasattr(socket.getaddrinfo, "__wrapped__")


        def test_guard_is_back_after_the_marked_test():
            assert hasattr(socket.getaddrinfo, "__wrapped__")


        @pytest.mark.xfail(reason="expected to fail")
        def test_unexpected_pass():
            pass


        def test_profile_comes_from_the_environment():
            assert settings.default.max_examples == 5000 and not settings.default.derandomize
        """,
        test_typo="""
        import pytest


        @pytest.mark.netwrok
        def test_misspelt_marker():
            pass
        """,
    )
    monkeypatch.setenv("HYPOTHESIS_PROFILE", "nightly")
    result = pytester.runpytest_subprocess("test_inner.py", "test_typo.py", "--continue-on-collection-errors")
    result.assert_outcomes(passed=3, failed=2, errors=1)
    output = result.stdout.str()
    assert "FAILED test_inner.py::test_reaches_out" in output
    assert f"NetworkAccessBlocked: {BLOCKED} (getaddrinfo '{UNROUTABLE}')" in output
    assert "FAILED test_inner.py::test_unexpected_pass" in output and "[XPASS(strict)]" in output
    assert "ERROR test_typo.py" in output and "'netwrok' not found in `markers`" in output


def test_misspelt_settings_are_rejected(pytester):
    config = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    pytester.makepyprojecttoml(config.replace("strict_xfail = true", "xfail_stirct = true"))
    pytester.makepyfile(test_nothing="def test_nothing():\n    pass\n")
    result = pytester.runpytest_subprocess("--collect-only", "test_nothing.py")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert "Unknown config option: xfail_stirct" in result.stderr.str()


# --- Hypothesis profiles -------------------------------------------------------------------------------------------


def test_hypothesis_profiles():
    hypothesis = pytest.importorskip("hypothesis")
    settings = hypothesis.settings
    dev, ci, nightly = (settings.get_profile(name) for name in ("dev", "ci", "nightly"))
    assert dev.deadline is None and dev.max_examples == 100
    assert (ci.max_examples, ci.derandomize, ci.database, ci.deadline, ci.print_blob) == (300, True, None, None, True)
    assert hypothesis.HealthCheck.too_slow in ci.suppress_health_check
    assert (nightly.max_examples, nightly.derandomize, nightly.database) == (5000, False, None)
    expected = os.environ.get("HYPOTHESIS_PROFILE") or ("ci" if os.environ.get("CI") else "dev")
    active = settings.get_profile(expected)
    assert (settings.default.max_examples, settings.default.derandomize) == (active.max_examples, active.derandomize)


# --- Package data --------------------------------------------------------------------------------------------------


def _matches(pattern: str, path: str) -> bool:
    """setuptools package-data glob: ``**`` spans any number of directories, ``*`` stays within one."""

    def match(parts: list[str], names: list[str]) -> bool:
        if not parts:
            return not names
        if parts[0] == "**":
            return any(match(parts[1:], names[i:]) for i in range(len(names) + 1))
        return bool(names) and fnmatch.fnmatchcase(names[0], parts[0]) and match(parts[1:], names[1:])

    return match(pattern.split("/"), path.split("/"))


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("static/**/*", "static/fonts/IBMPlexSans-Regular.woff2", True),
        ("static/**/*", "static/LICENSE.txt", True),
        ("methodology/*", "methodology/pricing.md", True),
        ("methodology/*", "methodology/old/pricing.md", False),
        (".streamlit/config.toml", ".streamlit/config.toml", True),
        ("static/**/*", "statics/x.css", False),
    ],
)
def test_package_data_glob_semantics(pattern, path, expected):
    assert _matches(pattern, path) is expected


def test_every_data_file_inside_the_package_is_declared_as_package_data():
    # Fonts, stylesheets and Methodology text must reach the wheel, or the installed app silently loses them.
    package_data = PYPROJECT["tool"]["setuptools"]["package-data"]
    assert {"static/**/*", "methodology/*"} <= set(package_data["options_analyzer.ui"])
    package_root = ROOT / "options_analyzer"
    undeclared = []
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or path.suffix in {".py", ".pyc"} or "__pycache__" in path.parts:
            continue
        package = path.parent
        while not (package / "__init__.py").is_file():
            package = package.parent
        name = ".".join(package.relative_to(ROOT).parts)
        relative = path.relative_to(package).as_posix()
        if not any(_matches(pattern, relative) for pattern in package_data.get(name, [])):
            undeclared.append(path.relative_to(ROOT).as_posix())
    assert not undeclared, f"add these to [tool.setuptools.package-data] in pyproject.toml: {undeclared}"


# --- Pinned tools --------------------------------------------------------------------------------------------------


def _requirements(name: str) -> list[str]:
    lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
    return [line.split("#")[0].strip() for line in lines if line.split("#")[0].strip()]


@pytest.mark.parametrize("name", ["requirements-dev.txt", "requirements-oracle.txt"])
def test_tool_requirements_are_exact_pins(name):
    # These files are the only place the dev and oracle tools are pinned (Dependabot bumps them).
    loose = [line for line in _requirements(name) if not line.startswith("-r ") and "==" not in line]
    assert not loose, f"pin with == in {name}: {loose}"


def test_streamlit_floor_is_the_same_in_requirements_and_pyproject():
    app_extra = [d for d in PYPROJECT["project"]["optional-dependencies"]["app"] if d.startswith("streamlit")]
    assert app_extra == ["streamlit>=1.65,<2"]
    assert "streamlit>=1.65,<2" in _requirements("requirements.txt")


# --- Oracle environment (requirements-oracle.txt) -----------------------------------------------------------------


@pytest.mark.oracle
def test_oracle_libraries_are_the_pinned_versions_and_price_like_the_engine():
    ql = pytest.importorskip("QuantLib")
    lbr = pytest.importorskip("py_lets_be_rational")
    pins = dict(line.split("==") for line in _requirements("requirements-oracle.txt") if "==" in line)
    assert importlib.metadata.version("QuantLib") == pins["QuantLib"]
    assert importlib.metadata.version("lets-be-rational") == pins["lets-be-rational"]
    s, k, t, r, q, sigma = 100.0, 105.0, 0.75, 0.04, 0.015, 0.25
    forward, discount = s * math.exp((r - q) * t), math.exp(-r * t)
    ours = float(bs.price(s, k, t, r, sigma, q, "call"))
    quantlib = ql.BlackCalculator(ql.PlainVanillaPayoff(ql.Option.Call, k), forward, sigma * math.sqrt(t), discount)
    assert abs(ours - quantlib.value()) <= 1e-13 * s
    implied = lbr.implied_volatility_from_a_transformed_rational_guess(ours / discount, forward, k, t, 1.0)
    assert implied == pytest.approx(sigma, abs=1e-12)
