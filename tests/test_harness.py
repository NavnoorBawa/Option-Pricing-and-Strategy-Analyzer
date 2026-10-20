"""The test harness itself: network guard, strict pytest settings, Hypothesis profiles, package data and pins."""

import asyncio
import fnmatch
import http.client
import http.server
import importlib.metadata
import importlib.util
import math
import os
import re
import shutil
import site
import socket
import subprocess
import textwrap
import urllib.request
from pathlib import Path

import pytest
from _oracles import ORACLE_DEPENDENCIES, ORACLE_LIBRARIES

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


#: tests/conftest.py and the helper modules it imports, which a separate pytest session needs next to it.
_CONFTEST_FILES = ("conftest.py", "_apptest.py", "_oracles.py")


def _copy_conftest(pytester: pytest.Pytester) -> None:
    for name in _CONFTEST_FILES:
        (pytester.path / name).write_text((ROOT / "tests" / name).read_text(encoding="utf-8"), encoding="utf-8")


def _socket_call(kind: int, method: str, port: int) -> None:
    with socket.socket(socket.AF_INET, kind) as sock:
        if method == "sendto":
            sock.sendto(b"x", (UNROUTABLE, port))
        elif method == "sendmsg":
            sock.sendmsg([b"x"], [], 0, (UNROUTABLE, port))
        else:
            getattr(sock, method)((UNROUTABLE, port))


def _urlopen(url: str) -> None:
    urllib.request.urlopen(url, timeout=5)


def _requests_get(url: str) -> None:
    pytest.importorskip("requests").get(url, timeout=5)


@pytest.mark.parametrize(
    "reach_out",
    [
        pytest.param(lambda: socket.create_connection((UNROUTABLE, 80), timeout=1), id="create_connection"),
        pytest.param(lambda: _socket_call(socket.SOCK_STREAM, "connect", 80), id="connect"),
        pytest.param(lambda: _socket_call(socket.SOCK_STREAM, "connect_ex", 80), id="connect_ex"),
        pytest.param(lambda: _socket_call(socket.SOCK_DGRAM, "sendto", 53), id="udp_sendto"),
        pytest.param(lambda: _socket_call(socket.SOCK_DGRAM, "sendmsg", 9), id="udp_sendmsg"),
        pytest.param(lambda: socket.getaddrinfo("pypi.org", 443), id="getaddrinfo"),
        pytest.param(lambda: socket.gethostbyname("example.com"), id="gethostbyname"),
        pytest.param(lambda: socket.gethostbyname_ex("example.com"), id="gethostbyname_ex"),
        pytest.param(lambda: socket.gethostbyaddr(UNROUTABLE), id="gethostbyaddr"),
        pytest.param(lambda: socket.getnameinfo((UNROUTABLE, 80), 0), id="getnameinfo"),
        pytest.param(lambda: socket.getfqdn(UNROUTABLE), id="getfqdn"),
        # example.com is in no NO_PROXY list, so these would go through a proxy set in the environment.
        pytest.param(lambda: _urlopen("https://example.com/"), id="urlopen"),
        pytest.param(lambda: _requests_get("https://example.com/"), id="requests"),
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
    # Reverse look-ups of this machine, and numeric ones that send no query.
    assert socket.gethostbyaddr("127.0.0.1")
    assert socket.getnameinfo((UNROUTABLE, 80), socket.NI_NUMERICHOST | socket.NI_NUMERICSERV) == (UNROUTABLE, "80")
    assert socket.getnameinfo(("127.0.0.1", 80), socket.NI_NUMERICSERV)[1] == "80"
    # http.server bound to all interfaces looks up this machine's own name (socket.getfqdn("")).
    with http.server.HTTPServer(("", 0), http.server.BaseHTTPRequestHandler) as server:
        assert server.server_port > 0
    if hasattr(socket.socket, "sendmsg"):  # not on Windows
        # A local datagram, and buffers given as a tuple (not mistaken for an address).
        with (
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver,
            socket.socket(type=socket.SOCK_DGRAM) as sender,
        ):
            receiver.bind(("127.0.0.1", 0))
            assert sender.sendmsg((b"da", b"ta"), [], 0, receiver.getsockname()) == 4
            assert receiver.recv(4) == b"data"
            sender.connect(receiver.getsockname())
            assert sender.sendmsg((b"ok",)) == 2


def _proxy_variables() -> list[str]:
    return [name for name in os.environ if name.lower().endswith("_proxy") and name.lower() != "no_proxy"]


@pytest.mark.parametrize("fetch", [pytest.param(_urlopen, id="urllib"), pytest.param(_requests_get, id="requests")])
def test_a_proxy_on_this_machine_cannot_fetch_for_an_unmarked_test(fetch, monkeypatch):
    # The guard removes the proxy variables and sets NO_PROXY=*, so clients connect directly and meet it; otherwise
    # their connection to a local proxy (cntlm, px, mitmproxy, a sandbox's) would be a loopback one, and allowed.
    assert not _proxy_variables()
    assert os.environ["NO_PROXY"] == "*"  # on Windows the same variable as no_proxy
    assert set(urllib.request.getproxies()) <= {"no"}
    with socket.create_server(("127.0.0.1", 0)) as proxy:
        proxy_url = "http://{}:{}".format(*proxy.getsockname()[:2])
        for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "ALL_PROXY"):
            monkeypatch.setenv(name, proxy_url)  # as a test or a library might, after the guard started
        with pytest.raises(RuntimeError, match=f"{BLOCKED} \\(getaddrinfo 'example.com'\\)"):
            fetch("https://example.com/")
        proxy.setblocking(False)
        with pytest.raises(BlockingIOError):
            proxy.accept()  # nothing connected to the proxy


def test_a_proxy_configured_before_the_session_cannot_fetch_for_unmarked_tests(pytester, monkeypatch):
    # As in a sandbox or behind a local corporate proxy: HTTPS_PROXY names a proxy on this machine when pytest starts.
    with socket.create_server(("127.0.0.1", 0)) as proxy:
        host, port = proxy.getsockname()[:2]
        proxy_url = f"http://{host}:{port}"
        monkeypatch.setenv("HTTPS_PROXY", proxy_url)
        monkeypatch.setenv("HTTP_PROXY", proxy_url)
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)
        # pytester points HOME elsewhere; keep the user site-packages (where requests may live) visible.
        monkeypatch.setenv("PYTHONUSERBASE", site.getuserbase())
        pytester.makepyprojecttoml((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        _copy_conftest(pytester)
        pytester.makepyfile(
            test_proxy=f"""
            import http.client
            import os
            import socket
            import urllib.request

            import pytest


            def test_the_proxy_variables_are_removed():
                assert "HTTPS_PROXY" not in os.environ and "HTTP_PROXY" not in os.environ
                assert os.environ["NO_PROXY"] == "*" and set(urllib.request.getproxies()) <= {{"no"}}


            def test_urllib_connects_directly_and_is_blocked():
                with pytest.raises(RuntimeError, match="getaddrinfo 'example.com'"):
                    urllib.request.urlopen("https://example.com/", timeout=5)


            def test_requests_connects_directly_and_is_blocked():
                requests = pytest.importorskip("requests")
                with pytest.raises(RuntimeError, match="getaddrinfo 'example.com'"):
                    requests.get("https://example.com/", timeout=5)


            def test_a_client_that_still_knows_the_proxy_is_blocked_at_the_proxy():
                # For example a proxy URL read before the test started.
                with pytest.raises(RuntimeError, match="connect to the local proxy"):
                    socket.create_connection(("localhost", {port}), timeout=5)
                tunnel = http.client.HTTPConnection("{host}", {port}, timeout=5)
                tunnel.set_tunnel("example.com", 443)
                with pytest.raises(RuntimeError, match="connect to the local proxy"):
                    tunnel.request("GET", "/")


            def test_other_local_servers_stay_reachable():
                with socket.create_server(("127.0.0.1", 0)) as server:
                    socket.create_connection(server.getsockname(), timeout=5).close()


            @pytest.mark.network
            def test_marked_tests_keep_the_proxy_settings_and_use_the_proxy():
                assert os.environ["HTTPS_PROXY"] == "{proxy_url}" and "NO_PROXY" not in os.environ
                # Also after urlopen() ran in the unmarked tests above (it keeps its first opener).
                with pytest.raises(OSError):  # the stand-in proxy never answers
                    urllib.request.urlopen("http://example.invalid/", timeout=1)
            """,
            test_import_time=f"""
            import urllib.request

            # While modules are imported the settings stay (a client made now keeps them), but the proxy is blocked.
            PROXIES_AT_IMPORT = urllib.request.getproxies()
            try:
                urllib.request.urlopen("https://example.com/", timeout=5)
            except RuntimeError as error:
                IMPORT_TIME_FETCH = str(error)


            def test_an_import_time_fetch_is_blocked_at_the_proxy():
                assert PROXIES_AT_IMPORT["https"] == "{proxy_url}"
                assert "connect to the local proxy ('127.0.0.1', {port})" in IMPORT_TIME_FETCH
            """,
        )
        result = pytester.runpytest_subprocess("test_proxy.py")
        has_requests = importlib.util.find_spec("requests") is not None
        result.assert_outcomes(passed=6 if has_requests else 5, skipped=0 if has_requests else 1)
        pytester.runpytest_subprocess("test_import_time.py").assert_outcomes(passed=1)
        # Exactly one connection reached the proxy: the marked test's request.
        proxy.settimeout(5)
        connection, _ = proxy.accept()
        with connection:
            assert connection.recv(64).startswith(b"GET http://example.invalid/ HTTP/1.1")
        proxy.setblocking(False)
        with pytest.raises(BlockingIOError):
            proxy.accept()


def test_a_separate_session_fails_unmarked_network_tests_and_lets_marked_ones_through(pytester, monkeypatch):
    # The repository's own pytest settings and conftest, in a fresh interpreter (no guard inherited from this run).
    pytester.makepyprojecttoml((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    _copy_conftest(pytester)
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


def test_hypothesis_profiles(request):
    hypothesis = pytest.importorskip("hypothesis")
    settings = hypothesis.settings
    dev, ci, nightly = (settings.get_profile(name) for name in ("dev", "ci", "nightly"))
    assert dev.deadline is None and dev.max_examples == 100
    assert (ci.max_examples, ci.derandomize, ci.database, ci.deadline, ci.print_blob) == (300, True, None, None, True)
    assert hypothesis.HealthCheck.too_slow in ci.suppress_health_check
    assert (nightly.max_examples, nightly.derandomize, nightly.database) == (5000, False, None)
    chosen = request.config.getoption("hypothesis_profile", default=None)  # pytest --hypothesis-profile=<name>
    expected = chosen or os.environ.get("HYPOTHESIS_PROFILE") or ("ci" if os.environ.get("CI") else "dev")
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


def _package_files(root: Path) -> list[Path]:
    """The files of ``root/options_analyzer``: in a git checkout the tracked ones, as the CI package job lists them
    (so ignored files such as .DS_Store or editor swap files do not count); otherwise every file."""
    git = shutil.which("git")
    if git is not None:
        top = subprocess.run([git, "rev-parse", "--show-toplevel"], cwd=root, capture_output=True, text=True)
        if top.returncode == 0 and Path(top.stdout.strip()).resolve() == root.resolve():
            listed = subprocess.run(
                [git, "ls-files", "-z", "--", "options_analyzer"], cwd=root, capture_output=True, text=True, check=True
            )
            return sorted(root / name for name in listed.stdout.split("\0") if name)
    return sorted((root / "options_analyzer").rglob("*"))


def _undeclared_package_data(root: Path, package_data: dict[str, list[str]]) -> list[str]:
    """Non-Python files of the package that no [tool.setuptools.package-data] glob of their package matches."""
    undeclared = []
    for path in _package_files(root):
        if not path.is_file() or path.suffix in {".py", ".pyc"} or "__pycache__" in path.parts:
            continue
        package = path.parent
        while package != root and not (package / "__init__.py").is_file():
            package = package.parent
        name = ".".join(package.relative_to(root).parts)
        relative = path.relative_to(package).as_posix()
        if not any(_matches(pattern, relative) for pattern in package_data.get(name, [])):
            undeclared.append(path.relative_to(root).as_posix())
    return undeclared


def test_every_data_file_inside_the_package_is_declared_as_package_data():
    # Fonts, stylesheets and Methodology text must reach the wheel, or the installed app silently loses them.
    package_data = PYPROJECT["tool"]["setuptools"]["package-data"]
    assert {"static/**/*", "methodology/*"} <= set(package_data["options_analyzer.ui"])
    undeclared = _undeclared_package_data(ROOT, package_data)
    assert not undeclared, f"add these to [tool.setuptools.package-data] in pyproject.toml: {undeclared}"


def test_the_package_data_check_ignores_files_that_git_does_not_track(tmp_path):
    git = shutil.which("git")
    if git is None:
        pytest.skip("needs git")
    ui = tmp_path / "options_analyzer" / "ui"
    (ui / "static").mkdir(parents=True)
    for package in (ui.parent, ui):
        (package / "__init__.py").write_text("", encoding="utf-8")
    (ui / "static" / "font.woff2").write_bytes(b"font")
    (ui / "notes.txt").write_text("tracked but not declared", encoding="utf-8")
    (ui / ".DS_Store").write_bytes(b"Finder")  # ignored by .gitignore, as in the repository
    (ui / ".notes.txt.swp").write_bytes(b"vim")  # untracked
    (tmp_path / ".gitignore").write_text(".DS_Store\n", encoding="utf-8")
    declared = {"options_analyzer.ui": ["static/**/*"]}
    # Without a git checkout (an unpacked sdist, say) every file counts.
    assert _undeclared_package_data(tmp_path, declared) == [
        "options_analyzer/ui/.DS_Store",
        "options_analyzer/ui/.notes.txt.swp",
        "options_analyzer/ui/notes.txt",
    ]
    subprocess.run([git, "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        [git, "add", ".gitignore", "options_analyzer/__init__.py", "options_analyzer/ui"], cwd=tmp_path, check=True
    )
    subprocess.run([git, "rm", "-q", "--cached", "options_analyzer/ui/.notes.txt.swp"], cwd=tmp_path, check=True)
    assert _undeclared_package_data(tmp_path, declared) == ["options_analyzer/ui/notes.txt"]


# --- Pinned tools --------------------------------------------------------------------------------------------------


def _requirements(name: str) -> list[str]:
    lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
    return [line.split("#")[0].strip() for line in lines if line.split("#")[0].strip()]


@pytest.mark.parametrize("name", ["requirements-dev.txt", "requirements-oracle.txt"])
def test_tool_requirements_are_exact_pins(name):
    # These files are the only place the dev and oracle tools are pinned (Dependabot bumps them).
    loose = [line for line in _requirements(name) if not line.startswith("-r ") and "==" not in line]
    assert not loose, f"pin with == in {name}: {loose}"


def _distribution(requirement: str) -> str:
    return re.sub(r"[-_.]+", "-", re.split(r"[\s<>=!~;\[]", requirement, maxsplit=1)[0]).lower()


def test_the_oracle_job_checks_every_pinned_oracle_library():
    # tests/conftest.py imports these under REQUIRE_ORACLES=1; a library pinned but missing there would go unchecked.
    oracle_pins = {
        _distribution(line) for line in _requirements("requirements-oracle.txt") if not line.startswith("-r ")
    }
    dev_pins = {_distribution(line) for line in _requirements("requirements-dev.txt") if not line.startswith("-r ")}
    checked = {_distribution(name) for name in ORACLE_LIBRARIES}
    unchecked = oracle_pins - checked - ORACLE_DEPENDENCIES
    assert not unchecked, f"add to ORACLE_LIBRARIES (or ORACLE_DEPENDENCIES) in tests/_oracles.py: {sorted(unchecked)}"
    assert oracle_pins >= ORACLE_DEPENDENCIES  # an oracle's own dependencies are pinned too
    assert checked <= oracle_pins | dev_pins


def _ci_job(name: str) -> list[str]:
    """The lines of one job in .github/workflows/ci.yml, without comments (no YAML parser is installed in CI)."""
    lines = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8").splitlines()
    start = lines.index(f"  {name}:") + 1
    end = next((i for i in range(start, len(lines)) if re.match(r"^ {0,2}\S", lines[i])), len(lines))
    return [line.strip() for line in lines[start:end] if line.strip() and not line.strip().startswith("#")]


def test_the_ci_oracle_job_requires_the_oracles():
    job = _ci_job("oracle")
    assert 'REQUIRE_ORACLES: "1"' in job
    assert "run: pip install -r requirements-oracle.txt" in job
    assert any(line.startswith("run: pytest") and "-m oracle" in line for line in job)


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


def _oracle_session(pytester: pytest.Pytester) -> None:
    """A separate session laid out like the repository, with stand-ins for the oracle libraries and oracle tests that
    run, skip and xfail; tests/oracle/ starts with an importorskip in its conftest, as VL-1's suite will."""
    pytester.makepyprojecttoml((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    # Stand-ins that import, found before any installed copy (pyproject.toml puts the session's directory first).
    pytester.makepyfile(**dict.fromkeys(ORACLE_LIBRARIES.values(), ""))
    files = {
        **{f"tests/{name}": (ROOT / "tests" / name).read_text(encoding="utf-8") for name in _CONFTEST_FILES},
        "tests/test_oracles.py": """
            import pytest


            @pytest.mark.oracle
            def test_runs():
                pass


            @pytest.mark.oracle
            def test_library_missing():
                pytest.importorskip("an_oracle_library_that_is_not_installed")


            @pytest.mark.oracle
            @pytest.mark.skipif(True, reason="skipped by a marker")
            def test_skipped_by_a_marker():
                pass


            @pytest.mark.oracle
            @pytest.mark.xfail(reason="a known difference", strict=True)
            def test_known_difference():
                raise AssertionError


            def test_not_an_oracle_test():
                pytest.skip("other tests may still skip")
            """,
        "tests/test_versioned.py": """
            import pytest

            QuantLib = pytest.importorskip("QuantLib", minversion="999")


            def test_needs_a_newer_quantlib():
                pass
            """,
        "tests/oracle/conftest.py": """
            import pytest

            pytest.importorskip("another_missing_library")
            """,
        "tests/oracle/test_suite.py": """
            import pytest


            @pytest.mark.oracle
            def test_in_the_suite():
                pass
            """,
    }
    for name, text in files.items():
        path = pytester.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")


def test_oracle_tests_skip_without_the_libraries_unless_the_oracles_are_required(pytester, monkeypatch):
    _oracle_session(pytester)
    keep_going = "--continue-on-collection-errors"
    monkeypatch.delenv("REQUIRE_ORACLES", raising=False)
    pytester.runpytest_subprocess(keep_going).assert_outcomes(passed=1, skipped=5, xfailed=1)

    monkeypatch.setenv("REQUIRE_ORACLES", "1")  # as in the CI oracle job
    result = pytester.runpytest_subprocess("-rfEs", keep_going)
    # Collection skips of test_versioned.py and tests/oracle/, and the marker skip in setup, become errors; the
    # skipped oracle test fails; the expected failure and the ordinary skip stay as they were.
    result.assert_outcomes(passed=1, failed=1, errors=3, xfailed=1, skipped=1)
    output = result.stdout.str()
    must_run = "REQUIRE_ORACLES is set, so this oracle check must run, but it was skipped: "
    assert f"{must_run}could not import 'an_oracle_library_that_is_not_installed'" in output
    assert f"{must_run}skipped by a marker" in output
    assert f"{must_run}module 'QuantLib' has __version__ None, required is: '999'" in output
    assert f"{must_run}could not import 'another_missing_library'" in output
    assert "SKIPPED [1] tests/test_oracles.py" in output and "other tests may still skip" in output
    # The CI command: the collection errors stop the run before any test, which fails the job.
    assert pytester.runpytest_subprocess("-m", "oracle").ret == pytest.ExitCode.INTERRUPTED


def test_required_oracles_that_do_not_import_stop_the_run(pytester, monkeypatch):
    _oracle_session(pytester)
    # As if a release dropped the py_lets_be_rational alias.
    pytester.makepyfile(py_lets_be_rational='raise ImportError("the alias module is gone")')
    monkeypatch.setenv("REQUIRE_ORACLES", "1")
    result = pytester.runpytest_subprocess("-m", "oracle", "tests/test_oracles.py")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert (
        "REQUIRE_ORACLES is set, but these oracle libraries do not import: "
        "py_lets_be_rational (ImportError: the alias module is gone)"
    ) in result.stderr.str()
    monkeypatch.setenv("REQUIRE_ORACLES", "0")
    assert pytester.runpytest_subprocess("-m", "oracle", "tests/test_oracles.py").ret == pytest.ExitCode.OK
