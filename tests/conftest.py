"""Shared test set-up: Streamlit AppTest fixtures, Hypothesis profiles, the network guard and the oracle job.

AppTest fixtures
    ``app`` is the app as a visitor first sees it (default URL, already run, no exception). ``new_app`` is a factory
    for an app that has not run yet, optionally with URL query parameters. Both run ``main.py``, the entry point that
    Streamlit Community Cloud also uses. Streamlit is imported only when a test asks for one of them, so the
    numerical-core tests run without the ``app`` extra.

Hypothesis profiles
    ``dev`` (the default), ``ci`` (the default when the ``CI`` environment variable is set, as on GitHub Actions) and
    ``nightly``. Choose one with ``HYPOTHESIS_PROFILE=<name>`` or ``pytest --hypothesis-profile=<name>``.

Network guard
    Every test runs with outbound connections, host-name look-ups and reverse look-ups blocked: a test that needs the
    internet must be marked ``@pytest.mark.network``. Loopback and Unix-domain sockets stay allowed, because asyncio's
    internal socket pair and the local servers that browser tests start use them. The guard covers Python's ``socket``
    module, so it also catches urllib, http.client, requests and asyncio; libraries that open sockets in C code
    (libcurl through curl_cffi, for example) bypass it and must be faked in tests.

    A proxy on this machine (a corporate one such as cntlm or px, mitmproxy, or a sandbox's) would otherwise fetch a
    page on a test's behalf, because the connection to it is a loopback one. So connections to the port of any local
    proxy configured when a test (or collection) starts are blocked, and, for the duration of an unmarked test, the
    ``*_proxy`` environment variables are removed and ``NO_PROXY=*`` is set, so that clients connect directly and the
    guard sees the real destination. Tests marked ``network`` keep the proxy settings.

Oracle job
    With ``REQUIRE_ORACLES=1`` (set by the CI oracle job) the run stops at once if an oracle library does not import,
    and an oracle test that would be skipped fails instead: a test marked ``oracle``, anything collected from a path
    with "oracle" in it (``tests/oracle/``), or a module skipped because an oracle library is missing. Without the
    variable those tests are skipped as usual when the libraries are not installed.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import ipaddress
import os
import socket
import urllib.parse
import urllib.request
from collections.abc import Callable, Generator, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from streamlit.testing.v1 import AppTest

# pytester runs a separate pytest session inside a test (tests/test_harness.py checks this file's guards with it).
pytest_plugins = ["pytester"]

ROOT = Path(__file__).resolve().parents[1]
#: The repository's entry point, which is also the Streamlit Community Cloud entry point.
APP_SCRIPT = ROOT / "main.py"
#: Seconds that one script run may take. The slowest page (an American tree with 2,000 steps on a busy CI runner)
#: needs a few seconds; the generous limit only stops a hung run.
APP_TIMEOUT = 120


# --------------------------------------------------------------------------------------------- AppTest fixtures


@pytest.fixture
def new_app() -> Callable[..., AppTest]:
    """Factory for an app that has not run yet: ``new_app(spot=105, model="Monte Carlo")``.

    Keyword arguments become URL query parameters (values are converted to text), exactly as in a shared link.
    """
    from streamlit.testing.v1 import AppTest

    def make(**query: object) -> AppTest:
        at = AppTest.from_file(str(APP_SCRIPT), default_timeout=APP_TIMEOUT)
        for name, value in query.items():
            at.query_params[name] = str(value)
        return at

    return make


@pytest.fixture
def app(new_app: Callable[..., AppTest]) -> AppTest:
    """The app at its default URL, after its first run, with no exception raised."""
    at = new_app()
    at.run()
    assert not at.exception, at.exception
    return at


# ------------------------------------------------------------------------------------------ Hypothesis profiles

try:
    from hypothesis import HealthCheck, settings
except ModuleNotFoundError:  # Hypothesis is a dev tool (requirements-dev.txt); tests without properties run anyway.
    pass
else:
    # Local runs: 100 examples per property, remembering failing examples in .hypothesis/ so they are replayed
    # first. No deadline, because tree and Monte Carlo prices legitimately take longer than 200 ms on a busy machine.
    settings.register_profile("dev", deadline=None)
    # Pull requests: 300 examples from a fixed seed (the same examples on every run and every worker), no example
    # database, and a reproduction blob printed for any failure.
    settings.register_profile(
        "ci",
        max_examples=300,
        derandomize=True,
        database=None,
        deadline=None,
        print_blob=True,
        suppress_health_check=[HealthCheck.too_slow],
    )
    # Scheduled runs: 5,000 fresh random examples per property each night.
    settings.register_profile("nightly", settings.get_profile("ci"), max_examples=5000, derandomize=False)
    settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE") or ("ci" if os.environ.get("CI") else "dev"))


# ------------------------------------------------------------------------------------------------ Network guard


class NetworkAccessBlocked(RuntimeError):
    """A test without ``@pytest.mark.network`` tried to reach another machine."""


_INET = (socket.AF_INET, socket.AF_INET6)
#: Names of this machine. Its own host name is included because ``socket.getfqdn("")`` (which ``http.server`` calls
#: for a server bound to all interfaces) looks it up.
_LOCAL_NAMES = frozenset(
    {"", "localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback", socket.gethostname().lower()}
)
#: The port a client uses for a proxy URL that names none.
_PROXY_DEFAULT_PORTS = {"https": 443, "socks4": 1080, "socks4a": 1080, "socks5": 1080, "socks5h": 1080}


def _is_local(host: object) -> bool:
    """True for this machine: no host, ``localhost``, a loopback address, or the unspecified (bind-all) address."""
    if host is None:
        return True
    if isinstance(host, (bytes, bytearray)):
        host = host.decode("ascii", "replace")
    name = str(host).strip("[]").split("%", 1)[0].lower()  # IPv6 literals may carry brackets and a zone index
    if name in _LOCAL_NAMES:
        return True
    try:
        ip = ipaddress.ip_address(name)
    except ValueError:
        return False  # any other host name: resolving it would query DNS
    mapped = getattr(ip, "ipv4_mapped", None)
    return ip.is_loopback or ip.is_unspecified or (mapped is not None and mapped.is_loopback)


def _local_proxy_ports() -> frozenset[int]:
    """Ports of the proxies on this machine that the environment (or the system settings) configure for clients."""
    ports = set()
    for scheme, value in urllib.request.getproxies().items():
        if scheme == "no" or not value:
            continue
        try:
            url = urllib.parse.urlsplit(value if "://" in value else f"http://{value}")
            port = url.port or _PROXY_DEFAULT_PORTS.get(url.scheme, 80)
        except ValueError:  # not a proxy URL (an unrelated variable whose name ends in _proxy)
            continue
        if url.hostname is not None and _is_local(url.hostname):
            ports.add(port)
    return frozenset(ports)


def _blocked(what: str, target: object) -> NetworkAccessBlocked:
    return NetworkAccessBlocked(
        f"Network access is blocked in tests ({what} {target!r}). Mark the test @pytest.mark.network if it really "
        "needs the internet, or fake the call."
    )


def _guard_socket_method(
    method: Callable[..., Any], what: str, position: int, proxy_ports: frozenset[int]
) -> Callable[..., Any]:
    """``connect``/``connect_ex``/``sendto``/``sendmsg``: the address is the positional argument at ``position``."""

    @functools.wraps(method)
    def guarded(sock: socket.socket, *args: Any) -> Any:
        if sock.family in _INET and -len(args) <= position < len(args):
            address = args[position]
            if isinstance(address, tuple) and address:
                if not _is_local(address[0]):
                    raise _blocked(what, address)
                if len(address) > 1 and address[1] in proxy_ports:
                    raise _blocked(f"{what} to the local proxy", address)
        return method(sock, *args)

    return guarded


def _guard_lookup(function: Callable[..., Any], what: str) -> Callable[..., Any]:
    """``getaddrinfo``/``gethostbyname``/``gethostbyname_ex``/``gethostbyaddr``: the host is the first argument."""

    @functools.wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        host = args[0] if args else kwargs.get("host")
        if not _is_local(host):
            raise _blocked(what, host)
        return function(*args, **kwargs)

    return guarded


def _guard_getnameinfo(function: Callable[..., Any]) -> Callable[..., Any]:
    """``getnameinfo((host, port), flags)``: a reverse look-up of the host unless ``NI_NUMERICHOST`` is set."""

    @functools.wraps(function)
    def guarded(*args: Any) -> Any:
        sockaddr = args[0] if args else None
        flags = args[1] if len(args) > 1 else 0
        numeric = isinstance(flags, int) and bool(flags & socket.NI_NUMERICHOST)
        if isinstance(sockaddr, tuple) and sockaddr and not numeric and not _is_local(sockaddr[0]):
            raise _blocked("getnameinfo", sockaddr)
        return function(*args)

    return guarded


@contextlib.contextmanager
def network_blocked(*, clear_proxy_settings: bool = True) -> Iterator[None]:
    """Block outbound connections and DNS look-ups (see the module docstring) until the block exits.

    ``clear_proxy_settings=False`` keeps the proxy environment variables; connections to a local proxy are still
    blocked.
    """
    proxy_ports = _local_proxy_ports()  # before the proxy variables are removed
    with pytest.MonkeyPatch.context() as patch:
        # sendto(data[, flags], address); sendmsg(buffers[, ancdata[, flags[, address]]])
        for name, position in (("connect", -1), ("connect_ex", -1), ("sendto", -1), ("sendmsg", 3)):
            if hasattr(socket.socket, name):  # Windows has no sendmsg
                method = getattr(socket.socket, name)
                patch.setattr(socket.socket, name, _guard_socket_method(method, name, position, proxy_ports))
        for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
            patch.setattr(socket, name, _guard_lookup(getattr(socket, name), name))
        patch.setattr(socket, "getnameinfo", _guard_getnameinfo(socket.getnameinfo))
        if clear_proxy_settings:
            # Clients then connect directly, so the guard sees the real destination. urllib (and so requests and
            # httpx) reads any <scheme>_proxy variable in either case, and NO_PROXY=* also overrides the system proxy
            # settings that urllib falls back to on macOS and Windows.
            for name in list(os.environ):
                if name.lower().endswith("_proxy") and name.lower() != "no_proxy":
                    patch.delenv(name)
            patch.setenv("NO_PROXY", "*")
            patch.setenv("no_proxy", "*")
            # urlopen() keeps the opener (and so the proxy settings) of its first call; one made inside the block
            # must not serve a later network test, nor one made earlier the test inside it.
            patch.setattr(urllib.request, "_opener", None, raising=False)
        yield


@pytest.hookimpl(wrapper=True)
def pytest_collection(session: pytest.Session) -> Generator[None, object, object]:
    """Importing test modules must not touch the network either.

    The proxy settings stay, so that a client a module creates on import (which may read them once) still has them
    in network tests; a connection to a local proxy is blocked all the same.
    """
    with network_blocked(clear_proxy_settings=False):
        return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Generator[None, object, object]:
    """Applies to every test automatically: setup (fixtures of any scope), the test itself and teardown."""
    if item.get_closest_marker("network") is not None:
        return (yield)
    with network_blocked():
        return (yield)


# --------------------------------------------------------------------------------------------------- Oracle job

#: The independent libraries the oracle tests compare against: distribution name (as pinned in
#: requirements-oracle.txt or requirements-dev.txt) -> import name.
ORACLE_LIBRARIES = {"QuantLib": "QuantLib", "lets-be-rational": "py_lets_be_rational", "mpmath": "mpmath"}


def pytest_configure(config: pytest.Config) -> None:
    """With ``REQUIRE_ORACLES=1``, every oracle library must import before any test runs, and no oracle check skips."""
    if os.environ.get("REQUIRE_ORACLES", "").strip() in {"", "0"}:
        return
    missing = []
    for module in ORACLE_LIBRARIES.values():
        try:
            importlib.import_module(module)
        except Exception as error:  # an ImportError, or any error the library raises while it loads
            missing.append(f"{module} ({type(error).__name__}: {error})")
    if missing:
        raise pytest.UsageError(
            f"REQUIRE_ORACLES is set, but these oracle libraries do not import: {'; '.join(missing)}. "
            "Install requirements-oracle.txt, or unset REQUIRE_ORACLES to skip the oracle tests."
        )
    config.pluginmanager.register(_RequireOracles(), "require-oracles")


def _fail_instead_of_skip(report: pytest.CollectReport | pytest.TestReport) -> None:
    longrepr = report.longrepr
    reason = str(longrepr[2]) if isinstance(longrepr, tuple) and len(longrepr) == 3 else str(longrepr)
    report.outcome = "failed"
    report.longrepr = (
        "REQUIRE_ORACLES is set, so this oracle check must run, but it was skipped: " + reason.removeprefix("Skipped: ")
    )


class _RequireOracles:
    """Turns skips of oracle checks into failures. It is a plugin, not hooks of this conftest, because pytest calls
    no conftest's hooks for a directory whose own conftest skipped while loading (tests/oracle/ without QuantLib)."""

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_makereport(self, item: pytest.Item, call: pytest.CallInfo[None]) -> Generator[None, Any, Any]:
        """A test marked oracle that skips (``importorskip`` or ``skip`` in the test or a fixture, a skip marker)."""
        report = yield
        if (
            report.skipped
            and not hasattr(report, "wasxfail")  # an expected failure is reported as skipped too
            and item.get_closest_marker("oracle") is not None
        ):
            _fail_instead_of_skip(report)
        return report

    @pytest.hookimpl(wrapper=True)
    def pytest_make_collect_report(self, collector: pytest.Collector) -> Generator[None, Any, Any]:
        """A module or directory that skips while being collected, if it holds oracle tests.

        Its tests are not known then (the module did not import), so it counts as oracle tests when its path has
        "oracle" in it, as ``tests/oracle/`` does, or when the skip names an oracle library.
        """
        report = yield
        if report.skipped:
            path = report.nodeid.split("::", 1)[0].lower()
            reason = str(report.longrepr)
            modules = ORACLE_LIBRARIES.values()
            if "oracle" in path or any(f"'{module}'" in reason or f"'{module}." in reason for module in modules):
                _fail_instead_of_skip(report)
        return report
