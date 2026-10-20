"""Shared test set-up: Streamlit AppTest fixtures, Hypothesis profiles and the network guard.

AppTest fixtures
    ``app`` is the app as a visitor first sees it (default URL, already run, no exception). ``new_app`` is a factory
    for an app that has not run yet, optionally with URL query parameters. Both run ``main.py``, the entry point that
    Streamlit Community Cloud also uses. Streamlit is imported only when a test asks for one of them, so the
    numerical-core tests run without the ``app`` extra.

Hypothesis profiles
    ``dev`` (the default), ``ci`` (the default when the ``CI`` environment variable is set, as on GitHub Actions) and
    ``nightly``. Choose one with ``HYPOTHESIS_PROFILE=<name>`` or ``pytest --hypothesis-profile=<name>``.

Network guard
    Every test runs with outbound connections and host-name look-ups blocked: a test that needs the internet must be
    marked ``@pytest.mark.network``. Loopback and Unix-domain sockets stay allowed, because asyncio's internal
    socket pair and the local servers that browser tests start use them. The guard covers Python's ``socket``
    module, so it also catches urllib, http.client, requests and asyncio; libraries that open sockets in C code
    (libcurl through curl_cffi, for example) bypass it and must be faked in tests.
"""

from __future__ import annotations

import contextlib
import functools
import ipaddress
import os
import socket
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
_LOCAL_NAMES = frozenset({"", "localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"})


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


def _blocked(what: str, target: object) -> NetworkAccessBlocked:
    return NetworkAccessBlocked(
        f"Network access is blocked in tests ({what} {target!r}). Mark the test @pytest.mark.network if it really "
        "needs the internet, or fake the call."
    )


def _guard_socket_method(method: Callable[..., Any], what: str) -> Callable[..., Any]:
    """``connect``/``connect_ex``/``sendto``: the address is the last positional argument."""

    @functools.wraps(method)
    def guarded(sock: socket.socket, *args: Any) -> Any:
        if sock.family in _INET and args:
            address = args[-1]
            if isinstance(address, tuple) and address and not _is_local(address[0]):
                raise _blocked(what, address)
        return method(sock, *args)

    return guarded


def _guard_lookup(function: Callable[..., Any], what: str) -> Callable[..., Any]:
    """``getaddrinfo``/``gethostbyname``/``gethostbyname_ex``: the host is the first argument."""

    @functools.wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        host = args[0] if args else kwargs.get("host")
        if not _is_local(host):
            raise _blocked(what, host)
        return function(*args, **kwargs)

    return guarded


@contextlib.contextmanager
def network_blocked() -> Iterator[None]:
    """Block outbound connections and DNS look-ups (see the module docstring) until the block exits."""
    with pytest.MonkeyPatch.context() as patch:
        for name in ("connect", "connect_ex", "sendto"):
            patch.setattr(socket.socket, name, _guard_socket_method(getattr(socket.socket, name), name))
        for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex"):
            patch.setattr(socket, name, _guard_lookup(getattr(socket, name), name))
        yield


@pytest.hookimpl(wrapper=True)
def pytest_collection(session: pytest.Session) -> Generator[None, object, object]:
    """Importing test modules must not touch the network either."""
    with network_blocked():
        return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Generator[None, object, object]:
    """Applies to every test automatically: setup (fixtures of any scope), the test itself and teardown."""
    if item.get_closest_marker("network") is not None:
        return (yield)
    with network_blocked():
        return (yield)
