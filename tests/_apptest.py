"""Building Streamlit AppTests of the app, shared by ``conftest.py`` and the test modules (see ``_oracles.py`` for why
this is not in ``conftest.py``)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
#: The repository's entry point, which is also the Streamlit Community Cloud entry point.
APP_SCRIPT = ROOT / "main.py"
#: Seconds that one script run may take. The slowest page (an American tree with 2,000 steps on a busy CI runner)
#: needs a few seconds; the generous limit only stops a hung run.
APP_TIMEOUT = 120


def make_app(**query: object) -> AppTest:
    """An app that has not run yet; keyword arguments become URL query parameters, exactly as in a shared link."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_SCRIPT), default_timeout=APP_TIMEOUT)
    for name, value in query.items():
        at.query_params[name] = str(value)
    return at
