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


#: The valuation time that the tests of the synthetic index pin, so that their numbers do not depend on the day:
#: Friday 23 October 2026 at the cash close.
ASOF = "2026-10-23T16:00"
#: The parameter that a link to the textbook market holds, as the app writes it.
TEXTBOOK = {"preset": "Textbook"}


def make_app(**query: object) -> AppTest:
    """An app that has not run yet; keyword arguments become URL query parameters, exactly as in a shared link.

    The app opens on the textbook market (``preset=Textbook``: S = K = 100, one year, σ 20%, r 5%, q 0), which most
    tests check numbers against, unless ``preset`` is given: ``preset="index"`` for the synthetic index (pin ``asof``
    too, :data:`ASOF`), or ``preset=None`` for the URL without the parameter (the app's default market)."""
    from streamlit.testing.v1 import AppTest

    query.setdefault("preset", "Textbook")
    at = AppTest.from_file(str(APP_SCRIPT), default_timeout=APP_TIMEOUT)
    for name, value in query.items():
        if value is not None:
            at.query_params[name] = str(value)
    return at
