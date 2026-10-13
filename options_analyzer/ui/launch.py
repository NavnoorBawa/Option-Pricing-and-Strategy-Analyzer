"""``options-analyzer`` console command: starts the Streamlit app of the installed package."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    try:
        from streamlit.web import cli
    except ModuleNotFoundError as exc:
        if exc.name != "streamlit":
            raise
        sys.exit(
            "options-analyzer: the app needs the 'app' extra (Streamlit, Plotly, XlsxWriter).\n"
            'Install it with pip install ".[app]" from a checkout of the repository.'
        )

    sys.argv = ["streamlit", "run", str(Path(__file__).with_name("streamlit_app.py")), *sys.argv[1:]]
    sys.exit(cli.main())
