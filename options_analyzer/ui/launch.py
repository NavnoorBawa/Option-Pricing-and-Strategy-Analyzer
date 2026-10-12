"""``options-analyzer`` console command: starts the Streamlit app of the installed package."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    from streamlit.web import cli

    sys.argv = ["streamlit", "run", str(Path(__file__).with_name("streamlit_app.py")), *sys.argv[1:]]
    sys.exit(cli.main())
