"""Streamlit script for the installed package (``options-analyzer`` or ``streamlit run`` on this file).

Like ``main.py`` in the repository, it only calls :func:`options_analyzer.ui.app.run`. Streamlit reads the
``.streamlit/config.toml`` and serves the ``static/`` folder (the self-hosted fonts) next to the script it runs, so
both ship beside this file as package data. Streamlit executes its target as a top-level script, so this file uses
absolute imports only.
"""

from options_analyzer.ui.app import run

run()
