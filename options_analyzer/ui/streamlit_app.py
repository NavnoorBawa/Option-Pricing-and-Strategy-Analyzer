"""Streamlit script for the installed package (``options-analyzer`` or ``streamlit run`` on this file).

Streamlit executes its target as a top-level script, so this file uses absolute imports only.
"""

from options_analyzer.ui.app import run

run()
