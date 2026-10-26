"""The app's Streamlit script: ``options-analyzer`` (the installed command) and ``streamlit run`` on this file run it,
and ``main.py`` in the repository calls its :func:`main`.

Streamlit reads the ``.streamlit/config.toml`` and serves the ``static/`` folder (the self-hosted fonts) next to the
script it runs, so both ship beside this file as package data. Streamlit executes its target as a top-level script,
so this file uses absolute imports only.
"""

from options_analyzer.ui.app import run


def main() -> None:
    """One script run of the app (:func:`options_analyzer.ui.app.run`)."""
    run()


if __name__ == "__main__":
    main()
