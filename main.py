"""Options Pricing & Strategy Analyzer - launch with ``streamlit run main.py`` (also the Streamlit Community Cloud
entry point).

The application lives in the :mod:`options_analyzer` package: the numerical core (pricing, Greeks, strategies,
risk, volatility surfaces) is independent of Streamlit, and :mod:`options_analyzer.ui` contains the interface.
This file and ``options_analyzer/ui/streamlit_app.py`` (run by the installed ``options-analyzer`` command) both
just call :func:`options_analyzer.ui.app.run`. Streamlit serves the ``static/`` folder next to the script it runs,
so ``static/fonts`` here is a copy of ``options_analyzer/ui/static/fonts`` (tests/test_theme.py keeps them equal).
"""

from options_analyzer.ui.app import run

if __name__ == "__main__":
    run()
