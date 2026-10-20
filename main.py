"""Options Pricing & Strategy Analyzer - launch with ``streamlit run main.py`` (also the Streamlit Community Cloud
entry point).

The application lives in the :mod:`options_analyzer` package: the numerical core (pricing, Greeks, strategies,
risk, volatility surfaces) is independent of Streamlit, and :mod:`options_analyzer.ui` contains the interface.
This file only calls :func:`options_analyzer.ui.streamlit_app.main`, the script the installed ``options-analyzer``
command runs. Streamlit serves the ``static/`` folder next to the script it runs, so ``static/fonts`` here is a copy
of ``options_analyzer/ui/static/fonts`` (tests/test_theme.py keeps them equal).
"""

from options_analyzer.ui.streamlit_app import main

if __name__ == "__main__":
    main()
