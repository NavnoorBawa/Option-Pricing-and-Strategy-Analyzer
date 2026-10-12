"""Options Pricing & Strategy Analyzer - launch with ``streamlit run main.py``.

The application lives in the :mod:`options_analyzer` package: the numerical
core (pricing, Greeks, strategies, risk, volatility surfaces) is independent
of Streamlit, and :mod:`options_analyzer.ui` contains the interface.
"""

from options_analyzer.ui.app import run

if __name__ == "__main__":
    run()
