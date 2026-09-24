"""Option pricing, Greeks, strategy analytics, risk and volatility-surface tools.

The numerical core is independent of Streamlit and can be used on its own::

    from options_analyzer import black_scholes as bs
    bs.price(100, 105, 0.5, 0.04, 0.22, q=0.01, option_type="put")

The Streamlit user interface lives in :mod:`options_analyzer.ui` and is
launched with ``streamlit run main.py``.
"""

__version__ = "2.0.0"
