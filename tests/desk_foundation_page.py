"""A page of the desk foundation's shared pieces, for the browser tests of tests/test_desk_foundation_e2e.py: the
what-if strip, desk tables of 15 and 20 rows and one chart, in the app's theme and stylesheet. Run from the
repository root (``python -m streamlit run tests/desk_foundation_page.py``), so that the package imports and the
theme in .streamlit/config.toml applies."""

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from options_analyzer.ui import charts, components, theme, whatif

st.set_page_config(page_title="Desk foundation", layout="wide")
theme.apply_stylesheet()
# A sidebar, so that the main column is as wide as the app's (1,038 px at a 1,440 px window).
st.sidebar.number_input("Strike", value=5815.0, step=5.0, key="page_strike")
st.sidebar.selectbox("Expiry", ["20-Nov-26 AM · 28d", "18-Dec-26 AM · 56d"], key="page_expiry")
st.sidebar.number_input("Locked", value=1.0, disabled=True, key="page_locked")
shifts = whatif.strip(
    spot=5800.0,
    atm_vol=0.1552,
    tenor=55.7 / 365,
    rate=0.0475,
    as_of=datetime(2026, 10, 23, 16, 0),
    horizon=datetime(2026, 12, 18, 9, 30),
)
st.caption(whatif.note(shifts) or "No scenario.")
strikes = np.arange(5700.0, 5800.0, 5.0)
frame = pd.DataFrame(
    {
        "Strike": strikes,
        "IV": 15.5 - (strikes - 5750.0) * 0.01,
        "Mid vs fit": np.linspace(-0.4, 0.4, len(strikes)),
        "P&L": np.linspace(-7760.0, 2240.0, len(strikes)),
    }
)
kinds = {"Strike": "strike", "IV": "vol", "Mid vs fit": "vol_change", "P&L": "pnl"}
help = {c: f"About the {c} column." for c in frame}
components.table(frame.head(15), kinds, help, key="tbl_fifteen")
components.table(frame, kinds, help, key="tbl_twenty")
figure = go.Figure(go.Scatter(x=strikes, y=frame["P&L"], name="P&L"))
charts.show(figure, key="foundation_chart", alt="Profit and loss of a sample position against the strike")
