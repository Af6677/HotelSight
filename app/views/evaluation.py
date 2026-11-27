"""Model evaluation – can the model be trusted? Model selection, sealed test results and sensitivity checks.
How accurate the saved predictions turned out to be is on the Prediction history page."""
import sys as _sys, pathlib as _pl  # make app/common.py importable however the page is launched
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import streamlit as st

from common import load, nav_link, page_header
from evidence import render_selection, render_sensitivity, render_test

sel = load("selected")
page_header("Model", "Model evaluation",
            f"How {sel['model']} was chosen and how well it predicted cancellations on bookings it had never seen.")

tab_test, tab_sel, tab_sens = st.tabs([":material/fact_check: Test results", ":material/leaderboard: Model selection",
                                       ":material/tune: Sensitivity checks"])
with tab_test:
    render_test()
with tab_sel:
    render_selection()
with tab_sens:
    render_sensitivity()
nav_link("views/history.py", "See how accurate the saved predictions were", ":material/fact_check:")
