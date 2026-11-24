"""HotelSight – Hotel Booking Cancellation Prediction (FYP prototype).

Launch:  python -m streamlit run app/streamlit_app.py
"""
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="HotelSight – Reservation Cancellation Risk",
                   page_icon=str(Path(__file__).resolve().parents[1] / "assets" / "logo_icon.svg"), layout="wide")

from common import apply_style, currency_control, footer, sidebar_model_card   # noqa: E402  (after set_page_config)

pages = {
    "": [
        st.Page("views/home.py", title="Overview", icon=":material/space_dashboard:", default=True),
    ],
    "Reservations": [
        st.Page("views/predict.py", title="Check a reservation", icon=":material/event_available:"),
        st.Page("views/batch.py", title="Upload reservations", icon=":material/upload_file:"),
        st.Page("views/history.py", title="Prediction history", icon=":material/history:"),
    ],
    "Model evidence": [
        st.Page("views/comparison.py", title="Model comparison", icon=":material/leaderboard:"),
        st.Page("views/performance.py", title="Test performance", icon=":material/fact_check:"),
        st.Page("views/explanations.py", title="What drives predictions", icon=":material/insights:"),
    ],
    "About": [
        st.Page("views/method.py", title="Method & limitations", icon=":material/menu_book:"),
    ],
}
nav = st.navigation(pages)
apply_style()
currency_control()
sidebar_model_card()
nav.run()
footer()
