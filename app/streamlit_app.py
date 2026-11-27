"""HotelSight – Hotel Booking Cancellation Prediction (FYP prototype).

Launch:  python -m streamlit run app/streamlit_app.py
"""
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="HotelSight – Hotel Performance and Cancellation Risk",
                   page_icon=str(Path(__file__).resolve().parents[1] / "assets" / "logo_icon.svg"), layout="wide")

from common import apply_style, currency_control, footer, sidebar_model_card   # noqa: E402  (after set_page_config)

pages = {
    "": [
        st.Page("views/home.py", title="Overview", icon=":material/space_dashboard:", default=True),
    ],
    "Predict": [
        st.Page("views/predict.py", title="Check a reservation", icon=":material/event_available:"),
        st.Page("views/batch.py", title="Upload reservations", icon=":material/upload_file:"),
    ],
    "Analyse": [
        st.Page("views/insights.py", title="Risk outlook", icon=":material/analytics:"),
        st.Page("views/actions.py", title="Action list", icon=":material/checklist:"),
        st.Page("views/performance_hotel.py", title="Hotel performance", icon=":material/trending_up:"),
    ],
    "Records": [
        st.Page("views/history.py", title="Prediction history", icon=":material/history:"),
    ],
    "Model": [
        st.Page("views/evaluation.py", title="Model evaluation", icon=":material/fact_check:"),
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
