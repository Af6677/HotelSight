"""Overview – what HotelSight does, where to start, the saved-prediction summary and how reliable the model is."""
import sys as _sys, pathlib as _pl  # make app/common.py importable however the page is launched
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

from datetime import date
from html import escape

import pandas as pd
import streamlit as st

from common import euros, friendly, icon, load, nav_link, page_header, pct, section
from hotelsight import history

split = load("plan")["splits"]["retained"]
t = load("test_metrics")
rep, one = t["repeated"], t["one_copy"]

page_header("Reservation risk intelligence", "See which reservations are likely to cancel",
            "HotelSight estimates the cancellation risk of each reservation and shows which booking details drove "
            f"the estimate. Its XGBoost model learned from <b>{split['train']['rows']:,} historical bookings</b> at a "
            f"city hotel and a resort hotel in Portugal, and was checked on {rep['rows']:,} bookings it never saw.",
            chips=["Single reservations", "Excel / CSV uploads", "Explained with SHAP", "Saved history"],
            icon_name="hotel", large=True)

# ------------------------------------------------------------------ where to start
section("Start here")
ACTIONS = [
    ("concierge", "Check one reservation", "Enter the stay dates, room rate, booking channel and guest details to get "
     "an instant risk score with the reasons behind it.", "views/predict.py", "Open the reservation check"),
    ("upload_file", "Upload a reservation list", "Score up to 10,000 reservations from an Excel or CSV file, see the "
     "arrivals and room revenue at risk, and download the results.", "views/batch.py", "Upload reservations"),
    ("history", "Review saved predictions", "Find earlier checks and uploads, open the exact inputs that were used, "
     "and export them for your records.", "views/history.py", "Open the prediction history"),
]
for col, (ic, title, text, page, link) in zip(st.columns(3, gap="medium"), ACTIONS):
    with col.container(key=f"card_go_{ic}", height="stretch"):
        st.html(f'<div class="hs-action">{icon(ic, "big")}<h4>{title}</h4><p>{text}</p></div>')
        nav_link(page, link)

# ------------------------------------------------------------------ saved predictions
section("Saved predictions", "Summary of the prediction history")
h = history.load()
if h.empty:
    st.html(f'<div class="hs-note">{icon("info")}<span>No predictions saved yet. Every reservation check and upload is '
            'saved automatically and summarised here.</span></div>')
else:
    h = history.with_stay(h)
    flagged = h[h["predicted_cancel"] == 1]
    arrival = pd.to_datetime(flagged["arrival"], errors="coerce", format="%Y-%m-%d")
    upcoming = flagged[arrival >= pd.Timestamp(date.today())]
    k = st.columns(4)
    k[0].metric("Saved predictions", f"{len(h):,}", f"{h['batch_id'].nunique():,} checks and uploads",
                delta_color="off", delta_arrow="off", border=True)
    k[1].metric("Likely to cancel", f"{len(flagged):,}", f"{pct(len(flagged) / len(h))} of saved",
                delta_color="off", delta_arrow="off", border=True)
    k[2].metric("Revenue flagged", euros(flagged["stay_value"].sum()), "room rate × nights",
                delta_color="off", delta_arrow="off", border=True,
                help="Sum of room rate × nights for the saved reservations flagged as likely to cancel.")
    k[3].metric("Upcoming at risk", f"{len(upcoming):,}", "flagged, arriving from today", delta_color="off",
                delta_arrow="off", border=True)
    if len(flagged):
        recent = flagged.head(6)
        with st.container(key="card_home_recent"):
            st.html('<div class="hs-card-title">Most recent reservations flagged as likely to cancel</div>')
            st.dataframe(pd.DataFrame({
                "Saved": recent["created_at"], "Reference": recent["reference"], "Property": recent["hotel"],
                "Arrival": recent["arrival"], "Stay value (EUR)": recent["stay_value"],
                "Probability": recent["probability"] * 100, "Main factor raising risk": recent["top_raising"]}),
                hide_index=True, width="stretch",
                column_config={"Saved": st.column_config.DatetimeColumn(format="D MMM YYYY, HH:mm"),
                               "Stay value (EUR)": st.column_config.NumberColumn(format="€%.2f"),
                               "Probability": st.column_config.ProgressColumn(format="%.1f%%", min_value=0,
                                                                              max_value=100)})

# ------------------------------------------------------------------ reliability
section("How reliable is the model?", f"Sealed test set · {rep['rows']:,} reservations the model never saw")
k = st.columns(4)
off = {"delta_color": "off", "delta_arrow": "off", "border": True}
k[0].metric("Accuracy", pct(rep["accuracy"]), "classified correctly", **off,
            help="Share of test reservations classified correctly.")
k[1].metric("Recall", pct(rep["recall"]), "cancellations caught", **off,
            help="Share of the test reservations that were cancelled which the model flagged.")
k[2].metric("Precision", pct(rep["precision"]), "flags that were right", **off,
            help="Share of the flagged test reservations that really were cancelled.")
k[3].metric("ROC AUC", f"{rep['roc_auc']:.3f}", "ranking quality", **off,
            help="How well the scores rank cancelled above kept reservations (1.0 = perfect, 0.5 = chance).")
st.caption(f"F1 is {pct(rep['f1'])} on these rows and {pct(one['f1'])} when each repeated record is counted once. "
           "Full results are on the Test performance page.")

left, right = st.columns([1.15, 1], gap="medium")
with left.container(key="card_home_drivers", height="stretch"):
    imp = load("global_shap").sort_values("mean_abs_shap_repeated", ascending=False).head(7)
    top = float(imp["mean_abs_shap_repeated"].max())
    rows = "".join(f'<div class="hs-bar"><span>{escape(friendly(r.input))}</span><div class="track"><div class="fill" '
                   f'style="width:{100 * r.mean_abs_shap_repeated / top:.1f}%"></div></div>'
                   f'<b>{r.mean_abs_shap_repeated:.2f}</b></div>' for r in imp.itertuples())
    st.html(f'<div class="hs-card-title">What the model pays most attention to</div><div class="hs-bars">{rows}</div>')
    st.caption("Average influence on the test set (mean |SHAP|, log-odds). These are associations learned from "
               "historical records, not causes.")
with right.container(key="card_home_how", height="stretch"):
    st.html('<div class="hs-card-title">How it works</div><div class="hs-steps">'
            '<div><span><b>Enter or upload reservations</b>Stay dates, room rate, channel, deposit and guest details: '
            'the same 24 inputs the model was trained on.</span></div>'
            '<div><span><b>Score the cancellation risk</b>The saved model returns a probability; 0.50 or more is '
            'flagged as likely to cancel.</span></div>'
            '<div><span><b>Explain and record</b>SHAP shows which details raised or lowered the score, and every '
            'prediction is kept in the history.</span></div></div>'
            f'<div class="hs-note">{icon("info")}<span>Research prototype trained on historical records from two '
            'Portuguese hotels (2015–2017). Its accuracy on new or Malaysian reservations is unknown, so use the '
            'scores to prioritise follow-up, not as certainties.</span></div>')
