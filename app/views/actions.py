"""Action list – which reservations to contact, in what order, and what to do. Built for the reservations team."""
import sys as _sys, pathlib as _pl  # make app/common.py importable however the page is launched
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from common import (data_stamp, convert_euro_text, dash, euros, kpis, money_view, page_header, prediction_filters,
                    saved_predictions)
from hotelsight import batch, insights

XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
off = {"delta_color": "off", "delta_arrow": "off", "border": True}

page_header("Analyse", "Action list",
            "The reservations flagged as likely to cancel, soonest arrival first, each with the main reason and a "
            "suggested action for the reservations team.")
data_stamp()

h, data = saved_predictions()
if data is None:
    st.info("No saved predictions yet. Check a reservation or upload a file first, and the action list appears here.",
            icon=":material/checklist:")
    st.stop()

_, sel, _ = prediction_filters(data, "act")
todo = insights.contact_list(sel)

with st.container(key="card_act_more"):
    c1, c2, c3 = st.columns([1.4, 1.4, 1])
    bands = c1.multiselect("Risk band", insights.BAND_LABELS[2:], default=insights.BAND_LABELS[2:], key="act_bands",
                           help="Reservations from 50% are flagged as likely to cancel.")
    reasons = ["All reasons"] + sorted(r for r in todo["reason_factor"].unique() if r)
    reason = c2.selectbox("Main reason", reasons, key="act_reason")
    prop = c3.selectbox("Property", ["Both", "City Hotel", "Resort Hotel"], key="act_property")

if bands:
    todo = todo[todo["risk_band"].astype(str).isin(bands)]
if reason != "All reasons":
    todo = todo[todo["reason_factor"] == reason]
if prop != "Both":
    todo = todo[todo["hotel"] == prop]

k = kpis("act", ["checklist", "warning", "payments", "event"], {0: "coral", 1: "coral", 2: "coral"})
k[0].metric("Reservations to contact", f"{len(todo):,}", "flagged, each listed once", **off)
k[1].metric("Very high risk", f"{int((todo['probability'] >= .70).sum()):,}", "70% and above", **off)
k[2].metric("Stay value at risk", euros(todo["stay_value"].sum()), "room rate × nights", **off)
first = todo["arrival"].min()
k[3].metric("First arrival", f"{first:%d %b %Y}" if pd.notna(first) else "—", "contact before this date", **off)

if todo.empty:
    st.info("Nothing to contact for these filters.", icon=":material/task_alt:")
    st.stop()

table = pd.DataFrame({"Arrival": todo["arrival"].dt.strftime("%Y-%m-%d"), "Reference": dash(todo["reference"]),
                      "Risk": todo["probability"] * 100, "Main reason": todo["reason"].map(convert_euro_text),
                      "Suggested action": todo["action"], "Stay value (EUR)": todo["stay_value"],
                      "Property": todo["hotel"], "Booking source": todo["market_segment"],
                      "Risk band": todo["risk_band"].astype(str)})
view, cfg = money_view(table, ["Stay value (EUR)"])
with st.container(key="card_act_table"):
    st.html('<div class="hs-card-title">Contact list</div>')
    st.dataframe(view, hide_index=True, width="stretch", height=min(38 + 35 * len(view), 560), column_config={
        **cfg, "Risk": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100, width=110),
        "Suggested action": st.column_config.TextColumn(width="large")})
    st.caption("Sorted by arrival date, then by risk. A reservation saved more than once is listed once. Suggested "
               "actions come from patterns in past bookings, so they are for the team to judge.")

export = table.assign(Risk=todo["probability"].round(4).to_numpy())
d1, d2, _ = st.columns([1, 1, 2])
d1.download_button("Download the action list (Excel)", batch.to_excel({"Action list": export}),
                   file_name="HotelSight_action_list.xlsx", mime=XLSX, icon=":material/download:", width="stretch",
                   key="act_xlsx")
d2.download_button("Download the action list (CSV)", export.to_csv(index=False).encode("utf-8-sig"),
                   file_name="HotelSight_action_list.csv", mime="text/csv", icon=":material/download:",
                   width="stretch", key="act_csv")
