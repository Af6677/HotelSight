"""Overview – the hotel's condition at a glance and what needs attention now. Details live on the linked pages."""
import sys as _sys, pathlib as _pl  # make app/common.py importable however the page is launched
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

from html import escape

import pandas as pd
import streamlit as st

from common import (DATA_CURRENT, DATA_HISTORY, booking_history, convert_euro_text, dash, euros, icon, kpis,
                    money_view, nav_link, pct, saved_predictions, section)
from hotelsight import analytics, insights

off = {"delta_color": "off", "delta_arrow": "off", "border": True}

with st.container(key="hero"):
    st.html('<div class="hs-hero"><h1>Hotel performance and cancellation risk</h1>'
            '<p>See how the hotels are performing, which reservations are likely to cancel, and what to do about '
            'them.</p></div>')
    with st.container(horizontal=True, gap="small"):
        with st.container(key="hero_primary", width="content"):
            nav_link("views/predict.py", "Check a reservation")
        nav_link("views/batch.py", "Upload reservations")

# ------------------------------------------------------------------ hotel performance, latest month
d = booking_history()
m = analytics.monthly(d)
last = m.iloc[-1]
year_ago = m[m["month"] == last["month"] - pd.DateOffset(years=1)]
ya = year_ago.iloc[0] if len(year_ago) else None


def vs(now, before, pts=False):
    if before is None or before != before or before == 0:
        return None
    return f"{(now - before) * 100:+.1f} pts vs last year" if pts else f"{(now - before) / before:+.1%} vs last year"


h, data = saved_predictions()
mode = st.segmented_control("Data", [DATA_HISTORY, DATA_CURRENT], default=DATA_HISTORY, key="ov_mode",
                            label_visibility="collapsed",
                            help="Booking history dataset: real results of past bookings. Current predicted bookings: "
                                 "expected results of the reservations you checked or uploaded.") or DATA_HISTORY
current = None
if mode == DATA_CURRENT:
    dc = analytics.from_predictions(h) if h is not None else pd.DataFrame()
    if dc.empty:
        section("Current predicted bookings", "Expected results from the model")
        st.html(f'<div class="hs-note">{icon("info")}<span>No predictions saved yet. Check a reservation or upload a '
                'file, and their expected results appear here.</span></div>')
    else:
        current = analytics.summary(dc)
        hist = analytics.summary(d)
        section(f"Current predicted bookings · {current['bookings']:,} reservations",
                "Expected results from the model, compared with the booking history dataset")
        k = kpis("ov_perf", ["payments", "sell", "bed", "event_busy"], {3: "coral"})
        k[0].metric("Expected room revenue", euros(current["room_revenue"]),
                    f"of {euros(current['room_revenue'] + current['revenue_lost'])} booked", **off,
                    help="Room rate × nights of the reservations not flagged as likely to cancel.")
        k[1].metric("Expected ADR", euros(current["adr"]), f"{euros(hist['adr'])} in the booking history", **off)
        k[2].metric("Expected rooms sold", f"{int(current['rooms_sold']):,}", "room nights expected to stay", **off)
        k[3].metric("Likely to cancel", pct(current["cancel_rate"]),
                    f"{(current['cancel_rate'] - hist['cancel_rate']) * 100:+.1f} pts vs booking history",
                    delta_color="inverse", border=True,
                    help="Share of your predicted reservations flagged as likely to cancel, against the real "
                         "cancellation rate in the booking history.")
else:
    section(f"Hotel performance · {last['month']:%B %Y}", "Latest month in the booking history dataset, both hotels")
    k = kpis("ov_perf", ["payments", "sell", "bed", "event_busy"], {3: "coral"})
    k[0].metric("Room revenue", euros(last["room_revenue"]),
                vs(last["room_revenue"], ya is not None and ya["room_revenue"]), border=True)
    k[1].metric("Average daily rate", euros(last["adr"]), vs(last["adr"], ya is not None and ya["adr"]), border=True)
    k[2].metric("Rooms sold", f"{int(last['rooms_sold']):,}",
                vs(last["rooms_sold"], ya is not None and ya["rooms_sold"]), border=True)
    k[3].metric("Cancellation rate", pct(last["cancel_rate"]),
                vs(last["cancel_rate"], ya is not None and ya["cancel_rate"], pts=True), delta_color="inverse",
                border=True)
if mode == DATA_CURRENT:
    st.session_state["perf_mode"] = DATA_CURRENT          # open the performance page on the same data
nav_link("views/performance_hotel.py", "Open hotel performance", ":material/trending_up:")

# ------------------------------------------------------------------ reservations that need attention
section("Reservations that need attention", "From the saved predictions")
if data is None:
    st.html(f'<div class="hs-note">{icon("info")}<span>No predictions saved yet. Check a reservation or upload a file, '
            'and the reservations at risk appear here.</span></div>')
    window, label, contact = None, "", pd.DataFrame()
else:
    window, label = insights.attention_window(data)
    s = insights.summary(window) if len(window) else insights.summary(data.iloc[0:0])
    contact = insights.contact_list(window)
    very_high = contact[contact["probability"] >= .70]
    st.caption(label)
    k = kpis("ov_risk", ["luggage", "event_busy", "payments", "call"], {1: "coral", 2: "coral", 3: "coral"})
    k[0].metric("Arrivals", f"{s['bookings']:,}", "saved reservations", **off)
    k[1].metric("Likely to cancel", pct(s["share_flagged"]), f"{s['flagged']:,} reservations", **off)
    k[2].metric("Revenue flagged", euros(s["revenue_flagged"]), f"of {euros(s['revenue'])} booked", **off,
                help="Room rate × nights of the reservations flagged as likely to cancel. Not a forecast of lost revenue.")
    k[3].metric("Guests to contact", f"{len(very_high):,}", "very high risk (70% and above)", **off)

left, right = st.columns([1, 1.25], gap="medium")
with left.container(key="card_ov_alerts", height="stretch"):
    st.html('<div class="hs-card-title">Management highlights</div>')
    alerts = []
    if current is not None:
        booked = current["room_revenue"] + current["revenue_lost"]
        share = current["revenue_lost"] / booked if booked else 0
        alerts.append(("risk", "payments", f"Of <b>{euros(booked)}</b> booked in your predicted reservations, "
                                           f"<b>{euros(current['revenue_lost'])}</b> ({share:.0%}) is at risk, so the "
                                           f"expected room revenue is <b>{euros(current['room_revenue'])}</b>."))
    elif ya is not None:
        g = (last["room_revenue"] - ya["room_revenue"]) / ya["room_revenue"]
        alerts.append(("good" if g >= 0 else "risk", "trending_up" if g >= 0 else "trending_down",
                       f"Room revenue in <b>{last['month']:%B %Y}</b> was <b>{g:+.1%}</b> against the same month last "
                       f"year, with ADR at <b>{euros(last['adr'])}</b>."))
        c = (last["cancel_rate"] - ya["cancel_rate"]) * 100
        if abs(c) >= 1:
            alerts.append(("risk" if c > 0 else "good", "event_busy",
                           f"The cancellation rate {'rose' if c > 0 else 'fell'} by <b>{abs(c):.1f} points</b> to "
                           f"<b>{pct(last['cancel_rate'])}</b>, costing <b>{euros(last['revenue_lost'])}</b> in booked "
                           "room revenue."))
    if data is not None and len(window):
        top = insights.reasons(window, top=1)
        if len(top):
            alerts.append(("risk", "warning", f"The most common reason for risk is <b>{escape(top.iloc[0]['reason'])}</b> "
                                              f"({top.iloc[0]['share_of_flagged']:.0%} of flagged reservations). "
                                              f"{escape(top.iloc[0]['action'])}"))
    fs = {f["title"]: f for f in analytics.findings(d)}
    if "When guests cancel" in fs:
        alerts.append(("info", "schedule", "In the booking history dataset: "
                       + escape(fs["When guests cancel"]["finding"][0].lower() + fs["When guests cancel"]["finding"][1:])))
    st.html('<div class="hs-alerts">' + "".join(
        f'<div class="hs-alert {kind}">{icon(ic)}<span>{text}</span></div>' for kind, ic, text in alerts) + '</div>')
    nav_link("views/performance_hotel.py", "See all findings and actions", ":material/lightbulb:")

with right.container(key="card_ov_contact", height="stretch"):
    st.html('<div class="hs-card-title">Contact first</div>')
    if contact.empty:
        st.caption("No saved reservation in this period is flagged as likely to cancel.")
    else:
        top5 = contact.head(5)
        view, cfg = money_view(pd.DataFrame({
            "Reference": dash(top5["reference"]), "Arrival": top5["arrival"].dt.strftime("%d %b %Y"),
            "Risk": top5["probability"] * 100, "Stay value": top5["stay_value"],
            "Suggested action": top5["action"].map(convert_euro_text)}), ["Stay value"])
        st.dataframe(view, hide_index=True, width="stretch", column_config={
            **cfg, "Risk": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100, width=110)})
    c1, c2 = st.columns(2)
    with c1:
        nav_link("views/actions.py", "Open the action list", ":material/checklist:")
    with c2:
        nav_link("views/insights.py", "Open the risk outlook", ":material/analytics:")
