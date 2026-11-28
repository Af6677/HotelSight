"""Overview – the hotel at a glance: four numbers, two trends and who to contact now. Details live on the linked pages."""
import sys as _sys, pathlib as _pl  # make app/common.py importable however the page is launched
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

from html import escape

import altair as alt
import pandas as pd
import streamlit as st

from common import (data_stamp, DATA_CURRENT, DATA_HISTORY, OUTCOME, base_chart, booking_history, chart_card, convert_euro_text,
                    currency_code, dash, euros, icon, kpis, money_short, money_view, nav_link, page_header, pct,
                    saved_predictions, section, show_chart, to_display)
from hotelsight import analytics, insights

off = {"delta_color": "off", "delta_arrow": "off", "border": True}
code = currency_code()

head, actions = st.columns([2, 1.45], vertical_alignment="bottom")
with head:
    page_header("Overview", "Hotel performance and cancellation risk",
                "The four numbers that matter, the trend, and who to contact now.")
with actions, st.container(key="ov_actions", horizontal=True, gap="small", horizontal_alignment="right"):
    with st.container(key="ov_primary", width="content"):
        nav_link("views/predict.py", "Check a reservation", ":material/event_available:")
    with st.container(width="content"):
        nav_link("views/batch.py", "Upload reservations", ":material/upload_file:")

mode = st.segmented_control("Data", [DATA_HISTORY, DATA_CURRENT], default=DATA_HISTORY, key="ov_mode",
                            label_visibility="collapsed",
                            help="Booking history dataset: real results of past bookings. Current predicted bookings: "
                                 "expected results of the reservations you checked or uploaded.") or DATA_HISTORY
data_stamp()

# ------------------------------------------------------------------ data
d = booking_history()
m = analytics.monthly(d)
last = m.iloc[-1]
year_ago = m[m["month"] == last["month"] - pd.DateOffset(years=1)]
ya = year_ago.iloc[0] if len(year_ago) else None
h, data = saved_predictions()
window, label, contact, s = None, "", pd.DataFrame(), None
if data is not None:
    window, label = insights.attention_window(data)
    s = insights.summary(window) if len(window) else insights.summary(data.iloc[0:0])
    contact = insights.contact_list(window)
very_high = contact[contact["probability"] >= .70] if len(contact) else contact
dc = analytics.from_predictions(h) if (mode == DATA_CURRENT and h is not None) else pd.DataFrame()
current = analytics.summary(dc) if len(dc) else None
if mode == DATA_CURRENT:
    st.session_state["perf_mode"] = DATA_CURRENT          # open the performance page on the same data


def vs(now, before, pts=False):
    if before is None or before != before or before == 0:
        return None
    return f"{(now - before) * 100:+.1f} pts vs last year" if pts else f"{(now - before) / before:+.1%} vs last year"


# ------------------------------------------------------------------ 1. at a glance: four numbers
if current is not None:
    hist = analytics.summary(d)
    section("At a glance", f"Expected results of your {current['bookings']:,} predicted reservations")
elif mode == DATA_CURRENT:
    section("At a glance", "No predicted reservations yet, so the booking history dataset is shown")
else:
    section("At a glance", f"{last['month']:%B %Y}, the latest month in the booking history dataset")
k = kpis("ov_glance", ["payments", "event_busy", "money_off", "call"], {1: "coral", 2: "coral", 3: "coral"})
if current is not None:
    k[0].metric("Expected room revenue", euros(current["room_revenue"]),
                f"of {euros(current['room_revenue'] + current['revenue_lost'])} booked", **off,
                help="Room rate × nights of the reservations not flagged as likely to cancel.")
    k[1].metric("Likely to cancel", pct(current["cancel_rate"]),
                f"{(current['cancel_rate'] - hist['cancel_rate']) * 100:+.1f} pts vs booking history",
                delta_color="inverse", border=True,
                help="Share of your predicted reservations flagged as likely to cancel, against the real cancellation "
                     "rate in the booking history.")
else:
    k[0].metric("Room revenue", euros(last["room_revenue"]),
                vs(last["room_revenue"], ya is not None and ya["room_revenue"]), border=True)
    k[1].metric("Cancellation rate", pct(last["cancel_rate"]),
                vs(last["cancel_rate"], ya is not None and ya["cancel_rate"], pts=True), delta_color="inverse",
                border=True)
if s is not None:
    scope = "upcoming arrivals" if label.startswith("Arrivals from today") else "latest 4 weeks"
    k[2].metric("Revenue flagged", euros(s["revenue_flagged"]), f"{s['flagged']:,} of {s['bookings']:,} · {scope}",
                **off, help="Room rate × nights of the saved reservations flagged as likely to cancel. Not a forecast "
                            "of lost revenue.")
    k[3].metric("Guests to contact", f"{len(very_high):,}", f"very high risk · {scope}", **off)
else:
    k[2].metric("Revenue flagged", "—", "no saved predictions yet", **off)
    k[3].metric("Guests to contact", "—", "upload arrivals to see", **off)

# ------------------------------------------------------------------ 2. trends: revenue and the weeks at risk
section("Trends", "Last 12 months of revenue, and the arrivals at risk week by week")
left, right = st.columns(2, gap="medium")
trend = (analytics.monthly(dc) if current is not None else m).tail(12).reset_index(drop=True)
with left:
    if len(trend) >= 2:
        trend = trend.assign(rev=to_display(trend["room_revenue"]),
                             tag=["Latest" if i == len(trend) - 1 else "Earlier" for i in range(len(trend))])
        best = trend.loc[trend["room_revenue"].idxmax()]
        with chart_card("card_ov_rev", "How is revenue moving?",
                        f"<b>{best['month']:%B %Y}</b> was the best month (<b>{money_short(best['room_revenue'])}</b>)."):
            show_chart(base_chart(alt.Chart(trend).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X("yearmonth(month):O", title=None,
                        axis=alt.Axis(format="%b %y", formatType="time", labelAngle=0, labelOverlap="greedy")),
                y=alt.Y("rev:Q", title=None, axis=alt.Axis(format="~s")),
                color=alt.Color("tag:N", scale=alt.Scale(domain=["Latest", "Earlier"], range=["#1E40AF", "#93B4EA"]),
                                legend=None),
                tooltip=[alt.Tooltip("month:T", title="Month", format="%B %Y"),
                         alt.Tooltip("rev:Q", title=f"Room revenue ({code})", format=",.0f")]).properties(height=190)))
            nav_link("views/performance_hotel.py", "Open hotel performance", ":material/trending_up:")
with right:
    w = insights.weekly(window) if window is not None and len(window) else pd.DataFrame()
    if len(w):
        peak = w.loc[w["flagged"].idxmax()]
        with chart_card("card_ov_weeks", "Which weeks are at risk?",
                        f"Week of <b>{peak['week']:%d %b %Y}</b>: <b>{int(peak['flagged'])} of {int(peak['bookings'])}"
                        "</b> arrivals likely to cancel.", "risk"):
            long = pd.concat([w.assign(Prediction="Likely to cancel", count=w["flagged"]),
                              w.assign(Prediction="Likely to stay", count=w["bookings"] - w["flagged"])])
            show_chart(base_chart(alt.Chart(long).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
                x=alt.X("yearmonthdate(week):O", title=None,
                        axis=alt.Axis(format="%d %b", formatType="time", labelAngle=0, labelOverlap="greedy")),
                y=alt.Y("count:Q", title=None),
                color=alt.Color("Prediction:N", scale=OUTCOME, legend=alt.Legend(orient="top", title=None)),
                order=alt.Order("Prediction:N"),
                tooltip=[alt.Tooltip("week:T", title="Week of", format="%d %b %Y"), "Prediction:N",
                         alt.Tooltip("count:Q", title="Reservations")]).properties(height=190)))
            nav_link("views/insights.py", "Open the risk outlook", ":material/analytics:")
    else:
        with chart_card("card_ov_weeks", "Which weeks are at risk?"):
            st.caption("Upload upcoming arrivals to see the risk week by week.")

# ------------------------------------------------------------------ 3. act now: who to call, and why
section("Act now", "Who to contact, and why")
if label:
    st.caption(label)
left, right = st.columns([1.6, 1], gap="medium")
with left.container(key="card_ov_contact", height="stretch"):
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
            **cfg, "Risk": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100, width=100),
            "Suggested action": st.column_config.TextColumn(width="large")})
    nav_link("views/actions.py", f"Open the action list ({len(contact):,})", ":material/checklist:")

with right.container(key="card_ov_alerts", height="stretch"):
    st.html('<div class="hs-card-title">Key insights</div>')
    points = []
    if window is not None and len(window):
        top = insights.reasons(window, top=1)
        if len(top):
            points.append(("risk", "warning", f"Main risk reason: <b>{escape(top.iloc[0]['reason'])}</b>. "
                                              f"{escape(top.iloc[0]['action'])}"))
    for f in analytics.findings(d)[:3 - len(points)]:
        points.append((f["kind"] if f["kind"] in ("risk", "good") else "info",
                       {"risk": "warning", "good": "thumb_up"}.get(f["kind"], "insights"),
                       f"<b>{escape(f['title'])}.</b> {escape(f['action'])}"))
    st.html('<div class="hs-alerts">' + "".join(
        f'<div class="hs-alert {kind}">{icon(ic)}<span>{text}</span></div>' for kind, ic, text in points) + '</div>')
    nav_link("views/performance_hotel.py", "See all findings", ":material/lightbulb:")
