"""Risk outlook – what the saved predictions say about upcoming arrivals: when the risk falls, where it comes from,
how it is spread and what to do about it. The list of individual reservations is on the Action list page."""
import sys as _sys, pathlib as _pl  # make app/common.py importable however the page is launched
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import altair as alt
import pandas as pd
import streamlit as st

from common import (ALL_WEEKS, CANCEL, INK, OUTCOME, STAY, base_chart, currency_code, euros, kpis, money_view,
                    nav_link, page_header, pct, prediction_filters, saved_predictions, section, show_chart,
                    to_display)
from hotelsight import insights

BAND_COLOURS = ["#2A78D6", "#9EC5F4", "#F09595", "#E34948"]
off = {"delta_color": "off", "delta_arrow": "off", "border": True}

page_header("Analyse", "Risk outlook",
            "What the saved predictions say about the arrivals: how many are likely to cancel, when, where the risk "
            "comes from and what the hotel can do about it.")

h, data = saved_predictions()
if data is None:
    st.info("No saved predictions yet. Check a reservation or upload a file first, and the outlook appears here.",
            icon=":material/analytics:")
    st.stop()

d, sel, week = prediction_filters(data, "ins")
if sel.empty:
    st.warning("No saved predictions match these filters.", icon=":material/filter_alt_off:")
    st.stop()

prev = None
if week != ALL_WEEKS:
    before = d[d["week"] == week - pd.Timedelta(days=7)]
    prev = insights.summary(before) if len(before) else None

# ------------------------------------------------------------------ headline figures
s = insights.summary(sel)
section("This selection" if week == ALL_WEEKS else f"Week of {week:%d %b %Y}",
        "Compared with the previous week" if prev else "Saved predictions in the filters above")
k = kpis("ins", ["luggage", "event_busy", "payments", "how_to_reg", "bed"], {1: "coral", 2: "coral", 4: "coral"})
if prev:
    k[0].metric("Arrivals", f"{s['bookings']:,}", f"{s['bookings'] - prev['bookings']:+,} vs previous week",
                delta_color="off", border=True)
    k[1].metric("Likely to cancel", pct(s["share_flagged"]),
                f"{(s['share_flagged'] - prev['share_flagged']) * 100:+.1f} pts vs previous week",
                delta_color="inverse", border=True)
    diff = s["revenue_flagged"] - prev["revenue_flagged"]
    k[2].metric("Revenue flagged", euros(s["revenue_flagged"]),
                f"{'+' if diff >= 0 else '-'}{euros(abs(diff))} vs previous week", delta_color="inverse", border=True,
                help="Room rate × nights of the reservations flagged as likely to cancel. Not a forecast of lost "
                     "revenue.")
else:
    k[0].metric("Arrivals", f"{s['bookings']:,}", "saved reservations", **off)
    k[1].metric("Likely to cancel", pct(s["share_flagged"]), f"{s['flagged']:,} reservations", **off)
    k[2].metric("Revenue flagged", euros(s["revenue_flagged"]), f"of {euros(s['revenue'])} booked", **off,
                help="Room rate × nights of the reservations flagged as likely to cancel. Not a forecast of lost "
                     "revenue.")
k[3].metric("Expected arrivals", f"{s['expected_arrivals']:,}", "booked minus likely to cancel", **off,
            help="Reservations not flagged as likely to cancel. A planning guide, not a guarantee.")
k[4].metric("Room nights at risk", f"{int(s['nights_flagged']):,}", "nights of flagged reservations", **off)

# ------------------------------------------------------------------ over time
w = insights.weekly(d)
code = currency_code()
with st.container(key="card_ins_weeks"):
    c1, c2 = st.columns([2.2, 1], vertical_alignment="center")
    with c1:
        st.html('<div class="hs-card-title">Outlook over time</div>')
        view = st.segmented_control("Show", ["Arrivals", "Share likely to cancel", "Revenue flagged",
                                             "Expected arrivals"], default="Arrivals", key="ins_week_view",
                                    label_visibility="collapsed")
    grain = c2.segmented_control("Period", ["Week", "Month"], default="Week", key="ins_grain",
                                 label_visibility="collapsed")
    if grain == "Month":
        w = w.assign(week=w["week"].dt.to_period("M").dt.to_timestamp()).groupby("week", as_index=False).sum(
            numeric_only=True)
        w["share_flagged"] = w["flagged"] / w["bookings"]
        x = alt.X("yearmonth(week):O", title="Arrival month", scale=alt.Scale(paddingOuter=1),
                  axis=alt.Axis(format="%b %Y", formatType="time", labelOverlap="greedy", labelAngle=0))
        tip = alt.Tooltip("week:T", title="Month", format="%B %Y")
    else:
        x = alt.X("yearmonthdate(week):O", title="Arrival week (Monday)", scale=alt.Scale(paddingOuter=1),
                  axis=alt.Axis(format="%d %b %Y", formatType="time", labelOverlap="greedy", labelAngle=0,
                                labelSeparation=10))
        tip = alt.Tooltip("week:T", title="Week of", format="%d %b %Y")
    if view == "Share likely to cancel":
        chart = alt.Chart(w).mark_line(point=True, color=CANCEL, strokeWidth=2).encode(
            x=x, y=alt.Y("share_flagged:Q", title="Likely to cancel", axis=alt.Axis(format="%")),
            tooltip=[tip, alt.Tooltip("share_flagged:Q", title="Likely to cancel", format=".1%"),
                     alt.Tooltip("bookings:Q", title="Arrivals")])
    elif view == "Revenue flagged":
        chart = alt.Chart(w.assign(value=to_display(w["revenue_flagged"]))).mark_bar(color=CANCEL).encode(
            x=x, y=alt.Y("value:Q", title=f"Revenue flagged ({code})"),
            tooltip=[tip, alt.Tooltip("value:Q", title=f"Revenue flagged ({code})", format=",.0f")])
    elif view == "Expected arrivals":
        chart = alt.Chart(w).mark_bar(color=STAY).encode(
            x=x, y=alt.Y("expected_arrivals:Q", title="Expected arrivals"),
            tooltip=[tip, alt.Tooltip("expected_arrivals:Q", title="Expected arrivals"),
                     alt.Tooltip("bookings:Q", title="Booked")])
    else:
        long = pd.concat([w.assign(Prediction="Likely to cancel", count=w["flagged"]),
                          w.assign(Prediction="Likely to stay", count=w["bookings"] - w["flagged"])])
        chart = alt.Chart(long).mark_bar().encode(
            x=x, y=alt.Y("count:Q", title="Arrivals"),
            color=alt.Color("Prediction:N", scale=OUTCOME, legend=alt.Legend(orient="top", title=None)),
            order=alt.Order("Prediction:N"),
            tooltip=[tip, "Prediction:N", alt.Tooltip("count:Q", title="Reservations")])
    if week != ALL_WEEKS and grain != "Month":
        chart = chart + alt.Chart(pd.DataFrame({"week": [week]})).mark_rule(
            color=INK, strokeDash=[4, 3], strokeWidth=1.5).encode(x="yearmonthdate(week):O")
    show_chart(base_chart(chart.properties(height=260)))
    st.caption("One bar per period with arrivals." + (" Dashed line: the selected week."
                                                      if week != ALL_WEEKS and grain != "Month" else ""))

# ------------------------------------------------------------------ where the risk comes from
left, right = st.columns([1.35, 1], gap="medium")
with left.container(key="card_ins_where", height="stretch"):
    st.html('<div class="hs-card-title">Where the risk comes from</div>')
    by = st.selectbox("Break down by", list(insights.BREAKDOWNS), key="ins_by", label_visibility="collapsed")
    col = insights.BREAKDOWNS[by]
    b = insights.breakdown(sel, col, top=10 if col == "country" else None)
    b = b.assign(revenue=to_display(b["revenue_flagged"]))
    bars = alt.Chart(b).mark_bar(color=CANCEL, cornerRadiusEnd=3, height={"band": 0.6}).encode(
        y=alt.Y("group:N", sort=list(b["group"]), title=None),
        x=alt.X("share_flagged:Q", title="Share likely to cancel", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("group:N", title=by), alt.Tooltip("bookings:Q", title="Reservations"),
                 alt.Tooltip("flagged:Q", title="Likely to cancel"),
                 alt.Tooltip("share_flagged:Q", title="Share", format=".1%"),
                 alt.Tooltip("revenue:Q", title=f"Revenue flagged ({code})", format=",.0f")])
    labels = bars.mark_text(align="left", dx=4, color=INK, fontSize=12).encode(text=alt.Text("share_flagged:Q",
                                                                                            format=".0%"))
    show_chart(base_chart((bars + labels).properties(height=max(150, 34 * len(b)))))
    st.caption("Hover a bar for the number of reservations and the revenue flagged. Small groups give unstable shares.")

with right.container(key="card_ins_bands", height="stretch"):
    st.html('<div class="hs-card-title">How the risk is spread</div>')
    bd = insights.bands(sel)
    bd = bd.assign(revenue_display=to_display(bd["revenue"]))
    show_chart(base_chart(alt.Chart(bd).mark_bar(cornerRadiusEnd=3, height={"band": 0.6}).encode(
        y=alt.Y("band:N", sort=insights.BAND_LABELS, title=None),
        x=alt.X("bookings:Q", title="Reservations"),
        color=alt.Color("band:N", scale=alt.Scale(domain=insights.BAND_LABELS, range=BAND_COLOURS), legend=None),
        tooltip=[alt.Tooltip("band:N", title="Band"), alt.Tooltip("bookings:Q", title="Reservations"),
                 alt.Tooltip("revenue_display:Q", title=f"Booked revenue ({code})", format=",.0f")])
        .properties(height=150)))
    show_chart(base_chart(alt.Chart(sel[["probability"]]).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("probability:Q", bin=alt.Bin(extent=[0, 1], step=0.05), title="Cancellation probability",
                axis=alt.Axis(format="%", values=[0, .25, .5, .75, 1])),
        y=alt.Y("count():Q", title="Reservations"),
        color=alt.condition(alt.datum.probability >= .5, alt.value(CANCEL), alt.value(STAY)),
        tooltip=[alt.Tooltip("count():Q", title="Reservations")]).properties(height=150)))
    st.caption("Reservations from 50% are flagged as likely to cancel. Contact the very high band first.")

# ------------------------------------------------------------------ reasons and actions
section("Main reasons and suggested actions", "Among the reservations flagged as likely to cancel")
r = insights.reasons(sel)
if r.empty:
    st.caption("No reservation in this selection is flagged as likely to cancel.")
else:
    rv, rcfg = money_view(r.rename(columns={"reason": "Main reason", "bookings": "Reservations",
                                            "share_of_flagged": "Share of flagged",
                                            "revenue_flagged": "Revenue flagged (EUR)", "action": "Suggested action"})
                          .assign(**{"Share of flagged": r["share_of_flagged"] * 100}), ["Revenue flagged (EUR)"])
    st.dataframe(rv, hide_index=True, width="stretch",
                 column_config={**rcfg, "Share of flagged": st.column_config.ProgressColumn(
                     format="%.0f%%", min_value=0, max_value=100),
                     "Suggested action": st.column_config.TextColumn(width="large")})
    st.caption("The main reason is the input that raised the risk most (SHAP). It shows an association learned from "
               "past bookings, not a proven cause, so the suggested actions are for the hotel to judge.")
nav_link("views/actions.py", "Open the action list for individual reservations", ":material/checklist:")
